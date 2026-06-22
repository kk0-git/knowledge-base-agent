from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.apps import (
    CoachTurnRequest,
    InterviewCoachApp,
    InterviewInterviewerApp,
    InterviewTurnRequest,
    LibrarianApp,
    LibrarianRequest,
)
from agent.llm.tool_calling import OpenAICompatibleToolCallingClient
from agent.interview.state import build_interview_state_machine
from agent.runtime import AgentRuntime
from agent.skill_loader import SkillLoader
from agent.tool_registry import ToolRegistry
from agent.tools import register_debug_tools, register_interview_tools, register_profile_tools
from agent.tools.vault import register_vault_tools
from agent.trace import TraceRecorder
from knowledge_base_agent.config import load_llm_config
from knowledge_base_agent.llm import create_llm_client
from knowledge_base_agent.llm.schema import LLMRequest
from services.rag.agent_answer import (
    AgentAnswerConfig,
    AgentAnswerPipeline,
    agent_retrieval_result_to_dict,
)
from services.rag.intent_router import ConversationCommand, LLMIntentRouter
from services.rag.index_sync import RAGIndexSyncConfig, sync_rag_index
from services.rag.online_search import OnlineSearchClient
from services.rag.reranker import DEFAULT_RERANKER_MODEL
from services.rag.search_service import SearchOptions, SearchService
from services.rag.vector_store_loader import (
    DEFAULT_HNSW_EF_CONSTRUCTION,
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_HNSW_M,
)
from services.wiki.manager import WikiManager, rebuild_tag_index, set_tag_policy
from services.wiki.report_writer import write_obsidian_wiki_report
from services.wiki.state_store import WikiStateStore
from services.wiki.tag_consolidation import (
    LLMTagConsolidator,
    propose_deterministic_cleanup,
    tag_cleanup_proposal_to_dict,
)
from services.wiki.tag_refinement import LLMTagRefiner, tag_refinement_proposal_to_dict
from services.workflows.context_builder import ContextBuilder
from services.workflows.interview import (
    build_interview_messages,
    deterministic_interview_plan,
    format_interview_opening_message,
    generate_session_summary,
    interview_plan_from_dict,
    interview_session_state_from_dict,
    interview_plan_to_dict,
    prepare_interview_plan,
)
from services.workflows.interview_profile import (
    InterviewProfileStore,
    build_candidate_profile_debug,
    render_candidate_profile_context,
)
from services.workflows.interview_memory_commit import commit_interview_memory
from services.workflows.interview_sessions import InterviewSessionStore
from services.workflows.runner import WorkflowRunner, task_result_to_dict
from services.workflows.schema import ScopeSpec, WorkflowSpec, WritebackSpec
from services.workflows.scope_resolver import ScopeResolver


def resolve_profile_topic_for_request(
    *,
    query: str,
    current_topic: str | None,
    plan: Any,
) -> tuple[str | None, str]:
    query_text = str(query or "")
    if plan and getattr(plan, "topics", None):
        for topic in plan.topics:
            name = str(getattr(topic, "name", "") or "").strip()
            if name and name in query_text:
                return name, "query_topic_match"
    if current_topic:
        return current_topic, "session_state"
    if plan and getattr(plan, "topics", None):
        return plan.topics[0].name, "plan_default"
    return None, "unknown"


def add_profile_injection_audit(
    *,
    debug: dict[str, Any] | None,
    topic_source: str,
    candidate_profile_context: str | None,
) -> dict[str, Any]:
    result = dict(debug or {})
    context = str(candidate_profile_context or "")
    result["topic_source"] = topic_source
    result["prompt_section"] = "## Candidate Profile"
    result["injected_to_prompt"] = bool(context)
    result["prompt_context_sha256"] = hashlib.sha256(context.encode("utf-8")).hexdigest()[:16] if context else ""
    result["prompt_context_line_count"] = len(context.splitlines()) if context else 0
    result["prompt_context_preview"] = context[:1200] if context else ""
    return result


def latest_user_content(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if str(message.get("role") or "") == "user":
            return str(message.get("content") or "").strip()
    return ""


def previous_assistant_before_latest_user_content(messages: list[dict[str, Any]]) -> str:
    latest_user_index: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        if str(messages[index].get("role") or "") == "user":
            latest_user_index = index
            break
    if latest_user_index is None:
        return ""
    for index in range(latest_user_index - 1, -1, -1):
        if str(messages[index].get("role") or "") == "assistant":
            return str(messages[index].get("content") or "").strip()
    return ""


def classify_runtime_error(exc: BaseException | str) -> dict[str, Any]:
    raw = str(exc or "")
    lower = raw.lower()
    error_type = type(exc).__name__ if isinstance(exc, BaseException) else "Error"
    retryable = True
    category = "unknown"
    if "ssl" in lower or "certificate" in lower:
        category = "ssl"
    elif "timeout" in lower or "timed out" in lower:
        category = "timeout"
    elif "connection" in lower or "network" in lower or "dns" in lower or "temporary failure" in lower:
        category = "network"
    elif "429" in lower or "rate limit" in lower or "too many requests" in lower:
        category = "rate_limit"
    elif "500" in lower or "502" in lower or "503" in lower or "504" in lower:
        category = "server_error"
    elif "400" in lower or "401" in lower or "403" in lower or "unauthorized" in lower or "forbidden" in lower:
        category = "request_error"
        retryable = False
    return {
        "message": raw,
        "error_type": error_type,
        "category": category,
        "retryable": retryable,
    }


class SearchRequest(BaseModel):
    query: str = Field(default="")
    mode: str = Field(default="hybrid")
    top_k: int = Field(default=10, ge=1, le=100)
    enable_rewrite: bool = Field(default=False)
    rewrite_confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    rewrite_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    dense_top_k: int = Field(default=50, ge=1, le=500)
    bm25_top_k: int = Field(default=50, ge=1, le=500)
    rrf_k: int = Field(default=60, ge=1, le=500)
    reranker_type: str = Field(default="off")
    reranker_model: str = Field(default=DEFAULT_RERANKER_MODEL)
    rerank_candidates: int = Field(default=50, ge=1, le=500)
    rerank_batch_size: int = Field(default=16, ge=1, le=128)
    rerank_max_length: int = Field(default=512, ge=64, le=4096)
    include_debug: bool = Field(default=True)


class AgentRequest(BaseModel):
    query: str = Field(default="")
    chat_mode: str = Field(default="answer")
    command: str = Field(default="auto")
    scope_type: str = Field(default="tag")
    scope_value: str | None = Field(default=None)
    scope_paths: list[str] = Field(default_factory=list)
    chat_history: list[dict[str, str]] = Field(default_factory=list)
    notes_top_k: int = Field(default=5, ge=1, le=50)
    regex_top_k: int = Field(default=8, ge=1, le=50)
    bm25_top_k: int = Field(default=8, ge=1, le=50)
    dense_top_k: int = Field(default=50, ge=1, le=500)
    hybrid_bm25_top_k: int = Field(default=50, ge=1, le=500)
    rrf_k: int = Field(default=60, ge=1, le=500)
    max_chars_per_item: int = Field(default=1000, ge=100, le=5000)
    max_context_chars: int = Field(default=8000, ge=1000, le=30000)
    online_provider: str | None = Field(default=None)
    online_top_k: int = Field(default=5, ge=1, le=20)
    speculative_notes_search: bool = Field(default=True)
    strict_evidence: bool = Field(default=False)
    interview_max_chars_per_note: int = Field(default=4000, ge=500, le=12000)
    interview_max_context_chars: int = Field(default=24000, ge=2000, le=60000)
    interview_plan: dict[str, Any] | None = Field(default=None)
    interview_state: dict[str, Any] | None = Field(default=None)
    session_id: str | None = Field(default=None)


class InterviewSummaryRequest(BaseModel):
    scope_type: str = Field(default="tag")
    scope_value: str | None = Field(default=None)
    scope_paths: list[str] = Field(default_factory=list)
    chat_history: list[dict[str, str]] = Field(default_factory=list)
    answer: str = Field(default="")
    interview_plan: dict[str, Any] | None = Field(default=None)
    notes_top_k: int = Field(default=5, ge=1, le=50)
    dense_top_k: int = Field(default=50, ge=1, le=500)
    hybrid_bm25_top_k: int = Field(default=50, ge=1, le=500)
    rrf_k: int = Field(default=60, ge=1, le=500)
    interview_max_chars_per_note: int = Field(default=4000, ge=500, le=12000)
    interview_max_context_chars: int = Field(default=24000, ge=2000, le=60000)
    session_id: str | None = Field(default=None)
    user_message_id: str | None = Field(default=None)
    assistant_message_id: str | None = Field(default=None)
    interview_state: dict[str, Any] | None = Field(default=None)


class InterviewSessionCreateRequest(BaseModel):
    source_type: str = Field(default="tag")
    source_value: str | None = Field(default=None)
    source_paths: list[str] = Field(default_factory=list)
    source_note_paths: list[str] = Field(default_factory=list)
    interview_plan: dict[str, Any] | None = Field(default=None)
    interview_state: dict[str, Any] | None = Field(default=None)
    extra: dict[str, Any] = Field(default_factory=dict)


class InterviewSessionAppendTurnRequest(BaseModel):
    user_content: str = Field(default="")
    assistant_content: str = Field(default="")
    interview_plan: dict[str, Any] | None = Field(default=None)
    interview_state: dict[str, Any] | None = Field(default=None)
    source_note_paths: list[str] = Field(default_factory=list)
    agent_actions: list[dict[str, Any]] = Field(default_factory=list)


class InterviewSessionPendingTurnRequest(BaseModel):
    user_content: str = Field(default="")
    interview_plan: dict[str, Any] | None = Field(default=None)
    interview_state: dict[str, Any] | None = Field(default=None)
    source_note_paths: list[str] = Field(default_factory=list)


class InterviewSessionCompleteAssistantRequest(BaseModel):
    assistant_content: str = Field(default="")
    interview_plan: dict[str, Any] | None = Field(default=None)
    interview_state: dict[str, Any] | None = Field(default=None)
    source_note_paths: list[str] = Field(default_factory=list)
    agent_actions: list[dict[str, Any]] = Field(default_factory=list)


class InterviewSessionFailAssistantRequest(BaseModel):
    assistant_content: str = Field(default="")
    error_type: str = Field(default="Error")
    error_message: str = Field(default="")
    retryable: bool = Field(default=True)


class InterviewSessionReviewRequest(BaseModel):
    user_message_id: str = Field(default="")
    assistant_message_id: str = Field(default="")
    feedback: dict[str, Any] = Field(default_factory=dict)
    reference_answer: str = Field(default="")
    expression_example: str = Field(default="")
    context_note_paths: list[str] = Field(default_factory=list)
    profile_signals: list[dict[str, Any]] = Field(default_factory=list)


class InterviewSessionReviewPendingRequest(BaseModel):
    user_message_id: str = Field(default="")
    assistant_message_id: str = Field(default="")
    context_note_paths: list[str] = Field(default_factory=list)


class InterviewSessionReviewFailRequest(BaseModel):
    user_message_id: str = Field(default="")
    assistant_message_id: str = Field(default="")
    error: str = Field(default="")
    context_note_paths: list[str] = Field(default_factory=list)


class InterviewSessionTraceRequest(BaseModel):
    event: str = Field(default="")
    summary: str = Field(default="")
    details: dict[str, Any] = Field(default_factory=dict)


class InterviewSessionSelectTopicRequest(BaseModel):
    topic: str = Field(default="")
    reason: str = Field(default="")
    source: str = Field(default="ui")
    interview_plan: dict[str, Any] | None = Field(default=None)


class WikiSynthesizeTagRequest(BaseModel):
    tag: str = Field(default="")
    force: bool = Field(default=True)


class WikiSyncRequest(BaseModel):
    force: bool = Field(default=False)
    limit: int | None = Field(default=None, ge=0)
    include_embedding: bool = Field(default=True)
    allow_full_rebuild: bool = Field(default=False)


class WikiRefineTagRequest(BaseModel):
    tag: str = Field(default="")


class WikiConsolidateTagsRequest(BaseModel):
    include_llm: bool = Field(default=False)


class WikiSetPolicyRequest(BaseModel):
    tag: str = Field(default="")
    policy: str = Field(default="")


class WorkspaceConfigRequest(BaseModel):
    vault_path: str = Field(default="")
    wiki_dir: str = Field(default="")
    wiki_state_path: str = Field(default="")
    workspace_state_path: str = Field(default="")
    index_path: str = Field(default="")
    bm25_index_path: str = Field(default="")
    min_notes_per_tag: int = Field(default=2, ge=1)
    overview_note_threshold: int = Field(default=12, ge=1)


class WorkflowScopeRequest(BaseModel):
    type: str = Field(default="search")
    value: str | None = Field(default=None)
    paths: list[str] = Field(default_factory=list)
    top_k: int = Field(default=8, ge=1, le=100)
    options: dict[str, Any] = Field(default_factory=dict)


class WorkflowWritebackRequest(BaseModel):
    type: str = Field(default="none")
    path: str | None = Field(default=None)
    options: dict[str, Any] = Field(default_factory=dict)


class WorkflowRunRequest(BaseModel):
    task_type: str = Field(default="answer")
    scope: WorkflowScopeRequest = Field(default_factory=WorkflowScopeRequest)
    user_request: str = Field(default="")
    context_mode: str | None = Field(default=None)
    writeback: WorkflowWritebackRequest = Field(default_factory=WorkflowWritebackRequest)
    options: dict[str, Any] = Field(default_factory=dict)


@dataclass
class WorkspaceRuntimeConfig:
    vault_path: Path | None
    wiki_dir: Path | None
    wiki_state_path: Path | None
    workspace_state_path: Path | None
    index_path: Path
    bm25_index_path: Path
    min_notes_per_tag: int
    overview_note_threshold: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "vault_path": str(self.vault_path) if self.vault_path else "",
            "wiki_dir": str(self.wiki_dir) if self.wiki_dir else "",
            "wiki_state_path": str(self.wiki_state_path) if self.wiki_state_path else "",
            "workspace_state_path": str(self.workspace_state_path) if self.workspace_state_path else "",
            "index_path": str(self.index_path),
            "bm25_index_path": str(self.bm25_index_path),
            "min_notes_per_tag": self.min_notes_per_tag,
            "overview_note_threshold": self.overview_note_threshold,
        }


@dataclass
class PipelineTask:
    task_id: str
    kind: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    current_step: str | None = None
    events: list[dict[str, Any]] | None = None
    result: Any = None
    error: str | None = None
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "current_step": self.current_step,
            "events": list(self.events or []),
            "result": self.result,
            "error": self.error,
            "error_type": self.error_type,
        }


class PipelineTaskContext:
    def __init__(self, manager: "PipelineTaskManager", task_id: str) -> None:
        self.manager = manager
        self.task_id = task_id

    def emit(self, event: str, payload: dict[str, Any] | None = None) -> None:
        self.manager.emit(self.task_id, event, payload or {})


class PipelineTaskManager:
    def __init__(self, *, max_workers: int = 1, max_events: int = 200) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pipeline-task")
        self.max_events = max_events
        self.lock = threading.Lock()
        self.tasks: dict[str, PipelineTask] = {}

    def submit(self, kind: str, fn: Callable[[PipelineTaskContext], Any]) -> PipelineTask:
        task_id = uuid.uuid4().hex
        task = PipelineTask(
            task_id=task_id,
            kind=kind,
            status="queued",
            created_at=utc_now_iso(),
            events=[],
        )
        with self.lock:
            self.tasks[task_id] = task
        self.executor.submit(self._run, task_id, fn)
        return task

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self.lock:
            task = self.tasks.get(task_id)
            return task.to_dict() if task else None

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.lock:
            tasks = sorted(self.tasks.values(), key=lambda task: task.created_at, reverse=True)
            return [task.to_dict() for task in tasks[:limit]]

    def emit(self, task_id: str, event: str, payload: dict[str, Any]) -> None:
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return
            task.current_step = event
            task.events = task.events or []
            task.events.append(
                {
                    "at": utc_now_iso(),
                    "event": event,
                    "payload": payload,
                }
            )
            if len(task.events) > self.max_events:
                task.events = task.events[-self.max_events :]

    def _run(self, task_id: str, fn: Callable[[PipelineTaskContext], Any]) -> None:
        with self.lock:
            task = self.tasks[task_id]
            task.status = "running"
            task.started_at = utc_now_iso()
            task.current_step = "started"
        context = PipelineTaskContext(self, task_id)
        context.emit("started", {})
        try:
            result = fn(context)
            with self.lock:
                task = self.tasks[task_id]
                task.status = "succeeded"
                task.result = result
                task.finished_at = utc_now_iso()
                task.current_step = "succeeded"
        except Exception as exc:
            with self.lock:
                task = self.tasks[task_id]
                task.status = "failed"
                task.error_type = type(exc).__name__
                task.error = f"{type(exc).__name__}: {exc}"
                task.result = {"traceback": traceback.format_exc()}
                task.finished_at = utc_now_iso()
                task.current_step = "failed"


def load_workspace_runtime_config(
    *,
    config_path: Path,
    default_config: WorkspaceRuntimeConfig,
) -> WorkspaceRuntimeConfig:
    if not config_path.exists():
        return default_config
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        request = WorkspaceConfigRequest(**data)
        return workspace_config_from_request(
            request=request,
            project_root=config_path.parent,
            fallback=default_config,
        )
    except Exception:
        return default_config


def save_workspace_runtime_config(path: Path, config: WorkspaceRuntimeConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def workspace_config_from_request(
    *,
    request: WorkspaceConfigRequest,
    project_root: Path,
    fallback: WorkspaceRuntimeConfig,
) -> WorkspaceRuntimeConfig:
    vault = parse_optional_path(request.vault_path)
    wiki_dir = parse_optional_path(request.wiki_dir)
    wiki_state = parse_optional_path(request.wiki_state_path)
    workspace_state = parse_optional_path(request.workspace_state_path)
    index = parse_optional_path(request.index_path)
    bm25_index = parse_optional_path(request.bm25_index_path)

    if vault and not wiki_dir:
        wiki_dir = vault / "wiki"
    if not wiki_state:
        wiki_state = fallback.wiki_state_path or (project_root / "wiki-state" / "wiki_state.json")
    if not workspace_state:
        workspace_state = fallback.workspace_state_path or (project_root / "wiki-state" / "workspace_state.json")
    if not index:
        index = fallback.index_path
    if not bm25_index:
        bm25_index = fallback.bm25_index_path

    return WorkspaceRuntimeConfig(
        vault_path=vault,
        wiki_dir=wiki_dir,
        wiki_state_path=wiki_state,
        workspace_state_path=workspace_state,
        index_path=index,
        bm25_index_path=bm25_index,
        min_notes_per_tag=request.min_notes_per_tag,
        overview_note_threshold=request.overview_note_threshold,
    )


def parse_optional_path(value: str) -> Path | None:
    cleaned = str(value or "").strip().strip('"')
    return Path(cleaned).expanduser() if cleaned else None


def validate_workspace_config(config: WorkspaceRuntimeConfig) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if config.vault_path is None:
        errors.append("vault path is required")
    elif not config.vault_path.exists() or not config.vault_path.is_dir():
        errors.append(f"vault path does not exist or is not a directory: {config.vault_path}")
    if config.wiki_dir is None:
        warnings.append("wiki output directory is not configured")
    if config.wiki_state_path is None:
        errors.append("wiki state path is required")
    if config.workspace_state_path is None:
        errors.append("workspace state path is required")
    if not config.index_path.parent.exists():
        warnings.append(f"index directory does not exist and will be created on sync: {config.index_path.parent}")
    return {
        "ok": not errors,
        "message": "; ".join(errors or warnings or ["config is ready"]),
        "errors": errors,
        "warnings": warnings,
    }


def create_app(
    index_path: Path,
    bm25_index_path: Path | None,
    model_name: str,
    project_root: Path,
    vault_path: Path | None = None,
    embedding_provider: str = "local",
    embed_batch_size: int = 32,
    max_seq_length: int | None = None,
    vector_index: str = "flat",
    hnsw_m: int = DEFAULT_HNSW_M,
    hnsw_ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION,
    hnsw_ef_search: int = DEFAULT_HNSW_EF_SEARCH,
    max_chunk_chars: int = 1500,
    target_chunk_chars: int = 900,
    min_chunk_chars: int = 200,
    chunk_overlap: int = 200,
    chunk_split_mode: str = "indexed",
    strip_code_blocks: bool = False,
    wiki_state_path: Path | None = None,
    wiki_dir: Path | None = None,
    wiki_min_notes_per_tag: int = 2,
    wiki_overview_note_threshold: int = 12,
    sync_on_start: bool = False,
) -> FastAPI:
    app = FastAPI(title="Knowledge Agent")
    config_path = project_root / "workspace-config.json"
    runtime_config = load_workspace_runtime_config(
        config_path=config_path,
        default_config=WorkspaceRuntimeConfig(
            vault_path=vault_path,
            wiki_dir=wiki_dir,
            wiki_state_path=wiki_state_path,
            workspace_state_path=project_root / "wiki-state" / "workspace_state.json",
            index_path=index_path,
            bm25_index_path=bm25_index_path or index_path.with_suffix(".bm25.json"),
            min_notes_per_tag=wiki_min_notes_per_tag,
            overview_note_threshold=wiki_overview_note_threshold,
        ),
    )
    interview_session_store = InterviewSessionStore(project_root / "interview-sessions")
    interview_profile_store = InterviewProfileStore(project_root / "profile" / "interview_profile.json")
    startup_sync_status: dict[str, Any] = {
        "enabled": sync_on_start,
        "running": False,
        "started_at": None,
        "finished_at": None,
        "failed": False,
        "error": None,
        "result": None,
        "task_id": None,
    }
    task_manager = PipelineTaskManager(max_workers=1)
    service = SearchService(
        index_path=index_path,
        bm25_index_path=bm25_index_path,
        model_name=model_name,
        project_root=project_root,
        embedding_provider=embedding_provider,
        embed_batch_size=embed_batch_size,
        max_seq_length=max_seq_length,
        vector_index=vector_index,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
    )
    rag_prewarm_lock = threading.Lock()
    rag_prewarm_started = False

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return CHAT_HTML

    @app.get("/search", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/chat", response_class=HTMLResponse)
    def chat() -> str:
        return CHAT_HTML

    @app.get("/topics", response_class=HTMLResponse)
    def topics() -> str:
        return TOPICS_HTML

    @app.get("/audit", response_class=HTMLResponse)
    def audit() -> str:
        return AUDIT_HTML

    @app.get("/organize", response_class=HTMLResponse)
    def organize() -> str:
        return AUDIT_HTML

    @app.get("/wiki", response_class=HTMLResponse)
    def wiki() -> str:
        return WIKI_HTML

    @app.get("/admin/wiki", response_class=HTMLResponse)
    def admin_wiki() -> str:
        return WIKI_HTML

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "index": str(index_path),
            "bm25_index": str(service.bm25_index_path),
            "model": model_name,
            "embedding_provider": embedding_provider,
            "vector_index": vector_index,
            "hnsw": {
                "m": hnsw_m,
                "ef_construction": hnsw_ef_construction,
                "ef_search": hnsw_ef_search,
            },
            "wiki": {
                "enabled": runtime_config.vault_path is not None and runtime_config.wiki_state_path is not None,
                "state": str(runtime_config.wiki_state_path) if runtime_config.wiki_state_path else None,
                "wiki_dir": str(runtime_config.wiki_dir) if runtime_config.wiki_dir else None,
                "min_notes_per_tag": runtime_config.min_notes_per_tag,
                "overview_note_threshold": runtime_config.overview_note_threshold,
                "sync_on_start": sync_on_start,
                "startup_sync": startup_sync_status,
            },
            "chunker": {
                "max_chunk_chars": max_chunk_chars,
                "target_chunk_chars": target_chunk_chars,
                "min_chunk_chars": min_chunk_chars,
                "chunk_overlap": chunk_overlap,
                "chunk_split_mode": chunk_split_mode,
                "strip_code_blocks": strip_code_blocks,
            },
        }

    @app.get("/settings", response_class=HTMLResponse)
    def settings() -> str:
        return SETTINGS_HTML

    @app.get("/api/workspace/config")
    def workspace_config() -> dict[str, Any]:
        return {
            "config": runtime_config.to_dict(),
            "config_path": str(config_path),
            "validation": validate_workspace_config(runtime_config),
        }

    @app.post("/api/workspace/config")
    def save_workspace_config(request: WorkspaceConfigRequest) -> dict[str, Any]:
        nonlocal runtime_config
        try:
            next_config = workspace_config_from_request(
                request=request,
                project_root=project_root,
                fallback=runtime_config,
            )
            validation = validate_workspace_config(next_config)
            if not validation["ok"]:
                raise HTTPException(status_code=400, detail=validation["message"])
            save_workspace_runtime_config(config_path, next_config)
            runtime_config = next_config
            return {
                "config": runtime_config.to_dict(),
                "config_path": str(config_path),
                "validation": validation,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/chat/starters")
    def chat_starters() -> dict[str, Any]:
        return {
            "starters": [
                "How should Redis Stream consumers handle events?",
                "What is a process, and how is it different from a thread?",
                "How do I configure FastAPI CORS?",
                "What notes do I have about LLM Agent architecture?",
                "What can cause WinError 10060?",
            ]
        }

    @app.post("/api/search")
    def search(request: SearchRequest) -> dict[str, Any]:
        try:
            response = service.search(SearchOptions(**request.model_dump()))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return asdict(response)

    def build_wiki_manager() -> WikiManager:
        active_vault_path = runtime_config.vault_path
        active_wiki_state_path = runtime_config.wiki_state_path
        if active_vault_path is None:
            raise HTTPException(status_code=400, detail="vault path is required for wiki")
        if active_wiki_state_path is None:
            raise HTTPException(status_code=400, detail="wiki state path is required")
        llm_config = load_llm_config(project_root)
        llm_client = create_llm_client(llm_config)
        return WikiManager(
            vault_root=active_vault_path,
            state_store=WikiStateStore(active_wiki_state_path),
            llm_client=llm_client,
            llm_model=llm_config.model,
            wiki_dir=runtime_config.wiki_dir,
            min_notes_per_tag=runtime_config.min_notes_per_tag,
            overview_note_threshold=runtime_config.overview_note_threshold,
        )

    def build_answer_pipeline(request: AgentRequest | None = None) -> AgentAnswerPipeline:
        if runtime_config.vault_path is None:
            raise HTTPException(status_code=400, detail="vault path is required for answer workflow")
        request = request or AgentRequest()
        llm_config = load_llm_config(project_root)
        llm_client = create_llm_client(llm_config)
        router = build_web_router(request.command, llm_client, llm_config)
        manager = service.build_manager(
            SearchOptions(
                query=request.query,
                mode="hybrid",
                dense_top_k=request.dense_top_k,
                bm25_top_k=request.hybrid_bm25_top_k,
                rrf_k=request.rrf_k,
            )
        )
        return AgentAnswerPipeline(
            router=router,
            llm_client=llm_client,
            llm_model=llm_config.model,
            manager=manager,
            vault_root=runtime_config.vault_path,
            online_client=OnlineSearchClient(provider=request.online_provider),
            config=AgentAnswerConfig(
                notes_top_k=request.notes_top_k,
                regex_top_k=request.regex_top_k,
                bm25_top_k=request.bm25_top_k,
                dense_top_k=request.dense_top_k,
                hybrid_bm25_top_k=request.hybrid_bm25_top_k,
                rrf_k=request.rrf_k,
                max_chars_per_item=request.max_chars_per_item,
                max_context_chars=request.max_context_chars,
                online_top_k=request.online_top_k,
                speculative_notes_search=request.speculative_notes_search,
            ),
            answer_temperature=llm_config.temperature,
        )

    def build_workflow_runner(
        *,
        task_type: str,
        scope_type: str | None = None,
        answer_request: AgentRequest | None = None,
    ) -> WorkflowRunner:
        if runtime_config.vault_path is None:
            raise HTTPException(status_code=400, detail="vault path is required for workflows")
        rag_manager = None
        if task_type == "answer" or scope_type == "search":
            rag_manager = service.build_manager(
                SearchOptions(
                    query=answer_request.query if answer_request else "",
                    mode="hybrid",
                )
            )
        wiki_manager = build_wiki_manager() if task_type == "synthesize_wiki" else None
        llm_client = None
        llm_model = None
        llm_temperature = 0.2
        if task_type in {"organize_suggestions", "generate_review", "organize"}:
            llm_config = load_llm_config(project_root)
            llm_client = create_llm_client(llm_config)
            llm_model = llm_config.model
            llm_temperature = llm_config.temperature
        return WorkflowRunner(
            scope_resolver=ScopeResolver(
                vault_root=runtime_config.vault_path,
                wiki_state_store=WikiStateStore(runtime_config.wiki_state_path) if runtime_config.wiki_state_path else None,
                wiki_dir=runtime_config.wiki_dir,
                rag_manager=rag_manager,
                overview_note_threshold=runtime_config.overview_note_threshold,
            ),
            context_builder=ContextBuilder(vault_root=runtime_config.vault_path),
            answer_pipeline=build_answer_pipeline(answer_request) if answer_request else None,
            wiki_manager=wiki_manager,
            vault_root=runtime_config.vault_path,
            wiki_state_store=WikiStateStore(runtime_config.wiki_state_path) if runtime_config.wiki_state_path else None,
            wiki_dir=runtime_config.wiki_dir,
            overview_note_threshold=runtime_config.overview_note_threshold,
            llm_client=llm_client,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
        )

    def build_interview_agent_runtime(llm_client) -> AgentRuntime:
        registry = ToolRegistry()
        register_debug_tools(registry)
        register_vault_tools(registry)
        register_interview_tools(registry)
        register_profile_tools(registry)
        return AgentRuntime(
            llm_client=OpenAICompatibleToolCallingClient(llm_client),
            skill_loader=SkillLoader(project_root / "skills", registry=registry),
            tool_registry=registry,
            trace_recorder=TraceRecorder(project_root / "eval-results" / "agent-debug" / "traces"),
        )

    def agent_v2_interview_enabled() -> bool:
        value = os.getenv("AGENT_V2_INTERVIEW", "1").strip().lower()
        return value not in {"0", "false", "no", "off"}

    def agent_v2_coach_enabled() -> bool:
        value = os.getenv("AGENT_V2_COACH", "").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def agent_v2_librarian_enabled() -> bool:
        value = os.getenv("AGENT_V2_LIBRARIAN", "").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def librarian_online_enabled(request: AgentRequest) -> bool:
        provider = str(request.online_provider or "").strip().lower()
        return provider not in {"", "0", "false", "no", "none", "off", "disabled"}

    def build_interview_rag_manager(request: AgentRequest):
        return service.build_manager(
            SearchOptions(
                query=request.query,
                mode="hybrid",
                dense_top_k=request.dense_top_k,
                bm25_top_k=request.hybrid_bm25_top_k,
                rrf_k=request.rrf_k,
            )
        )

    def build_librarian_rag_manager(request: AgentRequest, *, query: str | None = None):
        return service.build_manager(
            SearchOptions(
                query=query or request.query,
                mode="hybrid",
                dense_top_k=request.dense_top_k,
                bm25_top_k=request.hybrid_bm25_top_k,
                rrf_k=request.rrf_k,
            )
        )

    def resolve_librarian_scope(request: AgentRequest):
        if runtime_config.vault_path is None:
            raise HTTPException(status_code=400, detail="vault path is required for librarian agent")
        scope_type = str(request.scope_type or "all_vault")
        if scope_type == "all_vault":
            return None, (), {}
        rag_manager = (
            build_librarian_rag_manager(request, query=request.scope_value or request.query)
            if scope_type == "search"
            else None
        )
        resolver = ScopeResolver(
            vault_root=runtime_config.vault_path,
            wiki_state_store=WikiStateStore(runtime_config.wiki_state_path) if runtime_config.wiki_state_path else None,
            wiki_dir=runtime_config.wiki_dir,
            rag_manager=rag_manager,
            overview_note_threshold=runtime_config.overview_note_threshold,
        )
        scope = resolver.resolve(
            ScopeSpec(
                type=scope_type,
                value=request.scope_value or None,
                paths=tuple(request.scope_paths or ()),
                top_k=request.notes_top_k,
                options={
                    "dense_top_k": request.dense_top_k,
                    "bm25_top_k": request.hybrid_bm25_top_k,
                    "rrf_k": request.rrf_k,
                },
            )
        )
        scope_paths = tuple(
            str(item.get("path") or item.get("relative_path") or "")
            for item in scope.notes
            if str(item.get("path") or item.get("relative_path") or "").strip()
        )
        return scope, scope_paths, scope.metadata

    def prewarm_interview_rag(request: AgentRequest) -> None:
        nonlocal rag_prewarm_started
        with rag_prewarm_lock:
            if rag_prewarm_started:
                return
            rag_prewarm_started = True

        def run() -> None:
            try:
                build_interview_rag_manager(request)
            except Exception:
                pass

        threading.Thread(target=run, name="interview-rag-prewarm", daemon=True).start()

    def stream_librarian_agent_v2(request: AgentRequest, llm_client, llm_config):
        if runtime_config.vault_path is None:
            raise HTTPException(status_code=400, detail="vault path is required for librarian agent")

        yield sse_event("status", {"stage": "agent_v2_started", "message": "starting librarian agent runtime"})
        scope, scope_note_paths, scope_metadata = resolve_librarian_scope(request)
        yield sse_event(
            "context",
            {
                "mode": "answer",
                "strict_evidence": bool(request.strict_evidence),
                "scope": {
                    "type": request.scope_type,
                    "value": request.scope_value,
                    "paths": request.scope_paths,
                    "metadata": scope_metadata,
                    "note_count": len(scope_note_paths),
                },
                "items": list(scope.notes) if scope is not None else [],
                "stats": {"context_items": len(scope_note_paths), "agent_v2": True},
            },
        )

        online_enabled = librarian_online_enabled(request)
        online_client = OnlineSearchClient(provider=request.online_provider) if online_enabled else None
        rag_manager_factory = lambda: build_librarian_rag_manager(request)
        app_runner = LibrarianApp(build_interview_agent_runtime(llm_client))
        for event in app_runner.run_stream(
            LibrarianRequest(
                query=request.query,
                scope_type=request.scope_type,
                scope_value=request.scope_value or "",
                scope_note_paths=scope_note_paths,
                selected_note_paths=tuple(request.scope_paths or ()) if request.scope_type == "selected_notes" else (),
                chat_history=list(request.chat_history or []),
                vault_root=runtime_config.vault_path,
                rag_manager=build_librarian_rag_manager(request),
                rag_manager_factory=rag_manager_factory,
                online_search_client=online_client,
                online_enabled=online_enabled,
                strict_evidence=bool(request.strict_evidence),
                model=llm_config.model,
                tool_mode="auto",
                trace_path=str(project_root / "eval-results" / "agent-debug" / "traces"),
                temperature=llm_config.temperature,
            )
        ):
            yield sse_event(event["type"], event.get("payload") or {})

    def stream_interview_agent_v2(request: AgentRequest, llm_client, llm_config):
        if runtime_config.vault_path is None:
            raise HTTPException(status_code=400, detail="vault path is required for interview agent")
        if runtime_config.wiki_state_path is None:
            raise HTTPException(status_code=400, detail="wiki state path is required for interview scope")

        yield sse_event("status", {"stage": "agent_v2_started", "message": "starting interview agent runtime"})
        rag_manager = build_interview_rag_manager(request) if request.scope_type == "search" else None
        rag_manager_factory = lambda: build_interview_rag_manager(request)
        scope = ScopeResolver(
            vault_root=runtime_config.vault_path,
            wiki_state_store=WikiStateStore(runtime_config.wiki_state_path),
            wiki_dir=runtime_config.wiki_dir,
            rag_manager=rag_manager,
            overview_note_threshold=runtime_config.overview_note_threshold,
        ).resolve(
            ScopeSpec(
                type=request.scope_type,
                value=request.scope_value,
                paths=tuple(request.scope_paths),
                top_k=request.notes_top_k,
            )
        )
        context = ContextBuilder(
            vault_root=runtime_config.vault_path,
            max_chars_per_note=request.interview_max_chars_per_note,
            max_context_chars=request.interview_max_context_chars,
        ).build(scope, mode="interview_context")
        source_note_paths = tuple(
            str(item.get("path") or item.get("relative_path") or "")
            for item in context.items
            if str(item.get("path") or item.get("relative_path") or "").strip()
        )
        yield sse_event("context", {"items": context.items, "stats": context.stats, "scope": scope.metadata})

        plan = None
        plan_error: str | None = None
        plan_source: str | None = None
        plan_started_at = current_time()
        plan_ms: int | None = None
        if request.interview_plan:
            try:
                plan = interview_plan_from_dict(request.interview_plan, context=context)
                plan_source = "client_reused"
                plan_ms = elapsed_since(plan_started_at)
            except Exception as exc:
                plan_error = f"client plan rejected: {exc}"
        if plan is None:
            try:
                plan = prepare_interview_plan(
                    context=context,
                    llm_client=llm_client,
                    model=llm_config.model,
                    temperature=min(llm_config.temperature, 0.2),
                )
                plan_source = "generated"
                plan_ms = elapsed_since(plan_started_at)
            except Exception as exc:
                plan_error = str(exc) if not plan_error else f"{plan_error}; generated plan failed: {exc}"
                plan = deterministic_interview_plan(context)
                plan_source = "deterministic_fallback"
                plan_ms = elapsed_since(plan_started_at)
        yield sse_event(
            "interview_plan",
            {
                "available": True,
                "fallback_used": plan_source == "deterministic_fallback",
                "source": plan_source,
                "plan": interview_plan_to_dict(plan),
                "error": plan_error,
                "latency_ms": plan_ms,
            },
        )

        server_interview_state = request.interview_state
        if request.session_id:
            try:
                session = interview_session_store.load_session(str(request.session_id))
                server_interview_state = session.get("interview_state") or request.interview_state
            except Exception:
                server_interview_state = request.interview_state
        state_machine = build_interview_state_machine(
            plan=plan,
            state_payload=server_interview_state,
            session_id=str(request.session_id or ""),
        )
        if state_machine.snapshot().get("topic_phase") == "awaiting_selection":
            state_snapshot = state_machine.snapshot()
            selection_prompt = format_interview_opening_message(plan)
            yield sse_event("state_updated", state_snapshot)
            yield sse_event("answer_delta", {"text": selection_prompt})
            yield sse_event(
                "answer",
                {"answer": selection_prompt, "model": llm_config.model},
            )
            yield sse_event(
                "done",
                {
                    "telemetry": {
                        "command": "InterviewAgentV2",
                        "agent_v2": True,
                        "awaiting_topic_selection": True,
                        "interview_state": state_snapshot,
                    }
                },
            )
            if request.scope_type in {"folder", "selected_notes"}:
                prewarm_interview_rag(request)
            return

        app_runner = InterviewInterviewerApp(build_interview_agent_runtime(llm_client))
        turn_request = InterviewTurnRequest(
            query=request.query,
            session_id=str(request.session_id or ""),
            chat_history=list(request.chat_history or []),
            interview_plan=plan,
            interview_state=server_interview_state,
            vault_root=runtime_config.vault_path,
            rag_manager=rag_manager,
            rag_manager_factory=rag_manager_factory,
            scope_note_paths=source_note_paths,
            scope_type=request.scope_type,
            scope_value=request.scope_value or "",
            session_store=interview_session_store,
            profile_store=interview_profile_store,
            model=llm_config.model,
            tool_mode="auto",
            trace_path=str(project_root / "eval-results" / "agent-debug" / "traces"),
            max_steps=6,
            max_tool_calls_per_step=4,
            temperature=llm_config.temperature,
        )
        for event in app_runner.run_turn_stream(turn_request):
            yield sse_event(event["type"], event.get("payload") or {})

    def run_workspace_sync(
        *,
        force: bool = False,
        limit: int | None = None,
        include_embedding: bool = True,
        allow_full_rebuild: bool = False,
        task_context: PipelineTaskContext | None = None,
    ) -> dict[str, Any]:
        if task_context:
            task_context.emit("workspace_sync_started", {"include_embedding": include_embedding})
        manager = build_wiki_manager()
        rag_result = None
        if include_embedding:
            if task_context:
                task_context.emit("rag_sync_started", {"index": str(runtime_config.index_path)})
            if runtime_config.vault_path is None:
                raise HTTPException(status_code=400, detail="vault path is required for RAG sync")
            rag_result = sync_rag_index(
                RAGIndexSyncConfig(
                    vault_path=runtime_config.vault_path,
                    index_path=runtime_config.index_path,
                    bm25_index_path=runtime_config.bm25_index_path,
                    project_root=project_root,
                    model_name=model_name,
                    embedding_provider=embedding_provider,
                    embed_batch_size=embed_batch_size,
                    max_seq_length=max_seq_length,
                    max_chunk_chars=max_chunk_chars,
                    target_chunk_chars=target_chunk_chars,
                    min_chunk_chars=min_chunk_chars,
                    chunk_overlap=chunk_overlap,
                    chunk_split_mode=chunk_split_mode,
                    strip_code_blocks=strip_code_blocks,
                    allow_full_rebuild=allow_full_rebuild,
                    excluded_roots=(runtime_config.wiki_dir,) if runtime_config.wiki_dir else (),
                ),
                progress=task_context.emit if task_context else None,
            )
            if task_context:
                task_context.emit("rag_sync_done", rag_result)
        elif task_context:
            task_context.emit("rag_sync_skipped", {"reason": "disabled"})
        if task_context:
            task_context.emit("wiki_tag_sync_started", {})
        tag_result = manager.tag_changed_notes(force=force, limit=limit)
        if task_context:
            task_context.emit("wiki_tag_sync_done", tag_result)
        report = manager.report()
        report_path = None
        if runtime_config.wiki_dir:
            if task_context:
                task_context.emit("report_write_started", {})
            report_path = write_obsidian_wiki_report(
                report=report,
                wiki_dir=runtime_config.wiki_dir,
                sync_result={"rag": rag_result, "wiki": tag_result},
            )
            if task_context:
                task_context.emit("report_write_done", {"report_path": str(report_path)})
        return {
            "rag": rag_result,
            "wiki": tag_result,
            "report_path": str(report_path) if report_path else None,
        }

    @app.get("/api/wiki/report")
    def wiki_report() -> dict[str, Any]:
        try:
            data = build_wiki_manager().report()
            data["obsidian_vault_name"] = runtime_config.vault_path.name if runtime_config.vault_path else None
            return data
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/wiki/read", response_class=PlainTextResponse)
    def wiki_read(tag: str) -> str:
        try:
            result = build_wiki_manager().read_wiki(tag)
            return result["content"]
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/wiki/startup-sync-status")
    def wiki_startup_sync_status() -> dict[str, Any]:
        status = dict(startup_sync_status)
        task_id = status.get("task_id")
        if task_id:
            status["task"] = task_manager.get(str(task_id))
        return status

    @app.get("/api/tasks")
    def tasks() -> dict[str, Any]:
        return {"tasks": task_manager.list_recent()}

    @app.get("/api/tasks/{task_id}")
    def task_status(task_id: str) -> dict[str, Any]:
        task = task_manager.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
        return task

    @app.post("/api/wiki/sync")
    def wiki_sync(request: WikiSyncRequest) -> dict[str, Any]:
        try:
            task = task_manager.submit(
                "workspace_sync",
                lambda context: {
                    "result": run_workspace_sync(
                        force=request.force,
                        limit=request.limit,
                        include_embedding=request.include_embedding,
                        allow_full_rebuild=request.allow_full_rebuild,
                        task_context=context,
                    ),
                    "report": build_wiki_manager().report(),
                },
            )
            return {
                "task_id": task.task_id,
                "status": task.status,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def run_startup_sync(context: PipelineTaskContext) -> dict[str, Any]:
        startup_sync_status.update(
            {
                "running": True,
                "started_at": utc_now_iso(),
                "finished_at": None,
                "failed": False,
                "error": None,
                "result": None,
            }
        )
        try:
            result = run_workspace_sync(
                force=False,
                limit=None,
                include_embedding=True,
                allow_full_rebuild=False,
                task_context=context,
            )
            startup_sync_status.update({"result": result, "failed": False})
            return {
                "result": result,
                "report": build_wiki_manager().report(),
            }
        except Exception as exc:
            startup_sync_status.update(
                {
                    "failed": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        finally:
            startup_sync_status.update(
                {
                    "running": False,
                    "finished_at": utc_now_iso(),
                }
            )

    @app.on_event("startup")
    def start_optional_sync() -> None:
        if not sync_on_start:
            return
        if runtime_config.vault_path is None or runtime_config.wiki_state_path is None:
            startup_sync_status.update(
                {
                    "failed": True,
                    "error": "vault path and wiki state path are required for sync_on_start",
                    "finished_at": utc_now_iso(),
                }
            )
            return
        task = task_manager.submit("startup_workspace_sync", run_startup_sync)
        startup_sync_status.update({"task_id": task.task_id, "running": True})

    @app.post("/api/wiki/consolidate-tags")
    def wiki_consolidate_tags(request: WikiConsolidateTagsRequest) -> dict[str, Any]:
        try:
            if runtime_config.vault_path is None:
                raise HTTPException(status_code=400, detail="vault path is required for wiki")
            if runtime_config.wiki_state_path is None:
                raise HTTPException(status_code=400, detail="wiki state path is required")
            state = rebuild_tag_index(
                WikiStateStore(runtime_config.wiki_state_path).load(),
                overview_note_threshold=runtime_config.overview_note_threshold,
            )
            deterministic = [
                tag_cleanup_proposal_to_dict(proposal)
                for proposal in propose_deterministic_cleanup(state)
            ]
            llm_proposals: list[dict[str, Any]] = []
            if request.include_llm:
                llm_config = load_llm_config(project_root)
                llm_client = create_llm_client(llm_config)
                llm_proposals = [
                    tag_cleanup_proposal_to_dict(proposal)
                    for proposal in LLMTagConsolidator(
                        client=llm_client,
                        model=llm_config.model,
                    ).propose_cleanup(state=state, vault_root=runtime_config.vault_path)
                ]
            return {
                "deterministic_proposals": deterministic,
                "llm_proposals": llm_proposals,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/wiki/refine-tag")
    def wiki_refine_tag(request: WikiRefineTagRequest) -> dict[str, Any]:
        tag = request.tag.strip()
        if not tag:
            raise HTTPException(status_code=400, detail="tag is required")
        try:
            if runtime_config.vault_path is None:
                raise HTTPException(status_code=400, detail="vault path is required for wiki")
            if runtime_config.wiki_state_path is None:
                raise HTTPException(status_code=400, detail="wiki state path is required")
            llm_config = load_llm_config(project_root)
            llm_client = create_llm_client(llm_config)
            state = WikiStateStore(runtime_config.wiki_state_path).load()
            proposal = LLMTagRefiner(
                client=llm_client,
                model=llm_config.model,
            ).refine_tag(state=state, vault_root=runtime_config.vault_path, tag=tag)
            return {"tag_refinement": tag_refinement_proposal_to_dict(proposal)}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/wiki/set-policy")
    def wiki_set_policy(request: WikiSetPolicyRequest) -> dict[str, Any]:
        tag = request.tag.strip()
        policy = request.policy.strip()
        if not tag:
            raise HTTPException(status_code=400, detail="tag is required")
        if policy not in {"generate", "overview", "skip"}:
            raise HTTPException(status_code=400, detail="policy must be generate, overview, or skip")
        try:
            if runtime_config.wiki_state_path is None:
                raise HTTPException(status_code=400, detail="wiki state path is required")
            store = WikiStateStore(runtime_config.wiki_state_path)
            state = rebuild_tag_index(
                store.load(),
                overview_note_threshold=runtime_config.overview_note_threshold,
            )
            updated = set_tag_policy(state, tag=tag, wiki_policy=policy)
            store.save(updated)
            return {
                "tag": tag,
                "policy": policy,
                "report": build_wiki_manager().report(),
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/wiki/synthesize-tag")
    def wiki_synthesize_tag(request: WikiSynthesizeTagRequest) -> dict[str, Any]:
        tag = request.tag.strip()
        if not tag:
            raise HTTPException(status_code=400, detail="tag is required")
        try:
            def run_task(context: PipelineTaskContext) -> dict[str, Any]:
                context.emit("wiki_synthesis_started", {"tag": tag})
                runner = build_workflow_runner(task_type="synthesize_wiki", scope_type="tag")
                workflow_result = runner.run(
                    WorkflowSpec(
                        task_type="synthesize_wiki",
                        scope=ScopeSpec(type="tag", value=tag),
                        user_request=f"Generate wiki for {tag}",
                        writeback=WritebackSpec(type="wiki_file"),
                        options={"force": request.force},
                    )
                )
                result = workflow_result.output["synthesize"]
                context.emit("wiki_synthesis_done", result)
                manager = build_wiki_manager()
                report = manager.report()
                if runtime_config.wiki_dir:
                    context.emit("report_write_started", {})
                    report_path = write_obsidian_wiki_report(
                        report=report,
                        wiki_dir=runtime_config.wiki_dir,
                        sync_result={"wiki": result},
                    )
                    context.emit("report_write_done", {"report_path": str(report_path)})
                return {
                    "tag": tag,
                    "result": result,
                    "report": manager.report(),
                    "workflow": task_result_to_dict(workflow_result),
                }

            task = task_manager.submit("synthesize_topic", run_task)
            return {
                "task_id": task.task_id,
                "tag": tag,
                "status": task.status,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/workflows/run")
    def workflow_run(request: WorkflowRunRequest) -> dict[str, Any]:
        try:
            scope = ScopeSpec(
                type=request.scope.type,
                value=request.scope.value,
                paths=tuple(request.scope.paths),
                top_k=request.scope.top_k,
                options=request.scope.options,
            )
            writeback = WritebackSpec(
                type=request.writeback.type,
                path=request.writeback.path,
                options=request.writeback.options,
            )
            spec = WorkflowSpec(
                task_type=request.task_type,
                scope=scope,
                user_request=request.user_request,
                context_mode=request.context_mode,
                writeback=writeback,
                options=request.options,
            )

            def run_task(context: PipelineTaskContext) -> dict[str, Any]:
                context.emit("workflow_started", {"task_type": spec.task_type, "scope_type": spec.scope.type})
                answer_request = workflow_answer_request(request) if spec.task_type == "answer" else None
                runner = build_workflow_runner(
                    task_type=spec.task_type,
                    scope_type=spec.scope.type,
                    answer_request=answer_request,
                )
                result = runner.run(spec)
                payload = {"workflow": task_result_to_dict(result)}
                if spec.task_type == "synthesize_wiki" and runtime_config.wiki_dir:
                    context.emit("report_write_started", {})
                    report = build_wiki_manager().report()
                    synthesize_result = result.output.get("synthesize", {})
                    report_path = write_obsidian_wiki_report(
                        report=report,
                        wiki_dir=runtime_config.wiki_dir,
                        sync_result={"wiki": synthesize_result},
                    )
                    context.emit("report_write_done", {"report_path": str(report_path)})
                    payload["report"] = report
                    payload["report_path"] = str(report_path)
                context.emit("workflow_done", {"task_type": spec.task_type})
                return payload

            task = task_manager.submit(f"workflow:{request.task_type}", run_task)
            return {
                "task_id": task.task_id,
                "status": task.status,
                "task_type": request.task_type,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def stream_study_or_interview(request: AgentRequest, llm_client, llm_config):
        if runtime_config.vault_path is None:
            raise HTTPException(status_code=400, detail="vault path is required for interview review")
        if runtime_config.wiki_state_path is None:
            raise HTTPException(status_code=400, detail="wiki state path is required for interview review")

        yield sse_event("status", {"stage": "interview_scope_started", "message": "loading source notes"})
        rag_manager = None
        if request.scope_type == "search":
            rag_manager = service.build_manager(
                SearchOptions(
                    query=request.scope_value or request.query,
                    mode="hybrid",
                    dense_top_k=request.dense_top_k,
                    bm25_top_k=request.hybrid_bm25_top_k,
                    rrf_k=request.rrf_k,
                )
            )
        resolver = ScopeResolver(
            vault_root=runtime_config.vault_path,
            wiki_state_store=WikiStateStore(runtime_config.wiki_state_path),
            wiki_dir=runtime_config.wiki_dir,
            rag_manager=rag_manager,
            overview_note_threshold=runtime_config.overview_note_threshold,
        )
        scope = resolver.resolve(
            ScopeSpec(
                type=request.scope_type,
                value=request.scope_value or None,
                paths=tuple(request.scope_paths or ()),
                top_k=request.notes_top_k,
            )
        )
        context = ContextBuilder(
            vault_root=runtime_config.vault_path,
            max_chars_per_note=request.interview_max_chars_per_note,
            max_context_chars=request.interview_max_context_chars,
        ).build(scope, mode="interview_context")
        yield sse_event(
            "context",
            {
                "items": list(context.items),
                "mode": request.chat_mode,
                "scope": {
                    "type": request.scope_type,
                    "value": request.scope_value,
                    "paths": request.scope_paths,
                },
                "stats": context.stats,
            },
        )
        plan = None
        plan_error: str | None = None
        plan_source: str | None = None
        plan_started_at = current_time()
        plan_ms: int | None = None
        if request.chat_mode == "interview":
            if request.interview_plan:
                try:
                    plan = interview_plan_from_dict(request.interview_plan, context=context)
                    plan_source = "client_reused"
                    plan_ms = elapsed_since(plan_started_at)
                except Exception as exc:
                    plan_error = f"client plan rejected: {exc}"
            if plan is None:
                try:
                    yield sse_event(
                        "status",
                        {"stage": "interview_plan_started", "message": "preparing interview plan"},
                    )
                    plan = prepare_interview_plan(
                        context=context,
                        llm_client=llm_client,
                        model=llm_config.model,
                        temperature=min(llm_config.temperature, 0.2),
                    )
                    plan_source = "generated"
                    plan_ms = elapsed_since(plan_started_at)
                except Exception as exc:
                    plan_error = str(exc) if not plan_error else f"{plan_error}; generated plan failed: {exc}"
                    plan = deterministic_interview_plan(context)
                    plan_source = "deterministic_fallback"
                    plan_ms = elapsed_since(plan_started_at)
            yield sse_event(
                "interview_plan",
                {
                    "available": True,
                    "fallback_used": plan_source == "deterministic_fallback",
                    "source": plan_source,
                    "plan": interview_plan_to_dict(plan),
                    "error": plan_error,
                    "latency_ms": plan_ms,
                },
            )
        yield sse_event(
            "status",
            {
                "stage": "interview_started",
                "message": f"loaded {context.stats.get('context_items', 0)} source notes",
            },
        )

        started_at = current_time()
        first_delta_ms: int | None = None
        answer_parts: list[str] = []
        session_state = interview_session_state_from_dict(request.interview_state)
        candidate_profile_context = None
        profile_debug: dict[str, Any] | None = None
        if request.chat_mode == "interview":
            resolved_profile_topic, profile_topic_source = resolve_profile_topic_for_request(
                query=request.query,
                current_topic=session_state.current_topic if session_state else None,
                plan=plan,
            )
            try:
                profile = interview_profile_store.load()
                candidate_profile_context = render_candidate_profile_context(
                    profile=profile,
                    current_topic=resolved_profile_topic,
                    plan=plan,
                )
                profile_debug = build_candidate_profile_debug(
                    profile=profile,
                    current_topic=resolved_profile_topic,
                    plan=plan,
                )
            except Exception as exc:
                candidate_profile_context = f"(candidate profile unavailable: {exc})"
                profile_debug = {"available": False, "error": str(exc)}
            profile_debug = add_profile_injection_audit(
                debug=profile_debug,
                topic_source=profile_topic_source,
                candidate_profile_context=candidate_profile_context,
            )
            yield sse_event("profile_debug", profile_debug)
        llm_request = LLMRequest(
            model=llm_config.model,
            messages=build_interview_messages(
                query=request.query,
                context=context,
                chat_history=list(request.chat_history or []),
                mode=request.chat_mode,
                plan=plan,
                plan_error=plan_error,
                session_state=session_state,
                candidate_profile_context=candidate_profile_context,
            ),
            temperature=llm_config.temperature,
        )
        for delta in llm_client.stream_complete(llm_request):
            if first_delta_ms is None:
                first_delta_ms = elapsed_since(started_at)
            answer_parts.append(delta)
            yield sse_event("answer_delta", {"text": delta})

        answer_text = "".join(answer_parts)
        answer_ms = elapsed_since(started_at)
        session_summary: str | None = None
        summary_ms: int | None = None
        telemetry = {
            "command": "StudyReview" if request.chat_mode == "study" else "InterviewReview",
            "context": context.stats,
            "interview_plan": {
                "available": plan is not None,
                "fallback_used": plan_source == "deterministic_fallback",
                "source": plan_source,
                "error": plan_error,
                "latency_ms": plan_ms,
            },
            "interview_state": request.interview_state,
            "profile_debug": profile_debug,
            "session_summary": {
                "available": session_summary is not None,
                "latency_ms": summary_ms,
                "deferred": request.chat_mode == "interview",
            },
            "generation": {
                "ttft_ms": first_delta_ms,
                "answer_ms": answer_ms,
                "output_chars": len(answer_text),
                "prompt_chars": len(context.context_text),
            },
            "scope": scope.metadata,
            "total_ms": answer_ms,
        }
        yield sse_event(
            "answer",
            {
                "answer": answer_text,
                "model": llm_config.model,
                "prompt_chars": len(context.context_text),
                "session_summary": session_summary,
            },
        )
        yield sse_event(
            "done",
            {
                "timing": {"answer_ms": answer_ms, "total_ms": answer_ms},
                "telemetry": telemetry,
                "command": "StudyReview" if request.chat_mode == "study" else "InterviewReview",
            },
        )

    @app.post("/api/interview/sessions")
    def create_interview_session(request: InterviewSessionCreateRequest) -> dict[str, Any]:
        try:
            session = interview_session_store.create_session(
                source_type=request.source_type,
                source_value=request.source_value,
                source_paths=request.source_paths,
                source_note_paths=request.source_note_paths,
                interview_plan=request.interview_plan,
                interview_state=request.interview_state,
                extra=request.extra,
            )
            return {"session": session}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/interview/sessions")
    def list_interview_sessions(limit: int = 50) -> dict[str, Any]:
        try:
            return {"sessions": interview_session_store.list_sessions(limit=limit)}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/interview/sessions/{session_id}")
    def get_interview_session(session_id: str) -> dict[str, Any]:
        try:
            return interview_session_store.load_session_bundle(session_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/interview/sessions/{session_id}/select-topic")
    def select_interview_session_topic(session_id: str, request: InterviewSessionSelectTopicRequest) -> dict[str, Any]:
        topic = request.topic.strip()
        if not topic:
            raise HTTPException(status_code=400, detail="topic is required")
        try:
            session = interview_session_store.load_session(session_id)
            plan = request.interview_plan or session.get("interview_plan") or {}
            machine = build_interview_state_machine(
                plan=plan,
                state_payload=session.get("interview_state") or None,
                session_id=session_id,
            )
            result = machine.select_topic(
                name=topic,
                reason=request.reason or "topic selected",
                source=request.source or "ui",
            )
            if not result.get("ok"):
                raise HTTPException(status_code=400, detail=result.get("message") or "topic selection failed")
            state = machine.snapshot()
            session["interview_state"] = state
            session["updated_at"] = utc_now_iso()
            interview_session_store.save_session(session)
            interview_session_store.append_trace_event(
                session_id=session_id,
                event="topic_selected",
                summary=f"topic selected: {state.get('current_topic') or ''}",
                details={"result": result, "interview_state": state},
            )
            return {"session": session, "interview_state": state, "result": result}
        except HTTPException:
            raise
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/interview/sessions/{session_id}/turns")
    def append_interview_turn(session_id: str, request: InterviewSessionAppendTurnRequest) -> dict[str, Any]:
        if not request.user_content.strip() or not request.assistant_content.strip():
            raise HTTPException(status_code=400, detail="user_content and assistant_content are required")
        try:
            result = interview_session_store.append_turn(
                session_id=session_id,
                user_content=request.user_content,
                assistant_content=request.assistant_content,
                interview_plan=request.interview_plan,
                interview_state=request.interview_state,
                source_note_paths=request.source_note_paths,
                agent_actions=request.agent_actions,
            )
            return {
                "user_message": result["user_message"],
                "assistant_message": result["assistant_message"],
                "session": result["session"],
            }
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/interview/sessions/{session_id}/turns/pending")
    def append_pending_interview_turn(session_id: str, request: InterviewSessionPendingTurnRequest) -> dict[str, Any]:
        if not request.user_content.strip():
            raise HTTPException(status_code=400, detail="user_content is required")
        try:
            result = interview_session_store.append_pending_turn(
                session_id=session_id,
                user_content=request.user_content,
                interview_plan=request.interview_plan,
                interview_state=request.interview_state,
                source_note_paths=request.source_note_paths,
            )
            return {
                "user_message": result["user_message"],
                "assistant_message": result["assistant_message"],
                "session": result["session"],
            }
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/interview/sessions/{session_id}/messages/{assistant_message_id}/complete")
    def complete_interview_assistant_message(
        session_id: str,
        assistant_message_id: str,
        request: InterviewSessionCompleteAssistantRequest,
    ) -> dict[str, Any]:
        if not request.assistant_content.strip():
            raise HTTPException(status_code=400, detail="assistant_content is required")
        try:
            result = interview_session_store.complete_assistant_message(
                session_id=session_id,
                assistant_message_id=assistant_message_id,
                assistant_content=request.assistant_content,
                interview_plan=request.interview_plan,
                interview_state=request.interview_state,
                source_note_paths=request.source_note_paths,
                agent_actions=request.agent_actions,
            )
            return {"assistant_message": result["assistant_message"], "session": result["session"]}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/interview/sessions/{session_id}/messages/{assistant_message_id}/fail")
    def fail_interview_assistant_message(
        session_id: str,
        assistant_message_id: str,
        request: InterviewSessionFailAssistantRequest,
    ) -> dict[str, Any]:
        try:
            result = interview_session_store.fail_assistant_message(
                session_id=session_id,
                assistant_message_id=assistant_message_id,
                assistant_content=request.assistant_content,
                error_type=request.error_type,
                error_message=request.error_message,
                retryable=request.retryable,
            )
            return {"assistant_message": result["assistant_message"], "session": result["session"]}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/interview/sessions/{session_id}/reviews")
    def save_interview_turn_review(session_id: str, request: InterviewSessionReviewRequest) -> dict[str, Any]:
        if not request.user_message_id or not request.assistant_message_id:
            raise HTTPException(status_code=400, detail="message ids are required")
        try:
            review = interview_session_store.save_turn_review(
                session_id=session_id,
                user_message_id=request.user_message_id,
                assistant_message_id=request.assistant_message_id,
                feedback=request.feedback,
                reference_answer=request.expression_example or request.reference_answer,
                context_note_paths=request.context_note_paths,
                profile_signals=request.profile_signals,
            )
            return {"review": review}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/interview/sessions/{session_id}/reviews/pending")
    def create_pending_interview_turn_review(session_id: str, request: InterviewSessionReviewPendingRequest) -> dict[str, Any]:
        if not request.user_message_id or not request.assistant_message_id:
            raise HTTPException(status_code=400, detail="message ids are required")
        try:
            review = interview_session_store.create_pending_review(
                session_id=session_id,
                user_message_id=request.user_message_id,
                assistant_message_id=request.assistant_message_id,
                context_note_paths=request.context_note_paths,
            )
            return {"review": review}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/interview/sessions/{session_id}/reviews/failed")
    def fail_interview_turn_review(session_id: str, request: InterviewSessionReviewFailRequest) -> dict[str, Any]:
        if not request.user_message_id or not request.assistant_message_id:
            raise HTTPException(status_code=400, detail="message ids are required")
        try:
            review = interview_session_store.mark_review_failed(
                session_id=session_id,
                user_message_id=request.user_message_id,
                assistant_message_id=request.assistant_message_id,
                error=request.error,
                context_note_paths=request.context_note_paths,
            )
            return {"review": review}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/interview/sessions/{session_id}/trace")
    def append_interview_session_trace(session_id: str, request: InterviewSessionTraceRequest) -> dict[str, Any]:
        if not request.event.strip():
            raise HTTPException(status_code=400, detail="event is required")
        try:
            entry = interview_session_store.append_trace_event(
                session_id=session_id,
                event=request.event,
                summary=request.summary or request.event,
                details=request.details,
            )
            return {"trace": entry}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/interview/sessions/{session_id}/end")
    def end_interview_session(session_id: str) -> dict[str, Any]:
        try:
            session = interview_session_store.load_session(session_id)
            if session.get("status") == "completed":
                return {"session": session}
            reviews_doc = interview_session_store.load_reviews(session_id)
            reviews = reviews_doc.get("reviews", [])
            llm_config = load_llm_config(project_root)
            llm_client = create_llm_client(llm_config)
            final_review, profile_update = commit_interview_memory(
                session_store=interview_session_store,
                profile_store=interview_profile_store,
                session=session,
                reviews=reviews,
                llm_client=llm_client,
                model=llm_config.model,
                temperature=min(llm_config.temperature, 0.1),
            )
            final_review = {
                **final_review,
                "message_count": len(session.get("messages") or []),
                "review_count": len(reviews),
                "source_note_count": (session.get("context") or {}).get("source_note_count", 0),
            }
            completed = interview_session_store.mark_completed(
                session_id=session_id,
                final_review=final_review,
                profile_update=profile_update,
            )
            return {"session": completed}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            try:
                failed = interview_session_store.mark_end_failed(session_id, str(exc))
                return {"session": failed}
            except Exception:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/interview/summary")
    def interview_summary(request: InterviewSummaryRequest) -> dict[str, Any]:
        if runtime_config.vault_path is None:
            raise HTTPException(status_code=400, detail="vault path is required for interview summary")
        if runtime_config.wiki_state_path is None:
            raise HTTPException(status_code=400, detail="wiki state path is required for interview summary")
        if not request.answer.strip():
            raise HTTPException(status_code=400, detail="answer is required")

        started_at = current_time()
        try:
            llm_config = load_llm_config(project_root)
            llm_client = create_llm_client(llm_config)
            rag_manager = None
            if request.scope_type == "search":
                rag_manager = service.build_manager(
                    SearchOptions(
                        query=request.scope_value or "",
                        mode="hybrid",
                        dense_top_k=request.dense_top_k,
                        bm25_top_k=request.hybrid_bm25_top_k,
                        rrf_k=request.rrf_k,
                    )
                )
            resolver = ScopeResolver(
                vault_root=runtime_config.vault_path,
                wiki_state_store=WikiStateStore(runtime_config.wiki_state_path),
                wiki_dir=runtime_config.wiki_dir,
                rag_manager=rag_manager,
                overview_note_threshold=runtime_config.overview_note_threshold,
            )
            scope = resolver.resolve(
                ScopeSpec(
                    type=request.scope_type,
                    value=request.scope_value or None,
                    paths=tuple(request.scope_paths or ()),
                    top_k=request.notes_top_k,
                )
            )
            context = ContextBuilder(
                vault_root=runtime_config.vault_path,
                max_chars_per_note=request.interview_max_chars_per_note,
                max_context_chars=request.interview_max_context_chars,
            ).build(scope, mode="interview_context")
            plan = interview_plan_from_dict(request.interview_plan, context=context)
            context_note_paths = [
                str(item.get("path") or item.get("relative_path") or "")
                for item in context.items
                if str(item.get("path") or item.get("relative_path") or "").strip()
            ]
            if agent_v2_coach_enabled():
                coach_app = InterviewCoachApp(build_interview_agent_runtime(llm_client))
                latest_user_answer = latest_user_content(list(request.chat_history or []))
                previous_question = previous_assistant_before_latest_user_content(list(request.chat_history or []))
                summary, _result = coach_app.run_review(
                    CoachTurnRequest(
                        session_id=request.session_id or "",
                        user_message_id=request.user_message_id or "",
                        assistant_message_id=request.assistant_message_id or "",
                        previous_interviewer_question=previous_question,
                        latest_user_answer=latest_user_answer,
                        interviewer_followup=request.answer,
                        chat_history=list(request.chat_history or []),
                        interview_plan=plan,
                        interview_state=request.interview_state,
                        context_note_paths=tuple(context_note_paths),
                        vault_root=runtime_config.vault_path,
                        session_store=interview_session_store,
                        profile_store=interview_profile_store,
                        model=llm_config.model,
                        tool_mode="auto",
                        trace_path=str(project_root / "eval-results" / "agent-debug" / "traces"),
                        temperature=min(llm_config.temperature, 0.2),
                        save_review=bool(request.session_id and request.user_message_id and request.assistant_message_id),
                    )
                )
                summary["latency_ms"] = elapsed_since(started_at)
                return summary
            summary = generate_session_summary(
                context=context,
                chat_history=list(request.chat_history or []),
                answer_text=request.answer,
                llm_client=llm_client,
                model=llm_config.model,
                plan=plan,
                temperature=min(llm_config.temperature, 0.2),
            )
            profile_signals = summary.get("profile_signals", [])
            if isinstance(profile_signals, list):
                for signal in profile_signals:
                    if isinstance(signal, dict) and not signal.get("context_note_paths"):
                        signal["context_note_paths"] = context_note_paths
            return {
                "available": True,
                "feedback": summary.get("feedback", {}),
                "expression_example": summary.get("expression_example", ""),
                "reference_answer": summary.get("expression_example", ""),
                "context_note_paths": context_note_paths,
                "profile_signals": profile_signals,
                "latency_ms": elapsed_since(started_at),
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/agent/stream")
    def agent_stream(request: AgentRequest) -> StreamingResponse:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="query is required")
        if runtime_config.vault_path is None:
            raise HTTPException(status_code=400, detail="vault path is required for chat agent")

        def event_stream():
            try:
                yield sse_event("status", {"stage": "started", "message": "request received"})

                llm_config = load_llm_config(project_root)
                llm_client = create_llm_client(llm_config)
                if request.chat_mode in {"interview", "study"}:
                    if request.chat_mode == "interview" and agent_v2_interview_enabled():
                        yield from stream_interview_agent_v2(request, llm_client, llm_config)
                        return
                    yield from stream_study_or_interview(request, llm_client, llm_config)
                    return
                if request.chat_mode == "answer" and agent_v2_librarian_enabled():
                    yield from stream_librarian_agent_v2(request, llm_client, llm_config)
                    return

                router = build_web_router(request.command, llm_client, llm_config)

                yield sse_event("status", {"stage": "pipeline_ready", "message": "pipeline initialized"})

                manager = service.build_manager(
                    SearchOptions(
                        query=request.query,
                        mode="hybrid",
                        dense_top_k=request.dense_top_k,
                        bm25_top_k=request.hybrid_bm25_top_k,
                        rrf_k=request.rrf_k,
                    )
                )
                pipeline = AgentAnswerPipeline(
                    router=router,
                    llm_client=llm_client,
                    llm_model=llm_config.model,
                    manager=manager,
                    vault_root=runtime_config.vault_path,
                    online_client=OnlineSearchClient(provider=request.online_provider),
                    config=AgentAnswerConfig(
                        notes_top_k=request.notes_top_k,
                        regex_top_k=request.regex_top_k,
                        bm25_top_k=request.bm25_top_k,
                        dense_top_k=request.dense_top_k,
                        hybrid_bm25_top_k=request.hybrid_bm25_top_k,
                        rrf_k=request.rrf_k,
                        max_chars_per_item=request.max_chars_per_item,
                        max_context_chars=request.max_context_chars,
                        online_top_k=request.online_top_k,
                        speculative_notes_search=request.speculative_notes_search,
                    ),
                    answer_temperature=llm_config.temperature,
                )

                yield sse_event("status", {"stage": "retrieval_started", "message": "routing and retrieval started"})
                retrieval = pipeline.retrieve(request.query)
                payload = agent_retrieval_result_to_dict(retrieval)
                reference_summary = build_reference_summary(payload)

                yield sse_event("router", payload["router_decision"])
                yield sse_event(
                    "status",
                    {
                        "stage": "retrieval_finished",
                        "message": reference_summary["message"],
                        "summary": reference_summary,
                    },
                )
                yield sse_event(
                    "retrieval",
                    {
                        "summary": {
                            "notes": len(payload["retrieval"]["notes"]),
                            "rg": len(payload["retrieval"]["rg"]),
                            "bm25": len(payload["retrieval"]["bm25"]),
                            "online": len(payload["retrieval"]["online"]["results"]),
                            "tool_errors": len(payload["tool_errors"]),
                            "local_references": reference_summary["local_references"],
                            "distinct_files": reference_summary["distinct_files"],
                        },
                        "reference_summary": reference_summary,
                        "tool_errors": payload["tool_errors"],
                    },
                )
                yield sse_event("context", {"items": payload["context_items"]})

                yield sse_event("status", {"stage": "answer_started", "message": "answer streaming started"})
                answer_parts: list[str] = []
                answer_started_at = current_time()
                first_delta_ms: int | None = None
                for delta in pipeline.stream_answer(
                    query=request.query,
                    decision=retrieval.router_decision,
                    context_text=retrieval.context_text,
                ):
                    if first_delta_ms is None:
                        first_delta_ms = elapsed_since(answer_started_at)
                    answer_parts.append(delta)
                    yield sse_event("answer_delta", {"text": delta})

                answer_text = "".join(answer_parts)
                answer_ms = elapsed_since(answer_started_at)
                timing = {
                    **payload["timing"],
                    "answer_ms": answer_ms,
                    "total_ms": payload["timing"].get("total_ms", 0) + answer_ms,
                }
                telemetry = {
                    **payload["telemetry"],
                    "generation": {
                        "ttft_ms": first_delta_ms,
                        "answer_ms": answer_ms,
                        "output_chars": len(answer_text),
                        "prompt_chars": None,
                    },
                    "total_ms": timing["total_ms"],
                }
                yield sse_event(
                    "answer",
                    {
                        "answer": answer_text,
                        "model": llm_config.model,
                        "prompt_chars": None,
                    },
                )
                yield sse_event("done", {"timing": timing, "telemetry": telemetry, "command": payload["command"]})
            except Exception as exc:
                yield sse_event("error", classify_runtime_error(exc))

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def build_web_router(command: str, llm_client, llm_config):
    forced_command = None if command == "auto" else ConversationCommand(command)
    return LLMIntentRouter(
        client=llm_client,
        model=llm_config.model,
        temperature=0.0,
        forced_command=forced_command,
    )


def workflow_answer_request(request: WorkflowRunRequest) -> AgentRequest:
    options = dict(request.options or {})
    query = request.user_request or request.scope.value or ""
    return AgentRequest(
        query=str(query),
        command=str(options.get("command", "auto")),
        notes_top_k=int(options.get("notes_top_k", 5)),
        regex_top_k=int(options.get("regex_top_k", 8)),
        bm25_top_k=int(options.get("bm25_top_k", 8)),
        dense_top_k=int(options.get("dense_top_k", 50)),
        hybrid_bm25_top_k=int(options.get("hybrid_bm25_top_k", 50)),
        rrf_k=int(options.get("rrf_k", 60)),
        max_chars_per_item=int(options.get("max_chars_per_item", 1000)),
        max_context_chars=int(options.get("max_context_chars", 8000)),
        online_provider=options.get("online_provider"),
        online_top_k=int(options.get("online_top_k", 5)),
        speculative_notes_search=bool(options.get("speculative_notes_search", True)),
    )


def sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def build_reference_summary(payload: dict[str, Any]) -> dict[str, Any]:
    local_items = [
        item
        for item in payload.get("context_items", [])
        if str(item.get("source_type", "")).startswith("local_")
    ]
    distinct_files = sorted(
        {
            str(item.get("path", "")).strip()
            for item in local_items
            if str(item.get("path", "")).strip()
        }
    )
    heading_set: set[str] = set()
    for item in local_items:
        heading = clean_heading(str(item.get("heading", "")))
        if heading:
            heading_set.add(heading)
    distinct_headings = sorted(heading_set)
    heading_preview = distinct_headings[:5]

    if local_items:
        message = f"Found {len(local_items)} references across {len(distinct_files)} files"
        if heading_preview:
            message += ": " + ", ".join(heading_preview)
    else:
        message = "No local references found"

    return {
        "local_references": len(local_items),
        "distinct_files": len(distinct_files),
        "distinct_headings": len(distinct_headings),
        "files": distinct_files[:10],
        "headings": heading_preview,
        "message": message,
    }


def clean_heading(heading: str) -> str:
    cleaned = re.sub(r"\s*\{[^}]*\}", "", heading).strip()
    if not cleaned:
        return ""
    short = cleaned.rsplit(" > ", 1)[-1].lstrip("#").strip()
    return short or cleaned.strip()


def current_time() -> float:
    import time

    return time.perf_counter()


def elapsed_since(started_at: float) -> int:
    import time

    return int((time.perf_counter() - started_at) * 1000)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


WIKI_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Wiki 维护</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f6f3;
      --panel: #ffffff;
      --text: #171717;
      --muted: #666666;
      --line: #d9d9d2;
      --accent: #0f766e;
      --danger: #b42318;
      --chip: #eef2f1;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.92);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 { margin: 0; font-size: 18px; }
    nav { display: flex; gap: 12px; }
    nav a { color: var(--accent); text-decoration: none; font-size: 14px; }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 22px;
    }
    .page-intro {
      margin-bottom: 16px;
    }
    .page-intro h2 {
      margin: 0 0 6px;
      font-size: 24px;
      line-height: 1.2;
    }
    .page-intro p {
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }
    .topic-summary {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 12px 0 14px;
      color: var(--muted);
      font-size: 13px;
    }
    .topic-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 16px;
    }
    .topic-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 210px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .topic-card.dirty {
      border-color: #dfb94a;
      box-shadow: 0 0 0 1px rgba(223, 185, 74, 0.16);
    }
    .topic-card.missing {
      border-style: dashed;
    }
    .topic-title {
      font-size: 17px;
      font-weight: 750;
      line-height: 1.25;
      word-break: break-word;
    }
    .topic-preview {
      color: #383838;
      line-height: 1.55;
      font-size: 14px;
      flex: 1;
    }
    .topic-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .topic-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    details.admin-tools {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      margin-top: 14px;
    }
    details.admin-tools > summary {
      cursor: pointer;
      font-weight: 700;
      color: #333;
    }
    details.admin-tools[open] > summary {
      margin-bottom: 12px;
    }
    .toolbar, .summary, .status {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 14px;
    }
    .maintenance {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 14px;
      display: grid;
      gap: 12px;
    }
    .maintenance-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    .proposal-box {
      display: none;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfbfa;
      padding: 10px;
      max-height: 420px;
      overflow: auto;
      white-space: pre-wrap;
      font-size: 12px;
      line-height: 1.5;
    }
    .proposal-box.visible { display: block; }
    .toolbar {
      display: grid;
      grid-template-columns: 1fr auto auto auto auto auto;
      gap: 10px;
      align-items: center;
    }
    input[type="search"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      background: #fff;
      color: var(--text);
    }
    label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    button {
      border: 1px solid var(--accent);
      background: var(--accent);
      color: white;
      border-radius: 6px;
      padding: 8px 11px;
      cursor: pointer;
      font-weight: 600;
    }
    button.secondary {
      background: #fff;
      color: var(--accent);
    }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfbfa;
    }
    .metric .value { font-size: 22px; font-weight: 700; }
    .metric .label { font-size: 12px; color: var(--muted); margin-top: 2px; }
    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    th, td {
      padding: 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }
    th {
      background: #f0f1ee;
      color: #333;
      font-weight: 700;
      position: sticky;
      top: 50px;
      z-index: 5;
    }
    tr:last-child td { border-bottom: 0; }
    .tag { font-weight: 700; }
    .path { color: var(--muted); word-break: break-all; }
    .pill {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 2px 8px;
      background: var(--chip);
      margin: 0 5px 5px 0;
      font-size: 12px;
      color: #333;
    }
    .pill.overview { background: #e7f3f1; color: #0f5f58; }
    .pill.generate { background: #eef0ff; color: #303986; }
    .pill.skip { background: #f5e9e7; color: var(--danger); }
    .pill.dirty { background: #fff4d7; color: #835b00; }
    .status {
      display: none;
      color: var(--muted);
      white-space: pre-wrap;
    }
    .status.visible { display: block; }
    .error { color: var(--danger); }
    .admin-hidden { display: none; }
    @media (max-width: 780px) {
      header { align-items: flex-start; gap: 10px; flex-direction: column; }
      .topic-grid { grid-template-columns: 1fr; }
      .toolbar { grid-template-columns: 1fr; }
      .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      table { display: block; overflow-x: auto; }
      th { top: 0; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Wiki 维护</h1>
    <nav>
      <a href="/chat">对话</a>
      <a href="/topics">知识主题</a>
      <a href="/organize">整理</a>
      <a href="/search">检索调试</a>
      <a href="/admin/wiki">维护</a>
      <a href="/settings">设置</a>
    </nav>
  </header>
  <main>
    <section class="page-intro">
      <h2>Wiki 维护后台</h2>
      <p>用于同步、整理 tag、调整策略和查看原始状态。日常阅读请使用知识主题页面。</p>
      <div id="topicSummary" class="topic-summary"></div>
    </section>
    <section class="toolbar">
      <input id="filterText" type="search" placeholder="搜索主题、策略、路径或维护提示..." />
      <label><input id="eligibleOnly" type="checkbox" checked /> 仅可生成</label>
      <label><input id="dirtyOnly" type="checkbox" /> 仅需更新</label>
      <label><input id="syncEmbedding" type="checkbox" checked /> 同步 RAG 索引</label>
      <button id="refreshBtn" class="secondary">刷新报告</button>
      <button id="syncBtn">同步工作区</button>
    </section>
    <section id="status" class="status"></section>
    <section id="topicCards" class="topic-grid admin-hidden"></section>
    <details class="admin-tools">
      <summary>维护工具与原始数据</summary>
      <section class="summary">
        <div class="summary-grid">
          <div class="metric"><div id="filesMetric" class="value">-</div><div class="label">files</div></div>
          <div class="metric"><div id="tagsMetric" class="value">-</div><div class="label">tags</div></div>
          <div class="metric"><div id="eligibleMetric" class="value">-</div><div class="label">eligible tags</div></div>
          <div class="metric"><div id="dirtyMetric" class="value">-</div><div class="label">dirty tags</div></div>
        </div>
      </section>
      <section class="maintenance">
        <div class="maintenance-row">
          <strong>Tag 维护</strong>
          <label><input id="includeLlmConsolidation" type="checkbox" /> 包含 LLM 建议</label>
          <button id="consolidateBtn" class="secondary">整理标签</button>
        </div>
        <div class="maintenance-row">
          <span id="selectedTagLabel" class="path">未选择 tag</span>
          <button id="refineSelectedBtn" class="secondary" disabled>分析选中 tag</button>
          <button id="policyGenerateBtn" class="secondary" disabled>设为知识页</button>
          <button id="policyOverviewBtn" class="secondary" disabled>设为导航页</button>
          <button id="policySkipBtn" class="secondary" disabled>设为跳过</button>
        </div>
        <pre id="proposalBox" class="proposal-box"></pre>
      </section>
      <table>
        <thead>
          <tr>
            <th>Tag</th>
          <th>笔记数</th>
            <th>Policy</th>
            <th>Evidence</th>
            <th>Hints</th>
          <th>Wiki 路径</th>
          <th>操作</th>
          </tr>
        </thead>
        <tbody id="tagRows"></tbody>
      </table>
    </details>
  </main>
  <script>
    let report = null;
    let selectedTag = "";
    const $ = (id) => document.getElementById(id);

    $("refreshBtn").addEventListener("click", loadReport);
    $("syncBtn").addEventListener("click", syncChangedNotes);
    $("consolidateBtn").addEventListener("click", consolidateTags);
    $("refineSelectedBtn").addEventListener("click", () => refineTag(selectedTag));
    $("policyGenerateBtn").addEventListener("click", () => setPolicy(selectedTag, "generate"));
    $("policyOverviewBtn").addEventListener("click", () => setPolicy(selectedTag, "overview"));
    $("policySkipBtn").addEventListener("click", () => setPolicy(selectedTag, "skip"));
    $("filterText").addEventListener("input", renderReport);
    $("eligibleOnly").addEventListener("change", renderReport);
    $("dirtyOnly").addEventListener("change", renderReport);
    loadReport();

    async function loadReport() {
      setStatus("正在加载 Wiki 报告...");
      try {
        const response = await fetch("/api/wiki/report");
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Failed to load report");
        report = data;
        renderReport();
        setStatus(`已加载 ${data.tags} 个 tag，其中 ${data.eligible_tags} 个可生成。`);
      } catch (error) {
        setStatus(error.message || String(error), true);
      }
    }

    function renderSummary() {
      $("filesMetric").textContent = report?.files ?? "-";
      $("tagsMetric").textContent = report?.tags ?? "-";
      $("eligibleMetric").textContent = report?.eligible_tags ?? "-";
      $("dirtyMetric").textContent = report?.dirty_tags ?? "-";
      const rows = getFilteredRows();
      const dirty = rows.filter((row) => row.dirty).length;
      const missing = rows.filter((row) => row.eligible && !row.wiki_exists).length;
      const generated = rows.filter((row) => row.wiki_exists).length;
      const failed = rows.filter((row) => row.last_error).length;
      $("topicSummary").innerHTML = [
        failed ? `<span class="pill skip">${failed} 个主题合成失败</span>` : "",
        `<span class="pill dirty">${dirty} 个主题需要更新</span>`,
        `<span class="pill">${missing} 个主题尚未合成</span>`,
        `<span class="pill">${generated} 个主题可阅读</span>`
      ].filter(Boolean).join("");
    }

    function renderReport() {
      renderSummary();
      renderTopics();
      renderRows();
    }

    function getFilteredRows() {
      if (!report) {
        return [];
      }
      const query = $("filterText").value.trim().toLowerCase();
      const eligibleOnly = $("eligibleOnly").checked;
      const dirtyOnly = $("dirtyOnly").checked;
      return report.tag_rows.filter((row) => {
        if (eligibleOnly && !row.eligible) return false;
        if (dirtyOnly && !row.dirty) return false;
        if (!query) return true;
        const text = JSON.stringify(row).toLowerCase();
        return text.includes(query);
      }).sort(topicSort);
    }

    function renderTopics() {
      const container = $("topicCards");
      const rows = getFilteredRows();
      if (!rows.length) {
        container.innerHTML = `<article class="topic-card missing"><div class="topic-title">没有匹配的主题</div><div class="topic-preview">调整搜索条件或关闭“仅可生成”后再看。</div></article>`;
        return;
      }
      container.innerHTML = rows.map(renderTopicCard).join("");
      container.querySelectorAll("[data-synthesize-tag]").forEach((button) => {
        button.addEventListener("click", () => synthesizeTag(button.dataset.synthesizeTag));
      });
      container.querySelectorAll("[data-select-tag]").forEach((button) => {
        button.addEventListener("click", () => selectTag(button.dataset.selectTag));
      });
      container.querySelectorAll("[data-copy-path]").forEach((button) => {
        button.addEventListener("click", () => navigator.clipboard.writeText(button.dataset.copyPath || ""));
      });
      container.querySelectorAll("[data-read-tag]").forEach((button) => {
        button.addEventListener("click", () => {
          window.open(`/api/wiki/read?tag=${encodeURIComponent(button.dataset.readTag)}`, "_blank");
        });
      });
    }

    function renderRows() {
      const tbody = $("tagRows");
      if (!report) {
        tbody.innerHTML = "";
        return;
      }
      const rows = getFilteredRows();
      tbody.innerHTML = rows.map(renderRow).join("");
      tbody.querySelectorAll("[data-synthesize-tag]").forEach((button) => {
        button.addEventListener("click", () => synthesizeTag(button.dataset.synthesizeTag));
      });
      tbody.querySelectorAll("[data-select-tag]").forEach((button) => {
        button.addEventListener("click", () => selectTag(button.dataset.selectTag));
      });
      tbody.querySelectorAll("[data-refine-tag]").forEach((button) => {
        button.addEventListener("click", () => refineTag(button.dataset.refineTag));
      });
      tbody.querySelectorAll("[data-copy-path]").forEach((button) => {
        button.addEventListener("click", () => navigator.clipboard.writeText(button.dataset.copyPath || ""));
      });
    }

    function renderTopicCard(row) {
      const policy = escapeHtml(row.wiki_policy || "");
      const status = row.last_error ? "合成失败" : (row.dirty ? "需要更新" : (row.wiki_exists ? "已生成" : "未合成"));
      const preview = row.wiki_preview || "尚未生成知识页。可以先生成，再把它作为跨笔记主题入口阅读。";
      const cardClass = row.last_error ? "missing" : (row.dirty ? "dirty" : (!row.wiki_exists ? "missing" : ""));
      const errorBlock = row.last_error
        ? `<div class="path error">${escapeHtml(row.last_error_type || "Error")}: ${escapeHtml(row.last_error)} · retry=${row.retry_count || 0}</div>`
        : "";
      const readButton = row.wiki_exists
        ? `<button class="secondary" data-read-tag="${escapeHtml(row.tag)}">阅读</button>`
        : `<button class="secondary" disabled>阅读</button>`;
      return `
        <article class="topic-card ${cardClass}">
          <div class="topic-title">${formatTagTitle(row.tag)}</div>
          <div class="topic-preview">${escapeHtml(preview)}</div>
          <div class="topic-meta">
            <span class="pill">${row.note_count} 篇笔记</span>
            <span class="pill ${policy}">${policyLabel(row.wiki_policy)}</span>
            <span class="pill ${row.last_error ? "skip" : (row.dirty ? "dirty" : "")}">${status}</span>
          </div>
          ${errorBlock}
          <div class="path">${escapeHtml(row.wiki_path || "")}</div>
          <div class="topic-actions">
            ${readButton}
            <button data-synthesize-tag="${escapeHtml(row.tag)}" ${row.eligible ? "" : "disabled"}>${row.wiki_exists ? "重新合成" : "生成"}</button>
            <button class="secondary" data-select-tag="${escapeHtml(row.tag)}">维护</button>
            ${row.wiki_path ? `<button class="secondary" data-copy-path="${escapeHtml(row.wiki_path)}">复制路径</button>` : ""}
          </div>
        </article>
      `;
    }

    function topicSort(a, b) {
      const rank = (row) => {
        if (row.dirty && row.wiki_exists) return 0;
        if (row.eligible && !row.wiki_exists) return 1;
        if (row.wiki_exists && row.wiki_policy === "generate") return 2;
        if (row.wiki_exists && row.wiki_policy === "overview") return 3;
        return 4;
      };
      const rankDelta = rank(a) - rank(b);
      if (rankDelta !== 0) return rankDelta;
      return (b.note_count || 0) - (a.note_count || 0) || String(a.tag).localeCompare(String(b.tag));
    }

    function formatTagTitle(tag) {
      return escapeHtml(String(tag || "").split("/").filter(Boolean).join(" / "));
    }

    function policyLabel(policy) {
      if (policy === "overview") return "导航页";
      if (policy === "generate") return "知识页";
      if (policy === "skip") return "跳过";
      return escapeHtml(policy || "-");
    }

    function renderRow(row) {
      const policy = escapeHtml(row.wiki_policy || "");
      const source = escapeHtml(row.wiki_policy_source || "");
      const evidence = Object.entries(row.evidence_counts || {})
        .map(([name, count]) => `<span class="pill">${escapeHtml(name)}: ${count}</span>`)
        .join("");
      const hints = (row.review_hints || [])
        .map((hint) => `<span class="pill dirty">${escapeHtml(hint)}</span>`)
        .join("");
      const errorHint = row.last_error
        ? `<div class="path error">${escapeHtml(row.last_error_type || "Error")}: ${escapeHtml(row.last_error)}<br/>retry=${row.retry_count || 0}, retryable=${row.retryable ? "yes" : "no"}</div>`
        : "";
      const wikiPath = row.wiki_path || "";
      const actionDisabled = row.eligible ? "" : "disabled";
      return `
        <tr>
          <td><div class="tag">${escapeHtml(row.tag)}</div>${row.last_error ? '<span class="pill skip">failed</span>' : ""}${row.dirty ? '<span class="pill dirty">dirty</span>' : ""}</td>
          <td>${row.note_count}</td>
          <td><span class="pill ${policy}">${policy}</span><div class="path">${source}</div></td>
          <td>${evidence || '<span class="path">none</span>'}</td>
          <td>${errorHint}${hints || (!errorHint ? '<span class="path">none</span>' : "")}</td>
          <td>
            <div class="path">${escapeHtml(wikiPath || "(not generated)")}</div>
            ${wikiPath ? `<button class="secondary" data-copy-path="${escapeHtml(wikiPath)}">复制</button>` : ""}
          </td>
          <td>
            <button class="secondary" data-select-tag="${escapeHtml(row.tag)}">选择</button>
            <button class="secondary" data-refine-tag="${escapeHtml(row.tag)}">分析</button>
            <button data-synthesize-tag="${escapeHtml(row.tag)}" ${actionDisabled}>合成</button>
          </td>
        </tr>
      `;
    }

    function selectTag(tag) {
      selectedTag = tag || "";
      $("selectedTagLabel").textContent = selectedTag ? `已选择：${selectedTag}` : "未选择 tag";
      const disabled = !selectedTag;
      $("refineSelectedBtn").disabled = disabled;
      $("policyGenerateBtn").disabled = disabled;
      $("policyOverviewBtn").disabled = disabled;
      $("policySkipBtn").disabled = disabled;
    }

    async function consolidateTags() {
      setStatus("正在生成标签整理建议...");
      setButtonsDisabled(true);
      try {
        const response = await fetch("/api/wiki/consolidate-tags", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({include_llm: $("includeLlmConsolidation").checked})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Failed to consolidate tags");
        showProposal(data);
        const deterministic = data.deterministic_proposals?.length ?? 0;
        const llm = data.llm_proposals?.length ?? 0;
        setStatus(`标签整理建议已生成。deterministic=${deterministic}, llm=${llm}`);
      } catch (error) {
        setStatus(error.message || String(error), true);
      } finally {
        setButtonsDisabled(false);
        selectTag(selectedTag);
      }
    }

    async function refineTag(tag) {
      if (!tag) return;
      selectTag(tag);
      setStatus(`正在分析 tag：${tag}...`);
      setButtonsDisabled(true);
      try {
        const response = await fetch("/api/wiki/refine-tag", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({tag})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Failed to refine tag");
        showProposal(data);
        const proposal = data.tag_refinement || {};
        setStatus(`分析完成。decision=${proposal.decision || "-"}, problem=${proposal.problem || "-"}`);
      } catch (error) {
        setStatus(error.message || String(error), true);
      } finally {
        setButtonsDisabled(false);
        selectTag(selectedTag);
      }
    }

    async function setPolicy(tag, policy) {
      if (!tag) return;
      setStatus(`正在设置 ${tag} 的策略为 ${policy}...`);
      setButtonsDisabled(true);
      try {
        const response = await fetch("/api/wiki/set-policy", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({tag, policy})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Failed to set policy");
        report = data.report;
        renderReport();
        selectTag(tag);
        setStatus(`策略已更新：${tag} -> ${policy}`);
      } catch (error) {
        setStatus(error.message || String(error), true);
      } finally {
        setButtonsDisabled(false);
        selectTag(selectedTag);
      }
    }

    function showProposal(data) {
      const box = $("proposalBox");
      box.textContent = JSON.stringify(data, null, 2);
      box.className = "proposal-box visible";
    }

    async function synthesizeTag(tag) {
      if (!tag) return;
      setStatus(`正在为 ${tag} 合成 wiki...`);
      setButtonsDisabled(true);
      try {
        const response = await fetch("/api/wiki/synthesize-tag", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({tag, force: true})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Failed to synthesize tag");
        const task = await waitForTask(data.task_id, "正在合成 wiki");
        if (task.status !== "succeeded") {
          throw new Error(task.error || "Wiki synthesis task failed");
        }
        const payload = task.result || {};
        report = payload.report;
        renderReport();
        const result = payload.result || {};
        const failedTags = result.failed_tags || [];
        const retriedTags = result.retried_tags || [];
        const failureText = failedTags.length
          ? `；失败：${failedTags.map((item) => `${item.tag}(${item.error_type}, attempts=${item.attempts})`).join("，")}`
          : "";
        const retryText = retriedTags.length
          ? `；重试后成功：${retriedTags.map((item) => `${item.tag}(${item.attempts})`).join("，")}`
          : "";
        setStatus(`已为 ${tag} 合成 ${result.generated} 个 wiki。skipped=${result.skipped}, failed=${result.failed}${retryText}${failureText}`, failedTags.length > 0);
      } catch (error) {
        setStatus(error.message || String(error), true);
      } finally {
        setButtonsDisabled(false);
        selectTag(selectedTag);
      }
    }

    async function syncChangedNotes() {
      setStatus("正在同步 RAG 索引和 wiki tags...");
      setButtonsDisabled(true);
      try {
        const response = await fetch("/api/wiki/sync", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            force: false,
            limit: null,
            include_embedding: $("syncEmbedding").checked,
            allow_full_rebuild: false
          })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Failed to sync workspace");
        const task = await waitForTask(data.task_id, "正在同步工作区");
        if (task.status !== "succeeded") {
          throw new Error(task.error || "Workspace sync task failed");
        }
        const payload = task.result || {};
        report = payload.report;
        renderReport();
        const result = payload.result || {};
        setStatus(formatSyncResult(result));
      } catch (error) {
        setStatus(error.message || String(error), true);
      } finally {
        setButtonsDisabled(false);
      }
    }

    function formatSyncResult(result) {
      const rag = result.rag;
      const wiki = result.wiki || {};
      const lines = ["同步完成。"];
      if (rag) {
        lines.push(
          `RAG: mode=${rag.mode ?? "-"}, added=${rag.added_files ?? "-"}, modified=${rag.modified_files ?? "-"}, deleted=${rag.deleted_files ?? "-"}, embedded_chunks=${rag.embedded_chunks ?? "-"}, total_chunks=${rag.total_chunks ?? "-"}`
        );
        if (rag.reason) {
          lines.push(`RAG note: ${rag.reason}`);
        }
      } else {
        lines.push("RAG: skipped by request");
      }
      lines.push(
        `Wiki tags: markdown_files=${wiki.markdown_files ?? "-"}, tagged=${wiki.tagged ?? "-"}, skipped=${wiki.skipped ?? "-"}, failed=${wiki.failed ?? "-"}, deleted=${wiki.deleted ?? "-"}, tags=${wiki.tags ?? "-"}`
      );
      return lines.join("\n");
    }

    async function waitForTask(taskId, label) {
      if (!taskId) throw new Error("Task id missing");
      while (true) {
        const response = await fetch(`/api/tasks/${encodeURIComponent(taskId)}`);
        const task = await response.json();
        if (!response.ok) throw new Error(task.detail || "Task status request failed");
        const lastEvent = (task.events || []).slice(-1)[0];
        const step = lastEvent ? lastEvent.event : task.current_step;
        setStatus(`${label}... ${step || task.status}`);
        if (["succeeded", "failed", "cancelled"].includes(task.status)) {
          return task;
        }
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    }

    function setButtonsDisabled(disabled) {
      document.querySelectorAll("button").forEach((button) => {
        if (button.id !== "refreshBtn") button.disabled = disabled;
      });
    }

    function setStatus(message, isError = false) {
      const node = $("status");
      node.textContent = message;
      node.className = `status visible ${isError ? "error" : ""}`;
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }
  </script>
</body>
</html>
"""


TOPICS_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>知识主题</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --text: #171717;
      --muted: #666666;
      --line: #d9d9d2;
      --accent: #0f766e;
      --danger: #b42318;
      --warn-bg: #fff7df;
      --chip: #eef2f1;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.94);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 { margin: 0; font-size: 18px; }
    nav { display: flex; gap: 12px; flex-wrap: wrap; }
    nav a { color: var(--accent); text-decoration: none; font-size: 14px; }
    main {
      width: min(920px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 26px 0 48px;
    }
    .hero {
      margin-bottom: 18px;
    }
    .hero h2 {
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.16;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
    }
    .notice {
      margin-top: 14px;
      padding: 10px 12px;
      border: 1px solid #ead18a;
      background: var(--warn-bg);
      border-radius: 8px;
      color: #694900;
      display: none;
    }
    .notice.visible { display: block; }
    .controls {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 16px;
      align-items: center;
    }
    .tabs {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .tab {
      border: 1px solid var(--line);
      background: #fff;
      color: #333;
      border-radius: 999px;
      padding: 7px 11px;
      cursor: pointer;
      font-weight: 650;
    }
    .tab.active {
      border-color: var(--accent);
      background: #e7f3f1;
      color: #0f5f58;
    }
    input[type="search"] {
      width: min(280px, 100%);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      background: #fff;
      color: var(--text);
    }
    .topics {
      display: grid;
      gap: 22px;
    }
    .topic-group {
      display: grid;
      gap: 10px;
    }
    .group-title {
      margin: 4px 0 0;
      font-size: 14px;
      line-height: 1.4;
      color: #424242;
      font-weight: 760;
    }
    .group-title span {
      color: var(--muted);
      font-weight: 600;
    }
    .topic {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 14px;
      align-items: center;
    }
    .topic.needs-update { border-color: #dfb94a; }
    .topic.missing { border-style: dashed; }
    .topic-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 8px;
    }
    .topic-title {
      font-size: 17px;
      font-weight: 760;
      margin: 0;
      line-height: 1.25;
      word-break: break-word;
    }
    .topic-badges {
      display: inline-flex;
      flex-wrap: wrap;
      gap: 6px;
      justify-content: flex-end;
      flex-shrink: 0;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 3px 8px;
      background: var(--chip);
      color: #3f3f3f;
      font-size: 12px;
      line-height: 1.35;
      white-space: nowrap;
    }
    .badge.update {
      background: #fff1bf;
      color: #704c00;
    }
    .badge.missing {
      background: #f3f3ef;
      color: #555;
    }
    .badge.generated {
      background: #e7f3f1;
      color: #0f5f58;
    }
    .wiki-link {
      color: #3f3f3f;
      font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      word-break: break-all;
    }
    .actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    button, .open-link {
      border: 1px solid var(--accent);
      background: var(--accent);
      color: white;
      border-radius: 7px;
      padding: 8px 11px;
      cursor: pointer;
      font-weight: 700;
      text-decoration: none;
      font-size: 13px;
      white-space: nowrap;
    }
    button.secondary, .open-link.secondary {
      background: #fff;
      color: var(--accent);
    }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    .empty {
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 24px;
      color: var(--muted);
      text-align: center;
      background: rgba(255, 255, 255, 0.6);
    }
    .status {
      display: none;
      margin-bottom: 12px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--muted);
      white-space: pre-wrap;
    }
    .status.visible { display: block; }
    .status.error { color: var(--danger); }
    @media (max-width: 760px) {
      header { flex-direction: column; align-items: flex-start; }
      .controls { flex-direction: column; align-items: stretch; }
      input[type="search"] { width: 100%; }
      .topic { grid-template-columns: 1fr; }
      .topic-head { flex-direction: column; }
      .topic-badges { justify-content: flex-start; }
      .actions { justify-content: flex-start; }
    }
  </style>
</head>
<body>
  <header>
    <h1>知识主题</h1>
    <nav>
      <a href="/chat">对话</a>
      <a href="/topics">知识主题</a>
      <a href="/organize">整理</a>
      <a href="/search">检索调试</a>
      <a href="/admin/wiki">维护</a>
      <a href="/settings">设置</a>
    </nav>
  </header>
  <main>
    <section class="hero">
      <h2>我的知识主题</h2>
      <p>这里只展示可以阅读或合成的主题页。完整内容在 Obsidian 中阅读，维护操作放在后台。</p>
      <div id="notice" class="notice"></div>
    </section>
    <section class="controls">
      <div class="tabs">
        <button class="tab active" data-filter="all">全部</button>
        <button class="tab" data-filter="dirty">需要更新</button>
        <button class="tab" data-filter="missing">未合成</button>
        <button class="tab" data-filter="generated">已生成</button>
      </div>
      <input id="topicSearch" type="search" placeholder="搜索主题..." />
    </section>
    <section id="status" class="status"></section>
    <section id="topics" class="topics"></section>
  </main>
  <script>
    let report = null;
    let activeFilter = "all";
    const $ = (id) => document.getElementById(id);

    document.querySelectorAll("[data-filter]").forEach((button) => {
      button.addEventListener("click", () => {
        activeFilter = button.dataset.filter;
        document.querySelectorAll("[data-filter]").forEach((node) => node.classList.remove("active"));
        button.classList.add("active");
        renderTopics();
      });
    });
    $("topicSearch").addEventListener("input", renderTopics);
    loadReport();

    async function loadReport() {
      setStatus("正在加载知识主题...");
      try {
        const response = await fetch("/api/wiki/report");
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "知识主题加载失败");
        report = data;
        renderNotice();
        renderTopics();
        setStatus("");
      } catch (error) {
        setStatus(error.message || String(error), true);
      }
    }

    function renderNotice() {
      const rows = topicRows();
      const dirty = rows.filter((row) => row.dirty && row.wiki_exists).length;
      const missing = rows.filter((row) => row.eligible && !row.wiki_exists).length;
      const notice = $("notice");
      const parts = [];
      if (dirty) parts.push(`${dirty} 个主题需要更新`);
      if (missing) parts.push(`${missing} 个主题尚未合成`);
      if (!parts.length) {
        notice.className = "notice";
        notice.textContent = "";
        return;
      }
      notice.textContent = parts.join("；") + "。需要更新表示源笔记已变更，重新合成后 wiki 会刷新。";
      notice.className = "notice visible";
    }

    function topicRows() {
      if (!report) return [];
      return report.tag_rows
        .filter((row) => row.eligible || row.wiki_exists)
        .sort(topicSort);
    }

    function filteredRows() {
      const query = $("topicSearch").value.trim().toLowerCase();
      return topicRows().filter((row) => {
        if (activeFilter === "dirty" && !(row.dirty && row.wiki_exists)) return false;
        if (activeFilter === "missing" && !(!row.wiki_exists && row.eligible)) return false;
        if (activeFilter === "generated" && !row.wiki_exists) return false;
        if (!query) return true;
        return String(row.tag || "").toLowerCase().includes(query)
          || String(row.wiki_path || "").toLowerCase().includes(query);
      });
    }

    function renderTopics() {
      const container = $("topics");
      const rows = filteredRows();
      if (!rows.length) {
        container.innerHTML = `<div class="empty">没有匹配的知识主题。</div>`;
        return;
      }
      container.innerHTML = groupedRows(rows).map(renderGroup).join("");
      container.querySelectorAll("[data-synthesize-tag]").forEach((button) => {
        button.addEventListener("click", () => synthesizeTag(button.dataset.synthesizeTag));
      });
    }

    function groupedRows(rows) {
      const groups = [
        {key: "dirty", title: "需要更新", rows: []},
        {key: "missing", title: "尚未合成", rows: []},
        {key: "generated", title: "已生成", rows: []}
      ];
      for (const row of rows) {
        if (row.dirty && row.wiki_exists) {
          groups[0].rows.push(row);
        } else if (row.eligible && !row.wiki_exists) {
          groups[1].rows.push(row);
        } else if (row.wiki_exists) {
          groups[2].rows.push(row);
        }
      }
      return groups.filter((group) => group.rows.length);
    }

    function renderGroup(group) {
      return `
        <section class="topic-group">
          <h3 class="group-title">${group.title} <span>${group.rows.length}</span></h3>
          ${group.rows.map(renderTopic).join("")}
        </section>
      `;
    }

    function renderTopic(row) {
      const title = formatTagTitle(row.tag);
      const klass = row.wiki_exists ? (row.dirty ? "needs-update" : "") : "missing";
      const wikiLink = row.wiki_exists
        ? `[[${escapeHtml(row.wiki_path)}]]`
        : "合成后会生成 Obsidian wiki 文件";
      const openHref = row.wiki_exists ? obsidianUrl(row.wiki_path) : "";
      const statusBadge = row.wiki_exists
        ? (row.dirty ? `<span class="badge update">需要更新</span>` : `<span class="badge generated">已生成</span>`)
        : `<span class="badge missing">未合成</span>`;
      return `
        <article class="topic ${klass}">
          <div>
            <div class="topic-head">
              <div class="topic-title">${title}</div>
              <div class="topic-badges">
                <span class="badge">${row.note_count} 篇笔记</span>
                ${statusBadge}
              </div>
            </div>
            <div class="wiki-link">${wikiLink}</div>
          </div>
          <div class="actions">
            ${row.wiki_exists ? `<a class="open-link" href="${openHref}">在 Obsidian 中打开</a>` : ""}
            <button class="${row.wiki_exists ? "secondary" : ""}" data-synthesize-tag="${escapeHtml(row.tag)}" ${row.eligible ? "" : "disabled"}>${row.wiki_exists ? "重新合成" : "合成 wiki"}</button>
          </div>
        </article>
      `;
    }

    async function synthesizeTag(tag) {
      if (!tag) return;
      setStatus(`正在合成 ${tag}...`);
      setButtonsDisabled(true);
      try {
        const response = await fetch("/api/wiki/synthesize-tag", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({tag, force: true})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Failed to synthesize topic");
        const task = await waitForTask(data.task_id, "正在合成 wiki");
        if (task.status !== "succeeded") {
          throw new Error(task.error || "Wiki synthesis task failed");
        }
        const payload = task.result || {};
        report = payload.report;
        renderNotice();
        renderTopics();
        const result = payload.result || {};
        const failedTags = result.failed_tags || [];
        const retriedTags = result.retried_tags || [];
        const retryText = retriedTags.length
          ? `；重试后成功：${retriedTags.map((item) => `${item.tag}(${item.attempts})`).join("，")}`
          : "";
        const failureText = failedTags.length
          ? `；失败：${failedTags.map((item) => `${item.tag}(${item.error_type}, attempts=${item.attempts})`).join("，")}`
          : "";
        setStatus(`已更新 ${tag}${retryText}${failureText}`, failedTags.length > 0);
      } catch (error) {
        setStatus(error.message || String(error), true);
      } finally {
        setButtonsDisabled(false);
      }
    }

    async function waitForTask(taskId, label) {
      if (!taskId) throw new Error("Task id missing");
      while (true) {
        const response = await fetch(`/api/tasks/${encodeURIComponent(taskId)}`);
        const task = await response.json();
        if (!response.ok) throw new Error(task.detail || "Task status request failed");
        const lastEvent = (task.events || []).slice(-1)[0];
        const step = lastEvent ? lastEvent.event : task.current_step;
        setStatus(`${label}... ${step || task.status}`);
        if (["succeeded", "failed", "cancelled"].includes(task.status)) {
          return task;
        }
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    }

    function topicSort(a, b) {
      const rank = (row) => {
        if (row.dirty && row.wiki_exists) return 0;
        if (row.eligible && !row.wiki_exists) return 1;
        if (row.wiki_exists) return 2;
        return 3;
      };
      const rankDelta = rank(a) - rank(b);
      if (rankDelta !== 0) return rankDelta;
      return (b.note_count || 0) - (a.note_count || 0) || String(a.tag).localeCompare(String(b.tag));
    }

    function obsidianUrl(path) {
      const vault = report?.obsidian_vault_name || "";
      return `obsidian://open?vault=${encodeURIComponent(vault)}&file=${encodeURIComponent(path || "")}`;
    }

    function formatTagTitle(tag) {
      return escapeHtml(String(tag || "").split("/").filter(Boolean).join(" / "));
    }

    function setButtonsDisabled(disabled) {
      document.querySelectorAll("button").forEach((button) => {
        if (!button.classList.contains("tab")) button.disabled = disabled;
      });
    }

    function setStatus(message, isError = false) {
      const node = $("status");
      node.textContent = message;
      node.className = message ? `status visible ${isError ? "error" : ""}` : "status";
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }
  </script>
</body>
</html>
"""


AUDIT_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>知识整理</title>
  <style>
    :root {
      --bg: #f7f7f4;
      --panel: #ffffff;
      --text: #181816;
      --muted: #67645d;
      --line: #d9d7ce;
      --accent: #0f766e;
      --accent-soft: #e7f3f1;
      --danger: #b42318;
      --warn: #986a00;
      --warn-bg: #fff7df;
      --code: #f1f0ea;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 18px;
      padding: 16px 24px;
      background: rgba(255, 255, 255, 0.94);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    h1 { margin: 0 0 6px; font-size: 22px; line-height: 1.2; }
    header p { margin: 0; color: var(--muted); font-size: 14px; line-height: 1.5; }
    nav { display: flex; gap: 12px; flex-wrap: wrap; }
    nav a { color: var(--accent); text-decoration: none; font-size: 14px; font-weight: 650; }
    main {
      width: min(1080px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 48px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }
    .form-grid {
      display: grid;
      grid-template-columns: 180px minmax(240px, 1fr) 140px auto;
      gap: 12px;
      align-items: end;
    }
    label { display: block; color: var(--muted); font-size: 13px; margin-bottom: 6px; }
    select, input, button {
      font: inherit;
    }
    select, input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      background: #fff;
      color: var(--text);
    }
    button {
      border: 1px solid var(--accent);
      background: var(--accent);
      color: #fff;
      border-radius: 6px;
      padding: 9px 14px;
      cursor: pointer;
      font-weight: 700;
    }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .hint {
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    .status {
      display: none;
      margin-bottom: 16px;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 12px 14px;
      color: var(--muted);
    }
    .status.visible { display: block; }
    .status.error { border-color: #f0b8b8; color: var(--danger); background: #fff6f6; }
    .summary {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    .metric strong {
      display: block;
      font-size: 24px;
      line-height: 1.1;
      margin-bottom: 6px;
    }
    .metric span { color: var(--muted); font-size: 13px; }
    .issues {
      display: grid;
      gap: 10px;
    }
    .issue {
      background: var(--panel);
      border: 1px solid var(--line);
      border-left: 4px solid var(--warn);
      border-radius: 8px;
      padding: 12px 14px;
    }
    .issue.error { border-left-color: var(--danger); }
    .issue.info { border-left-color: var(--accent); }
    .issue-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 6px;
    }
    .issue-code {
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      background: var(--code);
      border-radius: 4px;
      padding: 2px 5px;
      font-size: 12px;
    }
    .issue-path {
      color: var(--muted);
      font-size: 13px;
      word-break: break-all;
    }
    .issue-message {
      margin: 7px 0 0;
      line-height: 1.55;
    }
    .empty {
      background: var(--panel);
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 18px;
      color: var(--muted);
      text-align: center;
    }
    .markdown {
      white-space: pre-wrap;
      background: var(--code);
      border-radius: 8px;
      padding: 12px;
      overflow: auto;
      color: #333;
      font-size: 13px;
      line-height: 1.5;
      display: none;
    }
    details { margin-top: 14px; }
    summary { cursor: pointer; color: var(--accent); font-weight: 700; }
    @media (max-width: 820px) {
      header { display: block; }
      nav { margin-top: 12px; }
      .form-grid { grid-template-columns: 1fr; }
      .summary { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>知识整理</h1>
      <p>先做结构检查，再给出整理建议。这个入口会把确定性审计和 LLM 整理结果放到同一份报告里。</p>
    </div>
    <nav>
      <a href="/chat">对话</a>
      <a href="/topics">知识主题</a>
      <a href="/organize">整理</a>
      <a href="/search">检索调试</a>
      <a href="/admin/wiki">维护</a>
      <a href="/settings">设置</a>
    </nav>
  </header>
  <main>
    <section class="panel">
      <div class="form-grid">
        <div>
          <label for="scopeType">范围</label>
          <select id="scopeType">
            <option value="tag" selected>Tag</option>
            <option value="folder">文件夹</option>
            <option value="all_vault">全部 Vault</option>
            <option value="selected_notes">指定笔记</option>
          </select>
        </div>
        <div>
          <label for="scopeValue">Tag / 文件夹</label>
          <input id="scopeValue" placeholder="例如：java/servlet" />
        </div>
        <div>
          <label for="maxIssues">最多问题数</label>
          <input id="maxIssues" type="number" min="1" max="1000" value="50" />
        </div>
        <div>
          <button id="runBtn">开始整理</button>
        </div>
      </div>
      <div class="hint">
        指定笔记模式下，在输入框中按行填写相对路径。全部 Vault 可能输出较多，建议先限制最多问题数。
      </div>
    </section>

    <div id="status" class="status"></div>
    <section id="summary" class="summary"></section>
    <section id="review" class="issues"></section>
    <section id="issues" class="issues"></section>
    <details id="markdownBox" style="display:none;">
      <summary>查看 Markdown 报告</summary>
      <pre id="markdown" class="markdown"></pre>
    </details>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);

    $("scopeType").addEventListener("change", updateScopeHint);
    $("runBtn").addEventListener("click", runAudit);
    updateScopeHint();

    async function runAudit() {
      const scopeType = $("scopeType").value;
      const value = $("scopeValue").value.trim();
      const maxIssues = Number($("maxIssues").value || 50);
      const paths = scopeType === "selected_notes"
        ? value.split(/\r?\n/).map((item) => item.trim()).filter(Boolean)
        : [];
      if ((scopeType === "tag" || scopeType === "folder") && !value) {
        setStatus("请填写 tag 或文件夹。", true);
        return;
      }
      if (scopeType === "selected_notes" && paths.length === 0) {
        setStatus("请填写至少一条笔记相对路径。", true);
        return;
      }

      setBusy(true);
      clearResults();
      setStatus("正在提交整理任务...");
      try {
        const response = await fetch("/api/workflows/run", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            task_type: "organize",
            scope: {
              type: scopeType,
              value: scopeType === "selected_notes" ? null : value,
              paths,
            },
            options: {
              max_issues: maxIssues,
              max_notes: 8,
              max_chars_per_note: 1800,
              review_mode: "auto"
            }
          })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "提交整理失败");
        const task = await waitForTask(data.task_id);
        if (task.status !== "succeeded") {
          throw new Error(task.error || "整理任务失败");
        }
        const workflow = task.result?.workflow || {};
        const organize = workflow.output?.organize || {};
        const audit = organize.audit || workflow.output?.audit || {};
        const review = organize.review || null;
        renderAudit(audit);
        renderReview(review);
        setStatus("整理完成。");
      } catch (error) {
        setStatus(error.message || String(error), true);
      } finally {
        setBusy(false);
      }
    }

    async function waitForTask(taskId) {
      while (true) {
        const response = await fetch(`/api/tasks/${encodeURIComponent(taskId)}`);
        const task = await response.json();
        if (!response.ok) throw new Error(task.detail || "读取任务状态失败");
        const lastEvent = (task.events || []).slice(-1)[0];
        const step = lastEvent ? lastEvent.event : task.current_step;
        setStatus(`任务状态：${task.status}${step ? ` / ${step}` : ""}`);
        if (!["queued", "running"].includes(task.status)) return task;
        await new Promise((resolve) => setTimeout(resolve, 800));
      }
    }

    function renderAudit(audit) {
      const summary = audit.summary || {};
      $("summary").innerHTML = [
        metric(summary.notes_checked, "检查笔记"),
        metric(summary.issues, "问题总数"),
        metric(summary.errors, "错误"),
        metric(summary.warnings, "警告"),
        metric(summary.info, "提示"),
      ].join("");

      const issues = audit.issues || [];
      if (!issues.length) {
        $("issues").innerHTML = `<div class="empty">没有发现问题。</div>`;
      } else {
        $("issues").innerHTML = issues.map(renderIssue).join("");
      }

      if (audit.markdown) {
        $("markdownBox").style.display = "block";
        $("markdown").style.display = "block";
        $("markdown").textContent = audit.markdown;
      }
    }

    function renderReview(review) {
      if (!review) {
        $("review").innerHTML = "";
        return;
      }
      const suggestions = review.suggestions || {};
      const validation = review.validation || suggestions._validation || {};
      const nextActions = Array.isArray(suggestions.next_actions) ? suggestions.next_actions : [];
      const questions = Array.isArray(suggestions.review_questions) ? suggestions.review_questions : [];
      const summary = suggestions.summary || "";
      const validationLine = validation.warning_count || validation.correction_count
        ? `<p class="issue-message">建议校验：${validation.correction_count || 0} 个自动修正，${validation.warning_count || 0} 个需要人工确认。</p>`
        : "";
      $("review").innerHTML = `
        <article class="issue info">
          <div class="issue-head">
            <div>
              <span class="issue-code">organize_review</span>
              <div class="issue-path">整理建议 · 模式 ${escapeHtml(review.review_mode || "")}</div>
            </div>
            <strong>建议</strong>
          </div>
          ${summary ? `<p class="issue-message">${escapeHtml(summary)}</p>` : ""}
          ${nextActions.length ? `<p class="issue-message"><strong>下一步：</strong>${escapeHtml(nextActions.slice(0, 4).join("；"))}</p>` : ""}
          ${questions.length ? `<p class="issue-message"><strong>可用于复习：</strong>${escapeHtml(questions.slice(0, 3).map((item) => item.question || "").filter(Boolean).join("；"))}</p>` : ""}
          ${validationLine}
        </article>
      `;
    }

    function renderIssue(issue) {
      const line = issue.line ? `:${issue.line}` : "";
      const severity = issue.severity || "warning";
      return `
        <article class="issue ${escapeHtml(severity)}">
          <div class="issue-head">
            <div>
              <span class="issue-code">${escapeHtml(issue.code || "issue")}</span>
              <div class="issue-path">${escapeHtml(issue.path || "")}${line}</div>
            </div>
            <strong>${escapeHtml(severity)}</strong>
          </div>
          <p class="issue-message">${escapeHtml(issue.message || "")}</p>
        </article>
      `;
    }

    function metric(value, label) {
      return `<div class="metric"><strong>${escapeHtml(value ?? 0)}</strong><span>${escapeHtml(label)}</span></div>`;
    }

    function updateScopeHint() {
      const scopeType = $("scopeType").value;
      const input = $("scopeValue");
      if (scopeType === "all_vault") {
        input.placeholder = "全部 Vault 不需要填写";
        input.value = "";
        input.disabled = true;
      } else if (scopeType === "selected_notes") {
        input.placeholder = "每行一个相对路径，例如：courses/js/Servlet.md";
        input.disabled = false;
      } else if (scopeType === "folder") {
        input.placeholder = "例如：courses/js";
        input.disabled = false;
      } else {
        input.placeholder = "例如：java/servlet";
        input.disabled = false;
      }
    }

    function clearResults() {
      $("summary").innerHTML = "";
      $("review").innerHTML = "";
      $("issues").innerHTML = "";
      $("markdownBox").style.display = "none";
      $("markdown").textContent = "";
    }

    function setBusy(busy) {
      $("runBtn").disabled = busy;
    }

    function setStatus(message, isError = false) {
      const node = $("status");
      node.textContent = message || "";
      node.className = message ? `status visible ${isError ? "error" : ""}` : "status";
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }
  </script>
</body>
</html>
"""


SETTINGS_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>工作区设置</title>
  <style>
    :root { --bg:#f7f5ef; --panel:#fff; --line:#ddd7c8; --text:#26231d; --muted:#716b5f; --accent:#1f6f78; --danger:#b54747; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif; }
    main { max-width:980px; margin:0 auto; padding:28px; }
    header { display:flex; justify-content:space-between; align-items:flex-start; gap:20px; margin-bottom:20px; }
    h1 { margin:0 0 8px; font-size:28px; }
    p { margin:0; color:var(--muted); line-height:1.6; }
    nav { display:flex; gap:12px; flex-wrap:wrap; }
    nav a { color:var(--accent); text-decoration:none; font-size:14px; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; margin-bottom:16px; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
    label { display:block; font-size:13px; color:var(--muted); margin-bottom:6px; }
    input { width:100%; border:1px solid var(--line); border-radius:6px; padding:10px 11px; font-size:14px; background:#fff; color:var(--text); }
    .full { grid-column:1 / -1; }
    .actions { display:flex; gap:10px; align-items:center; margin-top:14px; }
    button { border:1px solid var(--accent); background:var(--accent); color:#fff; border-radius:6px; padding:9px 14px; cursor:pointer; font-size:14px; }
    button.secondary { background:transparent; color:var(--accent); }
    .status { margin-top:12px; font-size:14px; color:var(--muted); }
    .status.error { color:var(--danger); }
    .hint { font-size:13px; color:var(--muted); margin-top:8px; }
    code { background:#f0ece2; padding:2px 5px; border-radius:4px; }
    @media (max-width:760px) { .grid { grid-template-columns:1fr; } header { display:block; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>工作区设置</h1>
        <p>设置当前笔记库路径和相关状态文件。浏览器不能直接选择本机目录，第一版使用路径输入。</p>
      </div>
      <nav>
        <a href="/chat">对话</a>
        <a href="/topics">知识主题</a>
        <a href="/organize">整理</a>
        <a href="/search">检索调试</a>
        <a href="/admin/wiki">维护</a>
        <a href="/settings">设置</a>
      </nav>
    </header>

    <section>
      <div class="grid">
        <div class="full">
          <label for="vaultPath">笔记库路径</label>
          <input id="vaultPath" placeholder="D:\31002\Documents\MyNote" />
          <div class="hint">这是 Obsidian vault 根目录。后端会验证目录是否存在。</div>
        </div>
        <div><label for="wikiDir">Wiki 输出目录</label><input id="wikiDir" placeholder="D:\31002\Documents\MyNote\wiki_full" /></div>
        <div><label for="wikiStatePath">Wiki State</label><input id="wikiStatePath" placeholder="./wiki-state/wiki_state.full-test.json" /></div>
        <div><label for="workspaceStatePath">Workspace State</label><input id="workspaceStatePath" placeholder="./wiki-state/workspace_state.json" /></div>
        <div><label for="indexPath">向量索引</label><input id="indexPath" placeholder="./rag-index/mixed-siliconflow-bge-m3.json" /></div>
        <div><label for="bm25IndexPath">BM25 索引</label><input id="bm25IndexPath" placeholder="./rag-index/mixed-siliconflow-bge-m3.bm25.json" /></div>
        <div><label for="minNotes">最少笔记数</label><input id="minNotes" type="number" min="1" value="2" /></div>
        <div><label for="overviewThreshold">Overview 阈值</label><input id="overviewThreshold" type="number" min="1" value="12" /></div>
      </div>
      <div class="actions"><button id="saveBtn">保存设置</button><button id="reloadBtn" class="secondary">重新加载</button></div>
      <div id="status" class="status">正在加载配置...</div>
    </section>

    <section>
      <p>保存路径后，Wiki、同步、对话中的文件检索会使用新的笔记库路径。搜索服务的向量索引在服务启动时加载；如果切换到另一套索引文件，建议重启 Web 服务后再检索。</p>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    async function loadConfig() {
      setStatus("正在加载配置...");
      const response = await fetch("/api/workspace/config");
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "配置加载失败");
      fillForm(data.config);
      renderValidation(data.validation, data.config_path);
    }
    function fillForm(config) {
      $("vaultPath").value = config.vault_path || "";
      $("wikiDir").value = config.wiki_dir || "";
      $("wikiStatePath").value = config.wiki_state_path || "";
      $("workspaceStatePath").value = config.workspace_state_path || "";
      $("indexPath").value = config.index_path || "";
      $("bm25IndexPath").value = config.bm25_index_path || "";
      $("minNotes").value = config.min_notes_per_tag || 2;
      $("overviewThreshold").value = config.overview_note_threshold || 12;
    }
    function collectForm() {
      return {
        vault_path: $("vaultPath").value.trim(), wiki_dir: $("wikiDir").value.trim(),
        wiki_state_path: $("wikiStatePath").value.trim(), workspace_state_path: $("workspaceStatePath").value.trim(),
        index_path: $("indexPath").value.trim(), bm25_index_path: $("bm25IndexPath").value.trim(),
        min_notes_per_tag: Number($("minNotes").value || 2), overview_note_threshold: Number($("overviewThreshold").value || 12),
      };
    }
    async function saveConfig() {
      setStatus("正在保存配置...");
      const response = await fetch("/api/workspace/config", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(collectForm()) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "保存失败");
      fillForm(data.config);
      renderValidation(data.validation, data.config_path);
    }
    function renderValidation(validation, configPath) { setStatus(`${validation.message} 配置文件：${configPath}`, validation.ok ? "" : "error"); }
    function setStatus(message, kind="") { $("status").textContent = message; $("status").className = kind ? `status ${kind}` : "status"; }
    $("saveBtn").addEventListener("click", () => saveConfig().catch((error) => setStatus(error.message, "error")));
    $("reloadBtn").addEventListener("click", () => loadConfig().catch((error) => setStatus(error.message, "error")));
    loadConfig().catch((error) => setStatus(error.message, "error"));
  </script>
</body>
</html>
"""
INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>检索调试</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --text: #171717;
      --muted: #666666;
      --line: #d9d9d2;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --danger: #b42318;
      --chip: #eef2f1;
      --code: #f2f4f3;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }

    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 22px 0 40px;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }

    h1 {
      font-size: 22px;
      line-height: 1.2;
      margin: 0;
      font-weight: 700;
    }

    .status {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }

    .search-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }

    .search-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 96px 112px;
      gap: 10px;
      align-items: center;
    }

    input, select, button {
      font: inherit;
    }

    input[type="search"], input[type="number"], input[type="text"], select {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: #fff;
      color: var(--text);
    }

    button {
      min-height: 38px;
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: var(--accent);
      color: white;
      padding: 8px 12px;
      cursor: pointer;
      font-weight: 600;
    }

    button.secondary {
      background: white;
      color: var(--accent-dark);
      border-color: var(--line);
      font-weight: 500;
    }

    button:disabled {
      opacity: .58;
      cursor: not-allowed;
    }

    .inline-options {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      align-items: center;
      margin-top: 12px;
      color: var(--muted);
      font-size: 14px;
    }

    label.check {
      display: inline-flex;
      gap: 7px;
      align-items: center;
    }

    details {
      margin-top: 12px;
      border-top: 1px solid var(--line);
      padding-top: 10px;
    }

    summary {
      cursor: pointer;
      color: var(--accent-dark);
      font-weight: 600;
      width: fit-content;
    }

    .advanced-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }

    .field label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }

    .meta-panel {
      margin: 14px 0;
      display: grid;
      gap: 8px;
    }

    .meta-box {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 14px;
    }

    .meta-box strong {
      display: inline-block;
      margin-right: 8px;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }

    .chip {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 8px;
      border-radius: 999px;
      background: var(--chip);
      border: 1px solid var(--line);
      font-size: 12px;
      color: #2d3b39;
    }

    .results {
      display: grid;
      gap: 12px;
    }

    .stage-tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 12px;
    }

    .stage-tab {
      min-height: 32px;
      border-color: var(--line);
      background: var(--panel);
      color: var(--accent-dark);
      font-size: 13px;
      font-weight: 600;
    }

    .stage-tab.is-active {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }

    .result-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px;
    }

    .result-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
    }

    .title {
      font-weight: 700;
      font-size: 16px;
      margin-bottom: 5px;
    }

    .path, .heading, .line-score {
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }

    .preview {
      margin-top: 10px;
      line-height: 1.65;
      font-size: 14px;
      white-space: pre-wrap;
    }

    .full-text {
      display: none;
      margin-top: 10px;
      background: var(--code);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      white-space: pre-wrap;
      line-height: 1.62;
      font-size: 13px;
      overflow-x: auto;
    }

    .actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 10px;
    }

    .error {
      color: var(--danger);
      border-color: #f3b5ad;
      background: #fff7f5;
    }

    @media (max-width: 760px) {
      main {
        width: min(100vw - 20px, 1180px);
        padding-top: 12px;
      }

      .topbar, .result-head {
        display: block;
      }

      .status {
        margin-top: 6px;
      }

      .search-row {
        grid-template-columns: 1fr;
      }

      .advanced-grid {
        grid-template-columns: 1fr 1fr;
      }
    }
  </style>
</head>
<body>
  <main>
    <div class="topbar">
      <div>
        <h1>检索调试</h1>
        <div id="status" class="status">就绪</div>
      </div>
      <nav class="topnav">
        <a href="/chat">对话</a>
        <a href="/topics">知识主题</a>
        <a href="/organize">整理</a>
        <a href="/search">检索调试</a>
        <a href="/admin/wiki">维护</a>
      <a href="/settings">设置</a>
      </nav>
    </div>

    <section class="search-panel">
      <div class="search-row">
        <input id="query" type="search" placeholder="输入搜索内容" autocomplete="off" />
        <input id="topK" type="number" min="1" max="100" value="10" />
        <button id="searchBtn">搜索</button>
      </div>

      <div class="inline-options">
        <label class="check"><input id="enableRewrite" type="checkbox" /> query rewrite</label>
      </div>

      <details>
        <summary>高级选项</summary>
        <div class="advanced-grid">
          <div class="field">
            <label for="mode">模式</label>
            <select id="mode">
              <option value="hybrid" selected>hybrid</option>
              <option value="dense">dense</option>
              <option value="bm25">bm25</option>
              <option value="hybrid-rerank">hybrid-rerank</option>
            </select>
          </div>
          <div class="field">
            <label for="rerankerType">Reranker</label>
            <select id="rerankerType">
              <option value="off" selected>off</option>
              <option value="local">local</option>
              <option value="dashscope">dashscope</option>
            </select>
          </div>
          <div class="field">
            <label for="rerankerModel">reranker_model</label>
            <input id="rerankerModel" type="text" value="BAAI/bge-reranker-v2-m3" />
          </div>
          <div class="field">
            <label for="denseTopK">dense_top_k</label>
            <input id="denseTopK" type="number" min="1" max="500" value="50" />
          </div>
          <div class="field">
            <label for="bm25TopK">bm25_top_k</label>
            <input id="bm25TopK" type="number" min="1" max="500" value="50" />
          </div>
          <div class="field">
            <label for="rrfK">rrf_k</label>
            <input id="rrfK" type="number" min="1" max="500" value="60" />
          </div>
          <div class="field">
            <label for="rerankCandidates">rerank_candidates</label>
            <input id="rerankCandidates" type="number" min="1" max="500" value="50" />
          </div>
          <div class="field">
            <label for="rewriteThreshold">rewrite_threshold</label>
            <input id="rewriteThreshold" type="number" min="0" max="1" step="0.05" value="0.75" />
          </div>
          <div class="field">
            <label for="rewriteWeight">rewrite_weight</label>
            <input id="rewriteWeight" type="number" min="0" max="1" step="0.05" value="0.7" />
          </div>
        </div>
      </details>
    </section>

    <section id="meta" class="meta-panel"></section>
    <section id="stageTabs" class="stage-tabs"></section>
    <section id="results" class="results"></section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    let currentSearchData = null;
    let activeStageKey = "final";

    $("searchBtn").addEventListener("click", runSearch);
    $("rerankerType").addEventListener("change", updateRerankerModelDefault);
    $("query").addEventListener("keydown", (event) => {
      if (event.key === "Enter") runSearch();
    });

    async function runSearch() {
      const query = $("query").value.trim();
      if (!query) return;

      setBusy(true);
      $("meta").innerHTML = "";
      $("stageTabs").innerHTML = "";
      $("results").innerHTML = "";

      const payload = {
        query,
        mode: $("mode").value,
        top_k: numberValue("topK"),
        enable_rewrite: $("enableRewrite").checked,
        rewrite_confidence_threshold: numberValue("rewriteThreshold"),
        rewrite_weight: numberValue("rewriteWeight"),
        dense_top_k: numberValue("denseTopK"),
        bm25_top_k: numberValue("bm25TopK"),
        rrf_k: numberValue("rrfK"),
        reranker_type: $("rerankerType").value,
        reranker_model: $("rerankerModel").value.trim(),
        rerank_candidates: numberValue("rerankCandidates"),
        include_debug: true
      };

      try {
        const response = await fetch("/api/search", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "search failed");
        }
        currentSearchData = data;
        activeStageKey = "final";
        renderMeta(data);
        renderStageTabs(data);
        renderActiveStage();
      } catch (error) {
        $("meta").innerHTML = `<div class="meta-box error">${escapeHtml(error.message)}</div>`;
      } finally {
        setBusy(false);
      }
    }

    function renderMeta(data) {
      const variants = (data.variants || []).map((variant) =>
        `<span class="chip">${escapeHtml(variant.source)} · ${escapeHtml(variant.text)} · ${variant.weight}</span>`
      ).join("");

      let rewriteHtml = "";
      if (data.rewrite_enabled) {
        const rewrite = data.rewrite;
        if (rewrite) {
          const risk = (rewrite.risk_notes || []).map((item) =>
            `<span class="chip">${escapeHtml(item)}</span>`
          ).join("");
          rewriteHtml = `
            <div class="meta-box">
              <strong>rewrite</strong>
              <span>${data.rewrite_used ? "used" : "not used"}</span>
              <div class="chips">
                <span class="chip">${escapeHtml(rewrite.rewrite_type)}</span>
                <span class="chip">confidence ${rewrite.confidence}</span>
                ${rewrite.rewritten_query ? `<span class="chip">${escapeHtml(rewrite.rewritten_query)}</span>` : ""}
              </div>
              ${risk ? `<div class="chips">${risk}</div>` : ""}
            </div>
          `;
        } else {
          rewriteHtml = `<div class="meta-box"><strong>rewrite</strong><span>not available</span></div>`;
        }
      }

      $("meta").innerHTML = `
        <div class="meta-box">
          <strong>${escapeHtml(data.mode)}</strong>
          <span>${data.results.length} results · ${data.elapsed_ms} ms</span>
          <div class="chips">${variants}</div>
        </div>
        ${rewriteHtml}
      `;
    }

    function renderStageTabs(data) {
      const stages = stageList(data);
      if (stages.length <= 1) {
        $("stageTabs").innerHTML = "";
        return;
      }

      $("stageTabs").innerHTML = stages.map((stage) => `
        <button class="stage-tab ${stage.key === activeStageKey ? "is-active" : ""}" data-stage-key="${escapeHtml(stage.key)}">
          ${escapeHtml(stage.label)} ${stage.results.length}
        </button>
      `).join("");

      document.querySelectorAll("[data-stage-key]").forEach((button) => {
        button.addEventListener("click", () => {
          activeStageKey = button.dataset.stageKey;
          renderStageTabs(currentSearchData);
          renderActiveStage();
        });
      });
    }

    function renderActiveStage() {
      const stages = stageList(currentSearchData);
      const stage = stages.find((item) => item.key === activeStageKey) || stages[0];
      renderResults(stage ? stage.results : []);
    }

    function stageList(data) {
      const debugStages = data?.debug?.stages;
      if (Array.isArray(debugStages) && debugStages.length) {
        return debugStages;
      }
      return [
        {
          key: "final",
          label: "最终结果",
          query: data?.query || null,
          query_source: "original",
          retriever: data?.mode || null,
          results: data?.results || []
        }
      ];
    }

    function renderResults(results) {
      if (!results.length) {
        $("results").innerHTML = `<div class="meta-box">no results</div>`;
        return;
      }

      $("results").innerHTML = results.map((item) => {
        const heading = item.heading ? `<div class="heading">heading: ${escapeHtml(item.heading)}</div>` : "";
        const lines = item.start_line && item.end_line ? `${item.start_line}-${item.end_line}` : "";
        return `
          <article class="result-card">
            <div class="result-head">
              <div>
                <div class="title">#${item.rank} ${escapeHtml(item.title)}</div>
                <div class="path">${escapeHtml(item.note_path)}</div>
                ${heading}
              </div>
              <div class="line-score">score ${item.score}${lines ? ` · lines ${lines}` : ""}</div>
            </div>
            <div class="preview">${escapeHtml(item.preview)}</div>
            <pre class="full-text">${escapeHtml(item.text)}</pre>
            <div class="actions">
              <button class="secondary" data-copy="${escapeHtml(item.note_path)}">复制路径</button>
              <button class="secondary" data-toggle>展开</button>
            </div>
          </article>
        `;
      }).join("");

      document.querySelectorAll("[data-copy]").forEach((button) => {
        button.addEventListener("click", async () => {
          await navigator.clipboard.writeText(button.dataset.copy);
          button.textContent = "已复制";
          setTimeout(() => button.textContent = "复制路径", 900);
        });
      });

      document.querySelectorAll("[data-toggle]").forEach((button) => {
        button.addEventListener("click", () => {
          const fullText = button.closest(".result-card").querySelector(".full-text");
          const visible = fullText.style.display === "block";
          fullText.style.display = visible ? "none" : "block";
          button.textContent = visible ? "展开" : "收起";
        });
      });
    }

    function numberValue(id) {
      return Number($(id).value);
    }

    function updateRerankerModelDefault() {
      const type = $("rerankerType").value;
      const current = $("rerankerModel").value.trim();
      if (type === "dashscope" && (!current || current === "BAAI/bge-reranker-v2-m3")) {
        $("rerankerModel").value = "qwen3-rerank";
      }
      if (type === "local" && (!current || current === "qwen3-rerank")) {
        $("rerankerModel").value = "BAAI/bge-reranker-v2-m3";
      }
      if (type === "off" && !current) {
        $("rerankerModel").value = "BAAI/bge-reranker-v2-m3";
      }
    }

    function setBusy(isBusy) {
      $("searchBtn").disabled = isBusy;
      $("status").textContent = isBusy ? "检索中" : "就绪";
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }
  </script>
</body>
</html>
"""


CHAT_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Knowledge Agent - &#23545;&#35805;</title>
  <style>
    :root { color-scheme: light; --bg:#f7f7f4; --panel:#fff; --text:#171717; --muted:#666; --line:#d9d9d2; --accent:#0f766e; --accent-dark:#115e59; --soft:#eef2f1; --danger:#b42318; --code:#f2f4f3; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }
    main { width: min(980px, calc(100vw - 28px)); margin: 0 auto; min-height: 100vh; display: grid; grid-template-rows: auto 1fr auto; gap: 12px; padding: 18px 0; }
    .topbar { display: flex; align-items: center; justify-content: space-between; gap: 14px; }
    .topbar h1 { margin: 0; font-size: 22px; line-height: 1.2; }
    .topnav { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    .topnav a { color: var(--accent-dark); text-decoration: none; font-size: 14px; font-weight: 600; }
    .status { color: var(--muted); font-size: 13px; margin-top: 4px; }
    .messages { overflow-y: auto; display: grid; align-content: start; gap: 12px; padding: 2px; }
    .message { border: 1px solid var(--line); border-radius: 8px; background: var(--panel); padding: 13px; }
    .message.user { margin-left: auto; width: min(760px, 90%); background: #fdfdfb; }
    .message.assistant { margin-right: auto; width: min(860px, 100%); }
    .role { color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; margin-bottom: 8px; }
    .answer { line-height: 1.68; font-size: 15px; }
    .answer p { margin: 0 0 10px; }
    .answer ul, .answer ol { margin: 8px 0 10px 20px; padding: 0; }
    .answer pre { overflow-x: auto; padding: 10px; border-radius: 6px; background: var(--code); }
    .answer code { background: var(--code); padding: 1px 4px; border-radius: 4px; }
    .answer table { width: 100%; border-collapse: collapse; margin: 10px 0 12px; font-size: 14px; line-height: 1.5; }
    .answer th, .answer td { border: 1px solid var(--line); padding: 7px 9px; text-align: left; vertical-align: top; }
    .answer th { background: var(--soft); font-weight: 800; }
    .answer tr:nth-child(even) td { background: #fafbf9; }
    .answer hr { border: 0; border-top: 1px solid var(--line); margin: 12px 0; }
    .agent-process { border: 1px solid var(--line); border-radius: 8px; background: #f8fbfa; margin: 0 0 12px; overflow: hidden; }
    .agent-process[hidden] { display: none; }
    .agent-process summary { list-style: none; cursor: pointer; display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 10px 12px; color: var(--accent-dark); font-weight: 800; }
    .agent-process summary::-webkit-details-marker { display: none; }
    .agent-process-title { display: flex; align-items: center; gap: 8px; min-width: 0; }
    .agent-process-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 3px rgba(15,118,110,.12); flex: 0 0 auto; }
    .agent-process:not(.done) .agent-process-dot { animation: processPulse 1.3s ease-in-out infinite; }
    .agent-process.done .agent-process-dot { background: #64748b; box-shadow: none; }
    .agent-process-summary { color: var(--muted); font-size: 12px; font-weight: 600; white-space: nowrap; }
    .agent-process-body { padding: 0 12px 11px; display: grid; gap: 7px; }
    .agent-process-progress { height: 2px; background: rgba(15,118,110,.10); overflow: hidden; }
    .agent-process-progress-bar { height: 100%; background: var(--accent); transition: width .24s ease; }
    .agent-process-item { display: grid; grid-template-columns: 18px 1fr; gap: 8px; color: #2f3b37; font-size: 13px; line-height: 1.45; }
    .agent-process-icon { color: var(--accent-dark); font-weight: 900; }
    .agent-process-item.active .agent-process-icon { animation: processPulse 1.3s ease-in-out infinite; }
    .agent-process-item.error .agent-process-icon { color: #b42318; }
    .agent-process-item small { display: block; color: var(--muted); font-size: 12px; margin-top: 2px; word-break: break-word; }
    @keyframes processPulse { 0%, 100% { opacity: .42; } 50% { opacity: 1; } }
    .composer { border: 1px solid var(--line); border-radius: 8px; background: var(--panel); padding: 12px; display: grid; gap: 10px; }
    .controls, .session-toolbar { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    label { display: grid; gap: 5px; color: var(--muted); font-size: 12px; font-weight: 700; }
    input, select, textarea { border: 1px solid var(--line); border-radius: 6px; padding: 9px 10px; font: inherit; background: #fff; color: var(--text); }
    textarea { width: 100%; min-height: 78px; resize: vertical; }
    button { border: 1px solid var(--accent); border-radius: 6px; padding: 9px 13px; background: var(--accent); color: #fff; font-weight: 700; cursor: pointer; }
    button.secondary { background: #fff; color: var(--accent-dark); border-color: var(--line); }
    button.danger { background: #fff; color: var(--danger); border-color: #f0b8b1; }
    button.retry-answer { margin-top: 8px; width: 32px; height: 32px; padding: 0; border-radius: 50%; background: #fff; color: var(--accent-dark); border-color: var(--line); }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    .debug { border-top: 1px solid var(--line); padding-top: 8px; color: var(--muted); font-size: 12px; }
    .session-context { display: none; }
    .session-context.has-content { display: block; }
    .session-panel { border: 1px solid var(--line); border-radius: 8px; background: var(--panel); padding: 10px 12px; }
    .session-panel summary { cursor: pointer; color: var(--accent-dark); font-weight: 800; }
    .context-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-top: 10px; }
    .context-section { border: 1px solid var(--line); border-radius: 6px; background: #fdfdfb; padding: 10px; min-width: 0; }
    .context-section h3 { margin: 0 0 7px; font-size: 13px; }
    .context-section ul { margin: 6px 0 0 18px; padding: 0; }
    .context-section .meta-line { color: var(--muted); font-size: 12px; word-break: break-word; }
    details.refs, details.debug-details { margin-top: 12px; border-top: 1px solid var(--line); padding-top: 10px; }
    details.refs summary, details.debug-details summary { cursor: pointer; color: var(--accent-dark); font-weight: 700; }
    .reference-list { display: grid; gap: 10px; margin-top: 10px; }
    .reference { border: 1px solid var(--line); border-radius: 6px; background: #fdfdfb; padding: 10px; }
    .reference .path { color: var(--muted); font-size: 12px; word-break: break-all; margin-top: 3px; }
    .reference pre, .debug-box { white-space: pre-wrap; overflow-x: auto; background: var(--code); border-radius: 6px; padding: 10px; }
    .ref-id, .citation { color: var(--accent-dark); font-weight: 700; }
    details.review { margin-top: 12px; border-top: 1px solid var(--line); padding-top: 10px; }
    details.review summary { cursor: pointer; color: var(--accent-dark); font-weight: 700; }
    .review-section { margin-top: 10px; }
    .review-section h4 { margin: 0 0 6px; font-size: 14px; }
    .review-section ul { margin: 6px 0 0 18px; }
    .review-section.reference-answer { border-top: 1px solid var(--line); margin-top: 14px; padding-top: 12px; }
    .plan { border: 1px solid var(--line); border-radius: 8px; background: var(--soft); padding: 10px; margin-bottom: 10px; }
    .plan-title { font-weight: 800; margin-bottom: 6px; }
    .starter-grid { display: flex; gap: 8px; flex-wrap: wrap; }
    .starter-grid button { background: #fff; color: var(--accent-dark); border-color: var(--line); }
    .modal-backdrop { position: fixed; inset: 0; background: rgba(15,23,42,.28); display: none; align-items: stretch; justify-content: flex-end; z-index: 20; }
    .modal-backdrop.open { display: flex; }
    .modal { width: min(760px, 100vw); background: var(--panel); height: 100vh; overflow-y: auto; padding: 18px; border-left: 1px solid var(--line); box-shadow: -12px 0 32px rgba(15,23,42,.16); }
    .modal-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-bottom: 12px; }
    .history-item { display: block; width: 100%; text-align: left; margin-bottom: 8px; border: 1px solid var(--line); background: #fff; color: var(--text); }
    .history-meta { color: var(--muted); font-size: 12px; }
    .error { color: var(--danger); }
    @media (max-width: 720px) { main { width: min(100vw - 18px, 980px); padding: 10px 0; } .topbar { display: block; } .message.user, .message.assistant { width: 100%; } .controls, .context-grid { display: grid; grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div class="topbar">
        <div>
          <h1>&#23545;&#35805;</h1>
          <div id="status" class="status">&#23601;&#32490;</div>
        </div>
        <nav class="topnav">
          <a href="/chat">&#23545;&#35805;</a>
          <a href="/topics">&#30693;&#35782;&#20027;&#39064;</a>
          <a href="/organize">&#25972;&#29702;</a>
          <a href="/search">&#26816;&#32034;&#35843;&#35797;</a>
          <a href="/admin/wiki">&#32500;&#25252;</a>
          <a href="/settings">&#35774;&#32622;</a>
        </nav>
      </div>
    </header>

    <section id="sessionContext" class="session-context"></section>

    <section id="messages" class="messages">
      <article class="message assistant"><div class="role">Assistant</div><div class="answer">已准备。可以选择面试模式并发送消息开始。</div></article>
    </section>

    <section class="composer">
      <div class="controls">
        <label>模式<select id="chatMode"><option value="interview" selected>面试</option><option value="study">复习</option><option value="answer">问答</option></select></label>
        <label>范围<select id="scopeType"><option value="tag">标签</option><option value="folder" selected>文件夹</option><option value="selected_notes">指定笔记</option><option value="search">搜索</option></select></label>
        <label>范围值<input id="scopeValue" value="个人/面试/agent面试" /></label>
        <label><span>仅依据资料</span><input id="strictEvidence" type="checkbox" /></label>
      </div>
      <textarea id="query" placeholder="输入你的问题..."></textarea>
      <div id="starters" class="starter-grid"></div>
      <div class="session-toolbar">
        <button id="sendBtn" type="button">发送</button>
        <button id="newConversation" class="secondary" type="button">新建对话</button>
        <button id="historyBtn" class="secondary" type="button">历史</button>
        <button id="endInterview" class="danger" type="button" disabled>结束本次面试</button>
        <span id="sessionLabel" class="status">暂无进行中的面试</span>
      </div>
      <details class="debug">
        <summary>选项</summary>
        <div class="controls" style="margin-top:8px">
          <label>检索指令<select id="command"><option value="auto" selected>auto</option><option value="Notes">Notes</option><option value="RegexSearchFiles">RegexSearchFiles</option><option value="Notes+Online">Notes+Online</option></select></label>
          <label>引用数<input id="notesTopK" type="number" min="1" max="50" value="15" /></label>
          <label>联网<select id="onlineProvider"><option value="" selected>disabled</option><option value="tavily">tavily</option><option value="brave">brave</option></select></label>
          <label><span>预先检索本地笔记</span><input id="speculative" type="checkbox" checked /></label>
          <label>Dense top K<input id="denseTopK" type="number" min="1" max="500" value="50" /></label>
          <label>BM25 top K<input id="hybridBm25TopK" type="number" min="1" max="500" value="50" /></label>
          <label>RRF K<input id="rrfK" type="number" min="1" max="500" value="60" /></label>
        </div>
      </details>
    </section>
  </main>

  <div id="historyModal" class="modal-backdrop" aria-hidden="true">
    <aside class="modal" role="dialog" aria-modal="true" aria-label="面试历史">
      <div class="modal-head">
        <h2 style="margin:0;font-size:18px">面试历史</h2>
        <button id="closeHistory" class="secondary" type="button">关闭</button>
      </div>
      <div id="historyContent" class="history-meta">加载中...</div>
    </aside>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);
    const messages = $("messages");
    let chatHistory = [];
    let currentAssistant = null;
    let currentAssistantText = "";
    let currentPayload = null;
    let currentInterviewPlan = null;
    let currentInterviewPlanSignature = "";
    let currentInterviewState = null;
    let currentInterviewSessionId = "";
    let currentConversationSignature = conversationSignature();
    let lastContextItems = [];
    let sessionContextState = {scope: null, stats: null, references: [], interviewPlan: null, profileDebug: null, trace: []};
    let currentTurnTrace = {directorNoteInjected: false};
    let currentProcess = createAgentProcessState();
    const activeInterviewSessionKey = "knowledge_agent.active_interview_session_id";

    $("sendBtn").addEventListener("click", sendMessage);
    $("query").addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) sendMessage();
    });
    $("newConversation").addEventListener("click", startNewConversation);
    $("historyBtn").addEventListener("click", openHistory);
    $("closeHistory").addEventListener("click", closeHistory);
    $("historyModal").addEventListener("click", (event) => { if (event.target === $("historyModal")) closeHistory(); });
    $("endInterview").addEventListener("click", endInterviewSession);
    ["chatMode", "scopeType", "scopeValue", "strictEvidence"].forEach((id) => $(id).addEventListener("change", resetConversationIfNeeded));
    restoreActiveInterviewSession();

    function resetConversationIfNeeded() {
      const signature = conversationSignature();
      if (signature === currentConversationSignature) return;
      chatHistory = [];
      currentInterviewPlan = null;
      currentInterviewPlanSignature = "";
      currentInterviewState = null;
      currentInterviewSessionId = "";
      localStorage.removeItem(activeInterviewSessionKey);
      lastContextItems = [];
      resetSessionContext();
      currentConversationSignature = signature;
      updateSessionLabel();
      $("starters").innerHTML = "";
    }

    async function sendMessage() {
      const query = $("query").value.trim();
      if (!query) return;
      resetConversationIfNeeded();
      appendUserMessage(query);
      $("query").value = "";
      currentAssistant = appendAssistantMessage("");
      currentAssistantText = "";
      currentPayload = {router: null, retrieval: null, retrievalStatus: null, interviewPlan: null, profileDebug: null, context: null, answer: null, done: null, errors: []};
      currentProcess = createAgentProcessState();
      renderAgentProcess();
      setBusy(true, "Starting...");

      const body = buildAgentRequestBody(query);
      currentTurnTrace = {directorNoteInjected: shouldInjectDirectorNote(body.interview_state)};
      let turnIds = null;

      try {
        if ($("chatMode").value === "interview") {
          turnIds = await persistPendingInterviewTurn(query);
          body.session_id = currentInterviewSessionId;
          if (currentTurnTrace.directorNoteInjected) {
            recordSessionTrace("director_note_injected", `Director Note injected (follow_up=${body.interview_state?.follow_up_count || 0})`, {
              interview_state: body.interview_state || {}
            });
          }
        }
        await streamAgentAnswerWithRetry(body);
        const answerText = currentAssistantText.trim();
        if (answerText) {
          const assistantNode = currentAssistant;
          renderAssistantExtras();
          const shouldReview = $("chatMode").value === "interview" && shouldRequestTurnSummary(query, answerText);
          chatHistory.push({role:"user", content:query});
          chatHistory.push({role:"assistant", content:answerText});
          trimChatHistory();
          if ($("chatMode").value === "interview") {
            const stateChange = isServerInterviewState() ? null : updateInterviewState(query, answerText);
            if (turnIds) {
              await completeInterviewTurn(turnIds, answerText);
            } else {
              turnIds = await persistInterviewTurn(query, answerText);
            }
            recordInterviewStateTrace(stateChange);
            if (shouldReview) runTurnReviewInBackground(answerText, turnIds, assistantNode);
          }
        }
        setBusy(false, "就绪");
      } catch (error) {
        currentPayload.errors.push(error.message);
        if ($("chatMode").value === "interview" && turnIds) {
          await failInterviewTurn(turnIds, currentAssistantText.trim(), error);
        }
        if (currentAssistant) {
          currentAssistant.querySelector(".answer").innerHTML += `<p class="error">${escapeHtml(error.message)}</p>`;
          if (error.retryable !== false) attachRegenerateButton(currentAssistant, query, turnIds);
        }
        setBusy(false, "Error");
      }
    }

    function buildAgentRequestBody(query) {
      return {
        query,
        chat_mode: $("chatMode").value,
        command: $("command").value,
        scope_type: $("scopeType").value,
        scope_value: $("scopeValue").value.trim() || null,
        scope_paths: scopePaths(),
        chat_history: chatHistory.slice(-12),
        interview_plan: $("chatMode").value === "interview" && currentInterviewPlanSignature === conversationSignature() ? currentInterviewPlan : null,
        interview_state: $("chatMode").value === "interview" ? currentInterviewState : null,
        session_id: $("chatMode").value === "interview" ? currentInterviewSessionId : null,
        notes_top_k: numberValue("notesTopK"),
        dense_top_k: numberValue("denseTopK"),
        hybrid_bm25_top_k: numberValue("hybridBm25TopK"),
        rrf_k: numberValue("rrfK"),
        online_provider: $("onlineProvider").value.trim() || null,
        strict_evidence: $("strictEvidence").checked,
        speculative_notes_search: $("speculative").checked
      };
    }

    function shouldInjectDirectorNote(interviewState) {
      if ($("chatMode").value !== "interview") return false;
      if (!interviewState) return false;
      return Number(interviewState.follow_up_count || 0) >= 4;
    }

    async function recordSessionTrace(event, summary, details) {
      if (!currentInterviewSessionId) return null;
      const entry = {
        event,
        summary,
        details: details || {}
      };
      addLocalSessionTrace(entry);
      try {
        const response = await fetch(`/api/interview/sessions/${encodeURIComponent(currentInterviewSessionId)}/trace`, {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify(entry)
        });
        if (!response.ok) return null;
        return await response.json();
      } catch {
        return null;
      }
    }

    function addLocalSessionTrace(entry) {
      sessionContextState.trace = [...(sessionContextState.trace || []), {
        id: "local",
        created_at: new Date().toISOString(),
        ...entry
      }].slice(-50);
      renderSessionContext();
    }

    async function streamAgentAnswerWithRetry(body) {
      let lastError = null;
      for (let attempt = 0; attempt < 2; attempt += 1) {
        try {
          if (attempt > 0) {
            setBusy(true, "Retrying...");
            await delay(1200);
          }
          const response = await fetch("/api/agent/stream", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
          if (!response.ok || !response.body) throw new Error(`request failed: ${response.status}`);
          await readSse(response.body);
          return;
        } catch (error) {
          lastError = error;
          const retryable = error.retryable !== false;
          const noOutputYet = !currentAssistantText.trim();
          if (attempt === 0 && retryable && noOutputYet) continue;
          throw error;
        }
      }
      if (lastError) throw lastError;
    }

    function delay(ms) {
      return new Promise((resolve) => setTimeout(resolve, ms));
    }

    function attachRegenerateButton(assistantNode, query, turnIds) {
      if (!assistantNode || !turnIds) return;
      assistantNode.querySelectorAll(".retry-answer").forEach((node) => node.remove());
      const button = document.createElement("button");
      button.type = "button";
      button.className = "retry-answer";
      button.title = "Retry";
      button.innerHTML = "&#8635;";
      button.addEventListener("click", () => retryAssistantAnswer(query, turnIds, assistantNode));
      assistantNode.appendChild(button);
    }

    async function retryAssistantAnswer(query, turnIds, assistantNode) {
      if (!turnIds || !assistantNode) return;
      currentAssistant = assistantNode;
      currentAssistantText = "";
      currentPayload = {router: null, retrieval: null, retrievalStatus: null, interviewPlan: currentInterviewPlan, profileDebug: null, context: null, answer: null, done: null, errors: []};
      currentProcess = createAgentProcessState();
      assistantNode.querySelectorAll(".retry-answer, .assistant-extra, details.review").forEach((node) => node.remove());
      assistantNode.querySelector(".answer").innerHTML = "";
      renderAgentProcess();
      setBusy(true, "Retrying...");
      try {
        const body = buildAgentRequestBody(query);
        body.session_id = currentInterviewSessionId;
        currentTurnTrace = {directorNoteInjected: shouldInjectDirectorNote(body.interview_state)};
        if (currentTurnTrace.directorNoteInjected) {
          recordSessionTrace("director_note_injected", `Director Note injected (follow_up=${body.interview_state?.follow_up_count || 0})`, {
            interview_state: body.interview_state || {},
            retry: true
          });
        }
        await streamAgentAnswerWithRetry(body);
        const answerText = currentAssistantText.trim();
        if (!answerText) throw new Error("empty answer");
        renderAssistantExtras();
        const shouldReview = $("chatMode").value === "interview" && shouldRequestTurnSummary(query, answerText);
        chatHistory.push({role:"user", content:query});
        chatHistory.push({role:"assistant", content:answerText});
        trimChatHistory();
        const stateChange = isServerInterviewState() ? null : updateInterviewState(query, answerText);
        await completeInterviewTurn(turnIds, answerText);
        recordInterviewStateTrace(stateChange);
        if (shouldReview) runTurnReviewInBackground(answerText, turnIds, assistantNode);
        setBusy(false, "灏辩华");
      } catch (error) {
        await failInterviewTurn(turnIds, currentAssistantText.trim(), error);
        assistantNode.querySelector(".answer").innerHTML += `<p class="error">${escapeHtml(error.message)}</p>`;
        if (error.retryable !== false) attachRegenerateButton(assistantNode, query, turnIds);
        setBusy(false, "Error");
      }
    }

    async function readSse(stream) {
      const reader = stream.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const {value, done} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream:true});
        let boundary;
        while ((boundary = buffer.indexOf("\n\n")) >= 0) {
          const raw = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          handleSseEvent(raw);
        }
      }
      if (buffer.trim()) handleSseEvent(buffer);
    }

    function handleSseEvent(raw) {
      let eventName = "message";
      const dataLines = [];
      raw.split(/\r?\n/).forEach((line) => {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
      });
      if (!dataLines.length) return;
      let data;
      try { data = JSON.parse(dataLines.join("\n")); } catch { return; }
      if (eventName === "status") setBusy(true, data.message || data.stage || "处理中...");
      if (eventName === "router") currentPayload.router = data;
      if (eventName === "retrieval") {
        currentPayload.retrieval = data;
        currentPayload.retrievalStatus = data.reference_summary || data.summary || null;
      }
      if (eventName === "context") {
        currentPayload.context = data;
        lastContextItems = data.items || [];
        sessionContextState.scope = data.scope || null;
        sessionContextState.stats = data.stats || null;
        sessionContextState.references = data.items || [];
        updateProcessFromContext(data);
        renderSessionContext();
      }
      if (eventName === "interview_plan") {
        currentPayload.interviewPlan = data;
        sessionContextState.interviewPlan = data;
        renderSessionContext();
        recordSessionTrace(
          "interview_plan",
          `plan ${data.fallback_used ? "fallback" : "ready"} (${data.source || "unknown"})`,
          {
            source: data.source || "",
            fallback_used: Boolean(data.fallback_used),
            latency_ms: data.latency_ms ?? null,
            topic_count: Array.isArray(data.plan?.topics) ? data.plan.topics.length : 0,
            error: data.error || ""
          }
        );
        if (data.plan) {
          currentInterviewPlan = data.plan;
          currentInterviewPlanSignature = conversationSignature();
          if (shouldRenderInterviewPlanForCurrentTurn()) {
            renderInterviewPlan(data.plan);
          }
        }
      }
      if (eventName === "profile_debug") {
        currentPayload.profileDebug = data;
        sessionContextState.profileDebug = data;
        renderSessionContext();
        recordSessionTrace(
          "profile_injected",
          `profile injected: ${data.current_topic || "unknown"} (${data.weak_points_count ?? 0} weak, ${data.due_reviews_count ?? 0} due)`,
          {
            current_topic: data.current_topic || "",
            topic_source: data.topic_source || "",
            injected_to_prompt: Boolean(data.injected_to_prompt),
            weak_points_count: data.weak_points_count ?? 0,
            due_reviews_count: data.due_reviews_count ?? 0,
            strong_points_count: data.strong_points_count ?? 0,
            prompt_context_sha256: data.prompt_context_sha256 || "",
            prompt_context_line_count: data.prompt_context_line_count ?? 0
          }
        );
      }
      if (eventName === "agent_step") {
        if ($("chatMode").value === "interview") {
          addLocalSessionTrace({
            event: "agent_step",
            summary: `agent step ${data.index ?? ""}: ${data.kind || ""}`,
            details: data
          });
        }
      }
      if (eventName === "tool_started") {
        updateProcessFromToolStarted(data);
      }
      if (eventName === "tool_result") {
        updateProcessFromToolResult(data);
        if ($("chatMode").value === "interview") {
          addLocalSessionTrace({
            event: "tool_result",
            summary: `${data.name || "tool"}: ${data.status || ""}`,
            details: {name: data.name || "", ok: Boolean(data.ok), status: data.status || "", latency_ms: data.latency_ms ?? null}
          });
        }
      }
      if (eventName === "state_updated") {
        currentInterviewState = data || null;
        addLocalSessionTrace({
          event: "state_updated",
          summary: `state updated: ${data?.current_topic || ""} / ${data?.current_layer_name || ""}`,
          details: data || {}
        });
        renderSessionContext();
      }
      if (eventName === "answer_delta") {
        collapseProcessForAnswer();
        currentAssistantText += data.text || "";
        if (currentAssistant) currentAssistant.querySelector(".answer").innerHTML = renderMarkdown(currentAssistantText);
        scrollToBottom();
      }
      if (eventName === "answer") {
        currentPayload.answer = data;
        const answerText = String((data && (data.answer || data.text)) || "").trim();
        if (answerText && !currentAssistantText.trim()) {
          currentAssistantText = answerText;
          if (currentAssistant) currentAssistant.querySelector(".answer").innerHTML = renderMarkdown(currentAssistantText);
          scrollToBottom();
        }
      }
      if (eventName === "agent_stopped") {
        handleAgentStopped(data);
      }
      if (eventName === "done") {
        currentPayload.done = data;
        finishAgentProcess(data);
      }
      if (eventName === "error") {
        const error = new Error(data.message || "stream error");
        error.errorType = data.error_type || "Error";
        error.category = data.category || "unknown";
        error.retryable = data.retryable !== false;
        throw error;
      }
    }

    function createAgentProcessState() {
      return {
        visible: false,
        done: false,
        collapsed: false,
        items: [],
        readPaths: new Set(),
        searched: 0,
        checked: 0,
        listed: 0,
        online: 0,
        clueCount: 0,
        matchCount: 0,
        active: null,
        actionRuns: [],
        phase: "preparing",
        stopped: false,
        stopReason: "",
        stopMessage: ""
      };
    }

    function shouldShowAgentProcess() {
      return ["answer", "interview"].includes($("chatMode").value) && currentAssistant;
    }

    function ensureAgentProcessVisible() {
      if (!shouldShowAgentProcess()) return false;
      currentProcess.visible = true;
      return true;
    }

    function updateProcessFromContext(data) {
      if (!data || data.mode !== "answer" || !data.stats?.agent_v2) return;
      if (!ensureAgentProcessVisible()) return;
      currentProcess.phase = "preparing";
      const scope = data.scope || {};
      const noteCount = Number(scope.note_count ?? data.stats?.context_items ?? 0);
      const label = scopeLabel(scope.type, scope.value);
      addProcessItem("确定查阅范围", `${label}${noteCount ? `，约 ${noteCount} 篇候选笔记` : ""}`);
      renderAgentProcess();
    }

    function updateProcessFromToolStarted(data) {
      if (!data || !data.name) return;
      if (!ensureAgentProcessVisible()) return;
      const run = toolCallToActionRun(data);
      if (!run) return;
      currentProcess.phase = "tool_running";
      upsertActionRun(run);
      currentProcess.active = {name: data.name, actionId: run.id, index: addProcessItem(run.label, run.detail, "active")};
      renderAgentProcess();
    }

    function updateProcessFromToolResult(data) {
      if (!data || !data.name) return;
      if (!ensureAgentProcessVisible()) return;
      const run = toolResultToActionRun(data);
      if (!run) return;
      upsertActionRun(run);
      applyActionRunStats(run);
      finishActiveProcessItem(data.name, run.label, run.detail, run.status === "error" ? "error" : "done", run.id);
      if ($("chatMode").value !== "interview") currentProcess.phase = "analyzing";
      renderAgentProcess();
    }

    function upsertActionRun(run) {
      const index = currentProcess.actionRuns.findIndex((item) => item.id === run.id);
      if (index >= 0) currentProcess.actionRuns[index] = {...currentProcess.actionRuns[index], ...run};
      else currentProcess.actionRuns.push(run);
    }

    function applyActionRunStats(run) {
      const tool = String(run.tool || run.kind || "");
      const status = String(run.status || "");
      const stats = run.stats && typeof run.stats === "object" ? run.stats : {};
      const sourcePaths = Array.isArray(run.source_paths) ? run.source_paths : [];
      if (tool === "search_notes") {
        currentProcess.searched += status === "running" ? 0 : 1;
        currentProcess.clueCount += Number(stats.hit_count || 0);
      } else if (tool === "grep_vault") {
        currentProcess.checked += status === "running" ? 0 : 1;
        currentProcess.matchCount += Number(stats.hit_count || 0);
      } else if (tool === "list_notes") {
        currentProcess.listed += status === "running" ? 0 : 1;
      } else if (tool === "online_search") {
        currentProcess.online += status === "running" ? 0 : 1;
        currentProcess.clueCount += Number(stats.hit_count || 0);
      } else if (tool === "read_note") {
        sourcePaths.forEach((path) => {
          if (path) currentProcess.readPaths.add(String(path));
        });
      }
    }

    function collapseProcessForAnswer() {
      if (!currentProcess.visible || currentProcess.collapsed) return;
      currentProcess.collapsed = true;
      if ($("chatMode").value === "interview") {
        renderAgentProcess();
      } else {
        currentProcess.phase = "generating";
        addProcessItem("整理回答", "正在把查阅到的材料组织成回复", "active");
      }
      renderAgentProcess();
    }

    function handleAgentStopped(data) {
      if (!data) return;
      currentPayload.agentStopped = data;
      if (ensureAgentProcessVisible()) {
        currentProcess.stopped = true;
        currentProcess.stopReason = String(data.reason || "");
        currentProcess.stopMessage = String(data.message || "");
        currentProcess.collapsed = false;
        currentProcess.items = currentProcess.items.map((item) => item.status === "active" ? {...item, status: "done"} : item);
        currentProcess.active = null;
        addProcessItem("已停止继续查阅", stopReasonLabel(currentProcess.stopReason), "done");
        renderAgentProcess();
      }
      const message = String(data.message || "").trim();
      if (message && currentAssistant && !currentAssistantText.trim()) {
        currentAssistant.querySelector(".answer").innerHTML = renderMarkdown(message);
        scrollToBottom();
      }
    }

    function finishAgentProcess(data) {
      if (!currentProcess.visible) return;
      currentProcess.done = true;
      currentProcess.collapsed = true;
      currentProcess.phase = "done";
      currentProcess.items = currentProcess.items.map((item) => item.status === "active" ? {...item, status: "done"} : item);
      currentProcess.active = null;
      const metrics = data?.telemetry?.derived_metrics || {};
      if (Array.isArray(metrics.source_paths)) {
        metrics.source_paths.forEach((path) => {
          if (path) currentProcess.readPaths.add(String(path));
        });
      }
      renderAgentProcess();
    }

    function addProcessItem(title, detail, status = "done") {
      const last = currentProcess.items[currentProcess.items.length - 1];
      if (last && last.title === title && last.detail === detail) return currentProcess.items.length - 1;
      currentProcess.items.push({title, detail, status});
      if (currentProcess.items.length > 8) currentProcess.items = currentProcess.items.slice(-8);
      return currentProcess.items.length - 1;
    }

    function finishActiveProcessItem(toolName, title, detail, status = "done", actionId = "") {
      const active = currentProcess.active;
      if (active && active.name === toolName && (!actionId || !active.actionId || active.actionId === actionId) && currentProcess.items[active.index]) {
        currentProcess.items[active.index] = {title, detail, status};
        currentProcess.active = null;
        return;
      }
      addProcessItem(title, detail, status);
    }

    function toolCallToActionRun(call) {
      const name = String(call?.name || "").trim();
      if (!name || isHiddenTool(name)) return null;
      const args = call.arguments && typeof call.arguments === "object" ? call.arguments : {};
      return {
        id: String(call.id || `${name}-${Date.now()}`),
        kind: name,
        tool: name,
        label: toolLabel(name),
        detail: toolCallDetail(name, args),
        status: "running",
        stats: {},
        latency_ms: null,
        source_paths: []
      };
    }

    function toolResultToActionRun(result) {
      const name = String(result?.name || "").trim();
      if (!name || isHiddenTool(name)) return null;
      const output = result.output && typeof result.output === "object" ? result.output : {};
      const stats = result.stats && typeof result.stats === "object" ? result.stats : deriveToolStats(name, output);
      const sourcePaths = Array.isArray(stats.source_paths)
        ? stats.source_paths
        : sourcePathsFromOutput(output);
      const ok = result.ok !== false && result.status !== "error";
      return {
        id: String(result.call_id || `${name}-${Date.now()}`),
        kind: name,
        tool: name,
        label: toolLabel(name),
        detail: toolResultDetail(name, output, stats, ok, String(result.error || "")),
        status: ok ? "success" : "error",
        stats,
        latency_ms: Number(result.latency_ms || 0),
        source_paths: sourcePaths
      };
    }

    function isHiddenTool(name) {
      return ["get_interview_state", "list_plan_topics", "inspect_state"].includes(name);
    }

    function toolLabel(name) {
      const labels = {
        search_notes: "查找相关笔记",
        read_note: "阅读参考笔记",
        grep_vault: "核对关键词",
        list_notes: "浏览可用笔记",
        recall_profile: "回顾你的薄弱点",
        advance_layer: "推进追问层次",
        select_topic: "切换面试主题",
        online_search: "查找公开资料"
      };
      return labels[name] || "执行辅助动作";
    }

    function toolCallDetail(name, args) {
      if (["search_notes", "grep_vault", "online_search"].includes(name)) return queryDetail(args.query);
      if (name === "read_note") return [String(args.path || "指定笔记"), String(args.section_id || args.heading || "")].filter(Boolean).join(" · ");
      if (name === "list_notes") return args.filter ? `筛选：${args.filter}` : "查看当前范围内可用笔记";
      if (name === "recall_profile") return [String(args.topic || "当前主题"), String(args.planned_layer || "")].filter(Boolean).join(" · ");
      if (name === "advance_layer") return String(args.reason || "进入下一层追问");
      if (name === "select_topic") return [String(args.name || args.topic || "新主题"), String(args.reason || "")].filter(Boolean).join(" · ");
      return "";
    }

    function toolResultDetail(name, output, stats, ok, error) {
      if (!ok) return error || "工具调用未成功";
      if (name === "search_notes") return `${Number(stats.hit_count || 0)} 条线索${sourcePathSummary(stats.source_paths || output.source_paths)}`;
      if (name === "grep_vault") return `${Number(stats.hit_count || 0)} 处匹配${sourcePathSummary(stats.source_paths || output.source_paths)}`;
      if (name === "list_notes") return `${Number(stats.hit_count || output.result_count || 0)} 篇候选${output.truncated ? "，已截断" : ""}`;
      if (name === "read_note") {
        const path = String(output.path || "指定笔记");
        const headingPath = Array.isArray(output.heading_path) ? output.heading_path.join(" > ") : String(output.heading || "");
        const offsetLabel = Number(output.offset || 0) > 0 ? ` · offset ${output.offset}` : "";
        return `${path}${headingPath ? ` · ${headingPath}` : ""}${offsetLabel}${output.truncated ? "，内容较长已截断" : ""}`;
      }
      if (name === "recall_profile") {
        const weak = Number(stats.hit_count || output.matching_weak_count || output.weak_points_count || 0);
        const due = Number(output.due_review_count || output.due_reviews_count || 0);
        return [weak ? `${weak} 条相关弱项` : "", due ? `${due} 条到期复习` : ""].filter(Boolean).join("，") || "已读取相关画像提示";
      }
      if (name === "advance_layer") return String(output.current_layer_name || output.target_layer || output.summary || "追问层次已更新");
      if (name === "select_topic") return String(output.current_topic || output.topic || output.summary || "主题已更新");
      if (name === "online_search") return `${Number(stats.hit_count || output.result_count || 0)} 条结果`;
      return String(output.summary || "动作已完成");
    }

    function deriveToolStats(name, output) {
      const stats = {};
      const resultCount = Number(output.result_count ?? output.count ?? output.match_count ?? output.note_count ?? 0);
      if (resultCount) stats.hit_count = resultCount;
      const paths = sourcePathsFromOutput(output);
      if (paths.length) {
        stats.source_paths = paths;
        stats.source_count = paths.length;
        if (["search_notes", "grep_vault", "list_notes", "read_note"].includes(name)) stats.note_count = paths.length;
      }
      if (name === "read_note" && output.truncated !== undefined) stats.truncated = Boolean(output.truncated);
      return stats;
    }

    function sourcePathsFromOutput(output) {
      const paths = Array.isArray(output.source_paths) ? output.source_paths : (output.path ? [output.path] : []);
      return Array.from(new Set(paths.map((path) => String(path || "").trim()).filter(Boolean)));
    }

    function renderAgentProcess() {
      if (!currentAssistant) return;
      const node = currentAssistant.querySelector(".agent-process");
      if (!node) return;
      if (!currentProcess.visible || !currentProcess.items.length) {
        node.hidden = true;
        return;
      }
      node.hidden = false;
      node.open = !currentProcess.collapsed;
      node.classList.toggle("done", Boolean(currentProcess.done));
      const summary = processSummary();
      const processTitle = agentProcessTitle();
      node.innerHTML = `
        ${$("chatMode").value === "answer" ? `<div class="agent-process-progress"><div class="agent-process-progress-bar" style="width:${processProgress()}%"></div></div>` : ""}
        <summary>
          <span class="agent-process-title"><span class="agent-process-dot"></span><span>${escapeHtml(processTitle)}</span></span>
          <span class="agent-process-summary">${escapeHtml(summary)}</span>
        </summary>
        <div class="agent-process-body">
          ${currentProcess.items.map((item) => `
            <div class="agent-process-item ${item.status === "active" ? "active" : ""} ${item.status === "error" ? "error" : ""}">
              <span class="agent-process-icon">${processItemIcon(item.status)}</span>
              <span>${escapeHtml(item.title)}${item.detail ? `<small>${escapeHtml(item.detail)}</small>` : ""}</span>
            </div>
          `).join("")}
        </div>
      `;
      applyStoppedProcessTitle(node);
    }

    function agentProcessTitle() {
      if (currentProcess.stopped) return "已停止继续查阅";
      const mode = $("chatMode").value;
      if (mode === "interview") return currentProcess.done ? "已完成辅助动作" : "正在执行辅助动作";
      return currentProcess.done ? "已完成资料查阅" : "正在查阅资料";
    }

    function processItemIcon(status) {
      if (status === "active") return "•";
      if (status === "error") return "!";
      return "✓";
    }

    function processProgress() {
      if (currentProcess.done) return 100;
      if (currentProcess.phase === "generating") return 92;
      if (currentProcess.phase === "analyzing") return 78;
      if (currentProcess.phase === "tool_running") return 45;
      return 18;
    }

    function applyStoppedProcessTitle(node) {
      if (!currentProcess.stopped || !node) return;
      const titleNode = node.querySelector(".agent-process-title span:last-child");
      if (titleNode) titleNode.textContent = "已停止继续查阅";
      const summaryNode = node.querySelector(".agent-process-summary");
      if (summaryNode) summaryNode.textContent = stopReasonLabel(currentProcess.stopReason);
    }

    function processSummary() {
      const parts = [];
      const evidenceCount = currentProcess.clueCount + currentProcess.matchCount;
      if (evidenceCount) parts.push(`找到 ${evidenceCount} 条线索`);
      if (currentProcess.readPaths.size) parts.push(`已阅读 ${currentProcess.readPaths.size} 篇笔记`);
      if (currentProcess.online) parts.push("包含公开资料");
      if (!parts.length && currentProcess.active) return $("chatMode").value === "interview" ? "正在准备" : "正在查阅";
      return parts.length ? parts.join("，") : "准备中";
    }

    function stopReasonLabel(reason) {
      if (reason === "max_steps") return "\u8fbe\u5230\u672c\u8f6e\u6b65\u9aa4\u4e0a\u9650";
      if (reason === "tool_timeout") return "\u5de5\u5177\u8c03\u7528\u8d85\u65f6";
      if (reason === "tool_error") return "\u5de5\u5177\u8c03\u7528\u5931\u8d25";
      if (reason === "llm_error") return "\u6a21\u578b\u8c03\u7528\u5931\u8d25";
      return reason ? `\u505c\u6b62\u539f\u56e0\uff1a${reason}` : "\u5df2\u505c\u6b62";
    }

    function plannedProcessItem(name, args) {
      if (name === "search_notes") return ["查找相关笔记", queryDetail(args.query)];
      if (name === "grep_vault") return ["核对关键词出现位置", queryDetail(args.query)];
      if (name === "list_notes") return ["浏览可用笔记", args.filter ? `筛选：${args.filter}` : "查看当前范围内可用笔记"];
      if (name === "read_note") return ["阅读笔记", `${String(args.path || "指定笔记")}${args.reason ? `：${args.reason}` : ""}${args.section_id ? ` · ${args.section_id}` : args.heading ? ` · ${args.heading}` : ""}${args.offset ? ` · offset ${args.offset}` : ""}`];
      if (name === "online_search") return ["查找公开资料", queryDetail(args.query)];
      return null;
    }

    function queryDetail(query) {
      const text = String(query || "").trim();
      return text ? `关键词：${text.slice(0, 80)}` : "";
    }

    function sourcePathSummary(paths) {
      if (!Array.isArray(paths) || !paths.length) return "";
      const unique = Array.from(new Set(paths.map((path) => String(path || "").trim()).filter(Boolean)));
      if (!unique.length) return "";
      return `，涉及 ${unique.slice(0, 3).join("、")}${unique.length > 3 ? " 等" : ""}`;
    }

    function scopeLabel(type, value) {
      if (type === "selected_notes") return "指定笔记";
      if (type === "folder") return `文件夹 ${value || ""}`.trim();
      if (type === "tag") return `标签 ${value || ""}`.trim();
      if (type === "search") return `搜索范围 ${value || ""}`.trim();
      if (type === "all_vault") return "全库";
      return "当前范围";
    }

    function renderInterviewPlan(plan) {
      const topics = Array.isArray(plan.topics) ? plan.topics : [];
      if (!topics.length || !currentAssistant) return;
      currentAssistant.querySelectorAll(".plan").forEach((node) => node.remove());
      const html = `<div class="plan"><div class="plan-title">面试方向</div><div class="starter-grid">${topics.map((topic) => `<button type="button" data-topic="${escapeHtml(topic.name || "")}">${escapeHtml(topic.name || "Topic")}</button>`).join("")}</div></div>`;
      currentAssistant.querySelector(".answer").insertAdjacentHTML("beforebegin", html);
      currentAssistant.querySelectorAll("[data-topic]").forEach((button) => {
        button.addEventListener("click", async () => {
          await selectInterviewTopic(button.dataset.topic || "");
        });
      });
    }

    function shouldRenderInterviewPlanForCurrentTurn() {
      if ($("chatMode").value !== "interview") return false;
      if (!currentInterviewState) return true;
      return currentInterviewState.topic_phase === "awaiting_selection" || !currentInterviewState.current_topic;
    }

    async function selectInterviewTopic(topic) {
      const selected = String(topic || "").trim();
      if (!selected) return;
      const sessionId = await ensureInterviewSession();
      const response = await fetch(`/api/interview/sessions/${encodeURIComponent(sessionId)}/select-topic`, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({
          topic: selected,
          reason: "user selected topic from interview plan UI",
          source: "ui",
          interview_plan: currentInterviewPlan
        })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "topic selection failed");
      currentInterviewState = data.interview_state || data.session?.interview_state || currentInterviewState;
      recordSessionTrace("topic_selected", `topic selected: ${selected}`, {interview_state: currentInterviewState});
      $("query").value = `我想从${selected}开始`;
      $("query").focus();
    }

    function renderAssistantExtras() {
      if (!currentAssistant) return;
      currentAssistant.querySelectorAll(".assistant-extra").forEach((node) => node.remove());
      currentAssistant.insertAdjacentHTML(
        "beforeend",
        `<div class="assistant-extra">${renderDebug(buildTurnDebug())}</div>`
      );
    }

    function resetSessionContext() {
      sessionContextState = {scope: null, stats: null, references: [], interviewPlan: null, profileDebug: null, trace: []};
      renderSessionContext();
    }

    function renderSessionContext() {
      const node = $("sessionContext");
      if (!node) return;
      const hasContent = sessionContextState.scope || sessionContextState.interviewPlan || sessionContextState.profileDebug || (sessionContextState.references || []).length || (sessionContextState.trace || []).length;
      if (!hasContent) {
        node.className = "session-context";
        node.innerHTML = "";
        return;
      }
      node.className = "session-context has-content";
      const scope = sessionContextState.scope || {};
      const stats = sessionContextState.stats || {};
      const planPayload = sessionContextState.interviewPlan || {};
      const plan = planPayload.plan || planPayload || {};
      const topics = Array.isArray(plan.topics) ? plan.topics : [];
      const profile = sessionContextState.profileDebug || {};
      node.innerHTML = `<details class="session-panel" open><summary>Session Context</summary>
        <div class="context-grid">
          <section class="context-section">
            <h3>Scope</h3>
            <div class="meta-line">type: ${escapeHtml(scope.type || "")}</div>
            <div class="meta-line">value: ${escapeHtml(scope.value || "")}</div>
            ${Array.isArray(scope.paths) && scope.paths.length ? `<ul>${scope.paths.map((path) => `<li>${escapeHtml(path)}</li>`).join("")}</ul>` : ""}
            ${stats ? `<div class="meta-line">items: ${escapeHtml(stats.context_items ?? stats.items ?? "")} · chars: ${escapeHtml(stats.context_chars ?? "")}</div>` : ""}
          </section>
          <section class="context-section">
            <h3>Interview Plan</h3>
            ${planPayload.source ? `<div class="meta-line">source: ${escapeHtml(planPayload.source)}${planPayload.fallback_used ? " · fallback" : ""}</div>` : ""}
            ${topics.length ? `<ul>${topics.map((topic) => `<li><strong>${escapeHtml(topic.name || "")}</strong>${Array.isArray(topic.coverage) && topic.coverage.length ? `：${escapeHtml(topic.coverage.join(" / "))}` : ""}</li>`).join("")}</ul>` : `<div class="meta-line">not available</div>`}
          </section>
          <section class="context-section">
            <h3>Candidate Profile</h3>
            ${renderProfileDebug(profile)}
          </section>
          <section class="context-section">
            <h3>References</h3>
            ${renderReferences(sessionContextState.references || []) || `<div class="meta-line">not loaded</div>`}
          </section>
          <section class="context-section">
            <h3>Session Trace</h3>
            ${renderSessionTrace(sessionContextState.trace || [])}
          </section>
        </div>
      </details>`;
    }

    function renderSessionTrace(trace) {
      if (!Array.isArray(trace) || !trace.length) return `<div class="meta-line">not recorded yet</div>`;
      return `<ol class="trace-list">${trace.slice(-12).map((item, index) => {
        const details = item.details ? JSON.stringify(item.details, null, 2) : "";
        return `<li><strong>${escapeHtml(item.summary || item.event || `trace ${index + 1}`)}</strong>
          <div class="meta-line">${escapeHtml(item.event || "")} · ${escapeHtml(item.created_at || "")}</div>
          ${details && details !== "{}" ? `<details><summary>details</summary><pre class="debug-box">${escapeHtml(details)}</pre></details>` : ""}
        </li>`;
      }).join("")}</ol>`;
    }

    function renderProfileDebug(profile) {
      if (!profile || profile.available === false) {
        return `<div class="meta-line">${escapeHtml(profile?.error || "not available")}</div>`;
      }
      const mastery = profile.topic_mastery || {};
      const weak = Array.isArray(profile.weak_points) ? profile.weak_points : [];
      const due = Array.isArray(profile.due_reviews) ? profile.due_reviews : [];
      return `
        <div class="meta-line">topic: ${escapeHtml(profile.current_topic || "")}</div>
        <div class="meta-line">topic source: ${escapeHtml(profile.topic_source || "")}</div>
        <div class="meta-line">injected: ${profile.injected_to_prompt ? "yes" : "no"} · hash: ${escapeHtml(profile.prompt_context_sha256 || "")}</div>
        <div class="meta-line">weak: ${escapeHtml(profile.weak_points_count ?? 0)} · due: ${escapeHtml(profile.due_reviews_count ?? 0)} · strong: ${escapeHtml(profile.strong_points_count ?? 0)}</div>
        ${mastery.mastery_estimate !== undefined ? `<div class="meta-line">mastery estimate: ${escapeHtml(mastery.mastery_estimate)}/100 · active weak: ${escapeHtml(mastery.active_weak_count ?? "")}</div>` : ""}
        ${weak.length ? `<ul>${weak.map((item) => `<li>${escapeHtml(item.point || "")}<div class="meta-line">${escapeHtml(item.planned_layer || "")}</div></li>`).join("")}</ul>` : ""}
        ${due.length ? `<div class="meta-line">due reviews: ${escapeHtml(due.length)}</div>` : ""}
        ${profile.prompt_context_preview ? `<details><summary>Prompt profile context</summary><pre class="debug-box">${escapeHtml(profile.prompt_context_preview)}</pre></details>` : ""}
      `;
    }

    function renderReferences(items) {
      if (!Array.isArray(items) || !items.length) return "";
      return `<details class="refs"><summary>References (${items.length})</summary><div class="reference-list">${items.map(renderReference).join("")}</div></details>`;
    }

    function renderReference(item) {
      const path = item.path || item.url || item.provider || "";
      const lines = item.lines ? ` lines ${item.lines}` : item.line ? ` line ${item.line}` : "";
      const score = item.score !== null && item.score !== undefined ? ` score ${item.score}` : "";
      const citation = item.citation_id || item.id || "S";
      const originalText = item.text ? String(item.text) : "";
      const text = originalText.slice(0, 900);
      return `<div class="reference">
        <div><span class="ref-id">[${escapeHtml(citation)}]</span>${escapeHtml(lines)}${escapeHtml(score)}</div>
        <div class="path">${escapeHtml(path)}</div>
        ${item.heading ? `<div class="path">heading: ${escapeHtml(item.heading)}</div>` : ""}
        ${item.message ? `<div class="path">${escapeHtml(item.message)}</div>` : ""}
        ${text ? `<details><summary>excerpt</summary><pre>${escapeHtml(text)}${originalText.length > text.length ? "\n..." : ""}</pre></details>` : ""}
      </div>`;
    }

    function buildTurnDebug() {
      const slimItems = ((currentPayload.context && currentPayload.context.items) || []).map(slimReference);
      return {
        slim: {
          command: currentPayload.done?.command || null,
          retrieval: summarizeRetrieval(currentPayload.retrieval),
          context: {
            stats: currentPayload.context?.stats || null,
            item_count: slimItems.length,
            items: slimItems
          },
          interview_plan: summarizeInterviewPlan(currentPayload.interviewPlan),
          interview_state: currentInterviewState,
          profile_debug: currentPayload.profileDebug || currentPayload.done?.telemetry?.profile_debug || null,
          generation: currentPayload.done?.telemetry?.generation || currentPayload.done?.timing || null,
          errors: currentPayload.errors || []
        },
        raw: {
          router: currentPayload.router,
          retrieval: summarizeRetrieval(currentPayload.retrieval),
          context: currentPayload.context ? {...currentPayload.context, items: slimItems} : null,
          interview_plan: summarizeInterviewPlan(currentPayload.interviewPlan),
          profile_debug: currentPayload.profileDebug,
          answer: currentPayload.answer,
          done: currentPayload.done
        }
      };
    }

    function slimReference(item) {
      return {
        citation_id: item.citation_id || item.id || null,
        path: item.path || item.url || item.provider || null,
        heading: item.heading || null,
        lines: item.lines || item.line || null,
        score: item.score ?? null,
        message: item.message || null
      };
    }

    function summarizeRetrieval(retrieval) {
      if (!retrieval) return null;
      return {
        command: retrieval.command || null,
        summary: retrieval.reference_summary || retrieval.summary || null,
        total: retrieval.total ?? retrieval.count ?? null,
        elapsed_ms: retrieval.elapsed_ms ?? null
      };
    }

    function summarizeInterviewPlan(payload) {
      if (!payload) return null;
      const plan = payload.plan || {};
      return {
        available: payload.available ?? Boolean(payload.plan),
        source: payload.source || null,
        fallback_used: Boolean(payload.fallback_used),
        latency_ms: payload.latency_ms ?? null,
        topic_count: Array.isArray(plan.topics) ? plan.topics.length : 0,
        topics: Array.isArray(plan.topics) ? plan.topics.map((topic) => topic.name || "").filter(Boolean) : []
      };
    }

    function renderDebug(debug) {
      return `<details class="debug-details"><summary>Turn Debug</summary>
        <pre class="debug-box">${escapeHtml(JSON.stringify(debug.slim, null, 2))}</pre>
        <details><summary>Raw payload</summary><pre class="debug-box">${escapeHtml(JSON.stringify(debug.raw, null, 2))}</pre></details>
      </details>`;
    }

    function shouldRequestTurnSummary(query, answerText) {
      const priorAssistant = [...chatHistory].reverse().find((item) => item.role === "assistant")?.content || "";
      if (!priorAssistant.trim()) return false;
      const topicChoicePrompt =
        /which.*(topic|direction).*start|choose.*(topic|direction)/i.test(priorAssistant) ||
        priorAssistant.includes("\u5e0c\u671b\u4ece\u54ea\u4e2a") ||
        priorAssistant.includes("\u4f60\u4ece\u54ea\u4e2a\u5f00\u59cb") ||
        priorAssistant.includes("\u4ece\u54ea\u4e2a\u65b9\u5411") ||
        priorAssistant.includes("\u54ea\u4e2a\u4e3b\u9898");
      if (topicChoicePrompt) return false;
      const assistantText = answerText || "";
      const assistantOnlyAskedFirstQuestion =
        /first question/i.test(assistantText) ||
        assistantText.includes("\u5148\u95ee") ||
        assistantText.includes("\u5148\u6765") ||
        assistantText.includes("\u7b2c\u4e00\u4e2a\u95ee\u9898");
      const userText = query || "";
      const userOnlyChoseTopic =
        /^(I want to start with|start with)/i.test(userText) ||
        userText.includes("\u6211\u60f3\u4ece") ||
        (userText.includes("\u4ece") && userText.includes("\u5f00\u59cb"));
      if (assistantOnlyAskedFirstQuestion && userOnlyChoseTopic) return false;
      return true;
    }

    async function requestTurnSummary(answerText, turnIds) {
      try {
        const response = await fetch("/api/interview/summary", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({
          scope_type: $("scopeType").value,
          scope_value: $("scopeValue").value.trim() || null,
          scope_paths: scopePaths(),
          chat_history: chatHistory.slice(-12),
          answer: answerText,
          interview_plan: currentInterviewPlan,
          interview_state: currentInterviewState,
          session_id: currentInterviewSessionId || null,
          user_message_id: turnIds?.user?.id || null,
          assistant_message_id: turnIds?.assistant?.id || null,
          notes_top_k: numberValue("notesTopK"),
          dense_top_k: numberValue("denseTopK"),
          hybrid_bm25_top_k: numberValue("hybridBm25TopK"),
          rrf_k: numberValue("rrfK")
        })});
        if (!response.ok) return {available:false, error:"复盘生成失败"};
        return await response.json();
      } catch (error) {
        return {available:false, error:error.message};
      }
    }

    function renderSummaryIntoAssistant(assistantNode, summary) {
      if (!assistantNode) return;
      assistantNode.querySelectorAll("details.review").forEach((node) => node.remove());
      const html = renderSessionSummary(summary);
      if (html) assistantNode.insertAdjacentHTML("beforeend", html);
      scrollToBottom();
    }

    async function runTurnReviewInBackground(answerText, turnIds, assistantNode) {
      renderSummaryIntoAssistant(assistantNode, {pending: true});
      if (turnIds) await persistPendingTurnReview(turnIds);
      try {
        const summary = await requestTurnSummary(answerText, turnIds);
        renderSummaryIntoAssistant(assistantNode, summary);
        if (turnIds && summary && summary.available !== false) {
          await persistTurnReview(turnIds, summary);
        } else if (turnIds) {
          await persistFailedTurnReview(turnIds, new Error((summary && summary.error) || "review unavailable"));
        }
      } catch (error) {
        renderSummaryIntoAssistant(assistantNode, {available:false, error:error.message});
        if (turnIds) await persistFailedTurnReview(turnIds, error);
      }
    }

    function startNewConversation() {
      localStorage.removeItem(activeInterviewSessionKey);
      chatHistory = [];
      currentAssistant = null;
      currentAssistantText = "";
      currentPayload = null;
      currentInterviewPlan = null;
      currentInterviewPlanSignature = "";
      currentInterviewState = null;
      currentInterviewSessionId = "";
      lastContextItems = [];
      resetSessionContext();
      currentConversationSignature = conversationSignature();
      $("query").value = "";
      $("starters").innerHTML = "";
      messages.innerHTML = `<article class="message assistant"><div class="role">Assistant</div><div class="answer">已开始新对话。可以选择面试模式并发送消息开始。</div></article>`;
      updateSessionLabel();
      setBusy(false, "就绪");
      scrollToBottom();
    }

    async function restoreActiveInterviewSession() {
      const sessionId = localStorage.getItem(activeInterviewSessionKey);
      if (!sessionId) return;
      try {
        const response = await fetch(`/api/interview/sessions/${encodeURIComponent(sessionId)}`);
        if (!response.ok) {
          localStorage.removeItem(activeInterviewSessionKey);
          return;
        }
        const data = await response.json();
        const session = data.session || {};
        if (!["active", "end_failed"].includes(session.status)) {
          localStorage.removeItem(activeInterviewSessionKey);
          return;
        }
        restoreSessionIntoChat(session, data.reviews || []);
      } catch {
        localStorage.removeItem(activeInterviewSessionKey);
      }
    }

    function restoreSessionIntoChat(session, reviews) {
      const context = session.context || {};
      $("chatMode").value = "interview";
      if (context.source_type) $("scopeType").value = context.source_type;
      if (context.source_type === "selected_notes") {
        $("scopeValue").value = (context.source_paths || []).join("\n");
      } else if (context.source_value) {
        $("scopeValue").value = context.source_value;
      }

      currentInterviewSessionId = session.session_id || "";
      currentInterviewPlan = session.interview_plan || null;
      currentInterviewPlanSignature = currentInterviewPlan ? conversationSignature() : "";
      currentInterviewState = session.interview_state || null;
      currentConversationSignature = conversationSignature();
      localStorage.setItem(activeInterviewSessionKey, currentInterviewSessionId);
      sessionContextState = sessionContextFromSession(session);
      renderSessionContext();

      const reviewByAssistant = new Map((reviews || []).map((review) => [review.assistant_message_id, review]));
      const sessionMessages = Array.isArray(session.messages) ? session.messages : [];
      chatHistory = sessionMessages
        .filter((message) => ["user", "assistant"].includes(message.role))
        .filter((message) => message.role !== "assistant" || !["pending", "failed"].includes(message.status))
        .map((message) => ({role: message.role, content: message.content || ""}));
      trimChatHistory();
      messages.innerHTML = sessionMessages.length
        ? sessionMessages.map((message, index) => renderMessageWithReview(message, reviewByAssistant.get(message.id), sessionMessages[index - 1])).join("")
        : `<article class="message assistant"><div class="role">Assistant</div><div class="answer">已恢复未结束的面试记录。</div></article>`;
      updateSessionLabel();
      attachRestoredRetryButtons();
      setBusy(false, "已恢复未结束的面试");
      scrollToBottom();
    }

    function renderMessageWithReview(message, review, previousMessage) {
      const cls = message.role === "user" ? "user" : "assistant";
      const status = message.status || "completed";
      const errorHtml = status === "failed" ? `<p class="error">${escapeHtml(message.error_message || "assistant generation failed")}</p>` : "";
      const pendingHtml = status === "pending" ? `<p class="history-meta">pending generation</p>` : "";
      const retryHtml = status === "failed" && message.retryable !== false && previousMessage && previousMessage.role === "user"
        ? `<button type="button" class="retry-answer restored-retry" data-user-id="${escapeHtml(previousMessage.id || "")}" data-assistant-id="${escapeHtml(message.id || "")}" data-query="${escapeHtml(previousMessage.content || "")}">&#8635;</button>`
        : "";
      const reviewSummary = review ? {
        status: review.status,
        pending: review.status === "pending",
        error: review.error,
        available: review.status !== "failed",
        feedback: review.feedback || {},
        expression_example: review.expression_example || review.reference_answer || "",
        reference_answer: review.reference_answer || "",
        profile_signals: review.profile_signals || []
      } : null;
      return `<article class="message ${cls}"><div class="role">${escapeHtml(message.role || "")}</div><div class="answer">${renderMarkdown(message.content || "")}${errorHtml}${pendingHtml}</div>${retryHtml}${reviewSummary ? renderSessionSummary(reviewSummary) : ""}</article>`;
    }

    function attachRestoredRetryButtons() {
      messages.querySelectorAll(".restored-retry").forEach((button) => {
        button.addEventListener("click", () => {
          const article = button.closest(".message.assistant");
          const turnIds = {user:{id:button.dataset.userId || ""}, assistant:{id:button.dataset.assistantId || ""}};
          retryAssistantAnswer(button.dataset.query || "", turnIds, article);
        });
      });
    }

    async function ensureInterviewSession() {
      if (currentInterviewSessionId) return currentInterviewSessionId;
      const response = await fetch("/api/interview/sessions", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({
        source_type: $("scopeType").value,
        source_value: $("scopeValue").value.trim() || null,
        source_paths: scopePaths(),
        source_note_paths: sourceNotePaths(),
        interview_plan: currentInterviewPlan,
        interview_state: currentInterviewState,
        extra: {created_from:"chat"}
      })});
      if (!response.ok) throw new Error("创建面试记录失败");
      const data = await response.json();
      currentInterviewSessionId = data.session.session_id;
      localStorage.setItem(activeInterviewSessionKey, currentInterviewSessionId);
      updateSessionLabel();
      return currentInterviewSessionId;
    }

    async function persistPendingInterviewTurn(userContent) {
      const sessionId = await ensureInterviewSession();
      const response = await fetch(`/api/interview/sessions/${encodeURIComponent(sessionId)}/turns/pending`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({
        user_content: userContent,
        interview_plan: currentInterviewPlan,
        interview_state: currentInterviewState,
        source_note_paths: sourceNotePaths()
      })});
      if (!response.ok) throw new Error("failed to save pending interview turn");
      const data = await response.json();
      return {user: data.user_message, assistant: data.assistant_message};
    }

    async function completeInterviewTurn(turnIds, assistantContent) {
      if (!turnIds || !currentInterviewSessionId) return null;
      const agentActions = currentProcess.actionRuns.length ? currentProcess.actionRuns : (currentPayload?.done?.agent_actions || currentPayload?.done?.telemetry?.agent_actions || []);
      const response = await fetch(`/api/interview/sessions/${encodeURIComponent(currentInterviewSessionId)}/messages/${encodeURIComponent(turnIds.assistant.id)}/complete`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({
        assistant_content: assistantContent,
        interview_plan: currentInterviewPlan,
        interview_state: currentInterviewState,
        source_note_paths: sourceNotePaths(),
        agent_actions: agentActions
      })});
      if (!response.ok) return null;
      const data = await response.json();
      addLocalSessionTrace({
        event: "assistant_completed",
        summary: `assistant completed: ${turnIds.assistant.id}`,
        details: {assistant_message_id: turnIds.assistant.id, output_chars: assistantContent.length}
      });
      return data.assistant_message || null;
    }

    async function failInterviewTurn(turnIds, assistantContent, error) {
      if (!turnIds || !currentInterviewSessionId) return null;
      const response = await fetch(`/api/interview/sessions/${encodeURIComponent(currentInterviewSessionId)}/messages/${encodeURIComponent(turnIds.assistant.id)}/fail`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({
        assistant_content: assistantContent || "",
        error_type: error.errorType || error.name || "Error",
        error_message: error.message || String(error),
        retryable: error.retryable !== false
      })});
      if (!response.ok) return null;
      const data = await response.json();
      addLocalSessionTrace({
        event: "assistant_failed",
        summary: `assistant failed: ${turnIds.assistant.id}`,
        details: {
          assistant_message_id: turnIds.assistant.id,
          error_type: error.errorType || error.name || "Error",
          error_message: error.message || String(error),
          retryable: error.retryable !== false
        }
      });
      return data.assistant_message || null;
    }

    async function persistInterviewTurn(userContent, assistantContent) {
      const sessionId = await ensureInterviewSession();
      const agentActions = currentProcess.actionRuns.length ? currentProcess.actionRuns : (currentPayload?.done?.agent_actions || currentPayload?.done?.telemetry?.agent_actions || []);
      const response = await fetch(`/api/interview/sessions/${encodeURIComponent(sessionId)}/turns`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({
        user_content: userContent,
        assistant_content: assistantContent,
        interview_plan: currentInterviewPlan,
        interview_state: currentInterviewState,
        source_note_paths: sourceNotePaths(),
        agent_actions: agentActions
      })});
      if (!response.ok) return null;
      const data = await response.json();
      return {user: data.user_message, assistant: data.assistant_message};
    }

    async function persistPendingTurnReview(turnIds) {
      if (!turnIds || !currentInterviewSessionId) return null;
      const response = await fetch(`/api/interview/sessions/${encodeURIComponent(currentInterviewSessionId)}/reviews/pending`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({
        user_message_id: turnIds.user.id,
        assistant_message_id: turnIds.assistant.id,
        context_note_paths: sourceNotePaths()
      })});
      if (!response.ok) return null;
      addLocalSessionTrace({
        event: "turn_review_pending",
        summary: `turn_review pending: ${turnIds.assistant.id}`,
        details: {user_message_id: turnIds.user.id, assistant_message_id: turnIds.assistant.id}
      });
      return await response.json();
    }

    async function persistTurnReview(turnIds, summary) {
      if (!turnIds || !currentInterviewSessionId) return;
      await fetch(`/api/interview/sessions/${encodeURIComponent(currentInterviewSessionId)}/reviews`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({
        user_message_id: turnIds.user.id,
        assistant_message_id: turnIds.assistant.id,
        feedback: summary.feedback || {},
        expression_example: summary.expression_example || summary.reference_answer || "",
        reference_answer: summary.reference_answer || summary.expression_example || "",
        context_note_paths: summary.context_note_paths || sourceNotePaths(),
        profile_signals: summary.profile_signals || []
      })});
      const signals = Array.isArray(summary.profile_signals) ? summary.profile_signals : [];
      addLocalSessionTrace({
        event: "turn_review_completed",
        summary: "turn_review completed: profile via session_end_extractor",
        details: {
          user_message_id: turnIds.user.id,
          assistant_message_id: turnIds.assistant.id,
          profile_signal_count: signals.length,
          profile_signal_types: signals.map((signal) => signal.type || "").filter(Boolean),
          profile_signals_disabled: true,
          profile_write_source: "session_end_extractor"
        }
      });
    }

    async function persistFailedTurnReview(turnIds, error) {
      if (!turnIds || !currentInterviewSessionId) return;
      await fetch(`/api/interview/sessions/${encodeURIComponent(currentInterviewSessionId)}/reviews/failed`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({
        user_message_id: turnIds.user.id,
        assistant_message_id: turnIds.assistant.id,
        error: error.message || String(error),
        context_note_paths: sourceNotePaths()
      })});
      addLocalSessionTrace({
        event: "turn_review_failed",
        summary: "turn_review failed",
        details: {user_message_id: turnIds.user.id, assistant_message_id: turnIds.assistant.id, error: error.message || String(error)}
      });
    }

    async function endInterviewSession() {
      if (!currentInterviewSessionId) return;
      setBusy(true, "正在结束面试...");
      try {
        const response = await fetch(`/api/interview/sessions/${encodeURIComponent(currentInterviewSessionId)}/end`, {method:"POST"});
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || "end interview failed");
        const operations = (data.session && data.session.profile_update && data.session.profile_update.operations) || {};
        addLocalSessionTrace({
          event: data.session && data.session.status === "completed" ? "session_completed" : "session_end_failed",
          summary: data.session && data.session.status === "completed"
            ? `profile extraction completed: ${(operations.added || []).length} ADD, ${(operations.updated || []).length} UPDATE, ${(operations.partial || []).length} PARTIAL, ${(operations.improved || []).length} IMPROVE`
            : "session end failed",
          details: {
            status: data.session?.status || "",
            operations
          }
        });
        appendSystemMessage(data.session && data.session.status === "completed" ? "本次面试已保存，长期画像已更新。" : "本次面试已保存，但最终总结或画像更新失败。可以稍后从历史记录重试。");
        currentInterviewSessionId = "";
        localStorage.removeItem(activeInterviewSessionKey);
        updateSessionLabel();
      } catch (error) {
        appendSystemMessage(`结束失败：${error.message}`);
      } finally {
        setBusy(false, "就绪");
      }
    }

    async function openHistory() {
      $("historyModal").classList.add("open");
      $("historyModal").setAttribute("aria-hidden", "false");
      await loadInterviewHistoryListOnly();
    }

    function closeHistory() {
      $("historyModal").classList.remove("open");
      $("historyModal").setAttribute("aria-hidden", "true");
    }

    async function loadInterviewHistoryListOnly() {
      $("historyContent").innerHTML = "加载中...";
      try {
        const response = await fetch("/api/interview/sessions?limit=30");
        if (!response.ok) throw new Error("历史记录加载失败");
        const data = await response.json();
        renderInterviewHistory(data.sessions || []);
      } catch (error) {
        $("historyContent").innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
      }
    }

    function renderInterviewHistory(sessions) {
      if (!sessions.length) { $("historyContent").innerHTML = "暂无历史面试。"; return; }
      $("historyContent").innerHTML = sessions.map((session) => `<button class="history-item" type="button" data-session="${escapeHtml(session.session_id || "")}"><strong>${escapeHtml(session.title || session.session_id || "面试记录")}</strong><div class="history-meta">${escapeHtml(session.status || "")} | ${escapeHtml(session.updated_at || session.created_at || "")}</div></button>`).join("");
      document.querySelectorAll("[data-session]").forEach((button) => button.addEventListener("click", () => loadInterviewSession(button.dataset.session)));
    }

    async function loadInterviewSession(sessionId) {
      $("historyContent").innerHTML = "正在加载面试记录...";
      try {
        const response = await fetch(`/api/interview/sessions/${encodeURIComponent(sessionId)}`);
        if (!response.ok) throw new Error("面试记录加载失败");
        const data = await response.json();
        renderInterviewSessionDetail(data.session, data.reviews || []);
      } catch (error) {
        $("historyContent").innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
      }
    }

    function renderInterviewSessionDetail(session, reviews) {
      const reviewByAssistant = new Map((reviews || []).map((review) => [review.assistant_message_id, review]));
      const contextHtml = renderHistorySessionContext(session);
      $("historyContent").innerHTML = `<button class="secondary" type="button" id="backHistoryBtn">返回</button><div class="history-meta" style="margin:8px 0 12px">${escapeHtml(session.session_id || "")} | ${escapeHtml(session.status || "")}</div>${contextHtml}${(session.messages || []).map((message) => {
        const cls = message.role === "user" ? "user" : "assistant";
        const review = reviewByAssistant.get(message.id);
        return `<article class="message ${cls}"><div class="role">${escapeHtml(message.role || "")}</div><div class="answer">${renderMarkdown(message.content || "")}</div>${review ? renderSessionSummary({available:true, feedback:review.feedback || {}, expression_example:review.expression_example || review.reference_answer || "", reference_answer:review.reference_answer || "", profile_signals:review.profile_signals || []}) : ""}</article>`;
      }).join("") || `<article class="message assistant">暂无消息。</article>`}`;
      $("backHistoryBtn").addEventListener("click", loadInterviewHistoryListOnly);
    }

    function sessionContextFromSession(session) {
      const context = session.context || {};
      const references = (context.source_note_paths || []).map((path, index) => ({
        citation_id: `S${index + 1}`,
        path
      }));
      return {
        scope: {
          type: context.source_type || "",
          value: context.source_value || "",
          paths: context.source_paths || []
        },
        stats: {
          source_notes: references.length
        },
        references,
        interviewPlan: session.interview_plan ? {available: true, source: "session", plan: session.interview_plan} : null,
        profileDebug: null,
        trace: Array.isArray(session.trace) ? session.trace : []
      };
    }

    function renderHistorySessionContext(session) {
      const previous = sessionContextState;
      sessionContextState = sessionContextFromSession(session);
      const scope = sessionContextState.scope || {};
      const plan = (sessionContextState.interviewPlan && sessionContextState.interviewPlan.plan) || {};
      const topics = Array.isArray(plan.topics) ? plan.topics : [];
      const html = `<details class="session-panel" open><summary>Session Context</summary>
        <div class="context-grid">
          <section class="context-section">
            <h3>Scope</h3>
            <div class="meta-line">type: ${escapeHtml(scope.type || "")}</div>
            <div class="meta-line">value: ${escapeHtml(scope.value || "")}</div>
            ${Array.isArray(scope.paths) && scope.paths.length ? `<ul>${scope.paths.map((path) => `<li>${escapeHtml(path)}</li>`).join("")}</ul>` : ""}
          </section>
          <section class="context-section">
            <h3>Interview Plan</h3>
            ${topics.length ? `<ul>${topics.map((topic) => `<li><strong>${escapeHtml(topic.name || "")}</strong>${Array.isArray(topic.coverage) && topic.coverage.length ? `：${escapeHtml(topic.coverage.join(" / "))}` : ""}</li>`).join("")}</ul>` : `<div class="meta-line">not available</div>`}
          </section>
          <section class="context-section">
            <h3>References</h3>
            ${renderReferences(sessionContextState.references || []) || `<div class="meta-line">not loaded</div>`}
          </section>
          <section class="context-section">
            <h3>Session Trace</h3>
            ${renderSessionTrace(sessionContextState.trace || [])}
          </section>
        </div>
      </details>`;
      sessionContextState = previous;
      return html;
    }

    function renderSessionSummary(summary) {
      if (!summary) return "";
      if (summary.pending) return `<details class="review"><summary>本轮复盘</summary><div class="history-meta">生成中...</div></details>`;
      if (summary.available === false) return `<details class="review"><summary>本轮复盘</summary><div class="error">${escapeHtml(summary.error || "不可用")}</div></details>`;
      const feedback = summary.feedback || {};
      const gaps = feedback.gaps || feedback.missing || feedback.could_cover || [];
      const coachNote = feedback.coach_note || feedback.overall || feedback.summary || "";
      const followupNote = feedback.interviewer_followup_note || feedback.interviewer_direction || "";
      const feedbackText = [coachNote, followupNote].filter(Boolean).join("\n\n");
      const thinkingFramework = feedback.thinking_framework || feedback.next_focus || feedback.next_tip || feedback.next_step || "";
      const expressionExample = summary.expression_example || summary.reference_answer || "";
      const list = (items) => Array.isArray(items) && items.length ? `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : "";
      const paragraph = (text) => text ? `<p>${escapeHtml(text)}</p>` : "";
      const hasFeedback = feedbackText || (Array.isArray(gaps) && gaps.length) || thinkingFramework || expressionExample;
      if (!hasFeedback) return "";
      return `<details class="review" open><summary>本轮复盘</summary>
        <div class="review-section"><h4>反馈</h4>${paragraph(feedbackText)}</div>
        <div class="review-section"><h4>你可改进的点</h4>${list(gaps)}</div>
        <div class="review-section"><h4>思路构建</h4>${paragraph(thinkingFramework)}</div>
        ${expressionExample ? `<div class="review-section reference-answer"><h4>表达示例</h4>${renderMarkdown(expressionExample)}</div>` : ""}
      </details>`;
    }

    function updateInterviewState(userText, assistantText) {
      if (!currentInterviewState) currentInterviewState = {current_topic:null, current_layer_index:0, follow_up_count:0, sub_points_touched:[], last_user_answer:""};
      const before = JSON.parse(JSON.stringify(currentInterviewState || {}));
      let topicChanged = false;
      let layerTransition = false;
      currentInterviewState.last_user_answer = userText.slice(0, 300);
      const topic = inferTopic(userText + "\n" + assistantText);
      if (topic && topic !== currentInterviewState.current_topic) {
        currentInterviewState.current_topic = topic;
        currentInterviewState.current_layer_index = 0;
        currentInterviewState.follow_up_count = 0;
        currentInterviewState.sub_points_touched = [];
        topicChanged = true;
      }
      currentInterviewState.follow_up_count = (currentInterviewState.follow_up_count || 0) + 1;
      const question = extractLastQuestion(assistantText);
      if (question) {
        currentInterviewState.sub_points_touched = [...(currentInterviewState.sub_points_touched || []), question].slice(-8);
      }
      if (detectLayerTransition(assistantText)) {
        currentInterviewState.current_layer_index = (currentInterviewState.current_layer_index || 0) + 1;
        currentInterviewState.follow_up_count = 0;
        currentInterviewState.sub_points_touched = [];
        layerTransition = true;
      }
      return {
        before,
        after: JSON.parse(JSON.stringify(currentInterviewState || {})),
        topic_changed: topicChanged,
        layer_transition: layerTransition,
        transition_source: layerTransition ? (currentTurnTrace.directorNoteInjected ? "director_note" : "llm_self") : "",
      };
    }

    function isServerInterviewState() {
      if (!currentInterviewState) return false;
      if (currentInterviewState.source === "server") return true;
      if (currentInterviewState.topic_phase) return true;
      return false;
    }

    function recordInterviewStateTrace(change) {
      if (!change) return;
      if (change.topic_changed) {
        recordSessionTrace("topic_changed", `topic changed: ${change.after?.current_topic || ""}`, change);
      }
      if (change.layer_transition) {
        const source = change.transition_source === "director_note" ? "Director Note" : "LLM self";
        recordSessionTrace("layer_transition", `layer transition by ${source}`, change);
      }
    }

    function inferTopic(text) {
      const topics = currentInterviewPlan && Array.isArray(currentInterviewPlan.topics) ? currentInterviewPlan.topics : [];
      for (const topic of topics) {
        const name = topic.name || "";
        if (name && text.includes(name)) return name;
      }
      return null;
    }

    function detectLayerTransition(text) {
      return /(next layer|next dimension|switch to|move to|move on|next topic|next section)/i.test(text || "");
    }

    function extractLastQuestion(text) {
      const parts = String(text || "").split(/(?<=[??])/).map((item) => item.trim()).filter(Boolean);
      return parts.length ? parts[parts.length - 1].slice(0, 120) : "";
    }

    function appendUserMessage(text) {
      const node = document.createElement("article");
      node.className = "message user";
      node.innerHTML = `<div class="role">You</div><div class="answer">${renderMarkdown(text)}</div>`;
      messages.appendChild(node);
      scrollToBottom();
      return node;
    }

    function appendAssistantMessage(text = "") {
      const node = document.createElement("article");
      node.className = "message assistant";
      node.innerHTML = `<div class="role">Assistant</div><details class="agent-process" hidden open></details><div class="answer">${text ? renderMarkdown(text) : ""}</div>`;
      messages.appendChild(node);
      scrollToBottom();
      return node;
    }

    function appendSystemMessage(text) {
      const node = document.createElement("article");
      node.className = "message assistant";
      node.innerHTML = `<div class="role">System</div><div class="answer">${escapeHtml(text)}</div>`;
      messages.appendChild(node);
      scrollToBottom();
    }

    function renderMarkdown(text) {
      const codeBlocks = [];
      let safe = escapeHtml(text || "").replace(/```([\s\S]*?)```/g, (_, code) => {
        const key = `@@CODE_${codeBlocks.length}@@`;
        codeBlocks.push(`<pre><code>${code}</code></pre>`);
        return key;
      });
      const lines = safe.split(/\r?\n/);
      const html = [];
      let list = [];
      let orderedList = [];
      const flush = () => {
        if (list.length) { html.push(`<ul>${list.map((item) => `<li>${inlineMarkdown(item)}</li>`).join("")}</ul>`); list = []; }
        if (orderedList.length) { html.push(`<ol>${orderedList.map((item) => `<li>${inlineMarkdown(item)}</li>`).join("")}</ol>`); orderedList = []; }
      };
      for (let i = 0; i < lines.length; i += 1) {
        const line = lines[i];
        const trimmed = line.trim();
        if (!trimmed) {
          const previous = lines[i - 1] ? lines[i - 1].trim() : "";
          const next = lines[i + 1] ? lines[i + 1].trim() : "";
          if (isTableRow(previous) && isTableRow(next)) continue;
          flush();
          continue;
        }
        if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) { flush(); html.push("<hr>"); continue; }
        const heading = /^(#{1,4})\s+(.+)$/.exec(trimmed);
        if (heading) { flush(); html.push(`<h${Math.min(heading[1].length + 1, 5)}>${inlineMarkdown(heading[2])}</h${Math.min(heading[1].length + 1, 5)}>`); continue; }
        if (isTableRow(trimmed) && isTableSeparator(lines[i + 1] ? lines[i + 1].trim() : "")) {
          flush();
          const rows = [trimmed];
          i += 2;
          while (i < lines.length) {
            const row = lines[i].trim();
            if (!row) {
              const next = lines[i + 1] ? lines[i + 1].trim() : "";
              if (isTableRow(next)) { i += 1; continue; }
              break;
            }
            if (!isTableRow(row)) { i -= 1; break; }
            rows.push(row);
            i += 1;
          }
          html.push(renderMarkdownTable(rows));
          continue;
        }
        const bullet = /^[-*]\s+(.+)$/.exec(trimmed);
        if (bullet) { if (orderedList.length) flush(); list.push(bullet[1]); continue; }
        const ordered = /^\d+[.)]\s+(.+)$/.exec(trimmed);
        if (ordered) { if (list.length) flush(); orderedList.push(ordered[1]); continue; }
        flush();
        html.push(`<p>${inlineMarkdown(trimmed)}</p>`);
      }
      flush();
      let rendered = html.join("");
      codeBlocks.forEach((block, index) => { rendered = rendered.replace(`@@CODE_${index}@@`, block); });
      return rendered;
    }

    function isTableRow(text) {
      const value = String(text || "").trim();
      return value.includes("|") && /^\|?.+\|.+\|?$/.test(value);
    }

    function isTableSeparator(text) {
      const cells = splitTableRow(text);
      return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell.trim()));
    }

    function splitTableRow(text) {
      let value = String(text || "").trim();
      if (!value.includes("|")) return [];
      if (value.startsWith("|")) value = value.slice(1);
      if (value.endsWith("|")) value = value.slice(0, -1);
      return value.split("|").map((cell) => cell.trim());
    }

    function renderMarkdownTable(rows) {
      const header = splitTableRow(rows[0]);
      const body = rows.slice(1).map(splitTableRow).filter((cells) => cells.length);
      const headerHtml = header.map((cell) => `<th>${inlineMarkdown(cell)}</th>`).join("");
      const bodyHtml = body.map((cells) => {
        const padded = header.length ? cells.concat(Array(Math.max(0, header.length - cells.length)).fill("")) : cells;
        return `<tr>${padded.slice(0, Math.max(header.length, cells.length)).map((cell) => `<td>${inlineMarkdown(cell)}</td>`).join("")}</tr>`;
      }).join("");
      return `<table><thead><tr>${headerHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`;
    }

    function inlineMarkdown(text) {
      return String(text || "").replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\[(S|N|R|B|W|E)(\d+)\]/g, '<span class="citation">[$1$2]</span>');
    }

    function sourceNotePaths() {
      return Array.from(new Set((lastContextItems || []).map((item) => item.path || item.relative_path || "").filter(Boolean)));
    }

    function trimChatHistory() {
      if (chatHistory.length > 24) chatHistory = chatHistory.slice(-24);
    }

    function updateSessionLabel() {
      $("endInterview").disabled = !currentInterviewSessionId;
      $("sessionLabel").textContent = currentInterviewSessionId ? `面试记录：${currentInterviewSessionId}` : "暂无进行中的面试";
    }

    function setBusy(isBusy, text) {
      $("sendBtn").disabled = isBusy;
      $("status").textContent = text || (isBusy ? "处理中..." : "就绪");
    }

    function numberValue(id) { return Number($(id).value); }
    function scopePaths() { return $("scopeType").value === "selected_notes" ? $("scopeValue").value.split(/\r?\n/).map((item) => item.trim()).filter(Boolean) : []; }
    function conversationSignature() { return JSON.stringify({mode:$("chatMode").value, scopeType:$("scopeType").value, scopeValue:$("scopeValue").value.trim(), scopePaths:scopePaths(), strictEvidence:$("strictEvidence").checked}); }
    function scrollToBottom() { messages.scrollTop = messages.scrollHeight; }
    function escapeHtml(value) { return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;"); }
  </script>
</body>
</html>
"""
