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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, StreamingResponse
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
from agent.schema import AgentMessage, AgentRunConfig, AgentState, WorkingMemory
from agent.skill_loader import SkillLoader
from agent.tool_registry import ToolRegistry
from agent.tool_executor import ToolExecutionContext
from agent.tools import register_debug_tools, register_interview_tools, register_profile_tools, register_review_tools
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
from services.workflows.answer_sessions import AnswerSessionStore
from services.workflows.review_runs import ReviewRunStore
from services.workflows.review_run_service import ReviewRunService
from services.workflows.review_practice import (
    build_due_review_overview,
    build_correction_query,
    build_grouped_review_prompt,
    build_grouped_weak_point_verification_query,
    build_weak_point_verification_query,
    build_recall_prompt,
    commit_review_action,
    commit_review_outcome,
    find_weak_point,
    grouped_review_cards,
    matching_strategy_constraints_for_card,
    parse_correction_payload,
    parse_grouped_verification_payload,
    parse_verification_payload,
    read_review_card_cache,
    read_review_prompt_cache,
    review_prompt_cache_key,
    select_strategy_constraints,
    write_review_card_cache,
    write_review_prompt_cache,
    REVIEW_PROMPT_VERSION,
)
from services.workflows.runner import WorkflowRunner, task_result_to_dict
from services.workflows.schema import ScopeSpec, WorkflowSpec, WritebackSpec
from services.workflows.scope_resolver import ScopeResolver
from ports.file_answer_session_repository import FileAnswerSessionRepository
from ports.file_review_run_repository import FileReviewRunRepository
from ports.file_session_repository import FileSessionRepository
from ports.in_memory_agent_run_repository import InMemoryAgentRunRepository
from services.agent_turns import AgentTurnInput, AgentTurnRunner, AgentTurnRunnerDeps, AgentTurnService
from services.tasks.pipeline_task import PipelineTaskContext, PipelineTaskManager
from services.tasks.stream import stream_task_events


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
    assistant_message_id: str | None = Field(default=None)


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
    citations: list[dict[str, Any]] = Field(default_factory=list)


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
    citations: list[dict[str, Any]] = Field(default_factory=list)


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


class ReviewPlanRequest(BaseModel):
    topics: list[str] = Field(default_factory=list)
    question_types: list[str] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=50)
    allow_cross_topic: bool = Field(default=True)
    max_strategy_constraints: int = Field(default=2, ge=0, le=5)
    force_regenerate: bool = Field(default=False)


class ReviewPrepareRequest(BaseModel):
    topics: list[str] = Field(default_factory=list)
    limit: int = Field(default=12, ge=1, le=50)
    max_strategy_constraints: int = Field(default=2, ge=0, le=5)


class ReviewPromptRequest(BaseModel):
    allowed_question_types: list[str] = Field(default_factory=list)
    related_topics: list[str] = Field(default_factory=list)


class ReviewGradeRequest(BaseModel):
    prompt: str = Field(default="")
    answer: str = Field(default="")


class ReviewCommitRequest(BaseModel):
    outcome: str = Field(default="")


class ReviewVerifyRequest(BaseModel):
    weak_point_id: str = Field(default="")
    weak_point_ids: list[str] = Field(default_factory=list)
    card_id: str = Field(default="")
    answer: str = Field(default="")
    prompt: str = Field(default="")
    question_blocks: list[dict[str, Any]] = Field(default_factory=list)


class ReviewActionCommitRequest(BaseModel):
    weak_point_id: str = Field(default="")
    action: str = Field(default="")


class ReviewDialogueRequest(BaseModel):
    topic: str = Field(default="")
    topics: list[str] = Field(default_factory=list)
    message: str = Field(default="")
    chat_history: list[dict[str, str]] = Field(default_factory=list)


class ReviewRunWorkspacePatchRequest(BaseModel):
    mode: str | None = None
    selectionState: dict[str, Any] | None = None
    cardReviewState: dict[str, Any] | None = None
    dialogueReviewState: dict[str, Any] | None = None


class ReviewDialogueSessionRequest(BaseModel):
    topics: list[str] = Field(default_factory=list)


class AnswerSessionCreateRequest(BaseModel):
    scope_type: str = Field(default="all")
    scope_value: str | None = None
    scope_paths: list[str] = Field(default_factory=list)
    strict_evidence: bool = Field(default=False)
    extra: dict[str, Any] = Field(default_factory=dict)


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
    answer_session_store = AnswerSessionStore(project_root / "answer-sessions")
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
    review_prepare_cache: dict[str, dict[str, Any]] = {}
    review_cache_dir = project_root / "review-cache"
    review_run_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="review-run")
    review_run_store = ReviewRunStore(project_root / "review-runs")
    review_run_repo = FileReviewRunRepository(review_run_store)
    review_run_service = ReviewRunService(
        repository=review_run_repo,
        profile_store=interview_profile_store,
        review_cache_dir=review_cache_dir,
        project_root=project_root,
        executor=review_run_executor,
    )

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        if request.query_params.get("mode") == "study":
            return RedirectResponse("/review", status_code=307)
        return render_web_page(CHAT_HTML, "对话")

    @app.get("/search", response_class=HTMLResponse)
    def index(request: Request) -> str:
        title = "检索调试" if request.query_params.get("debug") == "true" else "笔记搜索"
        return render_web_page(INDEX_HTML, title)

    @app.get("/chat", response_class=HTMLResponse)
    def chat(request: Request):
        mode = request.query_params.get("mode")
        if mode == "study":
            return RedirectResponse("/review", status_code=307)
        target_mode = "interview" if mode == "interview" else "answer"
        return RedirectResponse(f"/?mode={target_mode}", status_code=307)

    @app.get("/review", response_class=HTMLResponse)
    def review() -> str:
        return render_web_page(REVIEW_HTML, "复习")

    @app.get("/topics", response_class=HTMLResponse)
    def topics() -> str:
        return render_web_page(TOPICS_HTML, "知识主题")

    @app.get("/audit")
    def audit():
        return RedirectResponse("/organize", status_code=307)

    @app.get("/organize", response_class=HTMLResponse)
    def organize() -> str:
        return render_web_page(AUDIT_HTML, "整理")

    @app.get("/wiki", response_class=HTMLResponse)
    def wiki() -> str:
        return render_web_page(WIKI_READER_HTML, "Wiki")

    @app.get("/admin/wiki", response_class=HTMLResponse)
    def admin_wiki() -> str:
        return render_web_page(WIKI_HTML, "Wiki Admin")

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
        return render_web_page(SETTINGS_HTML, "设置")

    @app.get("/api/workspace/config")
    def workspace_config() -> dict[str, Any]:
        return {
            "config": runtime_config.to_dict(),
            "config_path": str(config_path),
            "validation": validate_workspace_config(runtime_config),
        }

    @app.get("/api/review/due")
    def review_due(limit: int = 50) -> dict[str, Any]:
        profile = interview_profile_store.load()
        return build_due_review_overview(profile, limit=limit)

    def build_review_prompt_for_card(weak: dict[str, Any], request: ReviewPromptRequest | None = None) -> dict[str, Any]:
        prompt_request = request or ReviewPromptRequest()
        cache_key = review_prompt_cache_key(weak)
        cached = read_review_prompt_cache(review_cache_dir, cache_key)
        if cached:
            result = dict(cached)
            result["cache_hit"] = True
            return result
        try:
            llm_config = load_llm_config(project_root)
            llm_client = create_llm_client(llm_config)
            prompt = build_recall_prompt(
                weak,
                allowed_question_types=prompt_request.allowed_question_types,
                related_topics=prompt_request.related_topics,
                llm_client=llm_client,
                model=llm_config.model,
                temperature=min(llm_config.temperature, 0.2),
            )
            write_review_prompt_cache(review_cache_dir, cache_key, prompt)
            return prompt
        except Exception as exc:
            prompt = build_recall_prompt(
                weak,
                allowed_question_types=prompt_request.allowed_question_types,
                related_topics=prompt_request.related_topics,
            )
            prompt["error"] = str(exc)
            return prompt

    @app.post("/api/review/prepare")
    def review_prepare(request: ReviewPrepareRequest) -> dict[str, Any]:
        payload = review_run_service.create_plan_run(request)
        review_prepare_cache[payload["review_run_id"]] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        return payload

    @app.get("/api/review/prepare/{review_run_id}")
    def review_prepared_run(review_run_id: str) -> dict[str, Any]:
        snapshot = review_run_service.snapshot(review_run_id)
        if snapshot is not None:
            return snapshot
        cached = review_prepare_cache.get(review_run_id)
        if cached is None:
            raise HTTPException(status_code=404, detail=f"review run not found: {review_run_id}")
        return cached["payload"]

    @app.post("/api/review/plan")
    def review_plan(request: ReviewPlanRequest) -> dict[str, Any]:
        return review_run_service.create_plan_run(request)

    @app.get("/api/review/plan/{review_run_id}")
    def review_plan_run(review_run_id: str) -> dict[str, Any]:
        snapshot = review_run_service.snapshot(review_run_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail=f"review run not found: {review_run_id}")
        return snapshot

    @app.post("/api/review/plan/{review_run_id}/cards/{card_id}/regenerate")
    def regenerate_review_card(review_run_id: str, card_id: str) -> dict[str, Any]:
        try:
            return review_run_service.regenerate_card(review_run_id, card_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.patch("/api/review/runs/{review_run_id}/workspace")
    def patch_review_run_workspace(review_run_id: str, request: ReviewRunWorkspacePatchRequest) -> dict[str, Any]:
        workspace = request.model_dump(exclude_none=True)
        if not workspace:
            raise HTTPException(status_code=400, detail="workspace patch is empty")
        try:
            return review_run_service.patch_workspace(review_run_id, workspace)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/review/dialogue/sessions")
    def create_dialogue_review_session(request: ReviewDialogueSessionRequest) -> dict[str, Any]:
        try:
            return review_run_service.create_dialogue_run(request.topics)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/review/card/{card_id}/prompt")
    def review_card_prompt(card_id: str, request: ReviewPromptRequest) -> dict[str, Any]:
        profile = interview_profile_store.load()
        weak = find_weak_point(profile, card_id)
        if weak is None:
            raise HTTPException(status_code=404, detail=f"review card not found: {card_id}")
        prompt = build_review_prompt_for_card(weak, request)
        return {"card_id": card_id, **prompt}

    @app.post("/api/review/card/{card_id}/grade")
    def review_card_grade(card_id: str, request: ReviewGradeRequest) -> dict[str, Any]:
        if runtime_config.vault_path is None:
            raise HTTPException(status_code=400, detail="vault path is required for review grading")
        profile = interview_profile_store.load()
        weak = find_weak_point(profile, card_id)
        if weak is None:
            raise HTTPException(status_code=404, detail=f"review card not found: {card_id}")
        if not request.answer.strip():
            raise HTTPException(status_code=400, detail="answer is required")
        source_paths = tuple(str(path).strip() for path in weak.get("source_note_paths") or [] if str(path).strip())
        if not source_paths:
            return {
                "card_id": card_id,
                "covered": [],
                "missing": [],
                "corrections": ["这个薄弱点没有记录来源笔记，无法对照资料纠偏。"],
                "feedback": "缺少来源笔记，建议先在面试总结中补齐 source_note_paths。",
                "suggested_outcome": "fail",
                "parse_error": True,
                "citations": [],
            }
        try:
            llm_config = load_llm_config(project_root)
            llm_client = create_llm_client(llm_config)
            app_runner = LibrarianApp(build_interview_agent_runtime(llm_client))
            result = app_runner.run(
                LibrarianRequest(
                    query=build_correction_query(prompt=request.prompt, weak=weak, answer=request.answer),
                    scope_type="selected_notes",
                    scope_note_paths=source_paths,
                    selected_note_paths=source_paths,
                    vault_root=runtime_config.vault_path,
                    strict_evidence=True,
                    model=llm_config.model,
                    tool_mode="auto",
                    trace_path=str(project_root / "eval-results" / "agent-debug" / "traces"),
                    temperature=min(llm_config.temperature, 0.1),
                    max_tool_calls_per_step=3,
                )
            )
            citations = result.state.working.extra.get("citations", [])
            return {"card_id": card_id, "trace_path": result.trace_path, **parse_correction_payload(result.final_answer or "", citations=citations)}
        except Exception as exc:
            return {
                "card_id": card_id,
                "covered": [],
                "missing": [],
                "corrections": [f"纠偏失败：{exc}"],
                "feedback": "本轮纠偏失败，请稍后重试；当前建议按 fail 处理。",
                "suggested_outcome": "fail",
                "parse_error": True,
                "citations": [],
                "error": str(exc),
            }

    @app.post("/api/review/verify")
    def review_verify(request: ReviewVerifyRequest) -> dict[str, Any]:
        if runtime_config.vault_path is None:
            raise HTTPException(status_code=400, detail="vault path is required for review verification")
        card_id = request.card_id.strip() or request.weak_point_id.strip()
        weak_point_ids = [str(item).strip() for item in request.weak_point_ids if str(item).strip()]
        if not weak_point_ids and request.weak_point_id.strip():
            weak_point_ids = [request.weak_point_id.strip()]
        if not weak_point_ids:
            raise HTTPException(status_code=400, detail="weak_point_ids is required")
        if not request.answer.strip():
            raise HTTPException(status_code=400, detail="answer is required")
        profile = interview_profile_store.load()
        weak_points = [find_weak_point(profile, weak_id) for weak_id in weak_point_ids]
        weak_points = [weak for weak in weak_points if isinstance(weak, dict)]
        if not weak_points:
            raise HTTPException(status_code=404, detail=f"review weak point not found: {weak_point_ids[0]}")
        source_paths = tuple(
            sorted(
                {
                    str(path).strip()
                    for weak in weak_points
                    for path in weak.get("source_note_paths") or []
                    if str(path).strip()
                }
            )
        )
        if not source_paths:
            return {
                "weak_point_id": card_id,
                "correct": [],
                "missed": ["这个薄弱点没有记录来源笔记，无法对照资料纠偏。"],
                "example": "建议先完成一次带来源记录的面试复盘，再回到这里复习。",
                "feedback": "缺少来源笔记。",
                "suggested_action": "retry",
                "card_id": card_id,
                "weak_point_ids": weak_point_ids,
                "weak_results": [
                    {
                        "weak_point_id": weak_id,
                        "point": str((find_weak_point(profile, weak_id) or {}).get("point") or "").strip(),
                        "suggested_action": "retry",
                        "reason": "missing source_note_paths",
                    }
                    for weak_id in weak_point_ids
                ],
                "parse_error": True,
                "citations": [],
            }
        try:
            llm_config = load_llm_config(project_root)
            llm_client = create_llm_client(llm_config)
            app_runner = LibrarianApp(build_interview_agent_runtime(llm_client))
            strategy_constraints = matching_strategy_constraints_for_card(
                {
                    "topic": str(weak_points[0].get("topic") or "").strip(),
                    "planned_layer": str(weak_points[0].get("planned_layer") or "").strip(),
                },
                select_strategy_constraints(profile, max_items=None),
                max_items=2,
            )
            result = app_runner.run(
                LibrarianRequest(
                    query=build_grouped_weak_point_verification_query(
                        weak_points=weak_points,
                        answer=request.answer,
                        prompt=request.prompt,
                        question_blocks=request.question_blocks,
                        strategy_constraints=strategy_constraints,
                    ),
                    scope_type="selected_notes",
                    scope_note_paths=source_paths,
                    selected_note_paths=source_paths,
                    vault_root=runtime_config.vault_path,
                    strict_evidence=True,
                    model=llm_config.model,
                    tool_mode="auto",
                    trace_path=str(project_root / "eval-results" / "agent-debug" / "traces"),
                    temperature=min(llm_config.temperature, 0.1),
                    max_tool_calls_per_step=3,
                )
            )
            citations = result.state.working.extra.get("citations", [])
            parsed = parse_grouped_verification_payload(result.final_answer or "", weak_points=weak_points, citations=citations)
            return {
                "card_id": card_id,
                "weak_point_ids": weak_point_ids,
                "weak_point_id": card_id,
                "trace_path": result.trace_path,
                **parsed,
            }
        except Exception as exc:
            return {
                "weak_point_id": card_id,
                "correct": [],
                "missed": [],
                "example": f"纠偏失败：{exc}",
                "feedback": "本轮纠偏失败，请稍后重试；当前建议继续练习。",
                "suggested_action": "retry",
                "parse_error": True,
                "citations": [],
                "error": str(exc),
            }

    @app.post("/api/review/card/{card_id}/commit")
    def review_card_commit(card_id: str, request: ReviewCommitRequest) -> dict[str, Any]:
        try:
            return commit_review_outcome(interview_profile_store, card_id=card_id, outcome=request.outcome)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/review/commit")
    def review_commit(request: ReviewActionCommitRequest) -> dict[str, Any]:
        card_id = request.weak_point_id.strip()
        if not card_id:
            raise HTTPException(status_code=400, detail="weak_point_id is required")
        try:
            return commit_review_action(interview_profile_store, card_id=card_id, action=request.action)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/review/dialogue")
    def review_dialogue(request: ReviewDialogueRequest) -> dict[str, Any]:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="message is required")
        try:
            llm_config = load_llm_config(project_root)
            llm_client = create_llm_client(llm_config)
            runtime = build_interview_agent_runtime(llm_client)
            selected_topics = [str(topic).strip() for topic in request.topics if str(topic).strip()]
            legacy_topic = request.topic.strip()
            if not selected_topics and legacy_topic:
                selected_topics = [legacy_topic]
            working = WorkingMemory(current_topic=selected_topics[0] if len(selected_topics) == 1 else None)
            working.extra["review_topics"] = selected_topics
            state = AgentState(messages=[], working=working, skill_name="reviewer")
            for item in request.chat_history[-12:]:
                role = str(item.get("role") or "").strip()
                if role in {"user", "assistant"}:
                    state.messages.append(AgentMessage(role=role, content=str(item.get("content") or "")))
            topic_line = f"Topics: {', '.join(selected_topics)}\n" if selected_topics else ""
            result = runtime.run(
                config=AgentRunConfig(
                    skill_name="reviewer",
                    max_steps=8,
                    max_tool_calls_per_step=4,
                    temperature=min(llm_config.temperature, 0.2),
                    model=llm_config.model,
                    trace_path=str(project_root / "eval-results" / "agent-debug" / "traces"),
                    allowed_tools=["get_due_reviews", "verify_weak_point", "suggest_review_commit"],
                    tool_mode="auto",
                ),
                user_input=f"{topic_line}{request.message.strip()}",
                state=state,
                tool_context=ToolExecutionContext(
                    working=working,
                    profile_store=interview_profile_store,
                    turn_context={"review_topics": selected_topics},
                ),
            )
            return {
                "answer": result.final_answer,
                "trace_path": result.trace_path,
                "stopped_reason": result.stopped_reason,
                "error": result.error,
                "error_type": result.error_type,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
        register_review_tools(registry)
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
        value = os.getenv("AGENT_V2_LIBRARIAN", "1").strip().lower()
        return value not in {"0", "false", "no", "off"}

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

    session_repo = FileSessionRepository(interview_session_store)
    answer_session_repo = FileAnswerSessionRepository(answer_session_store)
    run_repo = InMemoryAgentRunRepository(task_manager)

    def build_agent_turn_runner(llm_client) -> AgentTurnRunner:
        llm_config = load_llm_config(project_root)
        return AgentTurnRunner(
            AgentTurnRunnerDeps(
                vault_path=runtime_config.vault_path,
                wiki_state_path=runtime_config.wiki_state_path,
                wiki_dir=runtime_config.wiki_dir,
                project_root=project_root,
                overview_note_threshold=runtime_config.overview_note_threshold,
                interview_session_store=interview_session_store,
                interview_profile_store=interview_profile_store,
                llm_model=llm_config.model,
                llm_temperature=llm_config.temperature,
                build_agent_runtime=build_interview_agent_runtime,
                build_interview_rag_manager=build_interview_rag_manager,
                build_librarian_rag_manager=build_librarian_rag_manager,
                resolve_librarian_scope=resolve_librarian_scope,
                librarian_online_enabled=librarian_online_enabled,
                prewarm_interview_rag=prewarm_interview_rag,
                load_session_state=lambda session_id: session_repo.load_session(session_id),
            ),
            llm_client=llm_client,
        )

    agent_turn_service = AgentTurnService(
        session_repo=session_repo,
        answer_session_repo=answer_session_repo,
        run_repo=run_repo,
        runner_factory=build_agent_turn_runner,
        classify_error=classify_runtime_error,
    )

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

    @app.get("/api/tasks/{task_id}/stream")
    def task_stream(task_id: str) -> StreamingResponse:
        if task_manager.get(task_id) is None:
            raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
        return StreamingResponse(stream_task_events(task_manager, task_id), media_type="text/event-stream")

    @app.post("/api/agent/runs")
    def agent_runs(request: AgentRequest) -> dict[str, Any]:
        if runtime_config.vault_path is None:
            raise HTTPException(status_code=400, detail="vault path is required for chat agent")
        if request.chat_mode == "interview" and not agent_v2_interview_enabled():
            raise HTTPException(status_code=400, detail="interview agent v2 is required for /api/agent/runs")
        if request.chat_mode == "answer" and not agent_v2_librarian_enabled():
            raise HTTPException(status_code=400, detail="librarian agent v2 is required for /api/agent/runs")
        if request.chat_mode not in {"interview", "answer"}:
            raise HTTPException(status_code=400, detail="only interview and answer modes are supported for /api/agent/runs")
        llm_config = load_llm_config(project_root)
        llm_client = create_llm_client(llm_config)
        turn_input = AgentTurnInput.from_mapping(request.model_dump())
        try:
            return agent_turn_service.start_run(turn_input, llm_client=llm_client)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
                citations=request.citations,
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
                citations=request.citations,
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

    @app.post("/api/answer/sessions")
    def create_answer_session(request: AnswerSessionCreateRequest) -> dict[str, Any]:
        try:
            session = answer_session_store.create_session(
                scope_type=request.scope_type,
                scope_value=request.scope_value,
                scope_paths=request.scope_paths,
                strict_evidence=request.strict_evidence,
                extra=request.extra,
            )
            return {"session": session}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/answer/sessions")
    def list_answer_sessions(limit: int = 50) -> dict[str, Any]:
        try:
            return {"sessions": answer_session_store.list_sessions(limit=limit)}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/answer/sessions/{session_id}")
    def get_answer_session(session_id: str) -> dict[str, Any]:
        try:
            return answer_session_store.load_session_bundle(session_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/answer/sessions/{session_id}/archive")
    def archive_answer_session(session_id: str) -> dict[str, Any]:
        try:
            session = answer_session_store.archive_session(session_id)
            return {"session": session}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
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

                scope = None
                scope_note_paths: tuple[str, ...] = ()
                scope_metadata: dict[str, Any] = {}
                if str(request.scope_type or "all_vault") != "all_vault":
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
                            "stats": {"context_items": len(scope_note_paths), "agent_v2": False},
                        },
                    )

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
                retrieval = pipeline.retrieve(
                    request.query,
                    scope_note_paths=scope_note_paths,
                    scope_type=str(request.scope_type or "all_vault"),
                )
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
    from services.tasks.stream import sse_event as _sse_event

    return _sse_event(event, payload)


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


SHELL_CSS = r"""
    .app-shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 236px minmax(0, 1fr);
      background: var(--bg, #f7f7f4);
      color: var(--text, #171717);
    }
    .app-sidebar {
      position: sticky;
      top: 0;
      height: 100vh;
      overflow-y: auto;
      border-right: 1px solid var(--line, #d9d9d2);
      background: rgba(255, 255, 255, 0.92);
      padding: 18px 14px;
      z-index: 30;
    }
    .app-brand {
      display: grid;
      gap: 3px;
      margin: 0 0 20px;
      padding: 0 7px;
    }
    .app-brand-title {
      font-size: 17px;
      font-weight: 850;
      color: var(--text, #171717);
      letter-spacing: -0.01em;
    }
    .app-brand-subtitle {
      font-size: 12px;
      color: var(--muted, #666666);
      line-height: 1.4;
    }
    .app-nav {
      display: grid;
      gap: 18px;
    }
    .app-nav-section {
      display: grid;
      gap: 6px;
    }
    .app-nav-title {
      padding: 0 7px;
      color: var(--muted, #666666);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.02em;
    }
    .app-nav-link {
      display: flex;
      align-items: center;
      gap: 9px;
      min-height: 34px;
      padding: 7px 9px;
      border-radius: 9px;
      color: #34423f;
      text-decoration: none;
      font-size: 14px;
      font-weight: 700;
      border: 1px solid transparent;
    }
    .app-nav-link:hover {
      background: rgba(15, 118, 110, 0.07);
      color: var(--accent-dark, #115e59);
    }
    .app-nav-link.active {
      background: #e7f3f1;
      border-color: rgba(15, 118, 110, 0.18);
      color: var(--accent-dark, #115e59);
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }
    .app-nav-icon {
      width: 20px;
      text-align: center;
      font-size: 15px;
    }
    .app-main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    .app-header {
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      min-height: 68px;
      padding: 14px 24px;
      border-bottom: 1px solid var(--line, #d9d9d2);
      background: rgba(247, 247, 244, 0.86);
      backdrop-filter: blur(12px);
    }
    .app-header-left {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }
    .app-title {
      margin: 0;
      font-size: 20px;
      line-height: 1.2;
      letter-spacing: -0.01em;
    }
    .app-sidebar-toggle {
      display: none;
      border: 1px solid var(--line, #d9d9d2);
      background: #fff;
      color: var(--accent-dark, #115e59);
      border-radius: 8px;
      padding: 7px 10px;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }
    .app-global-search {
      width: min(420px, 42vw);
      display: flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line, #d9d9d2);
      border-radius: 999px;
      background: #fff;
      padding: 7px 12px;
    }
    .app-global-search span {
      color: var(--muted, #666666);
      font-size: 13px;
    }
    .app-global-search input {
      width: 100%;
      min-width: 0;
      border: 0;
      outline: 0;
      background: transparent;
      padding: 2px 0;
      color: var(--text, #171717);
      font: inherit;
      font-size: 14px;
    }
    .app-content {
      min-width: 0;
    }
    .app-sidebar-overlay {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.26);
      z-index: 25;
    }
    @media (max-width: 768px) {
      .app-shell {
        display: block;
      }
      .app-sidebar {
        position: fixed;
        inset: 0 auto 0 0;
        width: min(286px, 86vw);
        transform: translateX(-100%);
        transition: transform 0.18s ease;
        box-shadow: 16px 0 36px rgba(15, 23, 42, 0.16);
      }
      .app-shell.nav-open .app-sidebar {
        transform: translateX(0);
      }
      .app-shell.nav-open .app-sidebar-overlay {
        display: block;
      }
      .app-sidebar-toggle {
        display: inline-flex;
      }
      .app-header {
        padding: 12px 14px;
        align-items: flex-start;
        flex-wrap: wrap;
      }
      .app-global-search {
        width: 100%;
        order: 2;
      }
    }
"""


SIDEBAR_HTML = r"""
    <aside class="app-sidebar" aria-label="主导航">
      <div class="app-brand">
        <div class="app-brand-title">Knowledge Agent</div>
        <div class="app-brand-subtitle">个人知识库助手</div>
      </div>
      <nav class="app-nav">
        <section class="app-nav-section">
          <div class="app-nav-title">对话</div>
          <a class="app-nav-link" href="/?mode=answer" data-nav-id="chat-answer"><span class="app-nav-icon">💬</span><span>对话</span></a>
          <a class="app-nav-link" href="/?mode=interview" data-nav-id="chat-interview"><span class="app-nav-icon">🎙</span><span>面试</span></a>
          <a class="app-nav-link" href="/review" data-nav-id="review"><span class="app-nav-icon">🧠</span><span>复习</span></a>
        </section>
        <section class="app-nav-section">
          <div class="app-nav-title">知识</div>
          <a class="app-nav-link" href="/topics" data-nav-id="topics"><span class="app-nav-icon">📚</span><span>Topics</span></a>
          <a class="app-nav-link" href="/wiki" data-nav-id="wiki"><span class="app-nav-icon">📖</span><span>Wiki</span></a>
        </section>
        <section class="app-nav-section">
          <div class="app-nav-title">检索</div>
          <a class="app-nav-link" href="/search" data-nav-id="search"><span class="app-nav-icon">🔍</span><span>笔记搜索</span></a>
        </section>
        <section class="app-nav-section">
          <div class="app-nav-title">维护</div>
          <a class="app-nav-link" href="/admin/wiki" data-nav-id="wiki-admin"><span class="app-nav-icon">🛠</span><span>Wiki Admin</span></a>
          <a class="app-nav-link" href="/search?debug=true" data-nav-id="search-debug"><span class="app-nav-icon">⚙</span><span>Search Debug</span></a>
          <a class="app-nav-link" href="/settings" data-nav-id="settings"><span class="app-nav-icon">⚙</span><span>设置</span></a>
          <a class="app-nav-link" href="/organize" data-nav-id="organize"><span class="app-nav-icon">🧹</span><span>笔记整理</span></a>
        </section>
      </nav>
    </aside>
"""


WORKSPACE_STORE_SCRIPT = r"""
    (function () {
      const REVIEW_KEY = "knowledge_agent.workspace.review.v1";
      const CHAT_KEY = "knowledge_agent.workspace.chat.v1";
      const UI_KEY = "knowledge_agent.workspace.ui.v1";
      const LEGACY_KEY = "knowledge_agent.workspace.v1";
      const LEGACY_INTERVIEW_KEY = "knowledge_agent.active_interview_session_id";
      const LEGACY_ANSWER_WORKSPACE_KEY = "knowledge_agent.answer_workspace";
      const LEGACY_ANSWER_HISTORY_KEY = "knowledge_agent.answer_history";
      const LEGACY_ANSWER_TASK_KEY = "knowledge_agent.active_answer_task_id";
      const REVIEW_SLICE_VERSION = 1;
      const CHAT_SLICE_VERSION = 1;
      const slices = {review: null, chat: null, ui: null};

      function migrateReviewSlice(review) {
        if (!review || typeof review !== "object") return null;
        let version = Number(review.version) || 0;
        if (version >= REVIEW_SLICE_VERSION) return review;
        const next = {...review, version: REVIEW_SLICE_VERSION};
        if (version < 1) {
          if (!next.promptVersion && next.cardReviewState && next.cardReviewState.promptVersion) {
            next.promptVersion = next.cardReviewState.promptVersion;
          }
          if (next.cardReviewState && !Array.isArray(next.cardReviewState.cards)) {
            next.cardReviewState.cards = [];
          }
        }
        return next;
      }

      function migrateChatSlice(chat) {
        if (!chat || typeof chat !== "object") return null;
        const version = Number(chat.version) || 0;
        if (version >= CHAT_SLICE_VERSION) return chat;
        return {...chat, version: CHAT_SLICE_VERSION};
      }

      function migrateAll() {
        let changed = false;
        if (slices.review) {
          const migrated = migrateReviewSlice(slices.review);
          if (migrated !== slices.review) {
            slices.review = migrated;
            changed = true;
          }
        }
        if (slices.chat) {
          const migrated = migrateChatSlice(slices.chat);
          if (migrated !== slices.chat) {
            slices.chat = migrated;
            changed = true;
          }
        }
        if (changed) persist();
      }

      function hydrate() {
        try {
          const legacy = sessionStorage.getItem(LEGACY_KEY);
          if (legacy) {
            const parsed = JSON.parse(legacy);
            if (parsed && typeof parsed === "object") {
              if (parsed.review) slices.review = parsed.review;
              if (parsed.chat) slices.chat = parsed.chat;
              if (parsed.ui) slices.ui = parsed.ui;
            }
            sessionStorage.removeItem(LEGACY_KEY);
          }
          const reviewRaw = sessionStorage.getItem(REVIEW_KEY);
          if (reviewRaw) {
            const parsed = JSON.parse(reviewRaw);
            if (parsed && typeof parsed === "object") {
              slices.review = parsed.review || (parsed.version ? parsed : null);
            }
          }
          const chatRaw = localStorage.getItem(CHAT_KEY);
          if (chatRaw) {
            const parsed = JSON.parse(chatRaw);
            if (parsed && typeof parsed === "object") slices.chat = parsed;
          }
          const uiRaw = sessionStorage.getItem(UI_KEY);
          if (uiRaw) {
            const parsed = JSON.parse(uiRaw);
            if (parsed && typeof parsed === "object") slices.ui = parsed.ui || parsed;
          }
          migrateAll();
        } catch {}
      }

      function get(slice) {
        return slices[slice] ?? null;
      }

      function patch(slice, partial) {
        slices[slice] = {...(slices[slice] || {}), ...(partial || {})};
      }

      function replace(slice, value) {
        slices[slice] = value;
      }

      function clear(slice) {
        slices[slice] = null;
      }

      function persist() {
        try {
          if (slices.review) {
            sessionStorage.setItem(REVIEW_KEY, JSON.stringify({version: 1, savedAt: new Date().toISOString(), review: slices.review}));
          } else {
            sessionStorage.removeItem(REVIEW_KEY);
          }
          if (slices.chat) {
            localStorage.setItem(CHAT_KEY, JSON.stringify({...slices.chat, version: 1, savedAt: new Date().toISOString()}));
          } else {
            localStorage.removeItem(CHAT_KEY);
          }
          if (slices.ui) {
            sessionStorage.setItem(UI_KEY, JSON.stringify({version: 1, savedAt: new Date().toISOString(), ui: slices.ui}));
          } else {
            sessionStorage.removeItem(UI_KEY);
          }
        } catch {}
      }

      function migrateLegacyChat() {
        try {
          const activeInterview = localStorage.getItem(LEGACY_INTERVIEW_KEY);
          const answerWorkspace = localStorage.getItem(LEGACY_ANSWER_WORKSPACE_KEY);
          const activeTask = sessionStorage.getItem(LEGACY_ANSWER_TASK_KEY);
          if (!activeInterview && !answerWorkspace && !activeTask) return;
          const chat = slices.chat || {version: 1};
          if (activeInterview && !chat.activeInterviewSessionId) {
            chat.activeInterviewSessionId = activeInterview;
          }
          if (answerWorkspace) {
            try {
              const parsed = JSON.parse(answerWorkspace);
              if (parsed && parsed.scope) {
                chat.scope = {
                  scopeType: parsed.scope.scopeType,
                  scopeValue: parsed.scope.scopeValue,
                  strictEvidence: parsed.scope.strictEvidence,
                };
              }
              if (parsed && parsed.pendingTurn) chat.pendingTurn = parsed.pendingTurn;
            } catch {}
          }
          if (activeTask) chat.activeAnswerTaskId = activeTask;
          slices.chat = migrateChatSlice(chat);
          persist();
          localStorage.removeItem(LEGACY_INTERVIEW_KEY);
          localStorage.removeItem(LEGACY_ANSWER_WORKSPACE_KEY);
          localStorage.removeItem(LEGACY_ANSWER_HISTORY_KEY);
          sessionStorage.removeItem(LEGACY_ANSWER_TASK_KEY);
        } catch {}
      }

      window.KnowledgeAgentWorkspace = {hydrate, get, patch, replace, clear, persist, migrateLegacyChat, migrate: migrateAll};
    })();
"""


SHELL_SCRIPT = r"""
    (function () {
      const shell = document.querySelector(".app-shell");
      const toggle = document.getElementById("appSidebarToggle");
      const overlay = document.getElementById("appSidebarOverlay");
      const globalSearch = document.getElementById("appGlobalSearch");

      function closeNav() {
        if (shell) shell.classList.remove("nav-open");
      }

      if (toggle && shell) {
        toggle.addEventListener("click", () => shell.classList.toggle("nav-open"));
      }
      if (overlay) overlay.addEventListener("click", closeNav);
      document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") closeNav();
      });

      if (globalSearch) {
        globalSearch.addEventListener("keydown", (event) => {
          if (event.key !== "Enter") return;
          const query = globalSearch.value.trim();
          if (!query) return;
          window.location.href = `/search?q=${encodeURIComponent(query)}`;
        });
      }

      const params = new URLSearchParams(window.location.search);
      const path = window.location.pathname;
      const debug = params.get("debug") === "true";
      const mode = params.get("mode") || "answer";
      let active = "";
      if (debug && path === "/search") active = "search-debug";
      else if (path === "/admin/wiki") active = "wiki-admin";
      else if (path === "/topics") active = "topics";
      else if (path === "/wiki") active = "wiki";
      else if (path === "/search") active = "search";
      else if (path === "/settings") active = "settings";
      else if (path === "/organize" || path === "/audit") active = "organize";
      else if (path === "/review") active = "review";
      else if (path === "/" || path === "/chat") active = mode === "interview" ? "chat-interview" : "chat-answer";

      if (active) {
        document.querySelectorAll(".app-nav-link").forEach((link) => {
          link.classList.toggle("active", link.dataset.navId === active);
        });
      }
    })();
"""


def extract_html_block(html: str, start: str, end: str) -> str:
    start_index = html.find(start)
    if start_index < 0:
        return ""
    start_index += len(start)
    end_index = html.find(end, start_index)
    if end_index < 0:
        return ""
    return html[start_index:end_index]


def strip_legacy_navigation(body_html: str) -> str:
    cleaned = body_html.strip()
    if cleaned.startswith("<header>"):
        header_end = cleaned.find("</header>")
        if header_end >= 0:
            cleaned = cleaned[header_end + len("</header>") :].lstrip()
    cleaned = re.sub(
        r"(<main[^>]*>)(.*?)(?:\s*<header>\s*<div class=\"topbar\">.*?</header>\s*)",
        r"\1\2",
        cleaned,
        count=1,
        flags=re.DOTALL,
    )
    cleaned = re.sub(
        r'(<main[^>]*>)(.*?)(?:\s*<div class="topbar">.*?</nav>\s*</div>\s*)',
        r"\1\2",
        cleaned,
        count=1,
        flags=re.DOTALL,
    )
    return cleaned


def render_web_page(raw_html: str, title: str) -> str:
    legacy_css = extract_html_block(raw_html, "<style>", "</style>")
    legacy_body = extract_html_block(raw_html, "<body>", "</body>")
    content = strip_legacy_navigation(legacy_body or raw_html)
    content = content.replace("__REVIEW_PROMPT_VERSION__", REVIEW_PROMPT_VERSION)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Knowledge Agent - {title}</title>
  <style>
{legacy_css}
{SHELL_CSS}
  </style>
  <script>
{WORKSPACE_STORE_SCRIPT}
  </script>
</head>
<body>
  <div class="app-shell">
{SIDEBAR_HTML}
    <div id="appSidebarOverlay" class="app-sidebar-overlay"></div>
    <section class="app-main">
      <header class="app-header">
        <div class="app-header-left">
          <button id="appSidebarToggle" class="app-sidebar-toggle" type="button" aria-label="打开导航">☰</button>
          <h1 class="app-title">{title}</h1>
        </div>
        <label class="app-global-search" aria-label="全局搜索">
          <span>搜索</span>
          <input id="appGlobalSearch" type="search" placeholder="搜索你的笔记..." autocomplete="off" />
        </label>
      </header>
      <div class="app-content">
{content}
      </div>
    </section>
  </div>
  <script>
{SHELL_SCRIPT}
  </script>
</body>
</html>"""


WIKI_READER_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Wiki</title>
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
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(1280px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 22px 0 40px;
    }
    .reader-shell {
      display: grid;
      grid-template-columns: minmax(260px, 340px) minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .sidebar {
      position: sticky;
      top: 16px;
      max-height: calc(100vh - 32px);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .sidebar-head {
      padding: 14px;
      border-bottom: 1px solid var(--line);
    }
    h2 {
      margin: 0 0 6px;
      font-size: 20px;
      line-height: 1.25;
    }
    .subtitle, .meta, .path {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    input[type="search"] {
      width: 100%;
      min-height: 38px;
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }
    .tag-list {
      overflow: auto;
      padding: 8px;
      display: grid;
      gap: 6px;
    }
    .tag-item {
      width: 100%;
      text-align: left;
      border: 1px solid transparent;
      border-radius: 7px;
      background: transparent;
      padding: 10px;
      cursor: pointer;
      font: inherit;
      color: var(--text);
    }
    .tag-item:hover {
      background: #f3f6f5;
      border-color: var(--line);
    }
    .tag-item.active {
      background: #e7f3f1;
      border-color: rgba(15, 118, 110, .32);
    }
    .tag-title {
      font-weight: 760;
      line-height: 1.35;
      word-break: break-word;
    }
    .tag-preview {
      margin-top: 5px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .badge-row {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-top: 7px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 2px 7px;
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
    .content {
      min-height: calc(100vh - 44px);
      overflow: hidden;
    }
    .content-head {
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: flex-start;
    }
    .content-title {
      margin: 0 0 6px;
      font-size: 24px;
      line-height: 1.2;
    }
    .actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
      flex-shrink: 0;
    }
    .action-link, button.secondary {
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--accent-dark);
      padding: 7px 10px;
      text-decoration: none;
      cursor: pointer;
      font: inherit;
      font-weight: 700;
      font-size: 13px;
      white-space: nowrap;
    }
    .wiki-body {
      padding: 20px 24px 32px;
      line-height: 1.75;
      overflow-wrap: anywhere;
    }
    .wiki-body h2, .wiki-body h3, .wiki-body h4, .wiki-body h5 {
      margin: 22px 0 8px;
      line-height: 1.35;
    }
    .wiki-body h2 { font-size: 22px; }
    .wiki-body h3 { font-size: 18px; }
    .wiki-body p { margin: 0 0 12px; }
    .wiki-body ul, .wiki-body ol { margin: 8px 0 14px 24px; padding: 0; }
    .wiki-body li { margin: 4px 0; }
    .wiki-body code {
      background: var(--code);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 1px 4px;
      font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
      font-size: .92em;
    }
    .wiki-body pre {
      background: var(--code);
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 12px;
      overflow-x: auto;
      line-height: 1.55;
    }
    .wiki-body pre code {
      border: 0;
      background: transparent;
      padding: 0;
    }
    .wiki-body table {
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 18px;
      font-size: 14px;
    }
    .wiki-body th, .wiki-body td {
      border: 1px solid var(--line);
      padding: 7px 9px;
      vertical-align: top;
    }
    .wiki-body th {
      background: #f3f6f5;
      text-align: left;
    }
    .wiki-body hr {
      border: 0;
      border-top: 1px solid var(--line);
      margin: 18px 0;
    }
    .empty, .status {
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 24px;
      color: var(--muted);
      text-align: center;
      background: rgba(255, 255, 255, 0.65);
    }
    .status.error {
      color: var(--danger);
      border-color: #f3b5ad;
      background: #fff7f5;
    }
    @media (max-width: 860px) {
      main { width: min(100vw - 20px, 1280px); padding-top: 12px; }
      .reader-shell { grid-template-columns: 1fr; }
      .sidebar { position: static; max-height: none; }
      .tag-list { max-height: 42vh; }
      .content-head { display: block; }
      .actions { justify-content: flex-start; margin-top: 12px; }
      .wiki-body { padding: 16px; }
    }
  </style>
</head>
<body>
  <main>
    <section class="reader-shell">
      <aside class="panel sidebar">
        <div class="sidebar-head">
          <div id="summary" class="subtitle">正在加载 Wiki...</div>
          <input id="tagSearch" type="search" placeholder="搜索 tag / 路径 / 摘要" autocomplete="off" />
        </div>
        <div id="tagList" class="tag-list"></div>
      </aside>
      <article class="panel content">
        <div class="content-head">
          <div>
            <h2 id="wikiTitle" class="content-title">选择一个 Wiki</h2>
            <div id="wikiMeta" class="meta">只读阅读器</div>
            <div id="wikiPath" class="path"></div>
          </div>
          <div id="wikiActions" class="actions"></div>
        </div>
        <div id="wikiBody" class="wiki-body">
          <div class="status">正在加载...</div>
        </div>
      </article>
    </section>
  </main>
  <script>
    let report = null;
    let rows = [];
    let activeTag = "";
    const $ = (id) => document.getElementById(id);

    $("tagSearch").addEventListener("input", renderTagList);
    window.addEventListener("popstate", () => {
      const tag = tagFromUrl();
      if (tag && tag !== activeTag) selectTag(tag, {push: false});
    });
    loadReport();

    async function loadReport() {
      renderBodyStatus("正在加载 Wiki...");
      try {
        const response = await fetch("/api/wiki/report");
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Wiki report 加载失败");
        report = data;
        rows = (data.tag_rows || [])
          .filter((row) => row.wiki_exists && row.eligible)
          .sort((a, b) => String(a.tag || "").localeCompare(String(b.tag || "")));
        renderSummary();
        renderTagList();
        if (!rows.length) {
          renderEmpty("暂无已生成 Wiki。可以先到 Wiki Admin 维护页生成主题页。");
          return;
        }
        const requested = tagFromUrl();
        const initial = rows.find((row) => row.tag === requested) || rows[0];
        await selectTag(initial.tag, {push: false, missingTag: requested && !rows.find((row) => row.tag === requested) ? requested : ""});
      } catch (error) {
        renderBodyStatus(error.message || String(error), true);
        $("summary").textContent = "Wiki 加载失败";
      }
    }

    function renderSummary() {
      const dirty = rows.filter((row) => row.dirty).length;
      $("summary").textContent = `${rows.length} 个已生成 Wiki${dirty ? ` · ${dirty} 个需要更新` : ""}`;
    }

    function filteredRows() {
      const query = $("tagSearch").value.trim().toLowerCase();
      if (!query) return rows;
      return rows.filter((row) => {
        return String(row.tag || "").toLowerCase().includes(query)
          || String(row.wiki_path || "").toLowerCase().includes(query)
          || String(row.wiki_preview || "").toLowerCase().includes(query);
      });
    }

    function renderTagList() {
      const container = $("tagList");
      const visible = filteredRows();
      if (!visible.length) {
        container.innerHTML = `<div class="empty">没有匹配的 Wiki。</div>`;
        return;
      }
      container.innerHTML = visible.map((row) => {
        const active = row.tag === activeTag ? " active" : "";
        const preview = row.wiki_preview || row.wiki_path || "";
        return `
          <button class="tag-item${active}" data-tag="${escapeHtml(row.tag || "")}">
            <div class="tag-title">${escapeHtml(formatTagTitle(row.tag))}</div>
            <div class="tag-preview">${escapeHtml(preview)}</div>
            <div class="badge-row">
              <span class="badge">${escapeHtml(row.note_count || 0)} 篇笔记</span>
              ${row.dirty ? `<span class="badge update">需要更新</span>` : ""}
            </div>
          </button>
        `;
      }).join("");
      container.querySelectorAll("[data-tag]").forEach((button) => {
        button.addEventListener("click", () => selectTag(button.dataset.tag || ""));
      });
    }

    async function selectTag(tag, options = {}) {
      const row = rows.find((item) => item.tag === tag);
      if (!row) {
        renderBodyStatus(`未找到已生成 Wiki：${tag || options.missingTag || ""}`, true);
        return;
      }
      activeTag = row.tag || "";
      renderTagList();
      if (options.push !== false) {
        const url = new URL(window.location.href);
        url.searchParams.set("tag", activeTag);
        history.pushState({}, "", url);
      }
      renderHeader(row);
      if (options.missingTag) {
        renderBodyStatus(`未找到已生成 Wiki：${options.missingTag}。已打开第一个可读 Wiki。`, true);
      } else {
        renderBodyStatus("正在读取 Wiki...");
      }
      try {
        const response = await fetch(`/api/wiki/read?tag=${encodeURIComponent(activeTag)}`);
        const text = await response.text();
        if (!response.ok) {
          let message = text || "Wiki 读取失败";
          try { message = JSON.parse(text).detail || message; } catch (_error) {}
          throw new Error(message);
        }
        $("wikiBody").innerHTML = renderMarkdown(text || "");
      } catch (error) {
        renderBodyStatus(error.message || String(error), true);
      }
    }

    function renderHeader(row) {
      $("wikiTitle").textContent = formatTagTitle(row.tag);
      $("wikiMeta").textContent = `${row.note_count || 0} 篇来源笔记${row.dirty ? " · 源笔记已变更，建议稍后在维护页更新" : ""}`;
      $("wikiPath").textContent = row.wiki_path || "";
      const actions = [];
      if (row.wiki_path) {
        actions.push(`<a class="action-link" href="${obsidianUrl(row.wiki_path)}">在 Obsidian 中打开</a>`);
        actions.push(`<button class="secondary" type="button" id="copyWikiPath">复制路径</button>`);
      }
      $("wikiActions").innerHTML = actions.join("");
      const copyButton = $("copyWikiPath");
      if (copyButton) {
        copyButton.addEventListener("click", async () => {
          try {
            await navigator.clipboard.writeText(row.wiki_path || "");
            copyButton.textContent = "已复制";
            setTimeout(() => { copyButton.textContent = "复制路径"; }, 1200);
          } catch (_error) {
            copyButton.textContent = "复制失败";
            setTimeout(() => { copyButton.textContent = "复制路径"; }, 1200);
          }
        });
      }
    }

    function renderBodyStatus(message, isError = false) {
      $("wikiBody").innerHTML = `<div class="status ${isError ? "error" : ""}">${escapeHtml(message)}</div>`;
    }

    function renderEmpty(message) {
      $("wikiTitle").textContent = "暂无 Wiki";
      $("wikiMeta").textContent = "只读阅读器";
      $("wikiPath").textContent = "";
      $("wikiActions").innerHTML = "";
      $("wikiBody").innerHTML = `<div class="empty">${escapeHtml(message)}</div>`;
    }

    function tagFromUrl() {
      return new URLSearchParams(window.location.search).get("tag") || "";
    }

    function obsidianUrl(path) {
      const vault = report?.obsidian_vault_name || "";
      return `obsidian://open?vault=${encodeURIComponent(vault)}&file=${encodeURIComponent(path || "")}`;
    }

    function formatTagTitle(tag) {
      return String(tag || "").split("/").filter(Boolean).join(" / ") || "未命名主题";
    }

    function renderMarkdown(text) {
      const codeBlocks = [];
      let source = String(text || "").replace(/```([\s\S]*?)```/g, (_match, code) => {
        const token = `@@CODE_${codeBlocks.length}@@`;
        codeBlocks.push(`<pre><code>${escapeHtml(code.trim())}</code></pre>`);
        return token;
      });
      const lines = source.split(/\r?\n/);
      const html = [];
      let list = [];
      let orderedList = [];
      const flush = () => {
        if (list.length) { html.push(`<ul>${list.map((item) => `<li>${inlineMarkdown(item)}</li>`).join("")}</ul>`); list = []; }
        if (orderedList.length) { html.push(`<ol>${orderedList.map((item) => `<li>${inlineMarkdown(item)}</li>`).join("")}</ol>`); orderedList = []; }
      };
      for (let i = 0; i < lines.length; i += 1) {
        const raw = lines[i];
        const trimmed = raw.trim();
        if (!trimmed) { flush(); continue; }
        if (/^@@CODE_\d+@@$/.test(trimmed)) { flush(); html.push(trimmed); continue; }
        if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) { flush(); html.push("<hr>"); continue; }
        const heading = /^(#{1,4})\s+(.+)$/.exec(trimmed);
        if (heading) {
          flush();
          const level = Math.min(heading[1].length + 1, 5);
          html.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
          continue;
        }
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
      return rendered || `<div class="empty">这个 Wiki 文件暂无内容。</div>`;
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
      return escapeHtml(text)
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\[\[([^\]]+)\]\]/g, '<code>[[$1]]</code>');
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
  <main>
    <section class="page-intro">
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
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      color: var(--muted);
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .notice.visible { display: flex; }
    .stat-pill {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 9px;
      background: var(--chip);
      color: #3f3f3f;
      font-size: 13px;
      font-weight: 700;
      white-space: nowrap;
    }
    .stat-pill.update {
      background: #fff1bf;
      color: #704c00;
    }
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
  <main>
    <section class="hero">
      <p>这里只展示知识主题的状态和阅读入口。合成、同步和策略调整放在 Wiki Admin。</p>
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
        renderOverview();
        renderTopics();
        setStatus("");
      } catch (error) {
        setStatus(error.message || String(error), true);
      }
    }

    function renderOverview() {
      const rows = topicRows();
      const total = rows.length;
      const generated = rows.filter((row) => row.wiki_exists).length;
      const dirty = rows.filter((row) => row.dirty && row.wiki_exists).length;
      const missing = rows.filter((row) => row.eligible && !row.wiki_exists).length;
      const notice = $("notice");
      notice.innerHTML = `
        <span class="stat-pill">${total} 个知识主题</span>
        <span class="stat-pill">${generated} 个已生成</span>
        <span class="stat-pill update">${dirty} 个需要更新</span>
        <span class="stat-pill">${missing} 个未合成</span>
      `;
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
          || String(row.wiki_path || "").toLowerCase().includes(query)
          || String(row.wiki_preview || "").toLowerCase().includes(query);
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
        : "尚未生成 Wiki 文件";
      const openHref = row.wiki_exists ? obsidianUrl(row.wiki_path) : "";
      const readHref = row.wiki_exists ? `/wiki?tag=${encodeURIComponent(row.tag || "")}` : "";
      const statusBadge = row.wiki_exists
        ? (row.dirty ? `<span class="badge update">需要更新</span>` : `<span class="badge generated">已生成</span>`)
        : `<span class="badge missing">未合成</span>`;
      const actions = row.wiki_exists
        ? `
            <a class="open-link" href="${readHref}">在 Wiki 中阅读</a>
            <a class="open-link secondary" href="${openHref}">在 Obsidian 中打开</a>
          `
        : "";
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
          <div class="actions">${actions}</div>
        </article>
      `;
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


REVIEW_OLD_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>复习</title>
  <style>
    :root { --bg:#f7f7f4; --panel:#fff; --line:#d9d9d2; --text:#171717; --muted:#666; --accent:#0f766e; --accent-dark:#115e59; --danger:#b42318; --chip:#eef2f1; --ok:#0f766e; --warn:#9a5b00; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font-family:ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { width:min(900px, calc(100vw - 32px)); margin:0 auto; padding:24px 0 48px; display:grid; gap:14px; }
    .panel { border:1px solid var(--line); border-radius:10px; background:var(--panel); padding:18px; }
    h2 { margin:0 0 8px; font-size:25px; }
    h3 { margin:0 0 10px; font-size:18px; }
    p { margin:0 0 12px; color:var(--muted); line-height:1.65; }
    textarea { width:100%; min-height:150px; resize:vertical; border:1px solid var(--line); border-radius:8px; padding:11px 12px; line-height:1.65; font:inherit; color:var(--text); background:#fff; }
    button { border:1px solid var(--accent); border-radius:7px; background:var(--accent); color:#fff; padding:8px 12px; cursor:pointer; font-weight:800; font:inherit; }
    button.secondary { background:#fff; color:var(--accent-dark); border-color:var(--line); }
    button.danger { background:#fff; color:var(--danger); border-color:#f3b5ad; }
    button:disabled { opacity:.55; cursor:not-allowed; }
    .status { color:var(--muted); font-size:13px; min-height:20px; }
    .status.error { color:var(--danger); }
    .overview { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
    .topic-chip, .badge { display:inline-flex; align-items:center; border-radius:999px; padding:4px 9px; background:var(--chip); color:#3f3f3f; font-size:12px; font-weight:750; }
    .card-head { display:flex; justify-content:space-between; gap:12px; align-items:flex-start; margin-bottom:14px; }
    .nav-row { display:flex; justify-content:space-between; gap:8px; align-items:center; margin-bottom:12px; }
    .weak-point { border-left:4px solid var(--accent); padding:12px 14px; background:#f5fbfa; border-radius:8px; line-height:1.7; font-size:18px; font-weight:760; }
    .strategy-tip { margin-top:8px; color:var(--muted); font-size:13px; line-height:1.55; }
    .evidence { margin-top:10px; color:#38423f; line-height:1.65; }
    .meta { margin-top:8px; color:var(--muted); font-size:13px; overflow-wrap:anywhere; }
    .actions { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
    .result { border-top:1px solid var(--line); margin-top:16px; padding-top:16px; display:grid; gap:12px; }
    .result.compact { gap:8px; line-height:1.65; }
    .result.compact p { margin:0; color:var(--text); }
    .result.compact .muted { color:var(--muted); }
    .result-line { border:1px solid var(--line); border-radius:8px; padding:12px; background:#fff; line-height:1.65; }
    .result-line.ok { border-color:rgba(15,118,110,.28); background:#f3fbf8; }
    .result-line.warn { border-color:#ead18a; background:#fffaf0; }
    .result-line h3 { font-size:15px; margin-bottom:6px; }
    .result-line ul { margin:6px 0 0 20px; padding:0; }
    .citations { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
    .citation { border:1px solid var(--line); border-radius:999px; padding:3px 8px; font-size:12px; background:#f8fbfa; color:#334155; }
    .hidden { display:none !important; }
    @media (max-width:760px) { .card-head, .nav-row { display:block; } .actions { margin-top:10px; } }
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h2>逐条对照</h2>
      <p>系统会先按建议复查弱项准备本轮题目；你逐条作答，再对照笔记纠偏。</p>
      <div class="actions">
        <button id="cardMode" type="button">逐条对照</button>
        <button id="dialogueMode" class="secondary" type="button">对话复查</button>
      </div>
      <div id="status" class="status">正在准备本轮复习题...</div>
      <div id="overview" class="overview"></div>
    </section>
    <section id="cardPanel" class="panel hidden"></section>
    <section id="donePanel" class="panel hidden"></section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const state = {cards: [], index: 0, verify: null, results: [], reviewRunId: "", runStatus: "", pollTimer: null};
    loadDue();

    async function loadDue() {
      stopPolling();
      setStatus("正在准备本轮复习题...");
      try {
        const data = await postJson("/api/review/plan", {limit: 12, max_strategy_constraints: 2});
        state.reviewRunId = data.review_run_id || "";
        state.runStatus = data.status || "";
        state.cards = data.cards || [];
        state.index = 0;
        state.verify = null;
        state.results = [];
        renderOverview(data);
        if (!state.cards.length) {
          hide($("cardPanel"));
          $("donePanel").classList.remove("hidden");
          const message = data.summary || "可以继续面试或稍后再来。复习不是面试的前置条件。";
          $("donePanel").innerHTML = `<h3>暂无可独立复习的知识弱项</h3><p>${escapeHtml(message)}</p>`;
          setStatus(data.summary || "暂无建议复查弱项。");
          return;
        }
        setStatus("正在生成复习题，第一题准备好后会自动显示。");
        selectFirstUsableCard();
        renderCard();
        startPolling();
      } catch (error) {
        setStatus(error.message || String(error), true);
      }
    }

    function startPolling() {
      stopPolling();
      if (!state.reviewRunId) return;
      state.pollTimer = window.setInterval(pollReviewRun, 1200);
      pollReviewRun();
    }

    function stopPolling() {
      if (state.pollTimer) window.clearInterval(state.pollTimer);
      state.pollTimer = null;
    }

    async function pollReviewRun() {
      if (!state.reviewRunId) return;
      try {
        const beforeCard = currentCard();
        const hadUsableCurrent = isCardUsable(beforeCard);
        const beforeCardId = beforeCard?.id || "";
        const data = await getJson(`/api/review/plan/${encodeURIComponent(state.reviewRunId)}`);
        state.runStatus = data.status || "";
        state.cards = mergeCards(state.cards, data.cards || []);
        renderOverview(data);
        if (!currentCard() || !isCardUsable(currentCard())) selectFirstUsableCard();
        const afterCard = currentCard();
        const shouldRender = !hadUsableCurrent || beforeCardId !== (afterCard?.id || "");
        if (shouldRender) renderCard();
        const ready = data.ready_count || 0;
        const pending = data.pending_count || 0;
        const failed = data.failed_count || 0;
        if (ready || failed) setStatus(`已准备 ${ready} 题，待生成 ${pending} 题${failed ? `，失败 ${failed} 题` : ""}。`);
        if ((data.status || "") === "done") stopPolling();
      } catch (error) {
        stopPolling();
        setStatus(error.message || String(error), true);
      }
    }

    function mergeCards(previous, next) {
      const byId = new Map((previous || []).map((card) => [card.id, card]));
      for (const card of next || []) byId.set(card.id, {...(byId.get(card.id) || {}), ...card});
      return (next || []).map((card) => byId.get(card.id) || card);
    }

    function renderOverview(data) {
      const topics = data.topics || [];
      $("overview").innerHTML = topics.length
        ? topics.slice(0, 6).map((item) => `<span class="topic-chip">${escapeHtml(item.topic)} ${escapeHtml(item.candidate_count || item.due)} 个可复习</span>`).join("")
        : "";
    }

    function renderCard() {
      const card = currentCard();
      if (!card) return renderDone(false);
      if (!isCardUsable(card)) {
        show($("cardPanel"));
        hide($("donePanel"));
        const ready = state.cards.filter(isCardUsable).length;
        const pending = state.cards.filter((item) => item.status === "pending").length;
        $("cardPanel").innerHTML = `
          <h3>正在准备复习题</h3>
          <p>已准备 ${ready} 题，待生成 ${pending} 题。第一题准备好后会自动显示。</p>
          <div class="actions">
            <button id="reloadDue" class="secondary" type="button">刷新建议复查弱项</button>
            <button id="endReview" class="secondary" type="button">结束本轮复习</button>
          </div>
        `;
        return;
      }
      show($("cardPanel"));
      hide($("donePanel"));
      const topicPosition = topicProgress(card);
      const promptPayload = card.review_prompt || null;
      const promptText = promptPayload?.prompt || "正在生成复习题...";
      const promptReady = Boolean(promptPayload?.prompt);
      $("cardPanel").innerHTML = `
        <div class="nav-row">
          <button class="secondary" type="button" data-nav="-1" aria-label="上一条">←</button>
          <span class="status">${escapeHtml(card.topic || "未分类")} · ${topicPosition}</span>
          <button class="secondary" type="button" data-nav="1" aria-label="下一条">→</button>
        </div>
        <div class="weak-point">${escapeHtml(promptText)}</div>
        ${promptPayload?.hint ? `<div class="meta">提示：${escapeHtml(promptPayload.hint)}</div>` : ""}
        <textarea id="reviewAnswer" style="margin-top:14px" placeholder="回答比如：我会先说明定义，再区分触发时机和工程边界。"></textarea>
        ${renderStrategyConstraints(card.strategy_constraints || [])}
        <div class="actions">
          <button id="submitVerify" type="button" ${promptReady ? "" : "disabled"}>提交</button>
          <button id="reloadDue" class="secondary" type="button">刷新建议复查弱项</button>
          <button id="endReview" class="secondary" type="button">结束本轮复习</button>
        </div>
        <div id="verifyResult" class="result hidden"></div>
      `;
    }

    function renderVerifyResult(result) {
      const node = $("verifyResult");
      node.classList.remove("hidden");
      node.className = "result compact";
      const correct = firstText(result.correct || [], "方向基本正确。");
      const missedValues = []
        .concat(Array.isArray(result.missed) ? result.missed : [])
        .concat(Array.isArray(result.strategy_feedback) ? result.strategy_feedback : [])
        .filter(Boolean);
      const missed = firstText(missedValues, "没有明显遗漏。");
      const example = result.example || result.feedback || result.overall || "暂无示例。";
      const card = currentCard();
      const fallbackWeakIds = card && Array.isArray(card.weak_point_ids) ? card.weak_point_ids : [];
      const weakResults = Array.isArray(result.weak_results) && result.weak_results.length
        ? result.weak_results
        : fallbackWeakIds.map((id) => ({weak_point_id: id, point: id, suggested_action: "retry", reason: result.feedback || result.overall || ""}));
      const weakResultsHtml = weakResults.length
        ? `<div class="weak-results">${weakResults.map((item) => {
            const action = item.suggested_action === "improve" ? "improve" : "retry";
            return `<div class="weak-result">
              <p><strong>${escapeHtml(action === "improve" ? "建议确认改善" : "建议继续练习")}</strong>：${escapeHtml(item.point || item.weak_point_id || "")}</p>
              ${item.reason ? `<p>${escapeHtml(item.reason)}</p>` : ""}
              <div class="actions">
                <button type="button" data-weak-id="${escapeHtml(item.weak_point_id || "")}" data-weak-action="improve">确认改善</button>
                <button type="button" class="danger" data-weak-id="${escapeHtml(item.weak_point_id || "")}" data-weak-action="retry">还要再练</button>
              </div>
            </div>`;
          }).join("")}</div>`
        : `<div class="actions">
          <button type="button" data-action="improve">确认改善</button>
          <button type="button" class="danger" data-action="retry">还要再练</button>
        </div>`;
      node.innerHTML = `
        <p>✓ ${escapeHtml(correct)}</p>
        <p>⚠ ${escapeHtml(missed)}</p>
        <p><strong>示例：</strong><br>${escapeHtml(example)}</p>
        ${renderCitations(result.citations || [])}
        ${weakResultsHtml}
      `;
    }
    document.addEventListener("click", async (event) => {
      if (event.target.dataset.nav) move(Number(event.target.dataset.nav));
      if (event.target.id === "submitVerify") await submitVerify();
      if (event.target.dataset.weakAction) await commitWeakAction(event.target.dataset.weakId, event.target.dataset.weakAction);
      if (event.target.dataset.action) await commitAction(event.target.dataset.action);
      if (event.target.id === "reloadDue") await loadDue();
      if (event.target.id === "newReview") await loadDue();
      if (event.target.id === "cardMode") await loadDue();
      if (event.target.id === "dialogueMode") renderDialogueReview();
      if (event.target.id === "sendDialogueReview") await sendDialogueReview();
      if (event.target.id === "endReview") {
        stopPolling();
        renderDone(true);
        setStatus("已结束本轮复习。");
      }
    });

    function move(delta) {
      if (!state.cards.length) return;
      let nextIndex = state.index;
      for (let offset = 0; offset < state.cards.length; offset += 1) {
        nextIndex = (nextIndex + delta + state.cards.length) % state.cards.length;
        if (isCardUsable(state.cards[nextIndex])) break;
      }
      state.index = nextIndex;
      state.verify = null;
      renderCard();
    }

    function isCardUsable(card) {
      return Boolean(card && card.review_prompt && card.review_prompt.prompt);
    }

    function selectFirstUsableCard() {
      if (isCardUsable(currentCard())) return true;
      const index = state.cards.findIndex(isCardUsable);
      if (index >= 0) {
        state.index = index;
        return true;
      }
      return false;
    }

    async function submitVerify() {
      const card = currentCard();
      const answer = $("reviewAnswer").value.trim();
      if (!answer) { setStatus("请先写下你的理解。", true); return; }
      setBusy(true, "正在对照笔记纠偏...");
      try {
        const prompt = card.review_prompt?.prompt || card.point || "";
        const weakPointIds = Array.isArray(card.weak_point_ids) && card.weak_point_ids.length ? card.weak_point_ids : [card.id];
        const result = await postJson("/api/review/verify", {
          card_id: card.card_id || card.id,
          weak_point_ids: weakPointIds,
          weak_point_id: weakPointIds[0] || card.id,
          answer,
          prompt,
          question_blocks: card.question_blocks || card.review_prompt?.question_blocks || [],
        });
        state.verify = result;
        renderVerifyResult(result);
        setStatus("纠偏完成。确认改善或继续练习后会进入下一条。");
      } catch (error) {
        setStatus(error.message || String(error), true);
      } finally {
        setBusy(false);
      }
    }

    async function commitAction(action) {
      const card = currentCard();
      setBusy(true, "正在保存复习结果...");
      try {
        const result = await postJson("/api/review/commit", {weak_point_id: card.id, action});
        state.results.push({card, action, result});
        state.cards.splice(state.index, 1);
        if (state.index >= state.cards.length) state.index = 0;
        if (!state.cards.length) renderDone(false);
        else renderCard();
        setStatus(action === "improve" ? "已标记改善。" : "已标记继续练习。");
      } catch (error) {
        setStatus(error.message || String(error), true);
      } finally {
        setBusy(false);
      }
    }

    async function commitWeakAction(weakPointId, action) {
      const card = currentCard();
      const weakId = String(weakPointId || "").trim();
      if (!weakId) return;
      setBusy(true, "正在保存该弱项的复习结果...");
      try {
        const result = await postJson("/api/review/commit", {weak_point_id: weakId, action});
        card.confirmed_weak_points = card.confirmed_weak_points || {};
        card.confirmed_weak_points[weakId] = action;
        state.results.push({card_id: card.id, weak_point_id: weakId, action, result});
        const expected = Array.isArray(card.weak_point_ids) && card.weak_point_ids.length ? card.weak_point_ids : [card.id];
        const allDone = expected.every((id) => card.confirmed_weak_points[id]);
        if (allDone) {
          state.cards.splice(state.index, 1);
          state.verify = null;
          if (state.index >= state.cards.length) state.index = 0;
          if (!state.cards.length) renderDone(false);
          else renderCard();
        } else if (state.verify) {
          renderVerifyResult(state.verify);
        }
        setStatus(action === "improve" ? "已确认该弱项改善。" : "已标记该弱项继续练习。");
      } catch (error) {
        setStatus(error.message || String(error), true);
      } finally {
        setBusy(false);
      }
    }

    function renderDialogueReview() {
      stopPolling();
      hide($("cardPanel"));
      show($("donePanel"));
      state.dialogueHistory = state.dialogueHistory || [];
      $("donePanel").innerHTML = `
        <h3>对话复查</h3>
        <p>Agent 会读取当前 topic 的建议复查弱项，逐条追问并给出建议；不会自动写入 profile。</p>
        <input id="dialogueTopic" style="width:100%;border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:10px" placeholder="可选 topic，例如 MCP 协议" />
        <div id="dialogueMessages" class="result compact"></div>
        <textarea id="dialogueMessage" placeholder="输入：开始对这个 topic 做对话复查，或回答 Agent 的追问。"></textarea>
        <div class="actions">
          <button id="sendDialogueReview" type="button">发送</button>
          <button id="cardMode" class="secondary" type="button">回到逐条对照</button>
        </div>
      `;
      setStatus("已切换到对话复查。");
    }

    async function sendDialogueReview() {
      const input = $("dialogueMessage");
      const topic = $("dialogueTopic")?.value.trim() || "";
      const message = input.value.trim();
      if (!message) { setStatus("请先输入对话内容。", true); return; }
      state.dialogueHistory = state.dialogueHistory || [];
      state.dialogueHistory.push({role: "user", content: message});
      input.value = "";
      renderDialogueMessages();
      setBusy(true, "正在进行对话复查...");
      try {
        const result = await postJson("/api/review/dialogue", {topic, message, chat_history: state.dialogueHistory});
        state.dialogueHistory.push({role: "assistant", content: result.answer || result.error || ""});
        renderDialogueMessages();
        setStatus("对话复查已更新。");
      } catch (error) {
        setStatus(error.message || String(error), true);
      } finally {
        setBusy(false);
      }
    }

    function renderDialogueMessages() {
      const node = $("dialogueMessages");
      if (!node) return;
      const history = state.dialogueHistory || [];
      node.innerHTML = history.map((item) => `<p><strong>${item.role === "user" ? "你" : "Agent"}：</strong>${escapeHtml(item.content || "")}</p>`).join("");
    }

    function renderDone(ended) {
      hide($("cardPanel"));
      show($("donePanel"));
      const improved = state.results.filter((item) => item.action === "improve").length;
      const retry = state.results.filter((item) => item.action === "retry").length;
      $("donePanel").innerHTML = `
        <h3>${ended ? "已结束复习" : "本轮复习完成"}</h3>
        <p>本轮处理 ${state.results.length} 条。确认改善 ${improved} 条，还要再练 ${retry} 条。</p>
        <p>未复习或还要再练的弱项仍会在后续面试中注入，复习只是加速器。</p>
        <div class="actions"><button id="newReview" type="button">重新加载复习</button></div>
      `;
    }

    function topicProgress(card) {
      const same = state.cards.filter((item) => item.topic === card.topic);
      const pos = same.findIndex((item) => item.id === card.id) + 1;
      return `${pos}/${same.length}`;
    }

    function currentCard() { return state.cards[state.index] || null; }
    async function loadCardPrompt(card) {
      if (!card || card.review_prompt || card.prompt_loading) return;
      card.prompt_loading = true;
      try {
        const prompt = await postJson(`/api/review/card/${encodeURIComponent(card.id)}/prompt`, {
          allowed_question_types: ["recall", "boundary", "compare", "scenario", "followup"],
          related_topics: [],
        });
        card.review_prompt = prompt;
      } catch (error) {
        card.review_prompt = {
          prompt: card.point || "请围绕这个知识弱点，用自己的话完整回答。",
          fallback_used: true,
          error: error.message || String(error),
        };
      } finally {
        card.prompt_loading = false;
        if (currentCard()?.id === card.id) renderCard();
      }
    }
    function renderStrategyConstraints(items) {
      const constraints = Array.isArray(items) ? items.filter((item) => item && item.point) : [];
      if (!constraints.length) return "";
      const tips = [];
      for (const item of constraints) {
        const tip = strategyTipLabel(item.point);
        if (tip && !tips.includes(tip)) tips.push(tip);
        if (tips.length >= 3) break;
      }
      if (!tips.length) return "";
      return `<div class="strategy-tip">💡 本次复习关注：${tips.map(escapeHtml).join(" · ")}</div>`;
    }
    function strategyTipLabel(point) {
      const text = String(point || "");
      if (!text.trim()) return "";
      if (text.includes("回答过短") || text.includes("推理展开") || text.includes("系统性工程思维")) return "展开推理过程";
      if (text.includes("简单归因") || text.includes("部署拓扑") || text.includes("部署场景")) return "按部署场景做判断";
      if (text.includes("边界") || text.includes("职责")) return "先区分边界";
      if (text.includes("选型") || text.includes("取舍")) return "说明工程取舍";
      if (text.includes("结构") || text.includes("框架")) return "先给结论再分层展开";
      return text.length > 18 ? `${text.slice(0, 18)}...` : text;
    }
    function firstText(items, emptyText) {
      const values = Array.isArray(items) ? items.filter(Boolean) : [];
      return values.length ? String(values[0]) : emptyText;
    }
    function renderList(items, emptyText) {
      const values = Array.isArray(items) ? items.filter(Boolean) : [];
      return values.length ? `<ul>${values.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : `<p>${escapeHtml(emptyText)}</p>`;
    }
    function renderCitations(citations) {
      if (!citations.length) return "";
      return `<div class="citations">${citations.slice(0, 8).map((item) => `<span class="citation">${escapeHtml(item.path || item.source_path || item.title || "source")}</span>`).join("")}</div>`;
    }
    async function getJson(url) {
      const response = await fetch(url);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || `${response.status} ${response.statusText}`);
      return data;
    }
    async function postJson(url, payload) {
      const response = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || `${response.status} ${response.statusText}`);
      return data;
    }
    function setBusy(busy, message="") { document.querySelectorAll("button").forEach((button) => button.disabled = busy); if (message) setStatus(message); }
    function setStatus(message, error=false) { $("status").textContent = message || ""; $("status").className = `status ${error ? "error" : ""}`; }
    function show(node) { node.classList.remove("hidden"); }
    function hide(node) { node.classList.add("hidden"); }
    function escapeHtml(value) { return String(value ?? "").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;"); }
  </script>
</body>
</html>
"""


REVIEW_LEGACY_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>复习</title>
  <style>
    :root { --bg:#f7f7f4; --panel:#fff; --line:#d9d9d2; --text:#171717; --muted:#666; --accent:#0f766e; --accent-dark:#115e59; --danger:#b42318; --chip:#eef2f1; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font-family:ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { width:min(980px, calc(100vw - 32px)); margin:0 auto; padding:24px 0 48px; display:grid; gap:14px; }
    .panel { border:1px solid var(--line); border-radius:10px; background:var(--panel); padding:18px; }
    details.panel { padding:0; }
    details.panel > summary { padding:16px 18px; cursor:pointer; font-weight:850; color:var(--accent-dark); }
    details.panel > .panel-body { border-top:1px solid var(--line); padding:16px 18px 18px; }
    h2 { margin:0 0 8px; font-size:25px; }
    h3 { margin:0 0 10px; font-size:18px; }
    p { margin:0 0 12px; color:var(--muted); line-height:1.65; }
    label { display:grid; gap:5px; color:var(--muted); font-size:13px; font-weight:750; }
    input, select, textarea, button { font:inherit; }
    input[type="number"], select, textarea { width:100%; border:1px solid var(--line); border-radius:7px; padding:8px 10px; background:#fff; color:var(--text); }
    textarea { min-height:130px; resize:vertical; line-height:1.6; }
    button { border:1px solid var(--accent); border-radius:7px; background:var(--accent); color:#fff; padding:8px 12px; cursor:pointer; font-weight:800; }
    button.secondary { background:#fff; color:var(--accent-dark); border-color:var(--line); }
    button.danger { background:#fff; color:var(--danger); border-color:#f3b5ad; }
    button:disabled { opacity:.55; cursor:not-allowed; }
    .setup-grid { display:grid; grid-template-columns: 1fr 220px 180px; gap:12px; align-items:end; }
    .chips { display:flex; flex-wrap:wrap; gap:7px; }
    .chip-check { display:inline-flex; align-items:center; gap:6px; border:1px solid var(--line); border-radius:999px; padding:6px 10px; background:#fff; color:#34423f; font-size:13px; cursor:pointer; }
    .chip-check input { margin:0; }
    .status { color:var(--muted); font-size:13px; min-height:20px; }
    .status.error { color:var(--danger); }
    .card-head { display:flex; justify-content:space-between; gap:12px; align-items:flex-start; margin-bottom:12px; }
    .badge { display:inline-flex; border-radius:999px; padding:3px 8px; background:var(--chip); color:#3f3f3f; font-size:12px; font-weight:750; }
    .question { border-left:4px solid var(--accent); padding:10px 12px; background:#f5fbfa; border-radius:7px; line-height:1.7; font-size:17px; }
    .hint { margin-top:8px; color:var(--muted); font-size:13px; }
    .actions { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
    .plan-list { display:grid; gap:6px; margin-top:8px; }
    .plan-item { display:flex; justify-content:space-between; gap:12px; border:1px solid var(--line); border-radius:7px; padding:8px 10px; color:#34423f; font-size:13px; }
    .plan-item.active { border-color:rgba(15,118,110,.35); background:#e7f3f1; }
    .feedback-grid { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:12px; }
    .feedback-box { border:1px solid var(--line); border-radius:8px; padding:12px; background:#fff; }
    .feedback-box ul { margin:6px 0 0 20px; padding:0; }
    .citations { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
    .citation { border:1px solid var(--line); border-radius:999px; padding:3px 8px; font-size:12px; background:#f8fbfa; color:#334155; }
    .hidden { display:none !important; }
    @media (max-width:760px) { .setup-grid, .feedback-grid { grid-template-columns:1fr; } .card-head { display:block; } }
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <p>选择要复习的主题和本轮题数，系统会逐题出题、纠偏，并在你确认 pass/fail 后自动进入下一题。</p>
      <div id="status" class="status">正在加载复习主题...</div>
    </section>

    <details id="setupPanel" class="panel" open>
      <summary>复习设置</summary>
      <div class="panel-body">
        <div class="setup-grid">
          <label>本轮题数<input id="limit" type="number" min="1" max="50" value="10" /></label>
          <label>允许交叉题<select id="allowCross"><option value="true" selected>允许</option><option value="false">不允许</option></select></label>
          <div class="actions" style="margin:0">
            <button id="startBtn" type="button">开始复习</button>
            <button id="newBtn" class="secondary" type="button">新建复习</button>
          </div>
        </div>
        <h3 style="margin-top:16px">主题</h3>
        <div id="topicFilters" class="chips"></div>
        <h3 style="margin-top:16px">允许题型</h3>
        <div id="typeFilters" class="chips"></div>
      </div>
    </details>

    <details id="planPanel" class="panel hidden">
      <summary>本轮计划</summary>
      <div class="panel-body"><div id="planList" class="plan-list"></div></div>
    </details>

    <section id="questionPanel" class="panel hidden"></section>
    <section id="reviewPanel" class="panel hidden"></section>
    <section id="donePanel" class="panel hidden"></section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const state = {cards: [], index: 0, prompt: null, grade: null, results: []};
    const questionTypeLabels = {recall:"主动回忆", boundary:"职责边界", compare:"概念对比", scenario:"场景设计", followup:"面试追问"};

    $("startBtn").addEventListener("click", startReview);
    $("newBtn").addEventListener("click", newReview);
    loadInitialPlan();

    async function loadInitialPlan() {
      try {
        const data = await postJson("/api/review/plan", {topics: [], question_types: [], limit: 10, allow_cross_topic: true});
        renderSetup(data);
        setStatus("选择主题后开始复习。未选择主题时会使用全部可复习薄弱点。");
      } catch (error) {
        setStatus(error.message || String(error), true);
      }
    }

    function renderSetup(data) {
      const topics = data.available_topics || [];
      $("topicFilters").innerHTML = topics.length ? topics.map((item) => `
        <label class="chip-check"><input type="checkbox" value="${escapeHtml(item.topic)}" />${escapeHtml(item.topic)} · ${item.due}/${item.total}</label>
      `).join("") : `<span class="status">暂无 profile weak point。先完成一次面试并结束 session 后再复习。</span>`;
      const types = data.question_types || [];
      $("typeFilters").innerHTML = types.map((item) => `
        <label class="chip-check"><input type="checkbox" value="${escapeHtml(item.value)}" checked />${escapeHtml(item.label)}</label>
      `).join("");
    }

    async function startReview() {
      setBusy(true, "正在生成复习计划...");
      try {
        const payload = {
          topics: checkedValues("topicFilters"),
          question_types: checkedValues("typeFilters"),
          limit: Number($("limit").value || 10),
          allow_cross_topic: $("allowCross").value === "true"
        };
        const plan = await postJson("/api/review/plan", payload);
        state.cards = plan.cards || [];
        state.index = 0;
        state.results = [];
        if (!state.cards.length) {
          setStatus("当前没有可复习卡片。可以换主题，或先完成一次面试生成 weak points。", true);
          $("setupPanel").open = true;
          return;
        }
        $("setupPanel").open = false;
        $("planPanel").classList.remove("hidden");
        renderPlan();
        await loadQuestion();
      } catch (error) {
        setStatus(error.message || String(error), true);
      } finally {
        setBusy(false);
      }
    }

    async function loadQuestion() {
      const card = currentCard();
      if (!card) return renderDone();
      hide($("reviewPanel"));
      hide($("donePanel"));
      show($("questionPanel"));
      $("questionPanel").innerHTML = renderQuestionShell(card, "正在出题...");
      renderPlan();
      setBusy(true, "正在出题...");
      try {
        state.prompt = await postJson(`/api/review/card/${encodeURIComponent(card.id)}/prompt`, {
          allowed_question_types: card.allowed_question_types || [],
          related_topics: card.candidate_related_topics || []
        });
        $("questionPanel").innerHTML = renderQuestion(card, state.prompt);
        $("answer").focus();
      } catch (error) {
        $("questionPanel").innerHTML = renderQuestionShell(card, error.message || String(error), true);
      } finally {
        setBusy(false);
      }
    }

    function renderQuestion(card, prompt) {
      const qtype = prompt.question_type || card.question_type || "auto";
      return `
        <div class="card-head">
          <div>
            <h3>第 ${state.index + 1} / ${state.cards.length} 题</h3>
            <div class="status">${escapeHtml(card.topic || "未分类")} · ${escapeHtml(questionTypeLabels[qtype] || qtype)}</div>
          </div>
          <span class="badge">${reviewStateLabel(card.review_state)}</span>
        </div>
        <div class="question">${escapeHtml(prompt.prompt || "")}</div>
        ${prompt.hint ? `<div class="hint">提示：${escapeHtml(prompt.hint)}</div>` : ""}
        ${prompt.reason ? `<div class="hint">题型选择：${escapeHtml(prompt.reason)}</div>` : ""}
        <label style="margin-top:14px">你的回答<textarea id="answer" placeholder="先凭记忆回答，不要看笔记。"></textarea></label>
        <div class="actions">
          <button id="submitAnswer" type="button">提交并纠偏</button>
          <button id="endBtn" class="secondary" type="button">结束复习</button>
        </div>
      `;
    }

    function renderQuestionShell(card, message, error=false) {
      return `
        <div class="card-head">
          <div><h3>第 ${state.index + 1} / ${state.cards.length} 题</h3><div class="status">${escapeHtml(card.topic || "未分类")}</div></div>
        </div>
        <div class="status ${error ? "error" : ""}">${escapeHtml(message)}</div>
      `;
    }

    document.addEventListener("click", async (event) => {
      if (event.target.id === "submitAnswer") await submitAnswer();
      if (event.target.id === "endBtn") renderDone(true);
      if (event.target.dataset.outcome) await commitOutcome(event.target.dataset.outcome);
    });

    async function submitAnswer() {
      const answer = $("answer").value.trim();
      if (!answer) { setStatus("请先填写回答。", true); return; }
      const card = currentCard();
      setBusy(true, "正在对照笔记纠偏...");
      try {
        state.grade = await postJson(`/api/review/card/${encodeURIComponent(card.id)}/grade`, {
          prompt: state.prompt?.prompt || "",
          answer
        });
        renderReview(card, state.grade);
      } catch (error) {
        setStatus(error.message || String(error), true);
      } finally {
        setBusy(false);
      }
    }

    function renderReview(card, grade) {
      hide($("questionPanel"));
      show($("reviewPanel"));
      const suggested = grade.suggested_outcome || "fail";
      $("reviewPanel").innerHTML = `
        <div class="card-head">
          <div><h3>纠偏结果</h3><div class="status">${escapeHtml(card.topic || "未分类")} · 建议 ${escapeHtml(suggested)}</div></div>
          <span class="badge">${escapeHtml(suggested)}</span>
        </div>
        <div class="feedback-grid">
          ${feedbackBox("已覆盖", grade.covered)}
          ${feedbackBox("遗漏点", grade.missing)}
          ${feedbackBox("纠偏", grade.corrections)}
          <div class="feedback-box"><h3>反馈</h3><p>${escapeHtml(grade.feedback || "无额外反馈。")}</p>${renderCitations(grade.citations || [])}</div>
        </div>
        <div class="actions">
          <button data-outcome="pass" type="button">Pass</button>
          <button data-outcome="fail" class="danger" type="button">Fail</button>
          <button id="endBtn" class="secondary" type="button">结束复习</button>
        </div>
      `;
    }

    function feedbackBox(title, items) {
      const values = Array.isArray(items) ? items : [];
      return `<div class="feedback-box"><h3>${escapeHtml(title)}</h3>${values.length ? `<ul>${values.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : "<p>无</p>"}</div>`;
    }

    function renderCitations(citations) {
      if (!citations.length) return "";
      return `<div class="citations">${citations.slice(0, 8).map((item) => `<span class="citation">${escapeHtml(item.path || item.source_path || item.title || "source")}</span>`).join("")}</div>`;
    }

    async function commitOutcome(outcome) {
      const card = currentCard();
      setBusy(true, "正在保存复习结果...");
      try {
        const result = await postJson(`/api/review/card/${encodeURIComponent(card.id)}/commit`, {outcome});
        state.results.push({card, outcome, result});
        state.index += 1;
        if (state.index >= state.cards.length) renderDone();
        else await loadQuestion();
      } catch (error) {
        setStatus(error.message || String(error), true);
      } finally {
        setBusy(false);
      }
    }

    function renderDone(ended=false) {
      hide($("questionPanel"));
      hide($("reviewPanel"));
      show($("donePanel"));
      const pass = state.results.filter((item) => item.outcome === "pass").length;
      const fail = state.results.filter((item) => item.outcome === "fail").length;
      $("donePanel").innerHTML = `
        <h3>${ended ? "已结束复习" : "本轮复习完成"}</h3>
        <p>已完成 ${state.results.length} / ${state.cards.length} 题。Pass ${pass}，Fail ${fail}。</p>
        <div class="actions"><button id="newBtnInline" type="button">新建复习</button></div>
      `;
      $("newBtnInline").addEventListener("click", newReview);
    }

    function newReview() {
      state.cards = []; state.index = 0; state.prompt = null; state.grade = null; state.results = [];
      document.querySelectorAll("#topicFilters input, #typeFilters input").forEach((node) => { node.checked = node.closest("#typeFilters") !== null; });
      $("allowCross").value = "true";
      $("setupPanel").open = true;
      hide($("planPanel")); hide($("questionPanel")); hide($("reviewPanel")); hide($("donePanel"));
      setStatus("重新选择主题后开始复习。");
    }

    function renderPlan() {
      $("planList").innerHTML = state.cards.map((card, index) => `
        <div class="plan-item ${index === state.index ? "active" : ""}">
          <span>${index + 1}. ${escapeHtml(card.topic || "未分类")}</span>
          <span>${escapeHtml(card.point || "")}</span>
        </div>
      `).join("");
    }

    function currentCard() { return state.cards[state.index] || null; }
    function checkedValues(id) { return Array.from(document.querySelectorAll(`#${id} input:checked`)).map((node) => node.value); }
    async function postJson(url, payload) {
      const response = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || `${response.status} ${response.statusText}`);
      return data;
    }
    function setStatus(message, error=false) { $("status").textContent = message || ""; $("status").className = `status ${error ? "error" : ""}`; }
    function setBusy(busy, message="") { $("startBtn").disabled = busy; if (message) setStatus(message); }
    function show(node) { node.classList.remove("hidden"); }
    function hide(node) { node.classList.add("hidden"); }
    function escapeHtml(value) { return String(value ?? "").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;"); }
  </script>
</body>
</html>
"""


REVIEW_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>复习</title>
  <style>
    :root { --bg:#f7f7f4; --panel:#fff; --line:#d9d9d2; --text:#171717; --muted:#666; --accent:#0f766e; --accent-dark:#115e59; --danger:#b42318; --chip:#eef2f1; --ok:#0f766e; --warn:#9a5b00; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font-family:ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { width:min(960px, calc(100vw - 32px)); margin:0 auto; padding:24px 0 48px; display:grid; gap:14px; }
    .panel { border:1px solid var(--line); border-radius:10px; background:var(--panel); padding:18px; }
    .tabs { display:flex; flex-wrap:wrap; gap:8px; align-items:center; border:1px solid var(--line); border-radius:10px; background:var(--panel); padding:8px; }
    .tab { display:inline-flex; align-items:center; gap:8px; border:1px solid transparent; border-radius:8px; background:#fff; color:var(--muted); padding:8px 10px; font-weight:850; }
    .tab.active { border-color:rgba(15,118,110,.26); background:#e7f3f1; color:var(--accent-dark); }
    .tab-close { border:0; background:transparent; color:inherit; padding:0 2px; min-width:0; font-size:16px; line-height:1; }
    .tab-close:hover { color:var(--danger); }
    .page-head { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; }
    h2 { margin:0 0 8px; font-size:25px; }
    h3 { margin:0 0 10px; font-size:18px; }
    p { margin:0 0 12px; color:var(--muted); line-height:1.65; }
    textarea { width:100%; min-height:140px; resize:vertical; border:1px solid var(--line); border-radius:8px; padding:11px 12px; line-height:1.65; font:inherit; color:var(--text); background:#fff; }
    button { border:1px solid var(--accent); border-radius:7px; background:var(--accent); color:#fff; padding:8px 12px; cursor:pointer; font-weight:800; font:inherit; }
    button.secondary { background:#fff; color:var(--accent-dark); border-color:var(--line); }
    button.danger { background:#fff; color:var(--danger); border-color:#f3b5ad; }
    button.close { width:34px; height:34px; padding:0; border-radius:999px; background:#fff; color:var(--muted); border-color:var(--line); font-size:18px; line-height:1; }
    button:disabled { opacity:.55; cursor:not-allowed; }
    .status { color:var(--muted); font-size:13px; min-height:20px; }
    .status.error { color:var(--danger); }
    .hidden { display:none !important; }
    .actions { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; align-items:center; }
    .topic-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(180px, 1fr)); gap:10px; margin-top:12px; }
    .topic-chip { display:flex; justify-content:space-between; align-items:center; gap:10px; border:1px solid rgba(15,118,110,.26); border-radius:10px; background:#e7f3f1; color:#143b36; padding:10px 12px; cursor:pointer; text-align:left; }
    .topic-chip .name { font-weight:850; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .topic-chip .count { color:var(--accent-dark); font-size:12px; font-weight:800; flex:0 0 auto; }
    .topic-chip.unselected { background:#fff; border-color:var(--line); color:var(--muted); opacity:.62; }
    .topic-chip.unselected .count { color:var(--muted); }
    .overview-line { display:flex; gap:8px; flex-wrap:wrap; color:var(--muted); font-size:13px; }
    .badge { display:inline-flex; border-radius:999px; padding:3px 8px; background:var(--chip); color:#3f3f3f; font-size:12px; font-weight:750; }
    .workspace-progress { display:flex; align-items:center; gap:10px; flex-wrap:wrap; padding:10px 12px; border:1px solid var(--line); border-radius:10px; background:var(--panel); }
    .tab-progress { font-size:12px; font-weight:700; opacity:.88; margin-left:2px; }
    .nav-row { display:flex; justify-content:space-between; gap:8px; align-items:center; margin-bottom:12px; }
    .weak-point { border-left:4px solid var(--accent); padding:12px 14px; background:#f5fbfa; border-radius:8px; line-height:1.7; font-size:18px; font-weight:760; }
    .strategy-tip { margin-top:8px; color:var(--muted); font-size:13px; line-height:1.55; }
    .meta { margin-top:8px; color:var(--muted); font-size:13px; overflow-wrap:anywhere; }
    .result { border-top:1px solid var(--line); margin-top:16px; padding-top:16px; display:grid; gap:12px; }
    .result.compact { gap:8px; line-height:1.65; }
    .result.compact p { margin:0; color:var(--text); }
    .weak-result { border:1px solid var(--line); border-radius:8px; padding:10px; background:#fff; margin-top:8px; }
    .citations { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
    .citation { border:1px solid var(--line); border-radius:999px; padding:3px 8px; font-size:12px; background:#f8fbfa; color:#334155; }
    .dialogue { display:grid; gap:10px; margin:12px 0; }
    .bubble { border:1px solid var(--line); border-radius:10px; padding:10px 12px; line-height:1.65; background:#fff; }
    .bubble.user { margin-left:8%; background:#fdfdfb; }
    .bubble.assistant { margin-right:8%; background:#f6fbfa; }
    .bubble p { margin:0 0 10px; color:var(--text); }
    .bubble p:last-child { margin-bottom:0; }
    .bubble ul, .bubble ol { margin:8px 0 10px 20px; padding:0; }
    .bubble hr { border:0; border-top:1px solid var(--line); margin:12px 0; }
    .bubble code { background:#eef2f1; border-radius:4px; padding:1px 4px; }
    .bubble pre { overflow-x:auto; background:#eef2f1; border-radius:7px; padding:10px; }
    .bubble table { width:100%; border-collapse:collapse; margin:10px 0 12px; font-size:14px; line-height:1.5; }
    .bubble th, .bubble td { border:1px solid var(--line); padding:7px 9px; text-align:left; vertical-align:top; }
    .bubble th { background:#eef2f1; font-weight:800; }
    .bubble tr:nth-child(even) td { background:#fafbf9; }
    .empty { border:1px dashed var(--line); border-radius:10px; padding:18px; color:var(--muted); background:#fff; }
    @media (max-width:760px) { .page-head, .nav-row { display:block; } .bubble.user, .bubble.assistant { margin:0; } }
  </style>
</head>
<body>
  <main>
    <nav id="reviewTabs" class="tabs" aria-label="复习模式"></nav>
    <section id="selectPanel" class="panel"></section>
    <section id="cardPanel" class="panel hidden"></section>
    <section id="dialoguePanel" class="panel hidden"></section>
    <section id="donePanel" class="panel hidden"></section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const CURRENT_REVIEW_PROMPT_VERSION = "__REVIEW_PROMPT_VERSION__";
    let mode = "selecting";
    let selectionState = {topics: [], selectedTopics: new Set(), loading: false};
    let cardReviewState = freshCardReviewState();
    let dialogueReviewState = freshDialogueReviewState();
    let reviewPollInFlight = false;

    document.addEventListener("click", async (event) => {
      const target = event.target;
      if (target.dataset.topicToggle) toggleTopic(target.dataset.topicToggle);
      if (target.dataset.tab) switchTab(target.dataset.tab);
      if (target.dataset.closeTab) closeReviewTab(target.dataset.closeTab, event);
      if (target.id === "selectAllTopics") selectAllTopics();
      if (target.id === "clearTopics") clearTopics();
      if (target.id === "startCardReview") await startCardReview();
      if (target.id === "startDialogueReview") await startDialogueReview();
      if (target.dataset.nav) moveCard(Number(target.dataset.nav));
      if (target.id === "submitVerify") await submitVerify();
      if (target.dataset.weakAction) await commitWeakAction(target.dataset.weakId, target.dataset.weakAction);
      if (target.id === "newReview") exitToSelecting();
      if (target.id === "sendDialogueReview") await sendDialogueReview();
      if (target.id === "retryCardReview") retryCardReview();
      if (target.id === "regenerateCardReview") await regenerateCurrentCard();
      if (target.id === "regenerateCardReviewRun") await regenerateCardReviewRun();
    });

    function freshCardReviewState() {
      return {active: false, reviewRunId: "", runStatus: "", cards: [], index: 0, verify: null, results: [], pollTimer: null, topics: [], doneTitle: "", doneMessage: "", runStale: false, promptVersion: CURRENT_REVIEW_PROMPT_VERSION, needsReplan: false};
    }

    function freshDialogueReviewState() {
      return {active: false, reviewRunId: "", topics: [], history: [], pendingSuggestions: [], busy: false};
    }

    let reviewWorkspacePatchTimer = null;
    const REVIEW_SNAPSHOT_VERSION = 1;

    function buildReviewSnapshot() {
      return {
        version: REVIEW_SNAPSHOT_VERSION,
        promptVersion: CURRENT_REVIEW_PROMPT_VERSION,
        savedAt: new Date().toISOString(),
        mode,
        selectionState: {
          topics: selectionState.topics,
          selectedTopics: Array.from(selectionState.selectedTopics || []),
          overview: selectionState.overview || null,
          error: selectionState.error || null,
        },
        cardReviewState: {
          active: cardReviewState.active,
          reviewRunId: cardReviewState.reviewRunId,
          runStatus: cardReviewState.runStatus,
          cards: cardReviewState.cards,
          index: cardReviewState.index,
          verify: cardReviewState.verify,
          results: cardReviewState.results,
          topics: cardReviewState.topics,
          doneTitle: cardReviewState.doneTitle,
          doneMessage: cardReviewState.doneMessage,
          runStale: Boolean(cardReviewState.runStale),
          promptVersion: cardReviewState.promptVersion || CURRENT_REVIEW_PROMPT_VERSION,
        },
        dialogueReviewState: {
          active: dialogueReviewState.active,
          reviewRunId: dialogueReviewState.reviewRunId || "",
          topics: dialogueReviewState.topics,
          history: dialogueReviewState.history,
          pendingSuggestions: dialogueReviewState.pendingSuggestions,
        },
      };
    }

    function buildReviewWorkspacePatch() {
      const snapshot = buildReviewSnapshot();
      return {
        promptVersion: snapshot.promptVersion,
        mode: snapshot.mode,
        selectionState: snapshot.selectionState,
        cardReviewState: snapshot.cardReviewState,
        dialogueReviewState: snapshot.dialogueReviewState,
      };
    }

    function getActiveReviewRunId() {
      if (cardReviewState.reviewRunId) return cardReviewState.reviewRunId;
      if (dialogueReviewState.reviewRunId) return dialogueReviewState.reviewRunId;
      return "";
    }

    function scheduleReviewWorkspacePatch() {
      const runId = getActiveReviewRunId();
      if (!runId) return;
      if (reviewWorkspacePatchTimer) window.clearTimeout(reviewWorkspacePatchTimer);
      reviewWorkspacePatchTimer = window.setTimeout(async () => {
        reviewWorkspacePatchTimer = null;
        try {
          await fetch(`/api/review/runs/${encodeURIComponent(runId)}/workspace`, {
            method: "PATCH",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(buildReviewWorkspacePatch()),
          });
        } catch {}
      }, 400);
    }

    function cancelReviewWorkspacePatch() {
      if (reviewWorkspacePatchTimer) window.clearTimeout(reviewWorkspacePatchTimer);
      reviewWorkspacePatchTimer = null;
    }

    function syncReviewWorkspace(options = {}) {
      const ws = window.KnowledgeAgentWorkspace;
      if (!ws) return;
      ws.replace("review", buildReviewSnapshot());
      ws.persist();
      if (options.patch !== false) scheduleReviewWorkspacePatch();
    }

    function applyServerReviewWorkspace(workspace) {
      if (!workspace || typeof workspace !== "object") return;
      applyReviewSnapshot({version: 1, ...workspace});
    }

    function inferCardPromptVersion(cards) {
      if (!Array.isArray(cards)) return "";
      for (const card of cards) {
        const version = card?.review_prompt?.prompt_version || card?.prompt_version || "";
        if (version) return version;
      }
      return "";
    }

    function normalizePromptVersion(value) {
      const text = String(value || "").trim();
      if (!text || text === "__REVIEW_PROMPT_VERSION__") return "";
      return text;
    }

    function resolveSavedPromptVersion(snapshot, cards) {
      const cardState = snapshot?.cardReviewState || {};
      return normalizePromptVersion(cardState.promptVersion)
        || normalizePromptVersion(snapshot?.promptVersion)
        || inferCardPromptVersion(cards)
        || CURRENT_REVIEW_PROMPT_VERSION;
    }

    function migrateReviewSnapshot(snapshot) {
      if (!snapshot || typeof snapshot !== "object") return null;
      let version = Number(snapshot.version) || 0;
      if (version >= REVIEW_SNAPSHOT_VERSION) return snapshot;
      const next = {...snapshot, version: REVIEW_SNAPSHOT_VERSION};
      if (version < 1) {
        if (!next.promptVersion && next.cardReviewState?.promptVersion) {
          next.promptVersion = next.cardReviewState.promptVersion;
        }
      }
      return next.version === REVIEW_SNAPSHOT_VERSION ? next : null;
    }

    function applyReviewSnapshot(snapshot) {
      const migrated = migrateReviewSnapshot(snapshot);
      if (!migrated) return false;
      snapshot = migrated;
      mode = snapshot.mode || "selecting";
      const sel = snapshot.selectionState || {};
      selectionState = {
        topics: Array.isArray(sel.topics) ? sel.topics : [],
        selectedTopics: new Set(Array.isArray(sel.selectedTopics) ? sel.selectedTopics.filter(Boolean) : []),
        loading: false,
        overview: sel.overview || null,
        error: sel.error || null,
      };
      const card = snapshot.cardReviewState || {};
      const savedPromptVersion = resolveSavedPromptVersion(snapshot, card.cards || []);
      const savedCards = Array.isArray(card.cards) ? card.cards : [];
      const staleCardPrompts = Boolean(card.active) && savedCards.length > 0 && savedPromptVersion !== CURRENT_REVIEW_PROMPT_VERSION;
      const cardTopics = Array.isArray(card.topics) && card.topics.length
        ? card.topics
        : Array.from(selectionState.selectedTopics || []);
      cardReviewState = {
        ...freshCardReviewState(),
        active: Boolean(card.active),
        reviewRunId: staleCardPrompts ? "" : (card.reviewRunId || ""),
        runStatus: staleCardPrompts ? "" : (card.runStatus || ""),
        cards: staleCardPrompts ? [] : savedCards,
        index: staleCardPrompts ? 0 : (Number(card.index) || 0),
        verify: staleCardPrompts ? null : (card.verify || null),
        results: staleCardPrompts ? [] : (Array.isArray(card.results) ? card.results : []),
        topics: cardTopics,
        doneTitle: staleCardPrompts ? "" : (card.doneTitle || ""),
        doneMessage: staleCardPrompts ? "" : (card.doneMessage || ""),
        runStale: staleCardPrompts ? false : Boolean(card.runStale),
        promptVersion: staleCardPrompts ? CURRENT_REVIEW_PROMPT_VERSION : (savedPromptVersion || CURRENT_REVIEW_PROMPT_VERSION),
        needsReplan: staleCardPrompts,
      };
      const dlg = snapshot.dialogueReviewState || {};
      dialogueReviewState = {
        ...freshDialogueReviewState(),
        active: Boolean(dlg.active),
        reviewRunId: dlg.reviewRunId || "",
        topics: Array.isArray(dlg.topics) ? dlg.topics : [],
        history: Array.isArray(dlg.history) ? dlg.history : [],
        pendingSuggestions: Array.isArray(dlg.pendingSuggestions) ? dlg.pendingSuggestions : [],
      };
      return true;
    }

    async function refreshSelectionOverview() {
      const previousSelected = new Set(selectionState.selectedTopics || []);
      selectionState.loading = true;
      render();
      try {
        const data = await getJson("/api/review/due?limit=50");
        selectionState.topics = Array.isArray(data.topics) ? data.topics : [];
        selectionState.overview = data;
        selectionState.error = null;
        const validTopics = new Set(selectionState.topics.map((item) => item.topic).filter(Boolean));
        if (previousSelected.size) {
          selectionState.selectedTopics = new Set([...previousSelected].filter((topic) => validTopics.has(topic)));
        } else {
          selectionState.selectedTopics = new Set([...validTopics]);
        }
      } catch (error) {
        selectionState.error = error.message || String(error);
      } finally {
        selectionState.loading = false;
        render();
        syncReviewWorkspace({patch: false});
      }
    }

    async function resyncCardReviewRun() {
      const runId = getActiveReviewRunId();
      if (!runId) return;
      if (cardReviewState.active && !cardReviewState.reviewRunId) return;
      if (dialogueReviewState.active && !dialogueReviewState.reviewRunId && !cardReviewState.reviewRunId) return;
      const beforeCard = currentCard();
      const beforeCardId = beforeCard?.id || "";
      const hadUsableCurrent = isCardUsable(beforeCard);
      try {
        const data = await getJson(`/api/review/plan/${encodeURIComponent(runId)}`);
        if (cardReviewState.active && cardReviewState.reviewRunId) {
          cardReviewState.runStale = false;
          cardReviewState.runStatus = data.status || "";
          cardReviewState.promptVersion = data.prompt_version || cardReviewState.promptVersion || CURRENT_REVIEW_PROMPT_VERSION;
          cardReviewState.cards = mergeCards(cardReviewState.cards, data.cards || []);
          if (!currentCard() || !isCardViewable(currentCard())) selectFirstUsableCard();
          if ((data.status || "") !== "done") startPolling();
          else stopPolling();
          const afterCard = currentCard();
          const shouldRender = !hadUsableCurrent || !isCardUsable(afterCard) || beforeCardId !== (afterCard?.id || "");
          if (mode === "card_review" && shouldRender) renderCardReview();
        } else if (data.workspace) {
          applyServerReviewWorkspace(data.workspace);
        }
        syncReviewWorkspace({patch: false});
      } catch (error) {
        stopPolling();
        if (cardReviewState.active) {
          cardReviewState.runStale = true;
          cardReviewState.runStatus = "stale";
        }
        setStatus("服务端 review run 已失效，本地进度仍保留。可继续逐条对照或新建复习。", true);
        render();
        syncReviewWorkspace({patch: false});
      }
    }

    async function initReviewPage() {
      const ws = window.KnowledgeAgentWorkspace;
      if (ws) ws.hydrate();
      const saved = ws ? ws.get("review") : null;
      if (saved && saved.version === 1) {
        stopPolling();
        applyReviewSnapshot(saved);
        render();
        if (cardReviewState.active && cardReviewState.needsReplan) {
          setStatus("复习题生成规则已更新，正在重新生成本轮卡片...");
          await startCardReview({force: true, topics: cardReviewState.topics});
        } else if ((cardReviewState.active && cardReviewState.reviewRunId && !cardReviewState.runStale)
          || (dialogueReviewState.active && dialogueReviewState.reviewRunId)) {
          await resyncCardReviewRun();
        } else if (cardReviewState.active && cardReviewState.reviewRunId && cardReviewState.runStale) {
          stopPolling();
        } else if (cardReviewState.active && cardReviewState.runStatus !== "done" && cardReviewState.cards.some((item) => item.status === "pending")) {
          startPolling();
        }
        syncReviewWorkspace({patch: false});
        await refreshSelectionOverview();
        return;
      }
      await loadSelection();
    }

    function pauseReviewPage() {
      stopPolling();
      cancelReviewWorkspacePatch();
      syncReviewWorkspace({patch: false});
    }

    function resumeReviewPage() {
      if (document.visibilityState === "hidden") return;
      if (cardReviewState.active && cardReviewState.reviewRunId && !cardReviewState.runStale) {
        resyncCardReviewRun();
      }
    }

    window.addEventListener("pagehide", pauseReviewPage);
    window.addEventListener("beforeunload", pauseReviewPage);
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden") pauseReviewPage();
      else resumeReviewPage();
    });
    initReviewPage();

    async function loadSelection() {
      mode = "selecting";
      selectionState.loading = true;
      render();
      try {
        const data = await getJson("/api/review/due?limit=50");
        selectionState.topics = Array.isArray(data.topics) ? data.topics : [];
        selectionState.selectedTopics = new Set(selectionState.topics.map((item) => item.topic).filter(Boolean));
        selectionState.overview = data;
      } catch (error) {
        selectionState.error = error.message || String(error);
      } finally {
        selectionState.loading = false;
        render();
        syncReviewWorkspace();
      }
    }

    function render() {
      $("donePanel").classList.add("hidden");
      renderTabs();
      $("selectPanel").classList.toggle("hidden", mode !== "selecting");
      $("cardPanel").classList.toggle("hidden", mode !== "card_review" || !cardReviewState.active);
      $("dialoguePanel").classList.toggle("hidden", mode !== "dialogue_review" || !dialogueReviewState.active);
      if (mode === "selecting") renderSelecting();
      if (mode === "card_review" && cardReviewState.active) renderCardReview();
      if (mode === "dialogue_review" && dialogueReviewState.active) renderDialogueReview();
    }

    function renderTabs() {
      const tabs = [
        `<button type="button" class="tab ${mode === "selecting" ? "active" : ""}" data-tab="selecting">复习总览</button>`
      ];
      if (cardReviewState.active) {
        tabs.push(`<button type="button" class="tab ${mode === "card_review" ? "active" : ""}" data-tab="card_review">逐条对照 <span class="tab-progress">${escapeHtml(cardProgressSummary())}</span> <span class="tab-close" data-close-tab="card_review" title="关闭逐条对照">×</span></button>`);
      }
      if (dialogueReviewState.active) {
        tabs.push(`<button type="button" class="tab ${mode === "dialogue_review" ? "active" : ""}" data-tab="dialogue_review">对话复查 <span class="tab-close" data-close-tab="dialogue_review" title="关闭对话复查">×</span></button>`);
      }
      $("reviewTabs").innerHTML = tabs.join("");
    }

    function switchTab(tab) {
      if (tab === "card_review" && !cardReviewState.active) return;
      if (tab === "dialogue_review" && !dialogueReviewState.active) return;
      if (!["selecting", "card_review", "dialogue_review"].includes(tab)) return;
      mode = tab;
      render();
      syncReviewWorkspace();
    }

    function closeReviewTab(tab, event) {
      event?.stopPropagation?.();
      if (tab === "card_review") {
        const total = cardReviewState.cards.length;
        const done = cardReviewState.cards.filter(isCardCompleted).length;
        const message = total
          ? `你在这轮逐条对照里已完成 ${done} / 共 ${total} 题，确定结束？`
          : "确定结束逐条对照？";
        if (!window.confirm(message)) return;
        stopPolling();
        cardReviewState = freshCardReviewState();
        if (mode === "card_review") mode = "selecting";
      } else if (tab === "dialogue_review") {
        const rounds = Math.floor((dialogueReviewState.history || []).length / 2);
        if (!window.confirm(`本轮对话已进行 ${rounds} 轮，确定结束？`)) return;
        dialogueReviewState = freshDialogueReviewState();
        if (mode === "dialogue_review") mode = "selecting";
      }
      render();
      syncReviewWorkspace();
    }

    function renderSelecting() {
      const selectedCount = selectionState.selectedTopics.size;
      const dueCount = selectionState.overview?.candidate_count || selectionState.overview?.due_count || 0;
      const recommendedCount = selectionState.overview?.recommended_count || 0;
      const neverReviewedCount = selectionState.overview?.never_reviewed_count || 0;
      const recentCount = selectionState.overview?.recent_count || 0;
      const strategyCount = selectionState.overview?.strategy_due_count || 0;
      const disabled = selectedCount === 0 || selectionState.loading;
      const topicsHtml = selectionState.topics.length
        ? `<div class="topic-grid">${selectionState.topics.map((item) => {
            const topic = item.topic || "";
            const selected = selectionState.selectedTopics.has(topic);
            return `<button type="button" class="topic-chip ${selected ? "" : "unselected"}" data-topic-toggle="${escapeHtml(topic)}">
              <span class="name">${escapeHtml(topic)}</span>
              <span class="count">${reviewTopicLabel(item)}</span>
            </button>`;
          }).join("")}</div>`
        : `<div class="empty">暂无可复习知识弱项。可以先完成一次面试并结束 session 后再回来复习。</div>`;
      const workspaceHtml = renderWorkspaceSummary();
      $("selectPanel").innerHTML = `
        <div class="page-head">
          <div>
            <h2>选择本轮复习范围</h2>
            <p>默认全选所有建议复查 topic。进入逐条对照或对话复查后，本轮范围会固定。</p>
          </div>
        </div>
        <div class="overview-line">
          <span>知识弱项 ${escapeHtml(dueCount)} 个可复习</span>
          <span>未复查 ${escapeHtml(neverReviewedCount)} 个</span>
          <span>建议复查 ${escapeHtml(recommendedCount)} 个</span>
          <span>最近确认 ${escapeHtml(recentCount)} 个</span>
          ${strategyCount ? `<span>策略弱项 ${escapeHtml(strategyCount)} 个建议在面试中训练</span>` : ""}
        </div>
        <div id="status" class="status ${selectionState.error ? "error" : ""}">${selectionState.loading ? "正在加载建议复查 topic..." : escapeHtml(selectionState.error || selectionState.overview?.summary || "")}</div>
        <div class="actions">
          <button id="selectAllTopics" class="secondary" type="button">全选</button>
          <button id="clearTopics" class="secondary" type="button">清空</button>
          <span class="status">已选 ${selectedCount} / ${selectionState.topics.length}</span>
        </div>
        ${topicsHtml}
        ${workspaceHtml}
        <div class="actions">
          <button id="startCardReview" type="button" ${disabled ? "disabled" : ""}>${cardReviewState.active ? "进入逐条对照" : "开始逐条对照"}</button>
          <button id="startDialogueReview" class="secondary" type="button" ${disabled ? "disabled" : ""}>${dialogueReviewState.active ? "进入对话复查" : "开始对话复查"}</button>
        </div>
      `;
    }

    function renderWorkspaceSummary() {
      const items = [];
      if (cardReviewState.active) {
        items.push(`<div class="workspace-progress"><strong>逐条对照进行中</strong><span class="badge">${escapeHtml(cardProgressSummary())}</span></div>`);
      }
      if (dialogueReviewState.active) {
        items.push(`<div class="empty"><strong>对话复查进行中</strong> (已聊 ${escapeHtml(dialogueRoundCount())} 轮)</div>`);
      }
      return items.length ? `<div style="display:grid;gap:8px;margin-top:12px">${items.join("")}</div>` : "";
    }

    function reviewTopicLabel(item) {
      const values = [];
      const neverReviewed = Number(item.never_reviewed_count || 0);
      const recommended = Number(item.recommended_count || 0);
      const recent = Number(item.recent_count || 0);
      if (neverReviewed) values.push(`未复查 ${neverReviewed}`);
      if (recommended) values.push(`建议复查 ${recommended}`);
      if (recent) values.push(`最近确认 ${recent}`);
      return escapeHtml(values.join(" · ") || `${item.candidate_count || item.due || 0} 可复习`);
    }

    function reviewStateLabel(state) {
      if (state === "never_reviewed") return "未复查";
      if (state === "recent") return "最近确认";
      return "建议复查";
    }

    function toggleTopic(topic) {
      if (!topic) return;
      if (selectionState.selectedTopics.has(topic)) selectionState.selectedTopics.delete(topic);
      else selectionState.selectedTopics.add(topic);
      renderSelecting();
      syncReviewWorkspace();
    }

    function selectAllTopics() {
      selectionState.selectedTopics = new Set(selectionState.topics.map((item) => item.topic).filter(Boolean));
      renderSelecting();
      syncReviewWorkspace();
    }

    function clearTopics() {
      selectionState.selectedTopics = new Set();
      renderSelecting();
      syncReviewWorkspace();
    }

    function selectedTopics() {
      return Array.from(selectionState.selectedTopics);
    }

    function cardProgressSummary() {
      const total = cardReviewState.cards.length;
      const done = cardReviewState.cards.filter(isCardCompleted).length;
      if (cardReviewState.doneTitle) return cardReviewState.doneTitle;
      if (!total) return "准备中";
      return `已完成 ${done} / 共 ${total}`;
    }

    function dialogueRoundCount() {
      return Math.floor((dialogueReviewState.history || []).length / 2);
    }

    async function startCardReview(options = {}) {
      const force = Boolean(options.force);
      if (cardReviewState.active && !force) {
        mode = "card_review";
        render();
        syncReviewWorkspace();
        return;
      }
      const topics = Array.isArray(options.topics) && options.topics.length ? options.topics : selectedTopics();
      if (!topics.length) return;
      stopPolling();
      mode = "card_review";
      cardReviewState = {...freshCardReviewState(), active: true, topics, promptVersion: CURRENT_REVIEW_PROMPT_VERSION};
      render();
      syncReviewWorkspace();
      setStatus(force ? "正在按最新规则重新生成逐条对照题目..." : "正在创建逐条对照本轮题目...");
      try {
        const data = await postJson("/api/review/plan", {topics, limit: 12, max_strategy_constraints: 2, force_regenerate: force});
        cardReviewState.reviewRunId = data.review_run_id || "";
        cardReviewState.runStatus = data.status || "";
        cardReviewState.runStale = false;
        cardReviewState.promptVersion = data.prompt_version || CURRENT_REVIEW_PROMPT_VERSION;
        cardReviewState.cards = data.cards || [];
        if (!cardReviewState.cards.length) {
          cardReviewState.doneTitle = "暂无可复习卡片";
          cardReviewState.doneMessage = data.summary || "所选 topic 当前没有可复习知识弱项。";
          renderCardReview();
          syncReviewWorkspace();
          return;
        }
        selectFirstUsableCard();
        renderCardReview();
        startPolling();
        syncReviewWorkspace();
      } catch (error) {
        setStatus(error.message || String(error), true);
        syncReviewWorkspace();
      }
    }

    function startPolling() {
      stopPolling();
      if (!cardReviewState.reviewRunId || cardReviewState.runStale) return;
      if (document.visibilityState === "hidden") return;
      cardReviewState.pollTimer = window.setInterval(pollReviewRun, 1200);
      pollReviewRun();
    }

    function stopPolling() {
      if (cardReviewState.pollTimer) window.clearInterval(cardReviewState.pollTimer);
      cardReviewState.pollTimer = null;
    }

    async function pollReviewRun() {
      if (!cardReviewState.reviewRunId || !cardReviewState.active || cardReviewState.runStale) return;
      if (document.visibilityState === "hidden") return;
      if (reviewPollInFlight) return;
      reviewPollInFlight = true;
      try {
        const beforeCard = currentCard();
        const beforeCardId = beforeCard?.id || "";
        const hadUsableCurrent = isCardUsable(beforeCard);
        const data = await getJson(`/api/review/plan/${encodeURIComponent(cardReviewState.reviewRunId)}`);
        cardReviewState.runStatus = data.status || "";
        cardReviewState.cards = mergeCards(cardReviewState.cards, data.cards || []);
        if (!currentCard() || !isCardViewable(currentCard())) selectFirstUsableCard();
        const afterCard = currentCard();
        const shouldRender = !hadUsableCurrent || beforeCardId !== (afterCard?.id || "");
        if (mode === "card_review" && shouldRender) renderCardReview();
        if (mode === "selecting") renderSelecting();
        const ready = data.ready_count || 0;
        const pending = data.pending_count || 0;
        const failed = data.failed_count || 0;
        if (mode === "card_review" && (ready || failed)) setStatus(`已准备 ${ready} 题，待生成 ${pending} 题${failed ? `，失败 ${failed} 题` : ""}。`);
        if ((data.status || "") === "done") stopPolling();
        syncReviewWorkspace({patch: false});
      } catch (error) {
        stopPolling();
        cardReviewState.runStale = true;
        cardReviewState.runStatus = "stale";
        setStatus(error.message || String(error), true);
        syncReviewWorkspace({patch: false});
      } finally {
        reviewPollInFlight = false;
      }
    }

    function renderCardReview() {
      const topicLine = cardReviewState.topics.join("、");
      const card = currentCard();
      if (cardReviewState.doneTitle) {
        $("cardPanel").innerHTML = `
          <div class="page-head"><div><h2>逐条对照</h2><p>范围：${escapeHtml(topicLine)}</p></div></div>
          <div class="empty"><strong>${escapeHtml(cardReviewState.doneTitle)}</strong><br>${escapeHtml(cardReviewState.doneMessage || "")}</div>
        `;
        return;
      }
      if (!card) {
        $("cardPanel").innerHTML = `
          <div class="page-head"><div><h2>逐条对照</h2><p>范围：${escapeHtml(topicLine)}</p></div></div>
          <div id="status" class="status">正在准备题目...</div>
        `;
        return;
      }
      if (!isCardViewable(card)) {
        const ready = cardReviewState.cards.filter(isCardPending).length;
        const pending = cardReviewState.cards.filter((item) => item.status === "pending").length;
        $("cardPanel").innerHTML = `
          <div class="page-head"><div><h2>逐条对照</h2><p>范围：${escapeHtml(topicLine)}</p></div></div>
          <div id="status" class="status">已准备 ${ready} 题，待生成 ${pending} 题。</div>
          <div class="empty">第一题准备好后会自动显示。</div>
        `;
        return;
      }
      if (isCardCompleted(card)) {
        renderCompletedCardReview(card, topicLine);
        return;
      }
      const promptPayload = card.review_prompt || null;
      const promptText = promptPayload?.prompt || "正在生成复习题...";
      const promptReady = Boolean(promptPayload?.prompt);
      const verifyActive = Boolean(cardReviewState.verify);
      const answerValue = card.last_answer || "";
      $("cardPanel").innerHTML = `
        <div class="page-head"><div><h2>逐条对照</h2><p>范围：${escapeHtml(topicLine)}</p></div></div>
        <div id="status" class="status">${escapeHtml(card.topic || "未分类")} · ${topicProgress(card)}</div>
        <div class="nav-row">
          <button class="secondary" type="button" data-nav="-1" aria-label="上一条">←</button>
          <span class="badge">第 ${cardReviewState.index + 1} / ${cardReviewState.cards.length} 题</span>
          <button class="secondary" type="button" data-nav="1" aria-label="下一条">→</button>
        </div>
        <div class="weak-point">${escapeHtml(promptText)}</div>
        ${promptPayload?.hint ? `<div class="meta">提示：${escapeHtml(promptPayload.hint)}</div>` : ""}
        <textarea id="reviewAnswer" style="margin-top:14px" placeholder="先凭记忆回答，再提交对照。" ${verifyActive ? "disabled" : ""}>${escapeHtml(answerValue)}</textarea>
        ${renderStrategyConstraints(card.strategy_constraints || [])}
        <div class="actions">
          <button id="submitVerify" type="button" ${promptReady && !verifyActive ? "" : "disabled"}>提交</button>
          <button id="regenerateCardReview" class="secondary" type="button">重新生成本题</button>
          <button id="regenerateCardReviewRun" class="secondary" type="button">重新生成本轮</button>
        </div>
        <div id="verifyResult" class="result hidden"></div>
      `;
      if (cardReviewState.verify) renderVerifyResult(cardReviewState.verify);
    }

    function renderCompletedCardReview(card, topicLine) {
      const promptPayload = card.review_prompt || null;
      const promptText = promptPayload?.prompt || card.point || "";
      const verify = card.last_verify || null;
      const confirmed = card.confirmed_weak_points || {};
      const actionSummary = Object.entries(confirmed).map(([id, action]) => {
        const label = action === "improve" ? "已确认改善" : "还要再练";
        return `<li>${escapeHtml(label)} · ${escapeHtml(id)}</li>`;
      }).join("");
      $("cardPanel").innerHTML = `
        <div class="page-head"><div><h2>逐条对照</h2><p>范围：${escapeHtml(topicLine)}</p></div></div>
        <div id="status" class="status">${escapeHtml(card.topic || "未分类")} · ${topicProgress(card)} · 已完成</div>
        <div class="nav-row">
          <button class="secondary" type="button" data-nav="-1" aria-label="上一条">←</button>
          <span class="badge">第 ${cardReviewState.index + 1} / ${cardReviewState.cards.length} 题</span>
          <button class="secondary" type="button" data-nav="1" aria-label="下一条">→</button>
        </div>
        <div class="weak-point">${escapeHtml(promptText)}</div>
        ${promptPayload?.hint ? `<div class="meta">提示：${escapeHtml(promptPayload.hint)}</div>` : ""}
        <div class="meta" style="margin-top:14px">你的回答</div>
        <div class="result compact" style="margin-top:6px"><p>${escapeHtml(card.last_answer || "")}</p></div>
        <div class="result compact" style="margin-top:12px">${renderReferenceAnswer(cardReferenceAnswer(card))}</div>
        ${verify ? renderCompletedVerifySummary(verify) : ""}
        ${actionSummary ? `<ul class="meta">${actionSummary}</ul>` : ""}
        <div class="actions">
          <button id="retryCardReview" class="secondary" type="button">重新回答</button>
          <button id="regenerateCardReview" class="secondary" type="button">重新生成本题</button>
          <button id="regenerateCardReviewRun" class="secondary" type="button">重新生成本轮</button>
        </div>
      `;
    }

    function renderCompletedVerifySummary(result) {
      const correct = firstText(result.correct || [], "方向基本正确。");
      const missedValues = [].concat(Array.isArray(result.missed) ? result.missed : []).concat(Array.isArray(result.strategy_feedback) ? result.strategy_feedback : []).filter(Boolean);
      const missed = firstText(missedValues, "没有明显遗漏。");
      const example = result.example || result.feedback || result.overall || "暂无示例。";
      return `
        <div class="result compact" style="margin-top:12px">
          <p>✓ ${escapeHtml(correct)}</p>
          <p>⚠ ${escapeHtml(missed)}</p>
          <p><strong>示例：</strong><br>${escapeHtml(example)}</p>
          ${renderCitations(result.citations || [])}
        </div>
      `;
    }

    function retryCardReview() {
      const card = currentCard();
      if (!card || !isCardCompleted(card)) return;
      card.review_completed = false;
      card.last_answer = "";
      card.last_verify = null;
      card.confirmed_weak_points = {};
      cardReviewState.verify = null;
      cardReviewState.doneTitle = "";
      cardReviewState.doneMessage = "";
      renderCardReview();
      syncReviewWorkspace();
      setStatus("已重置本题，可以重新作答。");
    }

    async function regenerateCardReviewRun() {
      const topics = cardReviewState.topics && cardReviewState.topics.length ? cardReviewState.topics : selectedTopics();
      if (!topics.length) return;
      if (!window.confirm("将按最新规则重新生成本轮所有卡片，当前本轮作答和反馈会清空。继续？")) return;
      await startCardReview({force: true, topics});
    }

    async function regenerateCurrentCard() {
      const card = currentCard();
      if (!card || !cardReviewState.reviewRunId) return;
      if (!window.confirm("将按最新规则重新生成当前题目，当前本题作答和反馈会清空。继续？")) return;
      setBusy(true, "正在重新生成当前题目...");
      try {
        const cardId = card.id || card.card_id;
        card.status = "pending";
        card.cache_hit = false;
        delete card.review_prompt;
        delete card.question_blocks;
        delete card.reference_answer;
        delete card.last_answer;
        delete card.last_verify;
        delete card.confirmed_weak_points;
        card.review_completed = false;
        cardReviewState.verify = null;
        renderCardReview();
        const data = await postJson(`/api/review/plan/${encodeURIComponent(cardReviewState.reviewRunId)}/cards/${encodeURIComponent(cardId)}/regenerate`, {});
        cardReviewState.runStatus = data.status || "";
        cardReviewState.cards = mergeCards(cardReviewState.cards, data.cards || []);
        const nextIndex = cardReviewState.cards.findIndex((item) => (item.id || item.card_id) === cardId);
        if (nextIndex >= 0) cardReviewState.index = nextIndex;
        startPolling();
        renderCardReview();
        syncReviewWorkspace();
        setStatus("当前题目正在重新生成，准备好后会自动显示。");
      } catch (error) {
        setStatus(error.message || String(error), true);
      } finally {
        setBusy(false);
      }
    }

    function renderVerifyResult(result) {
      const node = $("verifyResult");
      if (!node) return;
      node.classList.remove("hidden");
      node.className = "result compact";
      const referenceAnswer = result.reference_answer || result.referenceAnswer || cardReferenceAnswer(currentCard());
      if (result.loading) {
        node.innerHTML = `
          ${renderReferenceAnswer(referenceAnswer)}
          <p>正在根据你的回答生成对照反馈...</p>
        `;
        return;
      }
      if (result.error) {
        node.innerHTML = `
          ${renderReferenceAnswer(referenceAnswer)}
          <p class="danger-text">反馈生成失败：${escapeHtml(result.error)}</p>
        `;
        return;
      }
      const correct = firstText(result.correct || [], "方向基本正确。");
      const missedValues = [].concat(Array.isArray(result.missed) ? result.missed : []).concat(Array.isArray(result.strategy_feedback) ? result.strategy_feedback : []).filter(Boolean);
      const missed = firstText(missedValues, "没有明显遗漏。");
      const example = result.example || result.feedback || result.overall || "暂无示例。";
      const card = currentCard();
      const fallbackWeakIds = card && Array.isArray(card.weak_point_ids) ? card.weak_point_ids : [];
      const weakResults = Array.isArray(result.weak_results) && result.weak_results.length
        ? result.weak_results
        : fallbackWeakIds.map((id) => ({weak_point_id: id, point: id, suggested_action: "retry", reason: result.feedback || result.overall || ""}));
      node.innerHTML = `
        ${renderReferenceAnswer(referenceAnswer)}
        <p>✓ ${escapeHtml(correct)}</p>
        <p>⚠ ${escapeHtml(missed)}</p>
        <p><strong>示例：</strong><br>${escapeHtml(example)}</p>
        ${renderCitations(result.citations || [])}
        <div class="weak-results">${weakResults.map((item) => {
          const action = item.suggested_action === "improve" ? "improve" : "retry";
          return `<div class="weak-result">
            <p><strong>${escapeHtml(action === "improve" ? "建议确认改善" : "建议继续练习")}</strong>：${escapeHtml(item.point || item.weak_point_id || "")}</p>
            ${item.reason ? `<p>${escapeHtml(item.reason)}</p>` : ""}
            <div class="actions">
              <button type="button" data-weak-id="${escapeHtml(item.weak_point_id || "")}" data-weak-action="improve">确认改善</button>
              <button type="button" class="danger" data-weak-id="${escapeHtml(item.weak_point_id || "")}" data-weak-action="retry">还要再练</button>
            </div>
          </div>`;
        }).join("")}</div>
      `;
    }

    function cardReferenceAnswer(card) {
      if (!card) return "";
      return card.reference_answer || card.review_prompt?.reference_answer || "";
    }

    function renderReferenceAnswer(referenceAnswer) {
      const text = String(referenceAnswer || "").trim();
      if (!text) return `<p><strong>参考回答：</strong><br>暂无参考回答。</p>`;
      return `<p><strong>参考回答：</strong><br>${escapeHtml(text)}</p>`;
    }

    async function submitVerify() {
      const card = currentCard();
      const answer = $("reviewAnswer")?.value.trim() || "";
      if (!answer) { setStatus("请先写下你的理解。", true); return; }
      setBusy(true, "正在对照笔记纠偏...");
      card.last_answer = answer;
      cardReviewState.verify = {
        loading: true,
        reference_answer: cardReferenceAnswer(card),
      };
      renderCardReview();
      syncReviewWorkspace();
      try {
        const prompt = card.review_prompt?.prompt || card.point || "";
        const weakPointIds = Array.isArray(card.weak_point_ids) && card.weak_point_ids.length ? card.weak_point_ids : [card.id];
        const result = await postJson("/api/review/verify", {
          card_id: card.card_id || card.id,
          weak_point_ids: weakPointIds,
          weak_point_id: weakPointIds[0] || card.id,
          answer,
          prompt,
          question_blocks: card.question_blocks || card.review_prompt?.question_blocks || [],
        });
        cardReviewState.verify = {...result, reference_answer: cardReferenceAnswer(card)};
        renderVerifyResult(result);
        setStatus("纠偏完成。确认改善或继续练习后可保留本题反馈，并进入下一题。");
        syncReviewWorkspace();
      } catch (error) {
        cardReviewState.verify = {
          loading: false,
          reference_answer: cardReferenceAnswer(card),
          error: error.message || String(error),
        };
        renderVerifyResult(cardReviewState.verify);
        setStatus(error.message || String(error), true);
      } finally {
        setBusy(false);
      }
    }

    async function commitWeakAction(weakPointId, action) {
      const card = currentCard();
      const weakId = String(weakPointId || "").trim();
      if (!weakId || !card) return;
      setBusy(true, "正在保存该弱项的复习结果...");
      try {
        const result = await postJson("/api/review/commit", {weak_point_id: weakId, action});
        card.confirmed_weak_points = card.confirmed_weak_points || {};
        card.confirmed_weak_points[weakId] = action;
        cardReviewState.results.push({card_id: card.id, weak_point_id: weakId, action, result});
        const expected = Array.isArray(card.weak_point_ids) && card.weak_point_ids.length ? card.weak_point_ids : [card.id];
        const allDone = expected.every((id) => card.confirmed_weak_points[id]);
        if (allDone) {
          card.last_answer = card.last_answer || ($("reviewAnswer")?.value.trim() || "");
          card.last_verify = cardReviewState.verify;
          card.review_completed = true;
          cardReviewState.verify = null;
          if (cardReviewState.cards.every(isCardCompleted)) {
            setStatus("本轮已全部完成，可左右切换回看各题。");
          } else {
            setStatus("本题已完成，可回看反馈或用 → 进入下一题。");
          }
          renderCardReview();
        } else if (cardReviewState.verify) {
          renderVerifyResult(cardReviewState.verify);
        }
        setStatus(action === "improve" ? "已确认该弱项改善。" : "已标记该弱项继续练习。");
        syncReviewWorkspace();
      } catch (error) {
        setStatus(error.message || String(error), true);
      } finally {
        setBusy(false);
      }
    }

    async function startDialogueReview() {
      if (dialogueReviewState.active) {
        mode = "dialogue_review";
        render();
        syncReviewWorkspace();
        return;
      }
      const topics = selectedTopics();
      if (!topics.length) return;
      mode = "dialogue_review";
      dialogueReviewState = {...freshDialogueReviewState(), active: true, topics};
      render();
      syncReviewWorkspace();
      try {
        const data = await postJson("/api/review/dialogue/sessions", {topics});
        dialogueReviewState.reviewRunId = data.review_run_id || "";
        if (data.workspace) applyServerReviewWorkspace(data.workspace);
        syncReviewWorkspace();
        await sendDialogueReview("请开始本轮对话复查。先从选中 topic 的建议复查弱项中选择一个最适合开始的点，直接提出第一题。", true);
      } catch (error) {
        setStatus(error.message || String(error), true);
        syncReviewWorkspace();
      }
    }

    function renderDialogueReview() {
      const topics = dialogueReviewState.topics.join("、");
      $("dialoguePanel").innerHTML = `
        <div class="page-head"><div><h2>对话复查</h2><p>范围：${escapeHtml(topics)}</p></div></div>
        <div id="status" class="status">${dialogueReviewState.busy ? "正在进行对话复查..." : "对话复查不会自动写入 profile，确认建议后再提交更新。"}</div>
        <div id="dialogueMessages" class="dialogue">${renderDialogueMessages()}</div>
        <textarea id="dialogueMessage" placeholder="回答 Agent 的追问。"></textarea>
        <div class="actions"><button id="sendDialogueReview" type="button" ${dialogueReviewState.busy ? "disabled" : ""}>发送</button></div>
      `;
    }

    function renderDialogueMessages() {
      const history = dialogueReviewState.history || [];
      if (!history.length) return `<div class="empty">正在启动对话复查...</div>`;
      return history.map((item) => {
        const isUser = item.role === "user";
        const content = isUser ? escapeHtml(item.content || "") : renderReviewMarkdown(item.content || "");
        return `<div class="bubble ${isUser ? "user" : "assistant"}"><strong>${isUser ? "你" : "Agent"}：</strong>${content}</div>`;
      }).join("");
    }

    function renderReviewMarkdown(text) {
      const codeBlocks = [];
      let source = String(text || "").replace(/```([\s\S]*?)```/g, (_match, code) => {
        const token = `@@CODE_${codeBlocks.length}@@`;
        codeBlocks.push(`<pre><code>${escapeHtml(code.trim())}</code></pre>`);
        return token;
      });
      source = escapeHtml(source);
      const lines = source.split(/\r?\n/);
      const html = [];
      let list = [];
      let orderedList = [];
      const flush = () => {
        if (list.length) {
          html.push(`<ul>${list.map((item) => `<li>${inlineReviewMarkdown(item)}</li>`).join("")}</ul>`);
          list = [];
        }
        if (orderedList.length) {
          html.push(`<ol>${orderedList.map((item) => `<li>${inlineReviewMarkdown(item)}</li>`).join("")}</ol>`);
          orderedList = [];
        }
      };
      for (let i = 0; i < lines.length; i += 1) {
        const line = lines[i];
        const trimmed = line.trim();
        if (!trimmed) {
          const previous = lines[i - 1] ? lines[i - 1].trim() : "";
          const next = lines[i + 1] ? lines[i + 1].trim() : "";
          if (isReviewTableRow(previous) && isReviewTableRow(next)) continue;
          flush();
          continue;
        }
        if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
          flush();
          html.push("<hr>");
          continue;
        }
        const heading = /^(#{1,4})\s+(.+)$/.exec(trimmed);
        if (heading) {
          flush();
          const level = Math.min(heading[1].length + 2, 5);
          html.push(`<h${level}>${inlineReviewMarkdown(heading[2])}</h${level}>`);
          continue;
        }
        if (isReviewTableRow(trimmed) && isReviewTableSeparator(lines[i + 1] ? lines[i + 1].trim() : "")) {
          flush();
          const rows = [trimmed];
          i += 2;
          while (i < lines.length) {
            const row = lines[i].trim();
            if (!row) {
              const next = lines[i + 1] ? lines[i + 1].trim() : "";
              if (isReviewTableRow(next)) { i += 1; continue; }
              break;
            }
            if (!isReviewTableRow(row)) { i -= 1; break; }
            rows.push(row);
            i += 1;
          }
          html.push(renderReviewMarkdownTable(rows));
          continue;
        }
        const bullet = /^[-*]\s+(.+)$/.exec(trimmed);
        if (bullet) {
          if (orderedList.length) flush();
          list.push(bullet[1]);
          continue;
        }
        const ordered = /^\d+[.)]\s+(.+)$/.exec(trimmed);
        if (ordered) {
          if (list.length) flush();
          orderedList.push(ordered[1]);
          continue;
        }
        flush();
        html.push(`<p>${inlineReviewMarkdown(trimmed)}</p>`);
      }
      flush();
      let rendered = html.join("");
      codeBlocks.forEach((block, index) => {
        rendered = rendered.replace(`@@CODE_${index}@@`, block);
      });
      return rendered;
    }

    function isReviewTableRow(text) {
      const value = String(text || "").trim();
      return value.includes("|") && /^\|?.+\|.+\|?$/.test(value);
    }

    function isReviewTableSeparator(text) {
      const cells = splitReviewTableRow(text);
      return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell.trim()));
    }

    function splitReviewTableRow(text) {
      let value = String(text || "").trim();
      if (!value.includes("|")) return [];
      if (value.startsWith("|")) value = value.slice(1);
      if (value.endsWith("|")) value = value.slice(0, -1);
      return value.split("|").map((cell) => cell.trim());
    }

    function renderReviewMarkdownTable(rows) {
      const header = splitReviewTableRow(rows[0]);
      const body = rows.slice(1).map(splitReviewTableRow).filter((cells) => cells.length);
      const headerHtml = header.map((cell) => `<th>${inlineReviewMarkdown(cell)}</th>`).join("");
      const bodyHtml = body.map((cells) => {
        const padded = header.length ? cells.concat(Array(Math.max(0, header.length - cells.length)).fill("")) : cells;
        return `<tr>${padded.slice(0, Math.max(header.length, cells.length)).map((cell) => `<td>${inlineReviewMarkdown(cell)}</td>`).join("")}</tr>`;
      }).join("");
      return `<table><thead><tr>${headerHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`;
    }

    function inlineReviewMarkdown(text) {
      return String(text || "")
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.+?)\*/g, "<em>$1</em>")
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\[(S|N|R|B|W|E)(\d+)\]/g, '<span class="citation">[$1$2]</span>');
    }

    async function sendDialogueReview(kickoffMessage = "", hiddenUser = false) {
      const input = $("dialogueMessage");
      const message = kickoffMessage || (input?.value.trim() || "");
      if (!message) { setStatus("请先输入对话内容。", true); return; }
      if (!hiddenUser) dialogueReviewState.history.push({role: "user", content: message});
      if (input) input.value = "";
      dialogueReviewState.busy = true;
      renderDialogueReview();
      try {
        const result = await postJson("/api/review/dialogue", {
          topics: dialogueReviewState.topics,
          message,
          chat_history: dialogueReviewState.history,
        });
        dialogueReviewState.history.push({role: "assistant", content: result.answer || result.error || ""});
        setStatus("对话复查已更新。");
      } catch (error) {
        setStatus(error.message || String(error), true);
      } finally {
        dialogueReviewState.busy = false;
        renderDialogueReview();
        syncReviewWorkspace();
      }
    }

    function exitToSelecting() {
      mode = "selecting";
      $("donePanel").classList.add("hidden");
      render();
      syncReviewWorkspace();
    }

    function renderDone(title, message) {
      stopPolling();
      mode = "selecting";
      $("donePanel").classList.remove("hidden");
      $("donePanel").innerHTML = `<h3>${escapeHtml(title || "本轮完成")}</h3><p>${escapeHtml(message || "")}</p><div class="actions"><button id="newReview" type="button">回到选择面板</button></div>`;
      $("selectPanel").classList.add("hidden");
      $("cardPanel").classList.add("hidden");
      $("dialoguePanel").classList.add("hidden");
      syncReviewWorkspace();
    }

    function mergeCards(previous, next) {
      const byId = new Map((previous || []).map((card) => [card.id, card]));
      for (const card of next || []) {
        const local = byId.get(card.id) || {};
        const serverPrompt = card.review_prompt;
        const localPrompt = local.review_prompt;
        const review_prompt = (serverPrompt?.prompt ? serverPrompt : null)
          || (localPrompt?.prompt ? localPrompt : null)
          || serverPrompt
          || localPrompt
          || null;
        const status = review_prompt?.prompt
          ? (card.status === "failed" ? "failed" : "ready")
          : (card.status || local.status || "pending");
        byId.set(card.id, {
          ...local,
          ...card,
          status,
          review_prompt,
          review_completed: Boolean(local.review_completed),
          last_answer: local.last_answer || "",
          last_verify: local.last_verify || null,
          confirmed_weak_points: local.confirmed_weak_points || {},
        });
      }
      return (next || []).map((card) => byId.get(card.id) || card);
    }

    function moveCard(delta) {
      if (!cardReviewState.cards.length) return;
      let nextIndex = cardReviewState.index;
      for (let offset = 0; offset < cardReviewState.cards.length; offset += 1) {
        nextIndex = (nextIndex + delta + cardReviewState.cards.length) % cardReviewState.cards.length;
        if (isCardViewable(cardReviewState.cards[nextIndex])) break;
      }
      cardReviewState.index = nextIndex;
      const nextCard = currentCard();
      cardReviewState.verify = isCardCompleted(nextCard) ? null : cardReviewState.verify;
      renderCardReview();
      syncReviewWorkspace();
    }

    function isCardUsable(card) { return Boolean(card && card.review_prompt && card.review_prompt.prompt); }
    function isCardCompleted(card) { return Boolean(card && card.review_completed); }
    function isCardPending(card) { return isCardUsable(card) && !isCardCompleted(card); }
    function isCardViewable(card) { return Boolean(card && (isCardPending(card) || isCardCompleted(card) || card.status === "pending" || card.status === "failed")); }
    function currentCard() { return cardReviewState.cards[cardReviewState.index] || null; }
    function selectFirstUsableCard() {
      if (isCardViewable(currentCard())) return true;
      if (isCardPending(currentCard())) return true;
      const index = cardReviewState.cards.findIndex(isCardPending);
      if (index >= 0) { cardReviewState.index = index; return true; }
      const completedIndex = cardReviewState.cards.findIndex(isCardCompleted);
      if (completedIndex >= 0) { cardReviewState.index = completedIndex; return true; }
      return false;
    }
    function topicProgress(card) {
      const same = cardReviewState.cards.filter((item) => item.topic === card.topic);
      const pos = same.findIndex((item) => item.id === card.id) + 1;
      return `${pos}/${same.length}`;
    }
    function renderStrategyConstraints(items) {
      const constraints = Array.isArray(items) ? items.filter((item) => item && item.point) : [];
      if (!constraints.length) return "";
      return `<div class="strategy-tip">本次复习关注：${constraints.slice(0, 3).map((item) => escapeHtml(strategyTipLabel(item.point))).join(" · ")}</div>`;
    }
    function strategyTipLabel(point) {
      const text = String(point || "");
      if (text.includes("回答过短") || text.includes("推理展开") || text.includes("系统性工程思维")) return "展开推理过程";
      if (text.includes("简单归因") || text.includes("部署拓扑") || text.includes("部署场景")) return "按部署场景做判断";
      if (text.includes("边界") || text.includes("职责")) return "先区分边界";
      if (text.includes("选型") || text.includes("取舍")) return "说明工程取舍";
      if (text.includes("结构") || text.includes("框架")) return "先给结论再分层展开";
      return text.length > 18 ? `${text.slice(0, 18)}...` : text;
    }
    function firstText(items, emptyText) {
      const values = Array.isArray(items) ? items.filter(Boolean) : [];
      return values.length ? String(values[0]) : emptyText;
    }
    function renderCitations(citations) {
      if (!citations.length) return "";
      return `<div class="citations">${citations.slice(0, 8).map((item) => `<span class="citation">${escapeHtml(item.path || item.source_path || item.title || "source")}</span>`).join("")}</div>`;
    }
    async function getJson(url) {
      const response = await fetch(url);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || `${response.status} ${response.statusText}`);
      return data;
    }
    async function postJson(url, payload) {
      const response = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || `${response.status} ${response.statusText}`);
      return data;
    }
    function setBusy(busy, message="") { document.querySelectorAll("button").forEach((button) => button.disabled = busy); if (message) setStatus(message); }
    function setStatus(message, error=false) {
      const status = document.querySelector("section:not(.hidden) #status") || $("status");
      if (status) {
        status.textContent = message || "";
        status.className = `status ${error ? "error" : ""}`;
      }
    }
    function escapeHtml(value) { return String(value ?? "").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;"); }
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

    .search-columns {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      align-items: start;
    }

    .search-column {
      display: grid;
      gap: 10px;
      min-width: 0;
    }

    .column-title {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      padding: 0 2px;
    }

    .column-title strong {
      color: var(--text);
      font-size: 15px;
    }

    .user-result-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }

    .user-result-card a {
      color: var(--accent-dark);
      text-decoration: none;
      font-weight: 750;
    }

    .user-result-card a:hover {
      text-decoration: underline;
    }

    .user-result-title {
      font-size: 15px;
      font-weight: 750;
      line-height: 1.35;
      word-break: break-word;
    }

    .user-result-meta {
      color: var(--muted);
      font-size: 12px;
      margin-top: 4px;
      overflow-wrap: anywhere;
    }

    .user-result-preview {
      color: #303936;
      font-size: 13px;
      line-height: 1.58;
      margin-top: 8px;
      white-space: pre-wrap;
    }

    .empty-column {
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 18px 12px;
      color: var(--muted);
      background: rgba(255,255,255,.62);
      font-size: 13px;
      text-align: center;
    }

    .debug-only {
      display: none;
    }

    .debug-mode .debug-only {
      display: grid;
    }

    .debug-mode .stage-tabs {
      display: flex;
    }

    .debug-mode .search-columns {
      display: none;
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

      .search-columns {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main>
    <div id="status" class="status page-status">就绪</div>
    <div class="topbar">
      <div>
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

      <details id="advancedOptions">
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
    <section id="userResults" class="search-columns"></section>
    <section id="stageTabs" class="stage-tabs debug-only"></section>
    <section id="results" class="results debug-only"></section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    let currentSearchData = null;
    let currentWikiReport = null;
    let activeStageKey = "final";
    const searchParams = new URLSearchParams(window.location.search);
    const debugMode = searchParams.get("debug") === "true";

    $("searchBtn").addEventListener("click", runSearch);
    $("rerankerType").addEventListener("change", updateRerankerModelDefault);
    $("query").addEventListener("keydown", (event) => {
      if (event.key === "Enter") runSearch();
    });
    initializeSearchPage();

    function initializeSearchPage() {
      const initialQuery = searchParams.get("q") || "";
      if (debugMode) {
        document.body.classList.add("debug-mode");
        $("advancedOptions").open = true;
      } else {
        $("advancedOptions").open = false;
        $("userResults").innerHTML = renderEmptySearchState();
      }
      if (initialQuery) {
        $("query").value = initialQuery;
        runSearch();
      }
    }

    async function runSearch() {
      const query = $("query").value.trim();
      if (!query) return;

      setBusy(true);
      $("meta").innerHTML = "";
      $("userResults").innerHTML = "";
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
        include_debug: debugMode
      };

      try {
        const [data, wikiReport] = await Promise.all([
          fetchSearchResults(payload),
          debugMode ? Promise.resolve(null) : loadWikiReport()
        ]);
        currentSearchData = data;
        activeStageKey = "final";
        if (debugMode) {
          renderMeta(data);
          renderStageTabs(data);
          renderActiveStage();
        } else {
          renderUserSearchResults(data, wikiReport, query);
        }
      } catch (error) {
        $("meta").innerHTML = `<div class="meta-box error">${escapeHtml(error.message)}</div>`;
      } finally {
        setBusy(false);
      }
    }

    async function fetchSearchResults(payload) {
      const response = await fetch("/api/search", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "search failed");
      }
      return data;
    }

    async function loadWikiReport() {
      if (currentWikiReport) return currentWikiReport;
      const response = await fetch("/api/wiki/report");
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "wiki report failed");
      currentWikiReport = data;
      return data;
    }

    function renderEmptySearchState() {
      return `
        ${renderColumn("笔记片段", "Hybrid 检索结果", [])}
        ${renderColumn("Wiki 页面", "已生成主题页", [])}
        ${renderColumn("Tag 匹配", "主题候选", [])}
      `;
    }

    function renderUserSearchResults(data, wikiReport, query) {
      const noteItems = (data.results || []).slice(0, numberValue("topK")).map(renderNoteCard);
      const rows = wikiRows(wikiReport, query);
      const wikiItems = rows.filter((row) => row.wiki_exists).slice(0, 8).map(renderWikiCard);
      const tagItems = rows.slice(0, 12).map(renderTagCard);
      $("meta").innerHTML = `<div class="meta-box"><strong>搜索完成</strong><span>${data.results.length} 个笔记片段 · ${wikiItems.length} 个 Wiki 页面 · ${tagItems.length} 个 Tag 匹配 · ${data.elapsed_ms} ms</span></div>`;
      $("userResults").innerHTML = `
        ${renderColumn("笔记片段", `${noteItems.length} 个结果`, noteItems)}
        ${renderColumn("Wiki 页面", `${wikiItems.length} 个结果`, wikiItems)}
        ${renderColumn("Tag 匹配", `${tagItems.length} 个结果`, tagItems)}
      `;
    }

    function renderColumn(title, subtitle, items) {
      const body = items.length ? items.join("") : `<div class="empty-column">输入关键词后显示匹配结果</div>`;
      return `
        <section class="search-column">
          <div class="column-title"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(subtitle)}</span></div>
          ${body}
        </section>
      `;
    }

    function renderNoteCard(item) {
      const heading = item.heading ? `<div class="user-result-meta">${escapeHtml(item.heading)}</div>` : "";
      return `
        <article class="user-result-card">
          <div class="user-result-title">${escapeHtml(item.title || item.note_path || "笔记")}</div>
          <div class="user-result-meta">${escapeHtml(item.note_path || "")}</div>
          ${heading}
          <div class="user-result-preview">${escapeHtml(item.preview || item.text || "")}</div>
        </article>
      `;
    }

    function renderWikiCard(row) {
      const href = `/wiki?tag=${encodeURIComponent(row.tag || "")}`;
      return `
        <article class="user-result-card">
          <a class="user-result-title" href="${href}">${escapeHtml(formatTagTitle(row.tag))}</a>
          <div class="user-result-meta">${escapeHtml(row.wiki_path || "")}</div>
          <div class="user-result-preview">${escapeHtml(row.wiki_preview || "已生成 Wiki 页面。")}</div>
        </article>
      `;
    }

    function renderTagCard(row) {
      const href = row.wiki_exists ? `/wiki?tag=${encodeURIComponent(row.tag || "")}` : "";
      const title = escapeHtml(formatTagTitle(row.tag));
      const titleHtml = href ? `<a class="user-result-title" href="${href}">${title}</a>` : `<div class="user-result-title">${title}</div>`;
      const policy = row.wiki_policy ? ` · ${row.wiki_policy}` : "";
      const status = row.wiki_exists ? "已生成" : (row.eligible ? "未合成" : "跳过");
      return `
        <article class="user-result-card">
          ${titleHtml}
          <div class="user-result-meta">${Number(row.note_count || 0)} 篇笔记 · ${status}${escapeHtml(policy)}</div>
          <div class="user-result-preview">${escapeHtml(row.wiki_preview || row.wiki_path || "匹配到主题候选。")}</div>
        </article>
      `;
    }

    function wikiRows(report, query) {
      const needle = query.trim().toLowerCase();
      if (!report || !needle) return [];
      return (report.tag_rows || [])
        .filter((row) => rowMatches(row, needle))
        .sort((a, b) => Number(b.wiki_exists) - Number(a.wiki_exists) || Number(b.note_count || 0) - Number(a.note_count || 0));
    }

    function rowMatches(row, needle) {
      const haystack = [
        row.tag,
        row.wiki_path,
        row.wiki_preview,
        row.wiki_policy
      ].map((value) => String(value || "").toLowerCase()).join("\n");
      return haystack.includes(needle);
    }

    function formatTagTitle(tag) {
      return String(tag || "").split("/").filter(Boolean).join(" / ") || "未命名主题";
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
      const status = $("status");
      if (status) status.textContent = isBusy ? "检索中" : "就绪";
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
    main.chat-page {
      width: min(980px, calc(100vw - 28px));
      margin: 0 auto;
      height: calc(100dvh - 68px);
      max-height: calc(100dvh - 68px);
      min-height: 0;
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding: 12px 0 16px;
    }
    .app-main:has(.chat-page) .app-content {
      min-height: 0;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .chat-scroll {
      flex: 1 1 auto;
      min-height: 0;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .chat-dock {
      flex: 0 0 auto;
      display: flex;
      flex-direction: column;
      gap: 0;
    }
    .topbar { display: flex; align-items: center; justify-content: space-between; gap: 14px; }
    .topbar h1 { margin: 0; font-size: 22px; line-height: 1.2; }
    .topnav { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    .topnav a { color: var(--accent-dark); text-decoration: none; font-size: 14px; font-weight: 600; }
    .status { color: var(--muted); font-size: 13px; margin-top: 4px; }
    .messages { flex: 1 0 auto; min-height: 0; overflow: visible; display: grid; align-content: start; gap: 12px; padding: 2px; }
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
    .composer { border: 1px solid var(--line); border-radius: 0 0 8px 8px; background: var(--panel); padding: 12px; display: grid; gap: 10px; }
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
    .session-context { display: none; flex: 0 0 auto; min-height: 0; }
    .session-context.has-content { display: block; }
    .session-context .session-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }
    .session-context .session-panel summary {
      cursor: pointer;
      color: var(--accent-dark);
      font-weight: 800;
      padding: 10px 12px;
      list-style: none;
    }
    .session-context .session-panel summary::-webkit-details-marker { display: none; }
    .session-context .session-panel-body {
      max-height: min(32vh, 280px);
      overflow-x: hidden;
      overflow-y: auto;
      overscroll-behavior: contain;
      -webkit-overflow-scrolling: touch;
      padding: 0 12px 10px;
    }
    .review-notice {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border: 1px solid #d7e5df;
      border-radius: 8px;
      background: #f4fbf8;
      padding: 10px 12px;
      color: #263a35;
      font-size: 14px;
    }
    .review-notice[hidden] { display: none; }
    .review-notice strong { color: var(--accent-dark); }
    .review-notice-actions { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .review-notice-actions button { padding: 6px 10px; font-size: 12px; }
    .interview-direction-bar {
      display: none;
      align-items: center;
      gap: 8px;
      min-height: 36px;
      padding: 6px 12px;
      border: 1px solid var(--line);
      border-bottom: 0;
      border-radius: 8px 8px 0 0;
      background: rgba(255,255,255,.96);
      font-size: 12px;
      color: var(--muted);
      overflow: visible;
      position: relative;
      z-index: 6;
    }
    .interview-direction-bar.visible { display: flex; }
    .interview-direction-bar .dir-label {
      color: var(--accent-dark);
      font-weight: 800;
      white-space: nowrap;
      flex: 0 0 auto;
    }
    .interview-direction-bar .dir-value {
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      min-width: 0;
      flex: 1 1 auto;
    }
    .interview-direction-bar .dir-actions {
      display: flex;
      gap: 6px;
      flex-wrap: nowrap;
      overflow-x: auto;
      flex: 1 1 auto;
      min-width: 0;
      scrollbar-width: none;
    }
    .interview-direction-bar .dir-actions::-webkit-scrollbar { display: none; }
    .interview-direction-bar .dir-chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      color: var(--accent-dark);
      padding: 3px 10px;
      font-size: 11px;
      font-weight: 700;
      cursor: pointer;
      white-space: nowrap;
      flex: 0 0 auto;
    }
    .interview-direction-bar .dir-chip:hover { border-color: var(--accent); background: #f8fbfa; }
    .interview-direction-bar .dir-chip.active {
      border-color: var(--accent);
      background: #e7f3f1;
      color: var(--accent-dark);
    }
    .interview-direction-bar .dir-meta {
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      min-width: 0;
      flex: 0 1 auto;
      max-width: 42%;
    }
    .chat-dock .composer.no-direction-rail {
      border-radius: 8px;
      border-top: 1px solid var(--line);
    }
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
    .turn-citations { margin-top: 10px; border-top: 1px solid var(--line); padding-top: 8px; }
    .turn-citations summary { cursor: pointer; color: var(--muted); font-size: 12px; font-weight: 700; }
    .turn-citation-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
    .turn-citation-chip { display: inline-flex; align-items: center; gap: 4px; border: 1px solid var(--line); border-radius: 999px; background: #f8fbfa; padding: 3px 10px; font-size: 12px; color: #334155; max-width: 100%; }
    .turn-citation-chip .note-name { font-weight: 700; color: var(--accent-dark); }
    .turn-citation-chip .note-section { color: var(--muted); }
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
    @media (max-width: 720px) { main.chat-page { width: min(100vw - 18px, 980px); padding: 10px 0 12px; height: calc(100dvh - 60px); max-height: calc(100dvh - 60px); } .topbar { display: block; } .message.user, .message.assistant { width: 100%; } .controls, .context-grid { display: grid; grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main class="chat-page">
    <div id="status" class="status page-status">&#23601;&#32490;</div>
    <header>
      <div class="topbar">
        <div>
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

    <section id="reviewNotice" class="review-notice" hidden></section>

    <div class="chat-scroll">
      <section id="sessionContext" class="session-context"></section>
      <section id="messages" class="messages">
        <article class="message assistant"><div class="role">Assistant</div><div class="answer">已准备。可以直接提问，或切换到面试模式开始练习。</div></article>
      </section>
    </div>

    <div class="chat-dock">
      <div id="interviewDirectionBar" class="interview-direction-bar" hidden aria-live="polite"></div>
      <section id="composer" class="composer no-direction-rail">
      <div class="controls">
        <label>模式<select id="chatMode"><option value="answer" selected>问答</option><option value="interview">面试</option></select></label>
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
    </div>
  </main>

  <div id="historyModal" class="modal-backdrop" aria-hidden="true">
    <aside class="modal" role="dialog" aria-modal="true" aria-label="面试历史">
      <div class="modal-head">
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
    let currentTurnCitations = [];
    let currentPayload = null;
    let currentInterviewPlan = null;
    let currentInterviewPlanSignature = "";
    let currentInterviewState = null;
    let currentInterviewSessionId = "";
    let currentAnswerSessionId = "";
    let currentConversationSignature = "";
    let lastContextItems = [];
    let sessionContextState = {scope: null, stats: null, references: [], interviewPlan: null, profileDebug: null, trace: []};
    let currentTurnTrace = {directorNoteInjected: false};
    let currentProcess = createAgentProcessState();
    const activeAnswerTaskKey = "knowledge_agent.active_answer_task_id";
    let pendingAnswerTurn = null;
    let sessionContextPanelOpen = false;

    (function initChatWorkspace() {
      const ws = window.KnowledgeAgentWorkspace;
      if (ws) {
        ws.hydrate();
        ws.migrateLegacyChat();
        const chat = ws.get("chat") || {};
        if (chat.activeInterviewSessionId) currentInterviewSessionId = chat.activeInterviewSessionId;
        if (chat.activeAnswerSessionId) currentAnswerSessionId = chat.activeAnswerSessionId;
        if (chat.scope) {
          if (chat.scope.scopeType) $("scopeType").value = chat.scope.scopeType;
          if (typeof chat.scope.scopeValue === "string") $("scopeValue").value = chat.scope.scopeValue;
          if (typeof chat.scope.strictEvidence === "boolean") $("strictEvidence").checked = chat.scope.strictEvidence;
        }
        if (chat.pendingTurn) pendingAnswerTurn = {...chat.pendingTurn};
        if (chat.activeAnswerTaskId) sessionStorage.setItem(activeAnswerTaskKey, chat.activeAnswerTaskId);
      }
    })();

    applyInitialChatMode();
    currentConversationSignature = conversationSignature();

    $("sendBtn").addEventListener("click", sendMessage);
    $("query").addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) sendMessage();
    });
    $("newConversation").addEventListener("click", startNewConversation);
    $("historyBtn").addEventListener("click", openHistory);
    $("closeHistory").addEventListener("click", closeHistory);
    $("historyModal").addEventListener("click", (event) => { if (event.target === $("historyModal")) closeHistory(); });
    $("endInterview").addEventListener("click", endInterviewSession);
    $("chatMode").addEventListener("change", handleChatModeChange);
    ["scopeType", "scopeValue", "strictEvidence"].forEach((id) => $(id).addEventListener("change", resetConversationIfNeeded));
    restoreActiveInterviewSession();
    restoreAnswerWorkspace();
    restoreActiveAnswerTask();
    loadReviewNotice();
    window.addEventListener("beforeunload", () => {
      if ($("chatMode").value === "answer") syncChatWorkspace();
    });

    function buildChatWorkspaceSlice(overrides = {}) {
      return {
        version: 1,
        activeInterviewSessionId: currentInterviewSessionId || "",
        activeAnswerSessionId: currentAnswerSessionId || "",
        scope: {
          scopeType: $("scopeType").value,
          scopeValue: $("scopeValue").value,
          strictEvidence: $("strictEvidence").checked,
        },
        pendingTurn: pendingAnswerTurn ? {...pendingAnswerTurn} : null,
        activeAnswerTaskId: sessionStorage.getItem(activeAnswerTaskKey) || pendingAnswerTurn?.taskId || "",
        ...overrides,
      };
    }

    function syncChatWorkspace(overrides = {}) {
      const ws = window.KnowledgeAgentWorkspace;
      if (!ws) return;
      const slice = buildChatWorkspaceSlice(overrides);
      if (slice.activeAnswerTaskId) {
        sessionStorage.setItem(activeAnswerTaskKey, slice.activeAnswerTaskId);
      } else {
        sessionStorage.removeItem(activeAnswerTaskKey);
      }
      ws.replace("chat", slice);
      ws.persist();
    }

    function applyInitialChatMode() {
      const params = new URLSearchParams(window.location.search);
      const mode = params.get("mode") || "answer";
      if (mode === "study") {
        window.location.replace("/review");
        return;
      }
      $("chatMode").value = mode === "interview" ? "interview" : "answer";
      updateChatNavActive();
    }

    function handleChatModeChange() {
      const mode = $("chatMode").value;
      const target = mode === "interview" ? "/?mode=interview" : "/?mode=answer";
      window.history.replaceState(null, "", target);
      updateChatNavActive();
      resetConversationIfNeeded();
    }

    function updateChatNavActive() {
      const mode = $("chatMode").value;
      document.querySelectorAll(".app-nav-link").forEach((link) => {
        const id = link.dataset.navId || "";
        if (id === "chat-interview") {
          link.classList.toggle("active", mode === "interview");
        } else if (id === "chat-answer") {
          link.classList.toggle("active", mode !== "interview");
        } else if (["review", "topics", "wiki", "search", "wiki-admin", "search-debug", "settings", "organize"].includes(id)) {
          link.classList.remove("active");
        }
      });
    }

    function resetConversationIfNeeded() {
      const signature = conversationSignature();
      if (signature === currentConversationSignature) return;
      chatHistory = [];
      currentInterviewPlan = null;
      currentInterviewPlanSignature = "";
      currentInterviewState = null;
      currentInterviewSessionId = "";
      currentAnswerSessionId = "";
      syncChatWorkspace({activeInterviewSessionId: "", activeAnswerSessionId: ""});
      lastContextItems = [];
      sessionContextPanelOpen = false;
      resetSessionContext();
      currentConversationSignature = signature;
      updateSessionLabel();
      $("starters").innerHTML = "";
      renderInterviewDirectionBar();
    }

    async function sendMessage() {
      const query = $("query").value.trim();
      if (!query) return;
      resetConversationIfNeeded();
      if ($("chatMode").value === "answer") {
        pendingAnswerTurn = {query, taskId: null, status: "running"};
        syncChatWorkspace();
      }
      appendUserMessage(query);
      $("query").value = "";
      currentAssistant = appendAssistantMessage("");
      currentAssistantText = "";
      currentTurnCitations = [];
      currentPayload = {router: null, retrieval: null, retrievalStatus: null, interviewPlan: null, profileDebug: null, context: null, answer: null, done: null, errors: []};
      currentProcess = createAgentProcessState();
      renderAgentProcess();
      setBusy(true, "Starting...");

      const body = buildAgentRequestBody(query);
      currentTurnTrace = {directorNoteInjected: shouldInjectDirectorNote(body.interview_state)};
      let turnIds = null;

      try {
        if ($("chatMode").value === "interview") {
          await ensureInterviewSession();
          body.session_id = currentInterviewSessionId;
          if (currentTurnTrace.directorNoteInjected) {
            recordSessionTrace("director_note_injected", `Director Note injected (follow_up=${body.interview_state?.follow_up_count || 0})`, {
              interview_state: body.interview_state || {}
            });
          }
        } else if ($("chatMode").value === "answer") {
          await ensureAnswerSession();
          body.session_id = currentAnswerSessionId;
        }
        const streamResult = await streamAgentAnswerWithRetry(body);
        turnIds = streamResult?.turnIds || null;
        const answerText = currentAssistantText.trim();
        if (answerText) {
          const assistantNode = currentAssistant;
          renderAssistantExtras();
          const shouldReview = $("chatMode").value === "interview" && shouldRequestTurnSummary(query, answerText);
          if ($("chatMode").value === "interview") {
            chatHistory.push({role:"user", content:query});
            chatHistory.push({role:"assistant", content:answerText});
            trimChatHistory();
            const stateChange = isServerInterviewState() ? null : updateInterviewState(query, answerText);
            recordInterviewStateTrace(stateChange);
            renderInterviewDirectionBar();
            if (shouldReview && turnIds) runTurnReviewInBackground(answerText, turnIds, assistantNode);
          } else {
            pendingAnswerTurn = null;
            await reloadAnswerSessionFromServer();
          }
        }
        setBusy(false, "就绪");
      } catch (error) {
        currentPayload.errors.push(error.message);
        if ($("chatMode").value === "answer" && pendingAnswerTurn) {
          pendingAnswerTurn.status = "failed";
          pendingAnswerTurn.error = error.message;
          syncChatWorkspace();
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
        session_id: $("chatMode").value === "interview" ? currentInterviewSessionId : ($("chatMode").value === "answer" ? currentAnswerSessionId : null),
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
      if (sessionContextPanelOpen) renderSessionContext();
    }

    async function streamAgentAnswerWithRetry(body, options = {}) {
      if (shouldUseDecoupledAgentRuns(body.chat_mode)) {
        return streamDecoupledAgentRun(body, options);
      }
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
          return {turnIds: null, taskId: null};
        } catch (error) {
          lastError = error;
          const retryable = error.retryable !== false;
          const noOutputYet = !currentAssistantText.trim();
          if (attempt === 0 && retryable && noOutputYet) continue;
          throw error;
        }
      }
      if (lastError) throw lastError;
      return {turnIds: null, taskId: null};
    }

    function shouldUseDecoupledAgentRuns(chatMode) {
      return chatMode === "interview" || chatMode === "answer";
    }

    async function streamDecoupledAgentRun(body, options = {}) {
      let lastError = null;
      for (let attempt = 0; attempt < 2; attempt += 1) {
        try {
          if (attempt > 0) {
            setBusy(true, "Retrying...");
            await delay(1200);
          }
          const payload = {...body};
          if (payload.chat_mode === "interview") {
            await ensureInterviewSession();
            payload.session_id = currentInterviewSessionId;
            payload.source_note_paths = sourceNotePaths();
          } else if (payload.chat_mode === "answer") {
            await ensureAnswerSession();
            payload.session_id = currentAnswerSessionId;
          }
          if (options.assistantMessageId) payload.assistant_message_id = options.assistantMessageId;
          const response = await fetch("/api/agent/runs", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
          const data = await response.json().catch(() => ({}));
          if (!response.ok) {
            const error = new Error(data.detail || `request failed: ${response.status}`);
            error.retryable = true;
            throw error;
          }
          if (payload.chat_mode === "answer" && data.task_id) {
            if (pendingAnswerTurn) pendingAnswerTurn.taskId = data.task_id;
            syncChatWorkspace({activeAnswerTaskId: data.task_id, pendingTurn: pendingAnswerTurn});
          }
          const streamResponse = await fetch(`/api/tasks/${encodeURIComponent(data.task_id)}/stream`);
          if (!streamResponse.ok || !streamResponse.body) throw new Error(`task stream failed: ${streamResponse.status}`);
          await readSse(streamResponse.body);
          if (payload.chat_mode === "answer" && data.task_id) {
            const taskResponse = await fetch(`/api/tasks/${encodeURIComponent(data.task_id)}`);
            const task = await taskResponse.json().catch(() => ({}));
            if ((task.status || "") === "succeeded") sessionStorage.removeItem(activeAnswerTaskKey);
          }
          const turnIds = data.user_message_id && data.assistant_message_id
            ? {user: {id: data.user_message_id}, assistant: {id: data.assistant_message_id}}
            : null;
          return {turnIds, taskId: data.task_id || null};
        } catch (error) {
          lastError = error;
          const retryable = error.retryable !== false;
          const noOutputYet = !currentAssistantText.trim();
          if (attempt === 0 && retryable && noOutputYet) continue;
          throw error;
        }
      }
      if (lastError) throw lastError;
      return {turnIds: null, taskId: null};
    }

    async function restoreActiveAnswerTask() {
      if ($("chatMode").value !== "answer") return;
      const taskId = sessionStorage.getItem(activeAnswerTaskKey) || pendingAnswerTurn?.taskId || null;
      if (!taskId && !pendingAnswerTurn?.query) return;
      ensurePendingAnswerUserMessage();
      if (!taskId) {
        if (pendingAnswerTurn?.status === "running") setBusy(true, "查询中...");
        return;
      }
      try {
        const taskResponse = await fetch(`/api/tasks/${encodeURIComponent(taskId)}`);
        if (!taskResponse.ok) {
          sessionStorage.removeItem(activeAnswerTaskKey);
          return;
        }
        const task = await taskResponse.json();
        const status = task.status || "";
        if (status === "succeeded") {
          finalizePendingAnswerTurn(task);
          sessionStorage.removeItem(activeAnswerTaskKey);
          setBusy(false, "就绪");
          return;
        }
        if (status === "failed") {
          showPendingAnswerFailure(task.error || "assistant generation failed");
          sessionStorage.removeItem(activeAnswerTaskKey);
          setBusy(false, "Error");
          return;
        }
        if (!["queued", "running"].includes(status)) {
          sessionStorage.removeItem(activeAnswerTaskKey);
          return;
        }
        const lastMessage = messages.querySelector(".message:last-child");
        if (lastMessage && lastMessage.classList.contains("assistant")) {
          currentAssistant = lastMessage;
          currentAssistantText = pendingAnswerTurn?.partialAssistantText || "";
          if (currentAssistantText) {
            currentAssistant.querySelector(".answer").innerHTML = renderMarkdown(currentAssistantText);
          }
        } else {
          currentAssistant = appendAssistantMessage(pendingAnswerTurn?.partialAssistantText || "");
          currentAssistantText = pendingAnswerTurn?.partialAssistantText || "";
        }
        currentPayload = {router: null, retrieval: null, retrievalStatus: null, interviewPlan: null, profileDebug: null, context: null, answer: null, done: null, errors: []};
        currentProcess = createAgentProcessState();
        renderAgentProcess();
        setBusy(true, "恢复生成中...");
        const streamResponse = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/stream`);
        if (!streamResponse.ok || !streamResponse.body) throw new Error(`task stream failed: ${streamResponse.status}`);
        await readSse(streamResponse.body);
        const refreshed = await fetch(`/api/tasks/${encodeURIComponent(taskId)}`);
        const refreshedTask = refreshed.ok ? await refreshed.json() : task;
        if ((refreshedTask.status || "") === "succeeded") {
          finalizePendingAnswerTurn(refreshedTask);
        } else if ((refreshedTask.status || "") === "failed") {
          showPendingAnswerFailure(refreshedTask.error || "assistant generation failed");
        } else if (currentAssistantText.trim()) {
          finalizePendingAnswerTurn({result: {answer: currentAssistantText}});
        }
        sessionStorage.removeItem(activeAnswerTaskKey);
        setBusy(false, "就绪");
      } catch (error) {
        if (pendingAnswerTurn) {
          pendingAnswerTurn.status = "failed";
          pendingAnswerTurn.error = error.message || "restore failed";
          syncChatWorkspace();
        }
        sessionStorage.removeItem(activeAnswerTaskKey);
        setBusy(false, "Error");
      }
    }

    function ensurePendingAnswerUserMessage() {
      if (!pendingAnswerTurn?.query) return;
      const lastUser = [...chatHistory].reverse().find((item) => item.role === "user");
      if (lastUser && lastUser.content === pendingAnswerTurn.query) return;
      const nodes = messages.querySelectorAll(".message.user .answer");
      const lastDomUser = nodes.length ? nodes[nodes.length - 1] : null;
      if (lastDomUser && lastDomUser.textContent.trim() === pendingAnswerTurn.query.trim()) return;
      appendUserMessage(pendingAnswerTurn.query);
    }

    async function finalizePendingAnswerTurn(task) {
      pendingAnswerTurn = null;
      syncChatWorkspace({pendingTurn: null, activeAnswerTaskId: ""});
      await reloadAnswerSessionFromServer();
      if (!messages.querySelector(".message.user")) {
        const query = task?.query || "";
        const answerText = String(task?.result?.answer || currentAssistantText || "").trim();
        if (query) chatHistory.push({role: "user", content: query});
        if (answerText) chatHistory.push({role: "assistant", content: answerText});
        trimChatHistory();
        renderAnswerConversation();
      }
    }

    function showPendingAnswerFailure(message) {
      ensurePendingAnswerUserMessage();
      currentAssistant = appendAssistantMessage("");
      currentAssistantText = "";
      if (currentAssistant) {
        currentAssistant.querySelector(".answer").innerHTML = `<p class="error">${escapeHtml(message)}</p>`;
      }
      if (pendingAnswerTurn) {
        pendingAnswerTurn.status = "failed";
        pendingAnswerTurn.error = message;
        syncChatWorkspace();
      }
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
      currentTurnCitations = [];
      currentPayload = {router: null, retrieval: null, retrievalStatus: null, interviewPlan: currentInterviewPlan, profileDebug: null, context: null, answer: null, done: null, errors: []};
      currentProcess = createAgentProcessState();
      assistantNode.querySelectorAll(".retry-answer, .assistant-extra, details.review").forEach((node) => node.remove());
      assistantNode.querySelector(".answer").innerHTML = "";
      renderAgentProcess();
      setBusy(true, "Retrying...");
      try {
        const body = buildAgentRequestBody(query);
        if ($("chatMode").value === "interview") {
          body.session_id = currentInterviewSessionId;
          currentTurnTrace = {directorNoteInjected: shouldInjectDirectorNote(body.interview_state)};
          if (currentTurnTrace.directorNoteInjected) {
            recordSessionTrace("director_note_injected", `Director Note injected (follow_up=${body.interview_state?.follow_up_count || 0})`, {
              interview_state: body.interview_state || {},
              retry: true
            });
          }
        } else if ($("chatMode").value === "answer") {
          await ensureAnswerSession();
          body.session_id = currentAnswerSessionId;
        }
        await streamAgentAnswerWithRetry(body, {assistantMessageId: turnIds.assistant.id});
        const answerText = currentAssistantText.trim();
        if (!answerText) throw new Error("empty answer");
        renderAssistantExtras();
        const shouldReview = $("chatMode").value === "interview" && shouldRequestTurnSummary(query, answerText);
        if ($("chatMode").value === "interview") {
          chatHistory.push({role:"user", content:query});
          chatHistory.push({role:"assistant", content:answerText});
          trimChatHistory();
          const stateChange = isServerInterviewState() ? null : updateInterviewState(query, answerText);
          recordInterviewStateTrace(stateChange);
          renderInterviewDirectionBar();
          if (shouldReview) runTurnReviewInBackground(answerText, turnIds, assistantNode);
        } else {
          await reloadAnswerSessionFromServer();
        }
        if ($("chatMode").value === "answer") syncChatWorkspace();
        setBusy(false, "就绪");
      } catch (error) {
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
          renderInterviewDirectionBar();
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
        renderInterviewDirectionBar();
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
        const telemetryCitations = data?.telemetry?.citations;
        if (Array.isArray(telemetryCitations)) currentTurnCitations = telemetryCitations;
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
      const stoppedReason = String(data?.stopped_reason || "");
      if (stoppedReason && stoppedReason !== "final") {
        const stoppedPayload = {
          reason: stoppedReason,
          message: String(data?.error || stopReasonLabel(stoppedReason) || "Agent stopped"),
          trace_path: data?.trace_path || "",
          recoverable: data?.recoverable !== false,
          partial: Boolean(data?.partial)
        };
        if (!currentProcess.stopped) handleAgentStopped(stoppedPayload);
        currentProcess.done = true;
        currentProcess.phase = "done";
        currentProcess.items = currentProcess.items.map((item) => item.status === "active" ? {...item, status: "done"} : item);
        currentProcess.active = null;
        renderAgentProcess();
        return;
      }
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
        return [weak ? `${weak} 条相关弱项` : "", due ? `${due} 条建议复查` : ""].filter(Boolean).join("，") || "已读取相关画像提示";
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

    function bindInterviewDirectionTopicButtons(container) {
      if (!container) return;
      container.querySelectorAll("[data-topic]").forEach((button) => {
        button.addEventListener("click", async (event) => {
          event.stopPropagation();
          try {
            await selectInterviewTopic(button.dataset.topic || "");
          } catch (error) {
            setBusy(false, "Error");
            alert(error.message || "切换主题失败");
          }
        });
      });
    }

    function syncComposerDirectionRail(visible) {
      const composer = $("composer");
      if (!composer) return;
      composer.classList.toggle("no-direction-rail", !visible);
    }

    function renderInterviewTopicChips(topics, currentTopic) {
      return topics.map((topic) => {
        const name = String(topic.name || "").trim();
        if (!name) return "";
        const active = name === currentTopic;
        return `<button type="button" class="dir-chip${active ? " active" : ""}" data-topic="${escapeHtml(name)}">${escapeHtml(name)}</button>`;
      }).join("");
    }

    function interviewDirectionMeta(state) {
      const snapshot = state || {};
      const parts = [];
      if (snapshot.current_layer_name) parts.push(String(snapshot.current_layer_name));
      if (snapshot.topic_phase === "closing") parts.push("收尾");
      return parts.join(" · ");
    }

    function renderInterviewDirectionBar() {
      const bar = $("interviewDirectionBar");
      if (!bar) return;
      if ($("chatMode").value !== "interview") {
        bar.hidden = true;
        bar.classList.remove("visible");
        bar.innerHTML = "";
        syncComposerDirectionRail(false);
        return;
      }
      const plan = currentInterviewPlan || (sessionContextState.interviewPlan && sessionContextState.interviewPlan.plan) || null;
      const topics = Array.isArray(plan?.topics) ? plan.topics : [];
      const state = currentInterviewState || {};
      const currentTopic = String(state.current_topic || "").trim();
      const meta = interviewDirectionMeta(state);

      if (topics.length) {
        bar.hidden = false;
        bar.classList.add("visible");
        syncComposerDirectionRail(true);
        bar.innerHTML = `<span class="dir-label">面试方向</span>${meta ? `<span class="dir-meta">${escapeHtml(meta)}</span>` : ""}<div class="dir-actions">${renderInterviewTopicChips(topics, currentTopic)}</div>`;
        bindInterviewDirectionTopicButtons(bar);
        return;
      }

      bar.hidden = true;
      bar.classList.remove("visible");
      bar.innerHTML = "";
      syncComposerDirectionRail(false);
    }

    function renderInterviewPlan(plan) {
      if (plan) currentInterviewPlan = plan;
      renderInterviewDirectionBar();
    }

    async function selectInterviewTopic(topic) {
      const selected = String(topic || "").trim();
      if (!selected) return;
      const previousTopic = String(currentInterviewState?.current_topic || "").trim();
      const awaitingSelection = !previousTopic || currentInterviewState?.topic_phase === "awaiting_selection";
      if (!awaitingSelection && selected === previousTopic) return;
      const sessionId = await ensureInterviewSession();
      const response = await fetch(`/api/interview/sessions/${encodeURIComponent(sessionId)}/select-topic`, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({
          topic: selected,
          reason: awaitingSelection ? "user selected topic from interview plan UI" : "user switched topic from interview direction bar",
          source: "ui",
          interview_plan: currentInterviewPlan
        })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "topic selection failed");
      currentInterviewState = data.interview_state || data.session?.interview_state || currentInterviewState;
      recordSessionTrace("topic_selected", `topic selected: ${selected}`, {interview_state: currentInterviewState});
      renderInterviewDirectionBar();
      if (awaitingSelection) {
        $("query").value = `我想从${selected}开始`;
      } else if (selected !== previousTopic) {
        $("query").value = `我想切换到${selected}继续面试`;
      }
      $("query").focus();
    }

    function renderAssistantExtras() {
      if (!currentAssistant) return;
      currentAssistant.querySelectorAll(".assistant-extra").forEach((node) => node.remove());
      const parts = [];
      if ($("chatMode").value === "interview" && currentTurnCitations.length) {
        parts.push(renderTurnCitations(currentTurnCitations));
      }
      parts.push(`<div class="assistant-extra">${renderDebug(buildTurnDebug())}</div>`);
      currentAssistant.insertAdjacentHTML("beforeend", parts.join(""));
    }

    function renderTurnCitations(citations) {
      const items = Array.isArray(citations) ? citations : [];
      if (!items.length) return "";
      const chips = items.map(renderTurnCitationChip).join("");
      return `<details class="turn-citations assistant-extra" open><summary>来源 (${items.length})</summary><div class="turn-citation-list">${chips}</div></details>`;
    }

    function renderTurnCitationChip(citation) {
      const path = String(citation?.path || "").trim();
      const noteName = path ? path.split("/").pop() || path : "笔记";
      const headingPath = Array.isArray(citation?.heading_path) ? citation.heading_path.filter(Boolean) : [];
      const section = headingPath.length ? headingPath.join(" › ") : "";
      const lineStart = citation?.line_start;
      const lineEnd = citation?.line_end;
      let lineLabel = "";
      if (lineStart && lineEnd && lineStart !== lineEnd) lineLabel = `L${lineStart}-${lineEnd}`;
      else if (lineStart) lineLabel = `L${lineStart}`;
      const sectionHtml = section ? `<span class="note-section"> · ${escapeHtml(section)}</span>` : "";
      const lineHtml = lineLabel ? `<span class="note-section"> · ${escapeHtml(lineLabel)}</span>` : "";
      const title = escapeHtml([path, section, lineLabel].filter(Boolean).join(" · "));
      return `<span class="turn-citation-chip" title="${title}"><span class="note-name">${escapeHtml(noteName)}</span>${sectionHtml}${lineHtml}</span>`;
    }

    function resetSessionContext() {
      sessionContextState = {scope: null, stats: null, references: [], interviewPlan: null, profileDebug: null, trace: []};
      renderSessionContext();
      renderInterviewDirectionBar();
    }

    function renderSessionContext() {
      const node = $("sessionContext");
      if (!node) return;
      const existingPanel = node.querySelector(".session-panel");
      if (existingPanel) sessionContextPanelOpen = existingPanel.open;
      const hasContent = sessionContextState.scope || sessionContextState.interviewPlan || sessionContextState.profileDebug || (sessionContextState.references || []).length || (sessionContextState.trace || []).length;
      if (!hasContent) {
        node.className = "session-context";
        node.innerHTML = "";
        sessionContextPanelOpen = false;
        return;
      }
      node.className = "session-context has-content";
      const scope = sessionContextState.scope || {};
      const stats = sessionContextState.stats || {};
      const planPayload = sessionContextState.interviewPlan || {};
      const plan = planPayload.plan || planPayload || {};
      const topics = Array.isArray(plan.topics) ? plan.topics : [];
      const profile = sessionContextState.profileDebug || {};
      node.innerHTML = `<details class="session-panel"${sessionContextPanelOpen ? " open" : ""}><summary>Session Context（调试）</summary>
        <div class="session-panel-body">
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
        </div>
      </details>`;
      const panel = node.querySelector(".session-panel");
      const panelBody = node.querySelector(".session-panel-body");
      if (panelBody) {
        panelBody.addEventListener("wheel", (event) => {
          if (panelBody.scrollHeight <= panelBody.clientHeight + 1) return;
          const delta = event.deltaY;
          const atTop = panelBody.scrollTop <= 0;
          const atBottom = panelBody.scrollTop + panelBody.clientHeight >= panelBody.scrollHeight - 1;
          if ((delta < 0 && !atTop) || (delta > 0 && !atBottom)) {
            event.stopPropagation();
          }
        }, {passive: true});
      }
      if (panel) {
        panel.addEventListener("toggle", () => {
          sessionContextPanelOpen = panel.open;
          if (panel.open) renderSessionContext();
        });
      }
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

    async function startNewConversation() {
      if ($("chatMode").value === "answer" && currentAnswerSessionId) {
        try {
          await fetch(`/api/answer/sessions/${encodeURIComponent(currentAnswerSessionId)}/archive`, {method: "POST"});
        } catch {}
      }
      pendingAnswerTurn = null;
      chatHistory = [];
      currentAssistant = null;
      currentAssistantText = "";
      currentPayload = null;
      currentInterviewPlan = null;
      currentInterviewPlanSignature = "";
      currentInterviewState = null;
      currentInterviewSessionId = "";
      currentAnswerSessionId = "";
      lastContextItems = [];
      resetSessionContext();
      currentConversationSignature = conversationSignature();
      syncChatWorkspace({
        activeInterviewSessionId: "",
        activeAnswerSessionId: "",
        pendingTurn: null,
        activeAnswerTaskId: "",
      });
      $("query").value = "";
      $("starters").innerHTML = "";
      const newConversationHint = $("chatMode").value === "interview"
        ? "已开始新对话。可以选择面试模式并发送消息开始。"
        : "已开始新对话。可以直接提问。";
      messages.innerHTML = `<article class="message assistant"><div class="role">Assistant</div><div class="answer">${newConversationHint}</div></article>`;
      updateSessionLabel();
      setBusy(false, "就绪");
      scrollToBottom();
    }

    async function ensureAnswerSession() {
      if (currentAnswerSessionId) return currentAnswerSessionId;
      const response = await fetch("/api/answer/sessions", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          scope_type: $("scopeType").value,
          scope_value: $("scopeValue").value.trim() || null,
          scope_paths: scopePaths(),
          strict_evidence: $("strictEvidence").checked,
          extra: {created_from: "chat"},
        }),
      });
      if (!response.ok) throw new Error("创建问答记录失败");
      const data = await response.json();
      currentAnswerSessionId = data.session.session_id;
      syncChatWorkspace({activeAnswerSessionId: currentAnswerSessionId});
      updateSessionLabel();
      return currentAnswerSessionId;
    }

    function restoreAnswerSessionIntoChat(session) {
      const context = session.context || {};
      if (context.scope_type) $("scopeType").value = context.scope_type;
      if (context.scope_type === "selected_notes") {
        $("scopeValue").value = (context.scope_paths || []).join("\n");
      } else if (context.scope_value) {
        $("scopeValue").value = context.scope_value;
      }
      if (typeof context.strict_evidence === "boolean") $("strictEvidence").checked = context.strict_evidence;
      currentAnswerSessionId = session.session_id || "";
      currentConversationSignature = conversationSignature();
      syncChatWorkspace({activeAnswerSessionId: currentAnswerSessionId});
      const sessionMessages = Array.isArray(session.messages) ? session.messages : [];
      chatHistory = sessionMessages
        .filter((message) => ["user", "assistant"].includes(message.role))
        .filter((message) => message.role !== "assistant" || !["pending", "failed"].includes(message.status))
        .map((message) => ({role: message.role, content: message.content || ""}));
      trimChatHistory();
      const pendingAssistant = sessionMessages.find((message) => message.role === "assistant" && message.status === "pending");
      if (pendingAssistant) {
        const pendingIndex = sessionMessages.indexOf(pendingAssistant);
        const pendingUser = pendingIndex > 0 ? sessionMessages[pendingIndex - 1] : null;
        pendingAnswerTurn = {
          query: pendingUser && pendingUser.role === "user" ? pendingUser.content || "" : "",
          taskId: sessionStorage.getItem(activeAnswerTaskKey) || null,
          status: "running",
        };
      } else {
        pendingAnswerTurn = null;
      }
      messages.innerHTML = sessionMessages.length
        ? sessionMessages.map((message, index) => renderMessageWithReview(message, null, sessionMessages[index - 1])).join("")
        : `<article class="message assistant"><div class="role">Assistant</div><div class="answer">已恢复问答对话。</div></article>`;
      attachRestoredRetryButtons();
      updateSessionLabel();
    }

    async function reloadAnswerSessionFromServer() {
      if (!currentAnswerSessionId) return;
      try {
        const response = await fetch(`/api/answer/sessions/${encodeURIComponent(currentAnswerSessionId)}`);
        if (!response.ok) return;
        const data = await response.json();
        restoreAnswerSessionIntoChat(data.session || {});
        scrollToBottom();
      } catch {}
    }

    async function restoreAnswerWorkspace() {
      if ($("chatMode").value !== "answer") return;
      if (!currentAnswerSessionId) return;
      try {
        await reloadAnswerSessionFromServer();
        setBusy(false, "已恢复问答对话");
      } catch {}
    }

    function renderAnswerConversation() {
      const parts = chatHistory.map((message) => {
        const cls = message.role === "user" ? "user" : "assistant";
        const roleLabel = message.role === "user" ? "You" : "Assistant";
        return `<article class="message ${cls}"><div class="role">${roleLabel}</div><div class="answer">${renderMarkdown(message.content || "")}</div></article>`;
      });
      if (pendingAnswerTurn?.query) {
        const lastUser = [...chatHistory].reverse().find((item) => item.role === "user");
        if (!lastUser || lastUser.content !== pendingAnswerTurn.query) {
          parts.push(`<article class="message user"><div class="role">You</div><div class="answer">${renderMarkdown(pendingAnswerTurn.query)}</div></article>`);
        }
        if (pendingAnswerTurn.partialAssistantText) {
          parts.push(`<article class="message assistant"><div class="role">Assistant</div><div class="answer">${renderMarkdown(pendingAnswerTurn.partialAssistantText)}</div></article>`);
        } else if (pendingAnswerTurn.status === "failed" && pendingAnswerTurn.error) {
          parts.push(`<article class="message assistant"><div class="role">Assistant</div><div class="answer"><p class="error">${escapeHtml(pendingAnswerTurn.error)}</p></div></article>`);
        }
      }
      if (!parts.length) return false;
      messages.innerHTML = parts.join("");
      return true;
    }

    function renderAnswerMessagesFromChatHistory() {
      return renderAnswerConversation();
    }

    async function loadReviewNotice() {
      const node = $("reviewNotice");
      if (!node) return;
      const dismissedKey = "knowledge_agent.review_notice_dismissed";
      if (sessionStorage.getItem(dismissedKey) === "1") return;
      try {
        const response = await fetch("/api/review/due?limit=20");
        if (!response.ok) return;
        const data = await response.json();
        const dueCount = Number(data.due_count || 0);
        if (!dueCount) return;
        const topics = Array.isArray(data.topics) ? data.topics : [];
        const topicText = topics.slice(0, 3).map((item) => {
          const topic = item.topic || "未分组";
          const count = Number(item.count || 0);
          return `${escapeHtml(topic)} ${count} 个弱项建议复查`;
        }).join(" · ");
        node.innerHTML = `
          <div><strong>复习提醒</strong> ${topicText || `${dueCount} 个弱项建议复查`}</div>
          <div class="review-notice-actions">
            <button type="button" data-review-action="cards">逐条对照</button>
            <button type="button" class="secondary" data-review-action="chat" disabled title="对话复查将在下一阶段接入">对话复查</button>
            <button type="button" class="secondary" data-review-action="later">稍后提醒</button>
          </div>
        `;
        node.hidden = false;
        node.querySelector("[data-review-action='cards']").addEventListener("click", () => {
          window.location.href = "/review";
        });
        node.querySelector("[data-review-action='later']").addEventListener("click", () => {
          sessionStorage.setItem(dismissedKey, "1");
          node.hidden = true;
        });
      } catch {
        node.hidden = true;
      }
    }

    async function restoreActiveInterviewSession() {
      const urlMode = new URLSearchParams(window.location.search).get("mode") || "answer";
      if (urlMode === "answer") return;
      const sessionId = currentInterviewSessionId;
      if (!sessionId) return;
      try {
        const response = await fetch(`/api/interview/sessions/${encodeURIComponent(sessionId)}`);
        if (!response.ok) {
          syncChatWorkspace({activeInterviewSessionId: ""});
          return;
        }
        const data = await response.json();
        const session = data.session || {};
        if (!["active", "end_failed"].includes(session.status)) {
          syncChatWorkspace({activeInterviewSessionId: ""});
          return;
        }
        restoreSessionIntoChat(session, data.reviews || []);
      } catch {
        syncChatWorkspace({activeInterviewSessionId: ""});
      }
    }

    function restoreSessionIntoChat(session, reviews) {
      const context = session.context || {};
      $("chatMode").value = "interview";
      window.history.replaceState(null, "", "/?mode=interview");
      updateChatNavActive();
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
      syncChatWorkspace({activeInterviewSessionId: currentInterviewSessionId});
      sessionContextState = sessionContextFromSession(session);
      renderSessionContext();
      renderInterviewDirectionBar();

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
      const citationsHtml = message.role === "assistant" && Array.isArray(message.citations) && message.citations.length
        ? renderTurnCitations(message.citations)
        : "";
      return `<article class="message ${cls}"><div class="role">${escapeHtml(message.role || "")}</div><div class="answer">${renderMarkdown(message.content || "")}${errorHtml}${pendingHtml}</div>${citationsHtml}${retryHtml}${reviewSummary ? renderSessionSummary(reviewSummary) : ""}</article>`;
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
      syncChatWorkspace({activeInterviewSessionId: currentInterviewSessionId});
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
        agent_actions: agentActions,
        citations: currentTurnCitations
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
        agent_actions: agentActions,
        citations: currentTurnCitations
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
        syncChatWorkspace({activeInterviewSessionId: ""});
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
      if ($("chatMode").value === "interview") {
        await loadInterviewHistoryListOnly();
      } else {
        await loadAnswerHistoryListOnly();
      }
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

    async function loadAnswerHistoryListOnly() {
      $("historyContent").innerHTML = "加载中...";
      try {
        const response = await fetch("/api/answer/sessions?limit=30");
        if (!response.ok) throw new Error("问答历史加载失败");
        const data = await response.json();
        renderAnswerHistory(data.sessions || []);
      } catch (error) {
        $("historyContent").innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
      }
    }

    function renderAnswerHistory(sessions) {
      if (!sessions.length) {
        $("historyContent").innerHTML = "暂无问答历史。当前对话保存在服务端；新建对话时，上一轮会归档。";
        return;
      }
      $("historyContent").innerHTML = sessions.map((session) => `<button class="history-item" type="button" data-answer-session="${escapeHtml(session.session_id || "")}"><strong>${escapeHtml(session.title || "问答对话")}</strong><div class="history-meta">${escapeHtml(session.status || "")} | ${escapeHtml(session.updated_at || session.created_at || "")}</div></button>`).join("");
      document.querySelectorAll("[data-answer-session]").forEach((button) => button.addEventListener("click", () => loadAnswerSession(button.dataset.answerSession)));
    }

    async function loadAnswerSession(sessionId) {
      if (!sessionId) return;
      $("historyContent").innerHTML = "正在加载问答记录...";
      try {
        const response = await fetch(`/api/answer/sessions/${encodeURIComponent(sessionId)}`);
        if (!response.ok) throw new Error("问答记录加载失败");
        const data = await response.json();
        if ($("chatMode").value === "answer" && currentAnswerSessionId && currentAnswerSessionId !== sessionId) {
          try {
            await fetch(`/api/answer/sessions/${encodeURIComponent(currentAnswerSessionId)}/archive`, {method: "POST"});
          } catch {}
        }
        $("chatMode").value = "answer";
        window.history.replaceState(null, "", "/?mode=answer");
        updateChatNavActive();
        restoreAnswerSessionIntoChat(data.session || {});
        closeHistory();
        setBusy(false, "已恢复问答记录");
        scrollToBottom();
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
      const html = `<details class="session-panel"><summary>Session Context（调试）</summary>
        <div class="session-panel-body">
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
      const status = $("status");
      if (status) status.textContent = text || (isBusy ? "处理中..." : "就绪");
    }

    function numberValue(id) { return Number($(id).value); }
    function scopePaths() { return $("scopeType").value === "selected_notes" ? $("scopeValue").value.split(/\r?\n/).map((item) => item.trim()).filter(Boolean) : []; }
    function conversationSignature() { return JSON.stringify({mode:$("chatMode").value, scopeType:$("scopeType").value, scopeValue:$("scopeValue").value.trim(), scopePaths:scopePaths(), strictEvidence:$("strictEvidence").checked}); }
    function scrollToBottom() {
      const scrollNode = document.querySelector(".chat-scroll") || messages;
      scrollNode.scrollTop = scrollNode.scrollHeight;
    }
    function escapeHtml(value) { return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;"); }
  </script>
</body>
</html>
"""
