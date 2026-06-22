from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from agent.llm.tool_calling import LLMToolRequest, LLMToolResponse, OpenAICompatibleToolCallingClient
from agent.apps import LibrarianApp, LibrarianRequest
from agent.runtime import AgentRuntime
from agent.schema import AgentRunConfig, ToolCall
from agent.skill_loader import SkillLoader
from agent.tool_executor import ToolExecutionContext
from agent.tool_registry import ToolRegistry
from agent.tools import register_debug_tools, register_interview_tools, register_profile_tools
from agent.tools.vault import register_vault_tools
from agent.trace import TraceRecorder
from knowledge_base_agent.config import load_llm_config
from knowledge_base_agent.llm import create_llm_client
from services.rag.schema import SearchResult, TextChunk
from services.workflows.interview import InterviewPlan, TopicCard, interview_plan_from_payload
from services.workflows.schema import ScopeSpec
from services.workflows.scope_resolver import ScopeResolver


class DeterministicDebugLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete_with_tools(self, request: LLMToolRequest) -> LLMToolResponse:
        self.calls += 1
        tool_names = {schema.get("function", {}).get("name") for schema in request.tools}
        if is_coach_toolset(tool_names):
            return self._complete_coach(request, tool_names)
        if "get_interview_state" in tool_names or "advance_layer" in tool_names or "select_topic" in tool_names:
            return self._complete_interviewer(request, tool_names)
        if {"search_notes", "read_note"} & tool_names:
            return self._complete_librarian(request, tool_names)
        if self.calls == 1 and request.tools:
            user_text = latest_user_text(request)
            return LLMToolResponse(
                content="",
                tool_calls=[ToolCall(id="call_debug_1", name="echo", arguments={"text": user_text})],
                finish_reason="tool_calls",
                raw={"fake": True},
                used_mode=f"fake-{request.tool_mode}",
            )
        return LLMToolResponse(
            content="Phase 0 debug run completed. The runtime executed a tool call and returned a final answer.",
            tool_calls=[],
            finish_reason="stop",
            raw={"fake": True},
            used_mode=f"fake-{request.tool_mode}",
        )

    def _complete_coach(self, request: LLMToolRequest, tool_names: set[str]) -> LLMToolResponse:
        return LLMToolResponse(
            content=json.dumps(
                {
                    "feedback": {
                        "question_requires": ["协议目标", "Host/Client/Server 角色分工"],
                        "coach_note": "你的回答抓到了工具调用这个方向，但还没有拆出协议角色边界。",
                        "covered": ["提到了工具调用"],
                        "gaps": ["没有区分 Host、Client、Server 的职责"],
                        "thinking_framework": "协议类问题先定义目标，再拆角色和调用链路。",
                        "interviewer_followup_note": "面试官追问角色边界，是为了确认你是否理解协议落点。",
                    },
                    "expression_example": "MCP 可以理解为让 Agent 访问外部工具和数据源的协议层，但回答时要拆成 Host、Client、Server 三层说明。",
                    "profile_signals": [],
                },
                ensure_ascii=False,
            ),
            tool_calls=[],
            finish_reason="stop",
            raw={"fake": True},
            used_mode=f"fake-{request.tool_mode}",
        )
        if self.calls == 1 and "recall_profile" in tool_names:
            return LLMToolResponse(
                tool_calls=[ToolCall(id="call_recall_1", name="recall_profile", arguments={"topic": "MCP 协议", "limit": 3})],
                finish_reason="tool_calls",
                raw={"fake": True},
                used_mode=f"fake-{request.tool_mode}",
            )
        if self.calls == 2 and "record_signal" in tool_names:
            return LLMToolResponse(
                tool_calls=[
                    ToolCall(
                        id="call_signal_1",
                        name="record_signal",
                        arguments={
                            "signal_type": "possible_weak_point",
                            "point": "MCP 角色边界表述不清",
                            "topic": "MCP 协议",
                            "category": "knowledge_gap",
                            "scope_suggestion": "domain",
                            "evidence": "用户把 MCP 泛化为工具调用协议，没有区分 Host、Client、Server。",
                            "confidence": "medium",
                        },
                    )
                ],
                finish_reason="tool_calls",
                raw={"fake": True},
                used_mode=f"fake-{request.tool_mode}",
            )
        return LLMToolResponse(
            content='{"feedback":{"question_requires":["协议目标","Host/Client/Server 角色分工"],"coach_note":"你的回答抓到了工具调用这个方向，但还没有拆出协议角色边界。","covered":["提到了工具调用"],"gaps":["没有区分 Host、Client、Server 的职责"],"thinking_framework":"协议类问题先定义目标，再拆角色和调用链路。","interviewer_followup_note":"面试官追问角色边界，是为了确认你是否理解协议落点。"},"expression_example":"MCP 可以理解为让 Agent 访问外部工具和数据源的协议层，但回答时要拆成 Host、Client、Server 三层说明。","profile_signals":[]}',
            tool_calls=[],
            finish_reason="stop",
            raw={"fake": True},
            used_mode=f"fake-{request.tool_mode}",
        )

    def _complete_interviewer(self, request: LLMToolRequest, tool_names: set[str]) -> LLMToolResponse:
        user_text = latest_user_text(request)

        if self.calls == 1 and "select_topic" in tool_names and should_fake_select_topic(user_text):
            return LLMToolResponse(
                tool_calls=[
                    ToolCall(
                        id="call_select_topic_1",
                        name="select_topic",
                        arguments={"name": "MCP 协议", "reason": "debug request asked to switch topic", "source": "agent"},
                    )
                ],
                finish_reason="tool_calls",
                raw={"fake": True},
                used_mode=f"fake-{request.tool_mode}",
            )

        if self.calls == 1 and "advance_layer" in tool_names and should_fake_advance(request, user_text):
            return LLMToolResponse(
                tool_calls=[
                    ToolCall(
                        id="call_advance_1",
                        name="advance_layer",
                        arguments={"reason": "debug run has enough signal to verify layer transition", "force": True},
                    )
                ],
                finish_reason="tool_calls",
                raw={"fake": True},
                used_mode=f"fake-{request.tool_mode}",
            )

        if self.calls == 1 and "recall_profile" in tool_names and should_fake_recall_profile(user_text):
            return LLMToolResponse(
                tool_calls=[ToolCall(id="call_recall_1", name="recall_profile", arguments={"topic": "MCP 协议", "limit": 3})],
                finish_reason="tool_calls",
                raw={"fake": True},
                used_mode=f"fake-{request.tool_mode}",
            )

        if self.calls == 1 and "search_notes" in tool_names and should_fake_search(user_text):
            return LLMToolResponse(
                tool_calls=[
                    ToolCall(
                        id="call_search_1",
                        name="search_notes",
                        arguments={"query": user_text, "top_k": 3},
                    )
                ],
                finish_reason="tool_calls",
                raw={"fake": True},
                used_mode=f"fake-{request.tool_mode}",
            )

        if self.calls == 2 and "read_note" in tool_names:
            path = first_search_hit_path(request)
            if path:
                return LLMToolResponse(
                    tool_calls=[ToolCall(id="call_read_1", name="read_note", arguments={"path": path, "max_chars": 1200})],
                    finish_reason="tool_calls",
                    raw={"fake": True},
                    used_mode=f"fake-{request.tool_mode}",
                )

        return LLMToolResponse(
            content="你刚才把 MCP 说成“工具调用协议”，这个说法还太粗。请你具体区分一下 Host、Client、Server 三者分别承担什么职责？",
            tool_calls=[],
            finish_reason="stop",
            raw={"fake": True},
            used_mode=f"fake-{request.tool_mode}",
        )

    def _complete_librarian(self, request: LLMToolRequest, tool_names: set[str]) -> LLMToolResponse:
        user_text = latest_user_text(request)
        if is_strict_name_lookup(user_text, request):
            return self._complete_librarian_strict(request, tool_names)
        if is_agent_boundary_question(user_text):
            return self._complete_librarian_agent_boundary(request, tool_names)
        if self.calls == 1 and "search_notes" not in tool_names and "read_note" in tool_names:
            path = first_selected_note_path(request)
            if path:
                return LLMToolResponse(
                    tool_calls=[ToolCall(id="call_read_1", name="read_note", arguments={"path": path, "max_chars": 1200})],
                    finish_reason="tool_calls",
                    raw={"fake": True},
                    used_mode=f"fake-{request.tool_mode}",
                )
        if self.calls == 1 and "search_notes" in tool_names:
            return LLMToolResponse(
                tool_calls=[
                    ToolCall(
                        id="call_search_1",
                        name="search_notes",
                        arguments={"query": user_text, "top_k": 3},
                    )
                ],
                finish_reason="tool_calls",
                raw={"fake": True},
                used_mode=f"fake-{request.tool_mode}",
            )
        if self.calls == 2 and "read_note" in tool_names:
            path = first_search_hit_path(request)
            if path:
                return LLMToolResponse(
                    tool_calls=[ToolCall(id="call_read_1", name="read_note", arguments={"path": path, "max_chars": 1200})],
                    finish_reason="tool_calls",
                    raw={"fake": True},
                    used_mode=f"fake-{request.tool_mode}",
                )
        return LLMToolResponse(
            content="Phase 1 librarian debug run completed. The agent searched notes, read one note when available, and returned a final answer.",
            tool_calls=[],
            finish_reason="stop",
            raw={"fake": True},
            used_mode=f"fake-{request.tool_mode}",
        )

    def _complete_librarian_agent_boundary(self, request: LLMToolRequest, tool_names: set[str]) -> LLMToolResponse:
        if self.calls == 1 and "search_notes" in tool_names:
            return LLMToolResponse(
                tool_calls=[
                    ToolCall(
                        id="call_search_1",
                        name="search_notes",
                        arguments={"query": "memory profile coach agent", "top_k": 5},
                    )
                ],
                finish_reason="tool_calls",
                raw={"fake": True},
                used_mode=f"fake-{request.tool_mode}",
            )
        if self.calls == 2 and "read_note" in tool_names:
            return LLMToolResponse(
                tool_calls=[
                    ToolCall(id="call_read_1", name="read_note", arguments={"path": "memory.md", "max_chars": 4000}),
                    ToolCall(id="call_read_2", name="read_note", arguments={"path": "multi-agent.md", "max_chars": 4000}),
                ],
                finish_reason="tool_calls",
                raw={"fake": True},
                used_mode=f"fake-{request.tool_mode}",
            )
        if self.calls == 3 and "grep_vault" in tool_names:
            return LLMToolResponse(
                tool_calls=[
                    ToolCall(id="call_grep_1", name="grep_vault", arguments={"query": "profile", "ignore_case": True}),
                    ToolCall(id="call_grep_2", name="grep_vault", arguments={"query": "coach", "ignore_case": True}),
                ],
                finish_reason="tool_calls",
                raw={"fake": True},
                used_mode=f"fake-{request.tool_mode}",
            )
        return LLMToolResponse(
            content="Memory 负责存储与检索；Profile 维护用户稳定信息；Coach 负责评估与反思。vault 中没有独立的 profile/coach agent 命名，以上边界基于 memory 与 multi-agent 相关材料整理。",
            tool_calls=[],
            finish_reason="stop",
            raw={"fake": True},
            used_mode=f"fake-{request.tool_mode}",
        )

    def _complete_librarian_strict(self, request: LLMToolRequest, tool_names: set[str]) -> LLMToolResponse:
        if self.calls == 1 and "grep_vault" in tool_names:
            return LLMToolResponse(
                tool_calls=[ToolCall(id="call_grep_1", name="grep_vault", arguments={"query": "coach", "ignore_case": True})],
                finish_reason="tool_calls",
                raw={"fake": True},
                used_mode=f"fake-{request.tool_mode}",
            )
        return LLMToolResponse(
            content="vault 中未找到 coach 的直接证据，无法在 strict evidence 模式下描述 Coach Agent 设计。",
            tool_calls=[],
            finish_reason="stop",
            raw={"fake": True},
            used_mode=f"fake-{request.tool_mode}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agent runtime debug harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--skill", default="interviewer")
    run_parser.add_argument("--input", required=True)
    run_parser.add_argument("--max-steps", type=int, default=6)
    run_parser.add_argument("--max-tool-calls-per-step", type=int, default=4)
    run_parser.add_argument("--tool-mode", choices=["native", "json", "auto"], default="auto")
    run_parser.add_argument("--llm-mode", choices=["fake", "configured"], default="fake")
    run_parser.add_argument("--trace-out", default=str(PROJECT_ROOT / "eval-results" / "agent-debug" / "traces"))
    run_parser.add_argument("--print-steps", action="store_true")
    run_parser.add_argument("--vault", default=None)
    run_parser.add_argument("--rag-mode", choices=["fake", "configured"], default="fake")
    run_parser.add_argument("--index", default="./rag-index/index.json")
    run_parser.add_argument("--bm25-index", default=None)
    run_parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    run_parser.add_argument("--embedding-provider", choices=["local", "openai_compatible"], default="local")
    run_parser.add_argument("--scope-type", choices=["all_vault", "folder", "selected_notes"], default="all_vault")
    run_parser.add_argument("--scope-value", default=None)
    run_parser.add_argument("--scope-path", action="append", default=[])
    run_parser.add_argument("--notes-top-k", type=int, default=8)
    run_parser.add_argument("--session-id", default="")
    run_parser.add_argument("--interview-plan", default=None)
    args = parser.parse_args()

    if args.command == "run":
        return run_agent(args)
    raise ValueError(f"unsupported command: {args.command}")


def run_agent(args: argparse.Namespace) -> int:
    registry = ToolRegistry()
    register_debug_tools(registry)
    register_vault_tools(registry)
    register_interview_tools(registry)
    register_profile_tools(registry)

    loader = SkillLoader(PROJECT_ROOT / "skills", registry=registry)
    llm_client = build_llm_client(args.llm_mode)
    runtime = AgentRuntime(
        llm_client=llm_client,
        skill_loader=loader,
        tool_registry=registry,
        trace_recorder=TraceRecorder(PROJECT_ROOT / "eval-results" / "agent-debug" / "traces"),
    )
    model = "debug-fake" if args.llm_mode == "fake" else load_llm_config(PROJECT_ROOT).model
    if args.skill == "librarian":
        result = run_librarian_app(args=args, runtime=runtime, model=model)
        print_agent_result(result=result, print_steps=args.print_steps)
        return 0 if result.stopped_reason in {"final", "max_steps"} else 1

    result = runtime.run(
        config=AgentRunConfig(
            skill_name=args.skill,
            max_steps=args.max_steps,
            max_tool_calls_per_step=args.max_tool_calls_per_step,
            model=model,
            tool_mode=args.tool_mode,
            trace_path=args.trace_out,
        ),
        user_input=build_cli_user_input(args),
        tool_context=build_tool_context(args),
    )

    print_agent_result(result=result, print_steps=args.print_steps)
    return 0 if result.stopped_reason in {"final", "max_steps"} else 1


def print_agent_result(*, result, print_steps: bool) -> None:
    print(result.final_answer or f"(no final answer; stopped_reason={result.stopped_reason})")
    print()
    print(f"stopped_reason: {result.stopped_reason}")
    print(f"trace_id: {result.trace_id}")
    print(f"trace_path: {Path(result.trace_path).resolve() if result.trace_path else '(not saved)'}")
    if result.error:
        print(f"error: {result.error_type}: {result.error}")
    if print_steps:
        print()
        print("steps:")
        for step in result.steps:
            tool_names = ", ".join(call.name for call in step.tool_calls) or "-"
            statuses = ", ".join(tool.status for tool in step.tool_results) or "-"
            print(f"- {step.index}: {step.kind.value} tools=[{tool_names}] statuses=[{statuses}] latency_ms={step.latency_ms}")


def run_librarian_app(*, args: argparse.Namespace, runtime: AgentRuntime, model: str):
    if not args.vault:
        raise ValueError("--vault is required for librarian skill")
    vault_root = Path(args.vault)
    manager = build_rag_manager(args, vault_root)
    scope_paths = resolve_scope_note_paths(args, vault_root=vault_root, manager=manager)
    return LibrarianApp(runtime).run(
        LibrarianRequest(
            query=args.input,
            scope_type=args.scope_type,
            scope_value=args.scope_value or "",
            scope_note_paths=tuple(scope_paths),
            selected_note_paths=tuple(scope_paths) if args.scope_type == "selected_notes" else (),
            vault_root=vault_root,
            rag_manager=manager,
            rag_manager_factory=lambda: build_rag_manager(args, vault_root),
            model=model,
            tool_mode=args.tool_mode,
            trace_path=args.trace_out,
            max_tool_calls_per_step=args.max_tool_calls_per_step,
        )
    )


def build_llm_client(mode: str):
    if mode == "fake":
        return DeterministicDebugLLM()
    return OpenAICompatibleToolCallingClient(create_llm_client(load_llm_config(PROJECT_ROOT)))


def build_cli_user_input(args: argparse.Namespace) -> str:
    if args.skill != "interviewer":
        return args.input

    runtime_context = {
        "interview_mode": "mock",
        "session_id": args.session_id,
        "topic_phase": "active",
        "current_topic": "MCP 协议",
        "current_topic_index": 0,
        "current_layer_index": 0,
        "current_layer_name": "概念边界",
        "follow_up_count_before_this_turn": 1,
        "compact_plan": {
            "topic_count": 1,
            "topics": [
                {
                    "index": 0,
                    "name": "MCP 协议",
                    "coverage": ["概念边界", "角色分工", "调用链路", "工程取舍"],
                }
            ],
        },
        "scope": {
            "type": args.scope_type or "all_vault",
            "value": args.scope_value or "",
            "allowed_note_count": len(args.scope_path or []),
        },
        "profile": {
            "profile_available": True,
            "matching_weak_count": 1,
            "due_review_count": 1,
            "strong_point_count": 0,
        },
        "tool_boundaries": {
            "preloaded": ["interview_state", "compact_plan", "scope_summary", "profile_counts"],
            "on_demand": ["search_notes", "grep_vault", "read_note", "recall_profile"],
            "actions": ["advance_layer", "select_topic"],
        },
    }
    return "\n\n".join(
        [
            "# Runtime Context",
            json.dumps(runtime_context, ensure_ascii=False, indent=2),
            "",
            "# Current User Message",
            args.input,
            "",
            "# Task",
            (
                "Continue the mock interview from the authoritative runtime context above. "
                "Do not routinely call get_interview_state or list_plan_topics. "
                "Use note/profile tools only when specific details are needed. "
                "Use advance_layer or select_topic only when intentionally changing server state. "
                "End with exactly one question."
            ),
        ]
    )


def build_tool_context(args: argparse.Namespace) -> ToolExecutionContext | None:
    if args.skill not in {"librarian", "interviewer", "coach"}:
        return None
    if args.skill == "librarian" and not args.vault:
        raise ValueError("--vault is required for librarian skill")

    vault_root = Path(args.vault) if args.vault else None
    manager = build_rag_manager(args, vault_root) if vault_root else None
    scope_paths = resolve_scope_note_paths(args, vault_root=vault_root, manager=manager) if vault_root else []
    plan = build_interview_plan(args) if args.skill in {"interviewer", "coach"} else None

    return ToolExecutionContext(
        working=None,  # Runtime replaces this with the active WorkingMemory.
        confirmed_tools={"advance_layer", "select_topic"},
        vault_root=vault_root,
        rag_manager=manager,
        scope_note_paths=tuple(scope_paths),
        scope_type=args.scope_type,
        scope_value=args.scope_value or "",
        session_id=args.session_id,
        interview_plan=plan,
        interview_state=active_interviewer_state(args) if args.skill == "interviewer" else None,
        profile_store=FileProfileStore(),
        turn_context={
            "current_topic": "MCP 协议",
            "latest_user_answer": args.input,
            "interviewer_followup": args.input,
            "context_note_paths": tuple(scope_paths),
        },
    )


def active_interviewer_state(args: argparse.Namespace) -> dict[str, object]:
    return {
        "source": "server",
        "session_id": args.session_id,
        "current_topic": "MCP 协议",
        "current_topic_index": 0,
        "topic_phase": "active",
        "topic_selection_source": "cli_fake",
        "current_layer_index": 0,
        "current_layer_name": "概念边界",
        "follow_up_count": 1,
    }


def build_rag_manager(args: argparse.Namespace, vault_root: Path):
    if args.rag_mode == "fake":
        return FileScanRAGManager(vault_root)
    import rag_eval

    return rag_eval.build_manager(
        index_path=Path(args.index),
        bm25_index_path=Path(args.bm25_index) if args.bm25_index else None,
        model_name=args.embedding_model,
        embedding_provider=args.embedding_provider,
        mode="hybrid",
    )


def resolve_scope_note_paths(args: argparse.Namespace, *, vault_root: Path, manager) -> list[str]:
    if args.scope_type == "selected_notes":
        return [str(path).replace("\\", "/") for path in args.scope_path]
    if args.scope_type == "folder":
        if not args.scope_value:
            raise ValueError("--scope-value is required for folder scope")
        resolver = ScopeResolver(vault_root=vault_root)
        result = resolver.resolve(ScopeSpec(type="folder", value=args.scope_value, top_k=args.notes_top_k))
        return [str(note.get("path") or "") for note in result.notes if str(note.get("path") or "").strip()]
    return []


def build_interview_plan(args: argparse.Namespace) -> InterviewPlan:
    if args.interview_plan:
        payload = json.loads(Path(args.interview_plan).read_text(encoding="utf-8"))
        return interview_plan_from_payload(payload, context=fake_context_for_plan())
    return InterviewPlan(
        topics=(
            TopicCard(
                name="MCP 协议",
                coverage=("概念边界", "角色分工", "调用链路", "工程取舍"),
                source_note_paths=tuple(args.scope_path or []),
            ),
        ),
        suggested_order=("MCP 协议",),
    )


class FileScanRAGManager:
    """Lightweight CLI-only manager for deterministic dry runs without embedding dependencies."""

    def __init__(self, vault_root: Path):
        self.vault_root = vault_root

    def hybrid_search(self, *, query: str, top_k: int, dense_top_k: int, bm25_top_k: int, rrf_k: int):
        terms = [term.lower() for term in query.split() if term.strip()]
        results: list[SearchResult] = []
        for path in sorted(self.vault_root.rglob("*.md")):
            if len(results) >= top_k:
                break
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            haystack = text.lower()
            if terms and not any(term in haystack for term in terms):
                continue
            relative = path.relative_to(self.vault_root).as_posix()
            preview = text.strip()[:1200]
            results.append(file_scan_result(relative=relative, text=preview, score=1.0 / (len(results) + 1)))
        if not results:
            fallback = next(iter(sorted(self.vault_root.rglob("*.md"))), None)
            if fallback is not None:
                text = fallback.read_text(encoding="utf-8", errors="replace")
                results.append(
                    file_scan_result(
                        relative=fallback.relative_to(self.vault_root).as_posix(),
                        text=text.strip()[:1200],
                        score=0.01,
                    )
                )
        return results


class FileProfileStore:
    def load(self):
        return {
            "schema_version": 2,
            "weak_points": [
                {
                    "point": "MCP 角色边界表述不清",
                    "topic": "MCP 协议",
                    "scope": "domain",
                    "category": "knowledge_gap",
                    "planned_layer": "角色分工",
                    "improved": False,
                    "domain_anchor": {"plan_topic": "MCP 协议", "context_note_paths": [], "scope_path": ""},
                    "sr": {"next_review": "2000-01-01", "ease_factor": 2.5},
                }
            ],
            "strong_points": [],
            "topic_mastery": {},
            "communication": {"style": "", "suggestions": []},
        }


def file_scan_result(*, relative: str, text: str, score: float) -> SearchResult:
    return SearchResult(
        chunk=TextChunk(
            chunk_id=f"{relative}#filescan",
            note_path=relative,
            heading_path=[],
            text=text,
            start_line=1,
            end_line=min(len(text.splitlines()), 40),
        ),
        score=score,
    )


def is_agent_boundary_question(user_text: str) -> bool:
    lowered = user_text.lower()
    has_names = "profile" in lowered and "coach" in lowered
    has_topic = any(token in user_text for token in ["memory", "Memory", "记忆", "职责", "边界"])
    return has_names and has_topic


def is_strict_name_lookup(user_text: str, request: LLMToolRequest) -> bool:
    ctx = runtime_context_from_request(request)
    if not ctx.get("strict_evidence"):
        return False
    return "coach" in user_text.lower()


def runtime_context_from_request(request: LLMToolRequest) -> dict[str, Any]:
    for message in request.messages:
        if message.role != "user":
            continue
        content = str(message.content or "")
        marker = "# Runtime Context"
        if marker not in content:
            continue
        section = content.split(marker, 1)[1]
        for stop_marker in ("# Strict Evidence Constraint", "# Short Conversation History", "# Task"):
            if stop_marker in section:
                section = section.split(stop_marker, 1)[0]
        try:
            payload = json.loads(section.strip())
        except Exception:
            continue
        return payload if isinstance(payload, dict) else {}
    return {}


def latest_user_text(request: LLMToolRequest) -> str:
    for message in reversed(request.messages):
        if message.role != "user":
            continue
        content = message.content
        for marker, stop_markers in (
            ("# User Request", ("# Task",)),
            ("# Current User Message", ("# Task",)),
        ):
            if marker not in content:
                continue
            tail = content.split(marker, 1)[1]
            for stop in stop_markers:
                if stop in tail:
                    tail = tail.split(stop, 1)[0]
            text = tail.strip()
            if text:
                return text
        return content
    return ""


def is_coach_toolset(tool_names: set[str]) -> bool:
    if {"advance_layer", "select_topic", "search_notes", "grep_vault"} & tool_names:
        return False
    return "read_note" in tool_names and ("get_interview_state" in tool_names or "list_plan_topics" in tool_names)


def should_fake_advance(request: LLMToolRequest, user_text: str) -> bool:
    text = (user_text or "") + "\n" + "\n".join(message.content for message in request.messages if message.role == "tool")
    return "advance" in text.lower() or "切层" in text or "下一层" in text


def should_fake_search(user_text: str) -> bool:
    return any(marker in user_text for marker in ["搜索", "笔记", "note", "Redis", "Stream"])


def should_fake_recall_profile(user_text: str) -> bool:
    return any(marker in user_text for marker in ["profile", "画像", "弱点", "复习"])


def should_fake_select_topic(user_text: str) -> bool:
    lowered = user_text.lower()
    return "select" in lowered or "换 topic" in user_text or "选择 topic" in user_text or "切 topic" in user_text


def first_search_hit_path(request: LLMToolRequest) -> str:
    for message in reversed(request.messages):
        if message.role != "tool":
            continue
        try:
            payload = json.loads(message.content)
        except Exception:
            continue
        hits = payload.get("hits") if isinstance(payload, dict) else None
        if isinstance(hits, list) and hits:
            first = hits[0]
            if isinstance(first, dict) and first.get("path"):
                return str(first["path"])
    return ""


def first_selected_note_path(request: LLMToolRequest) -> str:
    for message in request.messages:
        if message.role != "user":
            continue
        content = str(message.content or "")
        marker = "# Runtime Context"
        if marker not in content:
            continue
        section = content.split(marker, 1)[1]
        if "# Short Conversation History" in section:
            section = section.split("# Short Conversation History", 1)[0]
        try:
            payload = json.loads(section.strip())
        except Exception:
            continue
        scope = payload.get("scope") if isinstance(payload, dict) else {}
        paths = scope.get("selected_note_paths") if isinstance(scope, dict) else []
        if isinstance(paths, list):
            for path in paths:
                if str(path or "").strip():
                    return str(path)
    return ""


class _FakeContext:
    items: list[dict] = []


def fake_context_for_plan():
    return _FakeContext()


if __name__ == "__main__":
    raise SystemExit(main())
