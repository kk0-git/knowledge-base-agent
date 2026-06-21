from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from agent.runtime import AgentRuntime
from agent.schema import AgentRunConfig, AgentState, WorkingMemory
from agent.serialization import to_jsonable
from agent.tool_executor import ToolExecutionContext


LOCAL_TOOLS = ["grep_vault", "inspect_note", "list_notes", "read_note", "search_notes"]
SCOPE_INDEX_COMPLETE_LIMIT = 30


@dataclass(frozen=True)
class LibrarianBudget:
    effort_level: str
    max_steps: int
    allowed_tools: list[str]


@dataclass
class LibrarianRequest:
    query: str
    scope_type: str = "all_vault"
    scope_value: str = ""
    scope_note_paths: tuple[str, ...] = ()
    selected_note_paths: tuple[str, ...] = ()
    chat_history: list[dict[str, Any]] = field(default_factory=list)
    vault_root: Path | None = None
    rag_manager: Any | None = None
    rag_manager_factory: Callable[[], Any] | None = None
    online_search_client: Any | None = None
    online_enabled: bool = False
    strict_evidence: bool = False
    model: str = ""
    tool_mode: str = "auto"
    trace_path: str | None = None
    temperature: float = 0.2
    max_tool_calls_per_step: int = 4


class LibrarianApp:
    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime

    def run(self, request: LibrarianRequest, *, event_sink=None) -> Any:
        budget = route_librarian_scope(
            scope_type=request.scope_type,
            online_enabled=request.online_enabled,
        )
        runtime_context = build_librarian_runtime_context(request=request, budget=budget)
        working = WorkingMemory()
        working.extra["runtime_context"] = runtime_context
        state = AgentState(messages=[], working=working, skill_name="librarian")
        tool_context = ToolExecutionContext(
            working=working,
            vault_root=request.vault_root,
            rag_manager=request.rag_manager,
            rag_manager_factory=request.rag_manager_factory,
            scope_note_paths=request.scope_note_paths,
            scope_type=request.scope_type,
            scope_value=request.scope_value,
            online_search_client=request.online_search_client if request.online_enabled else None,
            turn_context={"runtime_context": runtime_context},
        )
        result = self.runtime.run(
            config=AgentRunConfig(
                skill_name="librarian",
                max_steps=budget.max_steps,
                max_tool_calls_per_step=request.max_tool_calls_per_step,
                temperature=request.temperature,
                model=request.model,
                tool_mode=request.tool_mode,  # type: ignore[arg-type]
                trace_path=request.trace_path,
                allowed_tools=budget.allowed_tools,
            ),
            user_input=build_librarian_input(request=request, runtime_context=runtime_context),
            state=state,
            tool_context=tool_context,
            event_sink=event_sink,
        )
        result.state.working.extra["derived_metrics"] = derive_librarian_metrics(result)
        return result

    def run_stream(self, request: LibrarianRequest) -> Iterator[dict[str, Any]]:
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        holder: dict[str, Any] = {}

        def run_agent() -> None:
            try:
                holder["result"] = self.run(request, event_sink=events.put)
            except Exception as exc:
                holder["error"] = exc
            finally:
                events.put(None)

        thread = threading.Thread(target=run_agent, name="librarian-agent-stream", daemon=True)
        thread.start()
        while True:
            event = events.get()
            if event is None:
                break
            yield event
        thread.join()
        if holder.get("error") is not None:
            raise holder["error"]
        result = holder["result"]
        fallback = build_librarian_fallback(result)
        answer_text = result.final_answer or fallback.get("answer", "")
        if fallback:
            result.metadata.update(
                {
                    "partial": True,
                    "recoverable": True,
                    "fallback_answer": True,
                }
            )
            yield {"type": "agent_stopped", "payload": fallback["stopped"]}
        if answer_text:
            yield {"type": "answer_delta", "payload": {"text": answer_text}}
            yield {"type": "answer", "payload": {"answer": answer_text, "model": request.model, "partial": bool(fallback)}}
        yield {
            "type": "done",
            "payload": {
                "trace_id": result.trace_id,
                "trace_path": result.trace_path,
                "stopped_reason": result.stopped_reason,
                "error": result.error,
                "partial": bool(fallback),
                "recoverable": bool(fallback) or bool(result.metadata.get("recoverable")),
                "telemetry": {
                    "command": "LibrarianAgentV2",
                    "agent_v2": True,
                    "derived_metrics": result.state.working.extra.get("derived_metrics", {}),
                    "total_ms": result.total_ms,
                },
            },
        }


def build_librarian_fallback(result: Any) -> dict[str, Any]:
    if result.final_answer or result.stopped_reason != "max_steps":
        return {}
    successful_results = [
        tool_result
        for step in result.steps
        for tool_result in step.tool_results
        if tool_result.ok
    ]
    source_paths = sorted(
        {
            str(path)
            for step in result.steps
            for tool_result in step.tool_results
            if isinstance(tool_result.output, dict)
            for path in (
                list(tool_result.output.get("source_paths") or [])
                + ([tool_result.output.get("path")] if tool_result.output.get("path") else [])
            )
            if path
        }
    )
    if successful_results:
        source_line = ""
        if source_paths:
            source_line = "\n\n已查阅/命中的资料：" + "、".join(source_paths[:5])
            if len(source_paths) > 5:
                source_line += " 等"
        message = (
            "本轮查阅已达到步骤上限，未能生成完整回答。"
            "我先基于已完成的资料查阅给出阶段性结果；你可以继续追问，我会接着当前问题收敛。"
            f"{source_line}"
        )
    else:
        message = (
            "本轮查阅已达到步骤上限，且没有获得可用资料结果。"
            "请缩小范围、改写问题，或稍后重试。"
        )
    return {
        "answer": message,
        "stopped": {
            "reason": result.stopped_reason,
            "message": message,
            "trace_path": result.trace_path,
            "recoverable": True,
            "partial": bool(successful_results),
        },
    }


def route_librarian_scope(*, scope_type: str, online_enabled: bool) -> LibrarianBudget:
    normalized = str(scope_type or "").strip() or "all_vault"
    if normalized == "selected_notes":
        tools = ["grep_vault", "read_note"]
        budget = LibrarianBudget(effort_level="L2", max_steps=4, allowed_tools=tools)
    elif normalized in {"folder", "tag", "search", "all_vault", "current_context"}:
        tools = list(LOCAL_TOOLS)
        max_steps = 6
        effort = "L1" if normalized != "all_vault" else "L3"
        budget = LibrarianBudget(effort_level=effort, max_steps=max_steps, allowed_tools=tools)
    else:
        budget = LibrarianBudget(effort_level="L1", max_steps=6, allowed_tools=list(LOCAL_TOOLS))
    if online_enabled and "online_search" not in budget.allowed_tools:
        return LibrarianBudget(
            effort_level=budget.effort_level,
            max_steps=budget.max_steps + 1,
            allowed_tools=[*budget.allowed_tools, "online_search"],
        )
    return budget


def build_librarian_runtime_context(*, request: LibrarianRequest, budget: LibrarianBudget) -> dict[str, Any]:
    return {
        "mode": "answer",
        "effort_level": budget.effort_level,
        "online_enabled": bool(request.online_enabled),
        "strict_evidence": bool(request.strict_evidence),
        "scope": {
            "type": request.scope_type,
            "value": request.scope_value,
            "allowed_note_count": len(request.scope_note_paths),
            "selected_note_paths": list(request.selected_note_paths or ()),
        },
        "scope_index": build_scope_index(request),
        "tool_policy": {
            "allowed_tools": list(budget.allowed_tools),
            "max_steps": budget.max_steps,
        },
    }


def build_scope_index(request: LibrarianRequest) -> dict[str, Any]:
    paths = [str(path).strip() for path in request.scope_note_paths if str(path).strip()]
    note_count = len(paths)
    if note_count <= SCOPE_INDEX_COMPLETE_LIMIT:
        return {
            "type": request.scope_type,
            "note_count": note_count,
            "is_complete": True,
            "notes": [{"path": path, "title": Path(path).stem} for path in sorted(paths)],
        }
    return {
        "type": request.scope_type,
        "note_count": note_count,
        "is_complete": False,
        "notes": [],
        "hint": "Scope is large. Use list_notes, search_notes, or grep_vault to inspect relevant candidates.",
    }


def build_librarian_input(*, request: LibrarianRequest, runtime_context: dict[str, Any]) -> str:
    history = []
    for item in (request.chat_history or [])[-6:]:
        role = str(item.get("role") or "user")
        content = str(item.get("content") or "").strip()
        if content:
            history.append(f"{role}: {content}")
    sections = [
            "# Runtime Context",
            json.dumps(runtime_context, ensure_ascii=False, indent=2),
            "",
    ]
    if request.strict_evidence:
        sections.extend(
            [
                "# Strict Evidence Constraint",
                (
                    "You are answering as a careful vault librarian in 'only use my notes' mode. "
                    "Keep the answer natural and user-facing. "
                    "Use only information supported by the current vault scope and tool observations. "
                    "If the notes do not contain enough information, say so briefly in normal language. "
                    "Do not add architecture inferences or outside knowledge unless the user explicitly asks for inference. "
                    "Do not use audit-style labels such as 'directly supported', 'inferred', or 'missing' unless the user asks for evidence analysis."
                ),
                "",
            ]
        )
    sections.extend(
        [
            "# Short Conversation History",
            "\n".join(history) if history else "(none)",
            "",
            "# User Request",
            request.query,
            "",
            "# Task",
            "Answer the user using the bounded librarian tool loop. Respect scope and evidence policy.",
        ]
    )
    return "\n\n".join(sections)


def derive_librarian_metrics(result: Any) -> dict[str, Any]:
    tool_names = [call.name for step in result.steps for call in step.tool_calls]
    read_paths: set[str] = set()
    source_paths: set[str] = set()
    for step in result.steps:
        for tool_result in step.tool_results:
            output = tool_result.output if isinstance(tool_result.output, dict) else {}
            for path in output.get("source_paths") or []:
                source_paths.add(str(path))
            if tool_result.name == "read_note" and tool_result.ok and output.get("path"):
                read_paths.add(str(output.get("path")))
                source_paths.add(str(output.get("path")))
    return {
        "tool_call_count": len(tool_names),
        "tool_sequence": tool_names,
        "notes_read": len(read_paths),
        "source_paths": sorted(source_paths),
        "online_used": "online_search" in tool_names,
        "search_count": sum(1 for name in tool_names if name in {"search_notes", "grep_vault", "list_notes"}),
    }
