from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from agent.interview.state import InterviewStateMachine, build_interview_state_machine
from agent.runtime import AgentRuntime
from agent.schema import AgentRunConfig, AgentState, WorkingMemory
from agent.serialization import to_jsonable
from agent.tool_executor import ToolExecutionContext


@dataclass
class InterviewTurnRequest:
    query: str
    session_id: str = ""
    chat_history: list[dict[str, Any]] = field(default_factory=list)
    interview_plan: Any | None = None
    interview_state: dict[str, Any] | None = None
    vault_root: Path | None = None
    rag_manager: Any | None = None
    rag_manager_factory: Callable[[], Any] | None = None
    scope_note_paths: tuple[str, ...] = ()
    scope_type: str = ""
    scope_value: str = ""
    session_store: Any | None = None
    profile_store: Any | None = None
    model: str = ""
    tool_mode: str = "auto"
    trace_path: str | None = None
    max_steps: int = 6
    max_tool_calls_per_step: int = 4
    temperature: float = 0.2


class InterviewInterviewerApp:
    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime

    def run_turn(
        self,
        request: InterviewTurnRequest,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[Any, InterviewStateMachine]:
        machine = build_interview_state_machine(
            plan=request.interview_plan,
            state_payload=request.interview_state,
            session_id=request.session_id,
        )
        state_before = machine.snapshot()
        runtime_context = build_interviewer_runtime_context(request, machine, request.profile_store)
        working = working_from_machine(machine)
        working.extra["runtime_context"] = runtime_context
        state = AgentState(
            messages=[],
            working=working,
            skill_name="interviewer",
        )
        state.working.extra["interview_state_before"] = state_before
        tool_context = ToolExecutionContext(
            working=working,
            confirmed_tools={"advance_layer", "select_topic"},
            vault_root=request.vault_root,
            rag_manager=request.rag_manager,
            rag_manager_factory=request.rag_manager_factory,
            scope_note_paths=request.scope_note_paths,
            scope_type=request.scope_type,
            scope_value=request.scope_value,
            session_id=request.session_id,
            session_store=request.session_store,
            profile_store=request.profile_store,
            interview_plan=request.interview_plan,
            interview_state=state_before,
            state_machine=machine,
            metadata={"collect_citations": True},
            turn_context={"runtime_context": runtime_context},
        )
        result = self.runtime.run(
            config=AgentRunConfig(
                skill_name="interviewer",
                max_steps=request.max_steps,
                max_tool_calls_per_step=request.max_tool_calls_per_step,
                temperature=request.temperature,
                model=request.model,
                tool_mode=request.tool_mode,  # type: ignore[arg-type]
                trace_path=request.trace_path,
            ),
            user_input=build_turn_input(request, runtime_context=runtime_context),
            state=state,
            tool_context=tool_context,
            event_sink=event_sink,
        )
        if result.final_answer:
            machine.commit_turn(user_text=request.query, assistant_text=result.final_answer)
        state_after = machine.snapshot()
        result.state.working.extra["interview_state_before"] = state_before
        result.state.working.extra["interview_state_after"] = state_after
        result.state.working.extra["state_transitions"] = state_after.get("transition_history", [])
        result.state.working.extra["runtime_context"] = runtime_context
        result.state.working.extra["citations"] = list(tool_context.citations)
        result.state.working.extra["derived_metrics"] = derive_interview_metrics(result=result, state_before=state_before, state_after=state_after)
        sync_working_from_snapshot(result.state.working, state_after)
        rewrite_trace_interview_metadata(
            result=result,
            state_before=state_before,
            state_after=state_after,
            runtime_context=runtime_context,
        )
        persist_agent_trace(request=request, result=result, state_after=state_after)
        return result, machine

    def run_turn_stream(self, request: InterviewTurnRequest) -> Iterator[dict[str, Any]]:
        event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        holder: dict[str, Any] = {}

        def event_sink(event: dict[str, Any]) -> None:
            event_queue.put(event)

        def worker() -> None:
            try:
                result, machine = self.run_turn(request, event_sink=event_sink)
                holder["result"] = result
                holder["machine"] = machine
            except BaseException as exc:
                holder["error"] = exc
            finally:
                event_queue.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while True:
            event = event_queue.get()
            if event is None:
                break
            yield event
        thread.join()
        if holder.get("error") is not None:
            raise holder["error"]

        result = holder["result"]
        machine = holder["machine"]
        yield {"type": "state_updated", "payload": machine.snapshot()}
        if result.final_answer:
            yield {"type": "answer_delta", "payload": {"text": result.final_answer}}
            yield {"type": "answer", "payload": {"answer": result.final_answer, "model": request.model}}
        yield {
            "type": "done",
            "payload": {
                "trace_id": result.trace_id,
                "trace_path": result.trace_path,
                "stopped_reason": result.stopped_reason,
                "error": result.error,
                "telemetry": {
                    "command": "InterviewAgentV2",
                    "agent_v2": True,
                    "interview_state": machine.snapshot(),
                    "derived_metrics": result.state.working.extra.get("derived_metrics", {}),
                    "citations": result.state.working.extra.get("citations", []),
                    "total_ms": result.total_ms,
                },
            },
        }


def working_from_machine(machine: InterviewStateMachine) -> WorkingMemory:
    snapshot = machine.snapshot()
    working = WorkingMemory(
        session_id=snapshot.get("session_id") or None,
        current_topic=snapshot.get("current_topic"),
        current_layer_index=int(snapshot.get("current_layer_index") or 0),
        follow_up_count=int(snapshot.get("follow_up_count") or 0),
        plan_topic_names=[str(item.get("name") or "") for item in snapshot.get("plan_topics", []) if item.get("name")],
    )
    working.extra["interview_state"] = snapshot
    return working


def sync_working_from_snapshot(working: WorkingMemory, snapshot: dict[str, Any]) -> None:
    working.session_id = snapshot.get("session_id") or working.session_id
    working.current_topic = snapshot.get("current_topic")
    working.current_layer_index = int(snapshot.get("current_layer_index") or 0)
    working.follow_up_count = int(snapshot.get("follow_up_count") or 0)
    working.plan_topic_names = [str(item.get("name") or "") for item in snapshot.get("plan_topics", []) if item.get("name")]
    working.extra["interview_state"] = snapshot


def build_interviewer_runtime_context(
    request: InterviewTurnRequest,
    machine: InterviewStateMachine,
    profile_store: Any | None = None,
) -> dict[str, Any]:
    snapshot = machine.snapshot()
    current_topic = snapshot.get("current_topic")
    current_layer = snapshot.get("current_layer_name", "")
    memory_context = _render_interviewer_memory_context(
        profile_store=profile_store,
        current_topic=str(current_topic or ""),
        planned_layer=str(current_layer or ""),
        scope_note_paths=tuple(request.scope_note_paths or ()),
    )
    profile_summary = build_profile_availability_counts(
        profile_store=profile_store,
        topic=current_topic,
        current_layer=current_layer,
        plan=request.interview_plan,
        universal_limit=0 if memory_context else 3,
    )
    preloaded = ["interview_state", "compact_plan", "scope_summary", "profile_layer_counts"]
    if memory_context:
        preloaded.append("learner_memory_background")
    else:
        preloaded.append("universal_profile_weak_points")
    return {
        "interview_mode": "mock",
        "session_id": request.session_id,
        "topic_phase": snapshot.get("topic_phase") or "awaiting_selection",
        "current_topic": current_topic,
        "current_topic_index": snapshot.get("current_topic_index", 0),
        "current_layer_index": snapshot.get("current_layer_index", 0),
        "current_layer_name": snapshot.get("current_layer_name", ""),
        "next_layer_name": snapshot.get("next_layer_name", ""),
        "at_last_layer": snapshot.get("at_last_layer", False),
        "last_assistant_question": snapshot.get("last_assistant_question", ""),
        "follow_up_count_before_this_turn": snapshot.get("follow_up_count", 0),
        "should_consider_layer_transition": snapshot.get("should_consider_layer_transition", False),
        "compact_plan": {
            "topic_count": len(snapshot.get("plan_topics", []) or []),
            "topics": snapshot.get("plan_topics", []),
        },
        "scope": {
            "type": request.scope_type or "all_vault",
            "value": request.scope_value or "",
            "allowed_note_count": len(request.scope_note_paths or ()),
        },
        "profile": profile_summary,
        "memory_context": memory_context,
        "tool_boundaries": {
            "preloaded": preloaded,
            "on_demand": ["search_notes", "grep_vault", "read_note", "recall_profile"],
            "actions": ["advance_layer", "select_topic"],
        },
    }


def _render_interviewer_memory_context(
    *,
    profile_store: Any | None,
    current_topic: str,
    planned_layer: str,
    scope_note_paths: tuple[str, ...],
) -> str:
    if profile_store is None:
        return ""
    try:
        from services.memory.injection import render_interviewer_memory_context

        model = profile_store.ensure_derived_fresh()
        return render_interviewer_memory_context(
            model=model,
            current_topic=current_topic,
            planned_layer=planned_layer,
            scope_note_paths=scope_note_paths,
        )
    except Exception:
        return ""


def build_profile_availability_counts(
    *,
    profile_store: Any | None,
    topic: Any,
    current_layer: Any = "",
    plan: Any | None = None,
    universal_limit: int = 3,
) -> dict[str, Any]:
    if profile_store is None:
        return {
            "profile_available": False,
            "current_topic": topic,
            "universal_weak_points": [],
            "domain_weak_by_layer": {},
            "current_layer_domain_weak_count": 0,
            "matching_weak_count": 0,
            "domain_weak_count": 0,
            "due_review_count": 0,
            "other_due_reviews_count": 0,
            "strong_point_count": 0,
        }
    try:
        from services.workflows.interview_profile import build_profile_runtime_summary

        profile = profile_store.load()
        return build_profile_runtime_summary(
            profile=profile,
            current_topic=topic,
            current_layer=current_layer,
            plan=plan,
            universal_limit=universal_limit,
        )
    except Exception:
        return {
            "profile_available": False,
            "current_topic": topic,
            "universal_weak_points": [],
            "domain_weak_by_layer": {},
            "current_layer_domain_weak_count": 0,
            "matching_weak_count": 0,
            "domain_weak_count": 0,
            "due_review_count": 0,
            "other_due_reviews_count": 0,
            "strong_point_count": 0,
        }


def build_turn_input(request: InterviewTurnRequest, *, runtime_context: dict[str, Any] | None = None) -> str:
    history = []
    for item in (request.chat_history or [])[-6:]:
        role = str(item.get("role") or "user")
        content = str(item.get("content") or "").strip()
        if content:
            history.append(f"{role}: {content}")
    context = runtime_context or {}
    sections: list[str] = []
    memory_context = str(context.get("memory_context") or "").strip()
    if memory_context:
        sections.extend(["# Learner Memory Background", memory_context, ""])
    sections.extend(
        [
            "# Runtime Context",
            json.dumps(context, ensure_ascii=False, indent=2),
            "",
            "# Short Conversation History",
        ]
    )
    return "\n\n".join(
        [
            *sections,
            "\n".join(history) if history else "(new interview session)",
            "",
            "# Current User Message",
            request.query,
            "",
            "# Task",
            (
                "Continue the mock interview from the authoritative runtime context above. "
                "Do not routinely call get_interview_state or list_plan_topics; those are debug/refresh tools. "
                "Use note/profile tools only when specific details are needed. "
                "Use advance_layer or select_topic only when you are intentionally changing server state. "
                "End with exactly one question."
            ),
        ]
    )


def persist_agent_trace(*, request: InterviewTurnRequest, result: Any, state_after: dict[str, Any]) -> None:
    store = request.session_store
    if store is None or not request.session_id:
        return
    if hasattr(store, "record_agent_turn"):
        store.record_agent_turn(
            session_id=request.session_id,
            skill="interviewer",
            trace_path=result.trace_path,
            trace_id=result.trace_id,
            interview_state=state_after,
        )


def derive_interview_metrics(*, result: Any, state_before: dict[str, Any], state_after: dict[str, Any]) -> dict[str, Any]:
    tool_names: list[str] = []
    read_paths: set[str] = set()
    for step in result.steps:
        for call in step.tool_calls:
            tool_names.append(call.name)
        for tool_result in step.tool_results:
            if tool_result.name == "read_note" and tool_result.ok and isinstance(tool_result.output, dict):
                path = str(tool_result.output.get("path") or "").strip()
                if path:
                    read_paths.add(path)
    search_count = sum(1 for name in tool_names if name in {"search_notes", "grep_vault"})
    return {
        "routine_state_fetch": sum(1 for name in tool_names if name in {"get_interview_state", "list_plan_topics"}),
        "notes_read": len(read_paths),
        "profile_recalled": sum(1 for name in tool_names if name == "recall_profile"),
        "layer_advanced": transition_type_added(state_before, state_after, "advance_layer"),
        "topic_selected": transition_type_added(state_before, state_after, "select_topic"),
        "over_search": search_count > 2 and not read_paths,
    }


def transition_type_added(state_before: dict[str, Any], state_after: dict[str, Any], transition_type: str) -> bool:
    before_count = len(state_before.get("transition_history") or [])
    after = list(state_after.get("transition_history") or [])
    for transition in after[before_count:]:
        if isinstance(transition, dict) and transition.get("type") == transition_type:
            return True
    return False


def rewrite_trace_interview_metadata(
    *,
    result: Any,
    state_before: dict[str, Any],
    state_after: dict[str, Any],
    runtime_context: dict[str, Any],
) -> None:
    if not result.trace_path:
        return
    path = Path(result.trace_path)
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    derived_metrics = derive_interview_metrics(result=result, state_before=state_before, state_after=state_after)
    payload["working_memory"] = to_jsonable(result.state.working)
    payload["runtime_context"] = to_jsonable(runtime_context)
    payload["derived_metrics"] = to_jsonable(derived_metrics)
    payload["interview"] = {
        "state_before": to_jsonable(state_before),
        "state_after": to_jsonable(state_after),
        "state_transitions": to_jsonable(state_after.get("transition_history", [])),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
