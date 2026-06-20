from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.llm.tool_calling import parse_json_object
from agent.runtime import AgentRuntime
from agent.schema import AgentRunConfig, AgentState, WorkingMemory
from agent.serialization import to_jsonable
from agent.tool_executor import ToolExecutionContext


@dataclass
class CoachTurnRequest:
    session_id: str = ""
    user_message_id: str = ""
    assistant_message_id: str = ""
    previous_interviewer_question: str = ""
    latest_user_answer: str = ""
    interviewer_followup: str = ""
    chat_history: list[dict[str, Any]] = field(default_factory=list)
    interview_plan: Any | None = None
    interview_state: dict[str, Any] | None = None
    context_note_paths: tuple[str, ...] = ()
    vault_root: Path | None = None
    session_store: Any | None = None
    profile_store: Any | None = None
    model: str = ""
    tool_mode: str = "auto"
    trace_path: str | None = None
    max_steps: int = 4
    max_tool_calls_per_step: int = 4
    temperature: float = 0.2
    save_review: bool = True


class InterviewCoachApp:
    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime

    def run_review(self, request: CoachTurnRequest) -> tuple[dict[str, Any], Any]:
        working = WorkingMemory(
            session_id=request.session_id or None,
            current_topic=(request.interview_state or {}).get("current_topic") if isinstance(request.interview_state, dict) else None,
            current_layer_index=int((request.interview_state or {}).get("current_layer_index") or 0) if isinstance(request.interview_state, dict) else 0,
            follow_up_count=int((request.interview_state or {}).get("follow_up_count") or 0) if isinstance(request.interview_state, dict) else 0,
            notes_read_this_turn=list(request.context_note_paths),
        )
        state = AgentState(messages=[], working=working, skill_name="coach")
        turn_context = {
            "turn_id": "",
            "user_message_id": request.user_message_id,
            "assistant_message_id": request.assistant_message_id,
            "previous_interviewer_question": request.previous_interviewer_question,
            "latest_user_answer": request.latest_user_answer,
            "interviewer_followup": request.interviewer_followup,
            "context_note_paths": list(request.context_note_paths),
            "current_topic": working.current_topic or "",
            "planned_layer": (request.interview_state or {}).get("current_layer_name", "") if isinstance(request.interview_state, dict) else "",
        }
        tool_context = ToolExecutionContext(
            working=working,
            vault_root=request.vault_root,
            scope_note_paths=request.context_note_paths,
            session_id=request.session_id,
            session_store=request.session_store,
            interview_plan=request.interview_plan,
            interview_state=request.interview_state,
            profile_store=request.profile_store,
            turn_context=turn_context,
        )
        result = self.runtime.run(
            config=AgentRunConfig(
                skill_name="coach",
                max_steps=request.max_steps,
                max_tool_calls_per_step=request.max_tool_calls_per_step,
                temperature=request.temperature,
                model=request.model,
                tool_mode=request.tool_mode,  # type: ignore[arg-type]
                trace_path=request.trace_path,
            ),
            user_input=build_coach_input(request),
            state=state,
            tool_context=tool_context,
        )
        payload = normalize_coach_payload(parse_final_json(result.final_answer))
        payload["profile_signals"] = []
        if not payload.get("context_note_paths"):
            payload["context_note_paths"] = list(request.context_note_paths)
        payload["available"] = result.stopped_reason == "final"
        payload["trace_id"] = result.trace_id
        payload["trace_path"] = result.trace_path
        payload["stopped_reason"] = result.stopped_reason
        payload["latency_ms"] = result.total_ms
        result.state.working.extra["profile_signals"] = []
        result.state.working.extra["observation_drafts"] = tool_context.observation_drafts
        result.state.working.extra["coach_evaluation"] = {
            "question_requires": payload.get("feedback", {}).get("question_requires", []),
            "previous_interviewer_question": request.previous_interviewer_question,
        }
        rewrite_trace_memory_metadata(result=result, signals=[], drafts=tool_context.observation_drafts)
        if request.save_review:
            persist_review(request=request, payload=payload, result=result)
        return payload, result


def build_coach_input(request: CoachTurnRequest) -> str:
    state = request.interview_state if isinstance(request.interview_state, dict) else {}
    reference_context = {
        "current_topic": state.get("current_topic") or "",
        "planned_layer": state.get("current_layer_name") or "",
        "context_note_paths": list(request.context_note_paths),
        "note": "Optional paths only. Notes are not the default evaluation ground truth.",
    }
    return "\n\n".join(
        [
            "# Evaluation Zone",
            "Use ONLY this zone to derive question_requires, covered, gaps, and coach_note.",
            "Do not read the Follow-up Zone while deciding gaps.",
            "",
            "## Previous Interviewer Question",
            request.previous_interviewer_question or "(not available)",
            "",
            "## User Latest Answer",
            request.latest_user_answer or "(not available)",
            "",
            "# Follow-up Zone (read-only for evaluation)",
            "Use ONLY for interviewer_followup_note, thinking_framework, and expression_example.",
            "Do NOT copy follow-up dimensions into gaps.",
            "",
            request.interviewer_followup or "(not available)",
            "",
            "# Reference Context",
            json.dumps(reference_context, ensure_ascii=False, indent=2),
            "",
            "# Task",
            "Step 1 — Question decomposition: From Previous Interviewer Question alone, list question_requires.",
            "Step 2 — Coverage evaluation: Compare User Latest Answer to question_requires. Produce covered and gaps.",
            "Step 3 — Follow-up interpretation: Read Follow-up Zone only for interviewer_followup_note, thinking_framework, expression_example.",
            "",
            "Return only the required JSON object. Write feedback in Simplified Chinese.",
        ]
    )


def parse_final_json(text: str) -> dict[str, Any]:
    try:
        return parse_json_object(text)
    except Exception:
        return {
            "feedback": {
                "question_requires": [],
                "coach_note": text.strip(),
                "covered": [],
                "gaps": [],
                "thinking_framework": "",
                "interviewer_followup_note": "",
            },
            "expression_example": "",
            "profile_signals": [],
        }


def normalize_coach_payload(payload: dict[str, Any]) -> dict[str, Any]:
    feedback = payload.get("feedback") if isinstance(payload.get("feedback"), dict) else {}
    return {
        "feedback": {
            "question_requires": normalize_string_list(feedback.get("question_requires") or []),
            "coach_note": str(feedback.get("coach_note") or "").strip(),
            "covered": normalize_string_list(feedback.get("covered") or []),
            "gaps": normalize_string_list(feedback.get("gaps") or []),
            "thinking_framework": str(feedback.get("thinking_framework") or "").strip(),
            "interviewer_followup_note": str(feedback.get("interviewer_followup_note") or "").strip(),
        },
        "expression_example": str(payload.get("expression_example") or payload.get("reference_answer") or "").strip(),
        "reference_answer": str(payload.get("reference_answer") or payload.get("expression_example") or "").strip(),
        "profile_signals": [],
        "context_note_paths": normalize_string_list(payload.get("context_note_paths") or []),
    }


def normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def merge_signals(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for signal in [*(left or []), *(right or [])]:
        if not isinstance(signal, dict):
            continue
        key = (str(signal.get("type") or ""), str(signal.get("point") or signal.get("summary") or ""), str(signal.get("evidence") or ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append(signal)
    return merged


def persist_review(*, request: CoachTurnRequest, payload: dict[str, Any], result: Any) -> None:
    store = request.session_store
    if store is None or not request.session_id or not request.user_message_id or not request.assistant_message_id:
        return
    if hasattr(store, "save_turn_review"):
        store.save_turn_review(
            session_id=request.session_id,
            user_message_id=request.user_message_id,
            assistant_message_id=request.assistant_message_id,
            feedback=payload.get("feedback", {}),
            reference_answer=payload.get("expression_example") or payload.get("reference_answer") or "",
            context_note_paths=payload.get("context_note_paths") or list(request.context_note_paths),
            profile_signals=payload.get("profile_signals") or [],
        )
    if hasattr(store, "record_agent_turn"):
        store.record_agent_turn(
            session_id=request.session_id,
            skill="coach",
            trace_path=result.trace_path,
            trace_id=result.trace_id,
            interview_state=request.interview_state if isinstance(request.interview_state, dict) else None,
        )


def rewrite_trace_memory_metadata(*, result: Any, signals: list[dict[str, Any]], drafts: list[dict[str, Any]]) -> None:
    if not result.trace_path:
        return
    path = Path(result.trace_path)
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    payload["working_memory"] = to_jsonable(result.state.working)
    payload["memory"] = {
        "profile_signal_count": len(signals),
        "profile_signals": to_jsonable(signals),
        "observation_draft_count": len(drafts),
        "observation_drafts": to_jsonable(drafts),
    }
    coach_evaluation = result.state.working.extra.get("coach_evaluation")
    if isinstance(coach_evaluation, dict):
        payload["coach_evaluation"] = to_jsonable(coach_evaluation)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
