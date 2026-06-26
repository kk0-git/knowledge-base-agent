from __future__ import annotations

import json
import re
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
        parsed_payload = parse_final_json(result.final_answer)
        parse_error = str(parsed_payload.get("_parse_error") or "").strip()
        repaired = bool(parsed_payload.get("_repaired"))
        payload = normalize_coach_payload(parsed_payload)
        payload["profile_signals"] = []
        if not payload.get("context_note_paths"):
            payload["context_note_paths"] = list(request.context_note_paths)
        payload["available"] = result.stopped_reason == "final" and not parse_error
        if parse_error:
            payload["error"] = "本轮复盘生成失败，可重试。"
        if repaired:
            payload["repaired"] = True
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
        if request.save_review and payload.get("available") is not False:
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
            "Use ONLY for interviewer_followup_note and thinking_framework. Do NOT use it for expression_example.",
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
            "Step 3 — Follow-up interpretation: Read Follow-up Zone only for interviewer_followup_note and thinking_framework.",
            "Step 4 — Expression example: Write expression_example from Evaluation Zone only (question_requires + gaps). Do NOT incorporate Follow-up Zone direction into the example.",
            "",
            "Return only the required JSON object. Write feedback in Simplified Chinese.",
        ]
    )


def parse_final_json(text: str) -> dict[str, Any]:
    try:
        return parse_json_object(text)
    except Exception as exc:
        repaired = repair_turn_review_json(text)
        if repaired:
            repaired["_repaired"] = True
            return repaired
        return empty_parse_failure(str(exc))


def empty_parse_failure(error: str = "") -> dict[str, Any]:
    return {
        "_parse_error": error or "invalid turn review JSON",
        "feedback": {
            "question_requires": [],
            "coach_note": "",
            "covered": [],
            "gaps": [],
            "thinking_framework": "",
            "interviewer_followup_note": "",
        },
        "expression_example": "",
        "profile_signals": [],
    }


def repair_turn_review_json(text: str) -> dict[str, Any]:
    raw = str(text or "")
    if not raw.strip():
        return {}
    feedback_block = extract_object_block(raw, "feedback")
    feedback = {
        "question_requires": extract_string_array(feedback_block, "question_requires"),
        "coach_note": extract_string_field(feedback_block, "coach_note"),
        "covered": extract_string_array(feedback_block, "covered"),
        "gaps": extract_string_array(feedback_block, "gaps"),
        "thinking_framework": extract_string_field(feedback_block, "thinking_framework"),
        "interviewer_followup_note": extract_string_field(feedback_block, "interviewer_followup_note"),
    }
    expression_example = extract_string_field(raw, "expression_example")
    has_content = any(feedback[key] for key in feedback) or bool(expression_example)
    if not has_content:
        return {}
    return {
        "feedback": feedback,
        "expression_example": expression_example,
        "profile_signals": [],
    }


def extract_object_block(text: str, key: str) -> str:
    marker = re.search(rf'"{re.escape(key)}"\s*:\s*\{{', text)
    if not marker:
        return text
    start = marker.end() - 1
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def extract_string_field(text: str, key: str) -> str:
    marker = re.search(rf'"{re.escape(key)}"\s*:\s*"', text)
    if not marker:
        return ""
    start = marker.end()
    # A field value ends at the quote before the next JSON key, object close, or array close.
    end_pattern = re.compile(r'"\s*(?:,\s*"[A-Za-z_][A-Za-z0-9_]*"\s*:|\s*[}\]])', re.DOTALL)
    best_end = -1
    for match in end_pattern.finditer(text, start):
        best_end = match.start()
        break
    if best_end < start:
        return ""
    return decode_jsonish_string(text[start:best_end])


def extract_string_array(text: str, key: str) -> list[str]:
    marker = re.search(rf'"{re.escape(key)}"\s*:\s*\[', text)
    if not marker:
        return []
    start = marker.end()
    depth = 1
    end = -1
    for index in range(start, len(text)):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                end = index
                break
    if end < start:
        return []
    body = text[start:end]
    values: list[str] = []
    for line in body.splitlines():
        item = line.strip().rstrip(",").strip()
        if len(item) >= 2 and item[0] == '"' and item[-1] == '"':
            value = decode_jsonish_string(item[1:-1])
            if value:
                values.append(value)
    if values:
        return values
    try:
        parsed = json.loads(f"[{body}]")
    except Exception:
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def decode_jsonish_string(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(json.loads(f'"{text}"')).strip()
    except Exception:
        return text.replace("\\n", "\n").replace('\\"', '"').strip()


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
        "profile_signals_disabled": True,
        "profile_write_source": "session_end_extractor",
        "profile_signal_count": len(signals),
        "profile_signals": to_jsonable(signals),
        "observation_draft_count": len(drafts),
        "observation_drafts": to_jsonable(drafts),
    }
    coach_evaluation = result.state.working.extra.get("coach_evaluation")
    if isinstance(coach_evaluation, dict):
        payload["coach_evaluation"] = to_jsonable(coach_evaluation)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
