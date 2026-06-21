from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


TRANSITION_THRESHOLD = 4
MIN_FOLLOW_UPS_FOR_ADVANCE = 2
TOPIC_AWAITING_SELECTION = "awaiting_selection"
TOPIC_ACTIVE = "active"
TOPIC_CLOSING = "closing"
TOPIC_PHASES = {TOPIC_AWAITING_SELECTION, TOPIC_ACTIVE, TOPIC_CLOSING}
QUESTION_MARKS = ("?", "？")
ASSISTANT_ANCHOR_CHARS = 160


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class InterviewState:
    source: str = "server"
    session_id: str = ""
    current_topic: str | None = None
    current_topic_index: int = 0
    topic_phase: str = TOPIC_AWAITING_SELECTION
    topic_selection_source: str = ""
    current_layer_index: int = 0
    current_layer_name: str = ""
    follow_up_count: int = 0
    sub_points_touched: list[str] = field(default_factory=list)
    last_user_answer: str = ""
    last_assistant_question: str = ""
    transition_history: list[dict[str, Any]] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "session_id": self.session_id,
            "current_topic": self.current_topic,
            "current_topic_index": self.current_topic_index,
            "topic_phase": self.topic_phase,
            "topic_selection_source": self.topic_selection_source,
            "current_layer_index": self.current_layer_index,
            "current_layer_name": self.current_layer_name,
            "follow_up_count": self.follow_up_count,
            "sub_points_touched": list(self.sub_points_touched),
            "last_user_answer": self.last_user_answer,
            "last_assistant_question": self.last_assistant_question,
            "transition_history": list(self.transition_history),
            "updated_at": self.updated_at,
        }


class InterviewStateMachine:
    def __init__(self, *, plan: Any | None = None, state: InterviewState | None = None):
        self.plan = plan
        self.state = state or initialize_interview_state(plan=plan)
        self.refresh_plan_alignment()

    def snapshot(self) -> dict[str, Any]:
        topic = self.current_topic_card()
        coverage = topic_coverage(topic)
        next_layer = coverage[self.state.current_layer_index + 1] if self.state.current_layer_index + 1 < len(coverage) else ""
        payload = self.state.to_dict()
        payload.update(
            {
                "coverage": coverage,
                "current_layer_name": self.current_layer_name(),
                "next_layer_name": next_layer,
                "should_consider_layer_transition": self.should_consider_layer_transition(),
                "at_last_layer": bool(coverage) and self.state.current_layer_index >= len(coverage) - 1,
                "plan_topics": plan_topics_summary(self.plan, include_sources=False),
            }
        )
        return payload

    def commit_turn(self, *, user_text: str, assistant_text: str) -> InterviewState:
        self.state.last_user_answer = truncate_one_line(user_text, 300)
        question = extract_last_question(assistant_text)
        self.state.last_assistant_question = question
        if question:
            # Stores recent interviewer probes or fallback reply anchors, not only
            # punctuation-terminated questions.
            append_unique_tail(self.state.sub_points_touched, question, max_items=8)
        if self.state.topic_phase == TOPIC_ACTIVE and self.state.current_topic:
            self.state.follow_up_count = max(0, int(self.state.follow_up_count or 0)) + 1
        self.state.current_layer_name = self.current_layer_name()
        self.state.updated_at = utc_now()
        return self.state

    def select_topic(self, *, name: str, reason: str = "", source: str = "agent") -> dict[str, Any]:
        requested = str(name or "").strip()
        if not requested:
            return {
                "ok": False,
                "selected": False,
                "message": "topic name is required",
                "state": self.snapshot(),
            }
        topics = plan_topic_cards(self.plan)
        matched_index = -1
        matched_topic = None
        for index, topic in enumerate(topics):
            if topic_name(topic) == requested:
                matched_index = index
                matched_topic = topic
                break
        if matched_topic is None:
            return {
                "ok": False,
                "selected": False,
                "message": "topic not found in interview plan",
                "requested_topic": requested,
                "available_topics": [topic_name(topic) for topic in topics],
                "state": self.snapshot(),
            }

        before = self.state.to_dict()
        self.state.current_topic = topic_name(matched_topic)
        self.state.current_topic_index = matched_index
        self.state.topic_phase = TOPIC_ACTIVE
        self.state.topic_selection_source = truncate_one_line(source or "agent", 80)
        self.state.current_layer_index = 0
        self.state.current_layer_name = self.current_layer_name()
        self.state.follow_up_count = 0
        self.state.sub_points_touched = []
        transition = {
            "type": "select_topic",
            "reason": truncate_one_line(reason, 300),
            "source": self.state.topic_selection_source,
            "from_topic": before.get("current_topic"),
            "to_topic": self.state.current_topic,
            "from_topic_phase": before.get("topic_phase", TOPIC_AWAITING_SELECTION),
            "to_topic_phase": self.state.topic_phase,
            "created_at": utc_now(),
        }
        self.state.transition_history.append(transition)
        self.state.transition_history = self.state.transition_history[-20:]
        self.state.updated_at = utc_now()
        self.sync_topic_phase()
        return {
            "ok": True,
            "selected": True,
            "transition": transition,
            "state": self.snapshot(),
        }

    def sync_topic_phase(self) -> None:
        if not self.state.current_topic:
            self.state.topic_phase = TOPIC_AWAITING_SELECTION
            return
        coverage = topic_coverage(self.current_topic_card())
        if not coverage:
            return
        at_last = self.state.current_layer_index >= len(coverage) - 1
        if at_last:
            if self.state.topic_phase != TOPIC_AWAITING_SELECTION:
                self.state.topic_phase = TOPIC_CLOSING
            return
        if self.state.topic_phase == TOPIC_CLOSING:
            self.state.topic_phase = TOPIC_ACTIVE

    def advance_layer(self, *, reason: str, force: bool = False) -> dict[str, Any]:
        before = self.state.to_dict()
        coverage = topic_coverage(self.current_topic_card())
        if not coverage:
            return {
                "ok": False,
                "advanced": False,
                "reason": reason,
                "message": "current topic has no coverage layers",
                "state": self.snapshot(),
            }
        if self.state.current_layer_index >= len(coverage) - 1:
            return {
                "ok": False,
                "advanced": False,
                "reason": reason,
                "at_last_layer": True,
                "message": "already at last layer",
                "state": self.snapshot(),
            }
        if not force and self.state.follow_up_count < MIN_FOLLOW_UPS_FOR_ADVANCE:
            return {
                "ok": False,
                "advanced": False,
                "reason": reason,
                "follow_up_count": self.state.follow_up_count,
                "required_follow_up_count": MIN_FOLLOW_UPS_FOR_ADVANCE,
                "message": "not enough follow-up signal to advance layer",
                "state": self.snapshot(),
            }

        self.state.current_layer_index += 1
        self.state.current_layer_name = self.current_layer_name()
        self.state.follow_up_count = 0
        self.state.sub_points_touched = []
        transition = {
            "type": "advance_layer",
            "reason": truncate_one_line(reason, 300),
            "from_layer_index": before.get("current_layer_index", 0),
            "to_layer_index": self.state.current_layer_index,
            "from_layer_name": before.get("current_layer_name", ""),
            "to_layer_name": self.state.current_layer_name,
            "created_at": utc_now(),
        }
        self.state.transition_history.append(transition)
        self.state.transition_history = self.state.transition_history[-20:]
        self.state.updated_at = utc_now()
        self.sync_topic_phase()
        return {
            "ok": True,
            "advanced": True,
            "transition": transition,
            "state": self.snapshot(),
        }

    def should_consider_layer_transition(self) -> bool:
        return max(0, int(self.state.follow_up_count or 0)) >= TRANSITION_THRESHOLD

    def refresh_plan_alignment(self) -> None:
        topics = plan_topic_cards(self.plan)
        if not topics:
            self.state.current_layer_name = self.current_layer_name()
            return
        if self.state.current_topic:
            for index, topic in enumerate(topics):
                if topic_name(topic) == self.state.current_topic:
                    self.state.current_topic_index = index
                    self.state.topic_phase = normalize_topic_phase(self.state.topic_phase, has_topic=True)
                    break
        else:
            self.state.current_topic_index = 0
            self.state.current_layer_index = 0
            self.state.topic_phase = TOPIC_AWAITING_SELECTION
        coverage = topic_coverage(self.current_topic_card())
        if coverage:
            self.state.current_layer_index = clamp_index(self.state.current_layer_index, len(coverage))
        self.state.current_layer_name = self.current_layer_name()
        self.sync_topic_phase()

    def current_topic_card(self) -> Any | None:
        topics = plan_topic_cards(self.plan)
        if not topics:
            return None
        if self.state.current_topic:
            for topic in topics:
                if topic_name(topic) == self.state.current_topic:
                    return topic
        return None

    def current_layer_name(self) -> str:
        coverage = topic_coverage(self.current_topic_card())
        if not coverage:
            return self.state.current_layer_name or ""
        return coverage[clamp_index(self.state.current_layer_index, len(coverage))]


def build_interview_state_machine(
    *,
    plan: Any | None = None,
    state_payload: dict[str, Any] | None = None,
    session_id: str = "",
) -> InterviewStateMachine:
    state = interview_state_from_payload(state_payload, plan=plan, session_id=session_id)
    return InterviewStateMachine(plan=plan, state=state)


def interview_state_from_payload(
    payload: dict[str, Any] | None,
    *,
    plan: Any | None = None,
    session_id: str = "",
) -> InterviewState:
    if not payload:
        state = initialize_interview_state(plan=plan)
    else:
        current_topic = str(payload.get("current_topic") or "").strip() or None
        state = InterviewState(
            source="server",
            session_id=str(payload.get("session_id") or session_id or ""),
            current_topic=current_topic,
            current_topic_index=max(0, parse_int(payload.get("current_topic_index"), 0)),
            topic_phase=normalize_topic_phase(payload.get("topic_phase"), has_topic=bool(current_topic)),
            topic_selection_source=str(payload.get("topic_selection_source") or "").strip(),
            current_layer_index=max(0, parse_int(payload.get("current_layer_index"), 0)),
            current_layer_name=str(payload.get("current_layer_name") or "").strip(),
            follow_up_count=max(0, parse_int(payload.get("follow_up_count"), 0)),
            sub_points_touched=dedupe_strings(payload.get("sub_points_touched") or [], max_items=8),
            last_user_answer=str(payload.get("last_user_answer") or "").strip(),
            last_assistant_question=str(payload.get("last_assistant_question") or "").strip(),
            transition_history=normalize_transition_history(payload.get("transition_history") or []),
            updated_at=str(payload.get("updated_at") or utc_now()),
        )
    if session_id and not state.session_id:
        state.session_id = session_id
    return state


def initialize_interview_state(*, plan: Any | None = None) -> InterviewState:
    return InterviewState(
        current_topic_index=0,
        topic_phase=TOPIC_AWAITING_SELECTION,
        current_layer_index=0,
        current_layer_name="",
    )


def default_interview_state_dict(*, session_id: str = "") -> dict[str, Any]:
    state = initialize_interview_state()
    if session_id:
        state.session_id = session_id
    return state.to_dict()


def normalize_interview_state_payload(
    payload: dict[str, Any] | None,
    *,
    plan: Any | None = None,
    session_id: str = "",
) -> dict[str, Any]:
    machine = build_interview_state_machine(plan=plan, state_payload=payload, session_id=session_id)
    snapshot = machine.snapshot()
    snapshot["source"] = "server"
    return snapshot


def advance_interview_layer(machine: InterviewStateMachine, *, reason: str, force: bool = False) -> dict[str, Any]:
    return machine.advance_layer(reason=reason, force=force)


def plan_topic_cards(plan: Any | None) -> list[Any]:
    if isinstance(plan, dict):
        topics = plan.get("topics")
        return list(topics or []) if isinstance(topics, list) else []
    topics = getattr(plan, "topics", None)
    return list(topics or []) if topics else []


def plan_topics_summary(plan: Any | None, *, include_sources: bool) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, topic in enumerate(plan_topic_cards(plan)):
        item: dict[str, Any] = {
            "index": index,
            "name": topic_name(topic),
            "coverage": topic_coverage(topic),
        }
        if include_sources:
            item["source_note_paths"] = topic_source_paths(topic)
        items.append(item)
    return items


def topic_name(topic: Any | None) -> str:
    if topic is None:
        return ""
    if isinstance(topic, dict):
        return str(topic.get("name") or "").strip()
    return str(getattr(topic, "name", "") or "").strip()


def topic_coverage(topic: Any | None) -> list[str]:
    if topic is None:
        return []
    value = topic.get("coverage", ()) if isinstance(topic, dict) else getattr(topic, "coverage", ())
    return [str(item).strip() for item in (value or []) if str(item).strip()]


def topic_source_paths(topic: Any | None) -> list[str]:
    if topic is None:
        return []
    value = topic.get("source_note_paths", ()) if isinstance(topic, dict) else getattr(topic, "source_note_paths", ())
    return [str(path).replace("\\", "/") for path in (value or []) if str(path).strip()]


def normalize_topic_phase(value: Any, *, has_topic: bool) -> str:
    phase = str(value or "").strip()
    if phase in TOPIC_PHASES:
        return phase
    return TOPIC_ACTIVE if has_topic else TOPIC_AWAITING_SELECTION


def extract_last_question(text: str) -> str:
    clean = truncate_one_line(str(text or ""), 600)
    if not clean:
        return ""
    parts = re.split(r"(?<=[?？])", clean)
    questions = [part.strip() for part in parts if part.strip().endswith(QUESTION_MARKS)]
    if questions:
        return truncate_one_line(questions[-1], ASSISTANT_ANCHOR_CHARS)
    # Fallback mirrors MemCoach-style anchoring: keep a stable reply anchor even
    # when the interviewer asks an imperative question without punctuation.
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    fallback = lines[-1] if lines else clean
    return truncate_one_line(fallback, ASSISTANT_ANCHOR_CHARS)


def append_unique_tail(items: list[str], value: str, *, max_items: int) -> None:
    text = str(value or "").strip()
    if not text:
        return
    items[:] = [item for item in items if item != text]
    items.append(text)
    if len(items) > max_items:
        del items[:-max_items]


def dedupe_strings(value: Any, *, max_items: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= max_items:
            break
    return result


def normalize_transition_history(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            result.append(dict(item))
        if len(result) >= 20:
            break
    return result


def clamp_index(index: int, length: int) -> int:
    if length <= 0:
        return 0
    return max(0, min(index, length - 1))


def parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def truncate_one_line(text: str, max_chars: int) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."
