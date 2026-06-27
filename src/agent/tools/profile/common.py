from __future__ import annotations

from typing import Any


SIGNAL_TYPES = {"possible_weak_point", "possible_partial", "possible_improvement"}
FACETS = {"knowledge", "behavior"}
# Deprecated alias — kept for backward compat with agent tool schemas
CATEGORIES = {"knowledge_gap", "answer_structure", "communication", "thinking_pattern"}
SCOPES = {"domain", "universal"}
CONFIDENCE = {"low", "medium", "high"}


def current_topic(ctx: Any, fallback: str = "") -> str:
    topic = str((getattr(ctx, "turn_context", {}) or {}).get("current_topic") or "").strip()
    if topic:
        return topic
    working = getattr(ctx, "working", None)
    if working and getattr(working, "current_topic", None):
        return str(working.current_topic or "").strip()
    state = getattr(ctx, "interview_state", None)
    if isinstance(state, dict) and state.get("current_topic"):
        return str(state.get("current_topic") or "").strip()
    return fallback


def context_note_paths(ctx: Any) -> list[str]:
    working = getattr(ctx, "working", None)
    paths = list(getattr(working, "notes_read_this_turn", []) or []) if working else []
    return [str(path).replace("\\", "/") for path in paths if str(path).strip()]


def append_session_memory_signal(ctx: Any, signal: dict[str, Any]) -> None:
    store = getattr(ctx, "session_store", None)
    session_id = getattr(ctx, "session_id", "")
    if store is not None and session_id and hasattr(store, "append_memory_signal"):
        store.append_memory_signal(session_id=session_id, signal=signal)


def append_session_observation_draft(ctx: Any, draft: dict[str, Any]) -> None:
    store = getattr(ctx, "session_store", None)
    session_id = getattr(ctx, "session_id", "")
    if store is not None and session_id and hasattr(store, "append_observation_draft"):
        store.append_observation_draft(session_id=session_id, draft=draft)


def load_session_memory_signals(ctx: Any) -> list[dict[str, Any]]:
    store = getattr(ctx, "session_store", None)
    session_id = getattr(ctx, "session_id", "")
    if store is not None and session_id and hasattr(store, "list_memory_signals"):
        return store.list_memory_signals(session_id=session_id)
    return []


def normalize_enum(value: Any, allowed: set[str], default: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else default
