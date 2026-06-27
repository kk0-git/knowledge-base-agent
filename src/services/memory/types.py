"""Shared constants and normalizers for learner memory v4."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any


LIFECYCLES = {"candidate", "active", "archived"}
SOURCE_KINDS = {"interview", "answer", "review", "user"}
FACETS = {"knowledge", "behavior"}

# Deprecated: kept for backward compatibility during v4→v5 migration.
# New code should use FACETS and normalize_facet.
CATEGORIES = {
    "knowledge_gap",
    "answer_structure",
    "communication",
    "thinking_pattern",
}
BELIEF_KINDS = {"standard", "confusion_pair"}
OBSERVATION_OPS = {
    "propose_belief",
    "propose_procedure",
    "schedule_pass",
    "schedule_retry",
    "improvement",
    "user_commit",
}
CONFIDENCES = {"low", "medium", "high"}
SCOPES = {"domain", "universal"}


def today_iso() -> str:
    return date.today().isoformat()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_enum(value: Any, allowed: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def normalize_lifecycle(value: Any, default: str = "candidate") -> str:
    return _normalize_enum(value, LIFECYCLES, default)


def normalize_source_kind(value: Any, default: str = "interview") -> str:
    return _normalize_enum(value, SOURCE_KINDS, default)


def normalize_belief_kind(value: Any, default: str = "standard") -> str:
    return _normalize_enum(value, BELIEF_KINDS, default)


def normalize_observation_op(value: Any, default: str = "propose_belief") -> str:
    return _normalize_enum(value, OBSERVATION_OPS, default)


def normalize_confidence(value: Any, default: str = "medium") -> str:
    text = str(value or "").strip().lower()
    if text in {"strong"}:
        text = "high"
    if text in {"weak"}:
        text = "low"
    return text if text in CONFIDENCES else default


def normalize_facet(value: Any, default: str = "knowledge") -> str:
    text = str(value or "").strip().lower()
    # Map v4 categories and aliases to v5 facets
    knowledge_aliases = {
        "knowledge", "knowledge_gap", "domain", "gap", "weak_point",
    }
    behavior_aliases = {
        "behavior", "answer_structure", "answer", "structure",
        "thinking_pattern", "thinking", "pattern", "communication",
    }
    if text in knowledge_aliases:
        return "knowledge"
    if text in behavior_aliases:
        return "behavior"
    return default


def normalize_category(value: Any, default: str = "knowledge_gap") -> str:
    """Deprecated alias — maps to normalize_facet then back to v4 category for compatibility."""
    facet = normalize_facet(value, default="knowledge")
    return "knowledge_gap" if facet == "knowledge" else "answer_structure"


def normalize_scope(value: Any, default: str = "domain") -> str:
    text = str(value or "").strip().lower()
    return text if text in SCOPES else default


def normalize_float(value: Any, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def normalize_int(value: Any, default: int = 0, *, minimum: int | None = None) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    if minimum is not None:
        number = max(minimum, number)
    return number


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def unique_strings(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in ensure_list(values):
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def default_sr(today: str | None = None) -> dict[str, Any]:
    day = today or today_iso()
    return {
        "interval_days": 1,
        "ease_factor": 2.5,
        "repetitions": 0,
        "last_reviewed": "",
        "last_outcome": "",
        "next_review": day,
    }
