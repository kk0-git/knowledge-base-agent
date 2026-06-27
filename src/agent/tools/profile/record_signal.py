from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agent.schema import ToolSpec
from agent.tools.profile.common import (
    CATEGORIES,
    CONFIDENCE,
    FACETS,
    SCOPES,
    SIGNAL_TYPES,
    append_session_memory_signal,
    context_note_paths,
    current_topic,
    normalize_enum,
)


def record_signal(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
    turn_context = getattr(ctx, "turn_context", {}) or {}
    working = getattr(ctx, "working", None)
    signal = {
        "type": normalize_enum(args.get("signal_type") or args.get("type"), SIGNAL_TYPES, "possible_weak_point"),
        "point": str(args.get("point") or args.get("summary") or "").strip(),
        "topic": str(args.get("topic") or current_topic(ctx) or "").strip(),
        "planned_layer": str(args.get("planned_layer") or turn_context.get("planned_layer") or "").strip(),
        "category": normalize_enum(args.get("category"), FACETS | CATEGORIES, "knowledge"),
        "scope_suggestion": normalize_enum(args.get("scope_suggestion") or args.get("scope"), SCOPES, "domain"),
        "evidence": str(args.get("evidence") or "").strip(),
        "confidence": normalize_enum(args.get("confidence"), CONFIDENCE, "medium"),
        "weak_point_ref": str(args.get("weak_point_ref") or "").strip(),
        "context_note_paths": context_note_paths(ctx),
        "session_id": getattr(ctx, "session_id", "") or "",
        "turn_id": str(turn_context.get("turn_id") or "").strip(),
        "user_message_id": str(turn_context.get("user_message_id") or "").strip(),
        "assistant_message_id": str(turn_context.get("assistant_message_id") or "").strip(),
        "current_layer_index": getattr(working, "current_layer_index", 0) if working else 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if working is not None:
        working.signals_this_turn.append(signal)
    ctx.profile_signals.append(signal)
    append_session_memory_signal(ctx, signal)
    return {"recorded": True, "signal": signal}


def record_signal_spec() -> ToolSpec:
    return ToolSpec(
        name="record_signal",
        description="Record a session-level profile signal without committing long-term profile memory.",
        parameters={
            "type": "object",
            "properties": {
                "signal_type": {"type": "string", "enum": sorted(SIGNAL_TYPES)},
                "point": {"type": "string"},
                "topic": {"type": "string"},
                "planned_layer": {"type": "string"},
                "category": {"type": "string", "enum": sorted(FACETS)},
                "scope_suggestion": {"type": "string", "enum": sorted(SCOPES)},
                "evidence": {"type": "string"},
                "confidence": {"type": "string", "enum": sorted(CONFIDENCE)},
                "weak_point_ref": {"type": "string"},
            },
            "required": ["signal_type", "point", "evidence"],
        },
        handler=record_signal,
        timeout_s=5.0,
        permission_level="read",
        side_effect="none",
    )
