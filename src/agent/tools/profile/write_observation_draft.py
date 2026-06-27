from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agent.schema import ToolSpec
from agent.tools.profile.common import CATEGORIES, FACETS, SCOPES, append_session_observation_draft, context_note_paths, current_topic, normalize_enum


def write_observation_draft(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
    turn_context = getattr(ctx, "turn_context", {}) or {}
    draft = {
        "point": str(args.get("point") or "").strip(),
        "topic": str(args.get("topic") or current_topic(ctx) or "").strip(),
        "category": normalize_enum(args.get("category"), FACETS | CATEGORIES, "knowledge"),
        "scope": normalize_enum(args.get("scope") or args.get("scope_suggestion"), SCOPES, "domain"),
        "evidence": str(args.get("evidence") or "").strip(),
        "planned_layer": str(args.get("planned_layer") or turn_context.get("planned_layer") or "").strip(),
        "context_note_paths": context_note_paths(ctx),
        "session_id": getattr(ctx, "session_id", "") or "",
        "turn_id": str(turn_context.get("turn_id") or "").strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "draft",
    }
    ctx.observation_drafts.append(draft)
    append_session_observation_draft(ctx, draft)
    return {"written": True, "draft": draft}


def write_observation_draft_spec() -> ToolSpec:
    return ToolSpec(
        name="write_observation_draft",
        description="Write a session-level observation draft without committing long-term profile memory.",
        parameters={
            "type": "object",
            "properties": {
                "point": {"type": "string"},
                "topic": {"type": "string"},
                "category": {"type": "string", "enum": sorted(FACETS)},
                "scope": {"type": "string", "enum": sorted(SCOPES)},
                "evidence": {"type": "string"},
                "planned_layer": {"type": "string"},
            },
            "required": ["point", "evidence"],
        },
        handler=write_observation_draft,
        timeout_s=5.0,
        permission_level="read",
        side_effect="none",
    )
