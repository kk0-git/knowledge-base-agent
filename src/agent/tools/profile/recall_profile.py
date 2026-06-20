from __future__ import annotations

from typing import Any

from agent.schema import ToolSpec
from agent.tools.profile.common import current_topic
from services.workflows.interview_profile import build_candidate_profile_debug, filter_weak_points_by_planned_layer


def recall_profile(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
    if getattr(ctx, "profile_store", None) is None:
        return {"available": False, "error": "profile_store is not configured"}
    topic = str(args.get("topic") or current_topic(ctx) or "").strip() or None
    planned_layer = str(args.get("planned_layer") or "").strip()
    if not planned_layer:
        state = getattr(ctx, "interview_state", None)
        if isinstance(state, dict):
            planned_layer = str(state.get("current_layer_name") or "").strip()
    include_due_only = bool(args.get("include_due_only", False))
    limit = max(1, min(int(args.get("limit") or 4), 12))
    profile = ctx.profile_store.load()
    debug = build_candidate_profile_debug(
        profile=profile,
        current_topic=topic,
        plan=getattr(ctx, "interview_plan", None),
    )
    if planned_layer:
        weak_source = list(debug.get("domain_weak_points") or [])
        weak_points = filter_weak_points_by_planned_layer(weak_source, planned_layer)[:limit]
    else:
        weak_points = list(debug.get("weak_points") or [])[:limit]
    due_reviews = list(debug.get("due_reviews") or [])[:limit]
    if include_due_only:
        weak_points = []
    strengths = [
        {
            "point": item.get("point"),
            "topic": item.get("topic"),
            "evidence": item.get("evidence", ""),
        }
        for item in (profile.get("strong_points") or [])
        if not topic or str(item.get("topic") or "") == topic
    ][:limit]
    return {
        "available": bool(debug.get("available")),
        "topic": debug.get("current_topic") or topic,
        "planned_layer": planned_layer,
        "weak_points": weak_points,
        "due_reviews": due_reviews,
        "strengths": strengths,
        "counts": {
            "weak_points": debug.get("weak_points_count", 0),
            "returned_weak_points": len(weak_points),
            "domain_weak_by_layer": debug.get("domain_weak_by_layer", {}),
            "due_reviews": debug.get("due_reviews_count", 0),
            "strong_points": debug.get("strong_points_count", 0),
        },
        "debug": {
            "topic_mastery": debug.get("topic_mastery"),
            "other_due_reviews_count": debug.get("other_due_reviews_count", 0),
            "other_domain_weak_points_count": debug.get("other_domain_weak_points_count", 0),
        },
    }


def recall_profile_spec() -> ToolSpec:
    return ToolSpec(
        name="recall_profile",
        description="Recall structured candidate profile memory for the current topic.",
        parameters={
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "planned_layer": {"type": "string"},
                "include_due_only": {"type": "boolean"},
                "limit": {"type": "integer"},
            },
            "required": [],
        },
        handler=recall_profile,
        timeout_s=5.0,
        permission_level="read",
        side_effect="none",
    )
