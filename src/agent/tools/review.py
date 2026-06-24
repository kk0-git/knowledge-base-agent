from __future__ import annotations

from typing import Any

from agent.schema import ToolSpec
from agent.tool_registry import ToolRegistry
from services.workflows.review_practice import grouped_review_cards, weak_point_id


def get_due_reviews(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
    if getattr(ctx, "profile_store", None) is None:
        return {"available": False, "error": "profile_store is not configured"}
    topic = str(args.get("topic") or "").strip()
    topics = [str(item).strip() for item in args.get("topics") or [] if str(item).strip()]
    if not topics and topic:
        topics = [topic]
    if not topics:
        working = getattr(ctx, "working", None)
        working_extra = getattr(working, "extra", {}) if working is not None else {}
        context_topics = getattr(ctx, "turn_context", {}).get("review_topics") or working_extra.get("review_topics", [])
        topics = [str(item).strip() for item in context_topics if str(item).strip()]
    limit = max(1, min(int(args.get("limit") or 12), 50))
    profile = ctx.profile_store.load()
    payload = grouped_review_cards(profile, topics=topics, limit=limit)
    return {
        "available": True,
        "topic": topic,
        "selected_topics": topics,
        "today": payload.get("today"),
        "summary": payload.get("summary"),
        "cards": payload.get("cards", []),
        "topics": payload.get("topics", []),
        "due_count": payload.get("due_count", 0),
        "candidate_count": payload.get("candidate_count", 0),
        "recommended_count": payload.get("recommended_count", 0),
        "never_reviewed_count": payload.get("never_reviewed_count", 0),
        "recent_count": payload.get("recent_count", 0),
    }


def verify_weak_point(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
    weak_point_ids = [str(item).strip() for item in args.get("weak_point_ids") or [] if str(item).strip()]
    answer = str(args.get("answer") or "").strip()
    if not weak_point_ids:
        return {"ok": False, "error": "weak_point_ids is required"}
    if not answer:
        return {"ok": False, "error": "answer is required"}
    if getattr(ctx, "profile_store", None) is None:
        return {"ok": False, "error": "profile_store is not configured"}
    profile = ctx.profile_store.load()
    weak_by_id = {}
    for weak in profile.get("weak_points", []):
        if not isinstance(weak, dict):
            continue
        weak_id = weak_point_id(weak)
        if weak_id in weak_point_ids:
            weak_by_id[weak_id] = weak
    return {
        "ok": True,
        "note": "This tool is advisory only; final SM-2 writes require user confirmation through review commit.",
        "overall": "已记录你的回答。请继续根据薄弱点原文判断是否确认改善。",
        "weak_results": [
            {
                "weak_point_id": weak_id,
                "point": str((weak_by_id.get(weak_id) or {}).get("point") or "").strip(),
                "suggested_action": "retry",
                "reason": "对话工具第一版不自动判定通过；请要求用户确认后再提交。",
            }
            for weak_id in weak_point_ids
        ],
    }


def suggest_review_commit(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
    action = str(args.get("action") or "").strip().lower()
    if action not in {"improve", "retry"}:
        action = "retry"
    weak_point_id = str(args.get("weak_point_id") or "").strip()
    return {
        "ok": bool(weak_point_id),
        "weak_point_id": weak_point_id,
        "suggested_action": action,
        "requires_user_confirmation": True,
        "committed": False,
    }


def register_review_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="get_due_reviews",
            description="Return due review weak points grouped into review cards. Read-only.",
            parameters={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "topics": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer"},
                },
                "required": [],
            },
            handler=get_due_reviews,
            timeout_s=5.0,
            permission_level="read",
            side_effect="none",
        )
    )
    registry.register(
        ToolSpec(
            name="verify_weak_point",
            description="Advisory review verification for one or more weak points. Does not write profile.",
            parameters={
                "type": "object",
                "properties": {
                    "weak_point_ids": {"type": "array"},
                    "answer": {"type": "string"},
                },
                "required": ["weak_point_ids", "answer"],
            },
            handler=verify_weak_point,
            timeout_s=5.0,
            permission_level="read",
            side_effect="none",
        )
    )
    registry.register(
        ToolSpec(
            name="suggest_review_commit",
            description="Suggest a review commit action without writing profile.",
            parameters={
                "type": "object",
                "properties": {
                    "weak_point_id": {"type": "string"},
                    "action": {"type": "string", "enum": ["improve", "retry"]},
                },
                "required": ["weak_point_id", "action"],
            },
            handler=suggest_review_commit,
            timeout_s=5.0,
            permission_level="read",
            side_effect="none",
        )
    )


__all__ = ["register_review_tools"]
