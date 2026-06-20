from __future__ import annotations

from typing import Any

from agent.schema import ToolSpec
from agent.tools.interview.common import topics_payload


def list_plan_topics(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
    include_sources = bool(args.get("include_sources", False))
    plan = getattr(ctx, "interview_plan", None)
    if plan is None:
        return {"topic_count": 0, "topics": [], "suggested_order": []}
    return topics_payload(plan, include_sources=include_sources)


def list_plan_topics_spec() -> ToolSpec:
    return ToolSpec(
        name="list_plan_topics",
        description="List interview plan topics, coverage layers, and optional source note paths.",
        parameters={
            "type": "object",
            "properties": {
                "include_sources": {"type": "boolean", "description": "Whether to include source note paths."}
            },
            "required": [],
        },
        handler=list_plan_topics,
        timeout_s=5.0,
        permission_level="read",
        side_effect="none",
    )
