from __future__ import annotations

from typing import Any

from agent.schema import ToolSpec
from agent.tools.interview.common import ensure_state_machine, sync_working_from_machine


def select_topic(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
    name = str(args.get("name") or "").strip()
    reason = str(args.get("reason") or "").strip()
    source = str(args.get("source") or "agent").strip() or "agent"
    machine = ensure_state_machine(ctx)
    result = machine.select_topic(name=name, reason=reason, source=source)
    sync_working_from_machine(ctx.working, machine)
    return result


def select_topic_spec() -> ToolSpec:
    return ToolSpec(
        name="select_topic",
        description="Select or switch the active interview topic in the server state machine.",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Topic name exactly as listed in the interview plan."},
                "reason": {"type": "string", "description": "Why this topic should become active."},
                "source": {"type": "string", "description": "Selection source, for example agent or ui."},
            },
            "required": ["name", "reason"],
        },
        handler=select_topic,
        timeout_s=5.0,
        permission_level="write",
        side_effect="state_write",
    )
