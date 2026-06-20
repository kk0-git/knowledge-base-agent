from __future__ import annotations

from typing import Any

from agent.schema import ToolSpec
from agent.tools.interview.common import ensure_state_machine, sync_working_from_machine


def advance_layer(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
    reason = str(args.get("reason") or "").strip()
    force = bool(args.get("force", False))
    machine = ensure_state_machine(ctx)
    result = machine.advance_layer(reason=reason, force=force)
    sync_working_from_machine(ctx.working, machine)
    return result


def advance_layer_spec() -> ToolSpec:
    return ToolSpec(
        name="advance_layer",
        description="Advance the current interview topic to the next planned coverage layer.",
        parameters={
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Why the current layer has enough signal."},
                "force": {"type": "boolean", "description": "Force transition even before the normal follow-up threshold."},
            },
            "required": ["reason"],
        },
        handler=advance_layer,
        timeout_s=5.0,
        permission_level="write",
        side_effect="state_write",
    )
