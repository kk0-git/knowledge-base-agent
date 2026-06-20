from __future__ import annotations

from typing import Any

from agent.schema import ToolSpec
from agent.tools.interview.common import ensure_state_machine, sync_working_from_machine


def get_interview_state(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
    machine = ensure_state_machine(ctx)
    sync_working_from_machine(ctx.working, machine)
    return machine.snapshot()


def get_interview_state_spec() -> ToolSpec:
    return ToolSpec(
        name="get_interview_state",
        description="Read the server-authoritative interview topic/layer/follow-up state.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=get_interview_state,
        timeout_s=5.0,
        permission_level="read",
        side_effect="none",
    )
