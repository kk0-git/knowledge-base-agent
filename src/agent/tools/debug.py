from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agent.schema import ToolSpec
from agent.serialization import to_jsonable
from agent.tool_executor import ToolExecutionContext
from agent.tool_registry import ToolRegistry


def register_debug_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="echo",
            description="Echo input text for runtime debugging.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=echo,
            timeout_s=5.0,
        )
    )
    registry.register(
        ToolSpec(
            name="inspect_state",
            description="Return the current working memory snapshot.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=inspect_state,
            timeout_s=5.0,
        )
    )
    registry.register(
        ToolSpec(
            name="get_time",
            description="Return the current UTC time for runtime debugging.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=get_time,
            timeout_s=5.0,
        )
    )
    registry.register(
        ToolSpec(
            name="record_debug_signal",
            description="Record a debug signal into working memory.",
            parameters={
                "type": "object",
                "properties": {
                    "signal": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["signal"],
            },
            handler=record_debug_signal,
            timeout_s=5.0,
        )
    )


def echo(arguments: dict[str, Any], ctx: ToolExecutionContext) -> dict[str, Any]:
    return {"text": str(arguments.get("text", ""))}


def inspect_state(arguments: dict[str, Any], ctx: ToolExecutionContext) -> dict[str, Any]:
    return to_jsonable(ctx.working)


def get_time(arguments: dict[str, Any], ctx: ToolExecutionContext) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {"timezone": "UTC", "iso": now.isoformat()}


def record_debug_signal(arguments: dict[str, Any], ctx: ToolExecutionContext) -> dict[str, Any]:
    signal = {
        "signal": str(arguments.get("signal", "")).strip(),
        "confidence": str(arguments.get("confidence") or "medium"),
    }
    ctx.working.signals_this_turn.append(signal)
    return {"recorded": signal, "signal_count": len(ctx.working.signals_this_turn)}
