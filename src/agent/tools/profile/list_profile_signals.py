from __future__ import annotations

from typing import Any

from agent.schema import ToolSpec
from agent.tools.profile.common import load_session_memory_signals


def list_profile_signals(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
    local = list(getattr(ctx, "profile_signals", []) or [])
    persisted = load_session_memory_signals(ctx)
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for signal in [*persisted, *local]:
        key = (
            str(signal.get("type") or ""),
            str(signal.get("point") or ""),
            str(signal.get("evidence") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(signal)
    return {"signal_count": len(merged), "signals": merged[-50:]}


def list_profile_signals_spec() -> ToolSpec:
    return ToolSpec(
        name="list_profile_signals",
        description="List session-level profile memory signals recorded so far.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=list_profile_signals,
        timeout_s=5.0,
        permission_level="read",
        side_effect="none",
    )
