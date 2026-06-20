from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from agent.schema import AgentResult, AgentRunConfig
from agent.serialization import json_dumps, to_jsonable, truncate_text
from agent.skill_loader import LoadedSkill


TRACE_SCHEMA_VERSION = "agent_trace_v1"


class TraceRecorder:
    def __init__(self, default_root: Path | str = "eval-results/agent-debug/traces"):
        self.default_root = Path(default_root)

    def save(
        self,
        *,
        result: AgentResult,
        config: AgentRunConfig,
        skill: LoadedSkill | None,
        user_input: str,
    ) -> tuple[str, str]:
        trace_id = result.trace_id or new_trace_id()
        payload = build_trace_payload(
            trace_id=trace_id,
            result=result,
            config=config,
            skill=skill,
            user_input=user_input,
        )
        path = resolve_trace_path(config.trace_path, self.default_root, trace_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_dumps(payload, indent=2), encoding="utf-8")
        return trace_id, str(path)


def build_trace_payload(
    *,
    trace_id: str,
    result: AgentResult,
    config: AgentRunConfig,
    skill: LoadedSkill | None,
    user_input: str,
) -> dict[str, Any]:
    working_extra = getattr(result.state.working, "extra", {}) or {}
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_id": trace_id,
        "skill": {
            "name": skill.name if skill else result.state.skill_name,
            "version": skill.version if skill else None,
            "description": skill.description if skill else "",
        },
        "config": {
            "skill_name": config.skill_name,
            "max_steps": config.max_steps,
            "max_tool_calls_per_step": config.max_tool_calls_per_step,
            "temperature": config.temperature,
            "model": config.model,
            "tool_mode": config.tool_mode,
            "stream_final": config.stream_final,
        },
        "input": {
            "summary": truncate_text(user_input, 500),
            "chars": len(user_input),
        },
        "steps": [to_jsonable(step) for step in result.steps],
        "working_memory": to_jsonable(result.state.working),
        "interview": {
            "state_before": to_jsonable(working_extra.get("interview_state_before") or {}),
            "state_after": to_jsonable(working_extra.get("interview_state_after") or working_extra.get("interview_state") or {}),
            "state_transitions": to_jsonable(working_extra.get("state_transitions") or []),
        },
        "final": {
            "answer": truncate_text(result.final_answer, 4000),
            "answer_chars": len(result.final_answer),
            "stopped_reason": result.stopped_reason,
            "total_ms": result.total_ms,
            "error": result.error,
            "error_type": result.error_type,
        },
    }


def resolve_trace_path(raw_path: str | None, default_root: Path, trace_id: str) -> Path:
    if raw_path:
        path = Path(raw_path)
        if path.suffix.lower() == ".json":
            return path
        return path / f"{trace_id}.json"
    return default_root / f"{trace_id}.json"


def new_trace_id() -> str:
    return f"agent-{uuid.uuid4().hex[:12]}"
