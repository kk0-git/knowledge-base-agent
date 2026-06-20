from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent.errors import ToolNotFoundError, ToolValidationError
from agent.schema import ToolCall, ToolExecutionStatus, ToolResult, ToolSpec, WorkingMemory
from agent.serialization import json_dumps, to_jsonable, truncate_text
from agent.tool_registry import ToolRegistry


@dataclass
class ToolExecutionContext:
    working: WorkingMemory
    confirmed_tools: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    vault_root: Path | None = None
    rag_manager: Any | None = None
    rag_manager_factory: Callable[[], Any] | None = None
    scope_note_paths: tuple[str, ...] = ()
    scope_type: str = ""
    scope_value: str = ""
    max_result_chars: int = 1200
    max_tool_output_chars: int = 8000
    session_id: str = ""
    session_store: Any | None = None
    interview_plan: Any | None = None
    interview_state: Any | None = None
    state_machine: Any | None = None
    profile_store: Any | None = None
    turn_context: dict[str, Any] = field(default_factory=dict)
    profile_signals: list[dict[str, Any]] = field(default_factory=list)
    observation_drafts: list[dict[str, Any]] = field(default_factory=list)


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, ctx: ToolExecutionContext):
        self.registry = registry
        self.ctx = ctx

    def execute(self, call: ToolCall) -> ToolResult:
        started_at = time.perf_counter()
        try:
            spec = self.registry.get(call.name)
            self._check_permission(spec)
            validate_arguments(spec, call.arguments)
            output = self._run_handler(spec, call.arguments)
            latency_ms = elapsed_ms(started_at)
            return build_success_result(call, output, latency_ms)
        except ToolNotFoundError as exc:
            return build_error_result(call, exc, ToolExecutionStatus.NOT_FOUND.value, elapsed_ms(started_at))
        except ToolValidationError as exc:
            return build_error_result(call, exc, ToolExecutionStatus.VALIDATION_ERROR.value, elapsed_ms(started_at))
        except TimeoutError as exc:
            return build_error_result(call, exc, ToolExecutionStatus.TIMEOUT.value, elapsed_ms(started_at), "tool timed out")
        except PermissionError as exc:
            return build_error_result(call, exc, ToolExecutionStatus.PERMISSION_DENIED.value, elapsed_ms(started_at))
        except Exception as exc:
            return build_error_result(call, exc, ToolExecutionStatus.ERROR.value, elapsed_ms(started_at))

    def _check_permission(self, spec: ToolSpec) -> None:
        if spec.side_effect != "none" and spec.name not in self.ctx.confirmed_tools:
            raise PermissionError(f"tool requires confirmation because side_effect={spec.side_effect}: {spec.name}")
        if spec.requires_confirmation and spec.name not in self.ctx.confirmed_tools:
            raise PermissionError(f"tool requires confirmation: {spec.name}")

    def _run_handler(self, spec: ToolSpec, arguments: dict[str, Any]) -> Any:
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(spec.handler, arguments, self.ctx)
            return future.result(timeout=spec.timeout_s)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


def validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> None:
    if not isinstance(arguments, dict):
        raise ToolValidationError(f"{spec.name}: arguments must be an object")
    required = spec.parameters.get("required") or []
    properties = spec.parameters.get("properties") or {}
    for name in required:
        if name not in arguments:
            raise ToolValidationError(f"{spec.name}: missing required argument: {name}")
    for name, value in arguments.items():
        schema = properties.get(name)
        if not isinstance(schema, dict):
            continue
        expected_type = schema.get("type")
        if expected_type and not value_matches_json_type(value, expected_type):
            raise ToolValidationError(f"{spec.name}: argument {name} must be {expected_type}")
        if "enum" in schema and value not in schema["enum"]:
            raise ToolValidationError(f"{spec.name}: argument {name} must be one of {schema['enum']}")


def value_matches_json_type(value: Any, expected_type: str | list[str]) -> bool:
    if isinstance(expected_type, list):
        return any(value_matches_json_type(value, item) for item in expected_type)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "null":
        return value is None
    return True


def build_success_result(call: ToolCall, output: Any, latency_ms: int) -> ToolResult:
    jsonable = to_jsonable(output)
    rendered = json_dumps(jsonable)
    return ToolResult(
        call_id=call.id,
        name=call.name,
        ok=True,
        output=jsonable,
        status=ToolExecutionStatus.SUCCESS.value,
        latency_ms=latency_ms,
        result_size=len(rendered),
        summary=truncate_text(rendered, 500),
    )


def build_error_result(
    call: ToolCall,
    exc: BaseException,
    status: str,
    latency_ms: int,
    message: str | None = None,
) -> ToolResult:
    error = message or str(exc)
    return ToolResult(
        call_id=call.id,
        name=call.name,
        ok=False,
        output=None,
        status=status,
        error=error,
        error_type=type(exc).__name__,
        latency_ms=latency_ms,
        result_size=len(error),
        summary=error,
    )


def elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)
