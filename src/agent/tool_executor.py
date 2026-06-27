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
    online_search_client: Any | None = None
    turn_context: dict[str, Any] = field(default_factory=dict)
    profile_signals: list[dict[str, Any]] = field(default_factory=list)
    observation_drafts: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    current_call_id: str | None = None
    stats_holder: dict[str, Any] = field(default_factory=dict)

    def begin_tool_call(self, call_id: str) -> None:
        self.current_call_id = call_id
        self.stats_holder = {}

    def put_stats(self, stats: dict[str, Any]) -> None:
        if isinstance(stats, dict):
            self.stats_holder.update(to_jsonable(stats))

    def clear_tool_call(self) -> None:
        self.current_call_id = None
        self.stats_holder = {}


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
            self.ctx.begin_tool_call(call.id)
            try:
                output = self._run_handler(spec, call.arguments)
                collect_tool_citations(call.name, output, self.ctx)
                latency_ms = elapsed_ms(started_at)
                return build_success_result(call, output, latency_ms, extra_stats=dict(self.ctx.stats_holder))
            finally:
                self.ctx.clear_tool_call()
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


def build_success_result(call: ToolCall, output: Any, latency_ms: int, *, extra_stats: dict[str, Any] | None = None) -> ToolResult:
    jsonable = to_jsonable(output)
    rendered = json_dumps(jsonable)
    stats = extract_tool_stats(call.name, jsonable)
    if extra_stats:
        stats.update(to_jsonable(extra_stats))
    return ToolResult(
        call_id=call.id,
        name=call.name,
        ok=True,
        output=jsonable,
        status=ToolExecutionStatus.SUCCESS.value,
        latency_ms=latency_ms,
        result_size=len(rendered),
        summary=truncate_text(rendered, 500),
        stats=stats,
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
        stats={},
    )


def elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def extract_tool_stats(name: str, output: Any) -> dict[str, Any]:
    if not isinstance(output, dict):
        return {}
    stats: dict[str, Any] = {}
    result_count = first_int(output, "result_count", "count", "match_count", "note_count")
    if result_count is not None:
        stats["hit_count"] = result_count
    source_paths = output.get("source_paths")
    if isinstance(source_paths, list):
        unique = [str(path) for path in dict.fromkeys(source_paths) if str(path or "").strip()]
        stats["source_count"] = len(unique)
        stats["source_paths"] = unique[:10]
    elif output.get("path"):
        stats["source_count"] = 1
        stats["source_paths"] = [str(output.get("path"))]
    if name == "read_note":
        stats["note_count"] = 1 if output.get("path") else 0
        if output.get("truncated") is not None:
            stats["truncated"] = bool(output.get("truncated"))
        if output.get("heading_path"):
            stats["heading_path"] = output.get("heading_path")
    if name in {"search_notes", "grep_vault", "list_notes", "online_search"} and "source_count" in stats:
        stats["note_count"] = stats["source_count"]
    if output.get("scope_type"):
        stats["scope_type"] = output.get("scope_type")
    if output.get("scope_value"):
        stats["scope_value"] = output.get("scope_value")
    if name == "recall_profile":
        counts = output.get("counts") if isinstance(output.get("counts"), dict) else {}
        weak_count = first_int(counts, "returned_weak_points", "weak_points")
        due_count = first_int(counts, "due_reviews")
        strength_count = first_int(counts, "strong_points")
        if weak_count is not None:
            stats["weak_count"] = weak_count
        if due_count is not None:
            stats["due_count"] = due_count
        if strength_count is not None:
            stats["strength_count"] = strength_count
        stats["hit_count"] = int(stats.get("weak_count", 0)) + int(stats.get("due_count", 0)) + int(stats.get("strength_count", 0))
        if output.get("planned_layer"):
            stats["planned_layer"] = output.get("planned_layer")
        if output.get("topic"):
            stats["topic"] = output.get("topic")
    if name in {"record_signal", "write_observation_draft"}:
        key = "signal" if name == "record_signal" else "draft"
        payload = output.get(key) if isinstance(output.get(key), dict) else {}
        stats[f"{key}_count"] = 1 if payload or output.get("recorded") or output.get("written") else 0
        if payload.get("topic"):
            stats["topic"] = payload.get("topic")
        if payload.get("planned_layer"):
            stats["planned_layer"] = payload.get("planned_layer")
        if payload.get("context_note_paths"):
            stats["source_paths"] = [str(path) for path in payload.get("context_note_paths") or [] if str(path or "").strip()][:10]
            stats["source_count"] = len(stats["source_paths"])
    if name == "list_profile_signals":
        signal_count = first_int(output, "signal_count")
        if signal_count is not None:
            stats["signal_count"] = signal_count
            stats["hit_count"] = signal_count
    if name in {"advance_layer", "select_topic"}:
        transition = output.get("transition") if isinstance(output.get("transition"), dict) else {}
        state = output.get("state") if isinstance(output.get("state"), dict) else {}
        action = str(transition.get("type") or name)
        stats["state_action"] = action
        stats["action_ok"] = bool(output.get("advanced") or output.get("selected") or output.get("ok"))
        if name == "advance_layer":
            if transition.get("from_layer_name") is not None:
                stats["from_layer"] = transition.get("from_layer_name")
            if transition.get("to_layer_name") is not None:
                stats["to_layer"] = transition.get("to_layer_name")
            if transition.get("from_layer_index") is not None:
                stats["from_layer_index"] = transition.get("from_layer_index")
            if transition.get("to_layer_index") is not None:
                stats["to_layer_index"] = transition.get("to_layer_index")
            if state.get("follow_up_count") == 0 and bool(output.get("advanced")):
                stats["follow_up_reset"] = True
        if name == "select_topic":
            if transition.get("from_topic") is not None:
                stats["from_topic"] = transition.get("from_topic")
            if transition.get("to_topic") is not None:
                stats["to_topic"] = transition.get("to_topic")
            if transition.get("source") is not None:
                stats["action_source"] = transition.get("source")
            if state.get("follow_up_count") == 0 and bool(output.get("selected")):
                stats["follow_up_reset"] = True
    return stats


def collect_tool_citations(name: str, output: Any, ctx: ToolExecutionContext) -> None:
    if not bool(ctx.metadata.get("collect_citations")):
        return
    if name not in {"search_notes", "read_note", "grep_vault", "list_notes"}:
        return
    jsonable = to_jsonable(output)
    if not isinstance(jsonable, dict):
        return
    for citation in build_tool_citations(name, jsonable):
        key = citation_key(citation)
        if any(citation_key(existing) == key for existing in ctx.citations):
            continue
        ctx.citations.append(citation)


def build_tool_citations(name: str, output: dict[str, Any]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    if name == "read_note" and output.get("path"):
        citations.append(
            compact_citation(
                {
                    "source_type": "note",
                    "path": str(output.get("path")),
                    "heading_path": output.get("heading_path") or [],
                    "line_start": output.get("start_line"),
                    "line_end": output.get("end_line"),
                    "tool": name,
                }
            )
        )
    if name == "search_notes":
        for hit in output.get("hits") or []:
            if not isinstance(hit, dict) or not hit.get("path"):
                continue
            line_start, line_end = parse_line_range(hit.get("lines"))
            citations.append(
                compact_citation(
                    {
                        "source_type": "note",
                        "path": str(hit.get("path")),
                        "heading_path": parse_heading_text(hit.get("heading")),
                        "line_start": line_start,
                        "line_end": line_end,
                        "tool": name,
                    }
                )
            )
    if name == "grep_vault":
        for match in output.get("matches") or []:
            if not isinstance(match, dict) or not match.get("path"):
                continue
            line = match.get("line")
            citations.append(
                compact_citation(
                    {
                        "source_type": "note",
                        "path": str(match.get("path")),
                        "heading_path": [],
                        "line_start": line,
                        "line_end": line,
                        "tool": name,
                    }
                )
            )
    if name == "list_notes":
        for note in output.get("notes") or []:
            if not isinstance(note, dict) or not note.get("path"):
                continue
            citations.append(
                compact_citation(
                    {
                        "source_type": "note",
                        "path": str(note.get("path")),
                        "heading_path": [],
                        "tool": name,
                    }
                )
            )
    return citations


def compact_citation(citation: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in citation.items() if value not in ("", None, [])}


def citation_key(citation: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(citation.get("tool") or ""),
        str(citation.get("path") or ""),
        str(citation.get("line_start") or ""),
        str(citation.get("line_end") or ""),
    )


def parse_line_range(value: Any) -> tuple[int | None, int | None]:
    text = str(value or "").strip()
    if not text:
        return None, None
    if "-" in text:
        left, right = text.split("-", 1)
        return parse_int(left), parse_int(right)
    parsed = parse_int(text)
    return parsed, parsed


def parse_heading_text(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split(">") if part.strip()]


def parse_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def first_int(payload: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return None
