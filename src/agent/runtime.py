from __future__ import annotations

import copy
import json
import time
from typing import Callable, Iterator

from agent.errors import AgentRuntimeError
from agent.llm.tool_calling import LLMToolRequest, ToolCallingLLMClient, has_dsml_tool_intent, parse_dsml_tool_calls
from agent.schema import (
    AgentMessage,
    AgentResult,
    AgentRunConfig,
    AgentState,
    AgentStep,
    StepKind,
    ToolCall,
    ToolResult,
    ToolSpec,
    WorkingMemory,
)
from agent.serialization import json_dumps, to_jsonable
from agent.skill_loader import LoadedSkill, SkillLoader
from agent.tool_executor import ToolExecutionContext, ToolExecutor
from agent.tool_registry import ToolRegistry
from agent.trace.recorder import TraceRecorder, new_trace_id


class AgentRuntime:
    def __init__(
        self,
        *,
        llm_client: ToolCallingLLMClient,
        skill_loader: SkillLoader,
        tool_registry: ToolRegistry,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.skill_loader = skill_loader
        self.tool_registry = tool_registry
        self.trace_recorder = trace_recorder or TraceRecorder()

    def run(
        self,
        *,
        config: AgentRunConfig,
        user_input: str,
        state: AgentState | None = None,
        tool_context: ToolExecutionContext | None = None,
        event_sink: Callable[[dict], None] | None = None,
    ) -> AgentResult:
        started_at = time.perf_counter()
        trace_id = new_trace_id()
        skill: LoadedSkill | None = None
        steps: list[AgentStep] = []
        run_state: AgentState
        tool_call_cache: dict[tuple[str, str], ToolResult] = {}

        try:
            skill = self.skill_loader.load(config.skill_name)
            allowed_tools = effective_allowed_tools(skill=skill, config=config)
            registry = self.tool_registry.subset(allowed_tools)
            run_state = initialize_state(skill, state)
            run_state.messages.append(AgentMessage(role="user", content=user_input))
            execution_context = merge_tool_context(tool_context, run_state.working)
            executor = ToolExecutor(registry, execution_context)

            for step_index in range(max(1, config.max_steps)):
                run_state.step_index = step_index
                is_reserved_final_step = config.reserve_final_step and step_index == max(1, config.max_steps) - 1 and steps
                if is_reserved_final_step:
                    append_forced_final_instruction(run_state)
                request = build_llm_request(
                    config=config,
                    skill=skill,
                    state=run_state,
                    registry=registry,
                    allowed_tools=allowed_tools,
                    disable_tools=is_reserved_final_step,
                )
                llm_started_at = time.perf_counter()
                response = self.llm_client.complete_with_tools(request)
                latency_ms = elapsed_ms(llm_started_at)
                residual_dsml_calls = parse_dsml_tool_calls(response.content, allowed_tools) if is_reserved_final_step else []
                if is_reserved_final_step and residual_dsml_calls:
                    step = AgentStep(
                        index=step_index,
                        kind=StepKind.ERROR,
                        llm_input_chars=count_request_chars(request),
                        llm_output_chars=len(response.content),
                        assistant_text=response.content,
                        latency_ms=latency_ms,
                        metadata={
                            "finish_reason": response.finish_reason,
                            "llm_mode": response.used_mode,
                            "error_type": "DirtyFinalToolIntent",
                            "final_quality": "dirty_dsml",
                            "residual_tool_calls": to_jsonable(residual_dsml_calls[: max(1, config.max_tool_calls_per_step)]),
                        },
                        working_snapshot=to_jsonable(run_state.working),
                    )
                    steps.append(step)
                    emit_event(event_sink, "agent_step", to_jsonable(step))
                    result = AgentResult(
                        state=run_state,
                        steps=steps,
                        final_answer="",
                        total_ms=elapsed_ms(started_at),
                        stopped_reason="max_steps",
                        trace_id=trace_id,
                        error="reserved final step still contained tool intent",
                        error_type="DirtyFinalToolIntent",
                        metadata={
                            "forced_final": True,
                            "partial": True,
                            "recoverable": True,
                            "final_quality": "dirty_dsml",
                            "final_had_residual_tool_intent": True,
                            "dsml_parsed": False,
                            "fallback_reason": "dirty_dsml_reserved_final",
                            "last_tool_statuses": last_tool_statuses(steps),
                        },
                    )
                    return self._save_trace(result=result, config=config, skill=skill, user_input=user_input)
                tool_calls = [] if is_reserved_final_step else response.tool_calls[: max(1, config.max_tool_calls_per_step)]
                if tool_calls:
                    assistant_message = AgentMessage(
                        role="assistant",
                        content=response.content,
                        tool_calls=tool_calls,
                    )
                    run_state.messages.append(assistant_message)
                    tool_results = []
                    for call in tool_calls:
                        emit_event(event_sink, "tool_started", to_jsonable(call))
                        tool_result = execute_with_run_cache(
                            call=call,
                            registry=registry,
                            executor=executor,
                            cache=tool_call_cache,
                        )
                        tool_results.append(tool_result)
                        emit_event(event_sink, "tool_result", to_jsonable(tool_result))
                    for result in tool_results:
                        run_state.messages.append(
                            AgentMessage(
                                role="tool",
                                tool_call_id=result.call_id,
                                content=json_dumps(to_jsonable(result.output if result.ok else {"error": result.error, "status": result.status})),
                            )
                        )
                    step = AgentStep(
                        index=step_index,
                        kind=StepKind.TOOL,
                        llm_input_chars=count_request_chars(request),
                        llm_output_chars=count_response_chars(response.content, tool_calls),
                        tool_calls=tool_calls,
                        tool_results=tool_results,
                        assistant_text=response.content,
                        latency_ms=latency_ms + sum(result.latency_ms for result in tool_results),
                        metadata={
                            "finish_reason": response.finish_reason,
                            "llm_mode": response.used_mode,
                            "dsml_parsed": bool(getattr(response, "raw", {}).get("dsml_parsed")),
                        },
                        working_snapshot=to_jsonable(run_state.working),
                    )
                    steps.append(step)
                    emit_event(event_sink, "agent_step", to_jsonable(step))
                    continue

                if has_dsml_tool_intent(response.content):
                    step = AgentStep(
                        index=step_index,
                        kind=StepKind.ERROR,
                        llm_input_chars=count_request_chars(request),
                        llm_output_chars=len(response.content),
                        assistant_text=response.content,
                        latency_ms=latency_ms,
                        metadata={
                            "finish_reason": response.finish_reason,
                            "llm_mode": response.used_mode,
                            "error_type": "DirtyFinalToolIntent",
                            "final_quality": "dirty_dsml",
                        },
                        working_snapshot=to_jsonable(run_state.working),
                    )
                    steps.append(step)
                    emit_event(event_sink, "agent_step", to_jsonable(step))
                    result = AgentResult(
                        state=run_state,
                        steps=steps,
                        final_answer="",
                        total_ms=elapsed_ms(started_at),
                        stopped_reason="max_steps",
                        trace_id=trace_id,
                        error="model returned unparseable tool intent in content",
                        error_type="DirtyFinalToolIntent",
                        metadata={
                            "forced_final": is_reserved_final_step,
                            "partial": True,
                            "recoverable": True,
                            "final_quality": "dirty_dsml",
                            "final_had_residual_tool_intent": True,
                            "dsml_parsed": False,
                            "fallback_reason": "dirty_dsml_unparseable",
                            "last_tool_statuses": last_tool_statuses(steps),
                        },
                    )
                    return self._save_trace(result=result, config=config, skill=skill, user_input=user_input)

                if not response.content.strip():
                    step = AgentStep(
                        index=step_index,
                        kind=StepKind.ERROR,
                        llm_input_chars=count_request_chars(request),
                        llm_output_chars=0,
                        assistant_text=response.content,
                        latency_ms=latency_ms,
                        metadata={
                            "finish_reason": response.finish_reason,
                            "llm_mode": response.used_mode,
                            "error_type": "DirtyEmptyFinal",
                            "final_quality": "dirty_empty",
                        },
                        working_snapshot=to_jsonable(run_state.working),
                    )
                    steps.append(step)
                    emit_event(event_sink, "agent_step", to_jsonable(step))
                    result = AgentResult(
                        state=run_state,
                        steps=steps,
                        final_answer="",
                        total_ms=elapsed_ms(started_at),
                        stopped_reason="max_steps",
                        trace_id=trace_id,
                        error="model returned empty final answer",
                        error_type="DirtyEmptyFinal",
                        metadata={
                            "forced_final": is_reserved_final_step,
                            "partial": True,
                            "recoverable": True,
                            "final_quality": "dirty_empty",
                            "fallback_reason": "dirty_empty_final",
                            "last_tool_statuses": last_tool_statuses(steps),
                        },
                    )
                    return self._save_trace(result=result, config=config, skill=skill, user_input=user_input)

                run_state.final_answer = response.content
                run_state.finished = True
                step = AgentStep(
                    index=step_index,
                    kind=StepKind.FINAL,
                    llm_input_chars=count_request_chars(request),
                    llm_output_chars=len(response.content),
                    assistant_text=response.content,
                    latency_ms=latency_ms,
                    metadata={"finish_reason": response.finish_reason, "llm_mode": response.used_mode, "final_quality": "clean"},
                    working_snapshot=to_jsonable(run_state.working),
                )
                steps.append(step)
                emit_event(event_sink, "agent_step", to_jsonable(step))
                result = AgentResult(
                    state=run_state,
                    steps=steps,
                    final_answer=response.content,
                    total_ms=elapsed_ms(started_at),
                    stopped_reason="final",
                    trace_id=trace_id,
                    metadata={
                        "forced_final": is_reserved_final_step,
                        "partial": False,
                        "recoverable": False,
                        "final_quality": "clean" if response.content.strip() else "dirty_empty",
                        "dsml_parsed": bool(getattr(response, "raw", {}).get("dsml_parsed")),
                        "last_tool_statuses": last_tool_statuses(steps),
                    },
                )
                return self._save_trace(result=result, config=config, skill=skill, user_input=user_input)

            steps.append(
                AgentStep(
                    index=len(steps),
                    kind=StepKind.ERROR,
                    metadata={"error_type": "MaxStepsExceeded", "message": "agent reached max_steps"},
                    working_snapshot=to_jsonable(run_state.working),
                )
            )
            result = AgentResult(
                state=run_state,
                steps=steps,
                final_answer="",
                total_ms=elapsed_ms(started_at),
                stopped_reason="max_steps",
                trace_id=trace_id,
                error="agent reached max_steps",
                error_type="MaxStepsExceeded",
                metadata={
                    "forced_final": False,
                    "partial": False,
                    "recoverable": True,
                    "last_tool_statuses": last_tool_statuses(steps),
                },
            )
            return self._save_trace(result=result, config=config, skill=skill, user_input=user_input)
        except Exception as exc:
            if state is not None:
                run_state = state
            else:
                run_state = AgentState(messages=[], working=WorkingMemory(), skill_name=config.skill_name)
            steps.append(
                AgentStep(
                    index=len(steps),
                    kind=StepKind.ERROR,
                    metadata={"error_type": type(exc).__name__, "message": str(exc)},
                    working_snapshot=to_jsonable(run_state.working),
                )
            )
            result = AgentResult(
                state=run_state,
                steps=steps,
                final_answer="",
                total_ms=elapsed_ms(started_at),
                stopped_reason="llm_error",
                trace_id=trace_id,
                error=str(exc),
                error_type=type(exc).__name__,
                metadata={
                    "forced_final": False,
                    "partial": False,
                    "recoverable": True,
                    "last_tool_statuses": last_tool_statuses(steps),
                },
            )
            return self._save_trace(result=result, config=config, skill=skill, user_input=user_input)

    def run_stream(
        self,
        *,
        config: AgentRunConfig,
        user_input: str,
        state: AgentState | None = None,
        tool_context: ToolExecutionContext | None = None,
    ) -> Iterator[dict]:
        result = self.run(config=config, user_input=user_input, state=state, tool_context=tool_context)
        for step in result.steps:
            yield {"type": "agent_step", "payload": to_jsonable(step)}
        if result.final_answer:
            yield {"type": "answer_delta", "payload": {"text": result.final_answer}}
        yield {
            "type": "done",
            "payload": {
                "trace_id": result.trace_id,
                "trace_path": result.trace_path,
                "stopped_reason": result.stopped_reason,
                "error": result.error,
            },
        }

    def _save_trace(
        self,
        *,
        result: AgentResult,
        config: AgentRunConfig,
        skill: LoadedSkill | None,
        user_input: str,
    ) -> AgentResult:
        if not config.save_trace:
            return result
        trace_id, trace_path = self.trace_recorder.save(
            result=result,
            config=config,
            skill=skill,
            user_input=user_input,
        )
        result.trace_id = trace_id
        result.trace_path = trace_path
        return result


def initialize_state(skill: LoadedSkill, state: AgentState | None) -> AgentState:
    if state is not None:
        state.skill_name = skill.name
        if not state.messages or state.messages[0].role != "system":
            state.messages.insert(0, AgentMessage(role="system", content=skill.system_prompt))
        return state
    return AgentState(
        messages=[AgentMessage(role="system", content=skill.system_prompt)],
        working=WorkingMemory(),
        skill_name=skill.name,
    )


def merge_tool_context(
    tool_context: ToolExecutionContext | None,
    working: WorkingMemory,
) -> ToolExecutionContext:
    if tool_context is None:
        return ToolExecutionContext(working=working)
    tool_context.working = working
    return tool_context


def build_llm_request(
    *,
    config: AgentRunConfig,
    skill: LoadedSkill,
    state: AgentState,
    registry: ToolRegistry,
    allowed_tools: list[str] | None = None,
    disable_tools: bool = False,
) -> LLMToolRequest:
    tool_names = [] if disable_tools else allowed_tools or sorted(skill.allowed_tools)
    return LLMToolRequest(
        model=config.model,
        messages=state.messages,
        tools=registry.schemas_for(tool_names),
        temperature=config.temperature if config.temperature is not None else skill.temperature,
        tool_choice="auto",
        tool_mode=config.tool_mode,
    )


def append_forced_final_instruction(state: AgentState) -> None:
    if state.messages and state.messages[-1].role == "system" and "You must now produce the final answer" in state.messages[-1].content:
        return
    state.messages.append(
        AgentMessage(
            role="system",
            content=(
                "You must now produce the final answer. Do not call tools. "
                "Use only the prior observations and clearly state any evidence limits or uncertainty."
            ),
        )
    )


def last_tool_statuses(steps: list[AgentStep]) -> list[dict]:
    statuses: list[dict] = []
    for step in steps:
        for result in step.tool_results:
            statuses.append({"name": result.name, "ok": result.ok, "status": result.status})
    return statuses[-8:]


def effective_allowed_tools(*, skill: LoadedSkill, config: AgentRunConfig) -> list[str]:
    skill_tools = {str(name) for name in skill.allowed_tools}
    if config.allowed_tools is None:
        allowed = set(skill_tools)
    else:
        requested = {str(name) for name in config.allowed_tools}
        allowed = skill_tools.intersection(requested)
    disabled = {str(name) for name in (config.disabled_tools or [])}
    return sorted(allowed.difference(disabled))


def execute_with_run_cache(
    *,
    call: ToolCall,
    registry: ToolRegistry,
    executor: ToolExecutor,
    cache: dict[tuple[str, str], ToolResult],
) -> ToolResult:
    spec = registry.get(call.name)
    if not is_cacheable_tool(spec):
        return executor.execute(call)
    key = tool_cache_key(call)
    cached = cache.get(key)
    if cached is not None:
        return clone_cached_tool_result(cached, call_id=call.id)
    result = executor.execute(call)
    if result.ok:
        cache[key] = result
    return result


def is_cacheable_tool(spec: ToolSpec) -> bool:
    if spec.side_effect != "none" or spec.requires_confirmation:
        return False
    return spec.name not in {"advance_layer", "select_topic", "record_signal", "write_observation_draft"}


def tool_cache_key(call: ToolCall) -> tuple[str, str]:
    rendered_args = json.dumps(to_jsonable(call.arguments), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return call.name, rendered_args


def clone_cached_tool_result(result: ToolResult, *, call_id: str) -> ToolResult:
    stats = copy.deepcopy(result.stats)
    stats["cached"] = True
    return ToolResult(
        call_id=call_id,
        name=result.name,
        ok=result.ok,
        output=copy.deepcopy(result.output),
        status=result.status,
        error=result.error,
        error_type=result.error_type,
        latency_ms=0,
        result_size=result.result_size,
        summary=result.summary,
        stats=stats,
    )


def count_request_chars(request: LLMToolRequest) -> int:
    return len(json_dumps({"messages": request.messages, "tools": request.tools}))


def count_response_chars(content: str, tool_calls: list[ToolCall]) -> int:
    return len(content) + len(json_dumps(tool_calls))


def elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def emit_event(event_sink: Callable[[dict], None] | None, event_type: str, payload: dict) -> None:
    if event_sink is None:
        return
    event_sink({"type": event_type, "payload": payload})
