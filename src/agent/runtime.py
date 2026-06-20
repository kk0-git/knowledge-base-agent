from __future__ import annotations

import time
from typing import Iterator

from agent.errors import AgentRuntimeError
from agent.llm.tool_calling import LLMToolRequest, ToolCallingLLMClient
from agent.schema import (
    AgentMessage,
    AgentResult,
    AgentRunConfig,
    AgentState,
    AgentStep,
    StepKind,
    ToolCall,
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
    ) -> AgentResult:
        started_at = time.perf_counter()
        trace_id = new_trace_id()
        skill: LoadedSkill | None = None
        steps: list[AgentStep] = []
        run_state: AgentState

        try:
            skill = self.skill_loader.load(config.skill_name)
            registry = self.tool_registry.subset(sorted(skill.allowed_tools))
            run_state = initialize_state(skill, state)
            run_state.messages.append(AgentMessage(role="user", content=user_input))
            execution_context = merge_tool_context(tool_context, run_state.working)
            executor = ToolExecutor(registry, execution_context)

            for step_index in range(max(1, config.max_steps)):
                run_state.step_index = step_index
                request = build_llm_request(config=config, skill=skill, state=run_state, registry=registry)
                llm_started_at = time.perf_counter()
                response = self.llm_client.complete_with_tools(request)
                latency_ms = elapsed_ms(llm_started_at)
                tool_calls = response.tool_calls[: max(1, config.max_tool_calls_per_step)]
                if response.tool_calls:
                    assistant_message = AgentMessage(
                        role="assistant",
                        content=response.content,
                        tool_calls=tool_calls,
                    )
                    run_state.messages.append(assistant_message)
                    tool_results = [executor.execute(call) for call in tool_calls]
                    for result in tool_results:
                        run_state.messages.append(
                            AgentMessage(
                                role="tool",
                                tool_call_id=result.call_id,
                                content=json_dumps(to_jsonable(result.output if result.ok else {"error": result.error, "status": result.status})),
                            )
                        )
                    steps.append(
                        AgentStep(
                            index=step_index,
                            kind=StepKind.TOOL,
                            llm_input_chars=count_request_chars(request),
                            llm_output_chars=count_response_chars(response.content, tool_calls),
                            tool_calls=tool_calls,
                            tool_results=tool_results,
                            assistant_text=response.content,
                            latency_ms=latency_ms + sum(result.latency_ms for result in tool_results),
                            metadata={"finish_reason": response.finish_reason, "llm_mode": response.used_mode},
                            working_snapshot=to_jsonable(run_state.working),
                        )
                    )
                    continue

                run_state.final_answer = response.content
                run_state.finished = True
                steps.append(
                    AgentStep(
                        index=step_index,
                        kind=StepKind.FINAL,
                        llm_input_chars=count_request_chars(request),
                        llm_output_chars=len(response.content),
                        assistant_text=response.content,
                        latency_ms=latency_ms,
                        metadata={"finish_reason": response.finish_reason, "llm_mode": response.used_mode},
                        working_snapshot=to_jsonable(run_state.working),
                    )
                )
                result = AgentResult(
                    state=run_state,
                    steps=steps,
                    final_answer=response.content,
                    total_ms=elapsed_ms(started_at),
                    stopped_reason="final",
                    trace_id=trace_id,
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
                stopped_reason="error",
                trace_id=trace_id,
                error=str(exc),
                error_type=type(exc).__name__,
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
) -> LLMToolRequest:
    return LLMToolRequest(
        model=config.model,
        messages=state.messages,
        tools=registry.schemas_for(sorted(skill.allowed_tools)),
        temperature=config.temperature if config.temperature is not None else skill.temperature,
        tool_choice="auto",
        tool_mode=config.tool_mode,
    )


def count_request_chars(request: LLMToolRequest) -> int:
    return len(json_dumps({"messages": request.messages, "tools": request.tools}))


def count_response_chars(content: str, tool_calls: list[ToolCall]) -> int:
    return len(content) + len(json_dumps(tool_calls))


def elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)
