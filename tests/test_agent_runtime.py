from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
import sys
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from agent.llm.tool_calling import (
    LLMToolRequest,
    LLMToolResponse,
    OpenAICompatibleToolCallingClient,
    parse_json_object,
    parse_fallback_tool_calls,
)
from agent.runtime import AgentRuntime
from agent.schema import AgentRunConfig, ToolCall, ToolSpec, WorkingMemory
from agent.serialization import to_jsonable
from agent.skill_loader import SkillLoader
from agent.tool_executor import ToolExecutionContext, ToolExecutor
from agent.tool_registry import ToolRegistry
from agent.tools import register_debug_tools
from agent.trace import TraceRecorder
from knowledge_base_agent.llm.schema import LLMResponse


class SequenceLLM:
    def __init__(self, responses: list[LLMToolResponse]):
        self.responses = list(responses)
        self.requests: list[LLMToolRequest] = []

    def complete_with_tools(self, request: LLMToolRequest) -> LLMToolResponse:
        self.requests.append(request)
        if self.responses:
            return self.responses.pop(0)
        return LLMToolResponse(content="done", finish_reason="stop", used_mode="fake")


class AlwaysToolLLM:
    def complete_with_tools(self, request: LLMToolRequest) -> LLMToolResponse:
        return LLMToolResponse(
            tool_calls=[ToolCall(id="call_loop", name="echo", arguments={"text": "loop"})],
            finish_reason="tool_calls",
            used_mode="fake",
        )


class ReserveAwareToolLLM:
    def __init__(self) -> None:
        self.requests: list[LLMToolRequest] = []

    def complete_with_tools(self, request: LLMToolRequest) -> LLMToolResponse:
        self.requests.append(request)
        if not request.tools:
            return LLMToolResponse(content="forced final from observations", finish_reason="stop", used_mode="fake")
        return LLMToolResponse(
            tool_calls=[ToolCall(id=f"call_loop_{len(self.requests)}", name="echo", arguments={"text": "loop"})],
            finish_reason="tool_calls",
            used_mode="fake",
        )


class TwoToolLLM:
    def __init__(self) -> None:
        self.requests: list[LLMToolRequest] = []

    def complete_with_tools(self, request: LLMToolRequest) -> LLMToolResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return LLMToolResponse(
                tool_calls=[
                    ToolCall(id="call_echo", name="echo", arguments={"text": "现在几点了，回复里用 echo 重复这个问题"}),
                    ToolCall(id="call_time", name="get_time", arguments={}),
                ],
                finish_reason="tool_calls",
                used_mode="fake",
            )
        return LLMToolResponse(
            content="已用 echo 重复问题，并读取了当前时间。",
            finish_reason="stop",
            used_mode="fake",
        )


@dataclass
class FakeLLMConfig:
    timeout_seconds: int = 5
    api_key: str = ""


class JsonFallbackBaseClient:
    def __init__(self) -> None:
        self.calls = 0
        self.requests = []

    def complete(self, request):
        self.calls += 1
        self.requests.append(request)
        if self.calls == 1:
            return LLMResponse(
                content='{"tool_calls":[{"id":"call_1","name":"echo","arguments":{"text":"fallback hello"}}]}',
                raw={"fake": True},
            )
        return LLMResponse(content='{"final":"fallback final"}', raw={"fake": True})


class AgentRuntimeTests(unittest.TestCase):
    def test_schema_serialization_excludes_unserializable_handler(self) -> None:
        spec = ToolSpec(
            name="sample",
            description="sample tool",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda args, ctx: None,
        )
        payload = to_jsonable(spec)
        self.assertEqual(payload["name"], "sample")
        self.assertIsInstance(payload["handler"], str)

    def test_tool_registry_exports_openai_schema_and_blocks_duplicates(self) -> None:
        registry = ToolRegistry()
        register_debug_tools(registry)
        schemas = registry.schemas_for(["echo"])
        self.assertEqual(schemas[0]["type"], "function")
        self.assertEqual(schemas[0]["function"]["name"], "echo")
        with self.assertRaises(Exception):
            register_debug_tools(registry)

    def test_tool_executor_success_validation_error_exception_and_permission(self) -> None:
        registry = ToolRegistry()
        register_debug_tools(registry)
        registry.register(
            ToolSpec(
                name="explode",
                description="explode",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=lambda args, ctx: (_ for _ in ()).throw(RuntimeError("boom")),
            )
        )
        registry.register(
            ToolSpec(
                name="write_debug",
                description="side effect",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=lambda args, ctx: {"ok": True},
                side_effect="write",
            )
        )
        executor = ToolExecutor(registry, ToolExecutionContext(working=WorkingMemory()))
        ok = executor.execute(ToolCall(id="1", name="echo", arguments={"text": "hi"}))
        self.assertTrue(ok.ok)
        missing_arg = executor.execute(ToolCall(id="2", name="echo", arguments={}))
        self.assertFalse(missing_arg.ok)
        self.assertEqual(missing_arg.status, "validation_error")
        exploded = executor.execute(ToolCall(id="3", name="explode", arguments={}))
        self.assertFalse(exploded.ok)
        self.assertEqual(exploded.status, "error")
        denied = executor.execute(ToolCall(id="4", name="write_debug", arguments={}))
        self.assertFalse(denied.ok)
        self.assertEqual(denied.status, "permission_denied")

    def test_tool_executor_timeout(self) -> None:
        registry = ToolRegistry()

        def slow(args, ctx):
            time.sleep(0.2)
            return {"late": True}

        registry.register(
            ToolSpec(
                name="slow",
                description="slow",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=slow,
                timeout_s=0.01,
            )
        )
        result = ToolExecutor(registry, ToolExecutionContext(working=WorkingMemory())).execute(
            ToolCall(id="1", name="slow", arguments={})
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "timeout")

    def test_skill_loader_loads_and_validates_tools(self) -> None:
        registry = ToolRegistry()
        register_debug_tools(registry)
        loader = SkillLoader(PROJECT_ROOT / "skills", registry=registry)
        skill = loader.load("runtime_debug")
        self.assertEqual(skill.name, "runtime_debug")
        self.assertIn("echo", skill.allowed_tools)
        self.assertIn("runtime_debug", loader.list_skills())

    def test_runtime_single_tool_call_then_final_and_trace(self) -> None:
        registry = ToolRegistry()
        register_debug_tools(registry)
        llm = SequenceLLM(
            [
                LLMToolResponse(
                    tool_calls=[ToolCall(id="call_1", name="echo", arguments={"text": "hello"})],
                    finish_reason="tool_calls",
                    used_mode="fake",
                ),
                LLMToolResponse(content="final answer", finish_reason="stop", used_mode="fake"),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            runtime = AgentRuntime(
                llm_client=llm,
                skill_loader=SkillLoader(PROJECT_ROOT / "skills", registry=registry),
                tool_registry=registry,
                trace_recorder=TraceRecorder(tmp),
            )
            result = runtime.run(
                config=AgentRunConfig(skill_name="runtime_debug", model="fake", trace_path=tmp),
                user_input="hello",
            )
            self.assertEqual(result.stopped_reason, "final")
            self.assertEqual(result.final_answer, "final answer")
            self.assertTrue(Path(result.trace_path).exists())
            payload = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "agent_trace_v1")
            self.assertEqual(payload["steps"][0]["tool_results"][0]["status"], "success")

    def test_runtime_two_tool_calls_echo_and_get_time_then_final(self) -> None:
        registry = ToolRegistry()
        register_debug_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            runtime = AgentRuntime(
                llm_client=TwoToolLLM(),
                skill_loader=SkillLoader(PROJECT_ROOT / "skills", registry=registry),
                tool_registry=registry,
                trace_recorder=TraceRecorder(tmp),
            )
            result = runtime.run(
                config=AgentRunConfig(skill_name="runtime_debug", model="fake", trace_path=tmp),
                user_input="现在几点了，回复里用 echo 重复这个问题",
            )
            self.assertEqual(result.stopped_reason, "final")
            self.assertEqual([call.name for call in result.steps[0].tool_calls], ["echo", "get_time"])
            self.assertEqual([tool.status for tool in result.steps[0].tool_results], ["success", "success"])
            payload = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["steps"][0]["tool_calls"][1]["name"], "get_time")
            self.assertEqual(payload["steps"][0]["tool_results"][1]["output"]["timezone"], "UTC")

    def test_skill_manifest_filters_tool_schema(self) -> None:
        registry = ToolRegistry()
        register_debug_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "echo_only"
            skill_dir.mkdir()
            (skill_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "name": "echo_only",
                        "version": 1,
                        "description": "echo only",
                        "allowed_tools": ["echo"],
                        "denied_tools": [],
                        "max_steps": 2,
                        "temperature": 0.0,
                        "output_contract": {"type": "debug_text"},
                        "trace_policy": {"save": True},
                    }
                ),
                encoding="utf-8",
            )
            (skill_dir / "SKILL.md").write_text("Use allowed tools only.", encoding="utf-8")
            llm = SequenceLLM([LLMToolResponse(content="no time tool visible", finish_reason="stop", used_mode="fake")])
            runtime = AgentRuntime(
                llm_client=llm,
                skill_loader=SkillLoader(tmp, registry=registry),
                tool_registry=registry,
                trace_recorder=TraceRecorder(tmp),
            )
            result = runtime.run(
                config=AgentRunConfig(skill_name="echo_only", model="fake", trace_path=tmp),
                user_input="几点了",
            )
            self.assertEqual(result.stopped_reason, "final")
            tool_names = [schema["function"]["name"] for schema in llm.requests[0].tools]
            self.assertEqual(tool_names, ["echo"])

    def test_runtime_executes_real_json_fallback_adapter_path(self) -> None:
        registry = ToolRegistry()
        register_debug_tools(registry)
        base_client = JsonFallbackBaseClient()
        with tempfile.TemporaryDirectory() as tmp:
            runtime = AgentRuntime(
                llm_client=OpenAICompatibleToolCallingClient(base_client),
                skill_loader=SkillLoader(PROJECT_ROOT / "skills", registry=registry),
                tool_registry=registry,
                trace_recorder=TraceRecorder(tmp),
            )
            result = runtime.run(
                config=AgentRunConfig(skill_name="runtime_debug", model="fake", tool_mode="json", trace_path=tmp),
                user_input="hello fallback",
            )
            self.assertEqual(result.stopped_reason, "final")
            self.assertEqual(result.final_answer, "fallback final")
            self.assertEqual(result.steps[0].metadata["llm_mode"], "json")
            self.assertEqual(result.steps[0].tool_results[0].status, "success")
            self.assertGreaterEqual(base_client.calls, 2)

    def test_runtime_reserves_last_step_for_final_answer(self) -> None:
        registry = ToolRegistry()
        register_debug_tools(registry)
        llm = ReserveAwareToolLLM()
        with tempfile.TemporaryDirectory() as tmp:
            runtime = AgentRuntime(
                llm_client=llm,
                skill_loader=SkillLoader(PROJECT_ROOT / "skills", registry=registry),
                tool_registry=registry,
                trace_recorder=TraceRecorder(tmp),
            )
            result = runtime.run(
                config=AgentRunConfig(skill_name="runtime_debug", model="fake", max_steps=2, trace_path=tmp),
                user_input="loop",
            )
            self.assertEqual(result.stopped_reason, "final")
            self.assertEqual(result.final_answer, "forced final from observations")
            self.assertEqual(llm.requests[-1].tools, [])
            self.assertTrue(result.metadata["forced_final"])

    def test_runtime_max_steps_when_final_reserve_disabled(self) -> None:
        registry = ToolRegistry()
        register_debug_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            runtime = AgentRuntime(
                llm_client=AlwaysToolLLM(),
                skill_loader=SkillLoader(PROJECT_ROOT / "skills", registry=registry),
                tool_registry=registry,
                trace_recorder=TraceRecorder(tmp),
            )
            result = runtime.run(
                config=AgentRunConfig(
                    skill_name="runtime_debug",
                    model="fake",
                    max_steps=2,
                    trace_path=tmp,
                    reserve_final_step=False,
                ),
                user_input="loop",
            )
            self.assertEqual(result.stopped_reason, "max_steps")
            self.assertEqual(result.error_type, "MaxStepsExceeded")

    def test_json_fallback_parser(self) -> None:
        payload = parse_json_object(
            '{"tool_calls":[{"id":"call_1","name":"echo","arguments":{"text":"hi"}}]}'
        )
        calls = parse_fallback_tool_calls(payload["tool_calls"])
        self.assertEqual(calls[0].name, "echo")
        self.assertEqual(calls[0].arguments["text"], "hi")


if __name__ == "__main__":
    unittest.main()
