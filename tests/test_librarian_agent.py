from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from agent.llm.tool_calling import LLMToolRequest, LLMToolResponse, OpenAICompatibleToolCallingClient
from agent.apps.librarian import (
    LibrarianApp,
    LibrarianRequest,
    build_librarian_fallback,
    build_librarian_input,
    build_librarian_runtime_context,
    route_librarian_scope,
)
from agent.runtime import AgentRuntime
from agent.schema import AgentResult, AgentRunConfig, AgentState, AgentStep, StepKind, ToolCall, ToolResult, ToolSpec, WorkingMemory
from agent.skill_loader import SkillLoader
from agent.tool_executor import ToolExecutionContext
from agent.tool_registry import ToolRegistry
from agent.tools import register_debug_tools
from agent.tools.vault import register_vault_tools
from agent.trace import TraceRecorder
from knowledge_base_agent.llm.schema import LLMResponse
from services.rag.schema import SearchResult, TextChunk


class SearchReadLLM:
    def __init__(self):
        self.requests: list[LLMToolRequest] = []

    def complete_with_tools(self, request: LLMToolRequest) -> LLMToolResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return LLMToolResponse(
                tool_calls=[ToolCall(id="call_search", name="search_notes", arguments={"query": "Redis Stream", "top_k": 3})],
                finish_reason="tool_calls",
                used_mode="fake",
            )
        if len(self.requests) == 2:
            return LLMToolResponse(
                tool_calls=[ToolCall(id="call_read", name="read_note", arguments={"path": "redis.md", "max_chars": 1000})],
                finish_reason="tool_calls",
                used_mode="fake",
            )
        return LLMToolResponse(content="Redis Stream 可以用消费者组读取。", finish_reason="stop", used_mode="fake")


class JsonFallbackBaseClient:
    def __init__(self):
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content='{"tool_calls":[{"id":"call_search","name":"search_notes","arguments":{"query":"Redis Stream","top_k":3}}]}',
                raw={"fake": True},
            )
        if self.calls == 2:
            return LLMResponse(
                content='{"tool_calls":[{"id":"call_read","name":"read_note","arguments":{"path":"redis.md","max_chars":1000}}]}',
                raw={"fake": True},
            )
        return LLMResponse(content='{"final":"fallback librarian final"}', raw={"fake": True})


class FakeRAGManager:
    def hybrid_search(self, *, query: str, top_k: int, dense_top_k: int, bm25_top_k: int, rrf_k: int):
        return [
            SearchResult(
                chunk=TextChunk(
                    chunk_id="redis.md#1",
                    note_path="redis.md",
                    heading_path=["Stream"],
                    text="Redis Stream consumer group",
                    start_line=1,
                    end_line=4,
                ),
                score=0.9,
            )
        ]


class LibrarianAgentTests(unittest.TestCase):
    def test_librarian_runtime_context_injects_complete_scope_index(self) -> None:
        request = LibrarianRequest(
            query="memory",
            scope_type="folder",
            scope_value="agent",
            scope_note_paths=("memory.md", "MCP.md"),
        )
        context = build_librarian_runtime_context(
            request=request,
            budget=route_librarian_scope(scope_type="folder", online_enabled=False),
        )
        self.assertFalse(context["strict_evidence"])
        self.assertTrue(context["scope_index"]["is_complete"])
        self.assertEqual(context["scope_index"]["note_count"], 2)
        self.assertEqual([item["path"] for item in context["scope_index"]["notes"]], ["MCP.md", "memory.md"])

    def test_librarian_runtime_context_marks_large_scope_index_incomplete(self) -> None:
        paths = tuple(f"note-{index}.md" for index in range(31))
        request = LibrarianRequest(query="overview", scope_type="folder", scope_note_paths=paths)
        context = build_librarian_runtime_context(
            request=request,
            budget=route_librarian_scope(scope_type="folder", online_enabled=False),
        )
        self.assertFalse(context["scope_index"]["is_complete"])
        self.assertEqual(context["scope_index"]["note_count"], 31)
        self.assertEqual(context["scope_index"]["notes"], [])
        self.assertIn("list_notes", context["scope_index"]["hint"])

    def test_librarian_input_injects_strict_constraint_only_when_enabled(self) -> None:
        budget = route_librarian_scope(scope_type="folder", online_enabled=False)
        loose_request = LibrarianRequest(query="profile agent", strict_evidence=False)
        strict_request = LibrarianRequest(query="profile agent", strict_evidence=True)
        loose_context = build_librarian_runtime_context(request=loose_request, budget=budget)
        strict_context = build_librarian_runtime_context(request=strict_request, budget=budget)
        self.assertNotIn("# Strict Evidence Constraint", build_librarian_input(request=loose_request, runtime_context=loose_context))
        strict_input = build_librarian_input(request=strict_request, runtime_context=strict_context)
        self.assertIn("# Strict Evidence Constraint", strict_input)
        self.assertIn("only use my notes", strict_input)
        self.assertIn("Keep the answer natural and user-facing", strict_input)
        self.assertIn("Do not use audit-style labels", strict_input)

    def test_librarian_skill_only_exposes_vault_tools(self) -> None:
        registry = ToolRegistry()
        register_debug_tools(registry)
        register_vault_tools(registry)
        llm = SearchReadLLM()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "redis.md").write_text("# Redis\nStream", encoding="utf-8")
            runtime = AgentRuntime(
                llm_client=llm,
                skill_loader=SkillLoader(PROJECT_ROOT / "skills", registry=registry),
                tool_registry=registry,
                trace_recorder=TraceRecorder(tmp),
            )
            runtime.run(
                config=AgentRunConfig(skill_name="librarian", model="fake", trace_path=tmp),
                user_input="Redis Stream 怎么消费事件",
                tool_context=ToolExecutionContext(
                    working=WorkingMemory(),
                    vault_root=root,
                    rag_manager=FakeRAGManager(),
                ),
            )
        tool_names = [schema["function"]["name"] for schema in llm.requests[0].tools]
        self.assertEqual(tool_names, ["grep_vault", "list_notes", "online_search", "read_note", "search_notes"])

    def test_scope_router_maps_selected_notes_to_read_budget(self) -> None:
        budget = route_librarian_scope(scope_type="selected_notes", online_enabled=False)
        self.assertEqual(budget.effort_level, "L2")
        self.assertEqual(budget.max_steps, 4)
        self.assertEqual(budget.allowed_tools, ["grep_vault", "read_note"])

    def test_librarian_app_applies_scope_tool_whitelist(self) -> None:
        registry = ToolRegistry()
        register_vault_tools(registry)
        llm = SearchReadLLM()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "redis.md").write_text("# Redis\nStream", encoding="utf-8")
            app = LibrarianApp(
                AgentRuntime(
                    llm_client=llm,
                    skill_loader=SkillLoader(PROJECT_ROOT / "skills", registry=registry),
                    tool_registry=registry,
                    trace_recorder=TraceRecorder(tmp),
                )
            )
            app.run(
                LibrarianRequest(
                    query="summarize selected note",
                    scope_type="selected_notes",
                    scope_note_paths=("redis.md",),
                    selected_note_paths=("redis.md",),
                    vault_root=root,
                    model="fake",
                    trace_path=tmp,
                )
            )
        tool_names = [schema["function"]["name"] for schema in llm.requests[0].tools]
        self.assertEqual(tool_names, ["grep_vault", "read_note"])

    def test_librarian_run_stream_emits_tool_started_before_result(self) -> None:
        registry = ToolRegistry()
        register_vault_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "redis.md").write_text("# Redis\nStream", encoding="utf-8")
            app = LibrarianApp(
                AgentRuntime(
                    llm_client=SearchReadLLM(),
                    skill_loader=SkillLoader(PROJECT_ROOT / "skills", registry=registry),
                    tool_registry=registry,
                    trace_recorder=TraceRecorder(tmp),
                )
            )
            events = list(
                app.run_stream(
                    LibrarianRequest(
                        query="Redis Stream",
                        scope_type="all_vault",
                        vault_root=root,
                        rag_manager=FakeRAGManager(),
                        model="fake",
                        trace_path=tmp,
                    )
                )
            )
        event_types = [event["type"] for event in events]
        self.assertLess(event_types.index("tool_started"), event_types.index("tool_result"))
        self.assertIn("answer_delta", event_types)

    def test_librarian_fallback_for_max_steps_with_observations(self) -> None:
        result = AgentResult(
            state=AgentState(messages=[], working=WorkingMemory(), skill_name="librarian"),
            steps=[
                AgentStep(
                    index=0,
                    kind=StepKind.TOOL,
                    tool_results=[
                        ToolResult(
                            call_id="call_read",
                            name="read_note",
                            ok=True,
                            output={"path": "memory.md", "content": "memory"},
                        )
                    ],
                )
            ],
            final_answer="",
            total_ms=10,
            stopped_reason="max_steps",
            trace_path="trace.json",
            error="agent reached max_steps",
            error_type="MaxStepsExceeded",
        )
        fallback = build_librarian_fallback(result)
        self.assertTrue(fallback["stopped"]["partial"])
        self.assertTrue(fallback["stopped"]["recoverable"])
        self.assertEqual(fallback["stopped"]["reason"], "max_steps")
        self.assertIn("memory.md", fallback["answer"])

    def test_librarian_fallback_for_max_steps_without_observations(self) -> None:
        result = AgentResult(
            state=AgentState(messages=[], working=WorkingMemory(), skill_name="librarian"),
            steps=[],
            final_answer="",
            total_ms=10,
            stopped_reason="max_steps",
            trace_path="trace.json",
            error="agent reached max_steps",
            error_type="MaxStepsExceeded",
        )
        fallback = build_librarian_fallback(result)
        self.assertFalse(fallback["stopped"]["partial"])
        self.assertTrue(fallback["stopped"]["recoverable"])
        self.assertIn("没有获得可用资料结果", fallback["answer"])

    def test_librarian_runtime_search_read_final_trace_and_working_memory(self) -> None:
        registry = ToolRegistry()
        register_vault_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "redis.md").write_text("# Redis\n\nRedis Stream 可以用 XREADGROUP 消费。", encoding="utf-8")
            runtime = AgentRuntime(
                llm_client=SearchReadLLM(),
                skill_loader=SkillLoader(PROJECT_ROOT / "skills", registry=registry),
                tool_registry=registry,
                trace_recorder=TraceRecorder(tmp),
            )
            result = runtime.run(
                config=AgentRunConfig(skill_name="librarian", model="fake", trace_path=tmp),
                user_input="Redis Stream 怎么消费事件",
                tool_context=ToolExecutionContext(
                    working=WorkingMemory(),
                    vault_root=root,
                    rag_manager=FakeRAGManager(),
                    scope_note_paths=("redis.md",),
                ),
            )
            self.assertEqual(result.stopped_reason, "final")
            self.assertEqual(result.state.working.notes_read_this_turn, ["redis.md"])
            self.assertEqual([step.tool_calls[0].name for step in result.steps[:2]], ["search_notes", "read_note"])
            payload = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["working_memory"]["notes_read_this_turn"], ["redis.md"])
            self.assertEqual(payload["steps"][0]["tool_results"][0]["output"]["source_paths"], ["redis.md"])

    def test_read_note_preserves_selection_reason(self) -> None:
        registry = ToolRegistry()
        register_vault_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "memory.md").write_text("# Memory\n\nWorking memory.", encoding="utf-8")
            executor = registry.get("read_note").handler
            output = executor(
                {"path": "memory.md", "reason": "directly supports memory types"},
                ToolExecutionContext(working=WorkingMemory(), vault_root=root, scope_note_paths=("memory.md",)),
            )
        self.assertEqual(output["reason"], "directly supports memory types")
        self.assertEqual(output["path"], "memory.md")

    def test_librarian_json_fallback_executes_vault_tools(self) -> None:
        registry = ToolRegistry()
        register_vault_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "redis.md").write_text("# Redis\n\nRedis Stream", encoding="utf-8")
            runtime = AgentRuntime(
                llm_client=OpenAICompatibleToolCallingClient(JsonFallbackBaseClient()),
                skill_loader=SkillLoader(PROJECT_ROOT / "skills", registry=registry),
                tool_registry=registry,
                trace_recorder=TraceRecorder(tmp),
            )
            result = runtime.run(
                config=AgentRunConfig(skill_name="librarian", model="fake", tool_mode="json", trace_path=tmp),
                user_input="Redis Stream 怎么消费事件",
                tool_context=ToolExecutionContext(
                    working=WorkingMemory(),
                    vault_root=root,
                    rag_manager=FakeRAGManager(),
                ),
            )
            self.assertEqual(result.stopped_reason, "final")
            self.assertEqual(result.final_answer, "fallback librarian final")
            self.assertEqual(result.state.working.notes_read_this_turn, ["redis.md"])


if __name__ == "__main__":
    unittest.main()
