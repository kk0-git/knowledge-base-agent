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
from agent.runtime import AgentRuntime
from agent.schema import AgentRunConfig, ToolCall, ToolSpec, WorkingMemory
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
        self.assertEqual(tool_names, ["grep_vault", "read_note", "search_notes"])

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
