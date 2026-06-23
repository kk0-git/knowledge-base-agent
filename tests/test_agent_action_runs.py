from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from agent.schema import ToolCall, ToolSpec, WorkingMemory
from agent.tool_executor import ToolExecutionContext, ToolExecutor, build_success_result, build_tool_citations
from agent.tool_registry import ToolRegistry
from services.workflows.interview_sessions import InterviewSessionStore


class AgentActionRunTests(unittest.TestCase):
    def test_tool_result_stats_are_structured(self) -> None:
        result = build_success_result(
            ToolCall(id="call-1", name="search_notes", arguments={"query": "memory"}),
            {
                "result_count": 3,
                "source_paths": ["a.md", "b.md", "a.md"],
            },
            latency_ms=42,
        )
        self.assertEqual(result.stats["hit_count"], 3)
        self.assertEqual(result.stats["source_count"], 2)
        self.assertEqual(result.stats["source_paths"], ["a.md", "b.md"])
        self.assertEqual(result.stats["note_count"], 2)

    def test_read_note_stats_include_path_and_truncation(self) -> None:
        result = build_success_result(
            ToolCall(id="call-2", name="read_note", arguments={"path": "memory.md"}),
            {
                "path": "memory.md",
                "heading_path": ["Memory", "Long term"],
                "truncated": True,
            },
            latency_ms=13,
        )
        self.assertEqual(result.stats["note_count"], 1)
        self.assertEqual(result.stats["source_paths"], ["memory.md"])
        self.assertEqual(result.stats["heading_path"], ["Memory", "Long term"])
        self.assertTrue(result.stats["truncated"])

    def test_profile_and_state_action_stats_are_structured(self) -> None:
        recall = build_success_result(
            ToolCall(id="call-3", name="recall_profile", arguments={}),
            {
                "topic": "MCP",
                "planned_layer": "roles",
                "counts": {
                    "returned_weak_points": 2,
                    "due_reviews": 1,
                    "strong_points": 3,
                },
            },
            latency_ms=10,
        )
        self.assertEqual(recall.stats["weak_count"], 2)
        self.assertEqual(recall.stats["due_count"], 1)
        self.assertEqual(recall.stats["strength_count"], 3)
        self.assertEqual(recall.stats["hit_count"], 6)
        self.assertEqual(recall.stats["planned_layer"], "roles")

        selected = build_success_result(
            ToolCall(id="call-4", name="select_topic", arguments={}),
            {
                "ok": True,
                "selected": True,
                "transition": {
                    "type": "select_topic",
                    "from_topic": None,
                    "to_topic": "MCP",
                    "source": "ui",
                },
                "state": {"follow_up_count": 0},
            },
            latency_ms=10,
        )
        self.assertEqual(selected.stats["state_action"], "select_topic")
        self.assertEqual(selected.stats["to_topic"], "MCP")
        self.assertEqual(selected.stats["action_source"], "ui")
        self.assertTrue(selected.stats["follow_up_reset"])

    def test_tool_executor_collects_citations_when_enabled(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="read_note",
                description="read note",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=lambda args, ctx: {
                    "path": "memory.md",
                    "heading_path": ["Memory"],
                    "start_line": 3,
                    "end_line": 8,
                    "content": "text",
                },
            )
        )
        ctx = ToolExecutionContext(working=WorkingMemory(), metadata={"collect_citations": True})
        result = ToolExecutor(registry, ctx).execute(ToolCall(id="call-5", name="read_note", arguments={}))
        self.assertTrue(result.ok)
        self.assertEqual(
            ctx.citations,
            [
                {
                    "source_type": "note",
                    "path": "memory.md",
                    "title": "memory",
                    "heading_path": ["Memory"],
                    "line_start": 3,
                    "line_end": 8,
                    "tool": "read_note",
                }
            ],
        )

    def test_build_tool_citations_includes_search_title_and_score(self) -> None:
        citations = build_tool_citations(
            "search_notes",
            {
                "hits": [
                    {
                        "path": "个人/面试/MCP.md",
                        "title": "MCP",
                        "heading": "调用链路 > 传输方式",
                        "lines": "42-78",
                        "score": 0.9234,
                    }
                ]
            },
        )
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["title"], "MCP")
        self.assertEqual(citations[0]["score"], 0.9234)
        self.assertEqual(citations[0]["heading_path"], ["调用链路", "传输方式"])
        self.assertEqual(citations[0]["line_start"], 42)
        self.assertEqual(citations[0]["line_end"], 78)

    def test_tool_executor_does_not_collect_citations_by_default(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="read_note",
                description="read note",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=lambda args, ctx: {"path": "memory.md", "content": "text"},
            )
        )
        ctx = ToolExecutionContext(working=WorkingMemory())
        result = ToolExecutor(registry, ctx).execute(ToolCall(id="call-6", name="read_note", arguments={}))
        self.assertTrue(result.ok)
        self.assertEqual(ctx.citations, [])

    def test_complete_assistant_message_persists_agent_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = InterviewSessionStore(Path(tmp))
            created = store.create_session(
                source_type="folder",
                source_value="agent-interview",
                source_note_paths=[],
                interview_plan={},
                interview_state={},
            )
            session_id = created["session_id"]
            pending = store.append_pending_turn(session_id=session_id, user_content="hello")
            assistant_id = pending["assistant_message"]["id"]
            action = {
                "id": "call-1",
                "tool": "read_note",
                "kind": "read_note",
                "label": "Read note",
                "status": "success",
                "detail": "memory.md",
                "stats": {"note_count": 1},
                "source_paths": ["memory.md"],
            }
            completed = store.complete_assistant_message(
                session_id=session_id,
                assistant_message_id=assistant_id,
                assistant_content="next question?",
                agent_actions=[action],
            )
            self.assertEqual(completed["assistant_message"]["agent_actions"], [action])


if __name__ == "__main__":
    unittest.main()
