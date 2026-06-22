from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from agent.schema import ToolCall, WorkingMemory
from agent.tool_executor import ToolExecutionContext, ToolExecutor
from agent.tool_registry import ToolRegistry
from agent.tools.vault import register_vault_tools
from agent.tools.vault.guards import VaultPathError, normalize_relative_path
from services.rag.schema import SearchResult, TextChunk


class FakeRAGManager:
    def __init__(self, results: list[SearchResult]):
        self.results = results
        self.queries: list[str] = []

    def hybrid_search(self, *, query: str, top_k: int, dense_top_k: int, bm25_top_k: int, rrf_k: int):
        self.queries.append(query)
        return self.results[:top_k]


class FakeOnlineClient:
    def search(self, query: str, top_k: int = 5):
        from services.rag.online_search import OnlineSearchResponse, OnlineSearchResult

        return OnlineSearchResponse(
            enabled=True,
            provider="fake",
            results=[OnlineSearchResult(title="Example", url="https://example.com", snippet=query)],
        )


class VaultToolTests(unittest.TestCase):
    def test_normalize_relative_path_rejects_unsafe_paths(self) -> None:
        self.assertEqual(normalize_relative_path("folder\\note.md"), "folder/note.md")
        for path in ["../note.md", "/tmp/note.md", "folder/note.txt"]:
            with self.assertRaises(VaultPathError):
                normalize_relative_path(path)

    def test_search_notes_filters_to_scope(self) -> None:
        registry = ToolRegistry()
        register_vault_tools(registry)
        manager = FakeRAGManager(
            [
                search_result("allowed/a.md", "allowed text"),
                search_result("outside/b.md", "outside text"),
            ]
        )
        ctx = ToolExecutionContext(
            working=WorkingMemory(),
            rag_manager=manager,
            scope_note_paths=("allowed/a.md",),
        )
        result = ToolExecutor(registry, ctx).execute(
            ToolCall(id="1", name="search_notes", arguments={"query": "test", "top_k": 5})
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.output["result_count"], 1)
        self.assertEqual(result.output["hits"][0]["path"], "allowed/a.md")
        self.assertEqual(result.output["source_paths"], ["allowed/a.md"])

    def test_search_notes_lazily_builds_rag_manager(self) -> None:
        registry = ToolRegistry()
        register_vault_tools(registry)
        manager = FakeRAGManager([search_result("allowed/a.md", "allowed text")])
        factory_calls = 0

        def factory():
            nonlocal factory_calls
            factory_calls += 1
            return manager

        ctx = ToolExecutionContext(
            working=WorkingMemory(),
            rag_manager=None,
            rag_manager_factory=factory,
            scope_note_paths=("allowed/a.md",),
        )
        executor = ToolExecutor(registry, ctx)
        first = executor.execute(ToolCall(id="1", name="search_notes", arguments={"query": "test"}))
        second = executor.execute(ToolCall(id="2", name="search_notes", arguments={"query": "again"}))
        self.assertTrue(first.ok)
        self.assertTrue(second.ok)
        self.assertIs(ctx.rag_manager, manager)
        self.assertEqual(factory_calls, 1)

    def test_read_note_updates_notes_read_and_rejects_scope_outside(self) -> None:
        registry = ToolRegistry()
        register_vault_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "allowed").mkdir()
            (root / "outside").mkdir()
            (root / "allowed" / "a.md").write_text("# Title\n\ncontent", encoding="utf-8")
            (root / "outside" / "b.md").write_text("outside", encoding="utf-8")
            working = WorkingMemory()
            ctx = ToolExecutionContext(
                working=working,
                vault_root=root,
                scope_note_paths=("allowed/a.md",),
            )
            executor = ToolExecutor(registry, ctx)
            ok = executor.execute(ToolCall(id="1", name="read_note", arguments={"path": "allowed/a.md"}))
            self.assertTrue(ok.ok)
            self.assertEqual(ok.output["path"], "allowed/a.md")
            self.assertEqual(working.notes_read_this_turn, ["allowed/a.md"])

            denied = executor.execute(ToolCall(id="2", name="read_note", arguments={"path": "outside/b.md"}))
            self.assertFalse(denied.ok)
            self.assertEqual(denied.status, "permission_denied")

    def test_read_note_heading_extracts_section(self) -> None:
        registry = ToolRegistry()
        register_vault_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.md").write_text("# A\none\n## B\ntwo\n# C\nthree", encoding="utf-8")
            result = ToolExecutor(
                registry,
                ToolExecutionContext(working=WorkingMemory(), vault_root=root),
            ).execute(ToolCall(id="1", name="read_note", arguments={"path": "note.md", "heading": "B"}))
            self.assertTrue(result.ok)
            self.assertIn("## B", result.output["content"])
            self.assertNotIn("# C", result.output["content"])
            self.assertTrue(result.output["section_id"])

    def test_read_note_long_note_returns_content_with_truncation(self) -> None:
        registry = ToolRegistry()
        register_vault_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "detail line for pagination test\n" * 300
            (root / "long.md").write_text(body, encoding="utf-8")
            result = ToolExecutor(
                registry,
                ToolExecutionContext(working=WorkingMemory(), vault_root=root),
            ).execute(ToolCall(id="1", name="read_note", arguments={"path": "long.md", "max_chars": 4000}))
            self.assertTrue(result.ok)
            self.assertIn("content", result.output)
            self.assertTrue(result.output["truncated"])
            self.assertIn("sections", result.output)
            self.assertEqual(result.output["next_offset"], 4000)

    def test_read_note_offset_continues_within_note(self) -> None:
        registry = ToolRegistry()
        register_vault_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "".join(f"chunk-{i:04d}-" for i in range(500))
            (root / "long.md").write_text(body, encoding="utf-8")
            executor = ToolExecutor(registry, ToolExecutionContext(working=WorkingMemory(), vault_root=root))
            first = executor.execute(
                ToolCall(id="1", name="read_note", arguments={"path": "long.md", "max_chars": 100})
            )
            second = executor.execute(
                ToolCall(
                    id="2",
                    name="read_note",
                    arguments={"path": "long.md", "max_chars": 100, "offset": first.output["next_offset"]},
                )
            )
            self.assertTrue(first.ok)
            self.assertTrue(second.ok)
            self.assertEqual(first.output["offset"], 0)
            self.assertEqual(second.output["offset"], 100)
            self.assertNotEqual(first.output["content"], second.output["content"])
            self.assertTrue(body.startswith(first.output["content"]))
            self.assertTrue(body[100:].startswith(second.output["content"]))

    def test_read_note_short_note_has_no_sections(self) -> None:
        registry = ToolRegistry()
        register_vault_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "short.md").write_text("# Title\n\nshort content", encoding="utf-8")
            result = ToolExecutor(
                registry,
                ToolExecutionContext(working=WorkingMemory(), vault_root=root),
            ).execute(ToolCall(id="1", name="read_note", arguments={"path": "short.md"}))
            self.assertTrue(result.ok)
            self.assertFalse(result.output["truncated"])
            self.assertNotIn("sections", result.output)

    def test_read_note_missing_section_returns_error(self) -> None:
        registry = ToolRegistry()
        register_vault_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.md").write_text("# A\none", encoding="utf-8")
            result = ToolExecutor(
                registry,
                ToolExecutionContext(working=WorkingMemory(), vault_root=root),
            ).execute(ToolCall(id="1", name="read_note", arguments={"path": "note.md", "heading": "Missing"}))
            self.assertFalse(result.ok)
            self.assertIn("section not found", result.error)

    def test_grep_vault_filters_scope(self) -> None:
        registry = ToolRegistry()
        register_vault_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "allowed").mkdir()
            (root / "outside").mkdir()
            (root / "allowed" / "a.md").write_text("needle inside", encoding="utf-8")
            (root / "outside" / "b.md").write_text("needle outside", encoding="utf-8")
            result = ToolExecutor(
                registry,
                ToolExecutionContext(
                    working=WorkingMemory(),
                    vault_root=root,
                    scope_note_paths=("allowed/a.md",),
                ),
            ).execute(ToolCall(id="1", name="grep_vault", arguments={"query": "needle", "limit": 10}))
            self.assertTrue(result.ok)
            self.assertEqual(result.output["source_paths"], ["allowed/a.md"])
            self.assertEqual(result.output["matches"][0]["path"], "allowed/a.md")

    def test_list_notes_filters_scope_and_truncates(self) -> None:
        registry = ToolRegistry()
        register_vault_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "allowed").mkdir()
            (root / "outside").mkdir()
            (root / "allowed" / "a.md").write_text("a", encoding="utf-8")
            (root / "allowed" / "b.md").write_text("b", encoding="utf-8")
            (root / "outside" / "c.md").write_text("c", encoding="utf-8")
            result = ToolExecutor(
                registry,
                ToolExecutionContext(
                    working=WorkingMemory(),
                    vault_root=root,
                    scope_note_paths=("allowed/a.md", "allowed/b.md"),
                    scope_type="selected_notes",
                ),
            ).execute(ToolCall(id="1", name="list_notes", arguments={"limit": 1}))
            self.assertTrue(result.ok)
            self.assertEqual(result.output["total"], 2)
            self.assertEqual(result.output["result_count"], 1)
            self.assertTrue(result.output["truncated"])
            self.assertEqual(result.output["source_paths"], ["allowed/a.md"])

    def test_online_search_uses_configured_client(self) -> None:
        registry = ToolRegistry()
        register_vault_tools(registry)
        result = ToolExecutor(
            registry,
            ToolExecutionContext(working=WorkingMemory(), online_search_client=FakeOnlineClient()),
        ).execute(ToolCall(id="1", name="online_search", arguments={"query": "agent memory", "top_k": 2}))
        self.assertTrue(result.ok)
        self.assertTrue(result.output["enabled"])
        self.assertEqual(result.output["provider"], "fake")


def search_result(path: str, text: str) -> SearchResult:
    return SearchResult(
        chunk=TextChunk(
            chunk_id=f"{path}#abc",
            note_path=path,
            heading_path=["Heading"],
            text=text,
            start_line=1,
            end_line=3,
        ),
        score=0.5,
    )


if __name__ == "__main__":
    unittest.main()
