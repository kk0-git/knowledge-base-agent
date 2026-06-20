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
