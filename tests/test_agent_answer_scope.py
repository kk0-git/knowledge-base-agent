from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from services.rag.agent_answer import AgentAnswerPipeline
from services.rag.grep_search import GrepMatch
from services.rag.intent_router import ConversationCommand, RouterDecision
from services.rag.schema import SearchResult, TextChunk


def search_result(path: str, text: str) -> SearchResult:
    return SearchResult(
        score=1.0,
        chunk=TextChunk(
            chunk_id=f"{path}:1",
            note_path=path,
            heading_path=(),
            start_line=1,
            end_line=1,
            text=text,
        ),
    )


class AgentAnswerScopeTests(unittest.TestCase):
    def test_retrieve_filters_results_to_folder_scope(self) -> None:
        pipeline = AgentAnswerPipeline(
            router=MagicMock(),
            llm_client=MagicMock(),
            llm_model="test-model",
            manager=MagicMock(),
            vault_root=Path("."),
        )
        pipeline.router.route.return_value = RouterDecision(
            command=ConversationCommand.NOTES,
            reason="test",
            confidence=1.0,
            raw_response="",
            tool_args={},
            fallback_used=False,
        )
        pipeline.run_notes_search = MagicMock(
            return_value=[
                search_result("个人/面试/agent面试/a.md", "in scope"),
                search_result("个人/面试/基础/HTTP缓存.md", "outside scope"),
            ]
        )
        pipeline.run_rg_search = MagicMock(return_value=[])
        pipeline.run_bm25_search = MagicMock(return_value=[])

        result = pipeline.retrieve(
            "cache",
            scope_note_paths=("个人/面试/agent面试/a.md",),
            scope_type="folder",
        )

        self.assertEqual(len(result.notes_results), 1)
        self.assertEqual(result.notes_results[0].chunk.note_path, "个人/面试/agent面试/a.md")
        paths = {item["path"] for item in result.context_items if item.get("path")}
        self.assertNotIn("个人/面试/基础/HTTP缓存.md", paths)


if __name__ == "__main__":
    unittest.main()
