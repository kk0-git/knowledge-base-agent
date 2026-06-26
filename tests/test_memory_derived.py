import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from services.memory.derived import rebuild_derived
from services.memory.schema import default_learner_model
from services.memory.types import default_sr


TODAY = "2026-06-25"


def test_rebuild_derived_marks_fresh_and_sets_generation():
    model = default_learner_model()
    model["canonical_revision"] = 3
    model["beliefs"] = [
        {
            "id": "wp-1",
            "lifecycle": "active",
            "point": "RRF unclear",
            "topic": "RAG",
            "domain_anchor": {"topic": "RAG", "scope_path": ["个人/RAG"], "source_note_paths": ["notes/rag.md"]},
            "sr": {**default_sr(TODAY), "next_review": "2026-06-20"},
        },
        {
            "id": "wp-2",
            "lifecycle": "active",
            "point": "Later review",
            "topic": "RAG",
            "domain_anchor": {"topic": "RAG", "scope_path": ["个人/RAG"], "source_note_paths": ["notes/rag.md"]},
            "sr": {**default_sr(TODAY), "next_review": "2026-12-31"},
        },
        {"id": "wp-c", "lifecycle": "candidate", "point": "ignored"},
    ]

    rebuilt = rebuild_derived(model, today=TODAY)
    derived = rebuilt["derived"]

    assert derived["stale"] is False
    assert derived["generation"] == 3
    assert len(derived["domains"]) == 1
    assert derived["domains"][0]["active_belief_ids"] == ["wp-1", "wp-2"]
    assert derived["domains"][0]["due_belief_ids"] == ["wp-1"]
    assert "个人/RAG" in derived["inject_blurbs"]
