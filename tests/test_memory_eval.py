import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from services.memory.eval import memory_metrics
from services.memory.schema import default_learner_model
from services.memory.types import default_sr


TODAY = "2026-06-25"


def test_memory_metrics_counts_lifecycles_and_due():
    model = default_learner_model()
    model["canonical_revision"] = 3
    model["learner_items"] = [
        {
            "id": "wp-active",
            "lifecycle": "active",
            "point": "Active",
            "sr": {**default_sr(TODAY), "next_review": "2026-06-20"},
        },
        {"id": "wp-candidate", "lifecycle": "candidate", "point": "Candidate"},
        {"id": "wp-archived", "lifecycle": "archived", "point": "Archived"},
        {
            "id": "wp-confuse",
            "kind": "confusion_pair",
            "lifecycle": "active",
            "left": "BM25",
            "right": "Dense",
            "sr": {**default_sr(TODAY), "next_review": "2026-12-31"},
        },
    ]
    model["assistant_items"] = [
        {"id": "proc-active", "lifecycle": "active", "title": "Active procedure"},
        {"id": "proc-candidate", "lifecycle": "candidate", "title": "Candidate procedure"},
        {"id": "proc-archived", "lifecycle": "archived", "title": "Archived procedure"},
    ]
    model["derived"] = {"stale": True, "generation": 2}

    metrics = memory_metrics(model, today=TODAY)

    assert metrics["beliefs"]["active"] == 2
    assert metrics["beliefs"]["candidate"] == 1
    assert metrics["beliefs"]["archived"] == 1
    assert metrics["beliefs"]["confusion_pair"] == 1
    assert metrics["beliefs"]["due_active"] == 1
    assert metrics["procedures"]["active"] == 1
    assert metrics["procedures"]["candidate"] == 1
    assert metrics["procedures"]["archived"] == 1
    assert metrics["backlog"]["candidate_total"] == 2
    assert metrics["backlog"]["archived_total"] == 2
    assert metrics["backlog"]["due_active_beliefs"] == 1
    assert metrics["injection_preview"]["librarian_active_beliefs"] == 2
    assert metrics["injection_preview"]["librarian_due_markers"] == 1
    assert metrics["injection_preview"]["librarian_procedures"] == 1
    assert metrics["injection_preview"]["interviewer_active_beliefs"] == 2
    assert metrics["injection_preview"]["interviewer_due_markers"] == 1
    assert metrics["injection_preview"]["interviewer_procedures"] == 1
    assert metrics["derived"]["stale"] is True
    assert metrics["derived"]["generation_mismatch"] is True
    assert metrics["health"]["candidate_backlog"] == 2
    assert metrics["health"]["archived_total"] == 2
    assert metrics["health"]["derived_stale"] is True
