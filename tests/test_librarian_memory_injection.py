import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from services.memory.derived import rebuild_derived
from services.memory.injection import render_librarian_memory_context
from services.memory.schema import default_learner_model
from services.memory.types import default_sr


TODAY = "2026-06-25"


def _model_with_beliefs():
    model = default_learner_model()
    model["canonical_revision"] = 2
    model["beliefs"] = [
        {
            "id": f"wp-{index}",
            "lifecycle": lifecycle,
            "point": f"Gap {index}",
            "scope": "domain",
            "topic": "RAG",
            "domain_anchor": {
                "topic": "RAG",
                "scope_path": "个人/RAG",
                "source_note_paths": ["notes/rag/a.md"],
            },
            "sr": {**default_sr(TODAY), "next_review": "2026-06-20" if index <= 2 else "2026-12-31"},
        }
        for index, lifecycle in enumerate(
            ["active", "active", "active", "candidate", "archived"],
            start=1,
        )
    ]
    return rebuild_derived(model, today=TODAY)


def test_render_librarian_memory_context_respects_budget():
    rendered = render_librarian_memory_context(
        model=_model_with_beliefs(),
        scope_note_paths=("notes/rag/a.md",),
        scope_value="个人/RAG",
    )

    assert "Gap 1" in rendered
    assert "Gap 2" in rendered
    assert "Gap 3" in rendered
    assert "Gap 4" not in rendered
    assert "Gap 5" not in rendered
    assert rendered.count("[due]") <= 2


def test_render_librarian_memory_context_empty_when_no_active():
    model = default_learner_model()
    model["beliefs"] = [{"id": "wp-c", "lifecycle": "candidate", "point": "Hidden"}]
    rendered = render_librarian_memory_context(model=model, scope_note_paths=())
    assert rendered == ""


def test_render_librarian_memory_context_formats_confusion_pair():
    model = default_learner_model()
    model["canonical_revision"] = 1
    model["beliefs"] = [
        {
            "id": "wp-confuse",
            "kind": "confusion_pair",
            "lifecycle": "active",
            "left": "BM25",
            "right": "Dense",
            "distinction": "关键词匹配 vs 语义匹配",
            "scope": "domain",
            "topic": "RAG",
            "domain_anchor": {
                "topic": "RAG",
                "scope_path": "个人/RAG",
                "source_note_paths": ["notes/rag/a.md"],
            },
            "sr": {**default_sr(TODAY), "next_review": "2026-12-31"},
        }
    ]
    rendered = render_librarian_memory_context(
        model=rebuild_derived(model, today=TODAY),
        scope_note_paths=("notes/rag/a.md",),
        scope_value="个人/RAG",
    )

    assert "BM25 vs Dense: 关键词匹配 vs 语义匹配" in rendered
