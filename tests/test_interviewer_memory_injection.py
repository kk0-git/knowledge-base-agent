import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from agent.apps.interview_interviewer import (
    InterviewTurnRequest,
    build_interviewer_runtime_context,
    build_turn_input,
)
from agent.interview.state import build_interview_state_machine
from services.memory.derived import rebuild_derived
from services.memory.injection import render_interviewer_memory_context
from services.memory.schema import default_learner_model
from services.memory.types import default_sr
from services.workflows.interview_profile import InterviewProfileStore
from tests.test_interview_state_tools import active_state, sample_plan


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
            "topic": "MCP",
            "planned_layer": "definition" if index == 1 else "roles",
            "category": "knowledge_gap",
            "domain_anchor": {
                "topic": "MCP",
                "scope_path": "个人/MCP",
                "source_note_paths": ["notes/mcp/a.md"],
            },
            "evidence_refs": [{"summary": f"turn evidence {index}"}],
            "sr": {**default_sr(TODAY), "next_review": "2026-06-20" if index <= 3 else "2026-12-31"},
        }
        for index, lifecycle in enumerate(
            ["active", "active", "active", "active", "active", "candidate", "archived"],
            start=1,
        )
    ]
    model["procedures"] = [
        {"id": "proc-1", "lifecycle": "active", "title": "Prefer concise follow-ups"},
        {"id": "proc-2", "lifecycle": "active", "title": "Pause before tradeoff questions"},
        {"id": "proc-3", "lifecycle": "active", "title": "Hidden procedure"},
    ]
    model["commitments"] = [{"id": "c-1", "action": "confirm_belief", "note": "User confirmed improved on caching"}]
    return rebuild_derived(model, today=TODAY)


def test_render_interviewer_memory_context_respects_budget():
    rendered = render_interviewer_memory_context(
        model=_model_with_beliefs(),
        current_topic="MCP",
        planned_layer="definition",
        scope_note_paths=("notes/mcp/a.md",),
    )

    assert "Gap 1" in rendered
    assert "Gap 5" in rendered
    assert "Gap 6" not in rendered
    assert "Gap 7" not in rendered
    assert rendered.count("latest evidence:") <= 5
    assert "Prefer concise follow-ups" in rendered
    assert "Pause before tradeoff questions" in rendered
    assert "Hidden procedure" not in rendered
    assert "User confirmed improved on caching" in rendered
    assert "recall_profile" in rendered


def test_render_interviewer_memory_context_empty_when_no_active():
    model = default_learner_model()
    model["beliefs"] = [{"id": "wp-c", "lifecycle": "candidate", "point": "Hidden"}]
    rendered = render_interviewer_memory_context(model=model, current_topic="MCP")
    assert rendered == ""


def test_render_interviewer_memory_context_formats_confusion_pair():
    model = default_learner_model()
    model["canonical_revision"] = 1
    model["beliefs"] = [
        {
            "id": "wp-confuse",
            "kind": "confusion_pair",
            "lifecycle": "active",
            "left": "Host",
            "right": "Client",
            "distinction": "谁发起连接",
            "scope": "domain",
            "topic": "MCP",
            "domain_anchor": {
                "topic": "MCP",
                "scope_path": "个人/MCP",
                "source_note_paths": ["notes/mcp/a.md"],
            },
            "sr": {**default_sr(TODAY), "next_review": "2026-12-31"},
        }
    ]
    rendered = render_interviewer_memory_context(
        model=rebuild_derived(model, today=TODAY),
        current_topic="MCP",
        scope_note_paths=("notes/mcp/a.md",),
    )

    assert "Host vs Client: 谁发起连接" in rendered
    assert "probe:" in rendered


def test_build_turn_input_includes_memory_block(tmp_path):
    profile_store = InterviewProfileStore(tmp_path / "profile.json")
    profile_store.save_v4(_model_with_beliefs())
    plan = sample_plan()
    machine = build_interview_state_machine(plan=plan, state_payload=active_state())
    request = InterviewTurnRequest(
        query="hello",
        interview_plan=plan,
        interview_state=active_state(),
        profile_store=profile_store,
        scope_note_paths=("notes/mcp/a.md",),
    )
    context = build_interviewer_runtime_context(request, machine, profile_store)
    text = build_turn_input(request, runtime_context=context)

    assert "# Learner Memory Background" in text
    assert "Gap 1" in text
    assert context["memory_context"]
    assert context["profile"]["universal_weak_points"] == []
    assert "learner_memory_background" in context["tool_boundaries"]["preloaded"]
