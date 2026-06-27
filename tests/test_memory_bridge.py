import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from services.memory.bridge import (
    beliefs_to_weak_points,
    learner_model_to_profile_view,
    observations_from_profile_extractor,
    sync_profile_view_to_model,
)
from services.memory.commit import commit_observations
from services.memory.migration import migrate_v3_profile_to_v4
from services.memory.schema import default_learner_model


TODAY = "2026-06-25"


def test_beliefs_to_weak_points_preserves_lifecycle_and_id():
    model = default_learner_model()
    model["learner_items"] = [
        {
            "id": "weak-abc",
            "lifecycle": "candidate",
            "point": "Needs review",
            "evidence_refs": [{"summary": "turn evidence"}],
        }
    ]

    weak_points = beliefs_to_weak_points(model)

    assert weak_points[0]["id"] == "weak-abc"
    assert weak_points[0]["lifecycle"] == "candidate"
    assert weak_points[0]["evidence"] == ["turn evidence"]


def test_v3_save_round_trip_through_profile_view():
    v3 = {
        "schema_version": 3,
        "weak_points": [
            {
                "point": "RRF fusion rationale is unclear",
                "topic": "RAG",
                "planned_layer": "definition",
                "category": "knowledge_gap",
                "scope": "domain",
            }
        ],
        "strong_points": [],
    }
    model = migrate_v3_profile_to_v4(v3)
    profile = learner_model_to_profile_view(model)

    assert len(profile["weak_points"]) == 1
    assert profile["weak_points"][0]["point"].startswith("RRF")


def test_observations_from_profile_extractor_maps_weak_point():
    session = {
        "session_id": "sess-1",
        "context": {"source_note_paths": ["notes/rag.md"]},
        "interview_state": {"current_topic": "RAG"},
    }
    observations = [
        {
            "type": "weak_point",
            "point": "Chunk overlap unclear",
            "category": "knowledge_gap",
            "scope_suggestion": "domain",
            "confidence": "medium",
            "evidence": "Could not explain overlap tradeoff",
        }
    ]

    mapped = observations_from_profile_extractor(observations, session=session, model=default_learner_model())

    assert mapped[0]["op"] == "propose_belief"
    assert mapped[0]["source_kind"] == "interview"
    assert mapped[0]["session_id"] == "sess-1"


def test_sync_profile_view_to_model_migrates_v3_payload():
    profile = {
        "schema_version": 3,
        "weak_points": [{"point": "A gap", "topic": "Topic", "planned_layer": "definition"}],
        "strong_points": [],
    }
    model = sync_profile_view_to_model(profile, default_learner_model())

    assert len(model["learner_items"]) == 1
    assert model["learner_items"][0]["lifecycle"] == "active"


def test_commit_mapped_interview_observation():
    session = {
        "session_id": "sess-1",
        "context": {"source_note_paths": ["notes/rag.md"]},
        "interview_state": {"current_topic": "RAG"},
    }
    observations = observations_from_profile_extractor(
        [
            {
                "type": "weak_point",
                "point": "Chunk overlap unclear",
                "category": "knowledge_gap",
                "scope_suggestion": "domain",
                "confidence": "medium",
                "evidence": "missed overlap tradeoff",
            }
        ],
        session=session,
        model=default_learner_model(),
    )

    model, ops = commit_observations(default_learner_model(), observations, today=TODAY)

    assert len(model["learner_items"]) == 1
    assert model["learner_items"][0]["lifecycle"] == "active"
    assert ops["added_beliefs"] == 1
