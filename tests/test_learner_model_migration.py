import json
import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from services.memory.migration import migrate_v3_profile_to_v4
from services.memory.schema import default_learner_model, normalize_learner_model
from services.memory.store import LearnerModelStore


def test_empty_v3_profile_migrates_to_v4_default_shape():
    model = migrate_v3_profile_to_v4({})

    assert model["schema_version"] == 5
    assert model["canonical_revision"] == 1
    assert model["learner_items"] == []
    assert model["assistant_items"] == []
    assert model["derived"]["stale"] is True
    assert "legacy" in model


def test_v3_weak_points_migrate_to_active_beliefs_with_evidence_refs():
    model = migrate_v3_profile_to_v4(
        {
            "weak_points": [
                {
                    "point": "RRF k value rationale is fuzzy",
                    "topic": "RAG",
                    "category": "knowledge_gap",
                    "evidence": "Could not explain why k=60 stays default.",
                    "source_session_ids": ["s1"],
                    "domain_anchor": {
                        "topic": "RAG",
                        "scope_path": ["interview", "rag"],
                        "source_note_paths": ["notes/rag.md"],
                    },
                }
            ],
            "communication": {"style": "too verbose"},
            "topic_mastery": {"RAG": {"level": "medium"}},
        }
    )

    belief = model["learner_items"][0]
    assert belief["id"].startswith("weak-")
    assert belief["lifecycle"] == "active"
    assert belief["kind"] == "standard"
    assert belief["source_session_ids"] == ["s1"]
    assert belief["source_kinds"] == ["interview"]
    assert belief["evidence_refs"][0]["summary"] == "Could not explain why k=60 stays default."
    assert belief["evidence_refs"][0]["session_id"] == "s1"
    assert model["legacy"]["communication"]["style"] == "too verbose"
    assert model["legacy"]["topic_mastery"]["RAG"]["level"] == "medium"


def test_normalize_v4_input_preserves_unknown_fields():
    model = default_learner_model()
    model["custom_top_level"] = {"keep": True}
    model["learner_items"].append(
        {
            "id": "wp-1",
            "point": "Chunking tradeoff is unclear",
            "custom_belief_field": 123,
        }
    )

    normalized = normalize_learner_model(model)

    assert normalized["custom_top_level"] == {"keep": True}
    assert normalized["learner_items"][0]["custom_belief_field"] == 123
    assert normalized["learner_items"][0]["lifecycle"] == "candidate"


def test_store_loads_legacy_in_memory_and_save_writes_only_v4(tmp_path):
    v4_path = tmp_path / "learner_model.json"
    legacy_path = tmp_path / "interview_profile.json"
    legacy_path.write_text(
        json.dumps(
            {
                "weak_points": [
                    {
                        "point": "Needs clearer answer structure",
                        "evidence": "Answer missed conclusion.",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    store = LearnerModelStore(v4_path, legacy_path=legacy_path)
    model = store.load()

    assert not v4_path.exists()
    assert model["schema_version"] == 5
    assert model["learner_items"][0]["point"] == "Needs clearer answer structure"

    model["learner_items"][0]["point"] = "Updated canonical point"
    store.save(model)

    assert v4_path.exists()
    saved = json.loads(v4_path.read_text(encoding="utf-8"))
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    assert saved["learner_items"][0]["point"] == "Updated canonical point"
    assert legacy["weak_points"][0]["point"] == "Needs clearer answer structure"
