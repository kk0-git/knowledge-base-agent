import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from services.memory.commit import commit_observations
from services.memory.schema import default_learner_model


TODAY = "2026-06-25"


def domain_anchor(topic="RAG", note="notes/rag.md"):
    return {
        "topic": topic,
        "scope_path": ["interview", topic.lower()],
        "source_note_paths": [note],
        "evidence_terms": [topic],
    }


def test_low_confidence_is_filtered_without_revision_change():
    model, ops = commit_observations(
        default_learner_model(),
        [
            {
                "op": "propose_belief",
                "confidence": "low",
                "source_kind": "answer",
                "point": "Maybe weak",
            }
        ],
        today=TODAY,
    )

    assert model["learner_items"] == []
    assert ops["filtered_low_count"] == 1
    assert model["canonical_revision"] == 0


def test_propose_belief_default_lifecycle_rules():
    model, _ = commit_observations(
        default_learner_model(),
        [
            {
                "op": "propose_belief",
                "source_kind": "answer",
                "confidence": "high",
                "target_lifecycle": "active",
                "point": "Can explain RRF clearly",
                "domain_anchor": domain_anchor(),
            },
            {
                "op": "propose_belief",
                "source_kind": "review",
                "confidence": "high",
                "point": "Review finding stays candidate",
                "domain_anchor": domain_anchor("Review", "notes/review.md"),
            },
            {
                "op": "propose_belief",
                "source_kind": "interview",
                "confidence": "medium",
                "point": "Interview finding defaults active",
                "domain_anchor": domain_anchor("Interview", "notes/interview.md"),
            },
        ],
        today=TODAY,
    )

    # §5 evidence scoring: answer+high without explicit/corroboration → candidate; interview+medium → active
    assert [belief["lifecycle"] for belief in model["learner_items"]] == ["candidate", "candidate", "active"]


def test_standard_belief_merges_when_scope_domain_category_and_anchor_match():
    model = default_learner_model()
    model, _ = commit_observations(
        model,
        [
            {
                "op": "propose_belief",
                "source_kind": "interview",
                "point": "RRF fusion rationale is unclear",
                "category": "knowledge_gap",
                "scope": "domain",
                "domain_anchor": domain_anchor(),
                "evidence": "first",
            }
        ],
        today=TODAY,
    )
    model, ops = commit_observations(
        model,
        [
            {
                "op": "propose_belief",
                "source_kind": "interview",
                "point": "RRF fusion rationale is unclear.",
                "category": "knowledge_gap",
                "scope": "domain",
                "domain_anchor": domain_anchor(),
                "evidence": "second",
            }
        ],
        today=TODAY,
    )

    assert len(model["learner_items"]) == 1
    assert model["learner_items"][0]["times_seen"] == 2
    assert len(model["learner_items"][0]["evidence_refs"]) == 2
    assert ops["updated_beliefs"] == 1


def test_archived_belief_does_not_merge_or_revive():
    model = default_learner_model()
    model["learner_items"].append(
        {
            "id": "wp-old",
            "lifecycle": "archived",
            "point": "RRF fusion rationale is unclear",
            "category": "knowledge_gap",
            "scope": "domain",
            "domain_anchor": domain_anchor(),
        }
    )

    model, _ = commit_observations(
        model,
        [
            {
                "op": "propose_belief",
                "source_kind": "answer",
                "confidence": "high",
                "point": "RRF fusion rationale is unclear",
                "category": "knowledge_gap",
                "scope": "domain",
                "domain_anchor": domain_anchor(),
            }
        ],
        today=TODAY,
    )

    assert len(model["learner_items"]) == 2
    assert model["learner_items"][0]["lifecycle"] == "archived"
    assert model["learner_items"][1]["lifecycle"] == "candidate"


def test_contradiction_archives_old_active_belief():
    model = default_learner_model()
    model["learner_items"].append(
        {
            "id": "wp-old",
            "lifecycle": "active",
            "point": "Dense retrieval is always enough",
            "category": "knowledge_gap",
            "scope": "domain",
            "domain_anchor": domain_anchor(),
        }
    )

    model, ops = commit_observations(
        model,
        [
            {
                "op": "propose_belief",
                "source_kind": "interview",
                "confidence": "medium",
                "point": "Dense retrieval is not enough for exact terms",
                "category": "knowledge_gap",
                "scope": "domain",
                "domain_anchor": domain_anchor("BM25", "notes/bm25.md"),
                "contradicts_belief_id": "wp-old",
            }
        ],
        today=TODAY,
    )

    assert model["learner_items"][0]["lifecycle"] == "archived"
    assert ops["archived_beliefs"] == 1
    assert model["commitments"][0]["action"] == "superseded_by"


def test_confusion_pair_merges_by_unordered_key():
    model, _ = commit_observations(
        default_learner_model(),
        [
            {
                "op": "propose_belief",
                "kind": "confusion_pair",
                "source_kind": "interview",
                "left": "BM25",
                "right": "Dense",
                "distinction": "Keyword vs semantic.",
            },
            {
                "op": "propose_belief",
                "kind": "confusion_pair",
                "source_kind": "interview",
                "left": "Dense",
                "right": "BM25",
                "distinction": "Semantic vs keyword.",
            },
        ],
        today=TODAY,
    )

    assert len(model["learner_items"]) == 1
    assert model["learner_items"][0]["distinction"] == "Semantic vs keyword."
    assert model["learner_items"][0]["times_seen"] == 2


def test_schedule_pass_only_changes_sr_fields():
    model = default_learner_model()
    model["learner_items"].append(
        {
            "id": "wp-1",
            "lifecycle": "active",
            "point": "Original point",
            "improved": False,
            "sr": {"repetitions": 0, "interval_days": 1, "ease_factor": 2.5},
        }
    )

    model, ops = commit_observations(model, [{"op": "schedule_pass", "belief_id": "wp-1"}], today=TODAY)
    belief = model["learner_items"][0]

    assert belief["point"] == "Original point"
    assert belief["lifecycle"] == "active"
    assert belief["improved"] is False
    assert belief["sr"]["last_outcome"] == "pass"
    assert belief["sr"]["repetitions"] == 1
    assert ops["schedule_updates"] == 1


def test_schedule_retry_does_not_make_negative_repetitions():
    model = default_learner_model()
    model["learner_items"].append(
        {
            "id": "wp-1",
            "lifecycle": "active",
            "point": "Original point",
            "improved": False,
            "sr": {"repetitions": 0, "interval_days": 3, "ease_factor": 2.5},
        }
    )

    model, _ = commit_observations(model, [{"op": "schedule_retry", "belief_id": "wp-1"}], today=TODAY)
    belief = model["learner_items"][0]

    assert belief["sr"]["last_outcome"] == "fail"
    assert belief["sr"]["repetitions"] == 0
    assert belief["point"] == "Original point"
    assert belief["improved"] is False


def test_improvement_only_updates_active_belief_and_adds_commitment():
    model = default_learner_model()
    model["learner_items"].append({"id": "wp-1", "lifecycle": "active", "point": "Needs practice"})
    model["learner_items"].append({"id": "wp-2", "lifecycle": "candidate", "point": "Candidate only"})

    model, ops = commit_observations(
        model,
        [
            {"op": "improvement", "belief_id": "wp-1", "note": "Answered well"},
            {"op": "improvement", "belief_id": "wp-2", "note": "Ignored"},
        ],
        today=TODAY,
    )

    assert model["learner_items"][0]["improved"] is True
    assert model["learner_items"][1].get("improved") is False
    assert model["commitments"][0]["action"] == "improved"
    assert ops["commitments_added"] == 1


def test_user_commit_confirm_deny_and_restore():
    model = default_learner_model()
    model["learner_items"].extend(
        [
            {"id": "wp-candidate", "lifecycle": "candidate", "point": "Candidate"},
            {"id": "wp-active", "lifecycle": "active", "point": "Active"},
            {"id": "wp-archived", "lifecycle": "archived", "point": "Archived"},
        ]
    )

    model, ops = commit_observations(
        model,
        [
            {"op": "user_commit", "action": "confirm_candidate", "belief_id": "wp-candidate"},
            {"op": "user_commit", "action": "deny_belief", "belief_id": "wp-active"},
            {"op": "user_commit", "action": "restore_archived", "belief_id": "wp-archived"},
        ],
        today=TODAY,
    )

    assert [belief["lifecycle"] for belief in model["learner_items"]] == ["active", "archived", "active"]
    assert ops["commitments_added"] == 3


def test_propose_procedure_defaults_candidate_and_merges_by_key():
    model = default_learner_model()
    model, ops = commit_observations(
        model,
        [
            {
                "op": "propose_procedure",
                "source_kind": "answer",
                "confidence": "medium",
                "procedure_key": "answer_format.conclusion_first",
                "title": "回答先给结论",
                "steps": ["先给结论", "再展开原因"],
                "evidence": "用户要求先给结论",
                "session_id": "answer-1",
            }
        ],
        today=TODAY,
    )

    assert ops["added_procedures"] == 1
    assert len(model["assistant_items"]) == 1
    assert model["assistant_items"][0]["lifecycle"] == "candidate"
    assert model["assistant_items"][0]["procedure_key"] == "answer_format.conclusion_first"

    model, ops = commit_observations(
        model,
        [
            {
                "op": "propose_procedure",
                "source_kind": "interview",
                "confidence": "medium",
                "procedure_key": "answer_format.conclusion_first",
                "title": "回答先给结论",
                "steps": ["最后补充例子"],
                "evidence": "面试中也提到同一偏好",
                "session_id": "interview-1",
            }
        ],
        today=TODAY,
    )

    procedure = model["assistant_items"][0]
    assert len(model["assistant_items"]) == 1
    assert procedure["times_seen"] == 2
    assert procedure["steps"] == ["先给结论", "再展开原因", "最后补充例子"]
    assert set(procedure["source_kinds"]) == {"answer", "interview"}
    assert ops["updated_procedures"] == 1


def test_propose_procedure_merges_by_title_similarity():
    model = default_learner_model()
    model, _ = commit_observations(
        model,
        [{"op": "propose_procedure", "title": "回答先给结论", "steps": ["结论优先"]}],
        today=TODAY,
    )
    model, _ = commit_observations(
        model,
        [{"op": "propose_procedure", "title": "回答先给结论。", "steps": ["再解释权衡"]}],
        today=TODAY,
    )

    assert len(model["assistant_items"]) == 1
    assert model["assistant_items"][0]["times_seen"] == 2


def test_user_commit_confirm_deny_and_restore_procedure():
    model = default_learner_model()
    model["assistant_items"].extend(
        [
            {"id": "proc-candidate", "lifecycle": "candidate", "title": "Candidate"},
            {"id": "proc-active", "lifecycle": "active", "title": "Active"},
            {"id": "proc-archived", "lifecycle": "archived", "title": "Archived"},
        ]
    )

    model, ops = commit_observations(
        model,
        [
            {"op": "user_commit", "action": "confirm_procedure", "procedure_id": "proc-candidate"},
            {"op": "user_commit", "action": "deny_procedure", "procedure_id": "proc-active"},
            {"op": "user_commit", "action": "restore_procedure", "procedure_id": "proc-archived"},
        ],
        today=TODAY,
    )

    assert [procedure["lifecycle"] for procedure in model["assistant_items"]] == ["active", "archived", "active"]
    assert ops["commitments_added"] == 3
    assert ops["archived_procedures"] == 1


def test_corroboration_promotes_candidate_to_active():
    """§5: answer+medium = candidate; second answer+medium with corroboration → active via scoring."""
    model = default_learner_model()
    model, _ = commit_observations(
        model,
        [
            {
                "op": "propose_belief",
                "source_kind": "answer",
                "confidence": "medium",
                "point": "RRF fusion rationale is unclear",
                "category": "knowledge_gap",
                "scope": "domain",
                "domain_anchor": domain_anchor(),
            }
        ],
        today=TODAY,
    )
    assert model["learner_items"][0]["lifecycle"] == "candidate"

    model, _ = commit_observations(
        model,
        [
            {
                "op": "propose_belief",
                "source_kind": "answer",
                "confidence": "medium",
                "point": "RRF fusion rationale is unclear.",
                "category": "knowledge_gap",
                "scope": "domain",
                "domain_anchor": domain_anchor(),
            }
        ],
        today=TODAY,
    )

    assert len(model["learner_items"]) == 1
    assert model["learner_items"][0]["lifecycle"] == "active"
    assert "answer" in model["learner_items"][0]["source_kinds"]


def test_revision_increments_only_when_actual_change_happens():
    model = default_learner_model()
    model, _ = commit_observations(
        model,
        [{"op": "propose_belief", "source_kind": "interview", "point": "A real change", "domain_anchor": domain_anchor()}],
        today=TODAY,
    )
    assert model["canonical_revision"] == 1

    model, ops = commit_observations(model, [{"op": "schedule_pass", "belief_id": "missing"}], today=TODAY)
    assert model["canonical_revision"] == 1
    assert ops["changed"] is False
