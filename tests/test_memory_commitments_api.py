import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import pytest

from services.workflows.interview_profile import InterviewProfileStore


def _seed_model(path: Path) -> InterviewProfileStore:
    store = InterviewProfileStore(path / "learner_model.json")
    model = store.load_v4()
    model["beliefs"] = [
        {"id": "wp-candidate", "lifecycle": "candidate", "point": "Candidate"},
        {"id": "wp-active", "lifecycle": "active", "point": "Active"},
        {"id": "wp-archived", "lifecycle": "archived", "point": "Archived"},
    ]
    model["procedures"] = [
        {"id": "proc-candidate", "lifecycle": "candidate", "title": "Candidate procedure"},
        {"id": "proc-active", "lifecycle": "active", "title": "Active procedure"},
        {"id": "proc-archived", "lifecycle": "archived", "title": "Archived procedure"},
    ]
    store.save_v4(model)
    return store


def test_apply_user_commit_confirm_deny_restore(tmp_path):
    store = _seed_model(tmp_path)

    result = store.apply_user_commit(action="confirm_candidate", belief_id="wp-candidate")
    assert result["belief"]["lifecycle"] == "active"

    result = store.apply_user_commit(action="deny_belief", belief_id="wp-active")
    assert result["belief"]["lifecycle"] == "archived"

    result = store.apply_user_commit(action="restore_archived", belief_id="wp-archived")
    assert result["belief"]["lifecycle"] == "active"


def test_apply_user_commit_missing_belief_raises(tmp_path):
    store = _seed_model(tmp_path)
    with pytest.raises(KeyError):
        store.apply_user_commit(action="confirm_candidate", belief_id="missing")


def test_apply_user_commit_invalid_transition_raises(tmp_path):
    store = _seed_model(tmp_path)
    with pytest.raises(ValueError):
        store.apply_user_commit(action="confirm_candidate", belief_id="wp-active")


def test_list_candidate_beliefs(tmp_path):
    store = _seed_model(tmp_path)
    candidates = store.list_candidate_beliefs()
    assert len(candidates) == 1
    assert candidates[0]["id"] == "wp-candidate"


def test_apply_user_commit_procedure_actions(tmp_path):
    store = _seed_model(tmp_path)

    result = store.apply_user_commit(
        action="confirm_procedure",
        procedure_id="proc-candidate",
        target_type="procedure",
    )
    assert result["procedure"]["lifecycle"] == "active"

    result = store.apply_user_commit(
        action="deny_procedure",
        procedure_id="proc-active",
        target_type="procedure",
    )
    assert result["procedure"]["lifecycle"] == "archived"

    result = store.apply_user_commit(
        action="restore_procedure",
        procedure_id="proc-archived",
        target_type="procedure",
    )
    assert result["procedure"]["lifecycle"] == "active"


def test_list_memory_candidates_includes_beliefs_and_procedures(tmp_path):
    store = _seed_model(tmp_path)
    candidates = store.list_memory_candidates()

    assert candidates["count"] == 2
    assert candidates["beliefs"][0]["id"] == "wp-candidate"
    assert candidates["procedures"][0]["id"] == "proc-candidate"


def test_list_memory_archived_includes_beliefs_and_procedures(tmp_path):
    store = _seed_model(tmp_path)
    archived = store.list_memory_archived()

    assert archived["count"] == 2
    assert archived["beliefs"][0]["id"] == "wp-archived"
    assert archived["procedures"][0]["id"] == "proc-archived"
