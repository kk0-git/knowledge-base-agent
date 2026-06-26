import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from services.memory.bridge import should_skip_memory_extraction
from services.memory.bridge import observations_from_profile_extractor
from services.memory.schema import default_learner_model
from services.workflows.interview_profile import InterviewProfileStore


def test_should_skip_memory_extraction_when_checkpoint_matches():
    session = {
        "session_id": "sess-1",
        "messages": [{"role": "user", "content": "hello"}],
        "memory_extraction": {
            "evidence_hash": "abc",
            "commit_revision": 2,
        },
    }
    reviews = []

    assert should_skip_memory_extraction(session, reviews=reviews) is False

    from services.memory.bridge import compute_session_evidence_hash

    session["memory_extraction"]["evidence_hash"] = compute_session_evidence_hash(session=session, reviews=reviews)
    assert should_skip_memory_extraction(session, reviews=reviews) is True


def test_update_from_session_writes_v4_beliefs_without_llm(tmp_path):
    store = InterviewProfileStore(tmp_path / "learner_model.json")
    session = {
        "session_id": "sess-1",
        "messages": [],
        "context": {"source_note_paths": []},
        "interview_state": {},
        "interview_plan": {},
    }
    reviews = [
        {
            "feedback": {"gaps": ["Missed hybrid retrieval"], "coach_note": "needs detail"},
            "context_note_paths": [],
        }
    ]

    final_review, update = store.update_from_session(
        session=session,
        reviews=reviews,
        llm_client=None,
        model=None,
    )

    model = store.load_v4()
    profile = store.load()

    assert update["source"] == "turn_review_signals_fallback"
    assert len(model["beliefs"]) >= 1
    assert len(profile["weak_points"]) >= 1
    assert session["memory_extraction"]["commit_revision"] == model["canonical_revision"]
    assert isinstance(final_review, dict)


def test_profile_extractor_maps_procedure_without_belief_pollution():
    session = {"session_id": "sess-1", "context": {"source_note_paths": []}}
    mapped = observations_from_profile_extractor(
        [
            {
                "type": "procedure",
                "procedure_key": "answer_format.conclusion_first",
                "title": "回答先给结论",
                "steps": ["先给结论"],
                "evidence": "用户要求偏好",
                "confidence": "medium",
            },
            {
                "type": "weak_point",
                "category": "thinking_pattern",
                "point": "容易把架构问题简单归因",
                "evidence": "面试回答中多次出现",
                "confidence": "medium",
            },
        ],
        session=session,
        model=default_learner_model(),
    )

    assert [item["op"] for item in mapped] == ["propose_procedure", "propose_belief"]
    assert mapped[0]["procedure_key"] == "answer_format.conclusion_first"
    assert mapped[1]["category"] == "thinking_pattern"


def test_profile_extractor_maps_communication_suggestions_to_procedures():
    session = {"session_id": "sess-1", "context": {"source_note_paths": []}}
    mapped = observations_from_profile_extractor(
        [],
        session=session,
        model=default_learner_model(),
        communication={"suggestions": ["回答前先给一句结论"]},
    )

    assert len(mapped) == 1
    assert mapped[0]["op"] == "propose_procedure"
    assert mapped[0]["title"] == "回答前先给一句结论"


def test_profile_extractor_maps_confusion_pair():
    session = {"session_id": "sess-1", "context": {"source_note_paths": []}}
    mapped = observations_from_profile_extractor(
        [
            {
                "type": "confusion_pair",
                "left": "BM25",
                "right": "Dense",
                "distinction": "关键词匹配 vs 语义匹配",
                "evidence": "用户混淆",
                "confidence": "medium",
            }
        ],
        session=session,
        model=default_learner_model(),
    )

    assert mapped[0]["op"] == "propose_belief"
    assert mapped[0]["kind"] == "confusion_pair"
    assert mapped[0]["left"] == "BM25"
    assert mapped[0]["right"] == "Dense"
