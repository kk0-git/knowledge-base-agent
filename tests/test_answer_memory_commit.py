import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from services.memory.bridge import (
    collect_answer_citation_paths,
    compute_session_evidence_hash,
    observations_from_answer_extractor,
    should_skip_memory_extraction,
)
from services.workflows.answer_memory_commit import commit_answer_memory
from services.workflows.answer_sessions import AnswerSessionStore
from services.workflows.interview_profile import InterviewProfileStore


def _answer_session(**overrides):
    session = {
        "session_id": "20260624-rag-abc123",
        "status": "archived",
        "updated_at": "2026-06-24T10:00:00+00:00",
        "messages": [
            {"role": "user", "content": "RRF 怎么融合？"},
            {
                "role": "assistant",
                "content": "RRF 通过 reciprocal rank 融合 dense 和 sparse 结果。",
                "citations": [{"path": "notes/rag/rrf.md"}],
            },
        ],
        "context": {
            "scope_type": "tag",
            "scope_value": "个人/RAG",
            "scope_paths": [],
        },
    }
    session.update(overrides)
    return session


def test_collect_answer_citation_paths():
    paths = collect_answer_citation_paths(_answer_session())
    assert paths == ["notes/rag/rrf.md"]


def test_update_from_answer_session_commits_candidate_with_mock_llm(tmp_path):
    profile_store = InterviewProfileStore(tmp_path / "learner_model.json")
    session_store = AnswerSessionStore(tmp_path / "answer-sessions")
    session_store.save_session(_answer_session(status="active"))

    llm = MagicMock()
    llm.complete.return_value = MagicMock(
        content='{"observations":[{"type":"weak_point","point":"RRF 融合原理仍不稳定","evidence":"用户追问","confidence":"medium","category":"knowledge_gap","scope_suggestion":"domain","topic":"RAG"}]}'
    )

    session = session_store.load_session("20260624-rag-abc123")
    update, updated_session = profile_store.update_from_answer_session(
        session=session,
        llm_client=llm,
        model="test-model",
    )

    model = profile_store.load_v4()
    assert update["source"] == "llm"
    assert update["observation_count"] == 1
    assert len(model["learner_items"]) == 1
    assert model["learner_items"][0]["lifecycle"] == "candidate"
    assert model["learner_items"][0]["source_kinds"] == ["answer"]
    assert updated_session["memory_extraction"]["trigger"] == "answer.session_ended"


def test_high_confidence_answer_observation_stays_candidate_without_explicit(tmp_path):
    profile_store = InterviewProfileStore(tmp_path / "learner_model.json")
    session = _answer_session()

    mapped = observations_from_answer_extractor(
        [
            {
                "type": "weak_point",
                "point": "不懂 BM25",
                "evidence": "用户说不懂",
                "confidence": "high",
                "category": "knowledge_gap",
                "scope_suggestion": "domain",
                "topic": "RAG",
            }
        ],
        session=session,
        model=profile_store.load_v4(),
    )
    assert mapped[0]["target_lifecycle"] == "active"

    update, _ = profile_store.update_from_answer_session(
        session=session,
        llm_client=MagicMock(
            complete=MagicMock(
                return_value=MagicMock(
                    content='{"observations":[{"type":"weak_point","point":"不懂 BM25","evidence":"用户说不懂","confidence":"high","category":"knowledge_gap","scope_suggestion":"domain","topic":"RAG"}]}'
                )
            )
        ),
        model="test-model",
    )
    belief = profile_store.load_v4()["learner_items"][0]
    # Answer + high confidence without explicit admission or corroboration → candidate (§5 scoring)
    assert belief["lifecycle"] == "candidate"
    assert update["canonical_revision"] == 1


def test_citation_paths_produce_domain_anchor(tmp_path):
    profile_store = InterviewProfileStore(tmp_path / "learner_model.json")
    mapped = observations_from_answer_extractor(
        [
            {
                "type": "weak_point",
                "point": "chunk 策略不清楚",
                "evidence": "追问 chunk",
                "confidence": "medium",
                "category": "knowledge_gap",
                "scope_suggestion": "domain",
                "topic": "RAG",
            }
        ],
        session=_answer_session(),
        model=profile_store.load_v4(),
    )
    anchor = mapped[0]["domain_anchor"]
    assert anchor.get("context_note_paths") == ["notes/rag/rrf.md"]


def test_answer_extractor_maps_procedure_observation(tmp_path):
    profile_store = InterviewProfileStore(tmp_path / "learner_model.json")
    mapped = observations_from_answer_extractor(
        [
            {
                "type": "procedure",
                "procedure_key": "answer_format.conclusion_first",
                "title": "回答先给结论",
                "steps": ["先给结论", "再解释依据"],
                "evidence": "用户要求回答先给结论",
                "confidence": "medium",
            }
        ],
        session=_answer_session(),
        model=profile_store.load_v4(),
    )

    assert mapped == [
        {
            "op": "propose_procedure",
            "source_kind": "answer",
            "confidence": "medium",
            "procedure_key": "answer_format.conclusion_first",
            "title": "回答先给结论",
            "description": "",
            "steps": ["先给结论", "再解释依据"],
            "scope": "universal",
            "session_id": "20260624-rag-abc123",
            "evidence_summary": "用户要求回答先给结论",
        }
    ]


def test_answer_extractor_maps_confusion_pair_observation(tmp_path):
    profile_store = InterviewProfileStore(tmp_path / "learner_model.json")
    mapped = observations_from_answer_extractor(
        [
            {
                "type": "confusion_pair",
                "left": "BM25",
                "right": "Dense retrieval",
                "distinction": "关键词匹配 vs 语义匹配",
                "evidence": "用户混用了两者",
                "confidence": "medium",
                "topic": "RAG",
            }
        ],
        session=_answer_session(),
        model=profile_store.load_v4(),
    )

    assert mapped[0]["op"] == "propose_belief"
    assert mapped[0]["kind"] == "confusion_pair"
    assert mapped[0]["left"] == "BM25"
    assert mapped[0]["right"] == "Dense retrieval"
    assert mapped[0]["distinction"] == "关键词匹配 vs 语义匹配"


def test_rules_fallback_on_llm_failure(tmp_path):
    profile_store = InterviewProfileStore(tmp_path / "learner_model.json")
    session = _answer_session(
        messages=[{"role": "user", "content": "我不太懂 RRF 融合"}],
    )
    update, _ = profile_store.update_from_answer_session(
        session=session,
        llm_client=None,
        model=None,
    )
    assert update["source"] == "rules_fallback"
    assert profile_store.load_v4()["learner_items"][0]["lifecycle"] == "active"


def test_checkpoint_skips_second_extract(tmp_path):
    profile_store = InterviewProfileStore(tmp_path / "learner_model.json")
    session_store = AnswerSessionStore(tmp_path / "answer-sessions")
    session = _answer_session(status="active")
    session_store.save_session(session)

    llm = MagicMock()
    llm.complete.return_value = MagicMock(content='{"observations":[]}')

    session = session_store.load_session(session["session_id"])
    _, updated_session = profile_store.update_from_answer_session(session=session, llm_client=llm, model="test-model")
    session_store.save_session(updated_session)
    session = session_store.load_session(session["session_id"])

    update, _ = profile_store.update_from_answer_session(session=session, llm_client=llm, model="test-model")
    assert update["source"] == "memory_extraction_checkpoint"
    assert update["operations"]["skipped"] is True


def test_empty_messages_do_not_change_revision(tmp_path):
    profile_store = InterviewProfileStore(tmp_path / "learner_model.json")
    session = _answer_session(messages=[])
    llm = MagicMock()
    llm.complete.return_value = MagicMock(content='{"observations":[]}')

    update, _ = profile_store.update_from_answer_session(session=session, llm_client=llm, model="test-model")
    assert update["observation_count"] == 0
    assert profile_store.load_v4()["canonical_revision"] == 0


def test_commit_answer_memory_persists_session_checkpoint(tmp_path):
    profile_store = InterviewProfileStore(tmp_path / "learner_model.json")
    session_store = AnswerSessionStore(tmp_path / "answer-sessions")
    session_store.save_session(_answer_session(status="archived"))

    llm = MagicMock()
    llm.complete.return_value = MagicMock(
        content='{"observations":[{"type":"weak_point","point":"测试弱项","evidence":"ev","confidence":"medium","category":"knowledge_gap","scope_suggestion":"universal"}]}'
    )

    session = session_store.load_session("20260624-rag-abc123")
    audit, _ = commit_answer_memory(
        session_store=session_store,
        profile_store=profile_store,
        session=session,
        llm_client=llm,
        model="test-model",
    )
    saved = session_store.load_session("20260624-rag-abc123")
    assert saved.get("memory_extraction")
    assert audit["observation_count"] == 1
    assert should_skip_memory_extraction(saved, reviews=[])
