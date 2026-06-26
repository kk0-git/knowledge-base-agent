import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from services.memory.ui import enrich_memory_item_for_ui, evidence_links_from_refs
from services.workflows.interview_profile import InterviewProfileStore


def test_evidence_links_interview_session():
    refs = [
        {
            "source_kind": "interview",
            "session_id": "20260626-abc",
            "summary": "missed MCP host role",
        }
    ]
    links = evidence_links_from_refs(refs)

    assert len(links) == 1
    assert links[0]["href"] == "/?mode=interview&session_id=20260626-abc"
    assert "面试" in links[0]["label"]
    assert links[0]["summary"] == "missed MCP host role"


def test_evidence_links_answer_session():
    refs = [
        {
            "source_kind": "answer",
            "session_id": "answer-42",
            "summary": "confused BM25 with dense",
        }
    ]
    links = evidence_links_from_refs(refs)

    assert links[0]["href"] == "/?mode=answer&session_id=answer-42"
    assert "问答" in links[0]["label"]


def test_evidence_links_review_run():
    refs = [
        {
            "source_kind": "review",
            "review_run_id": "review-run-1",
            "summary": "retry on card",
        }
    ]
    links = evidence_links_from_refs(refs)

    assert links[0]["href"] == "/review?run=review-run-1"
    assert "复习" in links[0]["label"]


def test_evidence_links_summary_only_fallback():
    refs = [{"source_kind": "interview", "summary": "legacy text only"}]
    links = evidence_links_from_refs(refs)

    assert links[0]["href"] == ""
    assert links[0]["summary"] == "legacy text only"


def test_enrich_memory_item_for_ui_attaches_links():
    item = {
        "id": "wp-1",
        "point": "Gap",
        "evidence_refs": [
            {
                "source_kind": "interview",
                "session_id": "s1",
                "summary": "turn 3 gap",
            }
        ],
    }
    enriched = enrich_memory_item_for_ui(item)

    assert enriched["point"] == "Gap"
    assert len(enriched["evidence_links"]) == 1
    assert enriched["evidence_links"][0]["session_id"] == "s1"


def test_list_memory_archived_includes_evidence_links(tmp_path):
    store = InterviewProfileStore(tmp_path / "learner_model.json")
    model = store.load_v4()
    model["beliefs"] = [
        {
            "id": "wp-archived",
            "lifecycle": "archived",
            "point": "Archived gap",
            "evidence_refs": [
                {
                    "source_kind": "interview",
                    "session_id": "sess-1",
                    "summary": "from interview",
                }
            ],
        }
    ]
    model["procedures"] = [
        {
            "id": "proc-archived",
            "lifecycle": "archived",
            "title": "Archived procedure",
        }
    ]
    store.save_v4(model)

    archived = store.list_memory_archived()

    assert archived["count"] == 2
    assert archived["beliefs"][0]["id"] == "wp-archived"
    assert archived["beliefs"][0]["evidence_links"][0]["href"].endswith("sess-1")
    assert archived["procedures"][0]["id"] == "proc-archived"


def test_list_memory_candidates_includes_evidence_links(tmp_path):
    store = InterviewProfileStore(tmp_path / "learner_model.json")
    model = store.load_v4()
    model["beliefs"] = [
        {
            "id": "wp-candidate",
            "lifecycle": "candidate",
            "point": "Candidate gap",
            "evidence_refs": [
                {
                    "source_kind": "answer",
                    "session_id": "ans-9",
                    "summary": "from answer",
                }
            ],
        }
    ]
    store.save_v4(model)

    candidates = store.list_memory_candidates()

    assert candidates["beliefs"][0]["evidence_links"][0]["href"] == "/?mode=answer&session_id=ans-9"
