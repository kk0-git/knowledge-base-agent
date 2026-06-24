from __future__ import annotations

import sys
import shutil
import unittest
import uuid
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from services.workflows.interview_profile import InterviewProfileStore
from services.workflows.review_practice import (
    build_due_review_overview,
    build_grouped_review_prompt,
    build_weak_point_verification_query,
    build_recall_prompt,
    build_review_plan,
    commit_review_action,
    commit_review_outcome,
    grouped_review_cards,
    list_due_reviews,
    matching_strategy_constraints_for_card,
    parse_correction_payload,
    parse_grouped_verification_payload,
    parse_verification_payload,
    read_review_card_cache,
    review_card_cache_key,
    select_strategy_constraints,
    weak_point_content_hash,
    weak_point_id,
    write_review_card_cache,
)
from agent.schema import WorkingMemory
from agent.tool_executor import ToolExecutionContext
from agent.tools.review import get_due_reviews


def make_writable_test_dir() -> Path:
    root = PROJECT_ROOT / ".tmp" / "review_practice_tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    return path


class FailingLlm:
    def complete(self, request):  # noqa: ANN001
        raise TimeoutError("temporary llm failure")


class ChoosingLlm:
    def complete(self, request):  # noqa: ANN001
        class Response:
            content = (
                '{"question_type":"boundary","reason":"职责混淆适合边界题",'
                '"prompt":"任务分解和推理规划的职责边界分别是什么？",'
                '"hint":"只比较这两个概念","expected_focus":["任务分解拆目标","推理规划排执行"]}'
            )

        return Response()


class FakeProfileStore:
    def __init__(self, profile: dict):
        self.profile = profile

    def load(self) -> dict:
        return self.profile


def weak_point(
    point: str,
    *,
    next_review: str,
    topic: str = "Agent Memory",
    improved: bool = False,
    repetitions: int = 0,
    ease_factor: float = 2.2,
    category: str = "knowledge_gap",
    times_seen: int = 1,
    last_seen: str = "",
    last_reviewed: str = "",
) -> dict:
    return {
        "point": point,
        "topic": topic,
        "category": category,
        "scope": "universal",
        "planned_layer": "definition",
        "source_note_paths": ["personal/interview/agent/memory.md"],
        "times_seen": times_seen,
        "last_seen": last_seen,
        "improved": improved,
        "sr": {
            "interval_days": 1,
            "ease_factor": ease_factor,
            "repetitions": repetitions,
            "next_review": next_review,
            "last_outcome": "",
            "last_reviewed": last_reviewed,
        },
    }


class ReviewPracticeTests(unittest.TestCase):
    def test_list_due_reviews_includes_all_unimproved_with_soft_priority(self) -> None:
        today = date.today()
        profile = {
            "weak_points": [
                weak_point("recommended point", next_review=today.isoformat(), last_reviewed=(today - timedelta(days=5)).isoformat()),
                weak_point("never reviewed point", next_review=(today + timedelta(days=3)).isoformat(), topic="MCP", last_seen=(today - timedelta(days=10)).isoformat()),
                weak_point("recent point", next_review=(today + timedelta(days=3)).isoformat(), topic="Tool Use", last_reviewed=today.isoformat()),
                weak_point("improved point", next_review=(today - timedelta(days=1)).isoformat(), improved=True),
            ]
        }

        queue = list_due_reviews(profile, today=today.isoformat())

        self.assertEqual([card["point"] for card in queue["cards"]], ["never reviewed point", "recommended point", "recent point"])
        self.assertEqual([card["review_state"] for card in queue["cards"]], ["never_reviewed", "recommended", "recent"])
        self.assertEqual(queue["early_candidates"], [])
        self.assertEqual(queue["stats"]["due_count"], 3)
        self.assertEqual(queue["stats"]["candidate_count"], 3)
        self.assertEqual(queue["stats"]["overdue_count"], 0)
        self.assertEqual(queue["stats"]["topic_counts"]["Agent Memory"], 1)
        self.assertEqual(queue["stats"]["topic_counts"]["MCP"], 1)
        self.assertEqual(queue["stats"]["topic_counts"]["Tool Use"], 1)

    def test_build_review_plan_filters_topics_and_keeps_type_preferences(self) -> None:
        today = date.today()
        profile = {
            "weak_points": [
                weak_point("memory boundary", next_review=today.isoformat(), topic="Memory", last_seen=(today - timedelta(days=2)).isoformat()),
                weak_point("tool scenario", next_review=today.isoformat(), topic="Tool Use", last_seen=(today - timedelta(days=1)).isoformat()),
                weak_point("mcp excluded", next_review=today.isoformat(), topic="MCP"),
            ]
        }

        plan = build_review_plan(
            profile,
            topics=["Memory", "Tool Use"],
            question_types=["boundary", "scenario"],
            limit=4,
            allow_cross_topic=True,
        )

        self.assertEqual([card["topic"] for card in plan["cards"]], ["Memory", "Tool Use"])
        self.assertEqual([card["question_type"] for card in plan["cards"]], ["auto", "auto"])
        self.assertEqual([card["allowed_question_types"] for card in plan["cards"]], [["boundary", "scenario"], ["boundary", "scenario"]])
        self.assertEqual(plan["cards"][0]["candidate_related_topics"], ["Tool Use"])
        self.assertEqual(plan["cards"][1]["candidate_related_topics"], ["Memory"])
        self.assertEqual(plan["cards"][0]["related_topics"], [])
        mcp_topic = next(item for item in plan["available_topics"] if item["topic"] == "MCP")
        self.assertEqual(mcp_topic["candidate_count"], 1)

    def test_get_due_reviews_filters_multiple_topics(self) -> None:
        today = date.today()
        profile = {
            "weak_points": [
                weak_point("memory boundary", next_review=today.isoformat(), topic="Memory"),
                weak_point("tool scenario", next_review=today.isoformat(), topic="Tool Use"),
                weak_point("mcp excluded", next_review=today.isoformat(), topic="MCP"),
            ]
        }
        ctx = ToolExecutionContext(working=WorkingMemory(), profile_store=FakeProfileStore(profile))

        result = get_due_reviews({"topics": ["Memory", "Tool Use"], "limit": 10}, ctx)

        self.assertTrue(result["available"])
        self.assertEqual(result["selected_topics"], ["Memory", "Tool Use"])
        self.assertEqual([card["topic"] for card in result["cards"]], ["Memory", "Tool Use"])

    def test_build_review_plan_can_disable_cross_topic(self) -> None:
        today = date.today()
        profile = {
            "weak_points": [
                weak_point("memory boundary", next_review=today.isoformat(), topic="Memory"),
                weak_point("tool scenario", next_review=today.isoformat(), topic="Tool Use"),
            ]
        }

        plan = build_review_plan(
            profile,
            topics=["Memory", "Tool Use"],
            question_types=["boundary", "scenario"],
            allow_cross_topic=False,
        )

        self.assertEqual([card["candidate_related_topics"] for card in plan["cards"]], [[], []])

    def test_build_due_review_overview_groups_soft_candidates(self) -> None:
        today = date.today()
        profile = {
            "weak_points": [
                weak_point("memory due", next_review=today.isoformat(), topic="Memory", last_reviewed=(today - timedelta(days=2)).isoformat()),
                weak_point("mcp recommended", next_review=(today - timedelta(days=1)).isoformat(), topic="MCP", last_reviewed=(today - timedelta(days=3)).isoformat()),
                weak_point("mcp never", next_review=today.isoformat(), topic="MCP"),
                weak_point("future", next_review=(today + timedelta(days=2)).isoformat(), topic="Tool Use", last_reviewed=today.isoformat()),
            ]
        }

        overview = build_due_review_overview(profile, today=today.isoformat())

        self.assertEqual(overview["due_count"], 4)
        self.assertEqual(overview["candidate_count"], 4)
        self.assertEqual(overview["overdue_count"], 0)
        mcp = next(item for item in overview["topics"] if item["topic"] == "MCP")
        self.assertEqual(mcp["candidate_count"], 2)
        self.assertEqual(mcp["never_reviewed_count"], 1)
        self.assertEqual(overview["recent_count"], 1)
        self.assertNotIn("到期", overview["summary"])

    def test_due_review_overview_does_not_attach_cross_topic_strategy_constraints(self) -> None:
        today = date.today()
        profile = {
            "weak_points": [
                weak_point("memory knowledge", next_review=today.isoformat(), topic="Memory"),
                weak_point(
                    "answer too short",
                    next_review=(today - timedelta(days=2)).isoformat(),
                    topic="Interview",
                    category="answer_structure",
                    ease_factor=2.4,
                    times_seen=2,
                ),
                weak_point(
                    "missing tradeoff",
                    next_review=today.isoformat(),
                    topic="Interview",
                    category="thinking_pattern",
                    ease_factor=1.8,
                    times_seen=4,
                ),
                weak_point(
                    "third strategy ignored by max",
                    next_review=today.isoformat(),
                    topic="Interview",
                    category="communication",
                    ease_factor=1.6,
                    times_seen=5,
                ),
            ]
        }

        overview = build_due_review_overview(profile, today=today.isoformat(), max_strategy_constraints=2)

        self.assertEqual(overview["due_count"], 1)
        self.assertEqual([card["point"] for card in overview["cards"]], ["memory knowledge"])
        self.assertEqual(overview["strategy_due_count"], 3)
        strategy_points = [item["point"] for item in overview["cards"][0]["strategy_constraints"]]
        self.assertEqual(strategy_points, [])

    def test_due_review_overview_strategy_only_returns_empty_cards(self) -> None:
        today = date.today()
        profile = {
            "weak_points": [
                weak_point("answer too short", next_review=today.isoformat(), category="answer_structure"),
                weak_point("missing tradeoff", next_review=today.isoformat(), category="thinking_pattern"),
            ]
        }

        overview = build_due_review_overview(profile, today=today.isoformat())

        self.assertEqual(overview["cards"], [])
        self.assertEqual(overview["due_count"], 0)
        self.assertEqual(overview["candidate_count"], 2)
        self.assertEqual(overview["strategy_due_count"], 2)
        self.assertIn("建议", overview["summary"])

    def test_select_strategy_constraints_prefers_soft_review_priority(self) -> None:
        today = date.today()
        profile = {
            "weak_points": [
                weak_point("future recent", next_review=(today + timedelta(days=1)).isoformat(), category="answer_structure", last_reviewed=today.isoformat()),
                weak_point("due higher ease", next_review=today.isoformat(), category="answer_structure", ease_factor=2.5, times_seen=5, last_reviewed=(today - timedelta(days=1)).isoformat()),
                weak_point("due lower ease", next_review=today.isoformat(), category="thinking_pattern", ease_factor=1.7, times_seen=1, last_reviewed=(today - timedelta(days=2)).isoformat()),
                weak_point("recommended older wins", next_review=(today - timedelta(days=1)).isoformat(), category="communication", ease_factor=2.8, last_reviewed=(today - timedelta(days=3)).isoformat()),
            ]
        }

        constraints = select_strategy_constraints(profile, today=today.isoformat(), max_items=3)

        self.assertEqual([item["point"] for item in constraints], ["recommended older wins", "due lower ease", "due higher ease"])

    def test_strategy_constraints_only_attach_to_same_topic_cards(self) -> None:
        today = date.today().isoformat()
        profile = {
            "weak_points": [
                weak_point("agent mechanism gap", next_review=today, topic="Agent", category="knowledge_gap"),
                weak_point("mcp decision pattern", next_review=today, topic="MCP", category="thinking_pattern"),
                weak_point("agent answer too short", next_review=today, topic="Agent", category="answer_structure"),
            ]
        }

        grouped = grouped_review_cards(profile, topics=["Agent"], today=today)

        card = next(item for item in grouped["cards"] if item["category"] == "knowledge_gap")
        strategy_points = [item["point"] for item in card["strategy_constraints"]]
        self.assertEqual(strategy_points, ["agent answer too short"])
        self.assertNotIn("mcp decision pattern", strategy_points)

    def test_matching_strategy_constraints_prefers_same_layer_when_available(self) -> None:
        card = {"topic": "Agent", "planned_layer": "planning"}
        constraints = [
            {"point": "same topic fallback", "topic": "Agent", "planned_layer": "memory"},
            {"point": "same layer", "topic": "Agent", "planned_layer": "planning"},
            {"point": "cross topic", "topic": "MCP", "planned_layer": "planning"},
        ]

        matched = matching_strategy_constraints_for_card(card, constraints)

        self.assertEqual([item["point"] for item in matched], ["same layer"])

    def test_weak_point_content_hash_changes_when_stable_content_changes(self) -> None:
        today = date.today().isoformat()
        first = weak_point("content hash", next_review=today)
        second = weak_point("content hash", next_review=today)
        self.assertEqual(weak_point_content_hash(first), weak_point_content_hash(second))

        second["evidence"] = "new evidence"
        self.assertNotEqual(weak_point_content_hash(first), weak_point_content_hash(second))

    def test_grouped_review_cards_merge_same_topic_layer_category(self) -> None:
        today = date.today().isoformat()
        profile = {
            "weak_points": [
                weak_point("first gap", next_review=today, topic="MCP", category="knowledge_gap"),
                weak_point("second gap", next_review=today, topic="MCP", category="knowledge_gap"),
                weak_point("structure gap", next_review=today, topic="MCP", category="answer_structure"),
            ]
        }

        grouped = grouped_review_cards(profile, topics=["MCP"], today=today)

        counts = sorted(card["weak_point_count"] for card in grouped["cards"])
        self.assertEqual(counts, [1, 2])
        merged = [card for card in grouped["cards"] if card["weak_point_count"] == 2][0]
        self.assertEqual(merged["topic"], "MCP")
        self.assertEqual(merged["category"], "knowledge_gap")

    def test_review_card_cache_round_trips_success_payload(self) -> None:
        cache_dir = make_writable_test_dir()
        today = date.today().isoformat()
        weak_points = [weak_point("cached gap", next_review=today)]
        cache_key = review_card_cache_key(weak_points)
        self.assertIsNone(read_review_card_cache(cache_dir, cache_key))

        write_review_card_cache(cache_dir, cache_key, {"review_prompt": {"prompt": "cached prompt"}})

        self.assertEqual(read_review_card_cache(cache_dir, cache_key)["review_prompt"]["prompt"], "cached prompt")

    def test_grouped_review_prompt_and_verification_payload(self) -> None:
        today = date.today().isoformat()
        weak_points = [
            weak_point("first gap", next_review=today, topic="RAG"),
            weak_point("second gap", next_review=today, topic="RAG"),
        ]
        prompt = build_grouped_review_prompt(weak_points)

        self.assertIn("question_blocks", prompt)
        self.assertEqual(set(prompt["question_blocks"][0]["weak_point_ids"]), {weak_point_id(item) for item in weak_points})

        payload = parse_grouped_verification_payload(
            '{"overall":"ok","correct":["a"],"missed":["b"],"example":"e","weak_results":['
            f'{{"weak_point_id":"{weak_point_id(weak_points[0])}","suggested_action":"improve","reason":"good"}},'
            f'{{"weak_point_id":"{weak_point_id(weak_points[1])}","suggested_action":"retry","reason":"miss"}}'
            "]}",
            weak_points=weak_points,
        )

        self.assertEqual(payload["suggested_action"], "retry")
        self.assertEqual([item["suggested_action"] for item in payload["weak_results"]], ["improve", "retry"])

    def test_build_recall_prompt_falls_back_when_llm_fails(self) -> None:
        prompt = build_recall_prompt(
            weak_point("Explain working memory", next_review=date.today().isoformat()),
            allowed_question_types=["scenario"],
            related_topics=["Tool Use"],
            llm_client=FailingLlm(),
            model="fake-model",
        )

        self.assertTrue(prompt["fallback_used"])
        self.assertIn("Explain working memory", prompt["prompt"])
        self.assertEqual(prompt["question_type"], "scenario")
        self.assertEqual(prompt["related_topics"], ["Tool Use"])
        self.assertIn("error", prompt)

    def test_build_recall_prompt_lets_llm_choose_question_type(self) -> None:
        prompt = build_recall_prompt(
            weak_point("用户混淆任务分解和推理规划的职责边界", next_review=date.today().isoformat()),
            allowed_question_types=["recall", "boundary", "scenario"],
            related_topics=["Tool Use"],
            llm_client=ChoosingLlm(),
            model="fake-model",
        )

        self.assertFalse(prompt["fallback_used"])
        self.assertEqual(prompt["question_type"], "boundary")
        self.assertIn("任务分解", prompt["prompt"])
        self.assertEqual(prompt["expected_focus"], ["任务分解拆目标", "推理规划排执行"])

    def test_verification_query_includes_strategy_constraints(self) -> None:
        query = build_weak_point_verification_query(
            weak=weak_point("memory boundary missing", next_review=date.today().isoformat()),
            answer="working memory is current context",
            prompt="Explain the boundary between short-term memory and working memory.",
            strategy_constraints=[
                {"id": "s1", "point": "answer too short", "category": "answer_structure", "topic": "Interview", "evidence": "one sentence"},
            ],
        )

        self.assertIn("memory boundary missing", query)
        self.assertIn("Explain the boundary", query)
        self.assertIn("answer too short", query)
        self.assertIn("knowledge_correct", query)
        self.assertIn("strategy_feedback", query)

    def test_commit_review_outcome_updates_profile_for_pass_and_fail(self) -> None:
        today = date.today()
        tmp_path = make_writable_test_dir()
        try:
            store = InterviewProfileStore(tmp_path / "profile.json")
            first = weak_point("pass target", next_review=today.isoformat(), repetitions=0)
            second = weak_point("fail target", next_review=today.isoformat(), repetitions=2)
            store.save({"schema_version": 3, "weak_points": [first, second], "strong_points": []})

            pass_result = commit_review_outcome(store, card_id=weak_point_id(first), outcome="pass")
            fail_result = commit_review_outcome(store, card_id=weak_point_id(second), outcome="fail")
            profile = store.load()
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

        self.assertEqual(pass_result["outcome"], "pass")
        self.assertEqual(fail_result["outcome"], "fail")
        updated_first, updated_second = profile["weak_points"]
        self.assertEqual(updated_first["sr"]["last_outcome"], "pass")
        self.assertEqual(updated_first["sr"]["last_reviewed"], today.isoformat())
        self.assertGreaterEqual(updated_first["sr"]["repetitions"], 1)
        self.assertEqual(updated_second["sr"]["last_outcome"], "fail")
        self.assertEqual(updated_second["sr"]["last_reviewed"], today.isoformat())
        self.assertEqual(updated_second["sr"]["interval_days"], 1)
        self.assertLessEqual(updated_second["sr"]["repetitions"], 1)

    def test_commit_review_action_retry_keeps_ease_and_repetitions(self) -> None:
        today = date.today()
        tmp_path = make_writable_test_dir()
        try:
            store = InterviewProfileStore(tmp_path / "profile.json")
            target = weak_point("retry target", next_review=today.isoformat(), repetitions=3, ease_factor=2.4)
            store.save({"schema_version": 3, "weak_points": [target], "strong_points": []})

            result = commit_review_action(store, card_id=weak_point_id(target), action="retry")
            profile = store.load()
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

        updated = profile["weak_points"][0]
        self.assertEqual(result["action"], "retry")
        self.assertEqual(updated["sr"]["last_outcome"], "retry")
        self.assertEqual(updated["sr"]["ease_factor"], 2.4)
        self.assertEqual(updated["sr"]["repetitions"], 3)
        self.assertEqual(updated["sr"]["interval_days"], 1)
        self.assertEqual(updated["sr"]["last_reviewed"], today.isoformat())

    def test_parse_correction_payload_falls_back_on_non_json(self) -> None:
        payload = parse_correction_payload("Missing retrieval and injection details.")

        self.assertTrue(payload["parse_error"])
        self.assertEqual(payload["suggested_outcome"], "fail")
        self.assertEqual(payload["corrections"], ["Missing retrieval and injection details."])

    def test_parse_verification_payload_maps_improve_and_retry(self) -> None:
        payload = parse_verification_payload(
            '{"correct":["tools/list 是发现阶段"],"missed":["tools/call 响应格式"],'
            '"example":"Server 返回 result.content 数组。","suggested_action":"improve"}'
        )

        self.assertEqual(payload["suggested_action"], "improve")
        self.assertEqual(payload["correct"], ["tools/list 是发现阶段"])
        self.assertEqual(payload["missed"], ["tools/call 响应格式"])

        fallback = parse_verification_payload("Need mention result.content.")
        self.assertTrue(fallback["parse_error"])
        self.assertEqual(fallback["suggested_action"], "retry")


if __name__ == "__main__":
    unittest.main()
