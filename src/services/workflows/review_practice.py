from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

from knowledge_base_agent.llm.schema import LLMMessage, LLMRequest
from services.workflows.interview_profile import (
    InterviewProfileStore,
    normalize_category,
    profile_weak_point_for_agent,
)


QUESTION_TYPES = ("recall", "boundary", "compare", "scenario", "followup")
AUTO_QUESTION_TYPE = "auto"
STRATEGY_CATEGORIES = {"answer_structure", "communication", "thinking_pattern"}
DEFAULT_MAX_STRATEGY_CONSTRAINTS = 2
REVIEW_PROMPT_VERSION = "review-prompt-v3"
REVIEW_GROUPING_STRATEGY_VERSION = "topic-layer-category-v1"

QUESTION_TYPE_LABELS = {
    "recall": "\u4e3b\u52a8\u56de\u5fc6",
    "boundary": "\u804c\u8d23\u8fb9\u754c",
    "compare": "\u6982\u5ff5\u5bf9\u6bd4",
    "scenario": "\u573a\u666f\u8bbe\u8ba1",
    "followup": "\u9762\u8bd5\u8ffd\u95ee",
}

UNCATEGORIZED = "\u672a\u5206\u7c7b"

RECALL_PROMPT_SYSTEM = """You create one active-recall review question for a Chinese technical interview learner.

Write in Simplified Chinese. Use only the weak point payload and review options.
Do not answer the question.

Question type meanings:
- recall: ask for definition, flow, classification, or core mechanism from memory.
- boundary: ask for responsibilities, ownership, lifecycle, or module boundary.
- compare: ask the learner to compare two close concepts or tradeoffs.
- scenario: give a practical engineering scenario and ask for a design.
- followup: ask like an interviewer, pushing the weak point one layer deeper.

Choose exactly one question type based on the weak point.
Do not combine multiple question types in one question.
Ask one primary question only.
If the weak point contains many concepts, choose the smallest useful focus.
For recall questions, ask one concept only.
For boundary/compare questions, compare at most two concepts.
For scenario questions, use one concrete scenario and one decision point.

Return only JSON:
{
  "question_type": "recall|boundary|compare|scenario|followup",
  "reason": "why this question type fits this weak point",
  "prompt": "one concrete recall question",
  "hint": "one short hint, or empty string",
  "expected_focus": ["one expected answer point"]
}
"""


def build_review_plan(
    profile: dict[str, Any],
    *,
    topics: list[str] | tuple[str, ...] | None = None,
    question_types: list[str] | tuple[str, ...] | None = None,
    limit: int = 12,
    early_limit: int = 8,
    allow_cross_topic: bool = True,
    today: str | None = None,
) -> dict[str, Any]:
    today_value = today or date.today().isoformat()
    selected_topics = normalize_topics(topics)
    selected_question_types = normalize_question_types(question_types)
    available_topics = collect_review_topics(profile, today=today_value)

    due_weak = review_candidates(profile, topics=selected_topics, today=today_value)
    due_cards = assign_review_plan(
        [review_card_payload(weak, due=weak.get("review_state") != "recent") for weak in due_weak],
        selected_topics=selected_topics,
        question_types=selected_question_types,
        limit=limit,
        allow_cross_topic=allow_cross_topic,
    )

    early_cards: list[dict[str, Any]] = []

    topic_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    for card in due_cards:
        topic = str(card.get("topic") or UNCATEGORIZED)
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        qtype = str(card.get("question_type") or AUTO_QUESTION_TYPE)
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

    return {
        "today": today_value,
        "prompt_version": REVIEW_PROMPT_VERSION,
        "selected_topics": selected_topics,
        "selected_question_types": selected_question_types,
        "allow_cross_topic": allow_cross_topic,
        "available_topics": available_topics,
        "question_types": [{"value": value, "label": QUESTION_TYPE_LABELS[value]} for value in QUESTION_TYPES],
        "cards": due_cards,
        "early_candidates": early_cards,
        "stats": {
            "due_count": len(due_cards),
            "candidate_count": len(due_weak),
            "recommended_count": len([weak for weak in due_weak if weak.get("review_state") == "recommended"]),
            "never_reviewed_count": len([weak for weak in due_weak if weak.get("review_state") == "never_reviewed"]),
            "recent_count": len([weak for weak in due_weak if weak.get("review_state") == "recent"]),
            "overdue_count": 0,
            "early_count": 0,
            "topic_counts": topic_counts,
            "question_type_counts": type_counts,
        },
    }


def list_due_reviews(
    profile: dict[str, Any],
    *,
    topics: list[str] | tuple[str, ...] | None = None,
    question_types: list[str] | tuple[str, ...] | None = None,
    limit: int = 12,
    early_limit: int = 8,
    today: str | None = None,
) -> dict[str, Any]:
    return build_review_plan(
        profile,
        topics=topics,
        question_types=question_types,
        limit=limit,
        early_limit=early_limit,
        today=today,
    )


def build_due_review_overview(
    profile: dict[str, Any],
    *,
    limit: int = 50,
    today: str | None = None,
    max_strategy_constraints: int = DEFAULT_MAX_STRATEGY_CONSTRAINTS,
) -> dict[str, Any]:
    today_value = today or date.today().isoformat()
    candidates = review_candidates(profile, today=today_value)
    strategies = select_strategy_constraints(profile, max_items=None, today=today_value)
    cards = [
        attach_strategy_constraints(
            review_card_payload(weak, due=weak.get("review_state") != "recent"),
            strategies,
            max_items=max_strategy_constraints,
        )
        for weak in candidates
        if is_knowledge_weak_point(weak)
    ][: max(0, limit)]
    strategy_due_count = len([weak for weak in candidates if is_strategy_weak_point(weak)])
    topic_stats = review_topic_stats(candidates)
    for card in cards:
        card["review_state"] = card.get("review_state") or "recommended"
    summary = " · ".join(f"{item['topic']} {item['candidate_count']} 个可复习" for item in topic_stats[:3])
    if not cards and strategy_due_count:
        summary = f"当前只有 {strategy_due_count} 个回答策略弱项建议复查，需进入面试或知识卡中训练。"
    return {
        "today": today_value,
        "prompt_version": REVIEW_PROMPT_VERSION,
        "due_count": len(cards),
        "candidate_count": len(candidates),
        "recommended_count": len([weak for weak in candidates if weak.get("review_state") == "recommended"]),
        "never_reviewed_count": len([weak for weak in candidates if weak.get("review_state") == "never_reviewed"]),
        "recent_count": len([weak for weak in candidates if weak.get("review_state") == "recent"]),
        "overdue_count": 0,
        "strategy_due_count": strategy_due_count,
        "strategy_constraints": strategies[: max(0, max_strategy_constraints)],
        "topics": topic_stats,
        "cards": cards,
        "summary": summary,
    }


def review_card_payload(weak: dict[str, Any], *, due: bool, early: bool = False) -> dict[str, Any]:
    payload = profile_weak_point_for_agent(weak)
    payload.update(
        {
            "id": weak_point_id(weak),
            "main_weak_point": profile_weak_point_for_agent(weak),
            "due": due,
            "early": early,
            "source_note_paths": list(weak.get("source_note_paths") or []),
            "last_seen": weak.get("last_seen", ""),
            "last_reviewed": weak.get("last_reviewed", ""),
            "review_state": weak.get("review_state", ""),
            "review_priority": weak.get("review_priority", 0),
            "review_age_days": weak.get("review_age_days", 0),
            "strategy_constraints": [],
        }
    )
    return payload


def is_knowledge_weak_point(weak: dict[str, Any]) -> bool:
    return normalize_category(weak.get("category")) == "knowledge_gap"


def is_strategy_weak_point(weak: dict[str, Any]) -> bool:
    return normalize_category(weak.get("category")) in STRATEGY_CATEGORIES


def strategy_constraint_payload(weak: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": weak_point_id(weak),
        "point": str(weak.get("point") or "").strip(),
        "category": normalize_category(weak.get("category")),
        "topic": str(weak.get("topic") or "").strip(),
        "planned_layer": str(weak.get("planned_layer") or "").strip(),
        "evidence": str(weak.get("evidence") or "").strip(),
    }


def select_strategy_constraints(
    profile: dict[str, Any],
    *,
    max_items: int | None = DEFAULT_MAX_STRATEGY_CONSTRAINTS,
    today: str | None = None,
) -> list[dict[str, Any]]:
    today_value = today or date.today().isoformat()
    candidates = [
        weak
        for weak in review_candidates(profile, today=today_value)
        if is_strategy_weak_point(weak)
    ]
    payloads = [strategy_constraint_payload(weak) for weak in candidates]
    if max_items is None:
        return payloads
    return payloads[: max(0, max_items)]


def attach_strategy_constraints(card: dict[str, Any], constraints: list[dict[str, Any]], *, max_items: int | None = None) -> dict[str, Any]:
    enriched = dict(card)
    enriched["strategy_constraints"] = matching_strategy_constraints_for_card(card, constraints, max_items=max_items)
    return enriched


def matching_strategy_constraints_for_card(
    card: dict[str, Any],
    constraints: list[dict[str, Any]],
    *,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    topic = str(card.get("topic") or "").strip()
    planned_layer = str(card.get("planned_layer") or "").strip()
    same_topic = [
        dict(item)
        for item in constraints
        if str(item.get("topic") or "").strip() == topic
    ]
    if not planned_layer:
        matched = same_topic
        return matched if max_items is None else matched[: max(0, max_items)]
    same_layer = [
        item
        for item in same_topic
        if str(item.get("planned_layer") or "").strip() == planned_layer
    ]
    matched = same_layer or same_topic
    return matched if max_items is None else matched[: max(0, max_items)]


def weak_point_id(weak: dict[str, Any]) -> str:
    for key in ("id", "weak_id", "uid"):
        value = str(weak.get(key) or "").strip()
        if value:
            return value
    seed = "|".join(
        [
            str(weak.get("topic") or ""),
            str(weak.get("planned_layer") or ""),
            str(weak.get("point") or ""),
        ]
    )
    return "weak-" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def weak_point_cache_content(weak: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": weak_point_id(weak),
        "point": str(weak.get("point") or "").strip(),
        "evidence": str(weak.get("evidence") or "").strip(),
        "category": normalize_category(weak.get("category")),
        "topic": str(weak.get("topic") or "").strip(),
        "planned_layer": str(weak.get("planned_layer") or "").strip(),
        "source_note_paths": sorted(str(path).strip() for path in weak.get("source_note_paths") or [] if str(path).strip()),
    }


def weak_point_content_hash(weak: dict[str, Any], *, prompt_version: str = REVIEW_PROMPT_VERSION) -> str:
    payload = {"prompt_version": prompt_version, "weak_point": weak_point_cache_content(weak)}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def review_prompt_cache_key(weak: dict[str, Any], *, prompt_version: str = REVIEW_PROMPT_VERSION) -> str:
    return weak_point_content_hash(weak, prompt_version=prompt_version)


def review_card_cache_key(
    weak_points: list[dict[str, Any]],
    *,
    grouping_strategy_version: str = REVIEW_GROUPING_STRATEGY_VERSION,
    prompt_version: str = REVIEW_PROMPT_VERSION,
) -> str:
    payload = {
        "grouping_strategy_version": grouping_strategy_version,
        "prompt_version": prompt_version,
        "weak_point_hashes": sorted(weak_point_content_hash(weak, prompt_version=prompt_version) for weak in weak_points),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def load_review_cache(cache_dir: Path | str, filename: str) -> dict[str, Any]:
    path = Path(cache_dir) / filename
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_review_cache(cache_dir: Path | str, filename: str, payload: dict[str, Any]) -> None:
    path = Path(cache_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def read_review_card_cache(cache_dir: Path | str, cache_key: str) -> dict[str, Any] | None:
    cache = load_review_cache(cache_dir, "card-cache.json")
    item = cache.get(cache_key)
    if not isinstance(item, dict) or item.get("error"):
        return None
    return dict(item)


def read_review_prompt_cache(cache_dir: Path | str, cache_key: str) -> dict[str, Any] | None:
    cache = load_review_cache(cache_dir, "prompt-cache.json")
    item = cache.get(cache_key)
    if not isinstance(item, dict) or item.get("error"):
        return None
    return dict(item)


def write_review_card_cache(cache_dir: Path | str, cache_key: str, payload: dict[str, Any]) -> None:
    if payload.get("error"):
        return
    cache = load_review_cache(cache_dir, "card-cache.json")
    cache[cache_key] = dict(payload)
    save_review_cache(cache_dir, "card-cache.json", cache)


def write_review_prompt_cache(cache_dir: Path | str, cache_key: str, payload: dict[str, Any]) -> None:
    if payload.get("error"):
        return
    cache = load_review_cache(cache_dir, "prompt-cache.json")
    cache[cache_key] = dict(payload)
    save_review_cache(cache_dir, "prompt-cache.json", cache)


def review_candidates(
    profile: dict[str, Any],
    *,
    topics: list[str] | tuple[str, ...] | None = None,
    today: str | None = None,
) -> list[dict[str, Any]]:
    today_value = today or date.today().isoformat()
    selected_topics = normalize_topics(topics)
    candidates = [
        enrich_review_candidate(weak, today=today_value)
        for weak in profile.get("weak_points", [])
        if isinstance(weak, dict) and not weak.get("improved") and topic_matches(weak, selected_topics)
    ]
    return sorted(candidates, key=review_candidate_sort_key)


def enrich_review_candidate(weak: dict[str, Any], *, today: str) -> dict[str, Any]:
    item = dict(weak)
    sr = dict(item.get("sr") or {})
    item["sr"] = sr
    last_reviewed = str(sr.get("last_reviewed") or "").strip()
    next_review = str(sr.get("next_review") or "2000-01-01").strip()
    if not last_reviewed:
        review_state = "never_reviewed"
    elif next_review <= today:
        review_state = "recommended"
    else:
        review_state = "recent"
    review_age_days = days_since(last_reviewed or str(item.get("last_seen") or ""), today=today)
    state_bonus = {"never_reviewed": 3.0, "recommended": 2.0, "recent": 1.0}[review_state]
    ease = float(sr.get("ease_factor", 2.5) or 2.5)
    times_seen = int(item.get("times_seen", 0) or 0)
    item.update(
        {
            "review_state": review_state,
            "review_priority": round(state_bonus + min(review_age_days, 365) / 365 + max(0.0, 3.0 - ease) / 10 + min(times_seen, 20) / 100, 4),
            "last_reviewed": last_reviewed,
            "review_age_days": review_age_days,
        }
    )
    return item


def review_candidate_sort_key(item: dict[str, Any]) -> tuple[int, str, float, int, str]:
    state_rank = {"never_reviewed": 0, "recommended": 1, "recent": 2}.get(str(item.get("review_state") or ""), 3)
    sr = item.get("sr") or {}
    last_reviewed = str(sr.get("last_reviewed") or item.get("last_seen") or "0000-00-00")
    return (
        state_rank,
        last_reviewed,
        float(sr.get("ease_factor", 2.5) or 2.5),
        -int(item.get("times_seen", 0) or 0),
        weak_point_id(item),
    )


def review_topic_stats(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for weak in candidates:
        topic = str(weak.get("topic") or UNCATEGORIZED).strip() or UNCATEGORIZED
        item = stats.setdefault(
            topic,
            {
                "topic": topic,
                "due": 0,
                "candidate_count": 0,
                "recommended_count": 0,
                "never_reviewed_count": 0,
                "recent_count": 0,
                "max_review_priority": 0.0,
            },
        )
        item["candidate_count"] += 1
        item["due"] += 1
        state = str(weak.get("review_state") or "")
        if state == "recommended":
            item["recommended_count"] += 1
        elif state == "never_reviewed":
            item["never_reviewed_count"] += 1
        elif state == "recent":
            item["recent_count"] += 1
        item["max_review_priority"] = max(float(item["max_review_priority"]), float(weak.get("review_priority", 0.0) or 0.0))
    return sorted(
        stats.values(),
        key=lambda item: (
            -float(item.get("max_review_priority") or 0.0),
            -int(item.get("never_reviewed_count") or 0),
            -int(item.get("recommended_count") or 0),
            str(item.get("topic") or ""),
        ),
    )


def days_since(value: str, *, today: str) -> int:
    try:
        current = date.fromisoformat(today)
        previous = date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return 9999
    return max(0, (current - previous).days)


def grouped_review_cards(
    profile: dict[str, Any],
    *,
    topics: list[str] | tuple[str, ...] | None = None,
    limit: int = 12,
    today: str | None = None,
    max_strategy_constraints: int = DEFAULT_MAX_STRATEGY_CONSTRAINTS,
) -> dict[str, Any]:
    today_value = today or date.today().isoformat()
    selected_topics = normalize_topics(topics)
    due_weak_points = review_candidates(profile, topics=selected_topics, today=today_value)
    strategies = select_strategy_constraints(profile, max_items=None, today=today_value)
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for weak in due_weak_points:
        key = (
            str(weak.get("topic") or UNCATEGORIZED).strip() or UNCATEGORIZED,
            str(weak.get("planned_layer") or "").strip(),
            normalize_category(weak.get("category")),
        )
        buckets.setdefault(key, []).append(weak)

    cards: list[dict[str, Any]] = []
    for key in sorted(buckets, key=lambda item: (item[0], item[1], item[2])):
        weak_group = sorted(buckets[key], key=lambda weak: weak_point_id(weak))
        card_shell = build_grouped_review_card_payload(weak_group, strategies=[])
        card_shell["strategy_constraints"] = matching_strategy_constraints_for_card(
            card_shell,
            strategies,
            max_items=max_strategy_constraints,
        )
        card_shell["strategy_tips"] = [
            str(item.get("point") or "").strip()
            for item in card_shell["strategy_constraints"]
            if str(item.get("point") or "").strip()
        ]
        cards.append(card_shell)
        if len(cards) >= max(0, limit):
            break

    topics_payload = review_topic_stats(due_weak_points)
    summary = "；".join(f"{item['topic']} {item['candidate_count']} 个可复习" for item in topics_payload[:3])
    return {
        "today": today_value,
        "prompt_version": REVIEW_PROMPT_VERSION,
        "selected_topics": selected_topics,
        "due_count": sum(len(card.get("weak_point_ids") or []) for card in cards),
        "candidate_count": len(due_weak_points),
        "recommended_count": len([weak for weak in due_weak_points if weak.get("review_state") == "recommended"]),
        "never_reviewed_count": len([weak for weak in due_weak_points if weak.get("review_state") == "never_reviewed"]),
        "recent_count": len([weak for weak in due_weak_points if weak.get("review_state") == "recent"]),
        "card_count": len(cards),
        "overdue_count": 0,
        "strategy_constraints": strategies[: max(0, max_strategy_constraints)],
        "topics": topics_payload,
        "cards": cards,
        "summary": summary,
    }


def build_grouped_review_card_payload(weak_points: list[dict[str, Any]], *, strategies: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if not weak_points:
        raise ValueError("weak_points is required")
    first = weak_points[0]
    topic = str(first.get("topic") or UNCATEGORIZED).strip() or UNCATEGORIZED
    planned_layer = str(first.get("planned_layer") or "").strip()
    category = normalize_category(first.get("category"))
    cache_key = review_card_cache_key(weak_points)
    weak_ids = [weak_point_id(weak) for weak in weak_points]
    source_note_paths = sorted(
        {
            str(path).strip()
            for weak in weak_points
            for path in weak.get("source_note_paths") or []
            if str(path).strip()
        }
    )
    weak_summary = []
    for weak in weak_points:
        summary = profile_weak_point_for_agent(weak)
        summary.update(
            {
                "id": weak_point_id(weak),
                "sr": dict(weak.get("sr") or {}),
                "source_note_paths": list(weak.get("source_note_paths") or []),
                "review_state": weak.get("review_state", ""),
                "review_priority": weak.get("review_priority", 0),
                "last_reviewed": weak.get("last_reviewed", ""),
                "review_age_days": weak.get("review_age_days", 0),
            }
        )
        weak_summary.append(summary)
    return {
        "id": "card-" + cache_key[:16],
        "card_id": "card-" + cache_key[:16],
        "cache_key": cache_key,
        "topic": topic,
        "planned_layer": planned_layer,
        "category": category,
        "weak_point_ids": weak_ids,
        "weak_points_summary": weak_summary,
        "weak_point_count": len(weak_ids),
        "review_state": str(first.get("review_state") or ""),
        "review_priority": max(float(weak.get("review_priority", 0) or 0) for weak in weak_points),
        "point": "；".join(str(weak.get("point") or "").strip() for weak in weak_points if str(weak.get("point") or "").strip()),
        "evidence": "\n".join(str(weak.get("evidence") or "").strip() for weak in weak_points if str(weak.get("evidence") or "").strip()),
        "source_note_paths": source_note_paths,
        "strategy_constraints": [dict(item) for item in strategies or []],
        "strategy_tips": [str(item.get("point") or "").strip() for item in strategies or [] if str(item.get("point") or "").strip()],
        "status": "pending",
    }


def collect_review_topics(profile: dict[str, Any], *, today: str | None = None) -> list[dict[str, Any]]:
    today_value = today or date.today().isoformat()
    topics = review_topic_stats(review_candidates(profile, today=today_value))
    return [
        {
            "topic": item["topic"],
            "total": item["candidate_count"],
            "due": item["candidate_count"],
            "candidate_count": item["candidate_count"],
            "recommended_count": item["recommended_count"],
            "never_reviewed_count": item["never_reviewed_count"],
            "recent_count": item["recent_count"],
        }
        for item in topics
    ]


def assign_review_plan(
    cards: list[dict[str, Any]],
    *,
    selected_topics: list[str],
    question_types: list[str],
    limit: int,
    allow_cross_topic: bool,
) -> list[dict[str, Any]]:
    ordered = list(cards)
    planned: list[dict[str, Any]] = []
    topic_pool = selected_topics or sorted({str(card.get("topic") or "") for card in ordered if str(card.get("topic") or "").strip()})
    for index, card in enumerate(ordered[: max(0, limit)]):
        planned_card = dict(card)
        planned_card["allowed_question_types"] = list(question_types)
        planned_card["question_type"] = AUTO_QUESTION_TYPE
        planned_card["question_type_label"] = "\u81ea\u52a8\u9009\u62e9"
        planned_card["candidate_related_topics"] = related_topics_for_card(
            planned_card,
            topic_pool,
            index=index,
            allow_cross_topic=allow_cross_topic,
        )
        planned_card["related_topics"] = []
        planned.append(planned_card)
    return planned


def interleave_by_topic(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        topic = str(card.get("topic") or UNCATEGORIZED)
        buckets.setdefault(topic, []).append(card)
    ordered: list[dict[str, Any]] = []
    topics = sorted(buckets)
    while any(buckets.values()):
        for topic in topics:
            if buckets[topic]:
                ordered.append(buckets[topic].pop(0))
    return ordered


def related_topics_for_card(card: dict[str, Any], topics: list[str], *, index: int, allow_cross_topic: bool) -> list[str]:
    if not allow_cross_topic:
        return []
    current = str(card.get("topic") or "").strip()
    others = [topic for topic in topics if topic and topic != current]
    if not others:
        return []
    return [others[index % len(others)]]


def normalize_topics(topics: list[str] | tuple[str, ...] | None) -> list[str]:
    result: list[str] = []
    for item in topics or []:
        for part in str(item or "").split(","):
            value = part.strip()
            if value and value not in result:
                result.append(value)
    return result


def normalize_question_types(question_types: list[str] | tuple[str, ...] | None) -> list[str]:
    result: list[str] = []
    for item in question_types or []:
        for part in str(item or "").split(","):
            value = part.strip()
            if value in QUESTION_TYPES and value not in result:
                result.append(value)
    return result or ["recall", "boundary", "scenario", "followup"]


def filter_by_topics(items: list[dict[str, Any]], selected_topics: list[str]) -> list[dict[str, Any]]:
    return [item for item in items if topic_matches(item, selected_topics)]


def topic_matches(weak: dict[str, Any], selected_topics: list[str]) -> bool:
    if not selected_topics:
        return True
    topic = str(weak.get("topic") or "").strip()
    return topic in selected_topics


def early_sort_key(weak: dict[str, Any]) -> tuple[float, str, str]:
    sr = weak.get("sr") or {}
    return (
        float(sr.get("ease_factor", 2.5) or 2.5),
        str(sr.get("next_review") or "9999-12-31"),
        str(weak.get("last_seen") or "0000-00-00"),
    )


def find_weak_point(profile: dict[str, Any], card_id: str) -> dict[str, Any] | None:
    target = str(card_id or "").strip()
    for weak in profile.get("weak_points", []):
        if isinstance(weak, dict) and weak_point_id(weak) == target:
            return weak
    return None


def build_static_recall_prompt(
    weak: dict[str, Any],
    *,
    question_type: str = AUTO_QUESTION_TYPE,
    related_topics: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    point = str(weak.get("point") or "\u8fd9\u4e2a\u8584\u5f31\u70b9").strip()
    topic = str(weak.get("topic") or "").strip()
    related = [str(item).strip() for item in related_topics or [] if str(item).strip()]
    qtype = question_type if question_type in QUESTION_TYPES else choose_question_type_for_weak_point(weak)
    title = topic or "\u8fd9\u4e2a\u4e3b\u9898"
    if qtype == "boundary":
        prompt = f"\u56f4\u7ed5\u300c{title}\u300d\uff0c\u8bf7\u8bf4\u660e\u8fd9\u4e2a\u8584\u5f31\u70b9\u6d89\u53ca\u7684\u804c\u8d23\u8fb9\u754c\uff1a{point}"
    elif qtype == "compare":
        target = f"\u5e76\u7ed3\u5408\u300c{related[0]}\u300d\u505a\u5bf9\u6bd4" if related else "\u5e76\u548c\u5bb9\u6613\u6df7\u6dc6\u7684\u76f8\u8fd1\u6982\u5ff5\u505a\u5bf9\u6bd4"
        prompt = f"\u8bf7\u89e3\u91ca\u300c{title}\u300d\u4e2d\u7684\u8584\u5f31\u70b9\uff1a{point}\uff0c{target}\u3002"
    elif qtype == "scenario":
        target = f"\uff0c\u540c\u65f6\u8003\u8651\u300c{related[0]}\u300d" if related else ""
        prompt = f"\u7ed9\u4f60\u4e00\u4e2a\u5de5\u7a0b\u573a\u666f\uff1a\u9700\u8981\u8bbe\u8ba1\u548c\u300c{title}\u300d\u76f8\u5173\u7684\u65b9\u6848{target}\u3002\u8bf7\u4e0d\u7528\u770b\u7b14\u8bb0\uff0c\u8bf4\u660e\u4f60\u4f1a\u5982\u4f55\u5904\u7406\uff1a{point}"
    elif qtype == "followup":
        prompt = f"\u9762\u8bd5\u5b98\u7ee7\u7eed\u8ffd\u95ee\uff1a\u4f60\u521a\u624d\u63d0\u5230\u300c{title}\u300d\uff0c\u8bf7\u628a\u8fd9\u4e2a\u70b9\u8bb2\u5177\u4f53\u4e00\u5c42\uff1a{point}"
    else:
        prompt = f"\u56f4\u7ed5\u300c{topic or '\u8fd9\u4e2a\u77e5\u8bc6\u70b9'}\u300d\uff0c\u8bf7\u5148\u4e0d\u770b\u7b14\u8bb0\uff0c\u7528\u81ea\u5df1\u7684\u8bdd\u56de\u7b54\uff1a{point}"
    return {
        "prompt": prompt,
        "hint": str(weak.get("planned_layer") or "").strip(),
        "question_type": qtype,
        "question_type_label": QUESTION_TYPE_LABELS.get(qtype, qtype),
        "reason": "fallback rule selected this type from the weak point text",
        "expected_focus": [],
        "related_topics": related,
        "fallback_used": True,
    }


def build_recall_prompt(
    weak: dict[str, Any],
    *,
    question_type: str = AUTO_QUESTION_TYPE,
    allowed_question_types: list[str] | tuple[str, ...] | None = None,
    related_topics: list[str] | tuple[str, ...] | None = None,
    llm_client: Any | None = None,
    model: str = "",
    temperature: float = 0.2,
) -> dict[str, Any]:
    preferred_type = question_type if question_type in QUESTION_TYPES else AUTO_QUESTION_TYPE
    allowed_types = normalize_question_types(allowed_question_types) if allowed_question_types else list(QUESTION_TYPES)
    related = [str(item).strip() for item in related_topics or [] if str(item).strip()]
    fallback = build_static_recall_prompt(weak, question_type=preferred_type, related_topics=related)
    if fallback["question_type"] not in allowed_types:
        fallback = build_static_recall_prompt(weak, question_type=allowed_types[0], related_topics=related)
    if llm_client is None or not model:
        return fallback
    try:
        payload = {
            "weak_point": profile_weak_point_for_agent(weak),
            "preferred_question_type": preferred_type,
            "allowed_question_types": [
                {"value": value, "label": QUESTION_TYPE_LABELS.get(value, value)}
                for value in allowed_types
            ],
            "candidate_related_topics": related,
            "selection_rules": {
                "choose_exactly_one_type": True,
                "one_primary_question_only": True,
                "smallest_useful_focus": True,
                "max_compare_targets": 2,
            },
        }
        response = llm_client.complete(
            LLMRequest(
                model=model,
                temperature=temperature,
                messages=[
                    LLMMessage(role="system", content=RECALL_PROMPT_SYSTEM),
                    LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False, indent=2)),
                ],
            )
        )
        parsed = parse_json_object(response.content)
        selected_type = str(parsed.get("question_type") or "").strip()
        if selected_type not in allowed_types:
            selected_type = fallback["question_type"]
        prompt = str(parsed.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("empty prompt")
        return {
            "prompt": prompt,
            "hint": str(parsed.get("hint") or "").strip(),
            "question_type": selected_type,
            "question_type_label": QUESTION_TYPE_LABELS.get(selected_type, selected_type),
            "reason": str(parsed.get("reason") or "").strip(),
            "expected_focus": list_of_strings(parsed.get("expected_focus")),
            "related_topics": related if selected_type in {"boundary", "compare", "scenario"} else [],
            "fallback_used": False,
        }
    except Exception as exc:
        result = dict(fallback)
        result["error"] = str(exc)
        return result


def build_grouped_review_prompt(
    weak_points: list[dict[str, Any]],
    *,
    strategy_constraints: list[dict[str, Any]] | None = None,
    llm_client: Any | None = None,
    model: str = "",
    temperature: float = 0.2,
) -> dict[str, Any]:
    fallback = build_static_grouped_review_prompt(weak_points, strategy_constraints=strategy_constraints)
    if llm_client is None or not model:
        return fallback
    try:
        payload = {
            "weak_points": [profile_weak_point_for_agent(weak) | {"id": weak_point_id(weak)} for weak in weak_points],
            "strategy_constraints": list(strategy_constraints or []),
            "rules": {
                "write_in_simplified_chinese": True,
                "one_card_can_cover_multiple_weak_points": True,
                "question_blocks_must_test_only_weak_points": True,
                "strategy_constraints_are_expression_guidance_only": True,
                "do_not_create_question_blocks_from_strategy_constraints": True,
                "answer_structure_uses_evidence_reconstruction": True,
                "thinking_pattern_uses_new_scenario": True,
                "communication_reasks_for_clearer_expression": True,
            },
        }
        response = llm_client.complete(
            LLMRequest(
                model=model,
                temperature=temperature,
                messages=[
                    LLMMessage(
                        role="system",
                        content=(
                            "You generate one grouped review card for a Chinese interview learner. "
                            "Return only JSON with question_blocks, reference_answer, and strategy_tips. "
                            "Each question block must include type, prompt, and weak_point_ids. "
                            "Generate question_blocks only from weak_points. "
                            "Never create a separate question block from strategy_constraints; use them only as brief answer-quality tips."
                        ),
                    ),
                    LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False, indent=2)),
                ],
            )
        )
        parsed = parse_json_object(response.content)
        question_blocks = normalize_question_blocks(parsed.get("question_blocks"), weak_points)
        if not question_blocks:
            raise ValueError("empty question_blocks")
        return {
            "prompt": "\n\n".join(block["prompt"] for block in question_blocks),
            "question_blocks": question_blocks,
            "reference_answer": str(parsed.get("reference_answer") or "").strip() or fallback["reference_answer"],
            "strategy_tips": list_of_strings(parsed.get("strategy_tips")) or list(fallback.get("strategy_tips") or []),
            "fallback_used": False,
            "prompt_version": REVIEW_PROMPT_VERSION,
        }
    except Exception as exc:
        result = dict(fallback)
        result["error"] = str(exc)
        return result


def build_static_grouped_review_prompt(
    weak_points: list[dict[str, Any]],
    *,
    strategy_constraints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not weak_points:
        return {
            "prompt": "请用自己的话复述这个薄弱点。",
            "question_blocks": [],
            "reference_answer": "",
            "strategy_tips": [],
            "fallback_used": True,
            "prompt_version": REVIEW_PROMPT_VERSION,
        }
    category = normalize_category(weak_points[0].get("category"))
    topic = str(weak_points[0].get("topic") or UNCATEGORIZED).strip() or UNCATEGORIZED
    ids = [weak_point_id(weak) for weak in weak_points]
    points = [str(weak.get("point") or "").strip() for weak in weak_points if str(weak.get("point") or "").strip()]
    evidence = [str(weak.get("evidence") or "").strip() for weak in weak_points if str(weak.get("evidence") or "").strip()]
    joined_points = "；".join(points) or "这个薄弱点"
    if category == "answer_structure":
        prompt = f"围绕「{topic}」，请根据上次暴露出的回答问题，重新组织一版更完整的回答。重点修复：{joined_points}"
        if evidence:
            prompt += f"\n上次证据摘要：{evidence[0][:500]}"
        block_type = "answer_structure"
    elif category == "thinking_pattern":
        prompt = f"换一个新的工程场景来考察同一个思维模式：如果你在「{topic}」相关项目中遇到相似取舍，请说明你会如何判断。需要覆盖：{joined_points}"
        block_type = "thinking_pattern"
    elif category == "communication":
        prompt = f"请重新回答一道相似面试题，目标是表达更清晰、有层次。主题是「{topic}」，需要修复：{joined_points}"
        block_type = "communication"
    else:
        prompt = f"围绕「{topic}」，请用自己的话解释这些薄弱点，并说明定义、边界、触发时机和工程例子：{joined_points}"
        block_type = "knowledge_gap"
    tips = [str(item.get("point") or "").strip() for item in strategy_constraints or [] if str(item.get("point") or "").strip()]
    return {
        "prompt": prompt,
        "question_blocks": [{"type": block_type, "prompt": prompt, "weak_point_ids": ids}],
        "reference_answer": "参考答案应覆盖薄弱点原文、证据中暴露的遗漏，以及一个可迁移到面试表达的结构化示例。",
        "strategy_tips": tips[:3],
        "fallback_used": True,
        "prompt_version": REVIEW_PROMPT_VERSION,
    }


def normalize_question_blocks(value: Any, weak_points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed_ids = {weak_point_id(weak) for weak in weak_points}
    result: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return result
    for item in value:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            continue
        ids = [str(weak_id).strip() for weak_id in item.get("weak_point_ids") or [] if str(weak_id).strip() in allowed_ids]
        result.append(
            {
                "type": str(item.get("type") or "knowledge_gap").strip() or "knowledge_gap",
                "prompt": prompt,
                "weak_point_ids": ids or sorted(allowed_ids),
            }
        )
    return result


def choose_question_type_for_weak_point(weak: dict[str, Any]) -> str:
    text = " ".join(
        str(weak.get(key) or "")
        for key in ("point", "evidence", "planned_layer", "topic")
    ).lower()
    if any(marker in text for marker in ("boundary", "边界", "职责", "ownership", "lifecycle", "混淆")):
        return "boundary"
    if any(marker in text for marker in ("compare", "对比", "区别", "tradeoff", "取舍", " vs ")):
        return "compare"
    if any(marker in text for marker in ("scenario", "场景", "设计", "工程", "生产", "方案")):
        return "scenario"
    if any(marker in text for marker in ("followup", "追问", "深入", "具体一层")):
        return "followup"
    return "recall"


def grade_weak_point(
    store: InterviewProfileStore,
    profile: dict[str, Any],
    weak: dict[str, Any],
    *,
    outcome: str,
    today: str | None = None,
) -> dict[str, Any]:
    normalized = str(outcome or "").strip().lower()
    today_value = today or date.today().isoformat()
    belief_id = weak_point_id(weak)
    before = json.loads(json.dumps(weak.get("sr") or {}, ensure_ascii=False))
    previous_last_seen = weak.get("last_seen", "")
    if normalized == "pass":
        from services.memory.bridge import observation_schedule_pass

        observation = observation_schedule_pass(belief_id=belief_id)
    elif normalized == "fail":
        from services.memory.bridge import observation_schedule_retry

        observation = observation_schedule_retry(belief_id=belief_id, evidence_summary="review practice fail")
    else:
        raise ValueError("outcome must be pass or fail")

    from services.memory.commit import commit_observations

    model, _ = commit_observations(store.load_v4(), [observation], today=today_value)
    store.save_v4(model)
    updated_profile = store.load()
    updated_weak = find_weak_point(updated_profile, belief_id)
    if updated_weak is None:
        updated_weak = weak
    updated_weak["last_seen"] = previous_last_seen
    updated_weak.setdefault("sr", {})["last_reviewed"] = today_value
    after = updated_weak.get("sr") or {}
    return {
        "card_id": belief_id,
        "outcome": normalized,
        "before": before,
        "after": after,
        "improved": bool(updated_weak.get("improved")),
    }


def commit_review_outcome(store: InterviewProfileStore, *, card_id: str, outcome: str) -> dict[str, Any]:
    profile = store.load()
    weak = find_weak_point(profile, card_id)
    if weak is None:
        raise KeyError(f"review card not found: {card_id}")
    return grade_weak_point(store, profile, weak, outcome=outcome)


def commit_review_action(store: InterviewProfileStore, *, card_id: str, action: str) -> dict[str, Any]:
    normalized = str(action or "").strip().lower()
    if normalized == "improve":
        return commit_review_outcome(store, card_id=card_id, outcome="pass")
    if normalized != "retry":
        raise ValueError("action must be improve or retry")
    profile = store.load()
    weak = find_weak_point(profile, card_id)
    if weak is None:
        raise KeyError(f"review card not found: {card_id}")
    today_value = date.today().isoformat()
    belief_id = weak_point_id(weak)
    before = json.loads(json.dumps(weak.get("sr") or {}, ensure_ascii=False))
    from services.memory.bridge import apply_review_ui_retry

    model = store.load_v4()
    updated = apply_review_ui_retry(model, belief_id=belief_id, today=today_value)
    if updated is None:
        raise KeyError(f"review card not found: {card_id}")
    store.save_v4(model)
    profile = store.load()
    weak = find_weak_point(profile, card_id) or weak
    sr = weak.get("sr") or {}
    return {
        "card_id": card_id,
        "action": "retry",
        "before": before,
        "after": sr,
        "improved": False,
    }


def commit_review_suggestions_as_candidates(
    store: InterviewProfileStore,
    *,
    suggested_commits: list[dict[str, Any]],
    review_run_id: str,
    messages: list[dict[str, Any]] | None = None,
    today: str | None = None,
) -> dict[str, Any]:
    if not suggested_commits:
        return {"changed": False, "observations": 0, "added_candidate_ids": [], "warnings": []}

    profile = store.load()
    model_before = store.load_v4()
    before_ids = {str(item.get("id") or "") for item in model_before.get("beliefs") or [] if isinstance(item, dict)}
    warnings: list[dict[str, str]] = []
    observations: list[dict[str, Any]] = []
    latest_turn_id = _latest_dialogue_turn_id(messages or [])

    for suggestion in suggested_commits:
        if not isinstance(suggestion, dict):
            continue
        weak_id = str(suggestion.get("weak_point_id") or "").strip()
        if not weak_id:
            continue
        weak = find_weak_point(profile, weak_id)
        if weak is None:
            warnings.append({"weak_point_id": weak_id, "warning": "review weak point not found"})
            continue
        evidence_summary = _review_candidate_evidence_summary(suggestion, weak, messages or [])
        observations.append(
            {
                "op": "propose_belief",
                "source_kind": "review",
                "confidence": "medium",
                "target_lifecycle": "candidate",
                "force_new_candidate": True,
                "point": str(weak.get("point") or "").strip(),
                "topic": str(weak.get("topic") or "").strip(),
                "planned_layer": str(weak.get("planned_layer") or "").strip(),
                "category": normalize_category(weak.get("category")),
                "scope": str(weak.get("scope") or "domain").strip() or "domain",
                "domain_anchor": weak.get("domain_anchor") or {},
                "source_note_paths": list(weak.get("source_note_paths") or []),
                "review_run_id": review_run_id,
                "card_id": weak_id,
                "turn_id": latest_turn_id,
                "evidence_summary": evidence_summary,
            }
        )

    if not observations:
        return {"changed": False, "observations": 0, "added_candidate_ids": [], "warnings": warnings}

    from services.memory.commit import commit_observations

    model_after, operations = commit_observations(model_before, observations, today=today)
    store.save_v4(model_after)
    added_ids = [
        str(item.get("id") or "")
        for item in model_after.get("beliefs") or []
        if isinstance(item, dict)
        and str(item.get("id") or "") not in before_ids
        and str(item.get("lifecycle") or "") == "candidate"
    ]
    return {
        "changed": bool(operations.get("changed")),
        "observations": len(observations),
        "added_candidate_ids": added_ids,
        "warnings": warnings,
        "operations": operations,
        "canonical_revision": int(model_after.get("canonical_revision") or 0),
    }


def _latest_dialogue_turn_id(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("turn_id"):
            return str(message.get("turn_id") or "")
    return ""


def _review_candidate_evidence_summary(
    suggestion: dict[str, Any],
    weak: dict[str, Any],
    messages: list[dict[str, Any]],
) -> str:
    explicit = str(suggestion.get("evidence") or suggestion.get("reason") or "").strip()
    if explicit:
        return explicit
    action = str(suggestion.get("action") or suggestion.get("suggested_action") or "retry").strip().lower()
    latest_user = ""
    latest_assistant = ""
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role == "assistant" and not latest_assistant:
            latest_assistant = content
        elif role == "user" and not latest_user:
            latest_user = content
        if latest_user and latest_assistant:
            break
    snippets = [item[:120] for item in (latest_user, latest_assistant) if item]
    if snippets:
        return f"review dialogue suggested {action}: " + " / ".join(snippets)
    return f"review dialogue suggested {action} for: {str(weak.get('point') or '').strip()}"


def parse_correction_payload(text: str, *, citations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    try:
        payload = parse_json_object(text)
    except Exception:
        return {
            "covered": [],
            "missing": [],
            "corrections": [str(text or "").strip()] if str(text or "").strip() else [],
            "feedback": str(text or "").strip(),
            "suggested_outcome": "fail",
            "parse_error": True,
            "citations": list(citations or []),
        }
    outcome = str(payload.get("suggested_outcome") or "fail").strip().lower()
    if outcome not in {"pass", "fail"}:
        outcome = "fail"
    return {
        "covered": list_of_strings(payload.get("covered")),
        "missing": list_of_strings(payload.get("missing")),
        "corrections": list_of_strings(payload.get("corrections")),
        "feedback": str(payload.get("feedback") or "").strip(),
        "suggested_outcome": outcome,
        "parse_error": False,
        "citations": list(citations or []),
    }


def parse_verification_payload(text: str, *, citations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    try:
        payload = parse_json_object(text)
    except Exception:
        raw = str(text or "").strip()
        return {
            "correct": [],
            "missed": [],
            "knowledge_correct": False,
            "strategy_feedback": [],
            "example": raw,
            "feedback": raw,
            "suggested_action": "retry",
            "parse_error": True,
            "citations": list(citations or []),
        }
    suggested = str(payload.get("suggested_action") or payload.get("suggested_outcome") or "retry").strip().lower()
    if suggested in {"pass", "improved"}:
        suggested = "improve"
    if suggested not in {"improve", "retry"}:
        suggested = "retry"
    correct = payload.get("correct", payload.get("covered"))
    missed = payload.get("missed", payload.get("missing"))
    example = str(payload.get("example") or payload.get("feedback") or "").strip()
    return {
        "correct": list_of_strings(correct),
        "missed": list_of_strings(missed),
        "knowledge_correct": bool(payload.get("knowledge_correct", suggested == "improve")),
        "strategy_feedback": list_of_strings(payload.get("strategy_feedback")),
        "example": example,
        "feedback": str(payload.get("feedback") or "").strip(),
        "suggested_action": suggested,
        "parse_error": False,
        "citations": list(citations or []),
    }


def parse_grouped_verification_payload(
    text: str,
    *,
    weak_points: list[dict[str, Any]],
    citations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    weak_by_id = {weak_point_id(weak): weak for weak in weak_points}
    try:
        payload = parse_json_object(text)
    except Exception:
        raw = str(text or "").strip()
        return {
            "overall": raw,
            "correct": [],
            "missed": [],
            "example": raw,
            "feedback": raw,
            "suggested_action": "retry",
            "weak_results": [
                {
                    "weak_point_id": weak_id,
                    "point": str(weak.get("point") or "").strip(),
                    "suggested_action": "retry",
                    "reason": "verification response could not be parsed",
                }
                for weak_id, weak in weak_by_id.items()
            ],
            "parse_error": True,
            "citations": list(citations or []),
        }
    weak_results: list[dict[str, Any]] = []
    raw_results = payload.get("weak_results")
    if isinstance(raw_results, list):
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            weak_id = str(item.get("weak_point_id") or "").strip()
            if weak_id not in weak_by_id:
                continue
            action = normalize_review_action(item.get("suggested_action") or item.get("suggested_outcome"))
            weak_results.append(
                {
                    "weak_point_id": weak_id,
                    "point": str(item.get("point") or weak_by_id[weak_id].get("point") or "").strip(),
                    "suggested_action": action,
                    "reason": str(item.get("reason") or "").strip(),
                }
            )
    existing = {item["weak_point_id"] for item in weak_results}
    overall_action = normalize_review_action(payload.get("suggested_action") or payload.get("suggested_outcome"))
    for weak_id, weak in weak_by_id.items():
        if weak_id in existing:
            continue
        weak_results.append(
            {
                "weak_point_id": weak_id,
                "point": str(weak.get("point") or "").strip(),
                "suggested_action": overall_action,
                "reason": str(payload.get("feedback") or payload.get("overall") or "").strip(),
            }
        )
    correct = list_of_strings(payload.get("correct", payload.get("covered")))
    missed = list_of_strings(payload.get("missed", payload.get("missing")))
    suggested = "improve" if weak_results and all(item.get("suggested_action") == "improve" for item in weak_results) else "retry"
    return {
        "overall": str(payload.get("overall") or payload.get("feedback") or "").strip(),
        "correct": correct,
        "missed": missed,
        "knowledge_correct": suggested == "improve",
        "strategy_feedback": list_of_strings(payload.get("strategy_feedback")),
        "example": str(payload.get("example") or payload.get("feedback") or "").strip(),
        "feedback": str(payload.get("feedback") or payload.get("overall") or "").strip(),
        "suggested_action": suggested,
        "weak_results": weak_results,
        "parse_error": False,
        "citations": list(citations or []),
    }


def normalize_review_action(value: Any) -> str:
    action = str(value or "retry").strip().lower()
    if action in {"pass", "improved", "improve"}:
        return "improve"
    return "retry"


def build_correction_query(*, prompt: str, weak: dict[str, Any], answer: str) -> str:
    return "\n".join(
        [
            "\u8bf7\u4f5c\u4e3a\u590d\u4e60\u7ea0\u504f\u52a9\u624b\uff0c\u5bf9\u7167\u6307\u5b9a\u7b14\u8bb0\u68c0\u67e5\u8fd9\u6b21\u4e3b\u52a8\u56de\u5fc6\u4f5c\u7b54\u3002",
            "\u53ea\u8fd4\u56de JSON\uff0c\u4e0d\u8981\u8f93\u51fa Markdown\u3002",
            "",
            "JSON schema:",
            '{"covered":["\u5df2\u8986\u76d6\u70b9"],"missing":["\u9057\u6f0f\u70b9"],"corrections":["\u7ea0\u504f\u8bf4\u660e"],"feedback":"\u7b80\u77ed\u53cd\u9988","suggested_outcome":"pass|fail"}',
            "",
            "\u8584\u5f31\u70b9:",
            json.dumps(profile_weak_point_for_agent(weak), ensure_ascii=False),
            "",
            "\u56de\u5fc6\u9898:",
            prompt,
            "",
            "\u7528\u6237\u4f5c\u7b54:",
            answer,
        ]
    )


def build_weak_point_verification_query(
    *,
    weak: dict[str, Any],
    answer: str,
    prompt: str = "",
    strategy_constraints: list[dict[str, Any]] | None = None,
) -> str:
    return "\n".join(
        [
            "\u8bf7\u4f5c\u4e3a\u590d\u4e60\u7ea0\u504f\u52a9\u624b\uff0c\u5bf9\u7167\u6307\u5b9a\u7b14\u8bb0\u68c0\u67e5\u7528\u6237\u5bf9\u8fd9\u4e2a\u8584\u5f31\u70b9\u7684\u81ea\u8ff0\u3002",
            "\u540c\u65f6\u68c0\u67e5\u7528\u6237\u662f\u5426\u6ee1\u8db3\u672c\u9898\u4f5c\u7b54\u8981\u6c42\uff0c\u4f46\u4e0d\u8981\u628a\u4f5c\u7b54\u8981\u6c42\u5f53\u6210\u4e3b\u77e5\u8bc6\u70b9\u8bc4\u5206\u3002",
            "\u53ea\u8fd4\u56de JSON\uff0c\u4e0d\u8981\u8f93\u51fa Markdown\u3002",
            "",
            "JSON schema:",
            '{"knowledge_correct":true,"strategy_feedback":["\u4f5c\u7b54\u8981\u6c42\u7684\u8bc4\u4f30"],"correct":["\u65b9\u5411\u6b63\u786e\u7684\u70b9"],"missed":["\u8fd8\u7f3a\u7684\u70b9"],"example":"\u4e00\u6bb5\u53ef\u76f4\u63a5\u5b66\u4e60\u7684\u793a\u4f8b\u8868\u8fbe","feedback":"\u7b80\u77ed\u53cd\u9988","suggested_action":"improve|retry"}',
            "",
            "\u4e3b\u77e5\u8bc6\u8584\u5f31\u70b9:",
            json.dumps(profile_weak_point_for_agent(weak), ensure_ascii=False),
            "",
            "\u672c\u9898\u9898\u9762:",
            str(prompt or "").strip(),
            "",
            "\u672c\u9898\u4f5c\u7b54\u8981\u6c42\uff08\u4ec5\u4f5c\u4e3a\u8868\u8fbe\u548c\u601d\u8def\u8bc4\u4f30\uff09:",
            json.dumps(list(strategy_constraints or []), ensure_ascii=False),
            "",
            "\u7528\u6237\u5f53\u524d\u7406\u89e3:",
            answer,
        ]
    )


def build_grouped_weak_point_verification_query(
    *,
    weak_points: list[dict[str, Any]],
    answer: str,
    prompt: str = "",
    question_blocks: list[dict[str, Any]] | None = None,
    strategy_constraints: list[dict[str, Any]] | None = None,
) -> str:
    return "\n".join(
        [
            "请作为复习纠偏助手，对照指定笔记检查用户对一张复习卡的回答。",
            "这张卡可能覆盖多个薄弱点。请给出整体反馈，并对每个 weak_point 单独给 suggested_action。",
            "只返回 JSON，不要输出 Markdown。",
            "",
            "JSON schema:",
            '{"overall":"整体反馈","correct":["方向正确点"],"missed":["遗漏点"],"example":"两三句话示例回答","suggested_action":"improve|retry","weak_results":[{"weak_point_id":"...","point":"...","suggested_action":"improve|retry","reason":"..."}]}',
            "",
            "薄弱点列表:",
            json.dumps(
                [profile_weak_point_for_agent(weak) | {"id": weak_point_id(weak)} for weak in weak_points],
                ensure_ascii=False,
            ),
            "",
            "题面:",
            str(prompt or "").strip(),
            "",
            "题目分段:",
            json.dumps(list(question_blocks or []), ensure_ascii=False),
            "",
            "作答策略要求（仅作为表达和思路评估）:",
            json.dumps(list(strategy_constraints or []), ensure_ascii=False),
            "",
            "用户当前回答:",
            answer,
        ]
    )


def parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end >= start:
        raw = raw[start : end + 1]
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object")
    return payload


def list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
