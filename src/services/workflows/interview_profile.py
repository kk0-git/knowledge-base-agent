from __future__ import annotations

import json
from difflib import SequenceMatcher
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from knowledge_base_agent.llm.schema import LLMMessage, LLMRequest
from services.workflows.interview import InterviewPlan


PROFILE_SCHEMA_VERSION = 1


PROFILE_EXTRACTION_SYSTEM_PROMPT = """# Role

You are an interview memory curator for a personal technical interview practice system.

You are not the interviewer and you are not editing the profile directly. Your job is to read the completed session, the per-turn debriefs, and the existing profile, then extract durable learning observations for the profile updater.

# What To Extract

Extract only observations that are useful across future interview sessions:
- weak_point: a recurring, important, or interview-relevant gap.
- strong_point: a durable strength shown by the user.
- improvement: evidence that an existing weak point may have improved.

Prefer concrete engineering phrasing over generic labels. Keep each point short enough to become a future interview probe.

# Boundaries

- Write in Simplified Chinese.
- Do not score the user.
- Do not create weak points from one trivial typo or a single casual phrase.
- If a turn_review already contains profile_signals, treat them as evidence hints, not final truth.
- Use planned_layer only when it is clear from the session plan or turn review. Otherwise leave it empty.
- For improvement, include weak_point_ref when an existing weak point or signal reference is clear.

# Output

Return only JSON:
{
  "final_review": {
    "summary": "short session-level debrief",
    "clear_strengths": ["durable strengths"],
    "active_gaps": ["remaining gaps"],
    "next_review_focus": ["what should be reviewed next"]
  },
  "observations": [
    {
      "type": "weak_point|strong_point|improvement",
      "topic": "topic name if clear",
      "planned_layer": "planned layer if clear, otherwise empty string",
      "point": "short durable point",
      "weak_point_ref": "existing weak point text if improvement, otherwise empty string",
      "evidence": "brief evidence from the session",
      "confidence": "low|medium|high"
    }
  ],
  "communication": {
    "style": "stable expression pattern if any, otherwise empty string",
    "suggestions": ["durable communication suggestions"]
  }
}
"""


def default_interview_profile() -> dict[str, Any]:
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "updated_at": "",
        "weak_points": [],
        "strong_points": [],
        "topic_mastery": {},
        "communication": {
            "style": "",
            "suggestions": [],
        },
    }


class InterviewProfileStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            profile = default_interview_profile()
            self.save(profile)
            return profile
        profile = json.loads(self.path.read_text(encoding="utf-8-sig"))
        return normalize_interview_profile(profile)

    def save(self, profile: dict[str, Any]) -> None:
        profile = normalize_interview_profile(profile)
        profile["updated_at"] = datetime.now(timezone.utc).isoformat()
        profile = recompute_topic_mastery(profile)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    def update_from_session(
        self,
        *,
        session: dict[str, Any],
        reviews: list[dict[str, Any]],
        llm_client: Any | None = None,
        model: str | None = None,
        temperature: float = 0.1,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        profile_before = self.load()
        extraction_error = ""
        observations: list[dict[str, Any]]
        final_review: dict[str, Any]
        communication: dict[str, Any]

        try:
            if not llm_client or not model:
                raise ValueError("LLM client and model are required for profile extraction")
            extracted = extract_profile_observations(
                session=session,
                reviews=reviews,
                profile=profile_before,
                llm_client=llm_client,
                model=model,
                temperature=temperature,
            )
            observations = extracted["observations"]
            final_review = extracted["final_review"]
            communication = extracted["communication"]
            extraction_source = "llm"
        except Exception as exc:
            extraction_error = str(exc)
            observations = observations_from_turn_reviews(reviews)
            final_review = fallback_final_review(session, observations)
            communication = {}
            extraction_source = "turn_review_signals_fallback"

        profile_after, operations = apply_profile_observations(
            profile=profile_before,
            observations=observations,
            session=session,
            communication=communication,
        )
        self.save(profile_after)
        profile_after = self.load()
        update = {
            "source": extraction_source,
            "extraction_error": extraction_error,
            "operations": operations,
            "observation_count": len(observations),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        return final_review, update


def normalize_interview_profile(profile: dict[str, Any]) -> dict[str, Any]:
    result = default_interview_profile()
    result.update(profile or {})
    result["weak_points"] = list(result.get("weak_points") or [])
    result["strong_points"] = list(result.get("strong_points") or [])
    result["topic_mastery"] = dict(result.get("topic_mastery") or {})
    communication = result.get("communication")
    if not isinstance(communication, dict):
        communication = {}
    result["communication"] = {
        "style": str(communication.get("style") or "").strip(),
        "suggestions": [str(item).strip() for item in communication.get("suggestions", []) if str(item).strip()],
    }
    return recompute_topic_mastery(result)


def extract_profile_observations(
    *,
    session: dict[str, Any],
    reviews: list[dict[str, Any]],
    profile: dict[str, Any],
    llm_client: Any,
    model: str,
    temperature: float = 0.1,
) -> dict[str, Any]:
    user_content = "\n\n".join(
        [
            "# Existing Profile",
            json.dumps(compact_profile_for_extraction(profile), ensure_ascii=False, indent=2),
            "",
            "# Session Context",
            json.dumps(session_context_for_extraction(session), ensure_ascii=False, indent=2),
            "",
            "# Interview Plan",
            json.dumps(session.get("interview_plan") or {}, ensure_ascii=False, indent=2),
            "",
            "# Transcript",
            render_session_transcript(session.get("messages") or []),
            "",
            "# Turn Reviews And Signals",
            json.dumps(compact_reviews_for_extraction(reviews), ensure_ascii=False, indent=2),
        ]
    )
    response = llm_client.complete(
        LLMRequest(
            model=model,
            messages=[
                LLMMessage(role="system", content=PROFILE_EXTRACTION_SYSTEM_PROMPT),
                LLMMessage(role="user", content=user_content),
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    )
    payload = parse_json_object(response.content)
    return normalize_profile_extraction(payload)


def compact_profile_for_extraction(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "weak_points": [
            {
                "point": weak.get("point"),
                "topic": weak.get("topic"),
                "planned_layer": weak.get("planned_layer", ""),
                "improved": bool(weak.get("improved")),
            }
            for weak in profile.get("weak_points", [])[:60]
        ],
        "strong_points": [
            {
                "point": item.get("point"),
                "topic": item.get("topic"),
            }
            for item in profile.get("strong_points", [])[:40]
        ],
        "topic_mastery": profile.get("topic_mastery") or {},
        "communication": profile.get("communication") or {},
    }


def session_context_for_extraction(session: dict[str, Any]) -> dict[str, Any]:
    context = session.get("context") or {}
    state = session.get("interview_state") or {}
    return {
        "session_id": session.get("session_id"),
        "created_at": session.get("created_at"),
        "source_type": context.get("source_type"),
        "source_value": context.get("source_value"),
        "source_note_paths": context.get("source_note_paths") or [],
        "current_topic": state.get("current_topic"),
        "current_layer_index": state.get("current_layer_index"),
    }


def compact_reviews_for_extraction(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for review in reviews:
        feedback = review.get("feedback") or {}
        compact.append(
            {
                "turn_id": review.get("turn_id"),
                "user_message_id": review.get("user_message_id"),
                "assistant_message_id": review.get("assistant_message_id"),
                "feedback": {
                    "coach_note": feedback.get("coach_note") or feedback.get("overall") or feedback.get("summary") or "",
                    "covered": feedback.get("covered", []),
                    "gaps": feedback.get("gaps") or feedback.get("missing") or feedback.get("could_cover") or [],
                    "thinking_framework": feedback.get("thinking_framework")
                    or feedback.get("next_focus")
                    or feedback.get("next_tip")
                    or feedback.get("next_step")
                    or "",
                    "interviewer_followup_note": feedback.get("interviewer_followup_note")
                    or feedback.get("interviewer_direction")
                    or "",
                },
                "expression_example": review.get("expression_example") or review.get("reference_answer") or "",
                "profile_signals": review.get("profile_signals") or [],
            }
        )
    return compact


def render_session_transcript(messages: list[dict[str, Any]], max_chars: int = 18000) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "").strip() or "unknown"
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    transcript = "\n\n".join(lines)
    if len(transcript) <= max_chars:
        return transcript
    return transcript[-max_chars:]


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            payload = json.loads(text[start : end + 1])
            return payload if isinstance(payload, dict) else {}
        raise


def normalize_profile_extraction(payload: dict[str, Any]) -> dict[str, Any]:
    final_review = payload.get("final_review")
    if not isinstance(final_review, dict):
        final_review = {}
    communication = payload.get("communication")
    if not isinstance(communication, dict):
        communication = {}
    return {
        "final_review": {
            "summary": str(final_review.get("summary") or "").strip(),
            "clear_strengths": dedupe_strings(final_review.get("clear_strengths") or [], max_items=8),
            "active_gaps": dedupe_strings(final_review.get("active_gaps") or [], max_items=8),
            "next_review_focus": dedupe_strings(final_review.get("next_review_focus") or [], max_items=8),
        },
        "observations": normalize_observations(payload.get("observations") or []),
        "communication": {
            "style": str(communication.get("style") or "").strip(),
            "suggestions": dedupe_strings(communication.get("suggestions") or [], max_items=5),
        },
    }


def normalize_observations(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    observations: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        obs_type = normalize_observation_type(item.get("type"))
        if not obs_type:
            continue
        point = str(item.get("point") or item.get("summary") or "").strip()
        evidence = str(item.get("evidence") or "").strip()
        if not point and not evidence:
            continue
        observations.append(
            {
                "type": obs_type,
                "topic": str(item.get("topic") or "").strip(),
                "planned_layer": str(item.get("planned_layer") or "").strip(),
                "point": point,
                "weak_point_ref": str(item.get("weak_point_ref") or "").strip(),
                "evidence": evidence,
                "confidence": normalize_confidence(item.get("confidence")),
            }
        )
        if len(observations) >= 30:
            break
    return observations


def normalize_observation_type(value: Any) -> str:
    text = str(value or "").strip()
    aliases = {
        "possible_weak_point": "weak_point",
        "weak": "weak_point",
        "weak_point": "weak_point",
        "possible_improvement": "improvement",
        "improved": "improvement",
        "improvement": "improvement",
        "strong": "strong_point",
        "strong_point": "strong_point",
    }
    return aliases.get(text, "")


def normalize_confidence(value: Any) -> str:
    text = str(value or "medium").strip().lower()
    return text if text in {"low", "medium", "high"} else "medium"


def observations_from_turn_reviews(reviews: list[dict[str, Any]]) -> list[dict[str, str]]:
    raw: list[dict[str, Any]] = []
    for review in reviews:
        for signal in review.get("profile_signals") or []:
            if not isinstance(signal, dict):
                continue
            raw.append(
                {
                    "type": signal.get("type"),
                    "topic": signal.get("topic"),
                    "planned_layer": signal.get("planned_layer"),
                    "point": signal.get("summary") or signal.get("weak_point_ref"),
                    "weak_point_ref": signal.get("weak_point_ref"),
                    "evidence": signal.get("evidence"),
                    "confidence": signal.get("confidence"),
                }
            )
    return normalize_observations(raw)


def fallback_final_review(session: dict[str, Any], observations: list[dict[str, str]]) -> dict[str, Any]:
    weak = [item["point"] for item in observations if item["type"] == "weak_point" and item.get("point")]
    improved = [item["point"] for item in observations if item["type"] == "improvement" and item.get("point")]
    return {
        "summary": "本次面试已归档。长期画像抽取使用每轮复盘信号降级完成。",
        "clear_strengths": improved[:5],
        "active_gaps": weak[:5],
        "next_review_focus": weak[:5],
        "message_count": len(session.get("messages") or []),
    }


def apply_profile_observations(
    *,
    profile: dict[str, Any],
    observations: list[dict[str, str]],
    session: dict[str, Any],
    communication: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = normalize_interview_profile(profile)
    operations = {"added": [], "updated": [], "improved": [], "strong_added": [], "strong_updated": []}
    today = date.today().isoformat()
    session_id = str(session.get("session_id") or "")
    context = session.get("context") or {}
    source_note_paths = list(context.get("source_note_paths") or [])

    for observation in observations:
        obs_type = observation["type"]
        confidence = observation.get("confidence") or "medium"
        if confidence == "low":
            continue
        if obs_type == "weak_point":
            index = find_similar_profile_item(
                updated.get("weak_points", []),
                topic=observation.get("topic", ""),
                text=observation.get("point", ""),
                include_improved=False,
            )
            if index is None:
                weak = new_weak_point(
                    observation=observation,
                    session_id=session_id,
                    source_note_paths=source_note_paths,
                    today=today,
                )
                updated.setdefault("weak_points", []).append(weak)
                operations["added"].append(weak.get("point"))
            else:
                weak = updated["weak_points"][index]
                update_weak_point(weak, observation, session_id=session_id, today=today)
                operations["updated"].append(weak.get("point"))
        elif obs_type == "improvement":
            index = find_improvement_target(updated.get("weak_points", []), observation)
            if index is not None:
                weak = updated["weak_points"][index]
                mark_weak_point_improved(weak, observation, session_id=session_id, today=today)
                operations["improved"].append(weak.get("point"))
            else:
                upsert_strong_point(updated, observation, session_id=session_id, today=today, operations=operations)
        elif obs_type == "strong_point":
            upsert_strong_point(updated, observation, session_id=session_id, today=today, operations=operations)

    merge_communication(updated, communication or {})
    updated = recompute_topic_mastery(updated)
    return updated, operations


def new_weak_point(
    *,
    observation: dict[str, str],
    session_id: str,
    source_note_paths: list[str],
    today: str,
) -> dict[str, Any]:
    return {
        "point": observation.get("point") or observation.get("evidence"),
        "topic": observation.get("topic", ""),
        "planned_layer": observation.get("planned_layer", ""),
        "evidence": compact_evidence(observation.get("evidence", "")),
        "source_session_ids": [session_id] if session_id else [],
        "source_note_paths": source_note_paths[:12],
        "times_seen": 1,
        "first_seen": today,
        "last_seen": today,
        "improved": False,
        "improved_at": "",
        "sr": {
            "interval_days": 1,
            "ease_factor": 2.5,
            "repetitions": 0,
            "next_review": (date.today() + timedelta(days=1)).isoformat(),
        },
    }


def update_weak_point(weak: dict[str, Any], observation: dict[str, str], *, session_id: str, today: str) -> None:
    weak["times_seen"] = int(weak.get("times_seen", 0) or 0) + 1
    weak["last_seen"] = today
    if observation.get("planned_layer") and not weak.get("planned_layer"):
        weak["planned_layer"] = observation["planned_layer"]
    append_unique(weak.setdefault("source_session_ids", []), session_id, max_items=8)
    append_evidence(weak, observation.get("evidence", ""))
    sr = weak.setdefault("sr", {})
    sr["ease_factor"] = max(1.3, float(sr.get("ease_factor", 2.5)) - 0.15)
    sr["interval_days"] = 1
    sr["next_review"] = (date.today() + timedelta(days=1)).isoformat()


def mark_weak_point_improved(weak: dict[str, Any], observation: dict[str, str], *, session_id: str, today: str) -> None:
    weak["last_seen"] = today
    append_unique(weak.setdefault("source_session_ids", []), session_id, max_items=8)
    append_evidence(weak, observation.get("evidence", ""))
    sr = weak.setdefault("sr", {})
    repetitions = int(sr.get("repetitions", 0) or 0) + 1
    ease = min(3.0, float(sr.get("ease_factor", 2.5)) + 0.15)
    interval = next_sm2_interval(repetitions, int(sr.get("interval_days", 1) or 1), ease)
    sr.update(
        {
            "repetitions": repetitions,
            "ease_factor": round(ease, 2),
            "interval_days": interval,
            "next_review": (date.today() + timedelta(days=interval)).isoformat(),
        }
    )
    if repetitions >= 3:
        weak["improved"] = True
        weak["improved_at"] = today


def next_sm2_interval(repetitions: int, previous_interval: int, ease: float) -> int:
    if repetitions <= 1:
        return 3
    if repetitions == 2:
        return 7
    return min(60, max(14, int(previous_interval * ease)))


def upsert_strong_point(
    profile: dict[str, Any],
    observation: dict[str, str],
    *,
    session_id: str,
    today: str,
    operations: dict[str, list[Any]],
) -> None:
    index = find_similar_profile_item(
        profile.get("strong_points", []),
        topic=observation.get("topic", ""),
        text=observation.get("point", ""),
        include_improved=True,
    )
    if index is None:
        item = {
            "point": observation.get("point") or observation.get("evidence"),
            "topic": observation.get("topic", ""),
            "evidence": compact_evidence(observation.get("evidence", "")),
            "source_session_ids": [session_id] if session_id else [],
            "times_seen": 1,
            "first_seen": today,
            "last_seen": today,
        }
        profile.setdefault("strong_points", []).append(item)
        operations["strong_added"].append(item.get("point"))
    else:
        item = profile["strong_points"][index]
        item["times_seen"] = int(item.get("times_seen", 0) or 0) + 1
        item["last_seen"] = today
        append_unique(item.setdefault("source_session_ids", []), session_id, max_items=8)
        append_evidence(item, observation.get("evidence", ""))
        operations["strong_updated"].append(item.get("point"))


def find_improvement_target(items: list[dict[str, Any]], observation: dict[str, str]) -> int | None:
    ref = observation.get("weak_point_ref", "")
    if ref:
        index = find_similar_profile_item(items, topic=observation.get("topic", ""), text=ref, include_improved=False)
        if index is not None:
            return index
    return find_similar_profile_item(
        items,
        topic=observation.get("topic", ""),
        text=observation.get("point", ""),
        include_improved=False,
    )


def find_similar_profile_item(
    items: list[dict[str, Any]],
    *,
    topic: str,
    text: str,
    include_improved: bool,
) -> int | None:
    text_norm = normalize_match_text(text)
    if not text_norm:
        return None
    best_index: int | None = None
    best_score = 0.0
    for index, item in enumerate(items):
        if not include_improved and item.get("improved"):
            continue
        item_topic = str(item.get("topic") or "")
        if topic and item_topic and item_topic != topic:
            continue
        candidate = normalize_match_text(str(item.get("point") or ""))
        if not candidate:
            continue
        score = max(
            SequenceMatcher(None, text_norm, candidate).ratio(),
            containment_score(text_norm, candidate),
        )
        if score > best_score:
            best_score = score
            best_index = index
    return best_index if best_score >= 0.68 else None


def containment_score(left: str, right: str) -> float:
    if left in right or right in left:
        return min(len(left), len(right)) / max(len(left), len(right))
    return 0.0


def normalize_match_text(text: str) -> str:
    return "".join(str(text or "").lower().split())


def compact_evidence(text: str, max_chars: int = 300) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1] + "…"


def append_evidence(item: dict[str, Any], evidence: str) -> None:
    evidence = compact_evidence(evidence)
    if not evidence:
        return
    history = item.setdefault("evidence_history", [])
    if evidence not in history:
        history.append(evidence)
    del history[:-5]
    item["evidence"] = evidence


def append_unique(items: list[Any], value: Any, max_items: int = 10) -> None:
    if not value:
        return
    if value not in items:
        items.append(value)
    del items[:-max_items]


def merge_communication(profile: dict[str, Any], communication: dict[str, Any]) -> None:
    target = profile.setdefault("communication", {"style": "", "suggestions": []})
    style = str(communication.get("style") or "").strip()
    if style:
        target["style"] = style
    suggestions = target.setdefault("suggestions", [])
    for suggestion in communication.get("suggestions") or []:
        append_unique(suggestions, str(suggestion).strip(), max_items=8)


def dedupe_strings(value: Any, max_items: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= max_items:
            break
    return result


def recompute_topic_mastery(profile: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for weak in profile.get("weak_points", []):
        if weak.get("improved"):
            continue
        topic = str(weak.get("topic") or "").strip()
        if not topic:
            continue
        counts[topic] = counts.get(topic, 0) + 1

    mastery = dict(profile.get("topic_mastery") or {})
    today = date.today().isoformat()
    for topic, count in counts.items():
        entry = dict(mastery.get(topic) or {})
        entry["active_weak_count"] = count
        entry["mastery_estimate"] = 100 - min(count * 15, 70)
        entry["last_assessed"] = entry.get("last_assessed") or today
        mastery[topic] = entry
    for topic, entry in list(mastery.items()):
        if topic not in counts:
            updated = dict(entry or {})
            updated["active_weak_count"] = 0
            updated["mastery_estimate"] = 100
            mastery[topic] = updated
    profile["topic_mastery"] = mastery
    return profile


def render_candidate_profile_context(
    *,
    profile: dict[str, Any] | None,
    current_topic: str | None,
    plan: InterviewPlan | None,
) -> str:
    if not profile:
        return "(no candidate profile available)"

    topic = resolve_current_topic(current_topic=current_topic, plan=plan)
    lines: list[str] = []
    lines.append(f"Current topic: {topic or '(unknown)'}")

    mastery = (profile.get("topic_mastery") or {}).get(topic or "", {})
    if mastery:
        estimate = mastery.get("mastery_estimate")
        weak_count = mastery.get("active_weak_count", 0)
        if estimate is not None:
            level = mastery_level_label(float(estimate))
            lines.extend(
                [
                    "",
                    "Topic overview:",
                    f"- {topic}: estimated {level} level (~{estimate}/100), {weak_count} active weak point(s).",
                    "  Interview behavior: calibrate baseline difficulty from this estimate; do not mention the score to the user.",
                ]
            )

    topic_weak = [
        weak for weak in profile.get("weak_points", [])
        if not weak.get("improved") and topic and str(weak.get("topic") or "") == topic
    ]
    due_matching, due_other = split_due_reviews(profile, topic)
    strong = [
        item for item in profile.get("strong_points", [])
        if topic and str(item.get("topic") or "") == topic
    ][:4]

    if topic_weak:
        lines.extend(["", "Priority weak points for this topic:"])
        for weak in topic_weak[:8]:
            layer = str(weak.get("planned_layer") or "").strip()
            lines.append(f"- {weak.get('point')}")
            if layer:
                lines.append(f"  planned layer: {layer}")
            else:
                lines.append("  planned layer: topic-level")
            lines.append("  behavior: use as a private probe; ask one extra follow-up if the answer is vague.")
    else:
        lines.extend(["", "Priority weak points for this topic: none recorded."])

    if due_matching:
        lines.extend(["", "Due reviews for current topic:"])
        for weak in due_matching[:5]:
            lines.extend(render_due_review_lines(weak))

    if due_other:
        lines.extend(["", "Due reviews not matching current topic:"])
        for weak in due_other[:5]:
            lines.append(f"- {weak.get('topic')}: {weak.get('point')}")
            lines.append("  behavior: do not introduce unless the user switches to that topic.")

    if strong:
        lines.extend(["", "Recent strengths for this topic:"])
        for item in strong:
            lines.append(f"- {item.get('point')}")

    communication = profile.get("communication") or {}
    if communication.get("style") or communication.get("suggestions"):
        lines.extend(["", "Communication notes:"])
        if communication.get("style"):
            lines.append(f"- style: {communication['style']}")
        for suggestion in communication.get("suggestions", [])[:3]:
            lines.append(f"- suggestion: {suggestion}")

    lines.extend(
        [
            "",
            "Interview policy:",
            "- Do not say the user was previously weak on these points.",
            "- Use this profile only as private guidance for what to verify.",
            "- If the user shows clear improvement, leave that as a profile signal in the turn review; do not announce it as a profile update.",
        ]
    )
    return "\n".join(lines)


def resolve_current_topic(*, current_topic: str | None, plan: InterviewPlan | None) -> str | None:
    if current_topic:
        return current_topic
    if plan and plan.topics:
        return plan.topics[0].name
    return None


def split_due_reviews(profile: dict[str, Any], current_topic: str | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    today = date.today().isoformat()
    matching: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    for weak in profile.get("weak_points", []):
        if weak.get("improved"):
            continue
        sr = weak.get("sr") or {}
        next_review = str(sr.get("next_review") or "2000-01-01")
        if next_review > today:
            continue
        if current_topic and weak.get("topic") == current_topic:
            matching.append(weak)
        else:
            other.append(weak)
    key = lambda item: float((item.get("sr") or {}).get("ease_factor", 2.5))
    return sorted(matching, key=key), sorted(other, key=key)


def render_due_review_lines(weak: dict[str, Any]) -> list[str]:
    sr = weak.get("sr") or {}
    ease = float(sr.get("ease_factor", 2.5))
    repetitions = int(sr.get("repetitions", 0) or 0)
    priority = "high" if ease < 2.0 or repetitions >= 3 else "normal"
    return [
        f"- {weak.get('point')}",
        f"  priority: {priority}",
        f"  history: reviewed {repetitions} time(s), ease {ease}, interval {sr.get('interval_days', 1)} day(s)",
        "  interview behavior: spend extra follow-up depth only if the answer is vague.",
    ]


def mastery_level_label(estimate: float) -> str:
    if estimate >= 80:
        return "strong"
    if estimate >= 55:
        return "intermediate"
    return "foundational"
