from __future__ import annotations

import json
from difflib import SequenceMatcher
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from knowledge_base_agent.llm.schema import LLMMessage, LLMRequest
from services.workflows.interview import InterviewPlan


PROFILE_SCHEMA_VERSION = 3


PROFILE_EXTRACTION_SYSTEM_PROMPT = """# Role

You are an interview memory curator for a personal technical interview practice system.

You are not the interviewer and you are not editing the profile directly. Your job is to read the completed session, the per-turn debriefs, and the existing profile, then extract durable learning observations for the profile updater.

# What To Extract

Extract only observations that are useful across future interview sessions:
- weak_point: a recurring, important, or interview-relevant gap.
- partial: evidence that the user partially addressed an existing weak point, but not completely.
- strong_point: a durable strength shown by the user.
- improvement: evidence that an existing weak point may have improved.
- procedure: an explicit assistant interaction preference, such as "answer with conclusion first" or "use counterexamples".
- confusion_pair: a durable confusion between two concepts.

Prefer concrete engineering phrasing over generic labels. Keep each point short enough to become a future interview probe.

# Boundaries

- Write in Simplified Chinese.
- Do not score the user.
- Do not create weak points from one trivial typo or a single casual phrase.
- If a turn_review already contains profile_signals, treat them as evidence hints, not final truth.
- Use planned_layer only when it is clear from the session plan or turn review. Otherwise leave it empty.
- For improvement, include weak_point_ref when an existing weak point or signal reference is clear.
- For partial, include weak_point_ref when possible. Do not use partial for a brand-new weak point.
- For weak_point, include category and scope_suggestion:
  - category must be one of: knowledge_gap, answer_structure, communication, thinking_pattern.
  - scope_suggestion must be domain or universal.
  - Concrete knowledge/mechanism/component gaps should be domain.
  - Listening, communication, or answer-structure habits may be universal, but the system will make the final scope decision.
- Do not output domain_anchor; it is written by code.
- Do not use procedure for learner weaknesses such as unclear expression, shallow reasoning, or listening mistakes. Those are weak_point observations with category communication, answer_structure, or thinking_pattern.

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
      "type": "weak_point|partial|strong_point|improvement|procedure|confusion_pair",
      "topic": "topic name if clear",
      "planned_layer": "planned layer if clear, otherwise empty string",
      "category": "knowledge_gap|answer_structure|communication|thinking_pattern",
      "scope_suggestion": "domain|universal",
      "point": "short durable point",
      "procedure_key": "stable key for procedure, if type=procedure",
      "title": "procedure title, if type=procedure",
      "steps": ["procedure steps, if type=procedure"],
      "left": "left concept, if type=confusion_pair",
      "right": "right concept, if type=confusion_pair",
      "distinction": "short distinction, if type=confusion_pair",
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


ANSWER_MEMORY_EXTRACTION_SYSTEM_PROMPT = """# Role

You are a learner memory curator for a personal knowledge Q&A system.

You read a completed answer session transcript and extract durable learning observations. You do not edit memory directly.

# What To Extract

Extract observations that are useful across future Q&A sessions:
- recurring knowledge gaps
- answer-structure or thinking-pattern habits when clearly shown
- procedure observations only when the user explicitly asks for answer or interaction style
- confusion_pair observations when the user clearly mixes two concepts

# Boundaries

- Write in Simplified Chinese.
- Do not score the user.
- Do not create weak points from one trivial typo.
- category must be one of: knowledge_gap, answer_structure, communication, thinking_pattern.
- scope_suggestion must be domain or universal.
- confidence high only when the user clearly admits not understanding or gave an answer clearly wrong on a cited topic.
- Do not output domain_anchor; it is written by code.
- Do not use procedure for learner weaknesses such as unclear expression or shallow reasoning. Those remain weak_point observations.

# Output

Return only JSON:
{
  "observations": [
    {
      "type": "weak_point|procedure|confusion_pair",
      "topic": "topic if clear",
      "planned_layer": "",
      "category": "knowledge_gap|answer_structure|communication|thinking_pattern",
      "scope_suggestion": "domain|universal",
      "point": "short durable point",
      "procedure_key": "stable key for procedure, if type=procedure",
      "title": "procedure title, if type=procedure",
      "steps": ["procedure steps, if type=procedure"],
      "left": "left concept, if type=confusion_pair",
      "right": "right concept, if type=confusion_pair",
      "distinction": "short distinction, if type=confusion_pair",
      "evidence": "brief evidence from the session",
      "confidence": "low|medium|high"
    }
  ]
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
    def __init__(self, path: Path | str, *, legacy_path: Path | str | None = None):
        from services.memory.store import LearnerModelStore

        self.path = Path(path)
        self.legacy_path = Path(legacy_path) if legacy_path is not None else self.path
        self._learner_store = LearnerModelStore(self.path, self.legacy_path)

    def load_v4(self) -> dict[str, Any]:
        return self._learner_store.load()

    def save_v4(self, model: dict[str, Any]) -> None:
        from services.memory.bridge import learner_model_to_profile_view
        from services.memory.schema import normalize_learner_model

        normalized = normalize_learner_model(model)
        profile_view = learner_model_to_profile_view(normalized)
        profile_view = recompute_topic_mastery(profile_view)
        legacy = dict(normalized.get("legacy") or {})
        legacy["topic_mastery"] = profile_view.get("topic_mastery") or {}
        normalized["legacy"] = legacy
        normalized["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._learner_store.save(normalized)

    def load(self) -> dict[str, Any]:
        from services.memory.bridge import learner_model_to_profile_view

        model = self.load_v4()
        profile = learner_model_to_profile_view(model)
        return normalize_interview_profile(profile)

    def save(self, profile: dict[str, Any]) -> None:
        from services.memory.bridge import sync_profile_view_to_model

        if int(profile.get("schema_version") or 0) >= 4 and (profile.get("beliefs") is not None or profile.get("learner_items") is not None):
            self.save_v4(profile)
            return
        model = sync_profile_view_to_model(profile, self.load_v4())
        self.save_v4(model)

    def update_from_session(
        self,
        *,
        session: dict[str, Any],
        reviews: list[dict[str, Any]],
        llm_client: Any | None = None,
        model: str | None = None,
        temperature: float = 0.1,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from services.memory.bridge import (
            apply_legacy_profile_observations_to_model,
            build_memory_extraction_checkpoint,
            observations_from_profile_extractor,
            should_skip_memory_extraction,
        )
        from services.memory.commit import commit_observations

        session_id = str(session.get("session_id") or "")
        if should_skip_memory_extraction(session, reviews=reviews):
            checkpoint = session.get("memory_extraction") or {}
            return (
                fallback_final_review(session, []),
                {
                    "source": "memory_extraction_checkpoint",
                    "extraction_error": "",
                    "operations": {"skipped": True},
                    "observation_count": int(checkpoint.get("observation_count") or 0),
                    "filtered_low_count": int(checkpoint.get("filtered_low_count") or 0),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

        model_before = self.load_v4()
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

        commit_obs = observations_from_profile_extractor(
            observations,
            session=session,
            model=model_before,
            communication=communication,
        )
        model_after, commit_ops = commit_observations(model_before, commit_obs)
        model_after, legacy_ops = apply_legacy_profile_observations_to_model(
            model_after,
            observations,
            session=session,
            communication=communication,
        )
        self.save_v4(model_after)
        model_after = self.load_v4()

        if session_id:
            session.setdefault("memory_extraction", {})
            session["memory_extraction"] = build_memory_extraction_checkpoint(
                session=session,
                reviews=reviews,
                trigger="interview.session_ended",
                observation_count=len(observations),
                filtered_low_count=int(commit_ops.get("filtered_low_count") or 0),
                commit_revision=int(model_after.get("canonical_revision") or 0),
            )

        operations = {
            "added": [],
            "updated": [],
            "partial": legacy_ops.get("partial") or [],
            "improved": [],
            "strong_added": legacy_ops.get("strong_added") or [],
            "strong_updated": legacy_ops.get("strong_updated") or [],
            "commit": commit_ops,
        }
        update = {
            "source": extraction_source,
            "extraction_error": extraction_error,
            "operations": operations,
            "observation_count": len(observations),
            "filtered_low_count": int(commit_ops.get("filtered_low_count") or 0),
            "canonical_revision": int(model_after.get("canonical_revision") or 0),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        return final_review, update

    def update_from_answer_session(
        self,
        *,
        session: dict[str, Any],
        llm_client: Any | None = None,
        model: str | None = None,
        temperature: float = 0.1,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from services.memory.bridge import (
            build_memory_extraction_checkpoint,
            observations_from_answer_extractor,
            observations_from_answer_rules_fallback,
            should_skip_memory_extraction,
        )
        from services.memory.commit import commit_observations

        reviews: list[dict[str, Any]] = []
        session_id = str(session.get("session_id") or "")
        if should_skip_memory_extraction(session, reviews=reviews):
            checkpoint = session.get("memory_extraction") or {}
            return (
                {
                    "source": "memory_extraction_checkpoint",
                    "extraction_error": "",
                    "operations": {"skipped": True, "commit": {"skipped": True}},
                    "observation_count": int(checkpoint.get("observation_count") or 0),
                    "filtered_low_count": int(checkpoint.get("filtered_low_count") or 0),
                    "canonical_revision": int(checkpoint.get("commit_revision") or 0),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                session,
            )

        model_before = self.load_v4()
        profile_before = self.load()
        extraction_error = ""
        observations: list[dict[str, Any]]
        extraction_source = "llm"

        try:
            if not llm_client or not model:
                raise ValueError("LLM client and model are required for answer memory extraction")
            extracted = extract_answer_memory_observations(
                session=session,
                profile=profile_before,
                llm_client=llm_client,
                model=model,
                temperature=temperature,
            )
            observations = extracted["observations"]
        except Exception as exc:
            extraction_error = str(exc)
            observations = observations_from_answer_rules_fallback(session)
            extraction_source = "rules_fallback"

        commit_obs = observations_from_answer_extractor(
            observations,
            session=session,
            model=model_before,
        )
        model_after, commit_ops = commit_observations(model_before, commit_obs)
        self.save_v4(model_after)
        model_after = self.load_v4()

        if session_id:
            session.setdefault("memory_extraction", {})
            session["memory_extraction"] = build_memory_extraction_checkpoint(
                session=session,
                reviews=reviews,
                trigger="answer.session_ended",
                observation_count=len(observations),
                filtered_low_count=int(commit_ops.get("filtered_low_count") or 0),
                commit_revision=int(model_after.get("canonical_revision") or 0),
            )

        update = {
            "source": extraction_source,
            "extraction_error": extraction_error,
            "operations": {"commit": commit_ops},
            "observation_count": len(observations),
            "filtered_low_count": int(commit_ops.get("filtered_low_count") or 0),
            "canonical_revision": int(model_after.get("canonical_revision") or 0),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        return update, session

    def apply_user_commit(
        self,
        *,
        action: str,
        belief_id: str = "",
        procedure_id: str = "",
        target_type: str = "belief",
        note: str = "",
    ) -> dict[str, Any]:
        from services.memory.bridge import observation_user_commit
        from services.memory.commit import commit_observations

        model_before = self.load_v4()
        target = str(target_type or "belief").strip() or "belief"
        target_id = procedure_id if target == "procedure" else belief_id
        collection = "assistant_items" if target == "procedure" else "learner_items"
        target_before = next(
            (item for item in (model_before.get(collection) or model_before.get("procedures" if target == "procedure" else "beliefs") or []) if str(item.get("id") or "") == target_id),
            None,
        )
        if target_before is None:
            raise KeyError(f"{target} not found: {target_id}")

        model_after, operations = commit_observations(
            model_before,
            [
                observation_user_commit(
                    action=action,
                    belief_id=belief_id if target == "belief" else "",
                    procedure_id=procedure_id if target == "procedure" else "",
                    target_type=target,
                    note=note,
                )
            ],
        )
        if not operations.get("changed"):
            raise ValueError(f"user commit rejected for action={action} {target}_id={target_id}")

        self.save_v4(model_after)
        target_after = next(
            (item for item in model_after.get(collection) or [] if str(item.get("id") or "") == target_id),
            None,
        )
        return {
            target: target_after,
            "target_type": target,
            "operations": operations,
            "canonical_revision": int(model_after.get("canonical_revision") or 0),
        }

    def list_candidate_beliefs(self) -> list[dict[str, Any]]:
        model = self.load_v4()
        return [
            dict(belief)
            for belief in (model.get("learner_items") or model.get("beliefs") or [])
            if isinstance(belief, dict) and str(belief.get("lifecycle") or "") == "candidate"
        ]

    def list_candidate_procedures(self) -> list[dict[str, Any]]:
        model = self.load_v4()
        return [
            dict(procedure)
            for procedure in (model.get("assistant_items") or model.get("procedures") or [])
            if isinstance(procedure, dict) and str(procedure.get("lifecycle") or "") == "candidate"
        ]

    def list_memory_candidates(self) -> dict[str, Any]:
        from services.memory.ui import enrich_memory_item_for_ui

        beliefs = [enrich_memory_item_for_ui(item) for item in self.list_candidate_beliefs()]
        procedures = [enrich_memory_item_for_ui(item) for item in self.list_candidate_procedures()]
        return {
            "beliefs": beliefs,
            "procedures": procedures,
            "candidates": beliefs,
            "count": len(beliefs) + len(procedures),
        }

    def list_archived_beliefs(self) -> list[dict[str, Any]]:
        model = self.load_v4()
        return [
            dict(belief)
            for belief in (model.get("learner_items") or model.get("beliefs") or [])
            if isinstance(belief, dict) and str(belief.get("lifecycle") or "") == "archived"
        ]

    def list_archived_procedures(self) -> list[dict[str, Any]]:
        model = self.load_v4()
        return [
            dict(procedure)
            for procedure in (model.get("assistant_items") or model.get("procedures") or [])
            if isinstance(procedure, dict) and str(procedure.get("lifecycle") or "") == "archived"
        ]

    def list_memory_archived(self) -> dict[str, Any]:
        from services.memory.ui import enrich_memory_item_for_ui

        beliefs = [enrich_memory_item_for_ui(item) for item in self.list_archived_beliefs()]
        procedures = [enrich_memory_item_for_ui(item) for item in self.list_archived_procedures()]
        return {
            "beliefs": beliefs,
            "procedures": procedures,
            "count": len(beliefs) + len(procedures),
        }

    def ensure_derived_fresh(self) -> dict[str, Any]:
        from services.memory.derived import rebuild_derived

        model = self.load_v4()
        derived = model.get("derived") or {}
        if not derived.get("stale", True):
            return model
        model = rebuild_derived(model)
        self.save_v4(model)
        return model


def normalize_interview_profile(profile: dict[str, Any]) -> dict[str, Any]:
    result = default_interview_profile()
    result.update(profile or {})
    result["schema_version"] = PROFILE_SCHEMA_VERSION
    result["weak_points"] = [normalize_weak_point(item) for item in list(result.get("weak_points") or []) if isinstance(item, dict)]
    result["strong_points"] = [normalize_strong_point(item) for item in list(result.get("strong_points") or []) if isinstance(item, dict)]
    result["topic_mastery"] = dict(result.get("topic_mastery") or {})
    communication = result.get("communication")
    if not isinstance(communication, dict):
        communication = {}
    result["communication"] = {
        "style": str(communication.get("style") or "").strip(),
        "suggestions": [str(item).strip() for item in communication.get("suggestions", []) if str(item).strip()],
    }
    return recompute_topic_mastery(result)


def normalize_weak_point(item: dict[str, Any]) -> dict[str, Any]:
    weak = dict(item or {})
    from services.memory.types import normalize_facet as _normalize_facet
    facet = _normalize_facet(weak.get("facet") or weak.get("category"))
    weak["category"] = "knowledge_gap" if facet == "knowledge" else "answer_structure"
    weak["scope"] = normalize_scope(
        weak.get("scope") or weak.get("scope_suggestion"),
        facet=facet,
        existing_scope=weak.get("scope"),
    )
    weak["domain_anchor"] = normalize_domain_anchor(
        weak.get("domain_anchor"),
        legacy_anchor_note_paths=weak.get("anchor_note_paths"),
        topic=weak.get("topic"),
    )
    weak["source_note_paths"] = [str(path) for path in weak.get("source_note_paths", []) if str(path).strip()]
    weak["source_session_ids"] = [str(item) for item in weak.get("source_session_ids", []) if str(item).strip()]
    weak["times_seen"] = int(weak.get("times_seen", 0) or 0)
    weak["improved"] = bool(weak.get("improved"))
    sr = dict(weak.get("sr") or {})
    sr.setdefault("interval_days", 1)
    sr.setdefault("ease_factor", 2.5)
    sr.setdefault("repetitions", 0)
    sr.setdefault("next_review", date.today().isoformat())
    sr.setdefault("last_outcome", "")
    sr.setdefault("last_reviewed", "")
    weak["sr"] = sr
    return weak


def normalize_strong_point(item: dict[str, Any]) -> dict[str, Any]:
    strong = dict(item or {})
    strong["topic"] = str(strong.get("topic") or "").strip()
    strong["source_session_ids"] = [str(value) for value in strong.get("source_session_ids", []) if str(value).strip()]
    strong["times_seen"] = int(strong.get("times_seen", 0) or 0)
    return strong


def normalize_category(value: Any) -> str:
    """Deprecated — delegates to normalize_facet, maps back to v4 category for compatibility."""
    from services.memory.types import normalize_facet as _normalize_facet
    facet = _normalize_facet(value, default="knowledge")
    return "knowledge_gap" if facet == "knowledge" else "answer_structure"


def normalize_scope(value: Any, *, facet: str = "", category: str = "", existing_scope: Any = None) -> str:
    current = str(existing_scope or value or "").strip().lower()
    if current == "universal":
        return "universal"
    return "domain"


def normalize_scope_suggestion(value: Any) -> str:
    text = str(value or "").strip().lower()
    return "universal" if text == "universal" else "domain"


def normalize_domain_anchor(
    value: Any,
    *,
    legacy_anchor_note_paths: Any = None,
    topic: Any = None,
) -> dict[str, Any]:
    anchor = dict(value or {}) if isinstance(value, dict) else {}
    paths = anchor.get("context_note_paths")
    if paths is None and legacy_anchor_note_paths:
        paths = legacy_anchor_note_paths
    anchor["context_note_paths"] = [str(path) for path in (paths or []) if str(path).strip()]
    anchor["plan_topic"] = str(anchor.get("plan_topic") or topic or "").strip()
    anchor["scope_path"] = str(anchor.get("scope_path") or "").strip()
    return anchor


def extract_answer_memory_observations(
    *,
    session: dict[str, Any],
    profile: dict[str, Any],
    llm_client: Any,
    model: str,
    temperature: float = 0.1,
) -> dict[str, Any]:
    from services.memory.bridge import collect_answer_citation_paths, enrich_answer_session_for_memory

    enriched = enrich_answer_session_for_memory(session)
    citation_paths = collect_answer_citation_paths(enriched)
    citations_compact = [
        {
            "message_id": message.get("message_id"),
            "paths": [
                str(item.get("path") or item.get("relative_path") or item).replace("\\", "/")
                for item in (message.get("citations") or [])
                if str(item.get("path") if isinstance(item, dict) else item or "").strip()
            ],
        }
        for message in enriched.get("messages") or []
        if message.get("role") == "assistant" and message.get("citations")
    ]
    user_content = "\n\n".join(
        [
            "# Existing Profile",
            json.dumps(compact_profile_for_extraction(profile), ensure_ascii=False, indent=2),
            "",
            "# Session Context",
            json.dumps(answer_context_for_extraction(enriched), ensure_ascii=False, indent=2),
            "",
            "# Assistant Citations",
            json.dumps({"citation_paths": citation_paths[:20], "by_message": citations_compact[:12]}, ensure_ascii=False, indent=2),
            "",
            "# Transcript",
            render_session_transcript(enriched.get("messages") or []),
        ]
    )
    response = llm_client.complete(
        LLMRequest(
            model=model,
            messages=[
                LLMMessage(role="system", content=ANSWER_MEMORY_EXTRACTION_SYSTEM_PROMPT),
                LLMMessage(role="user", content=user_content),
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    )
    payload = parse_json_object(response.content)
    observations = normalize_observations(payload.get("observations") or [])
    return {"observations": observations}


def answer_context_for_extraction(session: dict[str, Any]) -> dict[str, Any]:
    context = session.get("context") or {}
    return {
        "session_id": session.get("session_id"),
        "created_at": session.get("created_at"),
        "scope_type": context.get("scope_type"),
        "scope_value": context.get("scope_value"),
        "scope_paths": context.get("scope_paths") or [],
        "strict_evidence": bool(context.get("strict_evidence")),
        "source_note_paths": context.get("source_note_paths") or [],
    }


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
                "scope": weak.get("scope", "domain"),
                "category": weak.get("category", "knowledge_gap"),
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
                    "question_requires": feedback.get("question_requires") or [],
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
                "context_note_paths": review.get("context_note_paths") or [],
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


def normalize_observations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    observations: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        obs_type = normalize_observation_type(item.get("type"))
        if not obs_type:
            continue
        point = str(item.get("point") or item.get("summary") or "").strip()
        evidence = str(item.get("evidence") or "").strip()
        title = str(item.get("title") or point).strip()
        left = str(item.get("left") or "").strip()
        right = str(item.get("right") or "").strip()
        if obs_type == "procedure" and not title and not item.get("steps"):
            continue
        if obs_type == "confusion_pair" and not (left and right):
            continue
        if obs_type not in {"procedure", "confusion_pair"} and not point and not evidence:
            continue
        observations.append(
            {
                "type": obs_type,
                "topic": str(item.get("topic") or "").strip(),
                "planned_layer": str(item.get("planned_layer") or "").strip(),
                "category": normalize_category(item.get("category")),
                "scope_suggestion": normalize_scope_suggestion(item.get("scope_suggestion") or item.get("scope")),
                "point": point,
                "weak_point_ref": str(item.get("weak_point_ref") or "").strip(),
                "evidence": evidence,
                "confidence": normalize_confidence(item.get("confidence")),
                "context_note_paths": [str(path) for path in item.get("context_note_paths", []) if str(path).strip()] if isinstance(item.get("context_note_paths"), list) else [],
                "procedure_key": str(item.get("procedure_key") or item.get("key") or "").strip(),
                "title": title,
                "steps": [str(step).strip() for step in item.get("steps", []) if str(step).strip()] if isinstance(item.get("steps"), list) else [],
                "left": left,
                "right": right,
                "distinction": str(item.get("distinction") or "").strip(),
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
        "possible_partial": "partial",
        "partial": "partial",
        "possible_improvement": "improvement",
        "improved": "improvement",
        "improvement": "improvement",
        "strong": "strong_point",
        "strong_point": "strong_point",
        "procedure": "procedure",
        "assistant_preference": "procedure",
        "preference": "procedure",
        "confusion": "confusion_pair",
        "confusion_pair": "confusion_pair",
    }
    return aliases.get(text, "")


def normalize_confidence(value: Any) -> str:
    text = str(value or "medium").strip().lower()
    return text if text in {"low", "medium", "high"} else "medium"


def observations_from_turn_reviews(reviews: list[dict[str, Any]]) -> list[dict[str, str]]:
    raw: list[dict[str, Any]] = []
    for review in reviews:
        signals = [signal for signal in (review.get("profile_signals") or []) if isinstance(signal, dict)]
        if signals:
            for signal in signals:
                raw.append(
                    {
                        "type": signal.get("type"),
                        "topic": signal.get("topic"),
                        "planned_layer": signal.get("planned_layer"),
                        "category": signal.get("category"),
                        "scope_suggestion": signal.get("scope_suggestion") or signal.get("scope"),
                        "point": signal.get("summary") or signal.get("weak_point_ref"),
                        "weak_point_ref": signal.get("weak_point_ref"),
                        "evidence": signal.get("evidence"),
                        "confidence": signal.get("confidence"),
                        "context_note_paths": signal.get("context_note_paths") or review.get("context_note_paths") or [],
                    }
                )
            continue
        feedback = review.get("feedback") or {}
        coach_note = str(feedback.get("coach_note") or "").strip()
        for gap in feedback.get("gaps") or []:
            point = str(gap or "").strip()
            if not point:
                continue
            raw.append(
                {
                    "type": "weak_point",
                    "topic": "",
                    "planned_layer": "",
                    "category": "knowledge_gap",
                    "scope_suggestion": "domain",
                    "point": point,
                    "weak_point_ref": "",
                    "evidence": coach_note or point,
                    "confidence": "medium",
                    "context_note_paths": review.get("context_note_paths") or [],
                }
            )
    return normalize_observations(raw)


def fallback_final_review(session: dict[str, Any], observations: list[dict[str, str]]) -> dict[str, Any]:
    weak = [item["point"] for item in observations if item["type"] == "weak_point" and item.get("point")]
    improved = [item["point"] for item in observations if item["type"] == "improvement" and item.get("point")]
    return {
        "summary": "本次面试已归档。长期画像抽取使用每轮复盘反馈降级完成。",
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
    operations = {"added": [], "updated": [], "partial": [], "improved": [], "strong_added": [], "strong_updated": []}
    today = date.today().isoformat()
    session_id = str(session.get("session_id") or "")

    for observation in observations:
        obs_type = observation["type"]
        confidence = observation.get("confidence") or "medium"
        if confidence == "low":
            continue
        if obs_type == "weak_point":
            observation = prepare_observation_for_write(observation, session)
            index = find_similar_weak_point(updated.get("weak_points", []), observation=observation)
            if index is None:
                weak = new_weak_point(
                    observation=observation,
                    session=session,
                    session_id=session_id,
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
        elif obs_type == "partial":
            index = find_improvement_target(updated.get("weak_points", []), observation)
            if index is not None:
                weak = updated["weak_points"][index]
                mark_weak_point_partial(weak, observation, session_id=session_id, today=today)
                operations["partial"].append(weak.get("point"))
        elif obs_type == "strong_point":
            upsert_strong_point(updated, observation, session_id=session_id, today=today, operations=operations)

    merge_communication(updated, communication or {})
    upgraded = promote_repeated_domain_habits(updated)
    if upgraded:
        operations["scope_upgraded"] = upgraded
    updated = recompute_topic_mastery(updated)
    return updated, operations


def prepare_observation_for_write(observation: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(observation or {})
    from services.memory.types import normalize_facet as _normalize_facet
    facet = _normalize_facet(prepared.get("facet") or prepared.get("category"))
    prepared["category"] = "knowledge_gap" if facet == "knowledge" else "answer_structure"
    prepared["scope"] = normalize_scope(prepared.get("scope_suggestion") or prepared.get("scope"), facet=facet)
    prepared["domain_anchor"] = resolve_domain_anchor(observation=prepared, session=session)
    return prepared


def new_weak_point(
    *,
    observation: dict[str, Any],
    session: dict[str, Any],
    session_id: str,
    today: str,
) -> dict[str, Any]:
    context = session.get("context") or {}
    source_note_paths = list(context.get("source_note_paths") or [])
    return {
        "point": observation.get("point") or observation.get("evidence"),
        "topic": observation.get("topic", ""),
        "planned_layer": observation.get("planned_layer", ""),
        "scope": observation.get("scope", "domain"),
        "category": observation.get("category", "knowledge_gap"),
        "domain_anchor": observation.get("domain_anchor") or {},
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
            "last_outcome": "fail",
        },
    }


def update_weak_point(weak: dict[str, Any], observation: dict[str, Any], *, session_id: str, today: str) -> None:
    weak["times_seen"] = int(weak.get("times_seen", 0) or 0) + 1
    weak["last_seen"] = today
    if observation.get("planned_layer") and not weak.get("planned_layer"):
        weak["planned_layer"] = observation["planned_layer"]
    append_unique(weak.setdefault("source_session_ids", []), session_id, max_items=8)
    append_evidence(weak, observation.get("evidence", ""))
    sr = weak.setdefault("sr", {})
    if sr.get("last_outcome") == "fail":
        sr["ease_factor"] = round(max(1.3, float(sr.get("ease_factor", 2.5)) - 0.15), 2)
    else:
        sr["ease_factor"] = round(float(sr.get("ease_factor", 2.5)), 2)
    sr["repetitions"] = max(0, int(sr.get("repetitions", 0) or 0) - 1)
    sr["interval_days"] = 1
    sr["next_review"] = (date.today() + timedelta(days=1)).isoformat()
    sr["last_outcome"] = "fail"


def mark_weak_point_partial(weak: dict[str, Any], observation: dict[str, Any], *, session_id: str, today: str) -> None:
    weak["last_seen"] = today
    weak["improved"] = False
    append_unique(weak.setdefault("source_session_ids", []), session_id, max_items=8)
    append_evidence(weak, observation.get("evidence", ""))
    sr = weak.setdefault("sr", {})
    previous_interval = int(sr.get("interval_days", 1) or 1)
    if previous_interval <= 1:
        interval = 2
    elif previous_interval < 3:
        interval = 3
    else:
        interval = previous_interval
    sr.update(
        {
            "ease_factor": round(float(sr.get("ease_factor", 2.5)), 2),
            "repetitions": int(sr.get("repetitions", 0) or 0),
            "interval_days": interval,
            "next_review": (date.today() + timedelta(days=interval)).isoformat(),
            "last_outcome": "partial",
        }
    )


def advance_review_schedule(weak: dict[str, Any], observation: dict[str, Any], *, session_id: str, today: str) -> None:
    """Advance SM-2 after a successful review pass. Does not set improved."""
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
            "last_outcome": "pass",
        }
    )


def mark_weak_point_improved(weak: dict[str, Any], observation: dict[str, Any], *, session_id: str, today: str) -> None:
    """Interview-confirmed improvement: SM-2 pass plus improved flag after sustained success."""
    advance_review_schedule(weak, observation, session_id=session_id, today=today)
    sr = weak.get("sr") or {}
    repetitions = int(sr.get("repetitions", 0) or 0)
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


def find_similar_weak_point(items: list[dict[str, Any]], *, observation: dict[str, Any]) -> int | None:
    text_norm = normalize_match_text(observation.get("point") or "")
    if not text_norm:
        return None
    obs_scope = observation.get("scope") or "domain"
    obs_category = normalize_category(observation.get("category"))
    obs_anchor = normalize_domain_anchor(observation.get("domain_anchor"), topic=observation.get("topic"))
    best_index: int | None = None
    best_score = 0.0

    for index, item in enumerate(items):
        if item.get("improved"):
            continue
        item_scope = item.get("scope") or "domain"
        if item_scope != obs_scope:
            continue
        item_category = normalize_category(item.get("category"))
        if item_category != obs_category:
            continue
        candidate = normalize_match_text(str(item.get("point") or ""))
        if not candidate:
            continue
        score = max(
            SequenceMatcher(None, text_norm, candidate).ratio(),
            containment_score(text_norm, candidate),
        )
        if obs_scope == "universal":
            threshold = 0.72
        else:
            relevance = domain_relevance_between_anchors(item.get("domain_anchor") or {}, obs_anchor)
            if relevance in {"strong", "medium"}:
                threshold = 0.68
            elif is_legacy_topic_match(item, observation):
                threshold = 0.78
            else:
                threshold = 0.85
        if score >= threshold and score > best_score:
            best_score = score
            best_index = index
    return best_index


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
        if topic and item_topic and not topic_matches(item_topic, topic):
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


def resolve_domain_anchor(*, observation: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
    if observation.get("scope") == "universal":
        return {}
    topic_card = resolve_topic_card_from_session(
        session=session,
        current_topic=(session.get("interview_state") or {}).get("current_topic") or observation.get("topic"),
    )
    context = session.get("context") or {}
    plan_topic = str((session.get("interview_state") or {}).get("current_topic") or observation.get("topic") or "").strip()
    scope_path = resolve_scope_path(context)
    turn_context_paths = [str(path) for path in observation.get("context_note_paths", []) if str(path).strip()]
    if turn_context_paths and len(turn_context_paths) <= 3:
        context_note_paths = turn_context_paths
    else:
        card_paths = list((topic_card or {}).get("source_note_paths") or [])
        context_note_paths = [str(path) for path in (card_paths if len(card_paths) <= 3 else card_paths[:3]) if str(path).strip()]
    if not context_note_paths and not plan_topic and not scope_path:
        return {}
    return {
        "plan_topic": plan_topic,
        "context_note_paths": context_note_paths,
        "scope_path": scope_path,
    }


def resolve_scope_path(context: dict[str, Any]) -> str:
    source_type = str(context.get("source_type") or "").strip()
    source_value = str(context.get("source_value") or "").strip()
    if source_type in {"folder", "tag", "search"} and source_value:
        return source_value
    paths = [str(path) for path in (context.get("source_paths") or []) if str(path).strip()]
    if paths:
        return str(Path(paths[0]).parent).replace("\\", "/")
    note_paths = [str(path) for path in (context.get("source_note_paths") or []) if str(path).strip()]
    if note_paths:
        return str(Path(note_paths[0]).parent).replace("\\", "/")
    return ""


def resolve_topic_card_from_session(*, session: dict[str, Any], current_topic: Any) -> dict[str, Any] | None:
    plan = session.get("interview_plan") or {}
    topics = [topic for topic in plan.get("topics", []) if isinstance(topic, dict)]
    if not topics:
        return None
    target = str(current_topic or "").strip()
    if target:
        target_key = normalize_topic_key(target)
        for topic in topics:
            if normalize_topic_key(topic.get("name")) == target_key:
                return topic
        scored = [
            (SequenceMatcher(None, normalize_topic_key(topic.get("name")), target_key).ratio(), topic)
            for topic in topics
        ]
        best_score, best_topic = max(scored, key=lambda item: item[0])
        if best_score >= 0.6:
            return best_topic
    if len(topics) == 1:
        return topics[0]
    return None


def topic_card_from_plan(plan: InterviewPlan | None, current_topic: str | None) -> dict[str, Any] | None:
    if not plan or not plan.topics:
        return None
    if current_topic:
        target = normalize_topic_key(current_topic)
        for topic in plan.topics:
            if normalize_topic_key(topic.name) == target:
                return {"name": topic.name, "source_note_paths": list(topic.source_note_paths), "coverage": list(topic.coverage)}
        scored = [
            (SequenceMatcher(None, normalize_topic_key(topic.name), target).ratio(), topic)
            for topic in plan.topics
        ]
        best_score, best_topic = max(scored, key=lambda item: item[0])
        if best_score >= 0.6:
            return {"name": best_topic.name, "source_note_paths": list(best_topic.source_note_paths), "coverage": list(best_topic.coverage)}
    if len(plan.topics) == 1:
        topic = plan.topics[0]
        return {"name": topic.name, "source_note_paths": list(topic.source_note_paths), "coverage": list(topic.coverage)}
    return None


def domain_relevance_between_anchors(left: dict[str, Any], right: dict[str, Any]) -> str:
    left_paths = set(str(path) for path in (left or {}).get("context_note_paths", []) if str(path).strip())
    right_paths = set(str(path) for path in (right or {}).get("context_note_paths", []) if str(path).strip())
    if left_paths and right_paths and left_paths.intersection(right_paths):
        return "strong"
    left_scope = str((left or {}).get("scope_path") or "").strip()
    right_scope = str((right or {}).get("scope_path") or "").strip()
    left_topic = str((left or {}).get("plan_topic") or "").strip()
    right_topic = str((right or {}).get("plan_topic") or "").strip()
    if left_scope and right_scope and path_prefix_related(left_scope, right_scope):
        if not left_topic or not right_topic:
            return "medium"
        if SequenceMatcher(None, normalize_topic_key(left_topic), normalize_topic_key(right_topic)).ratio() >= 0.8:
            return "medium"
    return "none"


def domain_relevance_for_current(weak: dict[str, Any], *, current_topic_card: dict[str, Any] | None, current_topic: str | None) -> str:
    anchor = normalize_domain_anchor(weak.get("domain_anchor"), topic=weak.get("topic"))
    card_paths = set(str(path) for path in ((current_topic_card or {}).get("source_note_paths") or []) if str(path).strip())
    anchor_paths: set[str] = set()
    for key in ("context_note_paths", "source_note_paths"):
        anchor_paths.update(str(path) for path in (anchor.get(key) or weak.get("domain_anchor", {}).get(key) or []) if str(path).strip())
    if card_paths and anchor_paths and card_paths.intersection(anchor_paths):
        return "strong"
    if anchor_paths:
        return "none"
    if is_legacy_topic_match(weak, {"topic": current_topic}):
        return "medium"
    return "none"


def path_prefix_related(left: str, right: str) -> bool:
    a = left.replace("\\", "/").strip("/")
    b = right.replace("\\", "/").strip("/")
    return bool(a and b and (a == b or a.startswith(b + "/") or b.startswith(a + "/")))


def is_legacy_topic_match(item: dict[str, Any], observation: dict[str, Any]) -> bool:
    anchor = item.get("domain_anchor") or {}
    if anchor.get("context_note_paths"):
        return False
    return topic_matches(item.get("topic"), observation.get("topic"))


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


def promote_repeated_domain_habits(profile: dict[str, Any]) -> list[str]:
    upgraded: list[str] = []
    weak_points = profile.get("weak_points", [])
    whitelist = {"answer_structure", "thinking_pattern", "communication"}
    for left_index, left in enumerate(list(weak_points)):
        if left.get("improved") or left.get("scope") != "domain":
            continue
        category = normalize_category(left.get("category"))
        if category not in whitelist:
            continue
        left_text = normalize_match_text(left.get("point") or "")
        if not left_text:
            continue
        for right in weak_points[left_index + 1 :]:
            if right.get("improved") or right.get("scope") != "domain":
                continue
            if normalize_category(right.get("category")) != category:
                continue
            right_text = normalize_match_text(right.get("point") or "")
            score = max(
                SequenceMatcher(None, left_text, right_text).ratio(),
                containment_score(left_text, right_text),
            )
            if score < 0.85:
                continue
            if domain_relevance_between_anchors(left.get("domain_anchor") or {}, right.get("domain_anchor") or {}) != "none":
                continue
            if not domains_are_distinct(left, right):
                continue
            left["scope"] = "universal"
            left["times_seen"] = int(left.get("times_seen", 0) or 0) + int(right.get("times_seen", 0) or 0)
            for session_id in right.get("source_session_ids", []) or []:
                append_unique(left.setdefault("source_session_ids", []), session_id, max_items=8)
            append_evidence(left, right.get("evidence", ""))
            also_seen = left.setdefault("also_seen_in", [])
            also_seen.append(
                {
                    "topic": right.get("topic", ""),
                    "domain_anchor": right.get("domain_anchor") or {},
                    "point": right.get("point", ""),
                }
            )
            right["improved"] = True
            right["improved_at"] = date.today().isoformat()
            upgraded.append(left.get("point", ""))
            break
    return upgraded


def domains_are_distinct(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_anchor = left.get("domain_anchor") or {}
    right_anchor = right.get("domain_anchor") or {}
    if domain_relevance_between_anchors(left_anchor, right_anchor) != "none":
        return False
    left_scope = str(left_anchor.get("scope_path") or "").strip()
    right_scope = str(right_anchor.get("scope_path") or "").strip()
    if left_scope and right_scope and left_scope != right_scope:
        return True
    left_topic = normalize_topic_key(left_anchor.get("plan_topic") or left.get("topic"))
    right_topic = normalize_topic_key(right_anchor.get("plan_topic") or right.get("topic"))
    if left_topic and right_topic and SequenceMatcher(None, left_topic, right_topic).ratio() < 0.6:
        return True
    return False


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
        if not is_injectable_weak_point(weak):
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
    current_topic_card = topic_card_from_plan(plan, topic)
    universal_weak, domain_weak, _other_domain_weak = split_weak_points_for_current(
        profile,
        current_topic=topic,
        current_topic_card=current_topic_card,
    )
    strong = [
        item for item in profile.get("strong_points", [])
        if topic_matches(item.get("topic"), topic)
    ][:4]

    lines: list[str] = [
        "## Candidate Profile Background",
        "Private background from prior interview sessions. These notes are not interview tasks.",
        "",
        f"Current topic: {topic or '(unknown)'}",
    ]
    if domain_weak:
        lines.extend(["", "Observed unstable areas in this topic:"])
        for weak in domain_weak[:4]:
            lines.append(f"- {weak.get('point')}")
    else:
        lines.extend(["", "Observed unstable areas in this topic: none recorded."])

    if universal_weak:
        lines.extend(["", "Cross-topic habits:"])
        for weak in universal_weak[:3]:
            lines.append(f"- {weak.get('point')}")

    if strong:
        lines.extend(["", "Recent strengths for this topic:"])
        for item in strong:
            lines.append(f"- {item.get('point')}")

    communication = profile.get("communication") or {}
    if communication.get("style"):
        lines.extend(["", "Communication notes:"])
        lines.append(f"- style: {communication['style']}")

    lines.extend(
        [
            "",
            "## Profile Use Boundary",
            "Use this profile only as quiet background while you conduct the interview.",
            "Let the user's current answer and the interview plan drive the next question.",
            "Do not force these notes into the conversation, and do not chase them as checklist items.",
            "Do not mention prior sessions, profile notes, weak points, scores, review history, or spaced-review status.",
            "If the conversation naturally reaches one of these areas, use the note only to judge whether the current answer is clearer than before.",
        ]
    )
    return "\n".join(lines)


def build_candidate_profile_debug(
    *,
    profile: dict[str, Any] | None,
    current_topic: str | None,
    plan: InterviewPlan | None,
) -> dict[str, Any]:
    topic = resolve_current_topic(current_topic=current_topic, plan=plan)
    current_topic_card = topic_card_from_plan(plan, topic)
    if not profile:
        return {
            "available": False,
            "current_topic": topic,
            "weak_points_count": 0,
            "due_reviews_count": 0,
            "strong_points_count": 0,
            "topic_mastery": None,
        }

    universal_weak, domain_weak, other_domain_weak = split_weak_points_for_current(
        profile,
        current_topic=topic,
        current_topic_card=current_topic_card,
    )
    due_matching, due_other = split_due_reviews(profile, topic, current_topic_card=current_topic_card)
    topic_weak = [*universal_weak, *domain_weak]
    strong = [
        item for item in profile.get("strong_points", [])
        if topic_matches(item.get("topic"), topic)
    ]
    mastery = find_topic_mastery(profile, topic)

    return {
        "available": True,
        "current_topic": topic,
        "weak_points_count": len(topic_weak),
        "due_reviews_count": len(due_matching),
        "other_due_reviews_count": len(due_other),
        "other_domain_weak_points_count": len(other_domain_weak),
        "strong_points_count": len(strong),
        "topic_mastery": {
            "active_weak_count": mastery.get("active_weak_count"),
            "mastery_estimate": mastery.get("mastery_estimate"),
            "last_assessed": mastery.get("last_assessed"),
        } if mastery else None,
        "weak_points": [
            profile_weak_point_for_agent(weak)
            for weak in topic_weak[:5]
        ],
        "due_reviews": [
            profile_weak_point_for_agent(weak)
            for weak in due_matching[:5]
        ],
        "universal_weak_points": [profile_weak_point_for_agent(weak) for weak in universal_weak[:5]],
        "domain_weak_points": [profile_weak_point_for_agent(weak) for weak in domain_weak[:8]],
        "domain_weak_by_layer": count_weak_points_by_layer(domain_weak),
    }


def build_profile_runtime_summary(
    *,
    profile: dict[str, Any] | None,
    current_topic: str | None,
    current_layer: str | None,
    plan: InterviewPlan | None,
    universal_limit: int = 3,
) -> dict[str, Any]:
    topic = resolve_current_topic(current_topic=current_topic, plan=plan)
    current_layer_key = normalize_planned_layer(current_layer)
    if not profile:
        return empty_profile_runtime_summary(topic=topic, current_layer=current_layer)

    current_topic_card = topic_card_from_plan(plan, topic)
    universal_weak, domain_weak, other_domain_weak = split_weak_points_for_current(
        profile,
        current_topic=topic,
        current_topic_card=current_topic_card,
    )
    due_matching, due_other = split_due_reviews(profile, topic, current_topic_card=current_topic_card)
    strong = [
        item for item in profile.get("strong_points", [])
        if topic_matches(item.get("topic"), topic)
    ]
    domain_by_layer = count_weak_points_by_layer(domain_weak)
    return {
        "profile_available": True,
        "current_topic": topic,
        "universal_weak_points": [
            profile_weak_point_for_agent(weak)
            for weak in sort_weak_points_for_agent(universal_weak)[:max(0, universal_limit)]
        ],
        "domain_weak_by_layer": domain_by_layer,
        "current_layer_domain_weak_count": int(domain_by_layer.get(current_layer_key, 0)) if current_layer_key else 0,
        "matching_weak_count": len(universal_weak) + len(domain_weak),
        "domain_weak_count": len(domain_weak),
        "other_domain_weak_points_count": len(other_domain_weak),
        "due_review_count": len(due_matching),
        "other_due_reviews_count": len(due_other),
        "strong_point_count": len(strong),
    }


def empty_profile_runtime_summary(*, topic: str | None, current_layer: str | None) -> dict[str, Any]:
    return {
        "profile_available": False,
        "current_topic": topic,
        "universal_weak_points": [],
        "domain_weak_by_layer": {},
        "current_layer_domain_weak_count": 0,
        "matching_weak_count": 0,
        "domain_weak_count": 0,
        "other_domain_weak_points_count": 0,
        "due_review_count": 0,
        "other_due_reviews_count": 0,
        "strong_point_count": 0,
    }


def profile_weak_point_for_agent(weak: dict[str, Any]) -> dict[str, Any]:
    return {
        "point": weak.get("point"),
        "topic": weak.get("topic", ""),
        "scope": weak.get("scope", "domain"),
        "category": weak.get("category", "knowledge_gap"),
        "planned_layer": weak.get("planned_layer") or "topic-level",
        "confidence": weak.get("confidence", "medium"),
        "evidence": weak.get("evidence", ""),
        "domain_anchor": weak.get("domain_anchor") or {},
        "sr": weak.get("sr") or {},
    }


def filter_weak_points_by_planned_layer(items: list[dict[str, Any]], planned_layer: str | None) -> list[dict[str, Any]]:
    layer_key = normalize_planned_layer(planned_layer)
    if not layer_key:
        return items
    return [item for item in items if normalize_planned_layer(item.get("planned_layer") or "topic-level") == layer_key]


def count_weak_points_by_layer(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        layer = normalize_planned_layer(item.get("planned_layer") or "topic-level") or "topic-level"
        counts[layer] = counts.get(layer, 0) + 1
    return counts


def normalize_planned_layer(value: Any) -> str:
    return str(value or "").strip() or "topic-level"


def sort_weak_points_for_agent(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    today = date.today().isoformat()
    confidence_score = {"high": 3, "medium": 2, "low": 1}

    def key(item: dict[str, Any]) -> tuple[int, int, float, str]:
        sr = item.get("sr") or {}
        due = str(sr.get("next_review") or "2000-01-01") <= today
        confidence = confidence_score.get(str(item.get("confidence") or "medium"), 2)
        ease = float(sr.get("ease_factor", 2.5))
        return (0 if due else 1, -confidence, ease, str(item.get("last_seen") or ""))

    return sorted(items, key=key)


def resolve_current_topic(*, current_topic: str | None, plan: InterviewPlan | None) -> str | None:
    if current_topic:
        return current_topic
    if plan and plan.topics:
        return plan.topics[0].name
    return None


def normalize_topic_key(value: Any) -> str:
    return "".join(str(value or "").lower().split())


def topic_matches(left: Any, right: Any) -> bool:
    left_key = normalize_topic_key(left)
    right_key = normalize_topic_key(right)
    return bool(left_key and right_key and left_key == right_key)


def find_topic_mastery(profile: dict[str, Any], current_topic: str | None) -> dict[str, Any]:
    if not current_topic:
        return {}
    target = normalize_topic_key(current_topic)
    for topic, mastery in (profile.get("topic_mastery") or {}).items():
        if normalize_topic_key(topic) == target:
            return mastery or {}
    return {}


def is_injectable_weak_point(weak: dict[str, Any]) -> bool:
    if weak.get("improved"):
        return False
    lifecycle = str(weak.get("lifecycle") or "active").strip().lower()
    return lifecycle == "active"


def split_weak_points_for_current(
    profile: dict[str, Any],
    *,
    current_topic: str | None,
    current_topic_card: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    universal: list[dict[str, Any]] = []
    domain: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    for weak in profile.get("weak_points", []):
        if not is_injectable_weak_point(weak):
            continue
        if weak.get("scope") == "universal":
            universal.append(weak)
            continue
        relevance = domain_relevance_for_current(
            weak,
            current_topic_card=current_topic_card,
            current_topic=current_topic,
        )
        if relevance in {"strong", "medium"}:
            domain.append(weak)
        else:
            other.append(weak)
    return sort_weak_points_for_prompt(universal), sort_weak_points_for_prompt(domain), other


def sort_weak_points_for_prompt(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    today = date.today().isoformat()

    def key(item: dict[str, Any]) -> tuple[int, float, str]:
        sr = item.get("sr") or {}
        due = str(sr.get("next_review") or "2000-01-01") <= today
        ease = float(sr.get("ease_factor", 2.5))
        return (0 if due else 1, ease, str(item.get("last_seen") or ""))

    return sorted(items, key=key)


def split_due_reviews(
    profile: dict[str, Any],
    current_topic: str | None,
    *,
    current_topic_card: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    today = date.today().isoformat()
    matching: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    for weak in profile.get("weak_points", []):
        if not is_injectable_weak_point(weak):
            continue
        sr = weak.get("sr") or {}
        next_review = str(sr.get("next_review") or "2000-01-01")
        if next_review > today:
            continue
        if weak.get("scope") == "universal":
            matching.append(weak)
        elif domain_relevance_for_current(
            weak,
            current_topic_card=current_topic_card,
            current_topic=current_topic,
        ) in {"strong", "medium"}:
            matching.append(weak)
        else:
            other.append(weak)
    key = lambda item: float((item.get("sr") or {}).get("ease_factor", 2.5))
    return sorted(matching, key=key), sorted(other, key=key)
