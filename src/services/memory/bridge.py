"""Bridge between learner memory v4 and legacy interview_profile views."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

from .schema import normalize_learner_model
from .types import normalize_scope, today_iso


def beliefs_to_weak_points(model: dict[str, Any]) -> list[dict[str, Any]]:
    weak_points: list[dict[str, Any]] = []
    for belief in (model.get("learner_items") or model.get("beliefs") or []):
        if not isinstance(belief, dict):
            continue
        weak = dict(belief)
        evidence_refs = list(weak.get("evidence_refs") or [])
        legacy_evidence = [str(ref.get("summary") or "") for ref in evidence_refs if str(ref.get("summary") or "").strip()]
        weak["evidence"] = legacy_evidence
        if not weak.get("id"):
            weak["id"] = weak.get("belief_id") or ""
        weak_points.append(weak)
    return weak_points


def learner_model_to_profile_view(model: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_learner_model(model)
    legacy = normalized.get("legacy") or {}
    communication = legacy.get("communication") if isinstance(legacy.get("communication"), dict) else {}
    topic_mastery = legacy.get("topic_mastery") if isinstance(legacy.get("topic_mastery"), dict) else {}
    return {
        "schema_version": 3,
        "updated_at": normalized.get("updated_at") or "",
        "weak_points": beliefs_to_weak_points(normalized),
        "strong_points": list(normalized.get("strong_points") or []),
        "communication": {
            "style": str(communication.get("style") or "").strip(),
            "suggestions": [
                str(item).strip()
                for item in (communication.get("suggestions") or [])
                if str(item).strip()
            ],
        },
        "topic_mastery": dict(topic_mastery),
        "_learner_model_revision": int(normalized.get("canonical_revision") or 0),
    }


def sync_profile_view_to_model(profile: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    updated = normalize_learner_model(model)
    if int(profile.get("schema_version") or 0) < 4 and profile.get("weak_points") is not None:
        from .migration import migrate_v3_profile_to_v4

        return migrate_v3_profile_to_v4(
            {
                "weak_points": profile.get("weak_points") or [],
                "strong_points": profile.get("strong_points") or [],
                "communication": profile.get("communication") or {},
                "topic_mastery": profile.get("topic_mastery") or {},
                "updated_at": profile.get("updated_at") or "",
            }
        )

    beliefs_by_id = {str(item.get("id") or ""): item for item in (updated.get("learner_items") or updated.get("beliefs") or []) if isinstance(item, dict)}
    for weak in profile.get("weak_points") or []:
        if not isinstance(weak, dict):
            continue
        belief_id = str(weak.get("id") or "").strip()
        belief = beliefs_by_id.get(belief_id)
        if belief is None:
            continue
        for key in ("point", "topic", "planned_layer", "scope", "category", "improved", "improved_at", "lifecycle"):
            if key in weak:
                belief[key] = weak.get(key)
        if weak.get("sr"):
            belief["sr"] = dict(weak.get("sr") or {})
        if weak.get("domain_anchor"):
            belief["domain_anchor"] = dict(weak.get("domain_anchor") or {})
        if weak.get("evidence_refs"):
            belief["evidence_refs"] = list(weak.get("evidence_refs") or [])
    if profile.get("strong_points") is not None:
        updated["strong_points"] = list(profile.get("strong_points") or [])
    legacy = dict(updated.get("legacy") or {})
    if profile.get("communication") is not None:
        legacy["communication"] = profile.get("communication") or {}
    if profile.get("topic_mastery") is not None:
        legacy["topic_mastery"] = profile.get("topic_mastery") or {}
    updated["legacy"] = legacy
    return normalize_learner_model(updated)


def collect_answer_citation_paths(session: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for message in session.get("messages") or []:
        if str(message.get("role") or "").strip() != "assistant":
            continue
        for citation in message.get("citations") or []:
            if isinstance(citation, dict):
                path = str(citation.get("path") or citation.get("relative_path") or "").strip()
            else:
                path = str(citation).strip()
            path = path.replace("\\", "/")
            if not path or path in seen:
                continue
            seen.add(path)
            paths.append(path)
    return paths


def enrich_answer_session_for_memory(session: dict[str, Any]) -> dict[str, Any]:
    enriched = deepcopy(session)
    context = dict(enriched.get("context") or {})
    citation_paths = collect_answer_citation_paths(enriched)
    scope_paths = [str(path).replace("\\", "/") for path in (context.get("scope_paths") or []) if str(path).strip()]
    source_note_paths = citation_paths or scope_paths
    context["source_note_paths"] = source_note_paths[:12]
    scope_type = str(context.get("scope_type") or "all").strip()
    scope_value = str(context.get("scope_value") or "").strip()
    if scope_type in {"folder", "tag", "search"} and scope_value:
        context["source_type"] = scope_type
        context["source_value"] = scope_value
    elif scope_paths:
        context["source_paths"] = scope_paths
    enriched["context"] = context
    return enriched


def resolve_anchor_from_citations(
    session: dict[str, Any],
    citation_paths: list[str],
    *,
    topic: str = "",
) -> tuple[str, dict[str, Any]]:
    if not citation_paths:
        return "universal", {}
    from services.workflows.interview_profile import normalize_domain_anchor

    context = session.get("context") or {}
    scope_value = str(context.get("scope_value") or "").strip()
    scope_path = [scope_value] if scope_value else [str(Path(citation_paths[0]).parent).replace("\\", "/")]
    anchor = normalize_domain_anchor(
        {
            "topic": topic,
            "scope_path": scope_path,
            "source_note_paths": citation_paths[:3],
        },
        topic=topic,
    )
    return "domain", anchor


def observations_from_answer_rules_fallback(session: dict[str, Any]) -> list[dict[str, Any]]:
    admission_pattern = re.compile(r"不懂|不会|不清楚|没搞懂|没理解|不太懂|不太会")
    observations: list[dict[str, Any]] = []
    for message in session.get("messages") or []:
        if str(message.get("role") or "").strip() != "user":
            continue
        content = str(message.get("content") or "").strip()
        if not content or not admission_pattern.search(content):
            continue
        snippet = content[:120]
        observations.append(
            {
                "type": "weak_point",
                "topic": "",
                "planned_layer": "",
                "category": "knowledge_gap",
                "scope_suggestion": "universal",
                "point": f"用户自承理解不足：{snippet}",
                "evidence": snippet,
                "confidence": "high",
                "explicit": True,
            }
        )
        break
    return observations


def observations_from_answer_extractor(
    observations: list[dict[str, Any]],
    *,
    session: dict[str, Any],
    model: dict[str, Any],
) -> list[dict[str, Any]]:
    from services.workflows.interview_profile import normalize_confidence, prepare_observation_for_write

    enriched = enrich_answer_session_for_memory(session)
    citation_paths = collect_answer_citation_paths(enriched)
    session_id = str(session.get("session_id") or "")
    mapped: list[dict[str, Any]] = []
    _ = model

    for raw in observations or []:
        if not isinstance(raw, dict):
            continue
        obs_type = str(raw.get("type") or "").strip()
        confidence = normalize_confidence(raw.get("confidence"))
        if confidence == "low":
            continue
        if obs_type in {"procedure", "assistant_preference", "preference"}:
            mapped.append(_procedure_observation_from_raw(raw, source_kind="answer", session_id=session_id, confidence=confidence))
            continue
        prepared = dict(raw)
        note_paths = [
            str(path).replace("\\", "/")
            for path in (prepared.get("context_note_paths") or citation_paths)
            if str(path).strip()
        ]
        scope, anchor = resolve_anchor_from_citations(
            enriched,
            note_paths[:3],
            topic=str(prepared.get("topic") or "").strip(),
        )
        prepared["scope_suggestion"] = scope
        prepared["context_note_paths"] = note_paths[:3]
        prepared = prepare_observation_for_write(prepared, enriched)
        if scope == "domain" and note_paths:
            from services.workflows.interview_profile import normalize_domain_anchor

            context = enriched.get("context") or {}
            prepared["domain_anchor"] = normalize_domain_anchor(
                {
                    "plan_topic": prepared.get("topic") or "",
                    "scope_path": str(context.get("scope_value") or Path(note_paths[0]).parent).replace("\\", "/"),
                    "context_note_paths": note_paths[:3],
                },
                topic=prepared.get("topic"),
            )
        if obs_type in {"confusion_pair", "confusion"}:
            item = {
                "op": "propose_belief",
                "kind": "confusion_pair",
                "source_kind": "answer",
                "confidence": confidence,
                "left": str(prepared.get("left") or "").strip(),
                "right": str(prepared.get("right") or "").strip(),
                "distinction": str(prepared.get("distinction") or prepared.get("point") or "").strip(),
                "point": str(prepared.get("point") or "").strip(),
                "topic": prepared.get("topic") or "",
                "scope": prepared.get("scope") or scope,
                "domain_anchor": prepared.get("domain_anchor") or {},
                "session_id": session_id,
                "evidence_summary": str(prepared.get("evidence") or prepared.get("distinction") or "").strip(),
                "source_note_paths": note_paths[:12],
            }
            if confidence == "high":
                item["target_lifecycle"] = "active"
            mapped.append(item)
            continue
        if obs_type not in {"weak_point", "possible_weak_point", "weak"}:
            continue
        item = {
            "op": "propose_belief",
            "source_kind": "answer",
            "confidence": confidence,
            "point": prepared.get("point") or prepared.get("evidence") or "",
            "topic": prepared.get("topic") or "",
            "planned_layer": prepared.get("planned_layer") or "",
            "category": prepared.get("category") or "knowledge_gap",
            "scope": prepared.get("scope") or scope,
            "domain_anchor": prepared.get("domain_anchor") or {},
            "session_id": session_id,
            "evidence_summary": str(prepared.get("evidence") or prepared.get("point") or "").strip(),
            "source_note_paths": note_paths[:12],
        }
        if confidence == "high":
            item["target_lifecycle"] = "active"
        if prepared.get("explicit"):
            item["explicit"] = True
        mapped.append(item)
    return mapped


def observation_user_commit(
    *,
    action: str,
    belief_id: str = "",
    procedure_id: str = "",
    target_type: str = "belief",
    note: str = "",
) -> dict[str, Any]:
    return {
        "op": "user_commit",
        "source_kind": "user",
        "confidence": "high",
        "action": action,
        "belief_id": belief_id,
        "procedure_id": procedure_id,
        "target_type": target_type,
        "note": note,
    }


def compute_session_evidence_hash(*, session: dict[str, Any], reviews: list[dict[str, Any]]) -> str:
    payload = {
        "session_id": str(session.get("session_id") or ""),
        "messages": session.get("messages") or [],
        "reviews": reviews or [],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_memory_extraction_checkpoint(
    *,
    session: dict[str, Any],
    reviews: list[dict[str, Any]],
    trigger: str,
    observation_count: int,
    filtered_low_count: int,
    commit_revision: int | None,
) -> dict[str, Any]:
    return {
        "extracted_at": session.get("updated_at") or today_iso(),
        "trigger": trigger,
        "evidence_hash": compute_session_evidence_hash(session=session, reviews=reviews),
        "observation_count": int(observation_count),
        "filtered_low_count": int(filtered_low_count),
        "commit_revision": commit_revision,
    }


def should_skip_memory_extraction(
    session: dict[str, Any],
    *,
    reviews: list[dict[str, Any]],
) -> bool:
    checkpoint = session.get("memory_extraction") or {}
    if not isinstance(checkpoint, dict):
        return False
    current_hash = compute_session_evidence_hash(session=session, reviews=reviews)
    return (
        str(checkpoint.get("evidence_hash") or "") == current_hash
        and checkpoint.get("commit_revision") is not None
    )


def observation_schedule_pass(*, belief_id: str, evidence_summary: str = "") -> dict[str, Any]:
    return {
        "op": "schedule_pass",
        "source_kind": "review",
        "confidence": "high",
        "belief_id": belief_id,
        "evidence_summary": evidence_summary or "review practice pass",
    }


def observation_schedule_retry(*, belief_id: str, evidence_summary: str = "") -> dict[str, Any]:
    return {
        "op": "schedule_retry",
        "source_kind": "review",
        "confidence": "high",
        "belief_id": belief_id,
        "evidence_summary": evidence_summary or "review practice fail",
    }


def observations_from_profile_extractor(
    observations: list[dict[str, Any]],
    *,
    session: dict[str, Any],
    model: dict[str, Any],
    communication: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    from services.workflows.interview_profile import (
        find_improvement_target,
        normalize_confidence,
        prepare_observation_for_write,
    )

    session_id = str(session.get("session_id") or "")
    mapped: list[dict[str, Any]] = []
    profile_view = learner_model_to_profile_view(model)

    for raw in observations or []:
        if not isinstance(raw, dict):
            continue
        obs_type = str(raw.get("type") or "").strip()
        confidence = normalize_confidence(raw.get("confidence"))
        if obs_type == "weak_point":
            if confidence == "low":
                continue
            prepared = prepare_observation_for_write(dict(raw), session)
            mapped.append(
                {
                    "op": "propose_belief",
                    "source_kind": "interview",
                    "confidence": confidence,
                    "point": prepared.get("point") or prepared.get("evidence") or "",
                    "topic": prepared.get("topic") or "",
                    "planned_layer": prepared.get("planned_layer") or "",
                    "category": prepared.get("category") or "knowledge_gap",
                    "scope": prepared.get("scope") or normalize_scope(prepared.get("scope_suggestion")),
                    "domain_anchor": prepared.get("domain_anchor") or {},
                    "session_id": session_id,
                    "evidence_summary": str(prepared.get("evidence") or prepared.get("point") or "").strip(),
                    "source_note_paths": list((session.get("context") or {}).get("source_note_paths") or [])[:12],
                }
            )
        elif obs_type == "confusion_pair":
            if confidence == "low":
                continue
            prepared = prepare_observation_for_write(dict(raw), session)
            mapped.append(
                {
                    "op": "propose_belief",
                    "kind": "confusion_pair",
                    "source_kind": "interview",
                    "confidence": confidence,
                    "left": str(prepared.get("left") or "").strip(),
                    "right": str(prepared.get("right") or "").strip(),
                    "distinction": str(prepared.get("distinction") or prepared.get("point") or "").strip(),
                    "point": str(prepared.get("point") or "").strip(),
                    "topic": prepared.get("topic") or "",
                    "scope": prepared.get("scope") or normalize_scope(prepared.get("scope_suggestion")),
                    "domain_anchor": prepared.get("domain_anchor") or {},
                    "session_id": session_id,
                    "evidence_summary": str(prepared.get("evidence") or prepared.get("distinction") or "").strip(),
                    "source_note_paths": list((session.get("context") or {}).get("source_note_paths") or [])[:12],
                }
            )
        elif obs_type == "procedure":
            if confidence == "low":
                continue
            mapped.append(_procedure_observation_from_raw(raw, source_kind="interview", session_id=session_id, confidence=confidence))
        elif obs_type == "improvement":
            if confidence == "low":
                continue
            index = find_improvement_target(profile_view.get("weak_points") or [], raw)
            if index is None:
                continue
            weak = profile_view["weak_points"][index]
            mapped.append(
                {
                    "op": "improvement",
                    "source_kind": "interview",
                    "confidence": confidence,
                    "belief_id": str(weak.get("id") or ""),
                    "evidence_summary": str(raw.get("evidence") or raw.get("point") or "").strip(),
                    "session_id": session_id,
                }
            )
    for suggestion in (communication or {}).get("suggestions") or []:
        text = str(suggestion or "").strip()
        if not text:
            continue
        mapped.append(
            _procedure_observation_from_raw(
                {
                    "title": text,
                    "steps": [text],
                    "evidence": "interview communication suggestion",
                    "procedure_key": _procedure_key_from_title(text),
                },
                source_kind="interview",
                session_id=session_id,
                confidence="medium",
            )
        )
    return mapped


def _procedure_observation_from_raw(
    raw: dict[str, Any],
    *,
    source_kind: str,
    session_id: str,
    confidence: str,
) -> dict[str, Any]:
    title = str(raw.get("title") or raw.get("point") or "").strip()
    steps = [str(step).strip() for step in raw.get("steps") or [] if str(step).strip()]
    if not steps and title:
        steps = [title]
    return {
        "op": "propose_procedure",
        "source_kind": source_kind,
        "confidence": confidence,
        "procedure_key": str(raw.get("procedure_key") or raw.get("key") or _procedure_key_from_title(title)).strip(),
        "title": title,
        "description": str(raw.get("description") or "").strip(),
        "steps": steps,
        "scope": "universal",
        "session_id": session_id,
        "evidence_summary": str(raw.get("evidence") or title).strip(),
    }


def _procedure_key_from_title(title: str) -> str:
    normalized = re.sub(r"\s+", "_", str(title or "").strip().lower())
    normalized = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", normalized).strip("_")
    return f"assistant_preference.{normalized[:48]}" if normalized else ""


def apply_legacy_profile_observations_to_model(
    model: dict[str, Any],
    observations: list[dict[str, Any]],
    *,
    session: dict[str, Any],
    communication: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from services.workflows.interview_profile import (
        find_improvement_target,
        mark_weak_point_partial,
        normalize_confidence,
    )

    updated = normalize_learner_model(deepcopy(model))
    profile_view = learner_model_to_profile_view(updated)
    operations = {
        "partial": [],
        "strong_added": [],
        "strong_updated": [],
        "communication_updated": False,
    }
    today = date.today().isoformat()
    session_id = str(session.get("session_id") or "")

    for raw in observations or []:
        if not isinstance(raw, dict):
            continue
        obs_type = str(raw.get("type") or "").strip()
        confidence = normalize_confidence(raw.get("confidence"))
        if confidence == "low":
            continue
        if obs_type == "partial":
            index = find_improvement_target(profile_view.get("weak_points") or [], raw)
            if index is None:
                continue
            weak_ref = profile_view["weak_points"][index]
            belief = _find_belief(updated, str(weak_ref.get("id") or ""))
            if belief is None:
                continue
            mark_weak_point_partial(belief, raw, session_id=session_id, today=today)
            operations["partial"].append(belief.get("point"))
        # strong_point writes are frozen (§6) — observations of type "strong_point" are silently ignored

    if communication:
        legacy = dict(updated.get("legacy") or {})
        comm = dict(legacy.get("communication") or {})
        style = str(communication.get("style") or "").strip()
        if style:
            comm["style"] = style
        suggestions = [
            str(item).strip()
            for item in (communication.get("suggestions") or [])
            if str(item).strip()
        ]
        if suggestions:
            existing = [str(item).strip() for item in comm.get("suggestions") or [] if str(item).strip()]
            comm["suggestions"] = existing + [item for item in suggestions if item not in existing]
        legacy["communication"] = comm
        updated["legacy"] = legacy
        operations["communication_updated"] = bool(style or suggestions)

    return normalize_learner_model(updated), operations


def apply_review_ui_retry(
    model: dict[str, Any],
    *,
    belief_id: str,
    today: str | None = None,
) -> dict[str, Any] | None:
    day = today or today_iso()
    belief = _find_belief(model, belief_id)
    if belief is None:
        return None
    belief["improved"] = False
    sr = dict(belief.get("sr") or {})
    sr["ease_factor"] = round(float(sr.get("ease_factor", 2.5) or 2.5), 2)
    sr["repetitions"] = int(sr.get("repetitions", 0) or 0)
    sr["interval_days"] = 1
    sr["next_review"] = day
    sr["last_outcome"] = "retry"
    sr["last_reviewed"] = day
    belief["sr"] = sr
    return belief


def find_belief_in_model(model: dict[str, Any], belief_id: str) -> dict[str, Any] | None:
    return _find_belief(model, belief_id)


def _find_belief(model: dict[str, Any], belief_id: str) -> dict[str, Any] | None:
    target = str(belief_id or "").strip()
    if not target:
        return None
    for belief in (model.get("learner_items") or model.get("beliefs") or []):
        if isinstance(belief, dict) and str(belief.get("id") or "") == target:
            return belief
    return None
