"""Commit engine for learner memory observations."""

from __future__ import annotations

import uuid
from copy import deepcopy
from datetime import date, timedelta
from difflib import SequenceMatcher
from typing import Any

from .schema import normalize_belief, normalize_evidence_refs, normalize_learner_model, normalize_procedure
from .types import (
    default_sr,
    normalize_belief_kind,
    normalize_confidence,
    normalize_facet,
    normalize_lifecycle,
    normalize_observation_op,
    normalize_scope,
    normalize_source_kind,
    today_iso,
    unique_strings,
    utc_now_iso,
)


def commit_observations(
    model: dict[str, Any],
    observations: list[dict[str, Any]],
    *,
    today: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    day = today or today_iso()
    updated = normalize_learner_model(deepcopy(model))
    initial_revision = int(updated.get("canonical_revision") or 0)
    operations = _default_operations(initial_revision)
    changed = False

    for raw_observation in observations or []:
        if not isinstance(raw_observation, dict):
            continue
        observation = dict(raw_observation)
        op = normalize_observation_op(observation.get("op") or observation.get("operation"))
        confidence = normalize_confidence(observation.get("confidence"))
        if confidence == "low":
            operations["filtered_low_count"] += 1
            continue

        if op == "propose_belief":
            changed |= _commit_propose_belief(updated, observation, day, operations)
        elif op == "propose_procedure":
            changed |= _commit_propose_procedure(updated, observation, day, operations)
        elif op in {"schedule_pass", "schedule_retry"}:
            changed |= _commit_schedule(updated, observation, day, op, operations)
        elif op == "improvement":
            changed |= _commit_improvement(updated, observation, day, operations)
        elif op == "user_commit":
            changed |= _commit_user_commit(updated, observation, day, operations)

    if changed:
        updated["canonical_revision"] = initial_revision + 1
        updated["updated_at"] = utc_now_iso()
        derived = updated.setdefault("derived", {})
        derived["stale"] = True
        operations["changed"] = True
        operations["canonical_revision"] = updated["canonical_revision"]
    return updated, operations


def _default_operations(revision: int) -> dict[str, Any]:
    return {
        "added_beliefs": 0,
        "updated_beliefs": 0,
        "archived_beliefs": 0,
        "added_procedures": 0,
        "updated_procedures": 0,
        "archived_procedures": 0,
        "filtered_low_count": 0,
        "schedule_updates": 0,
        "commitments_added": 0,
        "canonical_revision": revision,
        "changed": False,
    }


def _commit_propose_belief(
    model: dict[str, Any],
    observation: dict[str, Any],
    today: str,
    operations: dict[str, Any],
) -> bool:
    # Build preliminary belief to check for merge target (corroboration)
    preliminary = _belief_from_observation(observation, today, has_corroboration=False, is_explicit=bool(observation.get("explicit")))
    force_new_candidate = bool(observation.get("force_new_candidate"))
    target = None if force_new_candidate else _find_merge_target(model.get("learner_items", []), preliminary)
    has_corroboration = target is not None

    # Rebuild with correct lifecycle from evidence score
    new_belief = _belief_from_observation(
        observation, today,
        has_corroboration=has_corroboration,
        is_explicit=bool(observation.get("explicit")),
    )
    changed = False
    if target:
        changed |= _merge_into_belief(
            target,
            new_belief,
            today,
            observation_confidence=normalize_confidence(observation.get("confidence")),
        )
        operations["updated_beliefs"] += 1
    else:
        model.setdefault("learner_items", []).append(new_belief)
        operations["added_beliefs"] += 1
        changed = True

    if new_belief["lifecycle"] == "active":
        contradicted_id = str(observation.get("contradicts_belief_id") or "").strip()
        if contradicted_id:
            old = _find_belief_by_id(model, contradicted_id)
            current_id = target.get("id") if target else new_belief.get("id")
            if old and old.get("id") != current_id and old.get("lifecycle") == "active":
                old["lifecycle"] = "archived"
                _append_commitment(
                    model,
                    "superseded_by",
                    belief_id=old.get("id", ""),
                    target_id=current_id or "",
                    today=today,
                    note=str(observation.get("contradiction_note") or ""),
                )
                operations["archived_beliefs"] += 1
                operations["commitments_added"] += 1
                changed = True
    return changed


def _belief_from_observation(
    observation: dict[str, Any],
    today: str,
    *,
    has_corroboration: bool = False,
    is_explicit: bool = False,
) -> dict[str, Any]:
    source_kind = normalize_source_kind(observation.get("source_kind"))
    confidence = normalize_confidence(observation.get("confidence"))
    target_lifecycle = normalize_lifecycle(observation.get("target_lifecycle"), default="")

    # Evidence-score based lifecycle (§5)
    if target_lifecycle and source_kind in {"review", "user"}:
        # Review and user commits use explicit target_lifecycle
        lifecycle = target_lifecycle if target_lifecycle in {"active", "candidate"} else "candidate"
    else:
        score = _score_evidence(
            source_kind=source_kind,
            confidence=confidence,
            has_corroboration=has_corroboration,
            is_explicit=is_explicit,
        )
        lifecycle = _decide_lifecycle(score)

    kind = normalize_belief_kind(observation.get("kind"))

    evidence_refs = observation.get("evidence_refs")
    if not evidence_refs:
        evidence_summary = str(observation.get("evidence") or observation.get("evidence_summary") or "").strip()
        session_id = str(observation.get("session_id") or "").strip()
        evidence_refs = [
            {
                "source_kind": source_kind,
                "session_id": session_id,
                "turn_id": str(observation.get("turn_id") or ""),
                "review_run_id": str(observation.get("review_run_id") or ""),
                "card_id": str(observation.get("card_id") or ""),
                "at": today,
                "summary": evidence_summary,
            }
        ] if evidence_summary or session_id else []

    belief = {
        "id": str(observation.get("belief_id") or observation.get("id") or f"wp-{uuid.uuid4().hex[:12]}"),
        "kind": kind,
        "lifecycle": lifecycle,
        "point": str(observation.get("point") or "").strip(),
        "left": str(observation.get("left") or "").strip(),
        "right": str(observation.get("right") or "").strip(),
        "distinction": str(observation.get("distinction") or "").strip(),
        "category": normalize_facet(observation.get("category") or observation.get("facet")),
        "scope": normalize_scope(observation.get("scope")),
        "topic": str(observation.get("topic") or "").strip(),
        "planned_layer": str(observation.get("planned_layer") or "").strip(),
        "domain_anchor": observation.get("domain_anchor") or {},
        "source_note_paths": unique_strings(observation.get("source_note_paths") or []),
        "source_session_ids": unique_strings(observation.get("source_session_ids") or observation.get("session_id") or []),
        "source_kinds": [source_kind],
        "evidence_refs": normalize_evidence_refs(evidence_refs),
        "times_seen": 1,
        "first_seen": today,
        "last_seen": today,
        "improved": False,
        "improved_at": "",
        "sr": default_sr(today),
    }
    return normalize_belief(belief)


def _commit_propose_procedure(
    model: dict[str, Any],
    observation: dict[str, Any],
    today: str,
    operations: dict[str, Any],
) -> bool:
    new_procedure = _procedure_from_observation(observation, today)
    if not new_procedure.get("title") and not new_procedure.get("steps"):
        return False
    target = _find_procedure_merge_target(model.get("assistant_items", []), new_procedure)
    if target:
        changed = _merge_into_procedure(target, new_procedure, today)
        operations["updated_procedures"] += 1
        return changed
    model.setdefault("assistant_items", []).append(new_procedure)
    operations["added_procedures"] += 1
    return True


def _procedure_from_observation(observation: dict[str, Any], today: str) -> dict[str, Any]:
    source_kind = normalize_source_kind(observation.get("source_kind"))
    target_lifecycle = normalize_lifecycle(observation.get("target_lifecycle"), default="")
    lifecycle = target_lifecycle if source_kind == "user" and target_lifecycle == "active" else "candidate"
    evidence_refs = observation.get("evidence_refs")
    if not evidence_refs:
        evidence_summary = str(observation.get("evidence") or observation.get("evidence_summary") or "").strip()
        session_id = str(observation.get("session_id") or "").strip()
        evidence_refs = [
            {
                "source_kind": source_kind,
                "session_id": session_id,
                "turn_id": str(observation.get("turn_id") or ""),
                "review_run_id": str(observation.get("review_run_id") or ""),
                "card_id": str(observation.get("card_id") or ""),
                "at": today,
                "summary": evidence_summary,
            }
        ] if evidence_summary or session_id else []

    procedure = {
        "id": str(observation.get("procedure_id") or observation.get("id") or f"proc-{uuid.uuid4().hex[:12]}"),
        "procedure_key": str(observation.get("procedure_key") or observation.get("key") or "").strip(),
        "lifecycle": lifecycle,
        "scope": normalize_scope(observation.get("scope"), default="universal"),
        "title": str(observation.get("title") or observation.get("point") or "").strip(),
        "description": str(observation.get("description") or "").strip(),
        "steps": unique_strings(observation.get("steps") or []),
        "source_kinds": [source_kind],
        "source_session_ids": unique_strings(observation.get("source_session_ids") or observation.get("session_id") or []),
        "evidence_refs": normalize_evidence_refs(evidence_refs),
        "times_seen": 1,
        "first_seen": today,
        "last_seen": today,
    }
    return normalize_procedure(procedure)


# Evidence scoring weights for lifecycle decision (§5)
_W_SOURCE: dict[str, int] = {"interview": 3, "answer": 1, "review": 0, "user": 3}
_W_CONFIDENCE: dict[str, int] = {"high": 2, "medium": 1}
_W_CORROBORATION = 2
_W_EXPLICIT = 2
_T_ACTIVE = 4


def _score_evidence(
    *,
    source_kind: str,
    confidence: str,
    has_corroboration: bool = False,
    is_explicit: bool = False,
) -> int:
    w_source = _W_SOURCE.get(source_kind, 1)
    w_confidence = _W_CONFIDENCE.get(confidence, 1)
    w_corroboration = _W_CORROBORATION if has_corroboration else 0
    w_explicit = _W_EXPLICIT if is_explicit else 0
    return w_source + w_confidence + w_corroboration + w_explicit


def _decide_lifecycle(score: int) -> str:
    return "active" if score >= _T_ACTIVE else "candidate"


def _default_lifecycle(source_kind: str, confidence: str, target_lifecycle: str) -> str:
    """Deprecated — replaced by _score_evidence + _decide_lifecycle.
    Kept for backward compatibility with callers that haven't migrated."""
    if source_kind == "answer":
        if confidence == "high" and target_lifecycle == "active":
            return "active"
        return "candidate"
    if source_kind in {"review", "user"}:
        return target_lifecycle if target_lifecycle in {"active", "candidate"} else "candidate"
    if target_lifecycle == "candidate":
        return "candidate"
    return "active"


def _merge_into_belief(
    target: dict[str, Any],
    new_belief: dict[str, Any],
    today: str,
    *,
    observation_confidence: str = "medium",
) -> bool:
    before = deepcopy(target)
    target["times_seen"] = int(target.get("times_seen") or 0) + 1
    target["last_seen"] = today
    if target.get("lifecycle") == "candidate" and new_belief.get("lifecycle") == "active":
        target["lifecycle"] = "active"
    elif (
        target.get("lifecycle") == "candidate"
        and "interview" in (target.get("source_kinds") or [])
        and "answer" in (new_belief.get("source_kinds") or [])
        and observation_confidence in {"medium", "high"}
    ):
        target["lifecycle"] = "active"
    for key in ("source_note_paths", "source_session_ids", "source_kinds"):
        target[key] = unique_strings((target.get(key) or []) + (new_belief.get(key) or []))
    target["evidence_refs"] = _append_unique_evidence(target.get("evidence_refs") or [], new_belief.get("evidence_refs") or [])
    if target.get("kind") == "confusion_pair" and new_belief.get("distinction"):
        target["distinction"] = new_belief["distinction"]
        if new_belief.get("point"):
            target["point"] = new_belief["point"]
    return target != before


def _merge_into_procedure(target: dict[str, Any], new_procedure: dict[str, Any], today: str) -> bool:
    before = deepcopy(target)
    target["times_seen"] = int(target.get("times_seen") or 0) + 1
    target["last_seen"] = today
    if not target.get("procedure_key") and new_procedure.get("procedure_key"):
        target["procedure_key"] = new_procedure.get("procedure_key")
        target["key"] = new_procedure.get("procedure_key")
    if new_procedure.get("title") and not target.get("title"):
        target["title"] = new_procedure.get("title")
    if new_procedure.get("description"):
        target["description"] = new_procedure.get("description")
    target["steps"] = unique_strings((target.get("steps") or []) + (new_procedure.get("steps") or []))
    target["source_kinds"] = unique_strings((target.get("source_kinds") or []) + (new_procedure.get("source_kinds") or []))
    target["source_session_ids"] = unique_strings(
        (target.get("source_session_ids") or []) + (new_procedure.get("source_session_ids") or [])
    )
    target["evidence_refs"] = _append_unique_evidence(
        target.get("evidence_refs") or [],
        new_procedure.get("evidence_refs") or [],
    )
    return target != before


def _append_unique_evidence(existing: list[dict[str, Any]], new_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = list(existing)
    seen = {
        (
            str(item.get("source_kind") or ""),
            str(item.get("session_id") or ""),
            str(item.get("turn_id") or ""),
            str(item.get("review_run_id") or ""),
            str(item.get("card_id") or ""),
            str(item.get("summary") or ""),
        )
        for item in result
    }
    for item in new_items:
        key = (
            str(item.get("source_kind") or ""),
            str(item.get("session_id") or ""),
            str(item.get("turn_id") or ""),
            str(item.get("review_run_id") or ""),
            str(item.get("card_id") or ""),
            str(item.get("summary") or ""),
        )
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _commit_schedule(
    model: dict[str, Any],
    observation: dict[str, Any],
    today: str,
    op: str,
    operations: dict[str, Any],
) -> bool:
    belief = _find_belief_by_id(model, _target_belief_id(observation))
    if not belief:
        return False
    before = deepcopy(belief.get("sr") or {})
    sr = dict(belief.get("sr") or default_sr(today))
    if op == "schedule_pass":
        repetitions = max(0, int(sr.get("repetitions") or 0)) + 1
        sr["repetitions"] = repetitions
        sr["last_outcome"] = "pass"
        sr["last_reviewed"] = today
        current_interval = max(1, int(sr.get("interval_days") or 1))
        ease = max(1.3, float(sr.get("ease_factor") or 2.5))
        sr["interval_days"] = 1 if repetitions <= 1 else max(1, round(current_interval * ease))
    else:
        sr["repetitions"] = max(0, int(sr.get("repetitions") or 0) - 1)
        sr["last_outcome"] = "fail"
        sr["last_reviewed"] = today
        sr["interval_days"] = 1
    sr["next_review"] = _add_days(today, int(sr["interval_days"]))
    belief["sr"] = sr
    if before != sr:
        operations["schedule_updates"] += 1
        return True
    return False


def _commit_improvement(model: dict[str, Any], observation: dict[str, Any], today: str, operations: dict[str, Any]) -> bool:
    belief = _find_belief_by_id(model, _target_belief_id(observation))
    if not belief or belief.get("lifecycle") != "active":
        return False
    before = deepcopy(belief)
    sr = dict(belief.get("sr") or default_sr(today))
    sr["repetitions"] = max(0, int(sr.get("repetitions") or 0)) + 1
    sr["last_reviewed"] = today
    sr["last_outcome"] = "pass"
    sr["interval_days"] = max(1, int(sr.get("interval_days") or 1))
    sr["next_review"] = _add_days(today, sr["interval_days"])
    belief["sr"] = sr
    belief["improved"] = True
    belief["improved_at"] = today
    _append_commitment(
        model,
        "improved",
        belief_id=belief.get("id", ""),
        target_id="",
        today=today,
        note=str(observation.get("note") or ""),
    )
    operations["schedule_updates"] += 1
    operations["commitments_added"] += 1
    return belief != before or True


def _commit_user_commit(model: dict[str, Any], observation: dict[str, Any], today: str, operations: dict[str, Any]) -> bool:
    action = str(observation.get("action") or "").strip()
    if action in {"confirm_procedure", "set_procedure", "deny_procedure", "restore_procedure"}:
        return _commit_procedure_user_commit(model, observation, today, operations)

    belief = _find_belief_by_id(model, _target_belief_id(observation))
    if not belief:
        return False
    before = deepcopy(belief)
    if action == "confirm_candidate" and belief.get("lifecycle") == "candidate":
        belief["lifecycle"] = "active"
    elif action == "deny_belief" and belief.get("lifecycle") != "archived":
        belief["lifecycle"] = "archived"
        operations["archived_beliefs"] += 1
    elif action == "restore_archived" and belief.get("lifecycle") == "archived":
        belief["lifecycle"] = "active"
    else:
        return False
    _append_commitment(
        model,
        action,
        belief_id=belief.get("id", ""),
        target_id=str(observation.get("target_id") or ""),
        today=today,
        note=str(observation.get("note") or ""),
    )
    operations["commitments_added"] += 1
    operations["updated_beliefs"] += 1
    return belief != before


def _commit_procedure_user_commit(
    model: dict[str, Any],
    observation: dict[str, Any],
    today: str,
    operations: dict[str, Any],
) -> bool:
    action = str(observation.get("action") or "").strip()
    procedure = _find_procedure_by_id(model, _target_procedure_id(observation))
    if not procedure:
        return False
    before = deepcopy(procedure)
    if action in {"confirm_procedure", "set_procedure"} and procedure.get("lifecycle") == "candidate":
        procedure["lifecycle"] = "active"
    elif action == "deny_procedure" and procedure.get("lifecycle") != "archived":
        procedure["lifecycle"] = "archived"
        operations["archived_procedures"] += 1
    elif action == "restore_procedure" and procedure.get("lifecycle") == "archived":
        procedure["lifecycle"] = "active"
    else:
        return False
    _append_commitment(
        model,
        action,
        belief_id="",
        target_id=str(procedure.get("id") or ""),
        today=today,
        note=str(observation.get("note") or ""),
    )
    operations["commitments_added"] += 1
    operations["updated_procedures"] += 1
    return procedure != before


def _target_belief_id(observation: dict[str, Any]) -> str:
    return str(observation.get("belief_id") or observation.get("target_belief_id") or observation.get("id") or "").strip()


def _target_procedure_id(observation: dict[str, Any]) -> str:
    return str(
        observation.get("procedure_id")
        or observation.get("target_procedure_id")
        or observation.get("target_id")
        or observation.get("id")
        or ""
    ).strip()


def _find_belief_by_id(model: dict[str, Any], belief_id: str) -> dict[str, Any] | None:
    if not belief_id:
        return None
    for belief in model.get("learner_items", []):
        if belief.get("id") == belief_id:
            return belief
    return None


def _find_procedure_by_id(model: dict[str, Any], procedure_id: str) -> dict[str, Any] | None:
    if not procedure_id:
        return None
    for procedure in model.get("assistant_items", []):
        if procedure.get("id") == procedure_id:
            return procedure
    return None


def _find_merge_target(beliefs: list[dict[str, Any]], new_belief: dict[str, Any]) -> dict[str, Any] | None:
    if new_belief.get("kind") == "confusion_pair":
        new_key = _confusion_key(new_belief)
        for belief in beliefs:
            if belief.get("lifecycle") == "archived":
                continue
            if belief.get("kind") == "confusion_pair" and _confusion_key(belief) == new_key:
                return belief
        return None

    best: tuple[float, dict[str, Any] | None] = (0.0, None)
    for belief in beliefs:
        if belief.get("lifecycle") == "archived":
            continue
        if belief.get("kind") != "standard":
            continue
        if belief.get("scope") != new_belief.get("scope"):
            continue
        if (belief.get("facet") or belief.get("category")) != (new_belief.get("facet") or new_belief.get("category")):
            continue
        if new_belief.get("scope") == "domain" and not _domain_anchor_compatible(
            belief.get("domain_anchor") or {}, new_belief.get("domain_anchor") or {}
        ):
            continue
        ratio = SequenceMatcher(None, str(belief.get("point") or ""), str(new_belief.get("point") or "")).ratio()
        if ratio >= 0.72 and ratio > best[0]:
            best = (ratio, belief)
    return best[1]


def _find_procedure_merge_target(procedures: list[dict[str, Any]], new_procedure: dict[str, Any]) -> dict[str, Any] | None:
    new_key = str(new_procedure.get("procedure_key") or "").strip()
    if new_key:
        for procedure in procedures:
            if procedure.get("lifecycle") == "archived":
                continue
            if str(procedure.get("procedure_key") or "").strip() == new_key:
                return procedure

    new_title = str(new_procedure.get("title") or "").strip()
    if not new_title:
        return None
    best: tuple[float, dict[str, Any] | None] = (0.0, None)
    for procedure in procedures:
        if procedure.get("lifecycle") == "archived":
            continue
        title = str(procedure.get("title") or "").strip()
        if not title:
            continue
        ratio = SequenceMatcher(None, title, new_title).ratio()
        if ratio >= 0.72 and ratio > best[0]:
            best = (ratio, procedure)
    return best[1]


def _confusion_key(belief: dict[str, Any]) -> tuple[str, str]:
    return tuple(sorted([str(belief.get("left") or "").lower(), str(belief.get("right") or "").lower()]))  # type: ignore[return-value]


def _domain_anchor_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_notes = set(left.get("source_note_paths") or [])
    right_notes = set(right.get("source_note_paths") or [])
    if left_notes and right_notes and left_notes.intersection(right_notes):
        return True

    left_path = [str(item) for item in left.get("scope_path") or []]
    right_path = [str(item) for item in right.get("scope_path") or []]
    if left_path and right_path:
        if left_path == right_path:
            return True
        shortest = min(len(left_path), len(right_path))
        if shortest and left_path[:shortest] == right_path[:shortest]:
            return True

    left_topic = str(left.get("topic") or "").strip()
    right_topic = str(right.get("topic") or "").strip()
    if left_topic and right_topic and SequenceMatcher(None, left_topic, right_topic).ratio() >= 0.8:
        return True
    return False


def _append_commitment(
    model: dict[str, Any],
    action: str,
    *,
    belief_id: str,
    target_id: str,
    today: str,
    note: str,
) -> None:
    model.setdefault("commitments", []).append(
        {
            "id": f"commit-{uuid.uuid4().hex[:12]}",
            "action": action,
            "belief_id": belief_id,
            "target_id": target_id,
            "at": today,
            "note": note,
        }
    )


def _add_days(today: str, days: int) -> str:
    try:
        base = date.fromisoformat(today)
    except ValueError:
        base = date.today()
    return (base + timedelta(days=max(0, days))).isoformat()
