from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.workflows.review_practice import (
    build_grouped_review_prompt,
    find_weak_point,
    grouped_review_cards,
    read_review_card_cache,
    read_review_prompt_cache,
    write_review_card_cache,
    REVIEW_PROMPT_VERSION,
)
from services.workflows.review_runs import ReviewRunStore, empty_workspace, normalize_dialogue_run, utc_now_iso
from services.workflows.conversation_schema import (
    apply_assistant_completion,
    apply_assistant_failure,
    build_pending_assistant_message,
    build_user_message,
    extract_suggested_commits,
    find_message as schema_find_message,
    messages_for_agent_context,
    next_message_id,
    next_turn_id,
    normalize_session_messages,
    project_dialogue_history,
)


class ReviewRunService:
    def __init__(
        self,
        *,
        repository: Any,
        profile_store: Any,
        review_cache_dir: Path | str,
        project_root: Path | str,
        executor: Any,
    ) -> None:
        self.repository = repository
        self.profile_store = profile_store
        self.review_cache_dir = Path(review_cache_dir)
        self.project_root = Path(project_root)
        self.executor = executor
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def snapshot(self, review_run_id: str) -> dict[str, Any] | None:
        run = self._get_run(review_run_id)
        if run is None:
            return None
        return ReviewRunStore.snapshot_payload(run)

    def patch_workspace(self, review_run_id: str, workspace: dict[str, Any]) -> dict[str, Any]:
        run = self.repository.patch_workspace(review_run_id, workspace)
        with self._lock:
            self._cache[review_run_id] = run
        return ReviewRunStore.snapshot_payload(run)

    def create_plan_run(self, request: Any) -> dict[str, Any]:
        profile = self.profile_store.load()
        max_strategy_constraints = int(getattr(request, "max_strategy_constraints", 2) or 2)
        force_regenerate = bool(getattr(request, "force_regenerate", False))
        grouped = grouped_review_cards(
            profile,
            topics=getattr(request, "topics", []),
            limit=int(getattr(request, "limit", 12) or 12),
            max_strategy_constraints=max_strategy_constraints,
        )
        selected_topics = {str(topic).strip() for topic in grouped.get("selected_topics", []) if str(topic).strip()}
        limit = int(getattr(request, "limit", 12) or 12)
        cards: list[dict[str, Any]] = []
        card_weak_points: dict[str, list[dict[str, Any]]] = {}
        for card in grouped.get("cards", []):
            card_id = str(card.get("id") or "")
            weak_ids = [str(item).strip() for item in card.get("weak_point_ids") or [] if str(item).strip()]
            weak_group = [find_weak_point(profile, weak_id) for weak_id in weak_ids]
            weak_group = [weak for weak in weak_group if isinstance(weak, dict)]
            if not card_id or not weak_group:
                continue
            shell = dict(card)
            shell.pop("review_prompt", None)
            cache_key = str(shell.get("cache_key") or "")
            cached = None if force_regenerate else (read_review_card_cache(self.review_cache_dir, cache_key) if cache_key else None)
            if cached:
                review_prompt = cached.get("review_prompt") or cached
                shell.update(
                    {
                        "status": "ready",
                        "review_prompt": review_prompt,
                        "question_blocks": cached.get("question_blocks") or review_prompt.get("question_blocks") or [],
                        "reference_answer": cached.get("reference_answer") or review_prompt.get("reference_answer") or "",
                        "strategy_tips": cached.get("strategy_tips") or shell.get("strategy_tips") or [],
                        "cache_hit": True,
                    }
                )
            else:
                shell["status"] = "pending"
                shell["cache_hit"] = False
            cards.append(shell)
            card_weak_points[card_id] = json.loads(json.dumps(weak_group, ensure_ascii=False))
        review_run_id = f"review-{uuid.uuid4().hex[:12]}"
        ready_count = len([card for card in cards if card.get("status") == "ready"])
        pending_count = len([card for card in cards if card.get("status") == "pending"])
        now = utc_now_iso()
        run = dict(grouped)
        run.update(
            {
                "review_run_id": review_run_id,
                "type": "card",
                "status": "queued" if pending_count else "done",
                "cards": cards,
                "card_weak_points": card_weak_points,
                "due_count": sum(len(card.get("weak_point_ids") or []) for card in cards),
                "prepared_count": ready_count,
                "limit": limit,
                "selected_topics": sorted(selected_topics),
                "created_at": now,
                "updated_at": now,
                "finished_at": now if not pending_count else "",
                "prompt_version": REVIEW_PROMPT_VERSION,
                "workspace": empty_workspace(),
                "force_regenerate": force_regenerate,
                "messages": [],
            }
        )
        self._persist_run(run)
        if pending_count:
            self.executor.submit(self.generate_cards, review_run_id)
        return self.snapshot(review_run_id) or run

    def regenerate_card(self, review_run_id: str, card_id: str) -> dict[str, Any]:
        target = str(card_id or "").strip()
        if not target:
            raise ValueError("card_id is required")
        with self._lock:
            run = self._cache.get(review_run_id) or self.repository.load_run(review_run_id)
            if run is None:
                raise FileNotFoundError(f"review run not found: {review_run_id}")
            found = False
            for card in run.get("cards", []):
                if str(card.get("id") or card.get("card_id") or "") == target:
                    found = True
                    card["status"] = "pending"
                    card["cache_hit"] = False
                    card.pop("review_prompt", None)
                    card.pop("question_blocks", None)
                    card.pop("reference_answer", None)
                    card.pop("error", None)
                    break
            if not found:
                raise KeyError(f"review card not found: {target}")
            run["status"] = "queued"
            run["finished_at"] = ""
            run["prepared_count"] = len([card for card in run.get("cards", []) if card.get("status") == "ready"])
            run["updated_at"] = utc_now_iso()
            self._cache[review_run_id] = run
        self.repository.save_run(run)
        self.executor.submit(self.generate_cards, review_run_id)
        return self.snapshot(review_run_id) or run

    def create_dialogue_run(self, topics: list[str]) -> dict[str, Any]:
        cleaned = [str(topic).strip() for topic in topics if str(topic).strip()]
        if not cleaned:
            raise ValueError("topics is required")
        review_run_id = f"review-{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()
        run = {
            "review_run_id": review_run_id,
            "type": "dialogue",
            "status": "done",
            "topics": cleaned,
            "selected_topics": cleaned,
            "cards": [],
            "card_weak_points": {},
            "due_count": 0,
            "prepared_count": 0,
            "limit": 0,
            "messages": [],
            "created_at": now,
            "updated_at": now,
            "finished_at": now,
            "workspace": {
                "mode": "dialogue_review",
                "selectionState": None,
                "cardReviewState": None,
                "dialogueReviewState": {
                    "active": True,
                    "reviewRunId": review_run_id,
                    "topics": cleaned,
                    "history": [],
                    "pendingSuggestions": [],
                },
            },
        }
        self._persist_run(normalize_dialogue_run(run))
        return self.snapshot(review_run_id) or run

    def append_dialogue_pending_turn(self, review_run_id: str, user_content: str) -> dict[str, Any]:
        run = self._require_dialogue_run(review_run_id)
        now = utc_now_iso()
        messages = list(run.get("messages") or [])
        turn_id = next_turn_id(messages)
        user_message = build_user_message(
            message_id=next_message_id(messages),
            turn_id=turn_id,
            content=user_content,
            created_at=now,
        )
        assistant_message = build_pending_assistant_message(
            message_id=next_message_id([*messages, user_message]),
            turn_id=turn_id,
            created_at=now,
        )
        messages.extend([user_message, assistant_message])
        run["messages"] = messages
        run = self._finalize_dialogue_run(run)
        self._persist_run(run)
        return {"user_message": user_message, "assistant_message": assistant_message, "run": run}

    def complete_dialogue_assistant(
        self,
        review_run_id: str,
        *,
        assistant_message_id: str,
        assistant_content: str,
        agent_actions: list[dict[str, Any]] | None = None,
        trace_path: str = "",
        suggested_commits: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        run = self._require_dialogue_run(review_run_id)
        message = schema_find_message(run, assistant_message_id, role="assistant")
        now = utc_now_iso()
        commits = suggested_commits if suggested_commits is not None else extract_suggested_commits(agent_actions)
        apply_assistant_completion(
            message,
            assistant_content=assistant_content,
            updated_at=now,
            agent_actions=agent_actions,
        )
        meta = dict(message.get("meta") or {})
        if trace_path:
            meta["trace_path"] = trace_path
        if commits:
            meta["suggested_commits"] = commits
        message["meta"] = meta
        run = self._finalize_dialogue_run(run)
        self._persist_run(run)
        return {"assistant_message": message, "run": run, "suggested_commits": commits}

    def fail_dialogue_assistant(
        self,
        review_run_id: str,
        *,
        assistant_message_id: str,
        assistant_content: str = "",
        error_type: str = "Error",
        error_message: str = "",
        retryable: bool = True,
    ) -> dict[str, Any]:
        run = self._require_dialogue_run(review_run_id)
        message = schema_find_message(run, assistant_message_id, role="assistant")
        now = utc_now_iso()
        apply_assistant_failure(
            message,
            updated_at=now,
            assistant_content=assistant_content,
            error_type=error_type,
            error_message=error_message,
            retryable=retryable,
        )
        run = self._finalize_dialogue_run(run)
        self._persist_run(run)
        return {"assistant_message": message, "run": run}

    def prepare_dialogue_turn(
        self,
        review_run_id: str,
        *,
        message: str,
        assistant_message_id: str = "",
    ) -> dict[str, Any]:
        text = str(message or "").strip()
        if not text and not str(assistant_message_id or "").strip():
            raise ValueError("message is required")
        run = self._require_dialogue_run(review_run_id)
        if str(assistant_message_id or "").strip():
            pending = self._resolve_dialogue_pending_turn(run, assistant_message_id)
            user_message = pending["user_message"]
            assistant = pending["assistant_message"]
            user_text = str(user_message.get("content") or "")
        else:
            pending = self.append_dialogue_pending_turn(review_run_id, text)
            user_message = pending["user_message"]
            assistant = pending["assistant_message"]
            user_text = text
            run = pending["run"]
        topics = [str(topic).strip() for topic in run.get("topics") or run.get("selected_topics") or [] if str(topic).strip()]
        chat_history = messages_for_agent_context(
            run.get("messages") or [],
            before_message_id=str(user_message.get("id") or ""),
        )
        return {
            "run": run,
            "topics": topics,
            "user_message": user_message,
            "assistant_message": assistant,
            "user_text": user_text,
            "chat_history": chat_history,
        }

    def _require_dialogue_run(self, review_run_id: str) -> dict[str, Any]:
        run = self._get_run(review_run_id)
        if run is None:
            raise FileNotFoundError(f"review run not found: {review_run_id}")
        if str(run.get("type") or "") != "dialogue":
            raise ValueError("review run is not dialogue type")
        return normalize_dialogue_run(run)

    def _resolve_dialogue_pending_turn(self, run: dict[str, Any], assistant_message_id: str) -> dict[str, Any]:
        assistant = schema_find_message(run, assistant_message_id, role="assistant")
        if str(assistant.get("status") or "") != "pending":
            raise ValueError("assistant_message_id is not a pending message")
        messages = list(run.get("messages") or [])
        user_message: dict[str, Any] | None = None
        for index, message in enumerate(messages):
            if str(message.get("id") or "") == assistant_message_id and index > 0:
                previous = messages[index - 1]
                if str(previous.get("role") or "") == "user":
                    user_message = previous
                break
        if user_message is None:
            raise ValueError("pending assistant message has no paired user message")
        return {"user_message": user_message, "assistant_message": assistant}

    def _finalize_dialogue_run(self, run: dict[str, Any]) -> dict[str, Any]:
        run = normalize_session_messages(run)
        workspace = dict(run.get("workspace") or empty_workspace())
        dialogue = dict(workspace.get("dialogueReviewState") or {})
        messages = list(run.get("messages") or [])
        dialogue["history"] = project_dialogue_history(messages)
        dialogue["reviewRunId"] = str(run.get("review_run_id") or dialogue.get("reviewRunId") or "")
        commits: list[dict[str, Any]] = []
        for message in reversed(messages):
            if str(message.get("role") or "") != "assistant":
                continue
            meta = message.get("meta") or {}
            for item in meta.get("suggested_commits") or []:
                if isinstance(item, dict) and item.get("weak_point_id"):
                    commits.append(item)
            if commits:
                break
        dialogue["pendingSuggestions"] = commits
        workspace["dialogueReviewState"] = dialogue
        workspace["mode"] = "dialogue_review"
        run["workspace"] = workspace
        run["updated_at"] = utc_now_iso()
        return normalize_dialogue_run(run)

    def generate_cards(self, review_run_id: str) -> None:
        with self._lock:
            run = self._cache.get(review_run_id) or self.repository.load_run(review_run_id)
            if run is None:
                return
            run["status"] = "running"
            run["updated_at"] = utc_now_iso()
            card_ids = [str(card.get("id") or "") for card in run.get("cards", []) if card.get("status") == "pending"]
            card_weak_points = dict(run.get("card_weak_points") or {})
            self._cache[review_run_id] = run
        self.repository.save_run(run)

        llm_config = None
        llm_client = None
        try:
            from knowledge_base_agent.config import load_llm_config
            from knowledge_base_agent.llm import create_llm_client

            llm_config = load_llm_config(self.project_root)
            llm_client = create_llm_client(llm_config)
        except Exception:
            llm_config = None
            llm_client = None

        for card_id in card_ids:
            weak_group = list(card_weak_points.get(card_id) or [])
            if not weak_group:
                continue
            try:
                strategy_constraints: list[dict[str, Any]] = []
                cache_key = ""
                with self._lock:
                    run = self._cache.get(review_run_id)
                    if run is None:
                        return
                    for card in run.get("cards", []):
                        if str(card.get("id") or "") == card_id:
                            strategy_constraints = list(card.get("strategy_constraints") or [])
                            cache_key = str(card.get("cache_key") or "")
                            break
                if llm_client is not None and llm_config is not None:
                    prompt = build_grouped_review_prompt(
                        weak_group,
                        strategy_constraints=strategy_constraints,
                        llm_client=llm_client,
                        model=llm_config.model,
                        temperature=min(llm_config.temperature, 0.2),
                    )
                else:
                    prompt = build_grouped_review_prompt(weak_group, strategy_constraints=strategy_constraints)
                if cache_key:
                    write_review_card_cache(
                        self.review_cache_dir,
                        cache_key,
                        {
                            "review_prompt": prompt,
                            "question_blocks": prompt.get("question_blocks") or [],
                            "reference_answer": prompt.get("reference_answer") or "",
                            "strategy_tips": prompt.get("strategy_tips") or [],
                            "prompt_version": prompt.get("prompt_version") or "",
                        },
                    )
                with self._lock:
                    run = self._cache.get(review_run_id)
                    if run is None:
                        return
                    for card in run.get("cards", []):
                        if str(card.get("id") or "") == card_id:
                            card["review_prompt"] = prompt
                            card["question_blocks"] = prompt.get("question_blocks") or []
                            card["reference_answer"] = prompt.get("reference_answer") or ""
                            card["strategy_tips"] = prompt.get("strategy_tips") or card.get("strategy_tips") or []
                            card["status"] = "ready"
                            break
                    run["prepared_count"] = len([card for card in run.get("cards", []) if card.get("status") == "ready"])
                    run["updated_at"] = utc_now_iso()
                self.repository.save_run(run)
            except Exception as exc:
                with self._lock:
                    run = self._cache.get(review_run_id)
                    if run is None:
                        return
                    for card in run.get("cards", []):
                        if str(card.get("id") or "") == card_id:
                            card["status"] = "failed"
                            card["error"] = str(exc)
                            card["review_prompt"] = {
                                "prompt": card.get("point") or "请围绕这个知识弱点，用自己的话完整回答。",
                                "fallback_used": True,
                                "error": str(exc),
                            }
                            break
                    run["updated_at"] = utc_now_iso()
                self.repository.save_run(run)

        with self._lock:
            run = self._cache.get(review_run_id)
            if run is not None:
                run["status"] = "done"
                run["prepared_count"] = len([card for card in run.get("cards", []) if card.get("status") == "ready"])
                run["finished_at"] = utc_now_iso()
                run["updated_at"] = run["finished_at"]
        if run is not None:
            self.repository.save_run(run)

    def _get_run(self, review_run_id: str) -> dict[str, Any] | None:
        with self._lock:
            cached = self._cache.get(review_run_id)
            if cached is not None:
                return cached
        run = self.repository.load_run(review_run_id)
        if run is not None:
            with self._lock:
                self._cache[review_run_id] = run
        return run

    def _persist_run(self, run: dict[str, Any]) -> None:
        saved = self.repository.save_run(run)
        with self._lock:
            self._cache[str(run["review_run_id"])] = saved
