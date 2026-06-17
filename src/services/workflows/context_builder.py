from __future__ import annotations

from pathlib import Path
from typing import Any

from services.workflows.schema import ContextPack, ScopeResult


class ContextBuilder:
    def __init__(self, *, vault_root: Path, max_chars_per_note: int = 6000, max_context_chars: int = 30000) -> None:
        self.vault_root = vault_root
        self.max_chars_per_note = max_chars_per_note
        self.max_context_chars = max_context_chars

    def build(self, scope_result: ScopeResult, *, mode: str) -> ContextPack:
        if mode == "answer_context":
            return self.build_answer_context(scope_result)
        if mode == "wiki_context":
            return self.build_note_context(scope_result, mode=mode)
        if mode in {"audit_context", "review_context", "suggestion_context"}:
            return self.build_note_context(scope_result, mode=mode)
        return self.build_note_context(scope_result, mode=mode)

    def build_answer_context(self, scope_result: ScopeResult) -> ContextPack:
        if scope_result.chunks:
            sections: list[str] = []
            for item in scope_result.chunks:
                sections.append(render_chunk_item(item))
            context_text = "\n\n".join(sections)
            return ContextPack(
                mode="answer_context",
                scope_result=scope_result,
                context_text=truncate_text(context_text, self.max_context_chars),
                items=scope_result.chunks,
                citations=scope_result.chunks,
                stats={
                    "context_items": len(scope_result.chunks),
                    "context_chars": min(len(context_text), self.max_context_chars),
                },
            )
        return self.build_note_context(scope_result, mode="answer_context")

    def build_note_context(self, scope_result: ScopeResult, *, mode: str) -> ContextPack:
        items: list[dict[str, Any]] = []
        sections: list[str] = []
        total_chars = 0

        for index, note in enumerate(scope_result.notes, start=1):
            path = str(note["path"])
            full_path = self.vault_root / path
            if not full_path.exists():
                continue
            text = full_path.read_text(encoding="utf-8", errors="replace")
            excerpt = truncate_text(text.strip(), self.max_chars_per_note)
            item = {
                "citation_id": f"S{index}",
                "source_type": "source_note",
                "path": path,
                "title": note.get("title") or full_path.stem,
                "reason": note.get("reason", ""),
                "text": excerpt,
            }
            rendered = render_note_item(item)
            if total_chars + len(rendered) > self.max_context_chars and sections:
                break
            items.append(item)
            sections.append(rendered)
            total_chars += len(rendered)

        context_text = "\n\n".join(sections)
        return ContextPack(
            mode=mode,
            scope_result=scope_result,
            context_text=context_text,
            items=tuple(items),
            citations=tuple(items),
            stats={
                "context_items": len(items),
                "context_chars": len(context_text),
                "note_count": len(scope_result.notes),
            },
        )


def render_chunk_item(item: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"[{item.get('citation_id', '')}]",
            f"source: {item.get('path', '')}",
            f"heading: {item.get('heading') or '(no heading)'}",
            f"lines: {item.get('lines') or ''}",
            f"score: {item.get('score')} ({item.get('score_type')})",
            "content:",
            str(item.get("text", "")),
        ]
    )


def render_note_item(item: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"[{item.get('citation_id', '')}]",
            f"source: {item.get('path', '')}",
            f"title: {item.get('title', '')}",
            f"reason: {item.get('reason', '')}",
            "content:",
            str(item.get("text", "")),
        ]
    )


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"
