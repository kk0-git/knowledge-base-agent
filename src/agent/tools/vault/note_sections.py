from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.tool_executor import ToolExecutionContext
from agent.tools.vault.guards import require_scope_allowed, resolve_vault_note_path, truncate_text
from services.markdown.sections import (
    build_note_outline,
    child_sections,
    find_sections,
)


SHORT_NOTE_CHAR_THRESHOLD = 2500


def load_scoped_note_text(path: str, ctx: ToolExecutionContext) -> tuple[str, str]:
    relative = require_scope_allowed(path, ctx.scope_note_paths)
    relative, full_path = resolve_vault_note_path(ctx.vault_root, relative)
    text = full_path.read_text(encoding="utf-8", errors="replace")
    return relative, text


def parse_heading_path_argument(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    text = str(raw).strip()
    if not text:
        return []
    if ">" in text:
        return [part.strip() for part in text.split(">") if part.strip()]
    return [text]


def resolve_target_section(
    sections: list,
    *,
    heading: str,
    section_id: str,
    heading_path: list[str],
) -> Any:
    matches, error_code = find_sections(
        sections,
        heading=heading,
        section_id=section_id,
        heading_path=heading_path or None,
    )
    if error_code == "not_found":
        requested = section_id or " > ".join(heading_path) or heading
        raise ValueError(f"section not found: {requested}")
    if error_code == "ambiguous":
        candidates = [
            {
                "section_id": section.section_id(),
                "heading_path": section.heading_path,
                "start_line": section.start_line,
            }
            for section in matches
        ]
        raise ValueError(f"ambiguous heading '{heading}'; use heading_path or section_id. candidates={candidates}")
    return matches[0]


def build_section_read_output(
    *,
    relative: str,
    section,
    max_chars: int,
    reason: str,
    all_sections: list,
) -> dict[str, Any]:
    content = section.content
    truncated_content, truncated = truncate_text(content, max_chars)
    child_headings = [
        {
            "section_id": child.section_id(),
            "heading": child.heading,
            "heading_path": child.heading_path,
            "start_line": child.start_line,
            "end_line": child.end_line,
        }
        for child in child_sections(section, all_sections)
    ]
    return {
        "path": relative,
        "title": Path(relative).stem,
        "mode": "section",
        "section_id": section.section_id(),
        "heading": section.heading,
        "heading_path": list(section.heading_path),
        "reason": reason,
        "content": truncated_content,
        "start_line": section.start_line,
        "end_line": section.end_line,
        "char_count": section.char_count,
        "truncated": truncated,
        "child_headings": child_headings,
        "hint": "Section is still long; read a child section with section_id or heading_path."
        if truncated or child_headings
        else "",
    }


def build_full_read_output(*, relative: str, text: str, max_chars: int, reason: str) -> dict[str, Any]:
    content = text.strip()
    truncated_content, truncated = truncate_text(content, max_chars)
    return {
        "path": relative,
        "title": Path(relative).stem,
        "mode": "full",
        "reason": reason,
        "content": truncated_content,
        "char_count": len(content),
        "truncated": truncated,
    }


def build_outline_read_output(
    *,
    relative: str,
    text: str,
    reason: str,
    preview_chars: int = 120,
) -> dict[str, Any]:
    outline = build_note_outline(
        path=relative,
        title=Path(relative).stem,
        text=text,
        preview_chars=preview_chars,
    )
    outline["mode"] = "outline"
    outline["reason"] = reason
    outline["hint"] = (
        "Note is long; this is the outline only. "
        "Use read_note with heading, heading_path, or section_id to read specific sections."
    )
    return outline
