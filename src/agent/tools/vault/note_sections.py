from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.tool_executor import ToolExecutionContext
from agent.tools.vault.guards import require_scope_allowed, resolve_vault_note_path
from services.markdown.sections import build_section_map, find_sections


DEFAULT_READ_MAX_CHARS = 4000


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


def char_slice_to_line_range(text: str, start: int, end: int) -> tuple[int, int]:
    if not text or start >= len(text):
        line = text.count("\n") + 1 if text else 1
        return line, line
    start_line = text.count("\n", 0, start) + 1
    end_line = text.count("\n", 0, max(start, end - 1)) + 1
    return start_line, end_line


def build_read_note_output(
    *,
    relative: str,
    text: str,
    sections: list,
    max_chars: int,
    offset: int,
    reason: str,
    section: Any | None = None,
) -> dict[str, Any]:
    if section is not None:
        window = "\n".join(section.lines)
        line_offset = section.start_line - 1
        active_section = section
    else:
        window = text
        line_offset = 0
        active_section = None

    window_char_count = len(window)
    total_char_count = len(text)
    start = max(0, int(offset))
    end = min(len(window), start + max_chars)
    content = window[start:end]
    truncated = end < len(window)
    returned_chars = len(content)

    rel_start, rel_end = char_slice_to_line_range(window, start, end)
    start_line = rel_start + line_offset
    end_line = rel_end + line_offset

    output: dict[str, Any] = {
        "path": relative,
        "title": Path(relative).stem,
        "reason": reason,
        "content": content,
        "truncated": truncated,
        "offset": start,
        "returned_chars": returned_chars,
        "window_char_count": window_char_count,
        "total_char_count": total_char_count,
        "start_line": start_line,
        "end_line": end_line,
        "section_id": active_section.section_id() if active_section else "",
        "heading": active_section.heading if active_section else "",
        "heading_path": list(active_section.heading_path) if active_section else [],
    }

    if truncated:
        output["next_offset"] = end
        output["sections"] = build_section_map(sections)
        if active_section is not None:
            output["hint"] = (
                f"Section content was truncated. Continue with offset={end}, "
                "or read another section_id from sections."
            )
        else:
            output["hint"] = (
                f"Note content was truncated. Continue with offset={end}, "
                "or jump to a section_id from sections."
            )

    return output
