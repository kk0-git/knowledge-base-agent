from __future__ import annotations

from typing import Any

from agent.schema import ToolSpec
from agent.tool_executor import ToolExecutionContext
from agent.tools.vault.note_sections import (
    SHORT_NOTE_CHAR_THRESHOLD,
    build_full_read_output,
    build_outline_read_output,
    build_section_read_output,
    load_scoped_note_text,
    parse_heading_path_argument,
    resolve_target_section,
)
from services.markdown.sections import parse_markdown_sections


def read_note_spec() -> ToolSpec:
    return ToolSpec(
        name="read_note",
        description=(
            "Read a markdown note from the current vault scope. "
            "Short notes return full content; long notes return an outline unless a section is requested."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "heading": {"type": "string", "description": "Read a specific section by heading text."},
                "heading_path": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Read a section by full heading path, e.g. [\"Memory\", \"Types\"].",
                },
                "section_id": {"type": "string", "description": "Read a section by id from inspect_note or outline."},
                "max_chars": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["path"],
        },
        handler=read_note,
        timeout_s=10.0,
        side_effect="none",
    )


def read_note(arguments: dict[str, Any], ctx: ToolExecutionContext) -> dict[str, Any]:
    relative, text = load_scoped_note_text(str(arguments.get("path") or ""), ctx)
    max_chars = max(1, min(int(arguments.get("max_chars") or 4000), ctx.max_tool_output_chars))
    reason = str(arguments.get("reason") or "").strip()
    heading = str(arguments.get("heading") or "").strip()
    section_id = str(arguments.get("section_id") or "").strip()
    heading_path = parse_heading_path_argument(arguments.get("heading_path"))

    sections = parse_markdown_sections(text)
    if heading or section_id or heading_path:
        section = resolve_target_section(
            sections,
            heading=heading,
            section_id=section_id,
            heading_path=heading_path,
        )
        output = build_section_read_output(
            relative=relative,
            section=section,
            max_chars=max_chars,
            reason=reason,
            all_sections=sections,
        )
    elif len(text) <= SHORT_NOTE_CHAR_THRESHOLD:
        output = build_full_read_output(relative=relative, text=text, max_chars=max_chars, reason=reason)
    else:
        output = build_outline_read_output(relative=relative, text=text, reason=reason)

    if relative not in ctx.working.notes_read_this_turn:
        ctx.working.notes_read_this_turn.append(relative)
    return output
