from __future__ import annotations

from typing import Any

from agent.schema import ToolSpec
from agent.tool_executor import ToolExecutionContext
from agent.tools.vault.note_sections import (
    DEFAULT_READ_MAX_CHARS,
    build_read_note_output,
    load_scoped_note_text,
    parse_heading_path_argument,
    resolve_target_section,
)
from services.markdown.sections import parse_markdown_sections


def read_note_spec() -> ToolSpec:
    return ToolSpec(
        name="read_note",
        description=(
            "Read markdown note content from the current vault scope. "
            "Always returns content. Use section_id to jump to a section, "
            "or offset to continue within the current reading window."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "section_id": {
                    "type": "string",
                    "description": "Read a specific section by id from a previous read_note sections map.",
                },
                "heading": {"type": "string", "description": "Read a specific section by heading text."},
                "heading_path": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Read a section by full heading path, e.g. [\"Memory\", \"Types\"].",
                },
                "offset": {
                    "type": "integer",
                    "description": "Character offset within the current reading window for pagination.",
                },
                "max_chars": {"type": "integer", "description": "Maximum characters to return. Default 4000."},
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
    max_chars = max(1, min(int(arguments.get("max_chars") or DEFAULT_READ_MAX_CHARS), ctx.max_tool_output_chars))
    offset = max(0, int(arguments.get("offset") or 0))
    reason = str(arguments.get("reason") or "").strip()
    heading = str(arguments.get("heading") or "").strip()
    section_id = str(arguments.get("section_id") or "").strip()
    heading_path = parse_heading_path_argument(arguments.get("heading_path"))

    sections = parse_markdown_sections(text)
    section = None
    if section_id or heading or heading_path:
        section = resolve_target_section(
            sections,
            heading=heading,
            section_id=section_id,
            heading_path=heading_path,
        )

    output = build_read_note_output(
        relative=relative,
        text=text,
        sections=sections,
        max_chars=max_chars,
        offset=offset,
        reason=reason,
        section=section,
    )

    if relative not in ctx.working.notes_read_this_turn:
        ctx.working.notes_read_this_turn.append(relative)
    return output
