from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agent.schema import ToolSpec
from agent.tool_executor import ToolExecutionContext
from agent.tools.vault.guards import require_scope_allowed, resolve_vault_note_path, truncate_text


def read_note_spec() -> ToolSpec:
    return ToolSpec(
        name="read_note",
        description="Read a markdown note from the current vault scope.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "heading": {"type": "string"},
                "max_chars": {"type": "integer"},
            },
            "required": ["path"],
        },
        handler=read_note,
        timeout_s=10.0,
        side_effect="none",
    )


def read_note(arguments: dict[str, Any], ctx: ToolExecutionContext) -> dict[str, Any]:
    relative = require_scope_allowed(str(arguments.get("path") or ""), ctx.scope_note_paths)
    relative, full_path = resolve_vault_note_path(ctx.vault_root, relative)
    max_chars = max(1, min(int(arguments.get("max_chars") or 4000), ctx.max_tool_output_chars))
    heading = str(arguments.get("heading") or "").strip()
    text = full_path.read_text(encoding="utf-8", errors="replace")
    content = extract_heading_section(text, heading) if heading else text
    content = content.strip()
    truncated_content, truncated = truncate_text(content, max_chars)
    if relative not in ctx.working.notes_read_this_turn:
        ctx.working.notes_read_this_turn.append(relative)
    return {
        "path": relative,
        "title": Path(relative).stem,
        "heading": heading,
        "content": truncated_content,
        "char_count": len(content),
        "truncated": truncated,
    }


def extract_heading_section(text: str, heading: str) -> str:
    target = normalize_heading_text(heading)
    lines = text.splitlines()
    start_index: int | None = None
    start_level = 0
    for index, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            continue
        title = normalize_heading_text(match.group(2))
        if title == target:
            start_index = index
            start_level = len(match.group(1))
            break
    if start_index is None:
        return text
    end_index = len(lines)
    for index in range(start_index + 1, len(lines)):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", lines[index])
        if match and len(match.group(1)) <= start_level:
            end_index = index
            break
    return "\n".join(lines[start_index:end_index])


def normalize_heading_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())
