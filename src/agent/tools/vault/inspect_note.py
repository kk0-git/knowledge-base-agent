from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.schema import ToolSpec
from agent.tool_executor import ToolExecutionContext
from agent.tools.vault.note_sections import load_scoped_note_text
from services.markdown.sections import build_note_outline


def inspect_note_spec() -> ToolSpec:
    return ToolSpec(
        name="inspect_note",
        description="Inspect a note structure and section previews without reading full content.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "preview_chars": {"type": "integer", "description": "Max preview characters per section."},
                "max_sections": {"type": "integer", "description": "Max sections to return in the outline."},
                "reason": {"type": "string"},
            },
            "required": ["path"],
        },
        handler=inspect_note,
        timeout_s=10.0,
        side_effect="none",
    )


def inspect_note(arguments: dict[str, Any], ctx: ToolExecutionContext) -> dict[str, Any]:
    relative, text = load_scoped_note_text(str(arguments.get("path") or ""), ctx)
    preview_chars = max(40, min(int(arguments.get("preview_chars") or 120), 400))
    max_sections = max(1, min(int(arguments.get("max_sections") or 50), 100))
    reason = str(arguments.get("reason") or "").strip()
    output = build_note_outline(
        path=relative,
        title=Path(relative).stem,
        text=text,
        preview_chars=preview_chars,
        max_sections=max_sections,
    )
    output["reason"] = reason
    return output
