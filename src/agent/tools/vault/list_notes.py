from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.schema import ToolSpec
from agent.tool_executor import ToolExecutionContext
from agent.tools.vault.guards import VaultPathError, filter_items_by_scope, normalize_relative_path


def list_notes_spec() -> ToolSpec:
    return ToolSpec(
        name="list_notes",
        description="List markdown note paths within the current vault scope, optionally filtered by path or filename substring.",
        parameters={
            "type": "object",
            "properties": {
                "filter": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": [],
        },
        handler=list_notes,
        timeout_s=10.0,
        side_effect="none",
    )


def list_notes(arguments: dict[str, Any], ctx: ToolExecutionContext) -> dict[str, Any]:
    if ctx.vault_root is None:
        raise ValueError("vault_root is required for list_notes")
    limit = max(1, min(int(arguments.get("limit") or 80), 200))
    filter_text = str(arguments.get("filter") or "").replace("\\", "/").strip().lower()
    paths = candidate_note_paths(ctx.vault_root)
    scoped_paths = filter_items_by_scope(
        paths,
        ctx.scope_note_paths,
        lambda path: path,
        scope_type=ctx.scope_type,
    )
    if filter_text:
        scoped_paths = [path for path in scoped_paths if filter_text in path.lower()]
    total = len(scoped_paths)
    selected = sorted(scoped_paths)[:limit]
    return {
        "filter": filter_text,
        "scope_type": ctx.scope_type,
        "total": total,
        "result_count": len(selected),
        "truncated": total > len(selected),
        "notes": [
            {
                "path": path,
                "title": Path(path).stem,
            }
            for path in selected
        ],
        "source_paths": selected,
    }


def candidate_note_paths(vault_root: Path) -> list[str]:
    root = vault_root.resolve()
    paths: list[str] = []
    for path in root.rglob("*.md"):
        if not path.is_file():
            continue
        try:
            relative = path.resolve().relative_to(root).as_posix()
            paths.append(normalize_relative_path(relative))
        except (ValueError, VaultPathError):
            continue
    return paths
