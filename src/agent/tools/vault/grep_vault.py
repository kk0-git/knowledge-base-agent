from __future__ import annotations

from typing import Any

from agent.schema import ToolSpec
from agent.tool_executor import ToolExecutionContext
from agent.tools.vault.guards import filter_items_by_scope
from services.rag.grep_search import rg_search


def grep_vault_spec() -> ToolSpec:
    return ToolSpec(
        name="grep_vault",
        description="Search exact terms, commands, APIs, error codes, or filenames in markdown notes.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
                "ignore_case": {"type": "boolean"},
            },
            "required": ["query"],
        },
        handler=grep_vault,
        timeout_s=15.0,
        side_effect="none",
    )


def grep_vault(arguments: dict[str, Any], ctx: ToolExecutionContext) -> dict[str, Any]:
    if ctx.vault_root is None:
        raise ValueError("vault_root is required for grep_vault")
    query = str(arguments.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    limit = max(1, min(int(arguments.get("limit") or 10), 50))
    ignore_case = bool(arguments.get("ignore_case", True))
    raw_matches = rg_search(
        vault_root=ctx.vault_root,
        query=query,
        limit=max(limit * 4, limit),
        ignore_case=ignore_case,
    )
    scoped = filter_items_by_scope(raw_matches, ctx.scope_note_paths, lambda item: item.path)[:limit]
    matches = [
        {
            "path": item.path,
            "line": item.line,
            "text": item.text,
        }
        for item in scoped
    ]
    return {
        "query": query,
        "result_count": len(matches),
        "matches": matches,
        "source_paths": sorted({item["path"] for item in matches}),
    }
