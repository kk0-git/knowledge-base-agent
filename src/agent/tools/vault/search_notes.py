from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.schema import ToolSpec
from agent.tool_executor import ToolExecutionContext
from agent.tools.vault.guards import filter_items_by_scope, truncate_text


def search_notes_spec() -> ToolSpec:
    return ToolSpec(
        name="search_notes",
        description="Search candidate chunks in the current vault scope using hybrid semantic and BM25 retrieval.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        handler=search_notes,
        timeout_s=30.0,
        side_effect="none",
    )


def search_notes(arguments: dict[str, Any], ctx: ToolExecutionContext) -> dict[str, Any]:
    if ctx.rag_manager is None:
        if ctx.rag_manager_factory is None:
            raise ValueError("rag_manager is required for search_notes")
        ctx.rag_manager = ctx.rag_manager_factory()
    query = str(arguments.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    top_k = max(1, min(int(arguments.get("top_k") or 5), 20))
    raw_results = ctx.rag_manager.hybrid_search(
        query=query,
        top_k=max(top_k * 4, top_k),
        dense_top_k=50,
        bm25_top_k=50,
        rrf_k=60,
    )
    scoped_results = filter_items_by_scope(
        raw_results,
        ctx.scope_note_paths,
        lambda result: result.chunk.note_path,
    )[:top_k]
    hits: list[dict[str, Any]] = []
    truncated = False
    for result in scoped_results:
        chunk = result.chunk
        snippet, was_truncated = truncate_text(chunk.text.strip(), ctx.max_result_chars)
        truncated = truncated or was_truncated
        hits.append(
            {
                "path": chunk.note_path,
                "title": Path(chunk.note_path).stem,
                "heading": " > ".join(chunk.heading_path) if chunk.heading_path else "",
                "lines": line_range(chunk.start_line, chunk.end_line),
                "score": round(float(result.score), 6),
                "chunk_id": chunk.chunk_id,
                "snippet": snippet,
            }
        )
    return {
        "query": query,
        "result_count": len(hits),
        "hits": hits,
        "truncated": truncated,
        "source_paths": sorted({hit["path"] for hit in hits}),
    }


def line_range(start_line: int | None, end_line: int | None) -> str:
    if start_line is None or end_line is None:
        return ""
    return f"{start_line}-{end_line}"
