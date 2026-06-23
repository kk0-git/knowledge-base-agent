from __future__ import annotations

from dataclasses import asdict
from typing import Any

from agent.schema import ToolSpec
from agent.tool_executor import ToolExecutionContext


def online_search_spec() -> ToolSpec:
    return ToolSpec(
        name="online_search",
        description="Search the public web for information not available or insufficient in the vault. Only registered when online search is enabled.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        handler=online_search,
        timeout_s=30.0,
        side_effect="none",
    )


def online_search(arguments: dict[str, Any], ctx: ToolExecutionContext) -> dict[str, Any]:
    client = getattr(ctx, "online_search_client", None)
    if client is None:
        raise ValueError("online_search_client is not configured")
    query = str(arguments.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    options = ctx.metadata.get("retrieval_options") if isinstance(ctx.metadata, dict) else {}
    if not isinstance(options, dict):
        options = {}
    top_k = max(1, min(int(arguments.get("top_k") or options.get("online_top_k") or 5), 10))
    response = client.search(query=query, top_k=top_k)
    ctx.put_stats({"requested_top_k": top_k, "provider": response.provider})
    return {
        "query": query,
        "enabled": bool(response.enabled),
        "provider": response.provider,
        "message": response.message,
        "result_count": len(response.results),
        "results": [asdict(item) for item in response.results],
    }
