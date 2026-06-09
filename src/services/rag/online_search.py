from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class OnlineSearchResult:
    title: str
    url: str
    snippet: str
    source: str = "online"


@dataclass(frozen=True)
class OnlineSearchResponse:
    enabled: bool
    provider: str
    results: list[OnlineSearchResult]
    message: str = ""


class OnlineSearchClient:
    """Small extension point for web search.

    The first agent version can keep online search disabled while preserving the
    command/context boundary. A concrete provider can be added without changing
    router or answer generation code.
    """

    def __init__(self, provider: str | None = None) -> None:
        self.provider = (provider or os.getenv("ONLINE_SEARCH_PROVIDER", "disabled")).strip().lower()

    def search(self, query: str, top_k: int = 5) -> OnlineSearchResponse:
        if self.provider in {"", "disabled", "off", "none"}:
            return OnlineSearchResponse(
                enabled=False,
                provider=self.provider or "disabled",
                results=[],
                message="online search is disabled",
            )

        if self.provider == "tavily":
            return search_tavily(query=query, top_k=top_k)

        if self.provider == "brave":
            return search_brave(query=query, top_k=top_k)

        return OnlineSearchResponse(
            enabled=False,
            provider=self.provider,
            results=[],
            message=f"online search provider is not implemented: {self.provider}",
        )


def online_response_to_dict(response: OnlineSearchResponse) -> dict:
    return {
        "enabled": response.enabled,
        "provider": response.provider,
        "message": response.message,
        "results": [asdict(result) for result in response.results],
    }


def search_tavily(query: str, top_k: int) -> OnlineSearchResponse:
    api_key = os.getenv("TAVILY_API_KEY") or os.getenv("ONLINE_SEARCH_API_KEY")
    if not api_key:
        return OnlineSearchResponse(
            enabled=False,
            provider="tavily",
            results=[],
            message="TAVILY_API_KEY or ONLINE_SEARCH_API_KEY is required",
        )

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": top_k,
        "search_depth": "basic",
        "include_answer": False,
    }
    request = urllib.request.Request(
        url="https://api.tavily.com/search",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        return OnlineSearchResponse(
            enabled=False,
            provider="tavily",
            results=[],
            message=f"tavily search failed: {exc}",
        )

    results = [
        OnlineSearchResult(
            title=str(item.get("title", "")),
            url=str(item.get("url", "")),
            snippet=str(item.get("content", "")),
            source="tavily",
        )
        for item in raw.get("results", [])[:top_k]
    ]
    return OnlineSearchResponse(enabled=True, provider="tavily", results=results)


def search_brave(query: str, top_k: int) -> OnlineSearchResponse:
    api_key = os.getenv("BRAVE_SEARCH_API_KEY") or os.getenv("ONLINE_SEARCH_API_KEY")
    if not api_key:
        return OnlineSearchResponse(
            enabled=False,
            provider="brave",
            results=[],
            message="BRAVE_SEARCH_API_KEY or ONLINE_SEARCH_API_KEY is required",
        )

    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(
        {"q": query, "count": min(top_k, 20)}
    )
    request = urllib.request.Request(
        url=url,
        headers={
            "X-Subscription-Token": api_key,
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        return OnlineSearchResponse(
            enabled=False,
            provider="brave",
            results=[],
            message=f"brave search failed: {exc}",
        )

    web_results = raw.get("web", {}).get("results", [])
    results = [
        OnlineSearchResult(
            title=str(item.get("title", "")),
            url=str(item.get("url", "")),
            snippet=str(item.get("description", "")),
            source="brave",
        )
        for item in web_results[:top_k]
    ]
    return OnlineSearchResponse(enabled=True, provider="brave", results=results)
