from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Protocol

from services.rag.embedder import build_chunk_embedding_text
from services.rag.schema import SearchResult, TextChunk

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DASHSCOPE_RERANK_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"


class Reranker(Protocol):
    """Reranker interface for local cross-encoders or API-based rerankers."""

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Return reranked results sorted by reranker relevance score."""
        ...


def build_chunk_rerank_text(chunk: TextChunk) -> str:
    """Build the document side text for reranking.

    Reuse the embedding text format for the first version so reranker receives
    path, heading, and content. This keeps dense and rerank inputs comparable.
    """
    return build_chunk_embedding_text(chunk)


class CrossEncoderReranker:
    """Local sentence-transformers CrossEncoder reranker."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        batch_size: int = 16,
        max_length: int | None = 512,
        device: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        kwargs: dict[str, Any] = {}
        if max_length is not None:
            kwargs["max_length"] = max_length
        if device is not None:
            kwargs["device"] = device

        try:
            self.model = CrossEncoder(
                model_name,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        except TypeError:
            self.model = CrossEncoder(model_name, **kwargs)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        if not results:
            return []

        pairs = [
            (query, build_chunk_rerank_text(result.chunk))
            for result in results
        ]

        try:
            raw_scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except TypeError:
            raw_scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
            )

        scored_results = [
            (
                result,
                _to_float_score(raw_score),
            )
            for result, raw_score in zip(results, raw_scores, strict=False)
        ]

        scored_results.sort(key=lambda item: item[1], reverse=True)

        return [
            SearchResult(
                chunk=result.chunk,
                score=round(score, 6),
            )
            for result, score in scored_results[:top_k]
        ]


def _to_float_score(raw_score: Any) -> float:
    """Normalize model score output to a float."""
    if hasattr(raw_score, "item"):
        try:
            return float(raw_score.item())
        except ValueError:
            pass

    if isinstance(raw_score, (list, tuple)):
        if not raw_score:
            return 0.0
        if len(raw_score) == 1:
            return _to_float_score(raw_score[0])
        return _to_float_score(raw_score[-1])

    if hasattr(raw_score, "tolist"):
        return _to_float_score(raw_score.tolist())

    return float(raw_score)


class DashScopeReranker:
    """Alibaba DashScope API reranker (Qwen3-Rerank).

    Requires DASHSCOPE_API_KEY environment variable.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "qwen3-rerank",
        timeout: int = 30,
    ) -> None:
        self.model_name = model_name
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY is required. Set it as an environment variable "
                "or pass api_key= to DashScopeReranker()."
            )

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        if not results:
            return []

        documents = [build_chunk_rerank_text(result.chunk) for result in results]
        truncated_docs = [doc[:4000] for doc in documents]

        body = json.dumps({
            "model": self.model_name,
            "query": query,
            "documents": truncated_docs,
            "top_n": min(top_k, len(truncated_docs)),
        }).encode("utf-8")

        request = urllib.request.Request(
            DASHSCOPE_RERANK_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8") if exc.fp else ""
            raise RuntimeError(
                f"DashScope rerank API error {exc.code}: {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"DashScope rerank API network error: {exc.reason}"
            ) from exc

        rerank_results = data.get("results", [])
        if not rerank_results:
            return results[:top_k]

        scored_results = [
            (
                results[item["index"]],
                float(item["relevance_score"]),
            )
            for item in rerank_results
        ]

        scored_results.sort(key=lambda item: item[1], reverse=True)

        return [
            SearchResult(
                chunk=result.chunk,
                score=round(score, 6),
            )
            for result, score in scored_results[:top_k]
        ]
