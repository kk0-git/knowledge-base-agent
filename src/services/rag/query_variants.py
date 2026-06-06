from __future__ import annotations

import re
import string
from dataclasses import dataclass

from services.rag.query_rewrite import QueryRewrite
from services.rag.schema import SearchResult, TextChunk


@dataclass(frozen=True)
class QueryVariant:
    text: str
    source: str
    weight: float


@dataclass(frozen=True)
class RankedList:
    results: list[SearchResult]
    query_source: str
    retriever: str
    weight: float


def build_query_variants(
    original_query: str,
    rewrite: QueryRewrite | None = None,
    rewrite_confidence_threshold: float = 0.75,
    rewrite_weight: float = 0.7,
) -> list[QueryVariant]:
    variants = [
        QueryVariant(
            text=original_query,
            source="original",
            weight=1.0,
        )
    ]

    if rewrite is None:
        return variants

    if not rewrite.should_rewrite:
        return variants

    if rewrite.confidence < rewrite_confidence_threshold:
        return variants

    rewritten_query = rewrite.rewritten_query
    if not rewritten_query:
        return variants

    if not is_meaningful_rewrite(original_query, rewritten_query):
        return variants

    variants.append(
        QueryVariant(
            text=rewritten_query,
            source="rewrite",
            weight=rewrite_weight,
        )
    )
    return variants


def is_meaningful_rewrite(original_query: str, rewritten_query: str) -> bool:
    return normalize_for_rewrite_compare(original_query) != normalize_for_rewrite_compare(rewritten_query)


def normalize_for_rewrite_compare(text: str) -> str:
    punctuation = string.punctuation + "，。！？；：、“”‘’（）【】《》·…"
    pattern = "[" + re.escape(punctuation) + r"\s]+"
    return re.sub(pattern, "", text).lower()


def weighted_rrf_fuse(
    ranked_lists: list[RankedList],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    chunks: dict[str, TextChunk] = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list.results, start=1):
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + ranked_list.weight / (rrf_k + rank)
            chunks.setdefault(chunk_id, result.chunk)

    ranked_chunk_ids = sorted(
        scores,
        key=lambda chunk_id: scores[chunk_id],
        reverse=True,
    )[:top_k]

    return [
        SearchResult(
            chunk=chunks[chunk_id],
            score=round(scores[chunk_id], 6),
        )
        for chunk_id in ranked_chunk_ids
    ]
