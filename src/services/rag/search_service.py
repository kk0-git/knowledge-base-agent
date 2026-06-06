from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from knowledge_base_agent.config import load_dotenv, load_llm_config
from knowledge_base_agent.llm import create_llm_client
from services.rag.bm25 import BM25Index
from services.rag.chunker import HeadingChunker
from services.rag.embedder import Embedder, create_embedder
from services.rag.manager import RAGManager
from services.rag.memory_vector_store import MemoryVectorStore
from services.rag.query_rewrite import LLMQueryRewriter, QueryRewrite
from services.rag.query_variants import QueryVariant, RankedList, build_query_variants, weighted_rrf_fuse
from services.rag.reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker, DashScopeReranker
from services.rag.schema import SearchResult


SEARCH_MODES = {"dense", "bm25", "hybrid", "hybrid-rerank"}
RERANKER_TYPES = {"off", "local", "dashscope"}
DASHSCOPE_DEFAULT_RERANKER_MODEL = "qwen3-rerank"


@dataclass(frozen=True)
class SearchOptions:
    query: str
    mode: str = "hybrid"
    top_k: int = 10
    enable_rewrite: bool = False
    rewrite_confidence_threshold: float = 0.75
    rewrite_weight: float = 0.7
    dense_top_k: int = 50
    bm25_top_k: int = 50
    rrf_k: int = 60
    reranker_type: str = "off"
    reranker_model: str = DEFAULT_RERANKER_MODEL
    rerank_candidates: int = 50
    rerank_batch_size: int = 16
    rerank_max_length: int = 512
    include_debug: bool = True


@dataclass(frozen=True)
class SearchResponse:
    query: str
    mode: str
    elapsed_ms: int
    rewrite_enabled: bool
    rewrite_used: bool
    rewrite: dict[str, Any] | None
    variants: list[dict[str, Any]]
    results: list[dict[str, Any]]
    debug: dict[str, Any] | None = None


@dataclass(frozen=True)
class SearchDebugStage:
    key: str
    label: str
    query: str | None
    query_source: str | None
    retriever: str | None
    results: list[dict[str, Any]]


class SearchService:
    def __init__(
        self,
        index_path: Path,
        bm25_index_path: Path | None,
        model_name: str,
        project_root: Path,
        embedding_provider: str = "local",
        embed_batch_size: int = 32,
        max_seq_length: int | None = None,
    ) -> None:
        self.index_path = index_path
        self.bm25_index_path = bm25_index_path or derive_bm25_index_path(index_path)
        self.model_name = model_name
        self.project_root = project_root
        self.embedding_provider = embedding_provider
        self.embed_batch_size = embed_batch_size
        self.max_seq_length = max_seq_length

        load_dotenv(project_root / ".env")

        self._embedder: Embedder | None = None
        self._vector_store: MemoryVectorStore | None = None
        self._bm25_index: BM25Index | None = None
        self._rewriter: LLMQueryRewriter | None = None

    def search(self, options: SearchOptions) -> SearchResponse:
        started_at = time.perf_counter()
        validate_search_options(options)

        rewrite: QueryRewrite | None = None
        variants = [QueryVariant(text=options.query, source="original", weight=1.0)]

        if options.enable_rewrite and options.mode in {"hybrid", "hybrid-rerank"}:
            rewrite = self.get_rewriter().rewrite(options.query).rewrite
            variants = build_query_variants(
                original_query=options.query,
                rewrite=rewrite,
                rewrite_confidence_threshold=options.rewrite_confidence_threshold,
                rewrite_weight=options.rewrite_weight,
            )

        manager = self.build_manager(options)

        debug_stages: list[SearchDebugStage] = []

        if options.mode == "dense":
            results = manager.dense_search(query=options.query, top_k=options.top_k)
            debug_stages.append(
                build_debug_stage(
                    key="final",
                    label="最终结果",
                    query=options.query,
                    query_source="original",
                    retriever="dense",
                    results=results,
                )
            )
        elif options.mode == "bm25":
            results = manager.bm25_search(query=options.query, top_k=options.top_k)
            debug_stages.append(
                build_debug_stage(
                    key="final",
                    label="最终结果",
                    query=options.query,
                    query_source="original",
                    retriever="bm25",
                    results=results,
                )
            )
        elif options.mode == "hybrid":
            results, debug_stages = self.run_hybrid_with_debug(
                manager=manager,
                options=options,
                variants=variants,
                enable_rerank=False,
            )
        elif options.mode == "hybrid-rerank":
            results, debug_stages = self.run_hybrid_with_debug(
                manager=manager,
                options=options,
                variants=variants,
                enable_rerank=True,
            )
        else:
            raise ValueError(f"Unsupported search mode: {options.mode}")

        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return SearchResponse(
            query=options.query,
            mode=options.mode,
            elapsed_ms=elapsed_ms,
            rewrite_enabled=options.enable_rewrite,
            rewrite_used=len(variants) > 1,
            rewrite=asdict(rewrite) if rewrite else None,
            variants=[asdict(variant) for variant in variants],
            results=format_results(results),
            debug=build_debug_payload(debug_stages) if options.include_debug else None,
        )

    def run_hybrid_with_debug(
        self,
        manager: RAGManager,
        options: SearchOptions,
        variants: list[QueryVariant],
        enable_rerank: bool,
    ) -> tuple[list[SearchResult], list[SearchDebugStage]]:
        ranked_lists: list[RankedList] = []
        stages: list[SearchDebugStage] = []

        for variant in variants:
            dense_results = manager.dense_search(query=variant.text, top_k=options.dense_top_k)
            bm25_results = manager.bm25_search(query=variant.text, top_k=options.bm25_top_k)

            ranked_lists.append(
                RankedList(
                    results=dense_results,
                    query_source=variant.source,
                    retriever="dense",
                    weight=variant.weight,
                )
            )
            ranked_lists.append(
                RankedList(
                    results=bm25_results,
                    query_source=variant.source,
                    retriever="bm25",
                    weight=variant.weight,
                )
            )

            stages.append(
                build_debug_stage(
                    key=f"{variant.source}_dense",
                    label=f"{variant.source} dense",
                    query=variant.text,
                    query_source=variant.source,
                    retriever="dense",
                    results=dense_results,
                )
            )
            stages.append(
                build_debug_stage(
                    key=f"{variant.source}_bm25",
                    label=f"{variant.source} BM25",
                    query=variant.text,
                    query_source=variant.source,
                    retriever="bm25",
                    results=bm25_results,
                )
            )

        rrf_top_k = options.rerank_candidates if enable_rerank else options.top_k
        rrf_results = weighted_rrf_fuse(
            ranked_lists=ranked_lists,
            top_k=rrf_top_k,
            rrf_k=options.rrf_k,
        )
        stages.append(
            build_debug_stage(
                key="rrf",
                label="RRF 合并",
                query=None,
                query_source=None,
                retriever="rrf",
                results=rrf_results,
            )
        )

        if not enable_rerank:
            stages.insert(
                0,
                build_debug_stage(
                    key="final",
                    label="最终结果",
                    query=options.query,
                    query_source="original",
                    retriever="hybrid",
                    results=rrf_results,
                ),
            )
            return rrf_results, stages

        if manager.reranker is None:
            raise ValueError("reranker is required for hybrid rerank search")

        stages.append(
            build_debug_stage(
                key="rerank_candidates",
                label="Rerank 候选",
                query=options.query,
                query_source="original",
                retriever="rerank-candidates",
                results=rrf_results,
            )
        )
        reranked_results = manager.reranker.rerank(
            query=options.query,
            results=rrf_results,
            top_k=options.top_k,
        )
        stages.insert(
            0,
            build_debug_stage(
                key="final",
                label="最终结果",
                query=options.query,
                query_source="original",
                retriever="rerank",
                results=reranked_results,
            ),
        )
        return reranked_results, stages

    def build_manager(self, options: SearchOptions) -> RAGManager:
        needs_dense = options.mode in {"dense", "hybrid", "hybrid-rerank"}
        needs_bm25 = options.mode in {"bm25", "hybrid", "hybrid-rerank"}

        reranker = None
        if options.mode == "hybrid-rerank":
            reranker_model = resolve_reranker_model(options)
            if options.reranker_type == "dashscope":
                reranker = DashScopeReranker(model_name=reranker_model)
            elif options.reranker_type == "local":
                reranker = CrossEncoderReranker(
                    model_name=reranker_model,
                    batch_size=options.rerank_batch_size,
                    max_length=options.rerank_max_length,
                )
            else:
                raise ValueError("hybrid-rerank requires reranker_type local or dashscope")

        return RAGManager(
            chunker=HeadingChunker(),
            embedder=self.get_embedder() if needs_dense else None,
            vector_store=self.get_vector_store(),
            bm25_index=self.get_bm25_index() if needs_bm25 else None,
            reranker=reranker,
        )

    def get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = create_embedder(
                provider=self.embedding_provider,
                model_name=self.model_name,
                batch_size=self.embed_batch_size,
                max_seq_length=self.max_seq_length,
            )
        return self._embedder

    def get_vector_store(self) -> MemoryVectorStore:
        if self._vector_store is None:
            if not self.index_path.exists():
                raise FileNotFoundError(f"Vector index file not found: {self.index_path}")
            self._vector_store = MemoryVectorStore(persist_path=self.index_path)
        return self._vector_store

    def get_bm25_index(self) -> BM25Index:
        if self._bm25_index is None:
            if not self.bm25_index_path.exists():
                raise FileNotFoundError(f"BM25 index file not found: {self.bm25_index_path}")
            self._bm25_index = BM25Index(persist_path=self.bm25_index_path)
        return self._bm25_index

    def get_rewriter(self) -> LLMQueryRewriter:
        if self._rewriter is None:
            llm_config = load_llm_config(self.project_root)
            client = create_llm_client(llm_config)
            self._rewriter = LLMQueryRewriter(
                client=client,
                model=llm_config.model,
                temperature=llm_config.temperature,
            )
        return self._rewriter


def validate_search_options(options: SearchOptions) -> None:
    if not options.query.strip():
        raise ValueError("query is required")
    if options.mode not in SEARCH_MODES:
        raise ValueError(f"Unsupported search mode: {options.mode}")
    if options.reranker_type not in RERANKER_TYPES:
        raise ValueError(f"Unsupported reranker_type: {options.reranker_type}")
    if options.top_k <= 0:
        raise ValueError("top_k must be positive")
    if options.dense_top_k <= 0:
        raise ValueError("dense_top_k must be positive")
    if options.bm25_top_k <= 0:
        raise ValueError("bm25_top_k must be positive")
    if options.rerank_candidates <= 0:
        raise ValueError("rerank_candidates must be positive")
    if options.mode == "hybrid-rerank" and options.reranker_type == "off":
        raise ValueError("hybrid-rerank requires reranker_type local or dashscope")


def resolve_reranker_model(options: SearchOptions) -> str:
    if options.reranker_type == "dashscope" and options.reranker_model == DEFAULT_RERANKER_MODEL:
        return DASHSCOPE_DEFAULT_RERANKER_MODEL
    return options.reranker_model


def format_results(results: list[SearchResult]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        formatted.append(
            {
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "note_path": chunk.note_path,
                "title": Path(chunk.note_path).name,
                "heading_path": chunk.heading_path,
                "heading": " > ".join(chunk.heading_path) if chunk.heading_path else "",
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "score": result.score,
                "preview": build_preview(chunk.text),
                "text": chunk.text,
            }
        )
    return formatted


def build_debug_stage(
    key: str,
    label: str,
    query: str | None,
    query_source: str | None,
    retriever: str | None,
    results: list[SearchResult],
) -> SearchDebugStage:
    return SearchDebugStage(
        key=key,
        label=label,
        query=query,
        query_source=query_source,
        retriever=retriever,
        results=format_results(results),
    )


def build_debug_payload(stages: list[SearchDebugStage]) -> dict[str, Any]:
    return {
        "enabled": True,
        "stages": [asdict(stage) for stage in stages],
    }


def build_preview(text: str, limit: int = 420) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def derive_bm25_index_path(vector_index_path: Path) -> Path:
    return vector_index_path.with_suffix(".bm25.json")
