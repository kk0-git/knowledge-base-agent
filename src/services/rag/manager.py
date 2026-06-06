from __future__ import annotations

from pathlib import Path

from services.rag.bm25 import BM25Index
from services.rag.chunker import HeadingChunker
from services.rag.embedder import Embedder, build_chunk_embedding_text
from services.rag.query_variants import QueryVariant, RankedList, weighted_rrf_fuse
from services.rag.reranker import Reranker
from services.rag.schema import EmbeddingChunk, SearchResult, TextChunk
from services.rag.vector_store import VectorStore


class RAGManager:
    """Top-level RAG orchestrator for chunking, indexing, and search."""

    def __init__(
        self,
        chunker: HeadingChunker,
        embedder: Embedder | None,
        vector_store: VectorStore,
        bm25_index: BM25Index | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.reranker = reranker

    def index_markdown_files(self, vault_path: str | Path, markdown_files: list[Path]) -> int:
        """Chunk markdown files, embed chunks, and upsert them into vector store."""
        vault_root = Path(vault_path)
        chunks: list[TextChunk] = []

        for file_path in markdown_files:
            chunks.extend(self.chunker.chunk_file(vault_root=vault_root, file_path=file_path))

        return self.index_chunks(chunks)

    def index_chunks(self, chunks: list[TextChunk]) -> int:
        """Embed TextChunk objects and persist them to vector store."""
        if not chunks:
            return 0
        if self.embedder is None:
            raise ValueError("embedder is required to index vector chunks")

        texts = [build_chunk_embedding_text(chunk) for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)

        embedded_chunks = [
            EmbeddingChunk(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=False)
        ]

        self.vector_store.upsert(embedded_chunks)
        self.vector_store.persist()

        return len(embedded_chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Compatibility alias for dense vector search."""
        return self.dense_search(query=query, top_k=top_k)

    def dense_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Semantic vector search."""
        if self.embedder is None:
            raise ValueError("embedder is required for dense search")

        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.search(query_embedding=query_embedding, top_k=top_k)

    def bm25_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Lexical BM25 search."""
        if self.bm25_index is None:
            raise ValueError("bm25_index is required for BM25 search")

        return self.bm25_index.search(query=query, top_k=top_k)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        dense_top_k: int = 50,
        bm25_top_k: int = 50,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """Hybrid dense + BM25 search using Reciprocal Rank Fusion."""
        dense_results = self.dense_search(query=query, top_k=dense_top_k)
        bm25_results = self.bm25_search(query=query, top_k=bm25_top_k)

        return rrf_fuse(
            ranked_lists=[dense_results, bm25_results],
            top_k=top_k,
            rrf_k=rrf_k,
        )

    def hybrid_rerank_search(
        self,
        query: str,
        top_k: int = 5,
        rerank_candidates: int = 50,
        dense_top_k: int = 50,
        bm25_top_k: int = 50,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """Hybrid retrieval followed by cross-encoder reranking."""
        if self.reranker is None:
            raise ValueError("reranker is required for hybrid rerank search")

        candidates = self.hybrid_search(
            query=query,
            top_k=rerank_candidates,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            rrf_k=rrf_k,
        )

        return self.reranker.rerank(
            query=query,
            results=candidates,
            top_k=top_k,
        )

    def variant_hybrid_search(
        self,
        variants: list[QueryVariant],
        top_k: int = 5,
        dense_top_k: int = 50,
        bm25_top_k: int = 50,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """Hybrid search over original/rewrite query variants.

        Each variant is searched independently through dense and BM25, then all
        ranked lists are fused with weighted RRF. The caller decides which
        variants are safe to use; this method stays LLM-free.
        """
        ranked_lists: list[RankedList] = []

        for variant in variants:
            ranked_lists.append(
                RankedList(
                    results=self.dense_search(query=variant.text, top_k=dense_top_k),
                    query_source=variant.source,
                    retriever="dense",
                    weight=variant.weight,
                )
            )
            ranked_lists.append(
                RankedList(
                    results=self.bm25_search(query=variant.text, top_k=bm25_top_k),
                    query_source=variant.source,
                    retriever="bm25",
                    weight=variant.weight,
                )
            )

        return weighted_rrf_fuse(
            ranked_lists=ranked_lists,
            top_k=top_k,
            rrf_k=rrf_k,
        )

    def variant_hybrid_rerank_search(
        self,
        query: str,
        variants: list[QueryVariant],
        top_k: int = 5,
        rerank_candidates: int = 50,
        dense_top_k: int = 50,
        bm25_top_k: int = 50,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """Variant hybrid retrieval followed by reranking with original query."""
        if self.reranker is None:
            raise ValueError("reranker is required for variant hybrid rerank search")

        candidates = self.variant_hybrid_search(
            variants=variants,
            top_k=rerank_candidates,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            rrf_k=rrf_k,
        )

        return self.reranker.rerank(
            query=query,
            results=candidates,
            top_k=top_k,
        )

    def delete_note(self, note_path: str) -> None:
        """Delete all vector chunks for one note.

        BM25 debug index is rebuilt as a whole for now, so deletion remains
        vector-only until BM25 gets incremental APIs.
        """
        self.vector_store.delete(note_path)
        self.vector_store.persist()


def rrf_fuse(
    ranked_lists: list[list[SearchResult]],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """Fuse ranked lists with Reciprocal Rank Fusion.

    RRF uses rank positions, so dense cosine scores and BM25 scores do not need
    normalization. Returned SearchResult.score is the fused RRF score.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, TextChunk] = {}

    for ranked_results in ranked_lists:
        for rank, result in enumerate(ranked_results, start=1):
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
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
