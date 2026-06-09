from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import faiss
import numpy as np

from services.rag.memory_vector_store import normalize_matrix, normalize_vector
from services.rag.schema import EmbeddingChunk, SearchResult


FaissIndexType = Literal["flat", "hnsw", "ivf_flat", "ivf_pq"]


@dataclass(frozen=True)
class FaissIndexConfig:
    index_type: FaissIndexType = "flat"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 80
    hnsw_ef_search: int = 64
    nlist: int | None = None
    nprobe: int = 8
    pq_m: int = 16
    pq_nbits: int = 8
    normalize_embeddings: bool = True


class FaissVectorStore:
    """FAISS-backed vector store for dense retrieval experiments.

    This implementation is intentionally rebuild-oriented. It is designed for
    offline algorithm benchmarks over an existing JSON index, not for online
    incremental updates.
    """

    def __init__(self, config: FaissIndexConfig | None = None) -> None:
        self.config = config or FaissIndexConfig()
        self._chunks: list[EmbeddingChunk] = []
        self._index: faiss.Index | None = None
        self._dimension: int | None = None

    def clear(self) -> None:
        self._chunks = []
        self._index = None
        self._dimension = None

    def upsert(self, chunks: list[EmbeddingChunk]) -> None:
        self._chunks = list(chunks)
        self._index = self._build_index(self._chunks)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        if self._index is None or not self._chunks:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        if self.config.normalize_embeddings:
            query = normalize_vector(query)
        query_matrix = query.reshape(1, -1)

        limit = min(top_k, len(self._chunks))
        scores, indices = self._index.search(query_matrix, limit)

        results: list[SearchResult] = []
        for raw_score, raw_index in zip(scores[0], indices[0]):
            index = int(raw_index)
            if index < 0:
                continue
            results.append(
                SearchResult(
                    chunk=self._chunks[index].chunk,
                    score=round(float(raw_score), 4),
                )
            )
        return results

    def delete(self, note_path: str) -> None:
        self._chunks = [
            chunk
            for chunk in self._chunks
            if chunk.chunk.note_path != note_path
        ]
        self._index = self._build_index(self._chunks) if self._chunks else None

    def persist(self) -> None:
        return

    def count(self) -> int:
        return len(self._chunks)

    def serialized_size_bytes(self) -> int:
        if self._index is None:
            return 0
        return int(len(faiss.serialize_index(self._index)))

    def _build_index(self, chunks: list[EmbeddingChunk]) -> faiss.Index:
        matrix = np.asarray([chunk.embedding for chunk in chunks], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("FAISS index requires a non-empty 2D embedding matrix")

        if self.config.normalize_embeddings:
            matrix = normalize_matrix(matrix).astype(np.float32)

        count, dimension = matrix.shape
        self._dimension = int(dimension)
        index = self._create_empty_index(count=count, dimension=dimension)

        if not index.is_trained:
            index.train(matrix)

        index.add(matrix)
        return index

    def _create_empty_index(self, *, count: int, dimension: int) -> faiss.Index:
        index_type = self.config.index_type

        if index_type == "flat":
            return faiss.IndexFlatIP(dimension)

        if index_type == "hnsw":
            index = faiss.IndexHNSWFlat(
                dimension,
                self.config.hnsw_m,
                faiss.METRIC_INNER_PRODUCT,
            )
            index.hnsw.efConstruction = self.config.hnsw_ef_construction
            index.hnsw.efSearch = self.config.hnsw_ef_search
            return index

        nlist = resolve_nlist(count=count, requested_nlist=self.config.nlist)
        quantizer = faiss.IndexFlatIP(dimension)

        if index_type == "ivf_flat":
            index = faiss.IndexIVFFlat(
                quantizer,
                dimension,
                nlist,
                faiss.METRIC_INNER_PRODUCT,
            )
            index.nprobe = min(self.config.nprobe, nlist)
            return index

        if index_type == "ivf_pq":
            pq_m = resolve_pq_m(dimension=dimension, requested_m=self.config.pq_m)
            index = faiss.IndexIVFPQ(
                quantizer,
                dimension,
                nlist,
                pq_m,
                self.config.pq_nbits,
                faiss.METRIC_INNER_PRODUCT,
            )
            index.nprobe = min(self.config.nprobe, nlist)
            return index

        raise ValueError(f"Unsupported FAISS index type: {index_type}")


def resolve_nlist(*, count: int, requested_nlist: int | None) -> int:
    if requested_nlist is not None:
        if requested_nlist <= 0:
            raise ValueError("nlist must be positive")
        return min(requested_nlist, count)
    return max(1, min(int(np.sqrt(count)), count))


def resolve_pq_m(*, dimension: int, requested_m: int) -> int:
    if requested_m <= 0:
        raise ValueError("pq_m must be positive")
    if dimension % requested_m == 0:
        return requested_m

    divisors = [value for value in range(requested_m, 0, -1) if dimension % value == 0]
    if not divisors:
        raise ValueError(f"Cannot find a valid pq_m divisor for dimension={dimension}")
    return divisors[0]
