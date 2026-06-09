from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from services.rag.faiss_vector_store import FaissIndexConfig, FaissVectorStore
from services.rag.memory_vector_store import MemoryVectorStore
from services.rag.schema import embedded_chunk_from_dict
from services.rag.vector_store import VectorStore


VectorIndexType = Literal["flat", "hnsw"]

DEFAULT_HNSW_M = 48
DEFAULT_HNSW_EF_CONSTRUCTION = 80
DEFAULT_HNSW_EF_SEARCH = 64


def load_vector_store(
    *,
    index_path: Path,
    vector_index: VectorIndexType = "flat",
    hnsw_m: int = DEFAULT_HNSW_M,
    hnsw_ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION,
    hnsw_ef_search: int = DEFAULT_HNSW_EF_SEARCH,
) -> VectorStore:
    if vector_index == "flat":
        return MemoryVectorStore(persist_path=index_path)

    if vector_index == "hnsw":
        chunks = [
            embedded_chunk_from_dict(item)
            for item in memory_store_payload_chunks(index_path)
        ]
        store = FaissVectorStore(
            FaissIndexConfig(
                index_type="hnsw",
                hnsw_m=hnsw_m,
                hnsw_ef_construction=hnsw_ef_construction,
                hnsw_ef_search=hnsw_ef_search,
            )
        )
        store.upsert(chunks)
        return store

    raise ValueError(f"Unsupported vector index: {vector_index}")


def memory_store_payload_chunks(index_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not chunks:
        raise ValueError(f"No chunks found in vector index: {index_path}")
    return chunks
