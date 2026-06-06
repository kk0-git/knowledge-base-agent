from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from services.rag.schema import (
    EmbeddingChunk,
    SearchResult,
    TextChunk,
    embedded_chunk_from_dict,
    embedding_chunk_to_dict,
)


class MemoryVectorStore:
    """Small JSON-backed vector store used by the local RAG debug pipeline."""

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self.persist_path = Path(persist_path) if persist_path else None
        self._chunks: dict[str, EmbeddingChunk] = {}
        self.files: dict[str, dict[str, Any]] = {}
        self.index_config: dict[str, Any] = {}

        if self.persist_path and self.persist_path.exists():
            self.load()

    def clear(self) -> None:
        """Clear all in-memory data and remove the persisted index file."""
        self._chunks = {}
        self.files = {}
        self.index_config = {}
        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()

    def upsert(self, chunks: list[EmbeddingChunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk.chunk_id] = chunk

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        if not self._chunks:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        query = normalize_vector(query)

        chunk_items = list(self._chunks.values())
        matrix = np.asarray(
            [chunk.embedding for chunk in chunk_items],
            dtype=np.float32,
        )
        matrix = normalize_matrix(matrix)

        scores = matrix @ query
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results: list[SearchResult] = []
        for index in ranked_indices:
            embedded_chunk = chunk_items[int(index)]
            results.append(
                SearchResult(
                    chunk=embedded_chunk.chunk,
                    score=round(float(scores[int(index)]), 4),
                )
            )

        return results

    def delete(self, note_path: str) -> None:
        ids_to_delete = [
            chunk_id
            for chunk_id, embedded_chunk in self._chunks.items()
            if embedded_chunk.chunk.note_path == note_path
        ]
        self.delete_chunks(ids_to_delete)

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            self._chunks.pop(chunk_id, None)

    def get_text_chunks(self) -> list[TextChunk]:
        return [embedded.chunk for embedded in self._chunks.values()]

    def get_files_metadata(self) -> dict[str, dict[str, Any]]:
        return dict(self.files)

    def set_file_metadata(self, note_path: str, metadata: dict[str, Any]) -> None:
        self.files[note_path] = dict(metadata)

    def remove_file_metadata(self, note_path: str) -> None:
        self.files.pop(note_path, None)

    def set_index_config(self, config: dict[str, Any]) -> None:
        self.index_config = dict(config)

    def get_index_config(self) -> dict[str, Any]:
        return dict(self.index_config)

    def persist(self) -> None:
        if self.persist_path is None:
            return

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "index_config": self.index_config,
            "files": self.files,
            "chunks": [
                embedding_chunk_to_dict(chunk)
                for chunk in self._chunks.values()
            ],
        }

        self.persist_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        if self.persist_path is None or not self.persist_path.exists():
            return

        payload = json.loads(self.persist_path.read_text(encoding="utf-8"))
        self.index_config = dict(payload.get("index_config", {}))
        self.files = {
            str(note_path): dict(metadata)
            for note_path, metadata in payload.get("files", {}).items()
        }
        self._chunks = {
            item["chunk"]["chunk_id"]: embedded_chunk_from_dict(item)
            for item in payload.get("chunks", [])
        }

    def count(self) -> int:
        return len(self._chunks)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms
