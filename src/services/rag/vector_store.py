from __future__ import annotations

from typing import Protocol

from services.rag.schema import EmbeddingChunk, SearchResult


class VectorStore(Protocol):
    """向量存储接口协议。

    定义索引和检索的核心方法签名，运行时实现类无需显式继承。
    当前实现: MemoryVectorStore（numpy 内存矩阵），后续可替换为 SQLite/pgvector。
    """

    def upsert(self, chunks: list[EmbeddingChunk]) -> None:
        """插入或更新 chunk 向量。相同 chunk_id 会覆盖旧数据。"""
        ...

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """用查询向量检索 top_k 个最相似的 chunk，按分数降序返回。"""
        ...

    def delete(self, note_path: str) -> None:
        """删除指定笔记的所有 chunk。"""
        ...

    def persist(self) -> None:
        """将当前索引状态持久化到磁盘。"""
        ...
