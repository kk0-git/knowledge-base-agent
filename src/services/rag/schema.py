from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TextChunk:
    """笔记分块后的文本单元。

    每个 chunk 对应笔记中的一个 heading 段落（或其子段），
    是向量检索的最小粒度单位。
    """
    chunk_id: str          # 唯一标识，格式: {note_path}#{sha1[:12]}
    note_path: str         # 笔记在 vault 中的相对路径
    heading_path: list[str]  # 当前块所处的 heading 层级路径，如 ["全栈", "FastAPI"]
    text: str              # 块文本内容（已做空白规范化）
    start_line: int | None = None  # 块在原文件中的起始行号（1-based）
    end_line: int | None = None    # 块在原文件中的结束行号（1-based）
    metadata: dict[str, Any] = field(default_factory=dict)  # 扩展元数据，如合并标记、来源等


@dataclass(frozen=True)
class EmbeddingChunk:
    """带向量的 chunk，用于批量存入向量存储。"""
    chunk: TextChunk
    embedding: list[float]


@dataclass(frozen=True)
class SearchResult:
    """单条检索结果，包含 chunk 和相似度分数。"""
    chunk: TextChunk
    score: float  # cosine 相似度，范围 [-1, 1]，已归一化时为 [0, 1]


def text_chunk_to_dict(chunk: TextChunk) -> dict[str, Any]:
    """将 TextChunk 序列化为字典，用于 JSON 输出。"""
    return asdict(chunk)


def embedding_chunk_to_dict(chunk: EmbeddingChunk) -> dict[str, Any]:
    """将 EmbeddingChunk 序列化为字典，embedding 直接作为 list 存储。"""
    return {
        "chunk": asdict(chunk.chunk),
        "embedding": chunk.embedding,
    }


def search_result_to_dict(result: SearchResult) -> dict[str, Any]:
    """将 SearchResult 序列化为字典。"""
    return {
        "chunk": asdict(result.chunk),
        "score": result.score,
    }


def text_chunk_from_dict(data: dict[str, Any]) -> TextChunk:
    """从字典反序列化 TextChunk。"""
    return TextChunk(
        chunk_id=data["chunk_id"],
        note_path=data["note_path"],
        heading_path=data.get("heading_path", []),
        text=data["text"],
        start_line=data.get("start_line"),
        end_line=data.get("end_line"),
        metadata=dict(data.get("metadata", {})),
    )


def embedded_chunk_from_dict(data: dict[str, Any]) -> EmbeddingChunk:
    """从字典反序列化 EmbeddingChunk，含内部 TextChunk 和向量。"""
    return EmbeddingChunk(
        chunk=text_chunk_from_dict(data["chunk"]),
        embedding=data["embedding"],
    )