from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol

from services.rag.schema import TextChunk

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class Embedder(Protocol):
    """嵌入器接口协议。

    运行时实现类不需要显式继承此 Protocol，
    只要满足 embed_texts 和 embed_query 的方法签名即可。
    """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量编码文本，返回 [文本数, 维度] 的向量列表。"""
        ...

    def embed_query(self, query: str) -> list[float]:
        """编码单条查询文本，返回单个向量。"""
        ...


class SentenceTransformerEmbedder:
    """基于 sentence_transformers 的本地嵌入器实现。

    支持任意 HuggingFace 兼容模型，默认使用 BAAI/bge-m3。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        max_seq_length: int | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量编码文本为向量列表。空输入返回空列表。"""
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """编码查询文本为单个向量。实际上是 embed_texts 的便捷包装。"""
        embeddings = self.embed_texts([query])
        return embeddings[0]


class OpenAICompatibleEmbedder:
    """Embedding client for OpenAI-compatible /v1/embeddings APIs."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        base_url: str,
        batch_size: int = 32,
        timeout_seconds: int = 120,
    ) -> None:
        if not model_name:
            raise ValueError("embedding model_name is required")
        if not base_url:
            raise ValueError("embedding base_url is required")
        if not api_key:
            raise ValueError("embedding api_key is required")

        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for batch_index, start in enumerate(range(0, len(texts), self.batch_size), start=1):
            batch = texts[start : start + self.batch_size]
            print(f"Embedding API batch {batch_index}/{total_batches} ({len(batch)} texts)")
            embeddings.extend(self._embed_batch(batch))
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        embeddings = self.embed_texts([query])
        return embeddings[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        url = self.base_url if self.base_url.endswith("/embeddings") else f"{self.base_url}/embeddings"
        payload = {
            "model": self.model_name,
            "input": texts,
        }
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Embedding HTTP error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Embedding request failed: {exc}") from exc

        items = raw.get("data", [])
        if not isinstance(items, list):
            raise RuntimeError("Embedding response missing data list")

        items = sorted(items, key=lambda item: int(item.get("index", 0)))
        return [item["embedding"] for item in items]


def create_embedder(
    *,
    provider: str,
    model_name: str,
    batch_size: int = 32,
    max_seq_length: int | None = None,
) -> Embedder:
    if provider == "local":
        return SentenceTransformerEmbedder(
            model_name=model_name,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        )

    if provider == "openai_compatible":
        return OpenAICompatibleEmbedder(
            model_name=model_name,
            api_key=os.getenv("EMBEDDING_API_KEY", ""),
            base_url=os.getenv("EMBEDDING_BASE_URL", ""),
            batch_size=batch_size,
            timeout_seconds=int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "120")),
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")


def build_chunk_embedding_text(chunk: TextChunk) -> str:
    """将 TextChunk 组装为送入嵌入模型的文本。

    格式: File: {路径}\nHeading: {层级路径}\nContent:\n{文本}
    使用英文字段前缀的方式，参考 Lumina-Note 和 obsidian-graph 的做法。
    """
    parts: list[str] = []

    parts.append(f"File: {chunk.note_path}")

    if chunk.heading_path:
        parts.append("Heading: " + " > ".join(chunk.heading_path))

    parts.append("Content:")
    parts.append(chunk.text)

    return "\n".join(parts)


def embed_chunks(
    chunks: list[TextChunk],
    embedder: Embedder,
) -> list[list[float]]:
    """对 chunk 列表进行批量嵌入，返回等长的向量列表。"""
    texts = [build_chunk_embedding_text(chunk) for chunk in chunks]
    return embedder.embed_texts(texts)
