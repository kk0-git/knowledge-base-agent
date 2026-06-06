from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from services.rag.schema import SearchResult, TextChunk, text_chunk_from_dict, text_chunk_to_dict

ASCII_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_./:+-]*|\d+(?:\.\d+)*|[\u4e00-\u9fff]+")

@dataclass(frozen=True)
class BM25Config:
    # 这是 Robertson 在 TREC 数据集上网格搜索出的最优值
    k1: float = 1.5 # 词频饱和速度（词出现得越多就越重要词频饱和度。值越高，重复词的计数越接近线性。标准范围 [1.2, 2.0]；默认值 1.5。）
    b: float = 0.75 # 文档长度归一化（b越大，长文档的惩罚越重。1.0 完全惩罚长文档；0.0 禁用长度归一化。默认值 0.75。)

def tokenize_mixed_text(text: str) -> list[str]:
    """中英混合技术笔记的轻量 tokenizer。

    目标不是语言学分词正确，而是让技术关键词可被精确命中：
    - 英文、命令、API 名保留整体 token
    - 下划线、斜杠、点号等技术符号会额外拆分
    - 中文连续片段使用 2-gram
    """
    tokens: list[str] = []

    for raw_match in ASCII_TOKEN_RE.findall(text):
        match = raw_match.lower()
        if re.fullmatch(r"[\u4e00-\u9fff]+", match):
            # 中文连续片段，生成 2-gram
            tokens.extend(_tokenize_chinese_segment(match))
            continue

        tokens.append(match)
        parts = re.split(r"([_./:+-])", match)
        for part in parts:
            if len(part) >= 2 and part != match:
                tokens.append(part)

    return tokens


def _tokenize_chinese_segment(segment: str) -> list[str]:
    if not segment:
        return []
    if len(segment) == 1:
        return [segment]
    
    tokens = [segment[index: index + 2] for index in range(len(segment) - 1)]

    # 较短中文词组保留整体，方便“进程”“缓存”“系统调用”这类查询命中
    if len(segment) <= 8:
        tokens.append(segment)

    return tokens

def build_bm25_document_text(chunk: TextChunk) -> str:
    """构造 BM25 文档文本。

    BM25 更吃关键词，heading 和路径对技术笔记很重要，所以这里轻量加权：
    - note_path 出现一次
    - heading_path 出现两次
    - chunk.text 出现一次
    """
    heading_text = " ".join(chunk.heading_path) # 将heading_path放在前面，增加其权重

    return "\n".join(
        [
            chunk.note_path,
            heading_text,
            heading_text,
            chunk.text,
        ]
    )

class BM25Index:
    def __init__(self, persist_path: str | Path | None = None, config: BM25Config | None = None) -> None:
        self.persist_path = Path(persist_path) if persist_path else None
        self.config = config or BM25Config()

        self.chunks: dict[str, TextChunk] = {}
        self.term_frequencies: dict[str, dict[str, int]] = {}
        self.document_frequencies: dict[str, int] = {}
        self.document_lengths: dict[str, int] = {}
        self.avg_document_length: float = 0.0

        if self.persist_path and self.persist_path.exists():
            self.load()

    def build(self, chunks: list[TextChunk]) -> None:
        """对给定的 chunks 构建 BM25 索引。会替换掉原有索引数据。"""
        self.chunks = {}
        self.term_frequencies = {}
        self.document_frequencies = {}
        self.document_lengths = {}

        for chunk in chunks:
            tokens = tokenize_mixed_text(build_bm25_document_text(chunk))
            term_frequency = Counter(tokens)

            self.chunks[chunk.chunk_id] = chunk
            self.term_frequencies[chunk.chunk_id] = dict(term_frequency)
            self.document_lengths[chunk.chunk_id] = len(tokens)

            for token in term_frequency:
                self.document_frequencies[token] = self.document_frequencies.get(token, 0) + 1

        total_length = sum(self.document_lengths.values())
        self.avg_document_length = total_length / len(chunks) if chunks else 0.0

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_tokens = tokenize_mixed_text(query)
        if not query_tokens or not self.chunks:
            return []
        
        query_terms = Counter(query_tokens)
        scores: dict[str, float] = {}

        for term, query_weight in query_terms.items():
            document_frequency = self.document_frequencies.get(term, 0)
            if document_frequency == 0:
                continue

            idf = self._idf(document_frequency)
            for chunk_id, term_frequency in self.term_frequencies.items():
                tf = term_frequency.get(term, 0)
                if tf == 0:
                    continue

                score = self._term_score(
                    tf=tf,
                    idf=idf,
                    document_length=self.document_lengths[chunk_id],
                )
                scores[chunk_id] = scores.get(chunk_id, 0.0) + score * query_weight

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        return [
            SearchResult(
                chunk=self.chunks[chunk_id],
                score=round(score, 6),
            )
            for chunk_id, score in ranked
        ]

    def persist(self) -> None:
        if not self.persist_path:
            return
        
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "chunks": {
                chunk_id: text_chunk_to_dict(chunk)
                for chunk_id, chunk in self.chunks.items()
            },
            "term_frequencies": self.term_frequencies,
            "document_frequencies": self.document_frequencies,
            "document_lengths": self.document_lengths,
            "avg_document_length": self.avg_document_length,
        }
        self.persist_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        if not self.persist_path:
            raise ValueError("persist_path is required to load BM25 index")
        
        payload = json.loads(self.persist_path.read_text(encoding="utf-8"))

        self.config = BM25Config(**payload.get("config", {}))
        self.chunks = {
            chunk_id: text_chunk_from_dict(chunk_data)
            for chunk_id, chunk_data in payload.get("chunks", {}).items()
        }
        self.term_frequencies = {
            chunk_id: {term: int(count) for term, count in terms.items()}
            for chunk_id, terms in payload.get("term_frequencies", {}).items()
        }
        self.document_frequencies = {
            term: int(count)
            for term, count in payload.get("document_frequencies", {}).items()
        }
        self.document_lengths = {
            chunk_id: int(length)
            for chunk_id, length in payload.get("document_lengths", {}).items()
        }
        self.avg_document_length = float(payload.get("avg_document_length", 0.0))

    def _idf(self, document_frequency: int) -> float:
        document_count = len(self.chunks)
        return math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))

    def _term_score(self, tf: int, idf: float, document_length: int) -> float:
        if self.avg_document_length <= 0:
            return 0.0

        k1 = self.config.k1
        b = self.config.b

        denominator = tf + k1 * (1 - b + b * document_length / self.avg_document_length)
        return idf * (tf * (k1 + 1)) / denominator
