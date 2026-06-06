from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from knowledge_base_agent.llm.client import LLMClient
from knowledge_base_agent.llm.schema import LLMMessage, LLMRequest


REWRITE_TYPES = {
    "none",
    "normalize_colloquial",
    "fix_typo",
    "clarify_expression",
    "preserve_ambiguous",
}


@dataclass(frozen=True)
class QueryRewrite:
    original_query: str
    should_rewrite: bool
    rewritten_query: str | None
    rewrite_type: str
    confidence: float
    risk_notes: list[str]


@dataclass(frozen=True)
class QueryRewriteResult:
    rewrite: QueryRewrite
    raw_output: str
    validation_warnings: list[str]


class QueryRewriteParseError(ValueError):
    def __init__(self, message: str, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


def build_query_rewrite_messages(query: str) -> list[LLMMessage]:
    return [
        LLMMessage(role="system", content=build_system_prompt()),
        LLMMessage(role="user", content=build_user_prompt(query)),
    ]


def build_system_prompt() -> str:
    return """你是一个本地知识库检索系统中的 Query Rewriter。

你的任务是判断用户 query 是否需要被轻度规范化，并在需要时输出一个更清晰的 query。

你只做 query 改写，不回答问题，不生成关键词，不生成多个查询，不补充用户没有明确表达的知识背景。

改写原则：
- 如果 query 已经清晰，should_rewrite 为 false，rewritten_query 为 null。
- 如果 query 只是口语化、缺少标点、表达不完整，可以轻度规范化。
- 不扩大原 query 的问题范围。
- 不把缩写强行解释成具体含义。
- 对多义词、缩写、缺少上下文的问题，只做表面规范化，并在 risk_notes 中提示歧义。
- 只输出合法 JSON。

输出 JSON 字段：

{
  "original_query": "用户原始 query",
  "should_rewrite": true,
  "rewritten_query": "改写后的 query；不需要改写时为 null",
  "rewrite_type": "none | normalize_colloquial | fix_typo | clarify_expression | preserve_ambiguous",
  "confidence": 0.0,
  "risk_notes": ["风险提示"]
}
"""


def build_user_prompt(query: str) -> str:
    return f"""请判断下面的 query 是否需要轻度改写。

query:
{query}

输出示例，仅用于说明格式：

{{
  "original_query": "PCB是啥",
  "should_rewrite": true,
  "rewritten_query": "PCB 是什么？",
  "rewrite_type": "normalize_colloquial",
  "confidence": 0.86,
  "risk_notes": [
    "PCB 是多义缩写，缺少上下文时不应扩展为具体全称"
  ]
}}
"""


def parse_llm_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM output does not contain a JSON object")

    payload = stripped[start : end + 1]
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("LLM JSON output must be an object")
    return data


def normalize_query_rewrite(data: dict[str, Any], original_query: str) -> tuple[QueryRewrite, list[str]]:
    warnings: list[str] = []

    llm_original = _coerce_string(data.get("original_query"), fallback=original_query)
    if llm_original != original_query:
        warnings.append("original_query differs from input query")

    should_rewrite = _coerce_bool(data.get("should_rewrite"))
    rewritten_query = _coerce_optional_string(data.get("rewritten_query"))
    rewrite_type = _coerce_string(data.get("rewrite_type"), fallback="none")
    confidence = _coerce_float(data.get("confidence"), fallback=0.0)
    risk_notes = _coerce_string_list(data.get("risk_notes"))

    if rewrite_type not in REWRITE_TYPES:
        warnings.append(f"invalid rewrite_type: {rewrite_type}")
        rewrite_type = "clarify_expression" if should_rewrite else "none"

    if confidence < 0.0 or confidence > 1.0:
        warnings.append("confidence clamped to [0, 1]")
        confidence = min(max(confidence, 0.0), 1.0)

    if len(risk_notes) > 5:
        warnings.append("risk_notes truncated to 5 items")
        risk_notes = risk_notes[:5]

    if not should_rewrite:
        if rewritten_query and rewritten_query != original_query:
            warnings.append("rewritten_query was dropped because should_rewrite is false")
        rewritten_query = None
        rewrite_type = "none"
    elif not rewritten_query:
        warnings.append("should_rewrite is true but rewritten_query is empty; rewrite disabled")
        should_rewrite = False
        rewrite_type = "none"
    elif rewritten_query == original_query:
        warnings.append("rewritten_query equals original query; rewrite disabled")
        should_rewrite = False
        rewritten_query = None
        rewrite_type = "none"

    return (
        QueryRewrite(
            original_query=original_query,
            should_rewrite=should_rewrite,
            rewritten_query=rewritten_query,
            rewrite_type=rewrite_type,
            confidence=confidence,
            risk_notes=risk_notes,
        ),
        warnings,
    )


class LLMQueryRewriter:
    def __init__(
        self,
        client: LLMClient,
        model: str,
        temperature: float = 0.0,
        use_response_format: bool = True,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.use_response_format = use_response_format

    def rewrite(self, query: str) -> QueryRewriteResult:
        request = LLMRequest(
            model=self.model,
            messages=build_query_rewrite_messages(query),
            temperature=self.temperature,
            response_format={"type": "json_object"} if self.use_response_format else None,
        )
        response = self.client.complete(request)

        try:
            data = parse_llm_json(response.content)
            rewrite, warnings = normalize_query_rewrite(data, original_query=query)
        except Exception as exc:
            raise QueryRewriteParseError(f"Failed to parse query rewrite: {exc}", response.content) from exc

        return QueryRewriteResult(
            rewrite=rewrite,
            raw_output=response.content,
            validation_warnings=warnings,
        )


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "y"}
    return bool(value)


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "null":
            return None
        return stripped
    return str(value).strip() or None


def _coerce_string(value: Any, fallback: str) -> str:
    coerced = _coerce_optional_string(value)
    return coerced if coerced is not None else fallback


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []

    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _coerce_optional_string(item)
        if not text or text in seen:
            continue
        result.append(text)
        seen.add(text)
    return result
