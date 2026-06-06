from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from knowledge_base_agent.llm.client import LLMClient
from knowledge_base_agent.llm.schema import LLMMessage, LLMRequest


QUERY_QUALITIES = {
    "clear",
    "too_short",
    "ambiguous",
    "colloquial",
    "typo_or_alias",
    "multi_intent",
    "missing_context",
    "unknown",
}

QUERY_INTENTS = {
    "definition",
    "howto",
    "troubleshooting",
    "comparison",
    "exploratory",
    "command_or_config",
    "code_related",
    "unknown",
}


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    query_quality: str
    intent: str
    should_rewrite: bool
    should_hyde: bool
    normalized_query: str | None
    dense_queries: list[str]
    bm25_terms: list[str]
    hyde_answer: str | None
    risk_notes: list[str]


@dataclass(frozen=True)
class QueryPlanResult:
    plan: QueryPlan
    raw_output: str
    validation_warnings: list[str]


class QueryPlanParseError(ValueError):
    def __init__(self, message: str, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


def build_query_plan_messages(query: str) -> list[LLMMessage]:
    return [
        LLMMessage(role="system", content=build_system_prompt()),
        LLMMessage(role="user", content=build_user_prompt(query)),
    ]


def build_system_prompt() -> str:
    return """你是一个本地知识库检索系统中的 Query Planner。

你的工作是把用户的自然语言 query 分析成一个检索计划，供 dense 向量检索、BM25 关键词检索和可选 HyDE 检索使用。

这个检索计划不是最终答案。它只描述 query 的质量、意图、可选改写、关键词扩展和潜在风险。

输出需要是合法 JSON，并符合这些字段含义：

{
  "original_query": "用户原始 query",
  "query_quality": "clear | too_short | ambiguous | colloquial | typo_or_alias | multi_intent | missing_context | unknown",
  "intent": "definition | howto | troubleshooting | comparison | exploratory | command_or_config | code_related | unknown",
  "should_rewrite": "是否需要规范化 query",
  "should_hyde": "是否适合生成假设答案辅助 dense 检索",
  "normalized_query": "规范化后的 query；不需要改写时为 null",
  "dense_queries": "适合向量检索的自然语言 query 列表",
  "bm25_terms": "适合 BM25 的关键词列表",
  "hyde_answer": "用于 HyDE 的简短假设答案；不适合 HyDE 时为 null",
  "risk_notes": "改写或扩展时需要注意的风险"
}

判断原则：
- 如果原始 query 已经清晰，不需要强行改写。
- 改写应保持原意，不扩大问题范围。
- BM25 关键词应偏具体，避免泛词。
- HyDE 只适合抽象、探索性、缺少明确术语的问题。
- 不生成具体笔记路径、文件名或用户未提供的事实。
- 不回答用户问题。
"""


def build_user_prompt(query: str) -> str:
    return f"""背景：
用户正在检索个人 Obsidian 技术笔记。笔记可能包含中文说明、英文技术术语、命令、代码片段和零散记录。

请为下面的 query 生成 query plan。

query:
{query}

输出示例，仅用于说明格式：

{{
  "original_query": "agent咋做",
  "query_quality": "colloquial",
  "intent": "howto",
  "should_rewrite": true,
  "should_hyde": false,
  "normalized_query": "如何构建 LLM Agent？",
  "dense_queries": [
    "如何构建 LLM Agent？"
  ],
  "bm25_terms": [
    "LLM Agent",
    "工具调用",
    "规划",
    "记忆",
    "推理",
    "架构"
  ],
  "hyde_answer": null,
  "risk_notes": [
    "Agent 可能指代不同技术上下文，改写时不应扩展到具体框架"
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


def normalize_query_plan(data: dict[str, Any], original_query: str) -> tuple[QueryPlan, list[str]]:
    warnings: list[str] = []

    normalized_original = _coerce_string(data.get("original_query"), fallback=original_query)
    if normalized_original != original_query:
        warnings.append("original_query differs from input query")

    query_quality = _coerce_string(data.get("query_quality"), fallback="unknown")
    if query_quality not in QUERY_QUALITIES:
        warnings.append(f"invalid query_quality: {query_quality}")
        query_quality = "unknown"

    intent = _coerce_string(data.get("intent"), fallback="unknown")
    if intent not in QUERY_INTENTS:
        warnings.append(f"invalid intent: {intent}")
        intent = "unknown"

    should_rewrite = _coerce_bool(data.get("should_rewrite"))
    should_hyde = _coerce_bool(data.get("should_hyde"))

    normalized_query = _coerce_optional_string(data.get("normalized_query"))
    if not should_rewrite:
        if normalized_query and normalized_query != original_query:
            warnings.append("normalized_query was dropped because should_rewrite is false")
        normalized_query = None

    dense_queries = _coerce_string_list(data.get("dense_queries"))
    if len(dense_queries) > 3:
        warnings.append("dense_queries truncated to 3 items")
        dense_queries = dense_queries[:3]

    bm25_terms = _coerce_string_list(data.get("bm25_terms"))
    if len(bm25_terms) > 12:
        warnings.append("bm25_terms truncated to 12 items")
        bm25_terms = bm25_terms[:12]

    hyde_answer = _coerce_optional_string(data.get("hyde_answer"))
    if not should_hyde:
        if hyde_answer:
            warnings.append("hyde_answer was dropped because should_hyde is false")
        hyde_answer = None
    elif hyde_answer and len(hyde_answer) > 200:
        warnings.append("hyde_answer truncated to 200 characters")
        hyde_answer = hyde_answer[:200]

    risk_notes = _coerce_string_list(data.get("risk_notes"))
    if len(risk_notes) > 5:
        warnings.append("risk_notes truncated to 5 items")
        risk_notes = risk_notes[:5]

    return (
        QueryPlan(
            original_query=original_query,
            query_quality=query_quality,
            intent=intent,
            should_rewrite=should_rewrite,
            should_hyde=should_hyde,
            normalized_query=normalized_query,
            dense_queries=dense_queries,
            bm25_terms=bm25_terms,
            hyde_answer=hyde_answer,
            risk_notes=risk_notes,
        ),
        warnings,
    )


class LLMQueryPlanner:
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

    def plan(self, query: str) -> QueryPlanResult:
        request = LLMRequest(
            model=self.model,
            messages=build_query_plan_messages(query),
            temperature=self.temperature,
            response_format={"type": "json_object"} if self.use_response_format else None,
        )
        response = self.client.complete(request)

        try:
            data = parse_llm_json(response.content)
            plan, warnings = normalize_query_plan(data, original_query=query)
        except Exception as exc:
            raise QueryPlanParseError(f"Failed to parse query plan: {exc}", response.content) from exc

        return QueryPlanResult(
            plan=plan,
            raw_output=response.content,
            validation_warnings=warnings,
        )


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "y"}
    return bool(value)


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
