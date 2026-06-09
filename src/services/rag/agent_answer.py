from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from knowledge_base_agent.llm.client import LLMClient
from knowledge_base_agent.llm.schema import LLMMessage, LLMRequest
from services.rag.context_packer import score_type_for_mode
from services.rag.grep_search import GrepMatch, grep_matches_to_dict, rg_search
from services.rag.intent_router import (
    ConversationCommand,
    LLMIntentRouter,
    RouterDecision,
    router_decision_to_dict,
)
from services.rag.manager import RAGManager
from services.rag.online_search import OnlineSearchClient, OnlineSearchResponse, online_response_to_dict
from services.rag.schema import SearchResult


ANSWER_SYSTEM_PROMPT = """你是一个个人知识库助手。

回答规则:
- 优先使用提供的本地笔记、精确搜索和 BM25 检索结果。
- 可以用通用知识补充解释，但不能把通用知识伪装成用户笔记内容。
- 如果上下文不足以支持结论，需要明确说明依据不足。
- 引用本地上下文时使用 [N1]、[R1]、[B1] 这样的 citation id。
- 用中文回答，除非术语、命令、代码需要保留英文。"""


ANSWER_USER_PROMPT_TEMPLATE = """用户问题:
{query}

Router decision:
command: {command}
reason: {reason}

Context:
{context}

请回答用户问题。
要求:
1. 先给出直接答案。
2. 必要时补充解释和操作步骤。
3. 关键事实尽量标注 citation id。
4. 最后列出“参考来源”，按 citation id 汇总。"""


@dataclass(frozen=True)
class AgentAnswerConfig:
    notes_top_k: int = 5
    regex_top_k: int = 8
    bm25_top_k: int = 8
    dense_top_k: int = 50
    hybrid_bm25_top_k: int = 50
    rrf_k: int = 60
    max_chars_per_item: int = 1000
    max_context_chars: int = 8000
    online_top_k: int = 5
    speculative_notes_search: bool = True


@dataclass(frozen=True)
class AgentAnswer:
    answer: str
    model: str
    prompt_chars: int


@dataclass(frozen=True)
class AgentRetrievalResult:
    query: str
    command: ConversationCommand
    router_decision: RouterDecision
    notes_results: list[SearchResult]
    rg_results: list[GrepMatch]
    bm25_results: list[SearchResult]
    online_response: OnlineSearchResponse
    tool_errors: list[dict[str, str]]
    context_text: str
    context_items: list[dict[str, Any]]
    timing: dict[str, int]
    telemetry: dict[str, Any]


@dataclass(frozen=True)
class AgentRunResult:
    query: str
    command: ConversationCommand
    router_decision: RouterDecision
    notes_results: list[SearchResult]
    rg_results: list[GrepMatch]
    bm25_results: list[SearchResult]
    online_response: OnlineSearchResponse
    tool_errors: list[dict[str, str]]
    context_text: str
    context_items: list[dict[str, Any]]
    answer: AgentAnswer
    timing: dict[str, int]
    telemetry: dict[str, Any]


class AgentAnswerPipeline:
    def __init__(
        self,
        *,
        router: LLMIntentRouter,
        llm_client: LLMClient,
        llm_model: str,
        manager: RAGManager,
        vault_root: Path,
        online_client: OnlineSearchClient | None = None,
        config: AgentAnswerConfig | None = None,
        answer_temperature: float = 0.2,
    ) -> None:
        self.router = router
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.manager = manager
        self.vault_root = vault_root
        self.online_client = online_client or OnlineSearchClient()
        self.config = config or AgentAnswerConfig()
        self.answer_temperature = answer_temperature

    def run(self, query: str) -> AgentRunResult:
        started_at = time.perf_counter()
        retrieval = self.retrieve(query)

        answer_started_at = time.perf_counter()
        answer = self.answer(
            query=query,
            decision=retrieval.router_decision,
            context_text=retrieval.context_text,
        )
        answer_ms = elapsed_ms(answer_started_at)

        timing = {
            **retrieval.timing,
            "answer_ms": answer_ms,
            "total_ms": elapsed_ms(started_at),
        }
        telemetry = {
            **retrieval.telemetry,
            "generation": {
                "answer_ms": answer_ms,
                "output_chars": len(answer.answer),
                "prompt_chars": answer.prompt_chars,
                "ttft_ms": None,
            },
            "total_ms": timing["total_ms"],
        }

        return AgentRunResult(
            query=query,
            command=retrieval.command,
            router_decision=retrieval.router_decision,
            notes_results=retrieval.notes_results,
            rg_results=retrieval.rg_results,
            bm25_results=retrieval.bm25_results,
            online_response=retrieval.online_response,
            tool_errors=retrieval.tool_errors,
            context_text=retrieval.context_text,
            context_items=retrieval.context_items,
            answer=answer,
            timing=timing,
            telemetry=telemetry,
        )

    def retrieve(self, query: str) -> AgentRetrievalResult:
        started_at = time.perf_counter()

        executor = ThreadPoolExecutor(max_workers=4)
        notes_future: Future[list[SearchResult]] | None = None
        route_future: Future[RouterDecision] | None = None

        if self.config.speculative_notes_search:
            notes_future = executor.submit(timed_call, self.run_notes_search, query)
        route_future = executor.submit(self.router.route, query)

        route_started_at = time.perf_counter()
        tool_errors: list[dict[str, str]] = []
        try:
            decision = route_future.result()
        except Exception as exc:
            decision = RouterDecision(
                command=ConversationCommand.NOTES,
                reason=f"router failed, fallback to Notes: {exc}",
                confidence=0.0,
                raw_response="",
                tool_args={},
                fallback_used=True,
            )
            tool_errors.append(tool_error("router", exc))
        route_ms = elapsed_ms(route_started_at)

        if decision.command == ConversationCommand.REGEX_SEARCH_FILES and not valid_regex_pattern(
            regex_pattern_from_decision(decision)
        ):
            decision = RouterDecision(
                command=ConversationCommand.NOTES,
                reason=f"RegexSearchFiles missing valid regex_pattern; fallback to Notes. Original reason: {decision.reason}",
                confidence=decision.confidence,
                raw_response=decision.raw_response,
                tool_args={"q": query},
                fallback_used=True,
            )

        retrieval_started_at = time.perf_counter()
        tool_telemetry: dict[str, dict[str, Any]] = {
            "notes_search": empty_tool_telemetry(),
            "rg_search": empty_tool_telemetry(),
            "bm25_search": empty_tool_telemetry(),
            "online_search": empty_tool_telemetry(),
        }
        notes_results: list[SearchResult] = []
        rg_results: list[GrepMatch] = []
        bm25_results: list[SearchResult] = []
        online_response = OnlineSearchResponse(enabled=False, provider="disabled", results=[], message="not requested")

        if decision.command in {ConversationCommand.NOTES, ConversationCommand.NOTES_ONLINE}:
            if notes_future is None:
                notes_future = executor.submit(timed_call, self.run_notes_search, query)

        if decision.command == ConversationCommand.REGEX_SEARCH_FILES:
            if notes_future is not None:
                notes_future.cancel()
            regex_pattern = regex_pattern_from_decision(decision)
            rg_future = executor.submit(timed_call, self.run_rg_search, regex_pattern)
            bm25_future = executor.submit(timed_call, self.run_bm25_search, regex_pattern)
            rg_results = timed_future_result_or_empty("rg_search", rg_future, tool_errors, tool_telemetry)
            bm25_results = timed_future_result_or_empty("bm25_search", bm25_future, tool_errors, tool_telemetry)

        if decision.command == ConversationCommand.NOTES_ONLINE:
            online_future = executor.submit(timed_call, self.online_client.search, query, self.config.online_top_k)
            notes_results = timed_future_result_or_empty("notes_search", notes_future, tool_errors, tool_telemetry) if notes_future is not None else []
            online_response = timed_future_result_or_online_error("online_search", online_future, tool_errors, tool_telemetry)
        elif decision.command == ConversationCommand.NOTES:
            notes_results = timed_future_result_or_empty("notes_search", notes_future, tool_errors, tool_telemetry) if notes_future is not None else []

        executor.shutdown(wait=False, cancel_futures=True)

        retrieval_ms = elapsed_ms(retrieval_started_at)
        update_tool_result_count(tool_telemetry["notes_search"], notes_results)
        update_tool_result_count(tool_telemetry["rg_search"], rg_results)
        update_tool_result_count(tool_telemetry["bm25_search"], bm25_results)
        update_tool_result_count(tool_telemetry["online_search"], online_response.results)

        context_text, context_items = build_agent_context(
            notes_results=notes_results,
            rg_results=rg_results,
            bm25_results=bm25_results,
            online_response=online_response,
            tool_errors=tool_errors,
            max_chars_per_item=self.config.max_chars_per_item,
            max_context_chars=self.config.max_context_chars,
        )

        return AgentRetrievalResult(
            query=query,
            command=decision.command,
            router_decision=decision,
            notes_results=notes_results,
            rg_results=rg_results,
            bm25_results=bm25_results,
            online_response=online_response,
            tool_errors=tool_errors,
            context_text=context_text,
            context_items=context_items,
            timing={
                "route_ms": route_ms,
            "retrieval_ms": retrieval_ms,
                "total_ms": elapsed_ms(started_at),
            },
            telemetry={
                "command": decision.command.value,
                "router": {
                    "latency_ms": route_ms,
                    "confidence": decision.confidence,
                    "fallback_used": decision.fallback_used,
                },
                "tools": tool_telemetry,
                "context": {
                    "context_items": len(context_items),
                    "context_chars": len(context_text),
                    "distinct_files": count_distinct_local_files(context_items),
                },
            },
        )

    def run_notes_search(self, query: str) -> list[SearchResult]:
        return self.manager.hybrid_search(
            query=query,
            top_k=self.config.notes_top_k,
            dense_top_k=self.config.dense_top_k,
            bm25_top_k=self.config.hybrid_bm25_top_k,
            rrf_k=self.config.rrf_k,
        )

    def run_rg_search(self, query: str) -> list[GrepMatch]:
        return rg_search(
            vault_root=self.vault_root,
            query=query,
            limit=self.config.regex_top_k,
        )

    def run_bm25_search(self, query: str) -> list[SearchResult]:
        return self.manager.bm25_search(
            query=query,
            top_k=self.config.bm25_top_k,
        )

    def answer(self, *, query: str, decision: RouterDecision, context_text: str) -> AgentAnswer:
        user_prompt = self.build_answer_user_prompt(query=query, decision=decision, context_text=context_text)
        response = self.llm_client.complete(
            LLMRequest(
                model=self.llm_model,
                temperature=self.answer_temperature,
                messages=[
                    LLMMessage(role="system", content=ANSWER_SYSTEM_PROMPT),
                    LLMMessage(role="user", content=user_prompt),
                ],
            )
        )
        return AgentAnswer(
            answer=response.content,
            model=self.llm_model,
            prompt_chars=len(ANSWER_SYSTEM_PROMPT) + len(user_prompt),
        )

    def stream_answer(self, *, query: str, decision: RouterDecision, context_text: str):
        user_prompt = self.build_answer_user_prompt(query=query, decision=decision, context_text=context_text)
        request = LLMRequest(
            model=self.llm_model,
            temperature=self.answer_temperature,
            messages=[
                LLMMessage(role="system", content=ANSWER_SYSTEM_PROMPT),
                LLMMessage(role="user", content=user_prompt),
            ],
        )
        yield from self.llm_client.stream_complete(request)

    def build_answer_user_prompt(self, *, query: str, decision: RouterDecision, context_text: str) -> str:
        return ANSWER_USER_PROMPT_TEMPLATE.format(
            query=query,
            command=decision.command.value,
            reason=decision.reason,
            context=context_text or "(no context)",
        )


def build_agent_context(
    *,
    notes_results: list[SearchResult],
    rg_results: list[GrepMatch],
    bm25_results: list[SearchResult],
    online_response: OnlineSearchResponse,
    tool_errors: list[dict[str, str]],
    max_chars_per_item: int,
    max_context_chars: int,
) -> tuple[str, list[dict[str, Any]]]:
    sections: list[str] = []
    items: list[dict[str, Any]] = []
    total_chars = 0

    def append_item(item: dict[str, Any], rendered: str) -> None:
        nonlocal total_chars
        if total_chars + len(rendered) > max_context_chars and sections:
            return
        items.append(item)
        sections.append(rendered)
        total_chars += len(rendered)

    if tool_errors:
        sections.append("## 检索状态")
        total_chars += len(sections[-1])
        for index, error in enumerate(tool_errors, start=1):
            item = {
                "citation_id": f"E{index}",
                "source_type": "tool_error",
                **error,
            }
            append_item(
                item,
                f"[E{index}]\ntool: {error['tool']}\nerror: {error['error_type']}: {error['message']}",
            )

    if notes_results:
        sections.append("## 本地语义检索")
        total_chars += len(sections[-1])
        for index, result in enumerate(notes_results, start=1):
            label = f"N{index}"
            item = search_result_item(
                label=label,
                result=result,
                retriever="hybrid",
                score_type=score_type_for_mode("hybrid"),
                max_chars=max_chars_per_item,
            )
            append_item(item, render_search_result_item(item))

    if rg_results:
        sections.append("## 精确/正则搜索")
        total_chars += len(sections[-1])
        for index, result in enumerate(rg_results, start=1):
            label = f"R{index}"
            text = truncate_text(result.text, max_chars_per_item)
            item = {
                "citation_id": label,
                "source_type": "local_rg",
                "path": result.path,
                "line": result.line,
                "score": None,
                "score_type": "regex",
                "text": text,
            }
            append_item(item, render_grep_item(item))

    if bm25_results:
        sections.append("## BM25 关键词检索")
        total_chars += len(sections[-1])
        for index, result in enumerate(bm25_results, start=1):
            label = f"B{index}"
            item = search_result_item(
                label=label,
                result=result,
                retriever="bm25",
                score_type=score_type_for_mode("bm25"),
                max_chars=max_chars_per_item,
            )
            append_item(item, render_search_result_item(item))

    if online_response.results or online_response.message:
        sections.append("## 网络结果")
        total_chars += len(sections[-1])
        if not online_response.results:
            item = {
                "citation_id": "W0",
                "source_type": "online_status",
                "provider": online_response.provider,
                "message": online_response.message,
            }
            append_item(item, f"[W0]\nprovider: {online_response.provider}\nmessage: {online_response.message}")
        for index, result in enumerate(online_response.results, start=1):
            label = f"W{index}"
            item = {
                "citation_id": label,
                "source_type": "online",
                "title": result.title,
                "url": result.url,
                "text": truncate_text(result.snippet, max_chars_per_item),
            }
            append_item(item, render_online_item(item))

    return "\n\n".join(sections), items


def search_result_item(
    *,
    label: str,
    result: SearchResult,
    retriever: str,
    score_type: str,
    max_chars: int,
) -> dict[str, Any]:
    chunk = result.chunk
    return {
        "citation_id": label,
        "source_type": f"local_{retriever}",
        "path": chunk.note_path,
        "heading": " > ".join(chunk.heading_path) if chunk.heading_path else "",
        "lines": line_range(chunk.start_line, chunk.end_line),
        "score": round(float(result.score), 6),
        "score_type": score_type,
        "chunk_id": chunk.chunk_id,
        "text": truncate_text(chunk.text.strip(), max_chars),
    }


def render_search_result_item(item: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"[{item['citation_id']}]",
            f"source: {item['path']}",
            f"heading: {item.get('heading') or '(no heading)'}",
            f"lines: {item.get('lines') or ''}",
            f"score: {item.get('score')} ({item.get('score_type')})",
            "content:",
            item["text"],
        ]
    )


def render_grep_item(item: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"[{item['citation_id']}]",
            f"source: {item['path']}",
            f"line: {item['line']}",
            "content:",
            item["text"],
        ]
    )


def render_online_item(item: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"[{item['citation_id']}]",
            f"title: {item['title']}",
            f"url: {item['url']}",
            "content:",
            item["text"],
        ]
    )


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def line_range(start_line: int | None, end_line: int | None) -> str:
    if start_line is None or end_line is None:
        return ""
    return f"{start_line}-{end_line}"


def elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def regex_pattern_from_decision(decision: RouterDecision) -> str:
    value = decision.tool_args.get("regex_pattern", "")
    return str(value).strip()


def valid_regex_pattern(pattern: str) -> bool:
    stripped = pattern.strip()
    return len(stripped) >= 2


def timed_call(fn, *args, **kwargs):
    started_at = time.perf_counter()
    value = fn(*args, **kwargs)
    return value, elapsed_ms(started_at)


def empty_tool_telemetry() -> dict[str, Any]:
    return {
        "called": False,
        "latency_ms": None,
        "result_count": 0,
        "failed": False,
        "error_type": None,
        "message": None,
    }


def update_tool_success(telemetry: dict[str, Any], latency_ms: int, result_count: int) -> None:
    telemetry.update(
        {
            "called": True,
            "latency_ms": latency_ms,
            "result_count": result_count,
            "failed": False,
            "error_type": None,
            "message": None,
        }
    )


def update_tool_failure(telemetry: dict[str, Any], exc: Exception) -> None:
    telemetry.update(
        {
            "called": True,
            "failed": True,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    )


def update_tool_result_count(telemetry: dict[str, Any], results) -> None:
    if telemetry.get("called") and not telemetry.get("failed"):
        telemetry["result_count"] = len(results)


def count_distinct_local_files(context_items: list[dict[str, Any]]) -> int:
    return len(
        {
            str(item.get("path", "")).strip()
            for item in context_items
            if str(item.get("source_type", "")).startswith("local_") and str(item.get("path", "")).strip()
        }
    )


def tool_error(tool: str, exc: Exception) -> dict[str, str]:
    return {
        "tool": tool,
        "error_type": type(exc).__name__,
        "message": str(exc),
    }


def timed_future_result_or_empty(
    tool: str,
    future: Future,
    errors: list[dict[str, str]],
    telemetry: dict[str, dict[str, Any]],
):
    try:
        results, latency_ms = future.result()
        update_tool_success(telemetry[tool], latency_ms, len(results))
        return results
    except Exception as exc:
        errors.append(tool_error(tool, exc))
        update_tool_failure(telemetry[tool], exc)
        return []


def timed_future_result_or_online_error(
    tool: str,
    future: Future,
    errors: list[dict[str, str]],
    telemetry: dict[str, dict[str, Any]],
) -> OnlineSearchResponse:
    try:
        response, latency_ms = future.result()
        update_tool_success(telemetry[tool], latency_ms, len(response.results))
        if response.message and not response.enabled:
            telemetry[tool]["message"] = response.message
        return response
    except Exception as exc:
        errors.append(tool_error(tool, exc))
        update_tool_failure(telemetry[tool], exc)
        return OnlineSearchResponse(
            enabled=False,
            provider="error",
            results=[],
            message=f"{type(exc).__name__}: {exc}",
        )


def agent_run_result_to_dict(result: AgentRunResult) -> dict[str, Any]:
    return {
        "query": result.query,
        "command": result.command.value,
        "router_decision": router_decision_to_dict(result.router_decision),
        "answer": asdict(result.answer),
        "timing": result.timing,
        "telemetry": result.telemetry,
        "tool_errors": result.tool_errors,
        "context_text": result.context_text,
        "context_items": result.context_items,
        "retrieval": {
            "notes": [
                search_result_item(
                    label=f"N{index}",
                    result=item,
                    retriever="hybrid",
                    score_type=score_type_for_mode("hybrid"),
                    max_chars=500,
                )
                for index, item in enumerate(result.notes_results, start=1)
            ],
            "rg": grep_matches_to_dict(result.rg_results),
            "bm25": [
                search_result_item(
                    label=f"B{index}",
                    result=item,
                    retriever="bm25",
                    score_type=score_type_for_mode("bm25"),
                    max_chars=500,
                )
                for index, item in enumerate(result.bm25_results, start=1)
            ],
            "online": online_response_to_dict(result.online_response),
        },
    }


def agent_retrieval_result_to_dict(result: AgentRetrievalResult) -> dict[str, Any]:
    return {
        "query": result.query,
        "command": result.command.value,
        "router_decision": router_decision_to_dict(result.router_decision),
        "timing": result.timing,
        "telemetry": result.telemetry,
        "tool_errors": result.tool_errors,
        "context_text": result.context_text,
        "context_items": result.context_items,
        "retrieval": {
            "notes": [
                search_result_item(
                    label=f"N{index}",
                    result=item,
                    retriever="hybrid",
                    score_type=score_type_for_mode("hybrid"),
                    max_chars=500,
                )
                for index, item in enumerate(result.notes_results, start=1)
            ],
            "rg": grep_matches_to_dict(result.rg_results),
            "bm25": [
                search_result_item(
                    label=f"B{index}",
                    result=item,
                    retriever="bm25",
                    score_type=score_type_for_mode("bm25"),
                    max_chars=500,
                )
                for index, item in enumerate(result.bm25_results, start=1)
            ],
            "online": online_response_to_dict(result.online_response),
        },
    }
