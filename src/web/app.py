from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from knowledge_base_agent.config import load_llm_config
from knowledge_base_agent.llm import create_llm_client
from services.rag.agent_answer import (
    AgentAnswerConfig,
    AgentAnswerPipeline,
    agent_retrieval_result_to_dict,
)
from services.rag.intent_router import ConversationCommand, LLMIntentRouter
from services.rag.online_search import OnlineSearchClient
from services.rag.reranker import DEFAULT_RERANKER_MODEL
from services.rag.search_service import SearchOptions, SearchService
from services.rag.vector_store_loader import (
    DEFAULT_HNSW_EF_CONSTRUCTION,
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_HNSW_M,
)


class SearchRequest(BaseModel):
    query: str = Field(default="")
    mode: str = Field(default="hybrid")
    top_k: int = Field(default=10, ge=1, le=100)
    enable_rewrite: bool = Field(default=False)
    rewrite_confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    rewrite_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    dense_top_k: int = Field(default=50, ge=1, le=500)
    bm25_top_k: int = Field(default=50, ge=1, le=500)
    rrf_k: int = Field(default=60, ge=1, le=500)
    reranker_type: str = Field(default="off")
    reranker_model: str = Field(default=DEFAULT_RERANKER_MODEL)
    rerank_candidates: int = Field(default=50, ge=1, le=500)
    rerank_batch_size: int = Field(default=16, ge=1, le=128)
    rerank_max_length: int = Field(default=512, ge=64, le=4096)
    include_debug: bool = Field(default=True)


class AgentRequest(BaseModel):
    query: str = Field(default="")
    command: str = Field(default="auto")
    notes_top_k: int = Field(default=5, ge=1, le=50)
    regex_top_k: int = Field(default=8, ge=1, le=50)
    bm25_top_k: int = Field(default=8, ge=1, le=50)
    dense_top_k: int = Field(default=50, ge=1, le=500)
    hybrid_bm25_top_k: int = Field(default=50, ge=1, le=500)
    rrf_k: int = Field(default=60, ge=1, le=500)
    max_chars_per_item: int = Field(default=1000, ge=100, le=5000)
    max_context_chars: int = Field(default=8000, ge=1000, le=30000)
    online_provider: str | None = Field(default=None)
    online_top_k: int = Field(default=5, ge=1, le=20)
    speculative_notes_search: bool = Field(default=True)


def create_app(
    index_path: Path,
    bm25_index_path: Path | None,
    model_name: str,
    project_root: Path,
    vault_path: Path | None = None,
    embedding_provider: str = "local",
    embed_batch_size: int = 32,
    max_seq_length: int | None = None,
    vector_index: str = "flat",
    hnsw_m: int = DEFAULT_HNSW_M,
    hnsw_ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION,
    hnsw_ef_search: int = DEFAULT_HNSW_EF_SEARCH,
) -> FastAPI:
    app = FastAPI(title="Knowledge Agent Search")
    service = SearchService(
        index_path=index_path,
        bm25_index_path=bm25_index_path,
        model_name=model_name,
        project_root=project_root,
        embedding_provider=embedding_provider,
        embed_batch_size=embed_batch_size,
        max_seq_length=max_seq_length,
        vector_index=vector_index,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/chat", response_class=HTMLResponse)
    def chat() -> str:
        return CHAT_HTML

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "index": str(index_path),
            "bm25_index": str(service.bm25_index_path),
            "model": model_name,
            "embedding_provider": embedding_provider,
            "vector_index": vector_index,
            "hnsw": {
                "m": hnsw_m,
                "ef_construction": hnsw_ef_construction,
                "ef_search": hnsw_ef_search,
            },
        }

    @app.get("/api/chat/starters")
    def chat_starters() -> dict[str, Any]:
        return {
            "starters": [
                "Redis Stream 任务队列怎么消费事件？",
                "进程是什么？和线程有什么区别？",
                "FastAPI CORS 跨域怎么配置？",
                "我有哪些关于 LLM Agent 架构的笔记？",
                "WinError 10060 可能是什么原因？",
            ]
        }

    @app.post("/api/search")
    def search(request: SearchRequest) -> dict[str, Any]:
        try:
            response = service.search(SearchOptions(**request.model_dump()))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return asdict(response)

    @app.post("/api/agent/stream")
    def agent_stream(request: AgentRequest) -> StreamingResponse:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="query is required")
        if vault_path is None:
            raise HTTPException(status_code=400, detail="vault path is required for chat agent")

        def event_stream():
            try:
                yield sse_event("status", {"stage": "started", "message": "request received"})

                llm_config = load_llm_config(project_root)
                llm_client = create_llm_client(llm_config)
                router = build_web_router(request.command, llm_client, llm_config)

                yield sse_event("status", {"stage": "pipeline_ready", "message": "pipeline initialized"})

                manager = service.build_manager(
                    SearchOptions(
                        query=request.query,
                        mode="hybrid",
                        dense_top_k=request.dense_top_k,
                        bm25_top_k=request.hybrid_bm25_top_k,
                        rrf_k=request.rrf_k,
                    )
                )
                pipeline = AgentAnswerPipeline(
                    router=router,
                    llm_client=llm_client,
                    llm_model=llm_config.model,
                    manager=manager,
                    vault_root=vault_path,
                    online_client=OnlineSearchClient(provider=request.online_provider),
                    config=AgentAnswerConfig(
                        notes_top_k=request.notes_top_k,
                        regex_top_k=request.regex_top_k,
                        bm25_top_k=request.bm25_top_k,
                        dense_top_k=request.dense_top_k,
                        hybrid_bm25_top_k=request.hybrid_bm25_top_k,
                        rrf_k=request.rrf_k,
                        max_chars_per_item=request.max_chars_per_item,
                        max_context_chars=request.max_context_chars,
                        online_top_k=request.online_top_k,
                        speculative_notes_search=request.speculative_notes_search,
                    ),
                    answer_temperature=llm_config.temperature,
                )

                yield sse_event("status", {"stage": "retrieval_started", "message": "routing and retrieval started"})
                retrieval = pipeline.retrieve(request.query)
                payload = agent_retrieval_result_to_dict(retrieval)
                reference_summary = build_reference_summary(payload)

                yield sse_event("router", payload["router_decision"])
                yield sse_event(
                    "status",
                    {
                        "stage": "retrieval_finished",
                        "message": reference_summary["message"],
                        "summary": reference_summary,
                    },
                )
                yield sse_event(
                    "retrieval",
                    {
                        "summary": {
                            "notes": len(payload["retrieval"]["notes"]),
                            "rg": len(payload["retrieval"]["rg"]),
                            "bm25": len(payload["retrieval"]["bm25"]),
                            "online": len(payload["retrieval"]["online"]["results"]),
                            "tool_errors": len(payload["tool_errors"]),
                            "local_references": reference_summary["local_references"],
                            "distinct_files": reference_summary["distinct_files"],
                        },
                        "reference_summary": reference_summary,
                        "tool_errors": payload["tool_errors"],
                    },
                )
                yield sse_event("context", {"items": payload["context_items"]})

                yield sse_event("status", {"stage": "answer_started", "message": "answer streaming started"})
                answer_parts: list[str] = []
                answer_started_at = current_time()
                first_delta_ms: int | None = None
                for delta in pipeline.stream_answer(
                    query=request.query,
                    decision=retrieval.router_decision,
                    context_text=retrieval.context_text,
                ):
                    if first_delta_ms is None:
                        first_delta_ms = elapsed_since(answer_started_at)
                    answer_parts.append(delta)
                    yield sse_event("answer_delta", {"text": delta})

                answer_text = "".join(answer_parts)
                answer_ms = elapsed_since(answer_started_at)
                timing = {
                    **payload["timing"],
                    "answer_ms": answer_ms,
                    "total_ms": payload["timing"].get("total_ms", 0) + answer_ms,
                }
                telemetry = {
                    **payload["telemetry"],
                    "generation": {
                        "ttft_ms": first_delta_ms,
                        "answer_ms": answer_ms,
                        "output_chars": len(answer_text),
                        "prompt_chars": None,
                    },
                    "total_ms": timing["total_ms"],
                }
                yield sse_event(
                    "answer",
                    {
                        "answer": answer_text,
                        "model": llm_config.model,
                        "prompt_chars": None,
                    },
                )
                yield sse_event("done", {"timing": timing, "telemetry": telemetry, "command": payload["command"]})
            except Exception as exc:
                yield sse_event("error", {"message": str(exc), "error_type": type(exc).__name__})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def build_web_router(command: str, llm_client, llm_config):
    forced_command = None if command == "auto" else ConversationCommand(command)
    return LLMIntentRouter(
        client=llm_client,
        model=llm_config.model,
        temperature=0.0,
        forced_command=forced_command,
    )


def sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def build_reference_summary(payload: dict[str, Any]) -> dict[str, Any]:
    local_items = [
        item
        for item in payload.get("context_items", [])
        if str(item.get("source_type", "")).startswith("local_")
    ]
    distinct_files = sorted(
        {
            str(item.get("path", "")).strip()
            for item in local_items
            if str(item.get("path", "")).strip()
        }
    )
    heading_set: set[str] = set()
    for item in local_items:
        heading = clean_heading(str(item.get("heading", "")))
        if heading:
            heading_set.add(heading)
    distinct_headings = sorted(heading_set)
    heading_preview = distinct_headings[:5]

    if local_items:
        message = f"Found {len(local_items)} references across {len(distinct_files)} files"
        if heading_preview:
            message += ": " + ", ".join(heading_preview)
    else:
        message = "No local references found"

    return {
        "local_references": len(local_items),
        "distinct_files": len(distinct_files),
        "distinct_headings": len(distinct_headings),
        "files": distinct_files[:10],
        "headings": heading_preview,
        "message": message,
    }


def clean_heading(heading: str) -> str:
    cleaned = re.sub(r"\s*\{[^}]*\}", "", heading).strip()
    if not cleaned:
        return ""
    short = cleaned.rsplit(" > ", 1)[-1].lstrip("#").strip()
    return short or cleaned.strip()


def current_time() -> float:
    import time

    return time.perf_counter()


def elapsed_since(started_at: float) -> int:
    import time

    return int((time.perf_counter() - started_at) * 1000)


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Knowledge Agent Search</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --text: #171717;
      --muted: #666666;
      --line: #d9d9d2;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --danger: #b42318;
      --chip: #eef2f1;
      --code: #f2f4f3;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }

    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 22px 0 40px;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }

    h1 {
      font-size: 22px;
      line-height: 1.2;
      margin: 0;
      font-weight: 700;
    }

    .status {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }

    .search-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }

    .search-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 96px 112px;
      gap: 10px;
      align-items: center;
    }

    input, select, button {
      font: inherit;
    }

    input[type="search"], input[type="number"], input[type="text"], select {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: #fff;
      color: var(--text);
    }

    button {
      min-height: 38px;
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: var(--accent);
      color: white;
      padding: 8px 12px;
      cursor: pointer;
      font-weight: 600;
    }

    button.secondary {
      background: white;
      color: var(--accent-dark);
      border-color: var(--line);
      font-weight: 500;
    }

    button:disabled {
      opacity: .58;
      cursor: not-allowed;
    }

    .inline-options {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      align-items: center;
      margin-top: 12px;
      color: var(--muted);
      font-size: 14px;
    }

    label.check {
      display: inline-flex;
      gap: 7px;
      align-items: center;
    }

    details {
      margin-top: 12px;
      border-top: 1px solid var(--line);
      padding-top: 10px;
    }

    summary {
      cursor: pointer;
      color: var(--accent-dark);
      font-weight: 600;
      width: fit-content;
    }

    .advanced-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }

    .field label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }

    .meta-panel {
      margin: 14px 0;
      display: grid;
      gap: 8px;
    }

    .meta-box {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 14px;
    }

    .meta-box strong {
      display: inline-block;
      margin-right: 8px;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }

    .chip {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 8px;
      border-radius: 999px;
      background: var(--chip);
      border: 1px solid var(--line);
      font-size: 12px;
      color: #2d3b39;
    }

    .results {
      display: grid;
      gap: 12px;
    }

    .stage-tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 12px;
    }

    .stage-tab {
      min-height: 32px;
      border-color: var(--line);
      background: var(--panel);
      color: var(--accent-dark);
      font-size: 13px;
      font-weight: 600;
    }

    .stage-tab.is-active {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }

    .result-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px;
    }

    .result-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
    }

    .title {
      font-weight: 700;
      font-size: 16px;
      margin-bottom: 5px;
    }

    .path, .heading, .line-score {
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }

    .preview {
      margin-top: 10px;
      line-height: 1.65;
      font-size: 14px;
      white-space: pre-wrap;
    }

    .full-text {
      display: none;
      margin-top: 10px;
      background: var(--code);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      white-space: pre-wrap;
      line-height: 1.62;
      font-size: 13px;
      overflow-x: auto;
    }

    .actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 10px;
    }

    .error {
      color: var(--danger);
      border-color: #f3b5ad;
      background: #fff7f5;
    }

    @media (max-width: 760px) {
      main {
        width: min(100vw - 20px, 1180px);
        padding-top: 12px;
      }

      .topbar, .result-head {
        display: block;
      }

      .status {
        margin-top: 6px;
      }

      .search-row {
        grid-template-columns: 1fr;
      }

      .advanced-grid {
        grid-template-columns: 1fr 1fr;
      }
    }
  </style>
</head>
<body>
  <main>
    <div class="topbar">
      <h1>Knowledge Agent Search</h1>
      <div id="status" class="status">ready</div>
    </div>

    <section class="search-panel">
      <div class="search-row">
        <input id="query" type="search" placeholder="输入搜索内容" autocomplete="off" />
        <input id="topK" type="number" min="1" max="100" value="10" />
        <button id="searchBtn">搜索</button>
      </div>

      <div class="inline-options">
        <label class="check"><input id="enableRewrite" type="checkbox" /> query rewrite</label>
      </div>

      <details>
        <summary>高级选项</summary>
        <div class="advanced-grid">
          <div class="field">
            <label for="mode">mode</label>
            <select id="mode">
              <option value="hybrid" selected>hybrid</option>
              <option value="dense">dense</option>
              <option value="bm25">bm25</option>
              <option value="hybrid-rerank">hybrid-rerank</option>
            </select>
          </div>
          <div class="field">
            <label for="rerankerType">reranker</label>
            <select id="rerankerType">
              <option value="off" selected>off</option>
              <option value="local">local</option>
              <option value="dashscope">dashscope</option>
            </select>
          </div>
          <div class="field">
            <label for="rerankerModel">reranker_model</label>
            <input id="rerankerModel" type="text" value="BAAI/bge-reranker-v2-m3" />
          </div>
          <div class="field">
            <label for="denseTopK">dense_top_k</label>
            <input id="denseTopK" type="number" min="1" max="500" value="50" />
          </div>
          <div class="field">
            <label for="bm25TopK">bm25_top_k</label>
            <input id="bm25TopK" type="number" min="1" max="500" value="50" />
          </div>
          <div class="field">
            <label for="rrfK">rrf_k</label>
            <input id="rrfK" type="number" min="1" max="500" value="60" />
          </div>
          <div class="field">
            <label for="rerankCandidates">rerank_candidates</label>
            <input id="rerankCandidates" type="number" min="1" max="500" value="50" />
          </div>
          <div class="field">
            <label for="rewriteThreshold">rewrite_threshold</label>
            <input id="rewriteThreshold" type="number" min="0" max="1" step="0.05" value="0.75" />
          </div>
          <div class="field">
            <label for="rewriteWeight">rewrite_weight</label>
            <input id="rewriteWeight" type="number" min="0" max="1" step="0.05" value="0.7" />
          </div>
        </div>
      </details>
    </section>

    <section id="meta" class="meta-panel"></section>
    <section id="stageTabs" class="stage-tabs"></section>
    <section id="results" class="results"></section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    let currentSearchData = null;
    let activeStageKey = "final";

    $("searchBtn").addEventListener("click", runSearch);
    $("rerankerType").addEventListener("change", updateRerankerModelDefault);
    $("query").addEventListener("keydown", (event) => {
      if (event.key === "Enter") runSearch();
    });

    async function runSearch() {
      const query = $("query").value.trim();
      if (!query) return;

      setBusy(true);
      $("meta").innerHTML = "";
      $("stageTabs").innerHTML = "";
      $("results").innerHTML = "";

      const payload = {
        query,
        mode: $("mode").value,
        top_k: numberValue("topK"),
        enable_rewrite: $("enableRewrite").checked,
        rewrite_confidence_threshold: numberValue("rewriteThreshold"),
        rewrite_weight: numberValue("rewriteWeight"),
        dense_top_k: numberValue("denseTopK"),
        bm25_top_k: numberValue("bm25TopK"),
        rrf_k: numberValue("rrfK"),
        reranker_type: $("rerankerType").value,
        reranker_model: $("rerankerModel").value.trim(),
        rerank_candidates: numberValue("rerankCandidates"),
        include_debug: true
      };

      try {
        const response = await fetch("/api/search", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "search failed");
        }
        currentSearchData = data;
        activeStageKey = "final";
        renderMeta(data);
        renderStageTabs(data);
        renderActiveStage();
      } catch (error) {
        $("meta").innerHTML = `<div class="meta-box error">${escapeHtml(error.message)}</div>`;
      } finally {
        setBusy(false);
      }
    }

    function renderMeta(data) {
      const variants = (data.variants || []).map((variant) =>
        `<span class="chip">${escapeHtml(variant.source)} · ${escapeHtml(variant.text)} · ${variant.weight}</span>`
      ).join("");

      let rewriteHtml = "";
      if (data.rewrite_enabled) {
        const rewrite = data.rewrite;
        if (rewrite) {
          const risk = (rewrite.risk_notes || []).map((item) =>
            `<span class="chip">${escapeHtml(item)}</span>`
          ).join("");
          rewriteHtml = `
            <div class="meta-box">
              <strong>rewrite</strong>
              <span>${data.rewrite_used ? "used" : "not used"}</span>
              <div class="chips">
                <span class="chip">${escapeHtml(rewrite.rewrite_type)}</span>
                <span class="chip">confidence ${rewrite.confidence}</span>
                ${rewrite.rewritten_query ? `<span class="chip">${escapeHtml(rewrite.rewritten_query)}</span>` : ""}
              </div>
              ${risk ? `<div class="chips">${risk}</div>` : ""}
            </div>
          `;
        } else {
          rewriteHtml = `<div class="meta-box"><strong>rewrite</strong><span>not available</span></div>`;
        }
      }

      $("meta").innerHTML = `
        <div class="meta-box">
          <strong>${escapeHtml(data.mode)}</strong>
          <span>${data.results.length} results · ${data.elapsed_ms} ms</span>
          <div class="chips">${variants}</div>
        </div>
        ${rewriteHtml}
      `;
    }

    function renderStageTabs(data) {
      const stages = stageList(data);
      if (stages.length <= 1) {
        $("stageTabs").innerHTML = "";
        return;
      }

      $("stageTabs").innerHTML = stages.map((stage) => `
        <button class="stage-tab ${stage.key === activeStageKey ? "is-active" : ""}" data-stage-key="${escapeHtml(stage.key)}">
          ${escapeHtml(stage.label)} ${stage.results.length}
        </button>
      `).join("");

      document.querySelectorAll("[data-stage-key]").forEach((button) => {
        button.addEventListener("click", () => {
          activeStageKey = button.dataset.stageKey;
          renderStageTabs(currentSearchData);
          renderActiveStage();
        });
      });
    }

    function renderActiveStage() {
      const stages = stageList(currentSearchData);
      const stage = stages.find((item) => item.key === activeStageKey) || stages[0];
      renderResults(stage ? stage.results : []);
    }

    function stageList(data) {
      const debugStages = data?.debug?.stages;
      if (Array.isArray(debugStages) && debugStages.length) {
        return debugStages;
      }
      return [
        {
          key: "final",
          label: "最终结果",
          query: data?.query || null,
          query_source: "original",
          retriever: data?.mode || null,
          results: data?.results || []
        }
      ];
    }

    function renderResults(results) {
      if (!results.length) {
        $("results").innerHTML = `<div class="meta-box">no results</div>`;
        return;
      }

      $("results").innerHTML = results.map((item) => {
        const heading = item.heading ? `<div class="heading">heading: ${escapeHtml(item.heading)}</div>` : "";
        const lines = item.start_line && item.end_line ? `${item.start_line}-${item.end_line}` : "";
        return `
          <article class="result-card">
            <div class="result-head">
              <div>
                <div class="title">#${item.rank} ${escapeHtml(item.title)}</div>
                <div class="path">${escapeHtml(item.note_path)}</div>
                ${heading}
              </div>
              <div class="line-score">score ${item.score}${lines ? ` · lines ${lines}` : ""}</div>
            </div>
            <div class="preview">${escapeHtml(item.preview)}</div>
            <pre class="full-text">${escapeHtml(item.text)}</pre>
            <div class="actions">
              <button class="secondary" data-copy="${escapeHtml(item.note_path)}">复制路径</button>
              <button class="secondary" data-toggle>展开</button>
            </div>
          </article>
        `;
      }).join("");

      document.querySelectorAll("[data-copy]").forEach((button) => {
        button.addEventListener("click", async () => {
          await navigator.clipboard.writeText(button.dataset.copy);
          button.textContent = "已复制";
          setTimeout(() => button.textContent = "复制路径", 900);
        });
      });

      document.querySelectorAll("[data-toggle]").forEach((button) => {
        button.addEventListener("click", () => {
          const fullText = button.closest(".result-card").querySelector(".full-text");
          const visible = fullText.style.display === "block";
          fullText.style.display = visible ? "none" : "block";
          button.textContent = visible ? "展开" : "收起";
        });
      });
    }

    function numberValue(id) {
      return Number($(id).value);
    }

    function updateRerankerModelDefault() {
      const type = $("rerankerType").value;
      const current = $("rerankerModel").value.trim();
      if (type === "dashscope" && (!current || current === "BAAI/bge-reranker-v2-m3")) {
        $("rerankerModel").value = "qwen3-rerank";
      }
      if (type === "local" && (!current || current === "qwen3-rerank")) {
        $("rerankerModel").value = "BAAI/bge-reranker-v2-m3";
      }
      if (type === "off" && !current) {
        $("rerankerModel").value = "BAAI/bge-reranker-v2-m3";
      }
    }

    function setBusy(isBusy) {
      $("searchBtn").disabled = isBusy;
      $("status").textContent = isBusy ? "searching" : "ready";
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }
  </script>
</body>
</html>
"""


CHAT_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Knowledge Agent Chat</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --text: #171717;
      --muted: #666666;
      --line: #d9d9d2;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --soft: #eef2f1;
      --danger: #b42318;
      --code: #f2f4f3;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }

    main {
      width: min(980px, calc(100vw - 28px));
      margin: 0 auto;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 12px;
      padding: 18px 0;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
    }

    .topbar h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
    }

    .topbar a {
      color: var(--accent-dark);
      text-decoration: none;
      font-size: 14px;
      font-weight: 600;
    }

    .status {
      color: var(--muted);
      font-size: 13px;
      margin-top: 4px;
    }

    .messages {
      overflow-y: auto;
      display: grid;
      align-content: start;
      gap: 12px;
      padding: 2px;
    }

    .message {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 13px;
    }

    .message.user {
      margin-left: auto;
      width: min(760px, 90%);
      background: #fdfdfb;
    }

    .message.assistant {
      margin-right: auto;
      width: min(860px, 100%);
    }

    .role {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      margin-bottom: 8px;
    }

    .answer {
      line-height: 1.68;
      font-size: 15px;
    }

    .status-line {
      display: inline-block;
      margin: 0 0 10px;
      padding: 5px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--soft);
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }

    .markdown-body h2,
    .markdown-body h3,
    .markdown-body h4,
    .markdown-body h5 {
      margin: 12px 0 7px;
      line-height: 1.35;
    }

    .markdown-body h2 {
      font-size: 18px;
    }

    .markdown-body h3 {
      font-size: 16px;
    }

    .markdown-body p {
      margin: 7px 0;
    }

    .markdown-body ul {
      margin: 7px 0;
      padding-left: 22px;
    }

    .markdown-body li {
      margin: 4px 0;
    }

    .markdown-body code {
      background: var(--code);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 1px 4px;
      font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
      font-size: 0.92em;
    }

    .markdown-body pre {
      margin: 10px 0;
      padding: 10px;
      background: var(--code);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow-x: auto;
    }

    .markdown-body pre code {
      border: 0;
      padding: 0;
      background: transparent;
    }

    .citation {
      display: inline-block;
      color: var(--accent-dark);
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 0 6px;
      font-weight: 700;
      line-height: 1.4;
    }

    details {
      margin-top: 12px;
      border-top: 1px solid var(--line);
      padding-top: 10px;
    }

    summary {
      cursor: pointer;
      color: var(--accent-dark);
      font-weight: 700;
      font-size: 14px;
    }

    .reference-list {
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }

    .reference {
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      font-size: 13px;
    }

    .reference .ref-id {
      font-weight: 800;
      color: var(--accent-dark);
    }

    .reference .path {
      overflow-wrap: anywhere;
      margin-top: 4px;
    }

    .reference pre {
      white-space: pre-wrap;
      margin: 8px 0 0;
      padding: 8px;
      background: var(--code);
      border-radius: 6px;
      max-height: 180px;
      overflow: auto;
      line-height: 1.55;
    }

    .debug-box {
      margin-top: 10px;
      background: var(--code);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      white-space: pre-wrap;
      overflow-x: auto;
      font-size: 12px;
      line-height: 1.5;
    }

    .composer {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }

    .starters {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }

    .starter-chip {
      min-height: 30px;
      border-color: var(--line);
      background: var(--soft);
      color: var(--accent-dark);
      font-size: 13px;
      font-weight: 600;
      padding: 5px 9px;
    }

    textarea {
      width: 100%;
      min-height: 78px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      font: inherit;
      line-height: 1.55;
    }

    .composer-row {
      display: flex;
      gap: 10px;
      align-items: center;
      margin-top: 10px;
      flex-wrap: wrap;
    }

    select, input, button {
      font: inherit;
    }

    select, input[type="number"], input[type="text"] {
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 7px 9px;
    }

    button {
      min-height: 36px;
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      padding: 7px 14px;
      cursor: pointer;
    }

    button.secondary {
      background: white;
      color: var(--accent-dark);
      border-color: var(--line);
    }

    button:disabled {
      opacity: .58;
      cursor: not-allowed;
    }

    .field {
      display: inline-flex;
      gap: 6px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
    }

    .error {
      color: var(--danger);
    }

    @media (max-width: 720px) {
      main {
        width: min(100vw - 18px, 980px);
        padding: 10px 0;
      }
      .topbar {
        display: block;
      }
      .message.user, .message.assistant {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div class="topbar">
        <div>
          <h1>Knowledge Agent Chat</h1>
          <div id="status" class="status">ready</div>
        </div>
        <a href="/">Search</a>
      </div>
    </header>

    <section id="messages" class="messages"></section>

    <section class="composer">
      <div id="starters" class="starters"></div>
      <textarea id="query" placeholder="输入问题，例如：Redis Stream 任务队列怎么消费事件"></textarea>
      <div class="composer-row">
        <button id="sendBtn">发送</button>
        <label class="field">command
          <select id="command">
            <option value="auto" selected>auto</option>
            <option value="Notes">Notes</option>
            <option value="RegexSearchFiles">RegexSearchFiles</option>
            <option value="Notes+Online">Notes+Online</option>
          </select>
        </label>
        <label class="field">top
          <input id="notesTopK" type="number" min="1" max="50" value="5" />
        </label>
        <label class="field">online
          <input id="onlineProvider" type="text" placeholder="disabled / tavily / brave" />
        </label>
        <label class="field">
          <input id="speculative" type="checkbox" checked />
          speculative notes
        </label>
      </div>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const messages = $("messages");
    let currentAssistant = null;
    let currentPayload = {
      router: null,
      retrieval: null,
      context: null,
      answer: null,
      done: null,
      errors: []
    };

    $("sendBtn").addEventListener("click", sendMessage);
    loadStarters();
    $("query").addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        sendMessage();
      }
    });

    async function loadStarters() {
      try {
        const response = await fetch("/api/chat/starters");
        const data = await response.json();
        const starters = Array.isArray(data.starters) ? data.starters : [];
        $("starters").innerHTML = starters.map((item) =>
          `<button class="starter-chip" data-starter="${escapeHtml(item)}">${escapeHtml(item)}</button>`
        ).join("");
        document.querySelectorAll("[data-starter]").forEach((button) => {
          button.addEventListener("click", () => {
            $("query").value = button.dataset.starter || "";
            $("query").focus();
          });
        });
      } catch {
        $("starters").innerHTML = "";
      }
    }

    async function sendMessage() {
      const query = $("query").value.trim();
      if (!query) return;

      appendUserMessage(query);
      currentAssistant = appendAssistantMessage();
      currentPayload = {router: null, retrieval: null, context: null, answer: null, done: null, errors: []};
      setBusy(true, "starting");

      const body = {
        query,
        command: $("command").value,
        notes_top_k: numberValue("notesTopK"),
        online_provider: $("onlineProvider").value.trim() || null,
        speculative_notes_search: $("speculative").checked
      };

      try {
        const response = await fetch("/api/agent/stream", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(body)
        });
        if (!response.ok || !response.body) {
          const data = await response.json().catch(() => ({}));
          throw new Error(data.detail || "request failed");
        }
        await readSseStream(response.body);
      } catch (error) {
        currentPayload.errors.push({message: error.message});
        renderAssistant(currentAssistant, currentPayload);
      } finally {
        setBusy(false, "ready");
      }
    }

    async function readSseStream(stream) {
      const reader = stream.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      while (true) {
        const {value, done} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream: true});
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        for (const part of parts) {
          handleSseBlock(part);
        }
      }
      if (buffer.trim()) handleSseBlock(buffer);
    }

    function handleSseBlock(block) {
      let eventName = "message";
      const dataLines = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (!dataLines.length) return;
      const data = JSON.parse(dataLines.join("\n"));

      if (eventName === "status") {
        setBusy(true, data.message || data.stage || "running");
        if (data.stage === "retrieval_finished" && data.summary) {
          currentPayload.retrievalStatus = data.summary;
        }
      } else if (eventName === "router") {
        currentPayload.router = data;
      } else if (eventName === "retrieval") {
        currentPayload.retrieval = data;
      } else if (eventName === "context") {
        currentPayload.context = data;
      } else if (eventName === "answer_delta") {
        if (!currentPayload.answer) currentPayload.answer = {answer: "", model: "", prompt_chars: null};
        currentPayload.answer.answer += data.text || "";
      } else if (eventName === "answer") {
        currentPayload.answer = data;
      } else if (eventName === "done") {
        currentPayload.done = data;
        setBusy(false, "done");
      } else if (eventName === "error") {
        currentPayload.errors.push(data);
      }
      renderAssistant(currentAssistant, currentPayload);
      scrollToBottom();
    }

    function appendUserMessage(text) {
      const node = document.createElement("article");
      node.className = "message user";
      node.innerHTML = `<div class="role">User</div><div class="answer">${escapeHtml(text)}</div>`;
      messages.appendChild(node);
      scrollToBottom();
    }

    function appendAssistantMessage() {
      const node = document.createElement("article");
      node.className = "message assistant";
      node.innerHTML = `<div class="role">Assistant</div><div class="answer">正在处理...</div>`;
      messages.appendChild(node);
      scrollToBottom();
      return node;
    }

    function renderAssistant(node, payload) {
      if (!node) return;
      const answerText = payload.answer?.answer || statusText(payload);
      const references = payload.context?.items || [];
      const debug = {
        router: payload.router,
        retrieval: payload.retrieval,
        retrievalStatus: payload.retrievalStatus,
        done: payload.done,
        errors: payload.errors
      };
      const retrievalStatus = payload.retrievalStatus
        ? `<div class="status-line">${escapeHtml(payload.retrievalStatus.message)}</div>`
        : "";
      node.innerHTML = `
        <div class="role">Assistant</div>
        ${retrievalStatus}
        <div class="answer markdown-body ${payload.errors.length ? "error" : ""}">${renderMarkdown(answerText)}</div>
        ${renderReferences(references)}
        <details>
          <summary>Debug</summary>
          <pre class="debug-box">${escapeHtml(JSON.stringify(debug, null, 2))}</pre>
        </details>
      `;
    }

    function renderReferences(items) {
      if (!items.length) return "";
      return `
        <details>
          <summary>References (${items.length})</summary>
          <div class="reference-list">
            ${items.map(renderReference).join("")}
          </div>
        </details>
      `;
    }

    function renderReference(item) {
      const path = item.path || item.url || item.provider || "";
      const lines = item.lines ? ` lines ${item.lines}` : item.line ? ` line ${item.line}` : "";
      const score = item.score !== null && item.score !== undefined ? ` score ${item.score}` : "";
      return `
        <div class="reference">
          <div><span class="ref-id">[${escapeHtml(item.citation_id)}]</span>${escapeHtml(lines)}${escapeHtml(score)}</div>
          <div class="path">${escapeHtml(path)}</div>
          ${item.heading ? `<div class="path">heading: ${escapeHtml(item.heading)}</div>` : ""}
          ${item.message ? `<div class="path">${escapeHtml(item.message)}</div>` : ""}
          ${item.text ? `<pre>${escapeHtml(item.text)}</pre>` : ""}
        </div>
      `;
    }

    function statusText(payload) {
      if (payload.errors.length) return payload.errors[payload.errors.length - 1].message || "发生错误";
      if (payload.retrieval) return "正在生成回答...";
      if (payload.router) return "正在检索上下文...";
      return "正在判断检索方式...";
    }

    function renderMarkdown(markdown) {
      const source = String(markdown ?? "");
      const codeBlocks = [];
      let text = source.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (_, lang, code) => {
        const token = `@@CODE_BLOCK_${codeBlocks.length}@@`;
        codeBlocks.push(`<pre><code>${escapeHtml(code.trim())}</code></pre>`);
        return token;
      });

      const lines = text.split(/\r?\n/);
      const html = [];
      let listItems = [];

      function flushList() {
        if (!listItems.length) return;
        html.push(`<ul>${listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ul>`);
        listItems = [];
      }

      for (const rawLine of lines) {
        const line = rawLine.trimEnd();
        if (!line.trim()) {
          flushList();
          continue;
        }

        const heading = /^(#{1,4})\s+(.+)$/.exec(line);
        if (heading) {
          flushList();
          const level = Math.min(heading[1].length + 1, 5);
          html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
          continue;
        }

        const bullet = /^[-*]\s+(.+)$/.exec(line);
        if (bullet) {
          listItems.push(bullet[1]);
          continue;
        }

        const ordered = /^\d+\.\s+(.+)$/.exec(line);
        if (ordered) {
          listItems.push(ordered[1]);
          continue;
        }

        flushList();
        html.push(`<p>${renderInlineMarkdown(line)}</p>`);
      }
      flushList();

      let rendered = html.join("");
      codeBlocks.forEach((block, index) => {
        rendered = rendered.replace(`@@CODE_BLOCK_${index}@@`, block);
      });
      return rendered;
    }

    function renderInlineMarkdown(text) {
      return escapeHtml(text)
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\[(N|R|B|W|E)(\d+)\]/g, '<span class="citation">[$1$2]</span>');
    }

    function numberValue(id) {
      return Number($(id).value);
    }

    function setBusy(isBusy, text) {
      $("sendBtn").disabled = isBusy;
      $("status").textContent = text;
    }

    function scrollToBottom() {
      messages.scrollTop = messages.scrollHeight;
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }
  </script>
</body>
</html>
"""
