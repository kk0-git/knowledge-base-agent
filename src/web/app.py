from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from services.rag.reranker import DEFAULT_RERANKER_MODEL
from services.rag.search_service import SearchOptions, SearchService


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


def create_app(
    index_path: Path,
    bm25_index_path: Path | None,
    model_name: str,
    project_root: Path,
    embedding_provider: str = "local",
    embed_batch_size: int = 32,
    max_seq_length: int | None = None,
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
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "index": str(index_path),
            "bm25_index": str(service.bm25_index_path),
            "model": model_name,
            "embedding_provider": embedding_provider,
        }

    @app.post("/api/search")
    def search(request: SearchRequest) -> dict[str, Any]:
        try:
            response = service.search(SearchOptions(**request.model_dump()))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return asdict(response)

    return app


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
