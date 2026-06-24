# RAG 命令速查

## 0. 文档转换

- `--input`：源文件目录（PDF）
- `--output`：Markdown 输出目录
- `--collection`：数据集标签
- `--converter`：auto / pymupdf / pymupdf4llm / docling
- `--limit N`：只转前 N 篇，快速试转
- `--dry-run`：预览计划，不实际转换
- `--overwrite`：覆盖已有输出

```powershell
# 批量转换
uv run python scripts\convert_documents.py `
  --input "D:\31002\Documents\MyNote\textbooks" `
  --output "D:\31002\Documents\MyNote\imported_docs\textbooks" `
  --collection textbooks --converter auto

# 工具对比（同一份 PDF，三种转换器）
uv run python scripts\convert_documents.py `
  --input "D:\31002\Documents\MyNote\textbooks" `
  --output "D:\31002\Documents\MyNote\imported_docs\textbooks-pymupdf4llm" `
  --collection textbooks --limit 1 --converter pymupdf4llm --overwrite

uv run python scripts\mineru_agent_extract.py `
  --file "D:\31002\Documents\MyNote\textbooks\操作系统导论\02.pdf" `
  --out "D:\31002\Documents\MyNote\imported_docs\textbooks-mineru-agent\操作系统导论\02.md" `
  --collection textbooks --language ch --ocr --timeout 300 `
  --transport manual --upload-no-proxy

# MinerU 批量转换
$chapterDir = "D:\31002\Documents\MyNote\textbooks\操作系统导论"
$outDir = "D:\31002\Documents\MyNote\imported_docs\textbooks-mineru-agent\操作系统导论"
$chapters = @("02", "04", "05", "06", "08", "09", "10")
foreach ($chapter in $chapters) {
  uv run python scripts\mineru_agent_extract.py `
    --file "$chapterDir\$chapter.pdf" --out "$outDir\$chapter.md" `
    --collection textbooks --language ch --ocr --timeout 300 `
    --transport manual --upload-no-proxy
}
```

## 1. 建索引

- `--reset-index`：删除已有索引再重建
- `--incremental`：只对变更文件做 chunk+embed，不变文件复用
- `--vault`：Obsidian vault 根目录
- `--index`：向量索引 JSON 路径
- `--model`：嵌入模型，默认 BAAI/bge-m3
- `--embedding-provider`：local / openai_compatible
- `--max-chunk-chars / --target-chunk-chars / --min-chunk-chars`：chunker 参数

```powershell
# 本地首次全量
uv run python scripts\rag_debug.py index `
  --vault "D:\31002\Documents\MyNote" `
  --model BAAI/bge-m3 --index ./rag-index/bge-m3-v2.json --reset-index

# 本地增量
uv run python scripts\rag_debug.py index `
  --vault "D:\31002\Documents\MyNote" `
  --model BAAI/bge-m3 --index ./rag-index/bge-m3-v2.json --incremental

# API 首次全量（硅基流动）
uv run python scripts\rag_debug.py index `
  --vault "D:\31002\Documents\MyNote" `
  --embedding-provider openai_compatible --model BAAI/bge-m3 `
  --index ./rag-index/mixed-siliconflow-bge-m3.json `
  --embed-batch-size 16 --reset-index

# API 增量
uv run python scripts\rag_debug.py index `
  --vault "D:\31002\Documents\MyNote" `
  --embedding-provider openai_compatible --model BAAI/bge-m3 `
  --index ./rag-index/mixed-siliconflow-bge-m3.json `
  --embed-batch-size 16 --incremental
```

## 2. 单条搜索

- `--query`：查询文本
- `--mode`：dense / bm25 / hybrid / hybrid-rerank
- `--reranker-type`：local（bge） / dashscope（Qwen3）
- `--dense-top-k / --bm25-top-k`：两路候选数，默认 50

```powershell
# hybrid
uv run python scripts\rag_debug.py search `
  --index ./rag-index/bge-m3-v2.json --model BAAI/bge-m3 `
  --query "进程是什么" --mode hybrid --top-k 5

# hybrid + reranker
uv run python scripts\rag_debug.py search `
  --index ./rag-index/bge-m3-v2.json --model BAAI/bge-m3 `
  --query "进程是什么" --mode hybrid-rerank `
  --reranker-type local --top-k 5
```

## 3. 评测

策略含义：
- `dense`：纯向量检索（BGE-M3 + cosine）
- `bm25`：纯关键词检索（自定义分词 + Okapi BM25）
- `hybrid`：Dense + BM25 经 RRF（k=60）融合，无 reranker
- `local-rerank`：hybrid + 本地 bge-reranker-v2-m3（568M）精排
- `dashscope-rerank`：hybrid + DashScope Qwen3-Reranker（4B）精排

```powershell
# 五种策略对比
uv run python scripts\rag_compare.py `
  --index ./rag-index/bge-m3-v2.json --model BAAI/bge-m3 `
  --top-k 20 --hit-ks 1,3,5,10,20 `
  --strategies dense,bm25,hybrid,local-rerank,dashscope-rerank `
  --out ./eval-results/compare

# 单策略
uv run python scripts\rag_eval.py `
  --index ./rag-index/bge-m3-v2.json --model BAAI/bge-m3 `
  --mode hybrid --top-k 20 --hit-ks 1,3,5,10,20 `
  --out ./eval-results/hybrid.json

# 依赖感知版本
uv run python scripts\rag_compare_dependency.py `
  --index ./rag-index/bge-m3-v2.json --model BAAI/bge-m3 `
  --top-k 20 --hit-ks 1,3,5,10,20 `
  --strategies dense,bm25,hybrid,local-rerank,dashscope-rerank `
  --out ./eval-results/compare-dependency
```

## 4. 实验脚本

```powershell
# RRF k 值网格搜索
uv run python scripts\rrf_k_sweep.py `
  --index ./rag-index/mixed-siliconflow-bge-m3.json `
  --model BAAI/bge-m3 `
  --k-values 0,10,20,30,40,50,60,80,100,120 `
  --out ./eval-results/rrf-k-sweep.json

# Rerank 位移分析
uv run python scripts\analyze_rerank_movement.py `
  --before ./eval-results/hybrid.json `
  --after ./eval-results/hybrid-rerank.json `
  --eval ./eval/rag_eval.json `
  --out ./eval-results/movement

# Query 改写调试
uv run python scripts\query_rewrite_debug.py `
  --query "PCB是啥" --out ./eval-results/query-rewrite.json
```

## 5. Web 搜索

```powershell
uv run python scripts\web_search.py `
  --index ./rag-index/bge-m3-v2.json --model BAAI/bge-m3 `
  --host 127.0.0.1 --port 8001
```
打开 http://127.0.0.1:8001

## 6. 索引文件

```text
rag-index/bge-m3-v2.json          # 本地向量索引
rag-index/bge-m3-v2.bm25.json     # 本地 BM25
mixed-siliconflow-bge-m3.json     # API 向量索引
mixed-siliconflow-bge-m3.bm25.json # API BM25
```

## 7. Wiki / Obsidian 工作流

日常只记两个动作：

```text
更新主题 Wiki：同步工作区 -> 更新指定 topic wiki -> 刷新 _Wiki Report.md
同步 RAG 索引：同步 embedding/BM25 -> 同步 tag state -> 刷新 _Wiki Report.md
```

### 7.1 更新主题 Wiki（Obsidian 常用）

适合配置到 Obsidian Shell Commands。该命令会先同步工作区，再更新指定主题 Wiki；如果只想跳过 embedding，可额外追加 `--no-rag`。
Shell Commands 中建议使用 Prompt 变量 `{{tag}}`，不要依赖选中文本变量。

```powershell
cd D:\Workspaces\Personal\agent\obsidian_vault\knowledge_agent
uv run python scripts\workspace_sync.py update-topic `
  --vault "D:\31002\Documents\MyNote" `
  --state ./wiki-state/wiki_state.full-test.json `
  --wiki-dir "D:\31002\Documents\MyNote\wiki_full" `
  --min-notes-per-tag 2 `
  --overview-note-threshold 12 `
  --tag "{{tag}}" `
  --force
```

成功输出：

```text
OK: synced 1 changed notes, refreshed topic wiki.
Topic wiki updated.
Report: D:\31002\Documents\MyNote\wiki_full\_Wiki Report.md
```

### 7.2 同步 RAG 索引（搜索/对话前使用）

用于更新 embedding、BM25、tag state 和 `_Wiki Report.md`，不自动合成 wiki 正文。
这条命令影响搜索、聊天和 eval；不需要在每次更新 wiki 时运行。

```powershell
uv run python scripts\workspace_sync.py watch `
  --vault "D:\31002\Documents\MyNote" `
  --state ./wiki-state/wiki_state.full-test.json `
  --wiki-dir "D:\31002\Documents\MyNote\wiki_full" `
  --min-notes-per-tag 2 `
  --overview-note-threshold 12 `
  --once
```

### 7.3 只刷新 Obsidian 状态面板

```powershell
uv run python scripts\wiki_debug.py write-report `
  --vault "D:\31002\Documents\MyNote" `
  --state ./wiki-state/wiki_state.full-test.json `
  --wiki-dir "D:\31002\Documents\MyNote\wiki_full" `
  --min-notes-per-tag 2 `
  --overview-note-threshold 12
```

### 7.4 后台轮询同步

不自动合成 wiki 正文，只自动维护 RAG index、tag state 和 `_Wiki Report.md`：

```powershell
uv run python scripts\workspace_sync.py watch `
  --vault "D:\31002\Documents\MyNote" `
  --state ./wiki-state/wiki_state.full-test.json `
  --wiki-dir "D:\31002\Documents\MyNote\wiki_full" `
  --min-notes-per-tag 2 `
  --overview-note-threshold 12 `
  --quiet-seconds 10800 `
  --poll-seconds 60
```

Obsidian Shell Commands 插件配置见：`docs/OBSIDIAN_WORKFLOW.md`。

## 8. Workflow Debug

用于验证统一工作流抽象的 `scope -> context` 是否正确。

前端入口：

```text
http://127.0.0.1:8001/organize
```

```powershell
uv run python scripts\workflow_debug.py inspect-scope `
  --vault "D:\31002\Documents\MyNote" `
  --state ./wiki-state/wiki_state.full-test.json `
  --wiki-dir "D:\31002\Documents\MyNote\wiki_full" `
  --scope-type tag `
  --value "java/servlet" `
  --context-mode wiki_context
```

预期会输出该 tag 命中的源笔记数、context items 数量和前几个 sample item。

### 8.1 确定性审计

不触发 LLM，用规则检查指定范围内的笔记和 wiki state。

```powershell
uv run python scripts\workflow_debug.py audit `
  --vault "D:\31002\Documents\MyNote" `
  --state ./wiki-state/wiki_state.full-test.json `
  --wiki-dir "D:\31002\Documents\MyNote\wiki_full" `
  --scope-type tag `
  --value "java/servlet" `
  --max-issues 20
```

当前会检查绝对图片路径、重复标题、空标题、标题无正文，以及 wiki 合成失败/缺失等状态问题。

### 8.3 统一整理 workflow（推荐入口）

`organize` 会把确定性结构审计和 LLM 整理建议合并到一个报告中：

- 结构检查：坏链接、空 section、短 note、图片路径、wiki 状态等确定性问题。
- 整理建议：主题覆盖、核心/边缘笔记、补链、补 tag、wiki 候选、复习问题。

```powershell
uv run python scripts\workflow_debug.py organize `
  --vault "D:\31002\Documents\MyNote" `
  --state ./wiki-state/wiki_state.full-test.json `
  --wiki-dir "D:\31002\Documents\MyNote\wiki_full" `
  --scope-type tag `
  --value "学习/全栈" `
  --max-issues 20 `
  --max-notes 8 `
  --max-chars-per-note 1600 `
  --review-mode topic
```

输出：

```text
eval-results/organize-tag-学习-全栈.json
eval-results/organize-tag-学习-全栈.md
```

旧命令仍可用于分开调试：

- `workflow_debug.py audit`：只跑确定性审计。
- `workflow_debug.py review-notes`：只跑 LLM 整理建议。

### 8.4 复习与面试入口

复习已经从 Chat 模式中拆出，使用独立 `/review` 页面。Chat 只保留问答和模拟面试。

- `/review`：主题复习入口，按主题和题数进入逐题回忆、作答、纠偏、pass/fail。
- `/?mode=interview`：模拟面试入口。LLM 扮演考官，笔记是私有参考答案，重点是抓住用户回答里的模糊点继续追问。
- `/wiki`：只读 Wiki 阅读器。
- `/admin/wiki`：Wiki 同步、合成、策略维护后台。
- `/organize`：笔记整理与审计入口。

本地测试入口：

```text
http://127.0.0.1:8003/review
http://127.0.0.1:8003/?mode=interview
```

示例选择：

```text
页面：/?mode=interview
范围：文件夹
范围值：个人/面试/agent面试
输入：开始面试复习，先问我一个问题
```

API smoke test：

```powershell
@'
import json
import urllib.request

payload = {
    "query": "开始面试复习，先问我一个问题",
    "chat_mode": "interview",
    "scope_type": "folder",
    "scope_value": "个人/面试/agent面试",
    "notes_top_k": 5,
}
req = urllib.request.Request(
    "http://127.0.0.1:8003/api/agent/stream",
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    print(resp.read(2000).decode("utf-8", errors="replace"))
'@ | uv run python -
```

复习页面不走 `chat_mode=study`；旧 `mode=study` 页面入口会重定向到 `/review`。

### 8.6 Agent 解耦 runs API

```powershell
# 创建后台 run（面试需先有 session_id）
python -m pytest tests/test_session_repository.py tests/test_agent_turn_service.py tests/test_agent_run_stream.py -q

# curl 示例（问答）
curl -s -X POST http://127.0.0.1:8003/api/agent/runs -H "Content-Type: application/json" -d "{\"query\":\"MCP 是什么\",\"chat_mode\":\"answer\",\"scope_type\":\"folder\",\"scope_value\":\"个人/面试/agent面试\"}"

# 订阅 task 事件流（将 TASK_ID 替换为上一步返回值）
curl -N http://127.0.0.1:8003/api/tasks/TASK_ID/stream
```

### 8.7 Phase B Review WorkspaceStore 手测

```powershell
python -m pytest tests/test_review_practice.py tests/test_web_workspace.py -q --tb=short
```

浏览器手测（同 tab）：

1. `/review` 逐条对照做到第 2 题 → 侧边栏去面试 → 回 `/review` → tab、题号、已答 results 恢复
2. 对话复查 2 轮 → 去问答 → 回 `/review` → 对话 history 恢复
3. 复习进行中 F5 → 状态恢复
4. 关闭 tab 再开 `/review` → 工作区清空
5. 卡片 pending 生成中切页回来 → poll 重启、卡片与 API 对齐
6. 服务端重启后旧 `reviewRunId` → 本地进度可见 + 「run 已失效」提示
7. 逐条对照完成 2 题 → 回「复习总览」→ 总览区与 tab 显示「已完成 2 / 共 N」

sessionStorage key: `knowledge_agent.workspace.review.v1`（Phase C 起 review slice 独立 key；旧 `knowledge_agent.workspace.v1` 会在 hydrate 时一次性迁移；无 `version` 的 snapshot 会在 hydrate 时升到 v1）

### 8.8 Phase C Server Sessions + Chat WorkspaceStore 手测

```powershell
python -m pytest tests/test_review_run_repository.py tests/test_answer_session_repository.py tests/test_agent_turn_service.py tests/test_review_practice.py tests/test_web_workspace.py -q --tb=short
```

浏览器手测：

1. 卡片复习中重启 uvicorn → 回 `/review` → `GET plan` 成功，无 runStale
2. 对话复查 2 轮 → 重启服务 → 回 `/review` → history 从 run workspace 恢复
3. 问答 2 轮 → 清 site data 仅保留 localStorage chat slice → `GET answer session` 恢复消息+citations
4. 问答历史按钮 → 列表来自 `/api/answer/sessions`
5. 面试 + 问答指针 → 仅存于 `localStorage` `knowledge_agent.workspace.chat.v1`，无散落旧 key

Storage keys:

- Review workspace: `sessionStorage` `knowledge_agent.workspace.review.v1`
- Chat workspace: `localStorage` `knowledge_agent.workspace.chat.v1`
- Review runs: `review-runs/*.json`
- Answer sessions: `answer-sessions/YYYY-MM/*.json`

### 8.5 复习页三态重构检查

```powershell
python -m py_compile "src/web/app.py" "src/agent/tools/review.py"
python -m pytest tests/test_review_practice.py -q --tb=short
```




