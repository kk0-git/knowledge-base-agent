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
