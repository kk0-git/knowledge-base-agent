"""
Download documentation and papers to expand the Obsidian vault.

Sources:
  - FastAPI docs (zh-CN preferred, en fallback)
  - Redis docs
  - Docker docs (get-started + reference subsets)
  - arXiv papers (RAG, DPR, RAPTOR, ColBERT, HNSW, BM25)

Output: D:\Workspaces\Personal\agent\obsidian_vault\mynote\
  mynote/
  ├── docs/
  │   ├── fastapi/
  │   ├── redis/
  │   └── docker/
  └── papers/
      ├── rag/
      ├── retrieval/
      └── indexing/

Usage:
  python download_vault_data.py                          # default
  python download_vault_data.py --proxy http://127.0.0.1:7890   # with proxy
  python download_vault_data.py --no-git                        # skip git, use raw API
  python download_vault_data.py --skip-docs --skip-papers       # selective
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests

# ── Config ──────────────────────────────────────────────────────────
VAULT_MYNOTE = Path(r"D:\Workspaces\Personal\agent\obsidian_vault\mynote")
DOCS_DIR = VAULT_MYNOTE / "docs"
PAPERS_DIR = VAULT_MYNOTE / "papers"

GIT_TIMEOUT = 600  # per git operation
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds base delay

# ── Documentation sources ───────────────────────────────────────────
DOC_SOURCES = {
    "fastapi": {
        "repo": "https://github.com/fastapi/fastapi.git",
        "owner": "fastapi",
        "repo_name": "fastapi",
        "branch": "master",
        "sparse_paths": ["docs/zh/docs", "docs/en/docs"],
        "target_dir": DOCS_DIR / "fastapi",
        "path_remap": {
            "docs/zh/docs/": "",
            "docs/en/docs/": "en/",
        },
    },
    "redis": {
        "repo": "https://github.com/redis/redis-doc.git",
        "owner": "redis",
        "repo_name": "redis-doc",
        "branch": "master",
        "sparse_paths": ["docs", "commands", "topics"],
        "target_dir": DOCS_DIR / "redis",
        "path_remap": {
            "docs/": "",
            "commands/": "commands/",
            "topics/": "topics/",
        },
    },
    "docker": {
        "repo": "https://github.com/docker/docs.git",
        "owner": "docker",
        "repo_name": "docs",
        "branch": "main",
        "sparse_paths": [
            "content/get-started",
            "content/manuals/docker-desktop",
            "content/manuals/compose",
            "content/reference/cli",
        ],
        "target_dir": DOCS_DIR / "docker",
        "path_remap": {
            "content/": "",
        },
    },
}

# ── Paper sources ───────────────────────────────────────────────────
PAPER_SOURCES = [
    {
        "id": "2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "category": "rag",
        "filename": "RAG_Lewis_2020",
    },
    {
        "id": "2004.04906",
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "category": "rag",
        "filename": "DPR_Karpukhin_2020",
    },
    {
        "id": "2401.18059",
        "title": "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval",
        "category": "indexing",
        "filename": "RAPTOR_Sarthi_2024",
    },
    {
        "id": "2004.12832",
        "title": "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT",
        "category": "retrieval",
        "filename": "ColBERT_Khattab_2020",
    },
    {
        "id": "1603.09320",
        "title": "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs",
        "category": "indexing",
        "filename": "HNSW_Malkov_2016",
    },
    {
        "id": "2301.09232",
        "title": "Faiss: A Library for Efficient Similarity Search and Clustering of Dense Vectors",
        "category": "indexing",
        "filename": "FAISS_Johnson_2023",
    },
    {
        "id": "bm25",
        "title": "The Probabilistic Relevance Framework: BM25 and Beyond",
        "category": "retrieval",
        "filename": "BM25_Robertson_2009",
        "url": "https://www.nowpublishers.com/article/DownloadSummary/INR-019",
        "note": "Now Publishers 出版，部分需付费。将生成知识点总结。",
    },
]


# ── Helpers ─────────────────────────────────────────────────────────

def _get_session(proxy: str | None) -> requests.Session:
    """Create a requests session with optional proxy and retry."""
    session = requests.Session()
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    session.headers.update({"User-Agent": "knowledge-agent/1.0"})
    return session


def _git_env() -> dict:
    """Build env dict for git subprocess calls."""
    env = os.environ.copy()
    # Disable interactive prompts
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_HTTP_LOW_SPEED_LIMIT"] = "0"
    env["GIT_HTTP_LOW_SPEED_TIME"] = "0"
    return env


def run_git(cmd: list[str], cwd: Path | None = None,
            timeout: int = GIT_TIMEOUT) -> str:
    """Run a git command. Retries with backoff on network errors."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
                env=_git_env(),
            )
            if result.returncode == 0:
                return result.stdout

            stderr = result.stderr.strip()
            # Network-related failures → retry
            if any(kw in stderr.lower() for kw in
                   ["early eof", "rpc failed", "ssl", "unexpected disconnect",
                    "fetch-pack", "connection reset", "timeout", "could not fetch"]):
                raise _RetryableError(f"Git network error (attempt {attempt}): {stderr}")

            # Non-network error → don't retry
            raise RuntimeError(f"Git command failed: {' '.join(cmd)}\n{stderr}")

        except _RetryableError as e:
            last_err = e
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY * attempt
                print(f"    ⚠ {e}")
                print(f"    ↻ Retrying in {delay}s (attempt {attempt}/{MAX_RETRIES})...")
                time.sleep(delay)
        except subprocess.TimeoutExpired:
            last_err = Exception(f"Git command timed out after {timeout}s (attempt {attempt})")
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY * attempt
                print(f"    ⚠ {last_err}")
                print(f"    ↻ Retrying in {delay}s...")
                time.sleep(delay)

    raise RuntimeError(f"Git command failed after {MAX_RETRIES} attempts: {last_err}")


class _RetryableError(Exception):
    """Marker for errors that should trigger a retry."""
    pass


def configure_git_for_large_transfers(proxy: str | None = None) -> None:
    """Set global git config for more reliable large transfers."""
    configs = [
        ("http.postBuffer", "524288000"),       # 500MB
        ("http.lowSpeedLimit", "0"),
        ("http.lowSpeedTime", "0"),
        ("core.compression", "0"),              # disable compression (faster on good CPU)
        ("fetch.fsckObjects", "false"),
        ("protocol.version", "2"),              # protocol v2 is more efficient
    ]
    for key, val in configs:
        try:
            subprocess.run(
                ["git", "config", "--global", key, val],
                capture_output=True, timeout=10, env=_git_env(),
            )
        except Exception:
            pass  # config might already exist, ignore

    if proxy:
        try:
            subprocess.run(
                ["git", "config", "--global", "http.proxy", proxy],
                capture_output=True, timeout=10, env=_git_env(),
            )
            subprocess.run(
                ["git", "config", "--global", "https.proxy", proxy],
                capture_output=True, timeout=10, env=_git_env(),
            )
            print(f"  🔧 Git proxy set: {proxy}")
        except Exception:
            pass


def clone_sparse(repo_url: str, sparse_paths: list[str],
                 work_dir: Path, use_git: bool = True) -> Path:
    """Clone a repo with sparse checkout. Falls back to raw downloads if use_git=False."""
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    clone_dir = work_dir / repo_name

    if clone_dir.exists():
        shutil.rmtree(clone_dir, ignore_errors=True)

    if use_git:
        return _clone_sparse_git(repo_url, sparse_paths, clone_dir)
    else:
        print("  ⚠ --no-git mode: will create empty directory")
        clone_dir.mkdir(parents=True, exist_ok=True)
        return clone_dir


def _clone_sparse_git(repo_url: str, sparse_paths: list[str],
                      clone_dir: Path) -> Path:
    """Git-based sparse clone with retries and optimized settings."""
    print(f"  Cloning {repo_url} (sparse, depth=1)...")

    # Build clone command with optimizations
    clone_cmd = [
        "git", "clone",
        "--depth", "1",
        "--filter=blob:none",
        "--sparse",
        "--single-branch",
        "--no-tags",
        "-c", "http.postBuffer=524288000",
        "-c", "core.compression=0",
        "-c", "protocol.version=2",
        repo_url, str(clone_dir),
    ]
    run_git(clone_cmd)

    # Sparse checkout
    print(f"  Setting sparse-checkout: {sparse_paths}")
    run_git(
        ["git", "-C", str(clone_dir), "sparse-checkout", "set"] + sparse_paths,
    )

    return clone_dir


def copy_markdown_files(src_dir: Path, target_dir: Path,
                        path_remap: dict[str, str]) -> int:
    """Copy .md files from src_dir to target_dir, applying path remapping."""
    count = 0
    target_dir.mkdir(parents=True, exist_ok=True)

    for md_file in sorted(src_dir.rglob("*.md")):
        rel = md_file.relative_to(src_dir).as_posix()

        # Apply remap
        output_rel = rel
        for prefix, replacement in path_remap.items():
            if rel.startswith(prefix):
                output_rel = replacement + rel[len(prefix):]
                break

        dest = target_dir / output_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md_file, dest)
        count += 1

    return count


def download_arxiv_pdf(paper_id: str, output_dir: Path,
                       session: requests.Session) -> Path:
    """Download a PDF from arXiv by paper ID."""
    url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    pdf_path = output_dir / f"{paper_id}.pdf"

    print(f"  Downloading {url} ...")
    resp = session.get(url, timeout=120)
    resp.raise_for_status()
    pdf_path.write_bytes(resp.content)
    print(f"    → {pdf_path} ({len(resp.content)} bytes)")
    return pdf_path


def pdf_to_markdown(pdf_path: Path, title: str, authors: str = "",
                    paper_id: str = "", abstract: str = "") -> str:
    """Extract text from PDF and wrap in a markdown file."""
    try:
        import pymupdf  # type: ignore
    except ImportError:
        print("    ⚠ pymupdf not available, creating metadata-only .md")
        return _pdf_metadata_md(title, authors, paper_id, abstract, pdf_path)

    try:
        doc = pymupdf.open(str(pdf_path))
        pages = []
        for page in doc:
            text = page.get_text("text")  # type: ignore
            if text.strip():
                pages.append(text.strip())
        doc.close()

        body = "\n\n".join(pages)

        return f"""---
title: "{title}"
paper_id: "{paper_id}"
source: "arXiv:{paper_id}"
type: paper
---

# {title}

{body}
"""
    except Exception as e:
        print(f"    ⚠ PDF extraction failed: {e}, fallback to metadata-only")
        return _pdf_metadata_md(title, authors, paper_id, abstract, pdf_path)


def _pdf_metadata_md(title: str, authors: str, paper_id: str,
                     abstract: str, pdf_path: Path) -> str:
    return f"""---
title: "{title}"
paper_id: "{paper_id}"
source: "arXiv:{paper_id}"
type: paper
status: pdf_not_extracted
pdf_path: "{pdf_path.as_posix()}"
---

# {title}

> **注意**: 此论文的全文文本未能从 PDF 中自动提取。请手动查看 PDF 文件。

## 摘要

{abstract}

## PDF 文件

PDF 位于: `{pdf_path.as_posix()}`
"""


# ── Main routines ───────────────────────────────────────────────────

def download_docs(proxy: str | None = None, use_git: bool = True) -> dict[str, int]:
    """Download all documentation sources. Returns {name: file_count}."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    if use_git:
        configure_git_for_large_transfers(proxy)

    results = {}

    with tempfile.TemporaryDirectory(prefix="kb_docs_") as tmp:
        tmp_path = Path(tmp)

        for name, cfg in DOC_SOURCES.items():
            print(f"\n{'='*60}")
            print(f"📚 Downloading {name} docs...")
            print(f"{'='*60}")

            try:
                clone_dir = clone_sparse(
                    cfg["repo"], cfg["sparse_paths"], tmp_path,
                    use_git=use_git,
                )
                count = copy_markdown_files(
                    clone_dir, cfg["target_dir"], cfg["path_remap"],
                )
                results[name] = count
                print(f"  ✅ {name}: {count} markdown files copied")
            except Exception as e:
                print(f"  ❌ {name}: FAILED — {e}")
                results[name] = 0

    return results


def download_papers(proxy: str | None = None) -> dict[str, int]:
    """Download all paper PDFs and create .md files."""
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    session = _get_session(proxy)
    results: dict[str, int] = {}

    for paper in PAPER_SOURCES:
        category = paper["category"]
        cat_dir = PAPERS_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📄 [{category}] {paper['title'][:80]}...")

        # Non-arXiv papers (e.g., BM25)
        if "url" in paper and "arxiv.org" not in paper.get("url", ""):
            md_content = _bm25_markdown(paper)
            md_path = cat_dir / f"{paper['filename']}.md"
            md_path.write_text(md_content, encoding="utf-8")
            results[category] = results.get(category, 0) + 1
            print(f"  ✅ Created knowledge-summary .md")
            continue

        # arXiv papers
        try:
            pdf_path = download_arxiv_pdf(paper["id"], cat_dir, session)

            # Try to get abstract from arXiv API
            abstract = ""
            try:
                api_url = f"https://export.arxiv.org/api/query?id_list={paper['id']}&max_results=1"
                resp = session.get(api_url, timeout=30)
                abs_match = re.search(
                    r"<summary>(.*?)</summary>", resp.text, re.DOTALL
                )
                if abs_match:
                    abstract = abs_match.group(1).strip().replace("\n", " ")
            except Exception:
                pass

            md_content = pdf_to_markdown(
                pdf_path, title=paper["title"],
                paper_id=paper["id"], abstract=abstract,
            )

            md_path = cat_dir / f"{paper['filename']}.md"
            md_path.write_text(md_content, encoding="utf-8")
            results[category] = results.get(category, 0) + 1
            print(f"  ✅ .md written: {md_path.name}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            md_path = cat_dir / f"{paper['filename']}.md"
            md_path.write_text(
                _pdf_metadata_md(paper["title"], "", paper["id"], "",
                                 cat_dir / f"{paper['id']}.pdf"),
                encoding="utf-8",
            )
            results[category] = results.get(category, 0) + 1

    return results


def _bm25_markdown(paper: dict) -> str:
    """Generate a knowledge-summary .md for the BM25 paper."""
    return f"""---
title: "{paper['title']}"
paper_id: "{paper['id']}"
source: "{paper.get('url', 'unknown')}"
type: paper
status: knowledge_summary
---

# {paper['title']}

> **注意**: BM25 原始论文由 Now Publishers 出版（Foundations and Trends in
> Information Retrieval, 2009）。部分内容需通过机构订阅访问。
>
> **替代资源**:
> - [DBLP 页面](https://dblp.org/rec/journals/ftir/RobertsonZ09)
> - [Now Publishers](https://www.nowpublishers.com/article/Details/INR-019)

## 核心知识点

BM25 (Best Match 25) 是概率相关性框架下的排名函数，是 TF-IDF 的后继者。

### 关键公式

BM25(q, d) = Σ IDF(qi) · (f(qi,d) · (k1+1)) / (f(qi,d) + k1 · (1-b + b · |d|/avgdl))

### 关键参数

- **k1** (~1.2-2.0): 控制词频饱和度。词在文档中出现次数越多，额外贡献递减
- **b** (0-1, 通常 0.75): 文档长度归一化强度。b=0 无归一化，b=1 完全归一化
- **IDF 变体**: 基于 Robertson-Spärck Jones 的概率 IDF，比传统 IDF 更稳健

### 核心思想

1. **词频饱和度 (Term Frequency Saturation)**: 不同于 TF-IDF 的线性 TF，
   BM25 使用非线性饱和函数 — 一个词出现 100 次不等于 10 次的 10 倍价值
2. **文档长度归一化**: 长文档天然包含更多词频，BM25 用 b 参数将文档长度
   与平均长度比较，惩罚过长文档
3. **概率推导**: 从 2-Poisson 模型出发，推导出相关性排序的概率框架

### 在 RAG 中的应用

- BM25 是**稀疏检索**的基线方法，常与 dense retrieval (DPR 等) 做对比
- 混合检索 (Hybrid Search): BM25 + Dense Retrieval 结合，利用互补优势
- BM25 对**精确关键词匹配**效果好，dense 对**语义匹配**好

### 扩展阅读

- BM25F: 结构化文档（多字段）的 BM25 扩展
- BM25+: 解决 BM25 对超长文档的过度惩罚问题
"""


def print_summary(doc_results: dict, paper_results: dict) -> None:
    """Print a summary of all downloads."""
    print(f"\n{'='*60}")
    print(f"📊 DOWNLOAD SUMMARY")
    print(f"{'='*60}")

    print(f"\n📚 Documentation:")
    total_docs = 0
    for name, count in doc_results.items():
        print(f"  {name}: {count} files")
        total_docs += count
    print(f"  TOTAL: {total_docs} documentation files")

    print(f"\n📄 Papers:")
    total_papers = 0
    for cat, count in paper_results.items():
        print(f"  {cat}: {count} papers")
        total_papers += count
    print(f"  TOTAL: {total_papers} papers")

    print(f"\n📁 Output directory: {VAULT_MYNOTE}")
    _count_tree(VAULT_MYNOTE)


def _count_tree(path: Path, indent: int = 0) -> None:
    """Print directory tree with file counts."""
    if not path.exists():
        return
    for child in sorted(path.iterdir()):
        if child.is_dir():
            md_count = len(list(child.rglob("*.md")))
            pdf_count = len(list(child.rglob("*.pdf")))
            extras = []
            if md_count:
                extras.append(f"{md_count} .md")
            if pdf_count:
                extras.append(f"{pdf_count} .pdf")
            extra_str = f" ({', '.join(extras)})" if extras else ""
            print(f"  {'  '*indent}{child.name}/{extra_str}")
            _count_tree(child, indent + 1)


# ── Entry point ─────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download docs + papers for Obsidian vault expansion",
    )
    p.add_argument("--proxy", default=None,
                   help="HTTPS proxy, e.g. http://127.0.0.1:7890")
    p.add_argument("--no-git", action="store_true",
                   help="Skip git clone (useful if git protocol is blocked)")
    p.add_argument("--skip-docs", action="store_true",
                   help="Skip documentation downloads")
    p.add_argument("--skip-papers", action="store_true",
                   help="Skip paper downloads")
    return p


def main() -> int:
    args = build_parser().parse_args()

    # Show proxy setting
    proxy = args.proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

    print("=" * 60)
    print("📥 vault data downloader")
    print(f"   target: {VAULT_MYNOTE}")
    print(f"   proxy:  {proxy or '(none — set --proxy or HTTPS_PROXY env)'}")
    print(f"   git:    {'disabled' if args.no_git else 'enabled'}")
    print("=" * 60)

    doc_results = {}
    paper_results = {}

    if not args.skip_docs:
        print(f"\n🔹 PHASE 1: Download documentation")
        doc_results = download_docs(proxy=proxy, use_git=not args.no_git)

    if not args.skip_papers:
        print(f"\n🔹 PHASE 2: Download papers")
        paper_results = download_papers(proxy=proxy)

    print_summary(doc_results, paper_results)

    # Warn if docs all failed
    if doc_results and all(v == 0 for v in doc_results.values()):
        print(f"\n{'!'*60}")
        print("⚠️  ALL documentation downloads failed.")
        print("   This is likely a network issue reaching GitHub.")
        print("   Try one of these:")
        print(f"   1. Use a proxy:  python {__file__} --proxy http://127.0.0.1:7890")
        print(f"   2. Set env var:   $env:HTTPS_PROXY='http://127.0.0.1:7890'")
        print(f"                     python {__file__}")
        print(f"   3. Check your VPN / network connection to github.com")
        print(f"{'!'*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
