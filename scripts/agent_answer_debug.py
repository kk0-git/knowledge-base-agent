from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import rag_eval
from knowledge_base_agent.config import load_llm_config
from knowledge_base_agent.llm import create_llm_client
from services.rag.agent_answer import (
    AgentAnswerConfig,
    AgentAnswerPipeline,
    agent_run_result_to_dict,
)
from services.rag.intent_router import ConversationCommand, LLMIntentRouter
from services.rag.online_search import OnlineSearchClient
from services.rag.vector_store_loader import (
    DEFAULT_HNSW_EF_CONSTRUCTION,
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_HNSW_M,
)


DEFAULT_MODEL = "BAAI/bge-m3"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Khoj-style router + tools + context + LLM answer debug flow")
    parser.add_argument("--index", default="./rag-index/index.json", help="Vector index JSON path")
    parser.add_argument("--bm25-index", default=None, help="BM25 index JSON path")
    parser.add_argument("--vault", required=True, help="Vault root for rg search")
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model")
    parser.add_argument("--embedding-provider", choices=["local", "openai_compatible"], default="local")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--vector-index", choices=["flat", "hnsw"], default="flat")
    parser.add_argument("--hnsw-m", type=int, default=DEFAULT_HNSW_M)
    parser.add_argument("--hnsw-ef-construction", type=int, default=DEFAULT_HNSW_EF_CONSTRUCTION)
    parser.add_argument("--hnsw-ef-search", type=int, default=DEFAULT_HNSW_EF_SEARCH)
    parser.add_argument(
        "--command",
        choices=["auto", "Notes", "RegexSearchFiles", "Notes+Online"],
        default="auto",
        help="Override router for controlled debugging",
    )
    parser.add_argument("--notes-top-k", type=int, default=5)
    parser.add_argument("--regex-top-k", type=int, default=8)
    parser.add_argument("--bm25-top-k", type=int, default=8)
    parser.add_argument("--dense-top-k", type=int, default=50)
    parser.add_argument("--hybrid-bm25-top-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--max-chars-per-item", type=int, default=1000)
    parser.add_argument("--max-context-chars", type=int, default=8000)
    parser.add_argument("--online-provider", default=None)
    parser.add_argument("--online-top-k", type=int, default=5)
    parser.add_argument(
        "--speculative-notes-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start local hybrid search in parallel with LLM routing",
    )
    parser.add_argument("--out", default="./eval-results/agent-answer-debug")
    args = parser.parse_args()

    llm_config = load_llm_config(PROJECT_ROOT)
    llm_client = create_llm_client(llm_config)
    router = build_router(args.command, llm_client, llm_config)

    manager = rag_eval.build_manager(
        index_path=Path(args.index),
        bm25_index_path=Path(args.bm25_index) if args.bm25_index else None,
        model_name=args.model,
        mode="hybrid",
        embedding_provider=args.embedding_provider,
        embed_batch_size=args.embed_batch_size,
        max_seq_length=args.max_seq_length,
        vector_index=args.vector_index,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
    )

    pipeline = AgentAnswerPipeline(
        router=router,
        llm_client=llm_client,
        llm_model=llm_config.model,
        manager=manager,
        vault_root=Path(args.vault),
        online_client=OnlineSearchClient(provider=args.online_provider),
        config=AgentAnswerConfig(
            notes_top_k=args.notes_top_k,
            regex_top_k=args.regex_top_k,
            bm25_top_k=args.bm25_top_k,
            dense_top_k=args.dense_top_k,
            hybrid_bm25_top_k=args.hybrid_bm25_top_k,
            rrf_k=args.rrf_k,
            max_chars_per_item=args.max_chars_per_item,
            max_context_chars=args.max_context_chars,
            online_top_k=args.online_top_k,
            speculative_notes_search=args.speculative_notes_search,
        ),
        answer_temperature=llm_config.temperature,
    )

    result = pipeline.run(args.query)
    payload = build_payload(args, agent_run_result_to_dict(result))
    write_outputs(Path(args.out), payload)

    print(payload["answer"]["answer"])
    print()
    print(f"Command: {payload['command']}")
    print(f"Saved: {Path(args.out).with_suffix('.json').resolve()}")
    return 0


def build_router(command: str, llm_client, llm_config):
    forced_command = None if command == "auto" else ConversationCommand(command)
    return LLMIntentRouter(
        client=llm_client,
        model=llm_config.model,
        temperature=0.0,
        forced_command=forced_command,
    )


def build_payload(args: argparse.Namespace, run_payload: dict) -> dict:
    return {
        "config": {
            "index": args.index,
            "bm25_index": args.bm25_index,
            "vault": args.vault,
            "query": args.query,
            "command_override": args.command,
            "embedding_model": args.model,
            "embedding_provider": args.embedding_provider,
            "vector_index": args.vector_index,
            "notes_top_k": args.notes_top_k,
            "regex_top_k": args.regex_top_k,
            "bm25_top_k": args.bm25_top_k,
            "dense_top_k": args.dense_top_k,
            "hybrid_bm25_top_k": args.hybrid_bm25_top_k,
            "rrf_k": args.rrf_k,
            "max_chars_per_item": args.max_chars_per_item,
            "max_context_chars": args.max_context_chars,
            "online_provider": args.online_provider,
            "speculative_notes_search": args.speculative_notes_search,
        },
        **run_payload,
    }


def write_outputs(out_base: Path, payload: dict) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_base if out_base.suffix == ".json" else out_base.with_suffix(".json")
    md_path = out_base.with_suffix(".md")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")


def render_markdown(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# Agent Answer Debug")
    lines.append("")
    lines.append("## Query")
    lines.append("")
    lines.append(payload["query"])
    lines.append("")
    lines.append("## Router")
    lines.append("")
    router = payload["router_decision"]
    lines.append(f"- command: `{router['command']}`")
    lines.append(f"- confidence: `{router['confidence']}`")
    lines.append(f"- fallback_used: `{router['fallback_used']}`")
    lines.append(f"- reason: {router['reason']}")
    lines.append("")
    lines.append("## Answer")
    lines.append("")
    lines.append(payload["answer"]["answer"])
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append(payload["context_text"] or "(no context)")
    lines.append("")
    lines.append("## Retrieval Summary")
    lines.append("")
    retrieval = payload["retrieval"]
    lines.append(f"- notes: `{len(retrieval['notes'])}`")
    lines.append(f"- rg: `{len(retrieval['rg'])}`")
    lines.append(f"- bm25: `{len(retrieval['bm25'])}`")
    lines.append(f"- online: `{len(retrieval['online']['results'])}` ({retrieval['online']['message']})")
    lines.append(f"- tool_errors: `{len(payload.get('tool_errors', []))}`")
    if payload.get("tool_errors"):
        lines.append("")
        for error in payload["tool_errors"]:
            lines.append(f"- `{error['tool']}`: `{error['error_type']}` {error['message']}")
    lines.append("")
    lines.append("## Timing")
    lines.append("")
    for key, value in payload["timing"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Telemetry")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(payload.get("telemetry", {}), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
