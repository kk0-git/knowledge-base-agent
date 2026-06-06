from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from knowledge_base_agent.config import load_llm_config
from knowledge_base_agent.llm import create_llm_client
from services.rag.query_plan import LLMQueryPlanner, QueryPlanParseError


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate standalone LLM query plans for debugging")
    parser.add_argument("--query", default=None, help="Single query to analyze")
    parser.add_argument("--input", default=None, help="JSON file containing query cases")
    parser.add_argument("--out", default="./eval-results/query-plan-debug.json", help="Output JSON path")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of queries")
    parser.add_argument("--temperature", type=float, default=None, help="Override LLM temperature")
    parser.add_argument(
        "--no-response-format",
        action="store_true",
        help="Do not send OpenAI-compatible response_format={type: json_object}",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed query instead of continuing the batch",
    )
    args = parser.parse_args()

    if bool(args.query) == bool(args.input):
        parser.error("Specify exactly one of --query or --input")

    cases = load_cases(args)
    if args.limit is not None:
        cases = cases[: args.limit]

    llm_config = load_llm_config(PROJECT_ROOT)
    if args.temperature is not None:
        llm_config = replace(llm_config, temperature=args.temperature)

    client = create_llm_client(llm_config)
    planner = LLMQueryPlanner(
        client=client,
        model=llm_config.model,
        temperature=llm_config.temperature,
        use_response_format=not args.no_response_format,
    )

    items: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        query = str(case["query"])
        print(f"[{index}/{len(cases)}] {query}")

        try:
            result = planner.plan(query)
            item = {
                "query": query,
                "ok": True,
                "input_case": case,
                "plan": asdict(result.plan),
                "validation_warnings": result.validation_warnings,
                "raw_output": result.raw_output,
                "error": None,
            }
        except QueryPlanParseError as exc:
            item = {
                "query": query,
                "ok": False,
                "input_case": case,
                "plan": None,
                "validation_warnings": [],
                "raw_output": exc.raw_output,
                "error": str(exc),
            }
            print(f"  failed: {exc}")
            if args.fail_fast:
                items.append(item)
                break
        except Exception as exc:
            item = {
                "query": query,
                "ok": False,
                "input_case": case,
                "plan": None,
                "validation_warnings": [],
                "raw_output": "",
                "error": str(exc),
            }
            print(f"  failed: {exc}")
            if args.fail_fast:
                items.append(item)
                break

        items.append(item)

    payload = {
        "config": {
            "query": args.query,
            "input": args.input,
            "limit": args.limit,
            "provider": llm_config.provider,
            "base_url": llm_config.base_url,
            "model": llm_config.model,
            "temperature": llm_config.temperature,
            "use_response_format": not args.no_response_format,
        },
        "summary": {
            "query_count": len(items),
            "ok_count": sum(1 for item in items if item["ok"]),
            "error_count": sum(1 for item in items if not item["ok"]),
            "warning_count": sum(len(item["validation_warnings"]) for item in items),
        },
        "items": items,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print(f"Saved query plan debug result: {out_path.resolve()}")
    print(
        "Summary: "
        f"ok={payload['summary']['ok_count']}, "
        f"errors={payload['summary']['error_count']}, "
        f"warnings={payload['summary']['warning_count']}"
    )
    return 0 if payload["summary"]["error_count"] == 0 else 1


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.query:
        return [{"query": args.query}]

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    raw_cases: list[Any]
    if isinstance(data, list):
        raw_cases = data
    elif isinstance(data, dict) and isinstance(data.get("cases"), list):
        raw_cases = data["cases"]
    elif isinstance(data, dict) and isinstance(data.get("items"), list):
        raw_cases = data["items"]
    else:
        raise ValueError("Input JSON must be a list, or an object with a cases/items list")

    cases: list[dict[str, Any]] = []
    for index, raw_case in enumerate(raw_cases):
        if isinstance(raw_case, str):
            cases.append({"query": raw_case})
            continue

        if not isinstance(raw_case, dict):
            raise ValueError(f"Input case #{index} must be an object or string")

        if "query" not in raw_case:
            raise ValueError(f"Input case #{index} missing query")

        cases.append(dict(raw_case))

    return cases


if __name__ == "__main__":
    raise SystemExit(main())
