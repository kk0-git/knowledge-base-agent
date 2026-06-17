from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from wiki_debug import (  # noqa: E402
    add_common_args,
    add_sync_args,
    build_manager,
    print_json,
    print_update_topic_success,
    run_update_topic,
    run_watch,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync workspace state, RAG index, wiki tags, and Obsidian report"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    watch_parser = subparsers.add_parser(
        "watch",
        help="Sync changed notes, optional RAG index, wiki tags, and _Wiki Report.md",
    )
    add_common_args(watch_parser)
    add_sync_args(watch_parser)
    watch_parser.add_argument("--quiet-seconds", type=float, default=300.0)
    watch_parser.add_argument("--poll-seconds", type=float, default=15.0)
    watch_parser.add_argument("--sync-on-start", action="store_true")
    watch_parser.add_argument("--once", action="store_true", help="Run one sync and exit")

    update_topic_parser = subparsers.add_parser(
        "update-topic",
        help="Sync changed notes, then generate one topic wiki and refresh _Wiki Report.md",
    )
    add_common_args(update_topic_parser)
    add_sync_args(update_topic_parser)
    update_topic_parser.add_argument("--tag", required=True, help="Tag to synthesize, e.g. java/servlet")

    args = parser.parse_args()
    manager = build_manager(args)

    if args.command == "watch":
        payload = run_watch(args, manager)
        print_json(payload)
        return 0

    if args.command == "update-topic":
        payload = run_update_topic(args, manager)
        print_update_topic_success(payload)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
