from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge_base_agent.config import load_exclusion_patterns, load_llm_config
from knowledge_base_agent.scanner import ExclusionFilter, scan_vault
from knowledge_base_agent.parser import parse_note
from knowledge_base_agent.profile import build_note_profile, profile_to_dict
from knowledge_base_agent.graph import build_knowledge_graph, graph_to_dict
from knowledge_base_agent.candidates import build_candidate_report, candidate_report_to_dict
from knowledge_base_agent.llm import create_llm_client
from knowledge_base_agent.packet import build_review_packets, review_packet_to_dict
from knowledge_base_agent.reviewer import (
    packet_review_report_to_dict,
    review_packets_with_llm,
)
from knowledge_base_agent.report import build_vault_audit_report



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="knowledge-agent",
        description="Local Obsidian vault audit tool",
    )

    subcommands = parser.add_subparsers(dest="command", required=True)

    audit = subcommands.add_parser("audit", help="Audit an Obsidian vault")
    audit.add_argument("--vault", required=True, help="Path to external Obsidian vault")
    audit.add_argument("--out", default="./audit-output", help="Output directory")
    audit.add_argument("--llm", action="store_true", help="Enable LLM review")
    audit.add_argument("--llm-limit", type=int, default=10, help="Max candidates to review with LLM")

    return parser

def run_audit(vault: str, out: str, enable_llm: bool = False, llm_limit: int = 10) -> int:
    vault_path = Path(vault)
    output_path = Path(out)

    exclusions = load_exclusion_patterns(vault_path)
    exclusion_filter = ExclusionFilter(exclusions)
    result = scan_vault(vault_path, exclusion_filter)

    parsed_notes = []
    profiles = []
    for note in result.notes:
        try:
            parsed_note = parse_note(note)
            parsed_notes.append(parsed_note)
            profiles.append(build_note_profile(parsed_note))
        except OSError as e:
            result.failed.append((note.relative_path, str(e)))

    output_path.mkdir(parents=True, exist_ok=True)
    profiles_path = output_path / "note_profiles.json"
    profiles_path.write_text(
        json.dumps([profile_to_dict(profile) for profile in profiles], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    graph = build_knowledge_graph(profiles)
    graph_path = output_path / "knowledge_graph.json"
    graph_path.write_text(
        json.dumps(graph_to_dict(graph), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    candidate_report = build_candidate_report(profiles, graph)
    suggestions_path = output_path / "candidate_report.json"
    suggestions_path.write_text(
        json.dumps(candidate_report_to_dict(candidate_report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    review_packets = build_review_packets(
        profiles=profiles,
        graph=graph,
        candidate_report=candidate_report,
        max_notes_per_packet=12,
        max_packets=5,
    )

    packets_path = output_path / "review_packets.json"
    packets_path.write_text(
        json.dumps(
            [review_packet_to_dict(packet) for packet in review_packets],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    llm_review_data = None
    if enable_llm:
        llm_config = load_llm_config(Path.cwd())
        llm_client = create_llm_client(llm_config)

        packet_review = review_packets_with_llm(
            packets=review_packets,
            client=llm_client,
            config=llm_config,
            limit=llm_limit,
        )

        llm_review_data = packet_review_report_to_dict(packet_review)

        llm_review_path = output_path / "llm_review.json"
        llm_review_path.write_text(
            json.dumps(packet_review_report_to_dict(packet_review), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"LLM review written: {llm_review_path}")
        print(f"LLM packet reviews: {len(packet_review.packet_reviews)}")

        audit_report = build_vault_audit_report(
            profiles=profiles,
            graph=graph,
            candidate_report=candidate_report,
            review_packets=review_packets,
            llm_review=llm_review_data,
        )

        audit_report_path = output_path / "vault_audit.md"
        audit_report_path.write_text(audit_report, encoding="utf-8")

        print(f"Audit report written: {audit_report_path}")




    print(f"Vault: {result.vault_path}")
    print(f"Markdown notes: {len(result.notes)}")
    print(f"Excluded markdown files: {result.excluded_count}")
    print(f"Failed files: {len(result.failed)}")
    print(f"Output: {output_path.resolve()}")
    print(f"Parsed notes: {len(parsed_notes)}")
    print(f"Profiles written: {profiles_path}")

    
    print(f"Graph written: {graph_path}")
    print(f"Graph nodes: {len(graph.nodes)}")
    print(f"Graph edges: {len(graph.edges)}")
    print(f"Orphan notes: {len(graph.orphan_notes)}")
    print(f"Unresolved links: {len(graph.unresolved_links)}")

    print(f"Suggestions written: {suggestions_path}")
    print(f"Orphan candidates: {len(candidate_report.orphan_notes)}")
    print(f"Weakly linked candidates: {len(candidate_report.weakly_linked_notes)}")
    print(f"Hub candidates: {len(candidate_report.hub_candidates)}")
    print(f"Unresolved link candidates: {len(candidate_report.unresolved_links)}")

    return 0

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "audit":
        # return run_audit(args.vault, args.out)
        return run_audit(args.vault, args.out, enable_llm=args.llm, llm_limit=args.llm_limit)

    parser.error(f"Unknown command: {args.command}")
    return 2

if __name__ == "__main__":
    raise SystemExit(main()) # 使用 SystemExit 以确保正确的退出代码被返回
