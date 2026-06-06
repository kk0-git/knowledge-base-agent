from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

@dataclass(frozen=True)
class EmbeddingPair:
    source: str
    target: str
    score: float
    same_directory: bool
    source_title: str
    target_title: str
    source_tags: list[str]
    target_tags: list[str]


@dataclass(frozen=True)
class NoteNeighbor:
    target: str
    score: float
    same_directory: bool
    target_title: str
    target_tags: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug note-level embedding similarity")
    parser.add_argument(
        "--profiles",
        default="audit-output/note_profiles.json",
        help="Path to note_profiles.json",
    )
    parser.add_argument(
        "--out",
        default="audit-output/embedding_debug.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-large-zh-v1.5",
        help="SentenceTransformer model name",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top neighbors per note")
    parser.add_argument("--global-top", type=int, default=50, help="Top global pairs")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum score for global pair output",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=2600,
        help="Max embedding input chars per note",
    )
    parser.add_argument(
        "--input-mode",
        choices=("labeled", "raw"),
        default="labeled",
        help="Embedding input format: labeled (Title:/Tags: prefixes) or raw (plain text)",
    )
    args = parser.parse_args()

    profiles_path = Path(args.profiles)
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))

    if not profiles:
        raise ValueError(f"No profiles found: {profiles_path}")

    build_fn = build_embedding_text_raw if args.input_mode == "raw" else build_embedding_text_labeled
    texts = [build_fn(profile, max_chars=args.max_text_chars) for profile in profiles]
    paths = [profile["path"] for profile in profiles]

    print(f"Profiles: {len(profiles)}")
    print(f"Model: {args.model}")
    print("Loading model...")
    model = SentenceTransformer(args.model)
    print("DEBUG profiles:", args.profiles)
    print("DEBUG out:", args.out)
    print("DEBUG model:", args.model)


    print("Encoding notes...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    similarity = embeddings @ embeddings.T

    global_pairs = build_global_pairs(
        profiles=profiles,
        similarity=similarity,
        global_top=args.global_top,
        min_score=args.min_score,
    )

    per_note_top_k = build_per_note_top_k(
        profiles=profiles,
        similarity=similarity,
        top_k=args.top_k,
    )

    payload = {
        "model": args.model,
        "profile_count": len(profiles),
        "config": {
            "top_k": args.top_k,
            "global_top": args.global_top,
            "min_score": args.min_score,
            "max_text_chars": args.max_text_chars,
            "input_mode": args.input_mode,
        },
        "global_top_pairs": [asdict(pair) for pair in global_pairs],
        "per_note_top_k": {
            path: [asdict(neighbor) for neighbor in neighbors]
            for path, neighbors in per_note_top_k.items()
        },
        "embedding_inputs_preview": [
            {
                "path": profile["path"],
                "title": profile.get("title", ""),
                "text_chars": len(text),
                "text_preview": text[:500],
            }
            for profile, text in zip(profiles, texts, strict=False)
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Global pairs: {len(global_pairs)}")
    print(f"Wrote: {out_path.resolve()}")
    return 0


def build_embedding_text_labeled(profile: dict[str, Any], max_chars: int) -> str:
    """带英文字段前缀的输入格式：Title: xxx, Tags: xxx, Content: xxx"""
    path = str(profile.get("path", ""))
    title = str(profile.get("title", ""))
    note_type = str(profile.get("note_type", "unknown"))
    tags = profile.get("tags", []) or []
    headings = profile.get("headings", []) or []
    semantic_excerpt = profile.get("semantic_excerpt") or profile.get("excerpt", "")

    parts = [
        f"Title: {title}",
        f"Path: {path}",
        f"Type: {note_type}",
    ]

    if tags:
        parts.append("Tags: " + ", ".join(str(tag) for tag in tags))

    if headings:
        parts.append("Headings: " + " | ".join(str(heading) for heading in headings[:30]))

    if semantic_excerpt:
        parts.append("Content:\n" + str(semantic_excerpt))

    text = "\n".join(parts).strip()

    if len(text) > max_chars:
        return text[:max_chars]

    return text


def build_embedding_text_raw(profile: dict[str, Any], max_chars: int) -> str:
    """无前缀的输入格式：标题 + 标签 + headings + 正文自然拼接"""
    title = str(profile.get("title", ""))
    tags = profile.get("tags", []) or []
    headings = profile.get("headings", []) or []
    semantic_excerpt = profile.get("semantic_excerpt") or profile.get("excerpt", "")

    parts: list[str] = []

    if title:
        parts.append(f"# {title}")

    if tags:
        parts.append(" ".join(str(tag) for tag in tags))

    if headings:
        parts.append(" | ".join(str(h) for h in headings[:30]))

    if semantic_excerpt:
        parts.append(str(semantic_excerpt))

    text = "\n\n".join(parts).strip()

    if len(text) > max_chars:
        return text[:max_chars]

    return text


def build_global_pairs(
    profiles: list[dict[str, Any]],
    similarity: np.ndarray,
    global_top: int,
    min_score: float,
) -> list[EmbeddingPair]:
    pairs: list[EmbeddingPair] = []

    for source_index, source_profile in enumerate(profiles):
        for target_index in range(source_index + 1, len(profiles)):
            target_profile = profiles[target_index]
            score = float(similarity[source_index, target_index])

            if score < min_score:
                continue

            pairs.append(
                EmbeddingPair(
                    source=source_profile["path"],
                    target=target_profile["path"],
                    score=round(score, 4),
                    same_directory=same_directory(
                        source_profile["path"],
                        target_profile["path"],
                    ),
                    source_title=str(source_profile.get("title", "")),
                    target_title=str(target_profile.get("title", "")),
                    source_tags=list(source_profile.get("tags", []) or []),
                    target_tags=list(target_profile.get("tags", []) or []),
                )
            )

    pairs.sort(key=lambda pair: pair.score, reverse=True)
    return pairs[:global_top]


def build_per_note_top_k(
    profiles: list[dict[str, Any]],
    similarity: np.ndarray,
    top_k: int,
) -> dict[str, list[NoteNeighbor]]:
    result: dict[str, list[NoteNeighbor]] = {}

    for source_index, source_profile in enumerate(profiles):
        neighbors: list[NoteNeighbor] = []

        for target_index, target_profile in enumerate(profiles):
            if source_index == target_index:
                continue

            score = float(similarity[source_index, target_index])

            neighbors.append(
                NoteNeighbor(
                    target=target_profile["path"],
                    score=round(score, 4),
                    same_directory=same_directory(
                        source_profile["path"],
                        target_profile["path"],
                    ),
                    target_title=str(target_profile.get("title", "")),
                    target_tags=list(target_profile.get("tags", []) or []),
                )
            )

        neighbors.sort(key=lambda neighbor: neighbor.score, reverse=True)
        result[source_profile["path"]] = neighbors[:top_k]

    return result


def same_directory(left_path: str, right_path: str) -> bool:
    return get_directory(left_path) == get_directory(right_path)


def get_directory(path: str) -> str:
    parent = PurePosixPath(path.replace("\\", "/")).parent.as_posix()
    if parent == ".":
        return "root"
    return parent


if __name__ == "__main__":
    raise SystemExit(main())
