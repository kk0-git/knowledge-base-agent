from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*")
API_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*(?:\.[A-Za-z_][A-Za-z0-9_+-]*)+")
ZH_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")

STOP_TOKENS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "with",
    "is",
    "are",
    "be",
    "this",
    "that",
    "use",
    "using",
    "used",
    "get",
    "set",
    "create",
    "update",
    "delete",
    "file",
    "config",
    "这个",
    "那个",
    "什么",
    "一个",
    "可以",
    "如果",
    "因为",
    "所以",
    "使用",
    "实现",
    "需要",
    "问题",
    "方式",
    "进行",
    "通过",
    "当前",
    "建议",
    "阶段",
    "内容",
    "记录",
}


@dataclass(frozen=True)
class PairEvidence:
    source: str
    target: str
    score: float
    shared_tokens: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug lexical tokens from note_profiles.json")
    parser.add_argument(
        "--profiles",
        default="audit-output/note_profiles.json",
        help="Path to note_profiles.json",
    )
    parser.add_argument(
        "--out",
        default="audit-output/token_debug.json",
        help="Output JSON path",
    )
    parser.add_argument("--top", type=int, default=40, help="Top tokens per note")
    parser.add_argument("--pairs", type=int, default=30, help="Top pair evidences")
    parser.add_argument("--threshold", type=float, default=0.18, help="Pair score threshold")
    parser.add_argument("--min-shared", type=int, default=3, help="Minimum shared tokens")
    args = parser.parse_args()

    profiles_path = Path(args.profiles)
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))

    token_counters = {profile["path"]: tokenize_profile(profile) for profile in profiles}

    note_tokens = [
        {
            "path": profile["path"],
            "title": profile.get("title", ""),
            "tags": profile.get("tags", []),
            "top_tokens": counter.most_common(args.top),
            "token_count": sum(counter.values()),
            "unique_token_count": len(counter),
        }
        for profile in profiles
        for counter in [token_counters[profile["path"]]]
    ]

    pair_evidence = build_pair_evidence(
        profiles=profiles,
        token_counters=token_counters,
        threshold=args.threshold,
        min_shared=args.min_shared,
    )

    payload = {
        "config": {
            "threshold": args.threshold,
            "min_shared": args.min_shared,
            "top_tokens_per_note": args.top,
            "top_pairs": args.pairs,
        },
        "notes": note_tokens,
        "pairs": [asdict(pair) for pair in pair_evidence[: args.pairs]],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Profiles: {len(profiles)}")
    print(f"Pairs over threshold: {len(pair_evidence)}")
    print(f"Wrote: {out_path.resolve()}")
    return 0


def tokenize_profile(profile: dict[str, Any]) -> Counter[str]:
    counter: Counter[str] = Counter()

    add_weighted(counter, tokenize_path(profile.get("path", "")), weight=2)
    add_weighted(counter, tokenize_text(profile.get("title", "")), weight=3)

    for tag in profile.get("tags", []):
        add_weighted(counter, tokenize_tag(tag), weight=3)

    for heading in profile.get("headings", []):
        add_weighted(counter, tokenize_text(heading), weight=2)

    semantic_text = profile.get("semantic_excerpt") or profile.get("excerpt", "")
    add_weighted(counter, tokenize_text(semantic_text), weight=1)

    return Counter({token: count for token, count in counter.items() if token not in STOP_TOKENS})


def add_weighted(counter: Counter[str], tokens: list[str], weight: int) -> None:
    for token in tokens:
        if token and token not in STOP_TOKENS:
            counter[token] += weight


def tokenize_path(path: str) -> list[str]:
    stem = path.replace("\\", "/").removesuffix(".md")
    parts = re.split(r"[/\s._-]+", stem)
    tokens: list[str] = []
    for part in parts:
        tokens.extend(tokenize_text(part))
    return tokens


def tokenize_tag(tag: str) -> list[str]:
    normalized = tag.strip().strip("#").replace("\\", "/").lower()
    parts = [part for part in normalized.split("/") if part]
    tokens: list[str] = []

    if len(parts) >= 3:
        tokens.append("/".join(parts[:2]))

    for part in parts:
        if part == "tags":
            continue
        tokens.extend(tokenize_text(part))

    tokens.append(normalized)
    return dedupe_keep_order(tokens)


def tokenize_text(text: str) -> list[str]:
    normalized = text.lower()
    tokens: list[str] = []

    # Keep API-like tokens as evidence, then also keep their components.
    for match in API_TOKEN_RE.finditer(normalized):
        api_token = match.group(0)
        tokens.append(api_token)
        tokens.extend(api_token.split("."))

    for match in EN_TOKEN_RE.finditer(normalized):
        tokens.append(match.group(0))

    for match in ZH_RUN_RE.finditer(normalized):
        tokens.extend(chinese_bigrams(match.group(0)))

    return dedupe_keep_order([token for token in tokens if len(token) >= 2])


def chinese_bigrams(text: str) -> list[str]:
    if len(text) < 2:
        return []
    if len(text) == 2:
        return [text]
    return [text[index : index + 2] for index in range(len(text) - 1)]


def dedupe_keep_order(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


def build_pair_evidence(
    profiles: list[dict[str, Any]],
    token_counters: dict[str, Counter[str]],
    threshold: float,
    min_shared: int,
) -> list[PairEvidence]:
    pairs: list[PairEvidence] = []

    for left_index, left in enumerate(profiles):
        for right in profiles[left_index + 1 :]:
            left_path = left["path"]
            right_path = right["path"]
            left_counter = token_counters[left_path]
            right_counter = token_counters[right_path]

            shared = shared_tokens(left_counter, right_counter)
            if len(shared) < min_shared:
                continue

            score = multiset_jaccard(left_counter, right_counter)
            if score < threshold:
                continue

            pairs.append(
                PairEvidence(
                    source=left_path,
                    target=right_path,
                    score=round(score, 4),
                    shared_tokens=shared[:20],
                )
            )

    pairs.sort(key=lambda pair: pair.score, reverse=True)
    return pairs


def multiset_jaccard(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0

    intersection = sum(min(left[key], right[key]) for key in keys)
    union = sum(max(left[key], right[key]) for key in keys)
    if union == 0:
        return 0.0

    return intersection / union


def shared_tokens(left: Counter[str], right: Counter[str]) -> list[str]:
    shared = []
    for token in set(left) & set(right):
        shared.append((token, min(left[token], right[token])))

    shared.sort(key=lambda item: (-item[1], item[0]))
    return [token for token, _ in shared]


if __name__ == "__main__":
    raise SystemExit(main())
