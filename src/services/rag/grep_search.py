from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class GrepMatch:
    path: str
    line: int
    text: str
    source: str = "rg"


def rg_search(
    *,
    vault_root: Path,
    query: str,
    limit: int = 10,
    ignore_case: bool = True,
    fixed_strings: bool = False,
) -> list[GrepMatch]:
    if not query.strip():
        return []
    if not vault_root.exists():
        raise FileNotFoundError(f"Vault path not found: {vault_root}")

    args = [
        "rg",
        "--json",
        "--line-number",
        "--trim",
        "--glob",
        "*.md",
    ]
    if ignore_case:
        args.append("--ignore-case")
    if fixed_strings:
        args.append("--fixed-strings")
    args.extend(["-e", query, str(vault_root)])

    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        return python_grep_search(
            vault_root=vault_root,
            query=query,
            limit=limit,
            ignore_case=ignore_case,
        )

    if completed.returncode not in {0, 1}:
        if not fixed_strings:
            return rg_search(
                vault_root=vault_root,
                query=query,
                limit=limit,
                ignore_case=ignore_case,
                fixed_strings=True,
            )
        return python_grep_search(
            vault_root=vault_root,
            query=query,
            limit=limit,
            ignore_case=ignore_case,
        )

    matches: list[GrepMatch] = []
    for line in completed.stdout.splitlines():
        if len(matches) >= limit:
            break
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue
        data = event.get("data", {})
        path = Path(data.get("path", {}).get("text", ""))
        lines = data.get("lines", {}).get("text", "")
        line_number = int(data.get("line_number", 0))
        matches.append(
            GrepMatch(
                path=relative_path(path, vault_root),
                line=line_number,
                text=lines.strip(),
                source="rg",
            )
        )

    return matches


def python_grep_search(
    *,
    vault_root: Path,
    query: str,
    limit: int = 10,
    ignore_case: bool = True,
) -> list[GrepMatch]:
    needle = query.lower() if ignore_case else query
    matches: list[GrepMatch] = []

    for path in vault_root.rglob("*.md"):
        if len(matches) >= limit:
            break
        if should_skip_path(path):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            haystack = line.lower() if ignore_case else line
            if needle in haystack:
                matches.append(
                    GrepMatch(
                        path=relative_path(path, vault_root),
                        line=line_number,
                        text=line.strip(),
                        source="python-grep",
                    )
                )
                if len(matches) >= limit:
                    break

    return matches


def should_skip_path(path: Path) -> bool:
    parts = set(path.parts)
    return any(part in parts for part in {".git", ".obsidian", "node_modules"})


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def grep_matches_to_dict(matches: list[GrepMatch]) -> list[dict]:
    return [asdict(match) for match in matches]
