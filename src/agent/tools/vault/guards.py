from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Iterable


class VaultPathError(ValueError):
    pass


def normalize_relative_path(path: str) -> str:
    raw = str(path or "").strip().replace("\\", "/")
    if not raw:
        raise VaultPathError("path is required")
    pure = PurePosixPath(raw)
    if pure.is_absolute():
        raise VaultPathError(f"absolute paths are not allowed: {path}")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise VaultPathError(f"path traversal is not allowed: {path}")
    normalized = pure.as_posix()
    if not normalized.lower().endswith(".md"):
        raise VaultPathError(f"only markdown notes are allowed: {path}")
    return normalized


def resolve_vault_note_path(vault_root: Path | None, path: str) -> tuple[str, Path]:
    if vault_root is None:
        raise VaultPathError("vault_root is required")
    relative = normalize_relative_path(path)
    root = vault_root.resolve()
    full_path = (root / relative).resolve()
    try:
        full_path.relative_to(root)
    except ValueError as exc:
        raise VaultPathError(f"path is outside vault root: {path}") from exc
    if not full_path.exists():
        raise FileNotFoundError(f"note not found: {relative}")
    if not full_path.is_file():
        raise VaultPathError(f"path is not a file: {relative}")
    return relative, full_path


def normalize_scope_paths(scope_note_paths: Iterable[str]) -> set[str]:
    paths: set[str] = set()
    for raw in scope_note_paths:
        try:
            paths.add(normalize_relative_path(str(raw)))
        except VaultPathError:
            continue
    return paths


def is_scope_allowed(path: str, scope_note_paths: Iterable[str]) -> bool:
    scope = normalize_scope_paths(scope_note_paths)
    if not scope:
        return True
    return normalize_relative_path(path) in scope


def require_scope_allowed(path: str, scope_note_paths: Iterable[str]) -> str:
    relative = normalize_relative_path(path)
    if not is_scope_allowed(relative, scope_note_paths):
        raise PermissionError(f"path is outside current scope: {relative}")
    return relative


def filter_items_by_scope(items: Iterable[Any], scope_note_paths: Iterable[str], path_getter) -> list[Any]:
    scope = normalize_scope_paths(scope_note_paths)
    if not scope:
        return list(items)
    filtered: list[Any] = []
    for item in items:
        raw_path = path_getter(item)
        try:
            relative = normalize_relative_path(str(raw_path))
        except VaultPathError:
            continue
        if relative in scope:
            filtered.append(item)
    return filtered


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars].rstrip() + "\n...[truncated]", True
