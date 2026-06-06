
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ScannedNote:
    path: Path
    relative_path: str
    size_bytes: int
    modified_time: float

@dataclass(frozen=True)
class ScanResult:
    vault_path: Path
    notes: list[ScannedNote]
    excluded_count: int
    failed: list[tuple[str, str]]

class ExclusionFilter:
    def __init__(self, patterns: list[str]) -> None:
        self.patterns = patterns

    def should_exclude(self, relative_path: str) -> bool:
        """检查路径是否匹配任何排除模式"""
        normalized = relative_path.replace("\\", "/")

        for pattern in self.patterns:
            pattern = pattern.replace("\\", "/")

            if pattern.endswith("/"):
                folder = pattern.rstrip("/")
                first_part = normalized.split("/")[0]

                if normalized == folder:
                    return True
                if normalized.startswith(folder + "/"):
                    return True
                if first_part == folder:
                    return True

            if fnmatch.fnmatch(normalized, pattern):
                return True

            if "/" not in pattern and fnmatch.fnmatch(Path(normalized).name, pattern):
                return True

        return False
    
def scan_vault(vault_path: Path, exclusion_filter: ExclusionFilter) -> ScanResult:
    """扫描 Obsidian vault 中的所有 markdown 文件，返回扫描结果。"""
    vault_path = vault_path.resolve()

    if not vault_path.exists():
        raise FileNotFoundError(f"Vault path does not exist: {vault_path}")

    if not vault_path.is_dir():
        raise NotADirectoryError(f"Vault path is not a directory: {vault_path}")
    
    notes: list[ScannedNote] = []
    failed: list[tuple[str, str]] = []
    excluded_count = 0

    for markdown_path in vault_path.rglob("*.md"):
        try:
            relative_path = markdown_path.relative_to(vault_path).as_posix() # 转换为 POSIX 风格路径（使用 / 作为分隔符）

            if exclusion_filter.should_exclude(relative_path):
                # 路径被排除，增加计数并跳过
                excluded_count += 1
                continue

            stat = markdown_path.stat() # 获取文件的统计信息（大小、修改时间等）
            notes.append(
                ScannedNote(
                    path=markdown_path,
                    relative_path=relative_path,
                    size_bytes=stat.st_size,
                    modified_time=stat.st_mtime,
                )
            )
        except OSError as exc:
            failed.append((str(markdown_path), str(exc)))

    notes.sort(key=lambda note: note.relative_path.lower())

    return ScanResult(
        vault_path=vault_path,
        notes=notes,
        excluded_count=excluded_count,
        failed=failed,
    )