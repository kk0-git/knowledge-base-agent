"""Persistence for learner memory v4/v5."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .migration import migrate_v3_profile_to_v4, migrate_v4_to_v5
from .schema import SCHEMA_VERSION, default_learner_model, normalize_learner_model


class LearnerModelStore:
    def __init__(self, path: str | Path, legacy_path: str | Path | None = None) -> None:
        self.path = Path(path)
        self.legacy_path = Path(legacy_path) if legacy_path else None

    def load(self) -> dict[str, Any]:
        if self.path.exists():
            data = self._read_json(self.path)
            if self._is_v4_payload(data):
                if int(data.get("schema_version") or 0) == 4:
                    return self._migrate_v4_to_v5_in_place(data)
                return normalize_learner_model(data)
            if data.get("weak_points") is not None:
                return migrate_v3_profile_to_v4(data)
        if self.legacy_path and self.legacy_path.exists() and self.legacy_path != self.path:
            return migrate_v3_profile_to_v4(self._read_json(self.legacy_path))
        if self.legacy_path and self.legacy_path.exists():
            return migrate_v3_profile_to_v4(self._read_json(self.legacy_path))
        return default_learner_model()

    def save(self, model: dict[str, Any]) -> None:
        normalized = normalize_learner_model(model)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        suffix = self.path.suffix or ".json"
        tmp_path = self.path.with_suffix(f"{suffix}.tmp")
        tmp_path.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)

    def _migrate_v4_to_v5_in_place(self, v4_data: dict[str, Any]) -> dict[str, Any]:
        """Migrate v4 data to v5, backup the original, and save the result."""
        # Backup before migration
        backup_path = self.path.with_suffix(self.path.suffix + ".v4.bak")
        if not backup_path.exists():
            try:
                shutil.copy2(str(self.path), str(backup_path))
            except OSError:
                pass

        v5_model = migrate_v4_to_v5(v4_data)
        self.save(v5_model)
        return v5_model

    @staticmethod
    def _is_v4_payload(data: dict[str, Any]) -> bool:
        if int(data.get("schema_version") or 0) >= 4:
            return True
        return isinstance(data.get("learner_items") or data.get("beliefs"), list)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            raise
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}
