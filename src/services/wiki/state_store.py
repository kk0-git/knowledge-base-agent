from __future__ import annotations

import json
from pathlib import Path

from services.wiki.schema import WikiState, wiki_state_from_dict, wiki_state_to_dict


class WikiStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> WikiState:
        if not self.path.exists():
            return WikiState()

        data = json.loads(self.path.read_text(encoding="utf-8"))
        return wiki_state_from_dict(data)

    def save(self, state: WikiState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(wiki_state_to_dict(state), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)

