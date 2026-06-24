from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

APP_PATH = PROJECT_SRC / "web" / "app.py"


def _load_web_app():
    sys.modules.setdefault("faiss", MagicMock())
    from web import app as web_app

    return web_app


class WebWorkspaceRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.web_app = _load_web_app()

    def test_workspace_script_in_head_before_body_content(self) -> None:
        render_web_page = self.web_app.render_web_page
        for raw_html, title in ((self.web_app.REVIEW_HTML, "复习"), (self.web_app.CHAT_HTML, "对话")):
            with self.subTest(page=title):
                html = render_web_page(raw_html, title)
                workspace_pos = html.find("KnowledgeAgentWorkspace")
                head_end = html.find("</head>")
                body_start = html.find("<body>")
                page_script_pos = html.find("initReviewPage" if title == "复习" else "initChatWorkspace", body_start)
                self.assertGreater(workspace_pos, 0)
                self.assertIn("KnowledgeAgentWorkspace", html)
                self.assertLess(workspace_pos, head_end)
                self.assertLess(workspace_pos, body_start)
                self.assertGreater(page_script_pos, workspace_pos)

    def test_workspace_store_exposes_migration_api(self) -> None:
        script = self.web_app.WORKSPACE_STORE_SCRIPT
        self.assertIn("migrateReviewSlice", script)
        self.assertIn("migrateChatSlice", script)
        self.assertIn("migrate: migrateAll", script)
        self.assertIn("REVIEW_SLICE_VERSION", script)

    def test_render_template_injects_workspace_before_page_content(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        workspace_in_template = source.find("{WORKSPACE_STORE_SCRIPT}")
        content_in_template = source.find("{content}", workspace_in_template)
        self.assertGreater(workspace_in_template, 0)
        self.assertGreater(content_in_template, workspace_in_template)

    def test_review_html_has_progress_copy(self) -> None:
        review_html = self.web_app.REVIEW_HTML
        self.assertIn("已完成", review_html)
        self.assertIn("共 ${total}", review_html)
        self.assertIn("migrateReviewSnapshot", review_html)


if __name__ == "__main__":
    unittest.main()
