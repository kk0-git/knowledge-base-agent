from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from services.markdown.sections import (
    child_sections,
    find_sections,
    parse_markdown_sections,
    section_to_dict,
)


class MarkdownSectionTests(unittest.TestCase):
    def test_parse_markdown_sections_builds_heading_path(self) -> None:
        text = "# Memory\nintro\n## Types\nshort\nlong\n# Other\nx"
        sections = parse_markdown_sections(text)
        self.assertEqual(len(sections), 3)
        self.assertEqual(sections[1].heading_path, ["Memory", "Types"])
        self.assertIn("short", sections[1].content)

    def test_parse_markdown_sections_ignores_headings_in_fenced_code(self) -> None:
        text = "# Real\n```md\n# Fake\n```\nbody"
        sections = parse_markdown_sections(text)
        self.assertEqual(len(sections), 1)
        self.assertIn("# Fake", sections[0].content)

    def test_find_sections_by_heading_is_ambiguous_when_duplicated(self) -> None:
        text = "# Topic\na\n## Summary\none\n# Topic 2\nb\n## Summary\ntwo"
        sections = parse_markdown_sections(text)
        matches, error = find_sections(sections, heading="Summary")
        self.assertEqual(error, "ambiguous")
        self.assertEqual(len(matches), 2)

    def test_find_sections_by_heading_path_is_exact(self) -> None:
        text = "# Topic\na\n## Summary\none\n# Topic 2\nb\n## Summary\ntwo"
        sections = parse_markdown_sections(text)
        matches, error = find_sections(sections, heading_path=["Topic 2", "Summary"])
        self.assertIsNone(error)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].content, "## Summary\ntwo")

    def test_child_sections_uses_heading_path_prefix(self) -> None:
        text = "# Memory\nintro\n## Short-term\ns\n## Long-term\nl"
        sections = parse_markdown_sections(text)
        parent = sections[0]
        children = child_sections(parent, sections)
        self.assertEqual([child.heading for child in children], ["Short-term", "Long-term"])

    def test_section_to_dict_includes_preview_and_id(self) -> None:
        section = parse_markdown_sections("# Title\nhello world")[0]
        payload = section_to_dict(section, preview_chars=20)
        self.assertEqual(payload["heading"], "Title")
        self.assertTrue(str(payload["id"]).startswith("h1-"))
        self.assertIn("hello", str(payload["preview"]))


if __name__ == "__main__":
    unittest.main()
