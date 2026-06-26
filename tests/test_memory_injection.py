import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from services.workflows.interview_profile import render_candidate_profile_context


def test_render_candidate_profile_context_skips_candidate_and_archived():
    profile = {
        "weak_points": [
            {"point": "Active gap", "scope": "universal", "lifecycle": "active"},
            {"point": "Candidate gap", "scope": "universal", "lifecycle": "candidate"},
            {"point": "Archived gap", "scope": "universal", "lifecycle": "archived"},
            {"point": "Improved gap", "scope": "universal", "lifecycle": "active", "improved": True},
        ],
        "strong_points": [],
        "communication": {},
    }

    rendered = render_candidate_profile_context(profile=profile, current_topic=None, plan=None)

    assert "Active gap" in rendered
    assert "Candidate gap" not in rendered
    assert "Archived gap" not in rendered
    assert "Improved gap" not in rendered
