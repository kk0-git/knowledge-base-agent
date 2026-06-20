from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from agent.apps import CoachTurnRequest, InterviewCoachApp
from agent.apps.interview_coach import build_coach_input, normalize_coach_payload
from agent.llm.tool_calling import LLMToolRequest, LLMToolResponse
from agent.runtime import AgentRuntime
from agent.schema import ToolCall, WorkingMemory
from agent.skill_loader import SkillLoader
from agent.tool_executor import ToolExecutionContext, ToolExecutor
from agent.tool_registry import ToolRegistry
from agent.tools.interview import register_interview_tools
from agent.tools.profile import register_profile_tools
from agent.tools.vault import register_vault_tools
from agent.trace import TraceRecorder
from services.workflows.interview import InterviewPlan, TopicCard
from services.workflows.interview_memory_commit import commit_interview_memory
from services.workflows.interview_profile import InterviewProfileStore, observations_from_turn_reviews
from services.workflows.interview_sessions import InterviewSessionStore


def sample_plan() -> InterviewPlan:
    return InterviewPlan(
        topics=(TopicCard(name="MCP", coverage=("definition", "roles"), source_note_paths=("mcp.md",)),),
        suggested_order=("MCP",),
    )


class PureReviewFinalLLM:
    def __init__(self):
        self.requests: list[LLMToolRequest] = []

    def complete_with_tools(self, request: LLMToolRequest) -> LLMToolResponse:
        self.requests.append(request)
        return LLMToolResponse(
            content=json.dumps(
                {
                    "feedback": {
                        "question_requires": ["protocol goal", "Host/Client/Server roles"],
                        "coach_note": "The answer mentions tool calling, but does not separate protocol roles.",
                        "covered": ["tool calling"],
                        "gaps": ["Host/Client/Server responsibilities are not separated"],
                        "thinking_framework": "For protocol questions, state the goal, then separate roles and call flow.",
                        "interviewer_followup_note": "The follow-up checks whether the role boundary is clear.",
                    },
                    "expression_example": "MCP should be explained as a protocol layer with Host, Client, and Server responsibilities.",
                    "profile_signals": [
                        {
                            "type": "possible_weak_point",
                            "point": "This model output must be ignored by CoachApp.",
                            "evidence": "synthetic",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            finish_reason="stop",
            used_mode="fake",
        )


class ProfileMemoryToolTests(unittest.TestCase):
    def test_recall_profile_and_record_signal_do_not_write_profile_file(self) -> None:
        registry = ToolRegistry()
        register_profile_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = Path(tmp) / "profile.json"
            store = InterviewProfileStore(profile_path)
            store.save(
                {
                    "schema_version": 3,
                    "weak_points": [
                        {
                            "point": "MCP role boundary is unclear",
                            "topic": "MCP",
                            "scope": "domain",
                            "category": "knowledge_gap",
                            "planned_layer": "roles",
                            "improved": False,
                            "domain_anchor": {"plan_topic": "MCP", "context_note_paths": ["mcp.md"], "scope_path": ""},
                            "sr": {"next_review": "2000-01-01", "ease_factor": 2.5},
                        }
                    ],
                    "strong_points": [],
                    "topic_mastery": {},
                    "communication": {"style": "", "suggestions": []},
                }
            )
            before = profile_path.read_text(encoding="utf-8")
            working = WorkingMemory(session_id="s1", current_topic="MCP", notes_read_this_turn=["mcp.md"])
            ctx = ToolExecutionContext(
                working=working,
                profile_store=store,
                interview_plan=sample_plan(),
                session_id="s1",
                turn_context={"current_topic": "MCP", "user_message_id": "u1", "assistant_message_id": "a1"},
            )
            executor = ToolExecutor(registry, ctx)
            recall = executor.execute(ToolCall(id="1", name="recall_profile", arguments={"topic": "MCP", "planned_layer": "roles"}))
            self.assertTrue(recall.ok)
            self.assertEqual(recall.output["counts"]["weak_points"], 1)
            signal = executor.execute(
                ToolCall(
                    id="2",
                    name="record_signal",
                    arguments={
                        "signal_type": "possible_weak_point",
                        "point": "MCP role boundary is unclear",
                        "evidence": "The user did not separate roles.",
                    },
                )
            )
            self.assertTrue(signal.ok)
            self.assertEqual(signal.output["signal"]["context_note_paths"], ["mcp.md"])
            self.assertEqual(profile_path.read_text(encoding="utf-8"), before)

    def test_session_signal_and_draft_storage(self) -> None:
        registry = ToolRegistry()
        register_profile_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            session_store = InterviewSessionStore(Path(tmp) / "sessions")
            session = session_store.create_session(source_type="folder", source_value="agent")
            ctx = ToolExecutionContext(
                working=WorkingMemory(session_id=session["session_id"], notes_read_this_turn=["mcp.md"]),
                session_id=session["session_id"],
                session_store=session_store,
            )
            executor = ToolExecutor(registry, ctx)
            executor.execute(
                ToolCall(
                    id="1",
                    name="record_signal",
                    arguments={"signal_type": "possible_weak_point", "point": "role boundary unclear", "evidence": "evidence"},
                )
            )
            executor.execute(
                ToolCall(
                    id="2",
                    name="write_observation_draft",
                    arguments={"point": "role boundary unclear", "evidence": "evidence"},
                )
            )
            listed = executor.execute(ToolCall(id="3", name="list_profile_signals", arguments={}))
            self.assertEqual(listed.output["signal_count"], 1)
            self.assertEqual(len(session_store.list_observation_drafts(session_id=session["session_id"])), 1)


class CoachAgentTests(unittest.TestCase):
    def test_build_coach_input_separates_evaluation_and_followup_zones(self) -> None:
        text = build_coach_input(
            CoachTurnRequest(
                previous_interviewer_question="How do you judge whether retrieval results are reliable?",
                latest_user_answer="I would use rerank.",
                interviewer_followup="What do you do when the pool is empty?",
                interview_state={"current_topic": "RAG", "current_layer_name": "fallback"},
                context_note_paths=("rag.md",),
            )
        )
        self.assertIn("# Evaluation Zone", text)
        self.assertIn("# Follow-up Zone (read-only for evaluation)", text)
        self.assertIn("Step 1", text)
        eval_zone = text.split("# Follow-up Zone")[0]
        self.assertNotIn("pool is empty", eval_zone)

    def test_normalize_coach_payload_keeps_profile_signals_empty(self) -> None:
        payload = normalize_coach_payload(
            {
                "feedback": {
                    "question_requires": ["reliable signal", "fallback strategy"],
                    "coach_note": "direction is ok",
                    "covered": ["rerank"],
                    "gaps": ["fallback condition is missing"],
                    "thinking_framework": "framework",
                    "interviewer_followup_note": "follow-up asks about fallback",
                },
                "expression_example": "",
                "profile_signals": [{"point": "must be dropped"}],
            }
        )
        self.assertEqual(payload["feedback"]["question_requires"], ["reliable signal", "fallback strategy"])
        self.assertEqual(payload["profile_signals"], [])

    def test_coach_skill_does_not_expose_profile_write_or_state_action_tools(self) -> None:
        registry = ToolRegistry()
        register_interview_tools(registry)
        register_vault_tools(registry)
        register_profile_tools(registry)
        skill = SkillLoader(PROJECT_ROOT / "skills", registry=registry).load("coach")
        self.assertNotIn("advance_layer", skill.allowed_tools)
        self.assertNotIn("select_topic", skill.allowed_tools)
        self.assertNotIn("record_signal", skill.allowed_tools)
        self.assertNotIn("recall_profile", skill.allowed_tools)
        self.assertIn("read_note", skill.allowed_tools)

    def test_coach_agent_pure_review_does_not_save_profile_signals(self) -> None:
        registry = ToolRegistry()
        register_interview_tools(registry)
        register_vault_tools(registry)
        register_profile_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_store = InterviewSessionStore(root / "sessions")
            session = session_store.create_session(source_type="folder", source_value="agent")
            pending = session_store.append_pending_turn(session_id=session["session_id"], user_content="MCP is tool calling.")
            profile_store = InterviewProfileStore(root / "profile.json")
            profile_store.save({"schema_version": 3, "weak_points": [], "strong_points": [], "topic_mastery": {}, "communication": {"style": "", "suggestions": []}})
            runtime = AgentRuntime(
                llm_client=PureReviewFinalLLM(),
                skill_loader=SkillLoader(PROJECT_ROOT / "skills", registry=registry),
                tool_registry=registry,
                trace_recorder=TraceRecorder(root / "traces"),
            )
            app = InterviewCoachApp(runtime)
            summary, result = app.run_review(
                CoachTurnRequest(
                    session_id=session["session_id"],
                    user_message_id=pending["user_message"]["id"],
                    assistant_message_id=pending["assistant_message"]["id"],
                    previous_interviewer_question="What is MCP?",
                    latest_user_answer="MCP is tool calling.",
                    interviewer_followup="Separate Host, Client, and Server.",
                    interview_plan=sample_plan(),
                    interview_state={"source": "server", "current_topic": "MCP", "current_layer_index": 0},
                    context_note_paths=("mcp.md",),
                    session_store=session_store,
                    profile_store=profile_store,
                    model="fake",
                    trace_path=str(root / "traces"),
                )
            )
            self.assertTrue(summary["available"])
            self.assertEqual(summary["feedback"]["gaps"], ["Host/Client/Server responsibilities are not separated"])
            self.assertEqual(summary["profile_signals"], [])
            reviews = session_store.load_reviews(session["session_id"])["reviews"]
            self.assertEqual(len(reviews), 1)
            self.assertEqual(reviews[0]["profile_signals"], [])
            payload = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["memory"]["profile_signal_count"], 0)

    def test_memory_commit_bridge_consumes_legacy_session_signals_and_drafts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_store = InterviewSessionStore(root / "sessions")
            session = session_store.create_session(
                source_type="folder",
                source_value="agent",
                interview_plan={"topics": [{"name": "MCP", "coverage": ["definition"], "source_note_paths": ["mcp.md"]}]},
                interview_state={"source": "server", "current_topic": "MCP", "current_layer_index": 0},
            )
            session_store.append_memory_signal(
                session_id=session["session_id"],
                signal={
                    "type": "possible_weak_point",
                    "point": "MCP role boundary is unclear",
                    "topic": "MCP",
                    "category": "knowledge_gap",
                    "scope_suggestion": "domain",
                    "evidence": "The user did not separate Host, Client, and Server.",
                    "confidence": "medium",
                    "context_note_paths": ["mcp.md"],
                },
            )
            session_store.append_observation_draft(
                session_id=session["session_id"],
                draft={
                    "point": "Answer structure misses role layering",
                    "topic": "MCP",
                    "category": "answer_structure",
                    "scope": "domain",
                    "evidence": "The user gave one generic definition.",
                    "context_note_paths": ["mcp.md"],
                },
            )
            profile_store = InterviewProfileStore(root / "profile.json")
            profile_store.save({"schema_version": 3, "weak_points": [], "strong_points": [], "topic_mastery": {}, "communication": {"style": "", "suggestions": []}})
            session = session_store.load_session(session["session_id"])
            final_review, profile_update = commit_interview_memory(
                session_store=session_store,
                profile_store=profile_store,
                session=session,
                reviews=[],
                llm_client=None,
                model=None,
            )
            bridge = profile_update["commit_bridge"]
            self.assertEqual(profile_update["source"], "commit_bridge")
            self.assertEqual(bridge["profile_signal_count"], 1)
            self.assertEqual(bridge["observation_draft_count"], 1)
            self.assertEqual(final_review["memory_commit"]["consumed_signal_count"], 1)
            trace = session_store.load_session(session["session_id"])["trace"]
            self.assertIn("memory_commit", [item.get("event") for item in trace])

    def test_observations_from_turn_reviews_uses_gaps_when_profile_signals_empty(self) -> None:
        observations = observations_from_turn_reviews(
            [
                {
                    "feedback": {
                        "question_requires": ["Host/Client/Server 角色分工"],
                        "coach_note": "回答太泛。",
                        "gaps": ["没有区分 Host 和 Client"],
                        "covered": [],
                    },
                    "profile_signals": [],
                    "context_note_paths": ["mcp.md"],
                }
            ]
        )
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["type"], "weak_point")
        self.assertEqual(observations[0]["point"], "没有区分 Host 和 Client")
        self.assertEqual(observations[0]["evidence"], "回答太泛。")


if __name__ == "__main__":
    unittest.main()
