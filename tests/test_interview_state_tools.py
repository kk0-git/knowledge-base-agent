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

from agent.apps import InterviewInterviewerApp, InterviewTurnRequest
from agent.apps.interview_interviewer import build_interviewer_runtime_context, build_turn_input
from agent.interview.state import TOPIC_CLOSING, build_interview_state_machine, extract_last_question, interview_state_from_payload
from agent.llm.tool_calling import LLMToolRequest, LLMToolResponse, OpenAICompatibleToolCallingClient
from agent.runtime import AgentRuntime
from agent.schema import ToolCall, WorkingMemory
from agent.skill_loader import SkillLoader
from agent.tool_executor import ToolExecutionContext, ToolExecutor
from agent.tool_registry import ToolRegistry
from agent.tools.interview import register_interview_tools
from agent.tools.profile import register_profile_tools
from agent.tools.vault import register_vault_tools
from agent.trace import TraceRecorder
from knowledge_base_agent.llm.schema import LLMResponse
from services.rag.schema import SearchResult, TextChunk
from services.workflows.interview import InterviewPlan, TopicCard, format_interview_opening_message
from services.workflows.interview_profile import InterviewProfileStore
from services.workflows.interview_sessions import InterviewSessionStore


def sample_plan() -> InterviewPlan:
    return InterviewPlan(
        topics=(
            TopicCard(
                name="MCP ??",
                coverage=("????", "????", "????"),
                source_note_paths=("mcp.md",),
            ),
        ),
        suggested_order=("MCP ??",),
    )


def sample_plan_two_layers() -> InterviewPlan:
    return InterviewPlan(
        topics=(
            TopicCard(
                name="MCP ??",
                coverage=("????", "????"),
                source_note_paths=("mcp.md",),
            ),
        ),
        suggested_order=("MCP ??",),
    )


def active_state() -> dict[str, object]:
    machine = build_interview_state_machine(plan=sample_plan(), session_id="s1")
    selected = machine.select_topic(name="MCP ??", reason="test setup", source="test")
    return selected["state"]


class DirectFinalLLM:
    def __init__(self):
        self.requests: list[LLMToolRequest] = []

    def complete_with_tools(self, request: LLMToolRequest) -> LLMToolResponse:
        self.requests.append(request)
        return LLMToolResponse(content="????? Host?Client?Server ???????????", finish_reason="stop", used_mode="fake")


class FiveTurnDirectFinalLLM:
    def __init__(self):
        self.requests: list[LLMToolRequest] = []
        self.turn_index = 0

    def complete_with_tools(self, request: LLMToolRequest) -> LLMToolResponse:
        self.requests.append(request)
        self.turn_index += 1
        return LLMToolResponse(content=f"??{self.turn_index}?????????????", finish_reason="stop", used_mode="fake")


class AdvanceThenFinalLLM:
    def __init__(self):
        self.requests: list[LLMToolRequest] = []

    def complete_with_tools(self, request: LLMToolRequest) -> LLMToolResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return LLMToolResponse(
                tool_calls=[ToolCall(id="call_advance", name="advance_layer", arguments={"reason": "enough signal", "force": True})],
                finish_reason="tool_calls",
                used_mode="fake",
            )
        return LLMToolResponse(content="?????????Host?Client?Server ?????????", finish_reason="stop", used_mode="fake")


class SelectTopicThenFinalLLM:
    def __init__(self):
        self.requests: list[LLMToolRequest] = []

    def complete_with_tools(self, request: LLMToolRequest) -> LLMToolResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return LLMToolResponse(
                tool_calls=[ToolCall(id="call_select", name="select_topic", arguments={"name": "MCP ??", "reason": "switch topic", "source": "agent"})],
                finish_reason="tool_calls",
                used_mode="fake",
            )
        return LLMToolResponse(content="???? MCP??????????????", finish_reason="stop", used_mode="fake")


class SearchReadFinalLLM:
    def __init__(self):
        self.requests: list[LLMToolRequest] = []

    def complete_with_tools(self, request: LLMToolRequest) -> LLMToolResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return LLMToolResponse(tool_calls=[ToolCall(id="call_search", name="search_notes", arguments={"query": "MCP", "top_k": 3})], finish_reason="tool_calls", used_mode="fake")
        if len(self.requests) == 2:
            return LLMToolResponse(tool_calls=[ToolCall(id="call_read", name="read_note", arguments={"path": "mcp.md"})], finish_reason="tool_calls", used_mode="fake")
        return LLMToolResponse(content="??? MCP????????????????????", finish_reason="stop", used_mode="fake")


class RecallProfileFinalLLM:
    def __init__(self):
        self.requests: list[LLMToolRequest] = []

    def complete_with_tools(self, request: LLMToolRequest) -> LLMToolResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return LLMToolResponse(tool_calls=[ToolCall(id="call_recall", name="recall_profile", arguments={"topic": "MCP ??", "planned_layer": "????"})], finish_reason="tool_calls", used_mode="fake")
        return LLMToolResponse(content="???????????????????? Host ? Server ????????", finish_reason="stop", used_mode="fake")


class JsonAdvanceBaseClient:
    def __init__(self):
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(content='{"tool_calls":[{"id":"call_advance","name":"advance_layer","arguments":{"reason":"json fallback advance","force":true}}]}', raw={"fake": True})
        return LLMResponse(content='{"final":"fallback interviewer final?"}', raw={"fake": True})


class FakeRAGManager:
    def hybrid_search(self, *, query: str, top_k: int, dense_top_k: int, bm25_top_k: int, rrf_k: int):
        return [
            SearchResult(
                chunk=TextChunk(
                    chunk_id="mcp.md#1",
                    note_path="mcp.md",
                    heading_path=["MCP"],
                    text="MCP Host Client Server",
                    start_line=1,
                    end_line=4,
                ),
                score=0.9,
            )
        ]


class InterviewOpeningMessageTests(unittest.TestCase):
    def test_format_interview_opening_message_lists_topics_with_coverage(self) -> None:
        plan = InterviewPlan(
            topics=(
                TopicCard(name="MCP 协议", coverage=("协议定义", "调用链路"), source_note_paths=("mcp.md",)),
                TopicCard(name="RAG 系统", coverage=("检索策略", "重排序"), source_note_paths=("rag.md",)),
            ),
            suggested_order=("MCP 协议", "RAG 系统"),
        )
        text = format_interview_opening_message(plan)
        self.assertIn("好的，我们开始面试。", text)
        self.assertIn("**MCP 协议**：协议定义、调用链路。", text)
        self.assertIn("你想从哪个方向开始？", text)


class InterviewStateMachineTests(unittest.TestCase):
    def test_extract_last_question_supports_fullwidth_halfwidth_and_fallback(self) -> None:
        self.assertEqual(extract_last_question("你能解释一下吗？"), "你能解释一下吗？")
        self.assertEqual(extract_last_question("What is MCP?"), "What is MCP?")
        self.assertEqual(extract_last_question("第一问? 第二问？"), "第二问？")
        fallback = extract_last_question("请你区分 Host、Client、Server 三者职责")
        self.assertEqual(fallback, "请你区分 Host、Client、Server 三者职责")
        long_text = "a" * 200
        self.assertTrue(extract_last_question(long_text).endswith("..."))
        self.assertLess(len(extract_last_question(long_text)), len(long_text))
        self.assertEqual(extract_last_question(""), "")

    def test_initialize_awaits_topic_selection_and_select_topic(self) -> None:
        machine = build_interview_state_machine(plan=sample_plan(), session_id="s1")
        snapshot = machine.snapshot()
        self.assertIsNone(snapshot["current_topic"])
        self.assertEqual(snapshot["topic_phase"], "awaiting_selection")
        self.assertEqual(snapshot["current_layer_name"], "")
        selected = machine.select_topic(name="MCP ??", reason="user picked", source="ui")
        self.assertTrue(selected["selected"])
        self.assertEqual(machine.snapshot()["current_topic"], "MCP ??")
        self.assertEqual(machine.snapshot()["topic_phase"], "active")
        self.assertEqual(machine.snapshot()["current_layer_name"], "????")
        machine.commit_turn(user_text="answer", assistant_text="你能解释这一层吗？")
        self.assertEqual(machine.snapshot()["follow_up_count"], 1)
        self.assertEqual(machine.snapshot()["last_assistant_question"], "你能解释这一层吗？")
        self.assertEqual(machine.snapshot()["sub_points_touched"], ["你能解释这一层吗？"])

    def test_commit_turn_tracks_assistant_anchor_and_dedupes_sub_points(self) -> None:
        machine = build_interview_state_machine(plan=sample_plan(), session_id="s1")
        machine.select_topic(name="MCP ??", reason="start", source="test")
        machine.commit_turn(user_text="answer 1", assistant_text="第一问？")
        machine.commit_turn(user_text="answer 2", assistant_text="第一问？")
        self.assertEqual(machine.snapshot()["follow_up_count"], 2)
        self.assertEqual(machine.snapshot()["sub_points_touched"], ["第一问？"])
        machine.commit_turn(user_text="answer 3", assistant_text="请你区分 Host、Client、Server 三者职责")
        snapshot = machine.snapshot()
        self.assertEqual(snapshot["last_assistant_question"], "请你区分 Host、Client、Server 三者职责")
        self.assertEqual(snapshot["sub_points_touched"][-1], snapshot["last_assistant_question"])

    def test_commit_turn_does_not_increment_when_topic_not_active(self) -> None:
        machine = build_interview_state_machine(plan=sample_plan(), session_id="s1")
        machine.commit_turn(user_text="answer", assistant_text="还没选 topic 吗？")
        self.assertEqual(machine.snapshot()["topic_phase"], "awaiting_selection")
        self.assertEqual(machine.snapshot()["follow_up_count"], 0)

    def test_legacy_payload_and_transition_threshold(self) -> None:
        state = interview_state_from_payload({"current_topic": "MCP ??", "current_layer_index": 1, "follow_up_count": 4}, plan=sample_plan())
        machine = build_interview_state_machine(plan=sample_plan(), state_payload=state.to_dict())
        self.assertEqual(machine.snapshot()["source"], "server")
        self.assertEqual(machine.snapshot()["topic_phase"], "active")
        self.assertTrue(machine.snapshot()["should_consider_layer_transition"])

    def test_advance_to_last_layer_sets_closing(self) -> None:
        machine = build_interview_state_machine(plan=sample_plan_two_layers(), session_id="s1")
        machine.select_topic(name="MCP ??", reason="start", source="test")
        self.assertEqual(machine.snapshot()["topic_phase"], "active")
        advanced = machine.advance_layer(reason="move to last layer", force=True)
        self.assertTrue(advanced["advanced"])
        self.assertEqual(machine.snapshot()["current_layer_index"], 1)
        self.assertEqual(machine.snapshot()["topic_phase"], TOPIC_CLOSING)
        self.assertTrue(machine.snapshot()["at_last_layer"])


class InterviewSessionNormalizationTests(unittest.TestCase):
    def test_create_session_defaults_awaiting_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = InterviewSessionStore(Path(tmp))
            session = store.create_session(source_type="folder", source_value="agent")
            state = session["interview_state"]
            self.assertIsNotNone(state)
            self.assertEqual(state["source"], "server")
            self.assertEqual(state["topic_phase"], "awaiting_selection")
            self.assertIsNone(state["current_topic"])

    def test_load_session_normalizes_legacy_state_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = InterviewSessionStore(root)
            session = store.create_session(
                source_type="folder",
                source_value="legacy",
                extra={"created_from": "chat"},
            )
            path = store.session_path(session["session_id"])
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["interview_state"] = {
                "current_topic": "MCP ??",
                "current_layer_index": 0,
                "follow_up_count": 1,
            }
            path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
            loaded = store.load_session(session["session_id"])
            state = loaded["interview_state"]
            self.assertEqual(state["source"], "server")
            self.assertEqual(state["topic_phase"], "active")
            self.assertEqual(state["current_topic"], "MCP ??")


class InterviewToolTests(unittest.TestCase):
    def test_get_list_select_and_advance_tools_update_working(self) -> None:
        registry = ToolRegistry()
        register_interview_tools(registry)
        working = WorkingMemory()
        ctx = ToolExecutionContext(
            working=working,
            confirmed_tools={"advance_layer", "select_topic"},
            interview_plan=sample_plan(),
            state_machine=build_interview_state_machine(plan=sample_plan()),
        )
        executor = ToolExecutor(registry, ctx)
        state = executor.execute(ToolCall(id="1", name="get_interview_state", arguments={}))
        self.assertTrue(state.ok)
        self.assertEqual(state.output["topic_phase"], "awaiting_selection")
        topics = executor.execute(ToolCall(id="2", name="list_plan_topics", arguments={"include_sources": True}))
        self.assertEqual(topics.output["topics"][0]["source_note_paths"], ["mcp.md"])
        selected = executor.execute(ToolCall(id="3", name="select_topic", arguments={"name": "MCP ??", "reason": "test"}))
        self.assertTrue(selected.output["selected"])
        blocked = executor.execute(ToolCall(id="4", name="advance_layer", arguments={"reason": "too early"}))
        self.assertFalse(blocked.output["advanced"])
        advanced = executor.execute(ToolCall(id="5", name="advance_layer", arguments={"reason": "force", "force": True}))
        self.assertTrue(advanced.output["advanced"])
        self.assertEqual(working.current_layer_index, 1)


class InterviewerRuntimeTests(unittest.TestCase):
    def make_runtime(self, llm, tmp: str, registry: ToolRegistry | None = None) -> AgentRuntime:
        registry = registry or ToolRegistry()
        if not registry.has("get_interview_state"):
            register_interview_tools(registry)
        if not registry.has("search_notes"):
            register_vault_tools(registry)
        if not registry.has("recall_profile"):
            register_profile_tools(registry)
        return AgentRuntime(
            llm_client=llm,
            skill_loader=SkillLoader(PROJECT_ROOT / "skills", registry=registry),
            tool_registry=registry,
            trace_recorder=TraceRecorder(tmp),
        )

    def test_runtime_context_injected_and_routine_final_does_not_fetch_state(self) -> None:
        registry = ToolRegistry()
        register_interview_tools(registry)
        register_vault_tools(registry)
        register_profile_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            llm = DirectFinalLLM()
            app = InterviewInterviewerApp(self.make_runtime(llm, tmp, registry))
            request = InterviewTurnRequest(
                query="MCP ???",
                interview_plan=sample_plan(),
                interview_state=active_state(),
                model="fake",
                trace_path=tmp,
            )
            result, _ = app.run_turn(request)
            self.assertEqual(result.stopped_reason, "final")
            self.assertEqual(result.steps[0].tool_calls, [])
            self.assertIn("# Runtime Context", llm.requests[0].messages[-1].content)
            self.assertNotIn("MCP Host Client Server", llm.requests[0].messages[-1].content)
            tool_names = [schema["function"]["name"] for schema in llm.requests[0].tools]
            self.assertEqual(tool_names, ["advance_layer", "get_interview_state", "grep_vault", "list_plan_topics", "read_note", "recall_profile", "search_notes", "select_topic"])
            payload = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["runtime_context"]["topic_phase"], "active")
            self.assertEqual(payload["derived_metrics"]["routine_state_fetch"], 0)

    def test_build_turn_input_contains_compact_context(self) -> None:
        request = InterviewTurnRequest(query="hello", interview_plan=sample_plan(), interview_state=active_state(), scope_note_paths=("mcp.md",))
        machine = build_interview_state_machine(plan=sample_plan(), state_payload=active_state())
        context = build_interviewer_runtime_context(request, machine, None)
        text = build_turn_input(request, runtime_context=context)
        self.assertIn('"interview_mode": "mock"', text)
        self.assertIn('"allowed_note_count": 1', text)
        self.assertNotIn("profile_context", text)
        self.assertIn("next_layer_name", context)
        self.assertIn("at_last_layer", context)
        self.assertIn("last_assistant_question", context)

    def test_runtime_context_preloads_universal_and_domain_layer_counts(self) -> None:
        plan = InterviewPlan(
            topics=(TopicCard(name="MCP", coverage=("definition", "roles"), source_note_paths=("mcp.md",)),),
            suggested_order=("MCP",),
        )
        machine = build_interview_state_machine(plan=plan, session_id="s1")
        active = machine.select_topic(name="MCP", reason="test", source="test")["state"]
        with tempfile.TemporaryDirectory() as tmp:
            profile_store = InterviewProfileStore(Path(tmp) / "profile.json")
            profile_store.save(
                {
                    "schema_version": 3,
                    "weak_points": [
                        {
                            "point": "Universal answer structure habit",
                            "topic": "",
                            "scope": "universal",
                            "category": "answer_structure",
                            "confidence": "high",
                            "improved": False,
                            "sr": {"next_review": "2000-01-01", "ease_factor": 2.3},
                        },
                        {
                            "point": "Domain definition gap",
                            "topic": "MCP",
                            "scope": "domain",
                            "category": "knowledge_gap",
                            "planned_layer": "definition",
                            "confidence": "high",
                            "improved": False,
                            "domain_anchor": {"plan_topic": "MCP", "context_note_paths": ["mcp.md"], "scope_path": ""},
                        },
                        {
                            "point": "Domain roles gap",
                            "topic": "MCP",
                            "scope": "domain",
                            "category": "knowledge_gap",
                            "planned_layer": "roles",
                            "confidence": "medium",
                            "improved": False,
                            "domain_anchor": {"plan_topic": "MCP", "context_note_paths": ["mcp.md"], "scope_path": ""},
                        },
                    ],
                    "strong_points": [],
                    "topic_mastery": {},
                    "communication": {"style": "", "suggestions": []},
                }
            )
            request = InterviewTurnRequest(query="hello", interview_plan=plan, interview_state=active, profile_store=profile_store)
            context = build_interviewer_runtime_context(request, machine, profile_store)
            profile = context["profile"]
            self.assertTrue(profile["profile_available"])
            self.assertEqual(profile["universal_weak_points"][0]["point"], "Universal answer structure habit")
            self.assertEqual(profile["domain_weak_by_layer"]["definition"], 1)
            self.assertEqual(profile["domain_weak_by_layer"]["roles"], 1)
            self.assertEqual(profile["current_layer_domain_weak_count"], 1)
            rendered = json.dumps(profile, ensure_ascii=False)
            self.assertNotIn("Domain definition gap", rendered)
            self.assertNotIn("Domain roles gap", rendered)

    def test_five_turn_mock_session_routine_state_fetch_zero(self) -> None:
        registry = ToolRegistry()
        register_interview_tools(registry)
        register_vault_tools(registry)
        register_profile_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            llm = FiveTurnDirectFinalLLM()
            app = InterviewInterviewerApp(self.make_runtime(llm, tmp, registry))
            state = active_state()
            total_routine = 0
            for turn in range(5):
                result, machine = app.run_turn(
                    InterviewTurnRequest(
                        query=f"answer {turn}",
                        interview_plan=sample_plan(),
                        interview_state=state,
                        model="fake",
                        trace_path=tmp,
                    )
                )
                state = machine.snapshot()
                payload = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
                total_routine += int(payload["derived_metrics"]["routine_state_fetch"])
            self.assertEqual(total_routine, 0)
            self.assertEqual(llm.turn_index, 5)

    def test_interviewer_advance_trace_and_working_memory(self) -> None:
        registry = ToolRegistry()
        register_interview_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            app = InterviewInterviewerApp(self.make_runtime(AdvanceThenFinalLLM(), tmp, registry))
            result, machine = app.run_turn(InterviewTurnRequest(query="??", interview_plan=sample_plan(), interview_state=active_state(), model="fake", trace_path=tmp))
            self.assertEqual(result.stopped_reason, "final")
            self.assertEqual(machine.snapshot()["current_layer_index"], 1)
            self.assertEqual(result.state.working.current_layer_index, 1)
            payload = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
            self.assertTrue(payload["derived_metrics"]["layer_advanced"])
            self.assertEqual(payload["steps"][0]["tool_calls"][0]["name"], "advance_layer")

    def test_interviewer_can_select_topic_action(self) -> None:
        registry = ToolRegistry()
        register_interview_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            app = InterviewInterviewerApp(self.make_runtime(SelectTopicThenFinalLLM(), tmp, registry))
            result, machine = app.run_turn(InterviewTurnRequest(query="? topic", interview_plan=sample_plan(), model="fake", trace_path=tmp))
            self.assertEqual(result.stopped_reason, "final")
            self.assertEqual(machine.snapshot()["current_topic"], "MCP ??")
            payload = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
            self.assertTrue(payload["derived_metrics"]["topic_selected"])

    def test_interviewer_search_read_final(self) -> None:
        registry = ToolRegistry()
        register_interview_tools(registry)
        register_vault_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mcp.md").write_text("# MCP\nHost Client Server", encoding="utf-8")
            app = InterviewInterviewerApp(self.make_runtime(SearchReadFinalLLM(), tmp, registry))
            result, _ = app.run_turn(
                InterviewTurnRequest(
                    query="?????? MCP",
                    interview_plan=sample_plan(),
                    interview_state=active_state(),
                    vault_root=root,
                    rag_manager=FakeRAGManager(),
                    scope_note_paths=("mcp.md",),
                    model="fake",
                    trace_path=tmp,
                )
            )
            self.assertEqual(result.stopped_reason, "final")
            self.assertEqual(result.state.working.notes_read_this_turn, ["mcp.md"])

    def test_interviewer_can_recall_profile(self) -> None:
        registry = ToolRegistry()
        register_interview_tools(registry)
        register_vault_tools(registry)
        register_profile_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            profile_store = InterviewProfileStore(Path(tmp) / "profile.json")
            profile_store.save(
                {
                    "schema_version": 2,
                    "weak_points": [
                        {
                            "point": "MCP ??????",
                            "topic": "MCP ??",
                            "scope": "domain",
                            "category": "knowledge_gap",
                            "improved": False,
                            "domain_anchor": {"plan_topic": "MCP ??", "context_note_paths": ["mcp.md"], "scope_path": ""},
                        }
                    ],
                    "strong_points": [],
                    "topic_mastery": {},
                    "communication": {"style": "", "suggestions": []},
                }
            )
            app = InterviewInterviewerApp(self.make_runtime(RecallProfileFinalLLM(), tmp, registry))
            result, _ = app.run_turn(
                InterviewTurnRequest(
                    query="MCP ???????",
                    interview_plan=sample_plan(),
                    interview_state=active_state(),
                    profile_store=profile_store,
                    model="fake",
                    trace_path=tmp,
                )
            )
            self.assertEqual(result.stopped_reason, "final")
            self.assertEqual(result.steps[0].tool_calls[0].name, "recall_profile")

    def test_recall_profile_filters_domain_weak_points_by_planned_layer(self) -> None:
        registry = ToolRegistry()
        register_profile_tools(registry)
        plan = InterviewPlan(
            topics=(TopicCard(name="MCP", coverage=("definition", "roles"), source_note_paths=("mcp.md",)),),
            suggested_order=("MCP",),
        )
        with tempfile.TemporaryDirectory() as tmp:
            profile_store = InterviewProfileStore(Path(tmp) / "profile.json")
            profile_store.save(
                {
                    "schema_version": 3,
                    "weak_points": [
                        {
                            "point": "Universal habit",
                            "scope": "universal",
                            "category": "answer_structure",
                            "improved": False,
                            "confidence": "high",
                        },
                        {
                            "point": "Definition layer gap",
                            "topic": "MCP",
                            "scope": "domain",
                            "category": "knowledge_gap",
                            "planned_layer": "definition",
                            "improved": False,
                            "domain_anchor": {"plan_topic": "MCP", "context_note_paths": ["mcp.md"], "scope_path": ""},
                        },
                        {
                            "point": "Roles layer gap",
                            "topic": "MCP",
                            "scope": "domain",
                            "category": "knowledge_gap",
                            "planned_layer": "roles",
                            "improved": False,
                            "domain_anchor": {"plan_topic": "MCP", "context_note_paths": ["mcp.md"], "scope_path": ""},
                        },
                    ],
                    "strong_points": [],
                    "topic_mastery": {},
                    "communication": {"style": "", "suggestions": []},
                }
            )
            ctx = ToolExecutionContext(
                working=WorkingMemory(session_id="s1", current_topic="MCP"),
                profile_store=profile_store,
                interview_plan=plan,
                interview_state={"current_topic": "MCP", "current_layer_name": "definition"},
            )
            executor = ToolExecutor(registry, ctx)
            recall = executor.execute(ToolCall(id="1", name="recall_profile", arguments={"topic": "MCP", "planned_layer": "definition"}))
            self.assertTrue(recall.ok)
            self.assertEqual([item["point"] for item in recall.output["weak_points"]], ["Definition layer gap"])
            self.assertEqual(recall.output["counts"]["domain_weak_by_layer"]["definition"], 1)
            self.assertEqual(recall.output["counts"]["domain_weak_by_layer"]["roles"], 1)

    def test_json_fallback_can_advance_layer(self) -> None:
        registry = ToolRegistry()
        register_interview_tools(registry)
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(OpenAICompatibleToolCallingClient(JsonAdvanceBaseClient()), tmp, registry)
            app = InterviewInterviewerApp(runtime)
            result, machine = app.run_turn(
                InterviewTurnRequest(
                    query="json fallback",
                    interview_plan=sample_plan(),
                    interview_state=active_state(),
                    model="fake",
                    tool_mode="json",
                    trace_path=tmp,
                )
            )
            self.assertEqual(result.final_answer, "fallback interviewer final?")
            self.assertEqual(machine.snapshot()["current_layer_index"], 1)

    def test_session_store_records_agent_trace_and_server_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = InterviewSessionStore(Path(tmp))
            session = store.create_session(source_type="folder", source_value="agent")
            updated = store.record_agent_turn(
                session_id=session["session_id"],
                skill="interviewer",
                trace_path="traces/turn.json",
                trace_id="agent-test",
                interview_state={"source": "server", "current_topic": "MCP ??"},
            )
            self.assertEqual(updated["interview_state"]["source"], "server")
            self.assertEqual(updated["agent"]["skill"], "interviewer")
            self.assertEqual(updated["agent"]["traces"][0]["trace_id"], "agent-test")
            trace = store.load_session(session["session_id"])["trace"]
            self.assertEqual(trace[-1]["details"]["state_phase"], "post_agent_commit")


if __name__ == "__main__":
    unittest.main()
