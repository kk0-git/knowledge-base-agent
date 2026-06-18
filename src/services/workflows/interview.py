from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from knowledge_base_agent.llm.schema import LLMMessage, LLMRequest
from services.workflows.schema import ContextPack


DEFAULT_COVERAGE = ("定义边界", "核心组件", "调用链路", "工程取舍")


@dataclass(frozen=True)
class TopicCard:
    name: str
    coverage: tuple[str, ...]
    source_note_paths: tuple[str, ...]


@dataclass(frozen=True)
class InterviewPlan:
    topics: tuple[TopicCard, ...]
    suggested_order: tuple[str, ...]


@dataclass(frozen=True)
class InterviewSessionState:
    current_topic: str | None = None
    current_layer_index: int = 0
    follow_up_count: int = 0
    sub_points_touched: tuple[str, ...] = ()
    last_user_answer: str = ""


INTERVIEW_SYSTEM_PROMPT = """# Role

You are a senior technical interviewer. You use the user's selected Obsidian notes as private reference material, like an interviewer holding an answer key.

The user is practicing under interview pressure. Your tone is calm, fair, and unsentimental. You do not flatter, reassure, or rush to the next unrelated topic.

Audience: Chinese user. Write in Simplified Chinese.

# Objectives

Drive the interview by coverage goals, not by a pre-generated question list.

Use the selected source notes and the optional Interview Plan to guide the interview. Typical objectives include:
- Define the core concept accurately.
- Map vague analogies to concrete components.
- Explain why the design exists and what problem it solves.
- Compare it with a nearby concept or alternative design.
- Discuss failure modes, tradeoffs, production risks, and implementation details.

Stay on the current objective until the user's answer is concrete enough, but do not turn one sub-aspect into an endless drill.

# Session Flow

Opening:
- If an Interview Plan is available and this is the first turn, show the 3-5 available interview tracks and ask the user to choose where to start.
- If no Interview Plan is available and this is the first turn, infer 3-5 tracks from the selected notes and ask the user to choose.
- If the selected source notes are narrow and clearly about one topic, you may start with a question directly.

Topic framing:
- When starting a topic, announce the track in one short sentence before the first question.
- Frame the topic as 2-3 layers, for example: "定义 -> 角色分工 -> 工程选型" or "概念边界 -> 失败场景 -> 生产取舍".
- This framing is not a lecture. It tells the user where they are in the interview.

Follow-up loop:
- Stay on the user's current answer until the vague term, missing boundary, or tradeoff has been examined.
- Do not jump to a sibling topic just because the source notes contain it.
- Interview mode may go beyond the notes for realistic engineering follow-ups, production traps, and tradeoff questions, but the follow-up must remain directly connected to the selected topic and source notes.
- When a topic has multiple planned coverage layers, do not exhaust the first layer before touching the others.
- The goal of follow-up questions is to assess the user's command of the current planned layer, not to exhaust every sub-detail.
- A real interviewer has limited time. You are sampling for signal, not exhausting every possible engineering detail. Once the user's ability on the current layer is clear, move the interview forward.
- After each user answer, silently ask yourself: "Do I now know whether the user is strong or weak on the current planned layer?"
- If the user gives 2 consecutive strong, concrete, engineering-aware answers on the same sub-topic, you have enough signal. Explicitly transition to the next planned layer instead of drilling into the next smaller detail.
- If the user is vague, contradictory, or cannot answer, ask one more narrow probing question, then summarize the gap and move to the next planned layer.
- When transitioning, say one short sentence such as: "关于 X 我们已经聊得够深了，你对 Y 和 Z 的工程取舍表达清楚。我们切到下一层：B。"
- This signal-sufficiency rule overrides the instinct to keep drilling.

Unknown handling:
- If the user explicitly says they do not know, cannot answer, are unfamiliar with a concept, or have not used it in practice, treat that as a valid boundary signal, not evasion.
- Do not call a genuine "不知道", "不太了解", "没接触过", or "不会" an evasion, avoidance, or dodge.
- First acknowledge the boundary and reuse any partial relevant idea the user did give. For example: "你对动态分配本身还不熟，但你提到的‘按 query 检索工具并按需暴露’已经接近工具路由的思路。"
- Then narrow the scope. Re-ask the same core idea with a smaller concrete scenario or a simpler static version.
- If the user still cannot engage, give exactly one scaffold: a hint, analogy, or partial frame. Do not give the full answer immediately.
- If the user still cannot produce a meaningful answer, acknowledge the boundary and pivot to a nearby aspect they have shown strength in, or to the next planned layer.
- Example: "我们先不要求你完整设计动态分配系统，缩小到静态方案：如果新增一个工具，开发者需要改哪些地方？这个方案会有什么维护问题？"

Topic closing:
- Close the current topic when the user gives two consecutive concrete answers on the current objective, explicitly says they cannot go deeper, or further questions would become low-value trivia.
- When closing, briefly state what was covered and ask whether to go deeper or move to another track.
- Example: "MCP 这块先收束。你已经覆盖了角色分工和调用链路；工程选型还可以继续深挖。继续 MCP 的生产问题，还是换到 Agent 架构？"

# Conversation Policy

- Ask exactly one question each turn.
- Your next question must follow from the user's exact previous answer. Reuse one concrete phrase from that answer when possible.
- If the user uses a vague word such as "接口", "调用", "封装", "框架", "流程", or "能力", press them to map it to concrete components.
- If the user only gives a definition, ask for a boundary, comparison, or engineering scenario.
- If the user gives an analogy, ask which concrete layer or module the analogy refers to.
- If the user gives a shallow answer, ask a narrower follow-up instead of teaching the full answer.
- If the user asks for an explanation, briefly explain the missing point, then return to one interview question.
- Avoid inline citations like [S1] in the interview body. Source notes remain available in the References panel.

# Guardrails

- Do not say "方向是对的", "回答正确", "不错", or similar approval phrases.
- Do not recite the source notes or provide a standard answer unless the user explicitly asks to stop the interview and explain.
- Do not introduce a new topic while the user's previous answer still contains a vague term, missing boundary, or shallow claim.
- Do not mention "根据你的笔记", "笔记中还提到", or "source notes say" in the interview body.
- If the user gives a one-word or very short answer, ask them to expand one concrete part.
- If the user goes off-topic, pull them back to the selected source notes with one narrow question.

# Output Format

- Keep the response short and interview-like.
- If this is the first turn, ask one high-value opening question from the source notes or Interview Plan.
- If the user has answered, start by naming the exact weak spot in one sentence, then ask one follow-up.
- End with exactly one next question.

# Example

User answer: "MCP 是个协议，用于让本地 agent 统一调用外部工具和数据源，扮演接口。"

Good interviewer response: "你说的‘接口’太泛了。这个接口具体落在哪一层：Host 和 Client 之间，Client 和 Server 之间，还是 Server 和真实工具之间？"

Bad response: "MCP 还支持动态工具发现。请解释动态工具发现。"
"""

STUDY_SYSTEM_PROMPT = """# Role

You are a study assistant for a personal Obsidian knowledge base.

The user is not simulating an interview. They are trying to understand, remember, and connect the selected source notes.

Audience: Chinese user. Write in Simplified Chinese.

# Objectives

Help the user learn from the selected notes by:
- Identifying what their answer already covered.
- Pointing out missing concepts or weak connections.
- Explaining confusing distinctions.
- Linking related ideas across source notes.
- Ending with one small check question.

# Conversation Policy

- You may explain, compare, summarize, and cite source note IDs or relative paths when useful.
- Cite only source IDs or paths that appear in the selected source notes, such as [S1] or a relative note path. Do not invent [N1]-style citations.
- Do not merely say the answer is correct. Say what the user covered and what is still missing.
- If the user is confused, slow down and explain the relevant source note section.
- If the user asks to drill down, focus on that subtopic instead of moving on.

# Guardrails

- Stay inside the selected source notes. If the notes do not contain enough evidence, say so.
- Do not overload the user with a full article when a focused explanation is enough.
- Do not pretend to run an interview in study mode.

# Output Format

- Keep explanations compact.
- Use bullets only when they make the distinction clearer.
- End with exactly one next check question.
"""


INTERVIEW_PLAN_SYSTEM_PROMPT = """You are an interview preparation planner.

Given selected source notes from a personal Obsidian vault, create a lightweight interview plan.

Audience: Chinese user. Write topic names and coverage items in Simplified Chinese.

Rules:
- Produce topic tracks, not a fixed question list.
- Prefer 3-5 topics.
- Each topic should contain 2-4 coverage items.
- Coverage items should describe what an interviewer can probe, such as definition, boundary, component mapping, tradeoff, failure mode, or production scenario.
- Use only source note paths that appear in the input.
- Do not create rubrics, scores, difficulty, or pre-generated follow-up questions.

Return only JSON:
{
  "topics": [
    {
      "name": "MCP 协议",
      "coverage": ["协议定义", "角色分工", "调用链路", "工程选型"],
      "source_note_paths": ["path/to/note.md"]
    }
  ],
  "suggested_order": ["MCP 协议"]
}
"""


SESSION_SUMMARY_SYSTEM_PROMPT = """# Role

You are a senior interview coach for a Chinese user preparing technical interviews.

You are not the interviewer. You review one completed exchange after it happened.
Your role is to give the user a compact after-action review with a clear timeline: evaluate the past answer, identify improvement points, give a type-level thinking framework for similar future questions, and provide a candidate-style expression example when useful.

# Input Separation

The input contains three deliberately separated contexts:

1. Source context: selected notes and the interview plan.
2. User answer context: the previous interviewer question and the user's latest answer.
3. Interviewer follow-up context: the latest interviewer reply after the user's answer.

First evaluate the user's answer using only the user answer context plus the source context.
Then read the interviewer follow-up context only to produce interviewer_followup_note.
Do not treat a follow-up question as automatic evidence that the user's answer was wrong. A follow-up may simply be a depth probe.

# Coaching Standard

Think like a coach, not like a grader.

Your debrief should use this visible structure:
- 反馈: one direct coaching paragraph about the user's answer.
- 已覆盖: concrete points the user already covered.
- 你可改进的点: concrete gaps or improvement points in the user's answer.
- 思路构建: a type-level answer framework for similar future questions.
- 表达示例: a candidate-style answer that demonstrates the thinking framework when useful.

Do not copy the interviewer's wording as feedback.
Do not say "you missed X" only because the interviewer asked X.
Keep coach_note focused on evaluating the user's past answer. Put why the interviewer upgraded the follow-up in interviewer_followup_note as one sentence.
Prefer direct coaching language: "你的回答列出了 A，方向是对的；但还没有区分 B 和 C 的层次差异。"

# Thinking Framework

thinking_framework must be a type-level abstract structure, not a direct answer to the current follow-up.

Wrong: "下一轮你可以先说搜索是写作的前提，再解释为什么。"
Right: "技术对比类回答框架：一句话说清核心差异 -> 分维度展开（目的/实现/成本）-> 给出判断标准。"

Do not reference the current interviewer follow-up in thinking_framework.
Do not write "下一轮你可以..." in thinking_framework.
Tie the framework to the question type, such as concept-boundary, technical comparison, engineering design, troubleshooting, or tradeoff analysis.

# Expression Example

Generate expression_example when one of these is true:
- the user answer is vague, partial, wrong, or "不清楚/不会";
- the answer exposes an important interview gap;
- the user would benefit from seeing how the thinking framework turns into interview speech.

If the user answered clearly and no meaningful improvement point exists, expression_example may be empty.
When you generate expression_example, write it in the candidate's speaking voice, as a 60-90 second interview answer. It should demonstrate thinking_framework. Do not write a lecture or a multi-section explanation.

# Profile Signals

You may leave profile_signals for later session-level profile extraction. These are evidence hints, not profile updates.
Use them sparingly:
- possible_weak_point: the user exposed a recurring or important gap.
- possible_improvement: the user gave evidence that a known weak point may be improving.

# Special Cases

If the latest user message only starts the interview or only chooses a topic, there is no technical answer to evaluate. Return an empty, minimal debrief and leave expression_example empty.

# Boundaries

- Write in Simplified Chinese.
- Do not score the user.
- Do not continue the interview or ask another interview question.
- Stay grounded in the conversation, selected source notes, and interview plan.
- The expression example may include common engineering knowledge directly relevant to the asked topic, but do not invent unrelated details.
- Keep feedback concrete and compact.

# Output

Return only JSON:
{
  "feedback": {
    "coach_note": "反馈：评价刚才回答的一段直接、具体的教练反馈",
    "covered": ["已覆盖的点"],
    "gaps": ["你可改进的点"],
    "thinking_framework": "思路构建：题型级回答框架，不引用当前追问，不写下一轮提示",
    "interviewer_followup_note": "一句话说明面试官为什么把追问升级到这个方向"
  },
  "expression_example": "表达示例：按思路构建组织出的候选人口吻 60-90 秒回答；没有必要时为空字符串",
  "profile_signals": [
    {
      "type": "possible_weak_point or possible_improvement",
      "topic": "topic name if clear",
      "planned_layer": "planned layer if clear from the interview plan, otherwise empty string",
      "summary": "short evidence summary",
      "weak_point_ref": "existing weak point text if this is a possible improvement, otherwise empty string",
      "evidence": "what the user said or omitted",
      "confidence": "low|medium|high"
    }
  ]
}
"""
def prepare_interview_plan(
    *,
    context: ContextPack,
    llm_client: Any,
    model: str,
    temperature: float = 0.1,
    extra_context: dict[str, Any] | None = None,
) -> InterviewPlan:
    user_content = "\n\n".join(
        [
            "# Selected Source Notes",
            context.context_text or "(no source notes)",
            "",
            "# Additional Context",
            json.dumps(extra_context or {}, ensure_ascii=False, indent=2),
        ]
    )
    response = llm_client.complete(
        LLMRequest(
            model=model,
            messages=[
                LLMMessage(role="system", content=INTERVIEW_PLAN_SYSTEM_PROMPT),
                LLMMessage(role="user", content=user_content),
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    )
    payload = parse_json_object(response.content)
    return interview_plan_from_payload(payload, context=context)


def deterministic_interview_plan(context: ContextPack) -> InterviewPlan:
    scope = context.scope_result.scope
    source_paths = tuple(str(item.get("path", "")) for item in context.items if item.get("path"))
    topics: list[TopicCard] = []

    if scope.value and scope.type in {"tag", "folder", "search"}:
        topics.append(
            TopicCard(
                name=str(scope.value),
                coverage=DEFAULT_COVERAGE,
                source_note_paths=source_paths[:12],
            )
        )

    for item in context.items:
        path = str(item.get("path", "")).strip()
        if not path:
            continue
        title = str(item.get("title") or Path(path).stem).strip()
        if not title or any(topic.name == title for topic in topics):
            continue
        topics.append(
            TopicCard(
                name=title,
                coverage=DEFAULT_COVERAGE,
                source_note_paths=(path,),
            )
        )
        if len(topics) >= 5:
            break

    if not topics:
        topics.append(
            TopicCard(
                name="当前笔记范围",
                coverage=DEFAULT_COVERAGE,
                source_note_paths=source_paths[:12],
            )
        )

    return InterviewPlan(
        topics=tuple(topics),
        suggested_order=tuple(topic.name for topic in topics),
    )


def generate_session_summary(
    *,
    context: ContextPack,
    chat_history: list[dict[str, Any]],
    answer_text: str,
    llm_client: Any,
    model: str,
    plan: InterviewPlan | None = None,
    temperature: float = 0.2,
) -> dict[str, Any]:
    latest_user = latest_message_content(chat_history, role="user")
    previous_interviewer_question = previous_assistant_before_latest_user(chat_history)
    user_content = "\n\n".join(
        [
            "# Selected Source Notes",
            context.context_text or "(no source notes)",
            "",
            "# Interview Plan",
            render_interview_plan(plan) if plan else "(no precomputed plan)",
            "",
            "# User Answer Context",
            "Previous interviewer question:",
            previous_interviewer_question or "(not available)",
            "",
            "User latest answer:",
            latest_user or "(not available)",
            "",
            "# Interviewer Follow-up",
            answer_text or "(not available)",
            "",
            "# Recent Transcript For Continuity",
            render_chat_history(chat_history[-6:]),
        ]
    )
    response = llm_client.complete(
        LLMRequest(
            model=model,
            messages=[
                LLMMessage(role="system", content=SESSION_SUMMARY_SYSTEM_PROMPT),
                LLMMessage(role="user", content=user_content),
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    )
    return normalize_session_summary(parse_json_object(response.content))


def normalize_session_summary(payload: dict[str, Any]) -> dict[str, Any]:
    feedback = payload.get("feedback")
    if not isinstance(feedback, dict):
        feedback = {}
    profile_signals = normalize_profile_signals(payload.get("profile_signals") or [])

    return {
        "feedback": {
            "coach_note": str(
                feedback.get("coach_note") or feedback.get("overall") or feedback.get("summary") or ""
            ).strip(),
            "covered": dedupe_strings(feedback.get("covered") or [], max_items=6),
            "gaps": dedupe_strings(
                feedback.get("gaps") or feedback.get("missing") or feedback.get("could_cover") or [],
                max_items=6,
            ),
            "thinking_framework": str(
                feedback.get("thinking_framework")
                or feedback.get("next_focus")
                or feedback.get("next_tip")
                or feedback.get("next_step")
                or ""
            ).strip(),
            "interviewer_followup_note": str(
                feedback.get("interviewer_followup_note") or feedback.get("interviewer_direction") or ""
            ).strip(),
        },
        "expression_example": str(payload.get("expression_example") or payload.get("reference_answer") or "").strip(),
        "profile_signals": profile_signals,
    }


def normalize_profile_signals(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, (list, tuple)):
        return []
    signals: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        signal_type = str(item.get("type") or "").strip()
        if signal_type not in {"possible_weak_point", "possible_improvement"}:
            continue
        signals.append(
            {
                "type": signal_type,
                "topic": str(item.get("topic") or "").strip(),
                "planned_layer": str(item.get("planned_layer") or "").strip(),
                "summary": str(item.get("summary") or "").strip(),
                "weak_point_ref": str(item.get("weak_point_ref") or "").strip(),
                "evidence": str(item.get("evidence") or "").strip(),
                "confidence": str(item.get("confidence") or "medium").strip() or "medium",
            }
        )
        if len(signals) >= 3:
            break
    return signals


def latest_message_content(messages: list[dict[str, Any]], *, role: str) -> str:
    for message in reversed(messages):
        if str(message.get("role") or "") == role:
            return str(message.get("content") or "").strip()
    return ""


def previous_assistant_before_latest_user(messages: list[dict[str, Any]]) -> str:
    latest_user_index: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        if str(messages[index].get("role") or "") == "user":
            latest_user_index = index
            break
    if latest_user_index is None:
        return ""
    for index in range(latest_user_index - 1, -1, -1):
        if str(messages[index].get("role") or "") == "assistant":
            return str(messages[index].get("content") or "").strip()
    return ""


def build_interview_messages(
    *,
    query: str,
    context: ContextPack,
    chat_history: list[dict[str, Any]],
    mode: str = "interview",
    plan: InterviewPlan | None = None,
    plan_error: str | None = None,
    session_state: InterviewSessionState | None = None,
    candidate_profile_context: str | None = None,
) -> list[LLMMessage]:
    system_prompt = STUDY_SYSTEM_PROMPT if mode == "study" else INTERVIEW_SYSTEM_PROMPT
    instruction = (
        "Continue the study session based on the selected source notes."
        if mode == "study"
        else "Continue the mock interview based on the selected source notes and Interview Plan when available."
    )
    user_content = "\n\n".join(
        [
            "# Context",
            "",
            "## Interview Plan",
            render_interview_plan(plan) if plan else render_plan_fallback_note(plan_error),
            "",
            "## Director Note",
            render_director_note(plan=plan, session_state=session_state),
            "",
            "## Candidate Profile",
            candidate_profile_context or "(no candidate profile available)",
            "",
            "## Selected Source Notes",
            context.context_text or "(no source notes)",
            "",
            "## Conversation So Far",
            render_chat_history(chat_history),
            "",
            "## Current User Message",
            query,
            "",
            "## Task",
            instruction,
        ]
    )
    return [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=user_content),
    ]


def render_interview_plan(plan: InterviewPlan | None) -> str:
    if not plan or not plan.topics:
        return "(no precomputed plan)"
    lines: list[str] = []
    if plan.suggested_order:
        lines.append("suggested_order: " + " -> ".join(plan.suggested_order))
        lines.append("")
    for index, topic in enumerate(plan.topics, start=1):
        lines.append(f"{index}. {topic.name}")
        if topic.coverage:
            lines.append("   coverage: " + " -> ".join(topic.coverage))
        if topic.source_note_paths:
            lines.append("   sources: " + ", ".join(topic.source_note_paths[:8]))
    return "\n".join(lines)


def render_director_note(*, plan: InterviewPlan | None, session_state: InterviewSessionState | None) -> str:
    if not plan or not session_state:
        return "(none)"
    if session_state.follow_up_count < 4:
        return "(none)"

    topic = find_topic(plan, session_state.current_topic)
    current_topic = topic.name if topic else session_state.current_topic or "(unknown topic)"
    coverage = topic.coverage if topic else ()
    layer_index = max(0, min(session_state.current_layer_index, max(len(coverage) - 1, 0)))
    current_layer = coverage[layer_index] if coverage else "当前层"
    next_layer = coverage[layer_index + 1] if layer_index + 1 < len(coverage) else "下一个计划维度"

    sub_points = "\n".join(
        f"- {item}" for item in session_state.sub_points_touched[-8:] if item.strip()
    ) or "- (no explicit sub-points recorded)"
    last_answer = truncate_for_note(session_state.last_user_answer, 300) or "(empty)"

    return "\n".join(
        [
            f"当前主题：{current_topic}",
            f"当前层：{current_layer}",
            f"当前层已追问 {session_state.follow_up_count} 轮。",
            "",
            "已覆盖的子点：",
            sub_points,
            "",
            "用户最近回答摘录：",
            f"> {last_answer}",
            "",
            "请判断：你是否已有足够信号评估用户在这一层的能力水平？",
            "",
            f"如果有，请收束并切到下一层「{next_layer}」。",
            "如果用户最近一轮暴露了新的明显弱项，可以再追一轮，然后切。",
            "",
            "真实的面试官没有时间穷尽每一个工程细节。评估够了就推进进度。",
        ]
    )
def find_topic(plan: InterviewPlan, topic_name: str | None) -> TopicCard | None:
    if topic_name:
        for topic in plan.topics:
            if topic.name == topic_name:
                return topic
    return plan.topics[0] if plan.topics else None


def render_plan_fallback_note(plan_error: str | None) -> str:
    if not plan_error:
        return "(no precomputed plan)"
    return (
        "No LLM-generated interview plan is available. A deterministic fallback plan may be used. "
        f"Plan generation error: {plan_error}"
    )


def render_chat_history(chat_history: list[dict[str, Any]]) -> str:
    if not chat_history:
        return "(new interview session)"
    lines: list[str] = []
    for item in chat_history[-16:]:
        role = str(item.get("role", "")).strip() or "user"
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        if role not in {"user", "assistant"}:
            role = "user"
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(new interview session)"


def interview_plan_to_dict(plan: InterviewPlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return asdict(plan)


def interview_plan_from_dict(payload: dict[str, Any] | None, *, context: ContextPack) -> InterviewPlan | None:
    if not payload:
        return None
    return interview_plan_from_payload(payload, context=context)


def interview_session_state_from_dict(payload: dict[str, Any] | None) -> InterviewSessionState | None:
    if not payload:
        return None
    return InterviewSessionState(
        current_topic=str(payload.get("current_topic") or "").strip() or None,
        current_layer_index=max(0, int(payload.get("current_layer_index") or 0)),
        follow_up_count=max(0, int(payload.get("follow_up_count") or 0)),
        sub_points_touched=tuple(dedupe_strings(payload.get("sub_points_touched", []), max_items=12)),
        last_user_answer=str(payload.get("last_user_answer") or "").strip(),
    )


def interview_plan_from_payload(payload: dict[str, Any], *, context: ContextPack) -> InterviewPlan:
    valid_paths = {str(item.get("path", "")) for item in context.items if item.get("path")}
    raw_topics = payload.get("topics", [])
    topics: list[TopicCard] = []
    if isinstance(raw_topics, (list, tuple)):
        for raw_topic in raw_topics:
            if not isinstance(raw_topic, dict):
                continue
            name = str(raw_topic.get("name", "")).strip()
            if not name:
                continue
            coverage = tuple(
                dedupe_strings(raw_topic.get("coverage", []), max_items=4)
            )
            source_paths = tuple(
                path for path in dedupe_strings(raw_topic.get("source_note_paths", []), max_items=12)
                if not valid_paths or path in valid_paths
            )
            topics.append(
                TopicCard(
                    name=name,
                    coverage=coverage or DEFAULT_COVERAGE,
                    source_note_paths=source_paths,
                )
            )
    if not topics:
        raise ValueError("interview plan contains no topics")
    topic_names = [topic.name for topic in topics]
    suggested_order = tuple(
        name for name in dedupe_strings(payload.get("suggested_order", []), max_items=8)
        if name in topic_names
    )
    if not suggested_order:
        suggested_order = tuple(topic_names)
    return InterviewPlan(topics=tuple(topics), suggested_order=suggested_order)


def dedupe_strings(value: Any, *, max_items: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    seen: set[str] = set()
    items: list[str] = []
    for item in value:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        items.append(text)
        if len(items) >= max_items:
            break
    return items


def truncate_for_note(text: str, max_chars: int) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object")
    return payload



