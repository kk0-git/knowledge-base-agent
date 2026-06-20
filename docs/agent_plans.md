# 路线 A：Agent-native Interview Coach 实施计划

> **目标**：将 Knowledge Agent 重定义为「Obsidian 上的 Agent Runtime」，以 Interview Coach 为 flagship agent；现有 RAG / Wiki / Organize 降级为 Tools，而非并列 Workflow。
>
> **原则**：先自建轻量 `AgentRuntime`（不引入 LangGraph），Interview 一条链 agent 化，再 MCP 暴露、Agent Eval 闭环。
>
> 相关文档：[INTERVIEW_PROFILE_DESIGN.md](./INTERVIEW_PROFILE_DESIGN.md)
>
> **当前执行入口**：Phase 0–1 已完成 → 按 [§15 落地路线图（已定边界版）](#15-落地路线图已定边界版) 推进。

---

## 1. 背景与现状

### 1.1 当前 Interview 管线（Pipeline）

```text
POST /api/agent/stream  (chat_mode=interview)
  │
  ├─ ScopeResolver.resolve()
  ├─ ContextBuilder.build(all notes)          ← 整包预灌 8 篇 note
  ├─ prepare_interview_plan() / client plan
  ├─ render_candidate_profile_context()       ← profile 整段注入 prompt
  ├─ build_interview_messages()
  └─ llm_client.stream_complete()             ← 单次生成，无 tool loop

POST /api/interview/summary                   ← turn review（独立 LLM）
POST /api/interview/sessions/{id}/end         ← profile extraction（批处理 ETL）
```

**问题摘要**：

| 问题 | 根因 |
|------|------|
| context 过宽 | `ContextBuilder` 预灌全 scope notes |
| profile anchor 断裂 | `context_note_paths` 非 turn 级真实引用 |
| session state 不可靠 | 前端 heuristic，非服务端 state machine |
| 无 Agent 作品集叙事 | 无 tool loop / trace / skill boundary |

### 1.2 目标 Interview 管线（Agent Loop）

```text
POST /api/agent/stream  (chat_mode=interview, agent_v2=true)
  │
  ├─ 加载 Skill: interviewer
  ├─ 初始化 AgentState（session + plan + working memory）
  ├─ AgentRuntime.run_stream(max_steps=6)
  │     loop:
  │       LLM → tool_calls?
  │         ├─ search_notes / read_note / recall_profile
  │         ├─ get_state / advance_layer / record_signal
  │         └─ observations 回灌 → 下一步
  │       LLM → final user-visible answer（恰好一个问题）
  ├─ 持久化 agent_trace + 更新 interview_state（服务端）
  └─ SSE: agent_step / answer_delta / done

POST /api/interview/summary  → Coach Agent（skill=coach, max_steps=2）
POST /api/interview/sessions/{id}/end  → Curator Agent 或保留现有 extraction + profile tools 审计
```

---

## 2. 目标目录结构

在现有 `src/` 上 **新增 `src/agent/`**，不推翻 `services/`；Interview 相关逐步从 `workflows/interview.py` 拆 prompt 到 skills，逻辑迁入 agent apps。

```text
knowledge_agent/
├── docs/
│   ├── agent_plans.md                 ← 本文档
│   └── INTERVIEW_PROFILE_DESIGN.md
├── skills/                            ← 产品 Skill（非 Cursor Skill）
│   ├── interviewer/
│   │   ├── SKILL.md                   ← 行为指令（从 INTERVIEW_SYSTEM_PROMPT 迁移）
│   │   └── manifest.yaml              ← allowed_tools, max_steps, temperature
│   ├── coach/
│   │   ├── SKILL.md                   ← turn debrief（从 SESSION_SUMMARY 迁移）
│   │   └── manifest.yaml
│   ├── curator/
│   │   ├── SKILL.md                   ← session 结束 profile extraction
│   │   └── manifest.yaml
│   └── librarian/
│       ├── SKILL.md                   ← 通用笔记检索助手（answer 模式后续用）
│       └── manifest.yaml
├── eval/
│   ├── rag_eval.json                  ← 已有
│   └── agent_eval/                    ← 新增
│       ├── interview_golden.json
│       └── rubric.md
├── src/
│   ├── agent/                         ← ★ 新增：Agent Runtime 核心
│   │   ├── __init__.py
│   │   ├── schema.py                  ← AgentState, ToolCall, AgentStep, AgentResult
│   │   ├── runtime.py                 ← AgentRuntime（主 loop）
│   │   ├── skill_loader.py            ← 加载 skills/*/manifest.yaml + SKILL.md
│   │   ├── tool_registry.py           ← Tool 注册、schema、权限
│   │   ├── tool_executor.py           ← 执行、timeout、错误 → observation
│   │   ├── memory/
│   │   │   ├── __init__.py
│   │   │   ├── working.py             ← WorkingMemory（当前 topic/layer/turn facts）
│   │   │   ├── episodic.py            ← Session 事件 log（与 interview_sessions 对齐）
│   │   │   └── semantic.py            ← Profile recall 适配层
│   │   ├── trace/
│   │   │   ├── __init__.py
│   │   │   ├── recorder.py            ← JSON trace 写入
│   │   │   └── serialize.py
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   └── tool_calling.py        ← 扩展 LLMClient：支持 tools + tool_calls 解析
│   │   ├── tools/                     ← Tool 实现（薄封装，复用 services/）
│   │   │   ├── __init__.py
│   │   │   ├── vault/
│   │   │   │   ├── search_notes.py    ← 包装 RAGManager.hybrid_search
│   │   │   │   ├── grep_vault.py      ← 包装 grep_search
│   │   │   │   └── read_note.py       ← 包装 ContextBuilder 单 note 读取
│   │   │   ├── interview/
│   │   │   │   ├── get_state.py
│   │   │   │   ├── advance_layer.py
│   │   │   │   ├── record_signal.py
│   │   │   │   └── list_plan_topics.py
│   │   │   └── profile/
│   │   │       ├── recall_profile.py  ← 包装 render 逻辑 → 结构化返回
│   │   │       └── write_observation_draft.py  ← session 内草稿，end 时 commit
│   │   └── apps/
│   │       ├── __init__.py
│   │       ├── interview_interviewer.py   ← 组装 interviewer turn
│   │       ├── interview_coach.py         ← 组装 coach turn review
│   │       └── interview_curator.py       ← session end profile
│   ├── mcp/                           ← ★ 新增：MCP Server（P2）
│   │   ├── __init__.py
│   │   ├── server.py                  ← stdio / sse server 入口
│   │   └── tool_adapter.py            ← 复用 agent/tools 注册表
│   ├── knowledge_base_agent/          ← 保留：vault 解析、LLM client
│   ├── services/                      ← 保留：RAG、wiki、workflows（逐步变薄）
│   │   ├── rag/                       ← Tools 底层实现
│   │   ├── workflows/
│   │   │   ├── interview.py           ← 逐步 deprecated；prompt 迁到 skills/
│   │   │   ├── interview_profile.py   ← semantic memory 底层，由 profile tools 调用
│   │   │   └── interview_sessions.py  ← episodic 持久化
│   │   └── ...
│   └── web/
│       └── app.py                     ← 新增 agent_v2 分支；旧 pipeline feature flag 保留
└── scripts/
    ├── agent_debug.py                 ← CLI：单步跑 agent turn
    └── agent_eval.py                  ← golden session 回归
```

### 2.1 与现有模块的映射

| 现有 | 路线 A 归宿 |
|------|-------------|
| `services/rag/agent_answer.py` | Phase 4 迁为 `librarian` skill + ReAct loop |
| `services/rag/intent_router.py` | 被 tool loop 取代；router 逻辑可作 `search_notes` 内部策略 |
| `services/workflows/interview.py` | Prompt → `skills/interviewer/`；plan/state 函数 → `agent/tools/interview/` |
| `services/workflows/interview_profile.py` | 底层 store 保留；注入改为 `recall_profile` tool |
| `services/workflows/context_builder.py` | 保留；`read_note` / 按需检索替代全量 build |
| `services/workflows/interview_sessions.py` | 保留；增加 `agent_trace` 字段写入 |
| `web/app.py` `stream_study_or_interview` | 薄封装 → 调用 `InterviewInterviewerApp` |

---

## 3. AgentRuntime 接口设计

### 3.1 核心类型（`src/agent/schema.py`）

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal


class StepKind(str, Enum):
    LLM = "llm"
    TOOL = "tool"
    FINAL = "final"
    ERROR = "error"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]          # JSON Schema
    handler: Callable[..., Any]         # 运行时注册，不序列化
    timeout_s: float = 30.0
    requires_confirmation: bool = False


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    call_id: str
    name: str
    ok: bool
    output: Any                         # JSON-serializable
    error: str = ""
    latency_ms: int = 0


@dataclass
class AgentMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class WorkingMemory:
    """当前 turn 的可变事实；每轮 interviewer 开始时注入 state snapshot。"""
    session_id: str | None = None
    current_topic: str | None = None
    current_layer_index: int = 0
    follow_up_count: int = 0
    plan_topic_names: list[str] = field(default_factory=list)
    notes_read_this_turn: list[str] = field(default_factory=list)
    signals_this_turn: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    messages: list[AgentMessage]
    working: WorkingMemory
    skill_name: str
    step_index: int = 0
    finished: bool = False
    final_answer: str = ""


@dataclass
class AgentStep:
    index: int
    kind: StepKind
    llm_input_chars: int = 0
    llm_output_chars: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    assistant_text: str = ""
    latency_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRunConfig:
    skill_name: str
    max_steps: int = 8
    max_tool_calls_per_step: int = 4
    temperature: float = 0.2
    model: str = ""
    stream_final: bool = True           # 最后一步是否流式输出给用户
    save_trace: bool = True
    trace_path: str | None = None


@dataclass
class AgentResult:
    state: AgentState
    steps: list[AgentStep]
    final_answer: str
    total_ms: int
    stopped_reason: Literal["final", "max_steps", "error", "cancelled"]
    trace_id: str = ""
```

### 3.2 Skill 定义（`skills/interviewer/manifest.yaml`）

```yaml
name: interviewer
version: 1
description: Mock technical interviewer for Obsidian note scope
temperature: 0.3
max_steps: 6
allowed_tools:
  - list_plan_topics
  - search_notes
  - read_note
  - recall_profile
  - get_interview_state
  - advance_layer
  - record_signal
denied_tools:
  - write_observation_draft   # interviewer 只 record_signal，不直接写 profile
output_contract:
  type: interview_question
  rules:
    - exactly_one_question
    - simplified_chinese
    - no_approval_phrases
```

```yaml
# skills/coach/manifest.yaml
name: coach
version: 1
max_steps: 2
allowed_tools:
  - read_note
  - recall_profile
  - record_signal
output_contract:
  type: turn_review_json
```

### 3.3 SkillLoader（`src/agent/skill_loader.py`）

```python
@dataclass(frozen=True)
class LoadedSkill:
    name: str
    system_prompt: str           # SKILL.md 正文
    allowed_tools: frozenset[str]
    max_steps: int
    temperature: float
    output_contract: dict[str, Any]


class SkillLoader:
    def __init__(self, skills_root: Path): ...

    def load(self, name: str) -> LoadedSkill: ...
    def list_skills(self) -> list[str]: ...
```

### 3.4 ToolRegistry（`src/agent/tool_registry.py`）

```python
class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None: ...

    def get(self, name: str) -> ToolSpec: ...

    def schemas_for(self, names: Iterable[str]) -> list[dict[str, Any]]:
        """返回 OpenAI function calling 格式的 tools 列表。"""

    def subset(self, allowed: Iterable[str]) -> ToolRegistry:
        """按 Skill manifest 过滤。"""
```

### 3.5 ToolExecutor（`src/agent/tool_executor.py`）

```python
class ToolExecutionContext:
    vault_root: Path
    rag_manager: Any | None
    session_store: Any | None
    profile_store: Any | None
    working: WorkingMemory
    scope_note_paths: tuple[str, ...]


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, ctx: ToolExecutionContext): ...

    def execute(self, call: ToolCall) -> ToolResult:
        """
        - JSON schema 校验
        - timeout 包裹
        - 异常 → ToolResult(ok=False)
        - 成功 → 更新 WorkingMemory（如 notes_read_this_turn）
        """
```

### 3.6 AgentRuntime 主接口（`src/agent/runtime.py`）

```python
class AgentRuntime:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        skill_loader: SkillLoader,
        tool_registry: ToolRegistry,
        trace_recorder: TraceRecorder | None = None,
    ): ...

    def run(self, *, config: AgentRunConfig, user_input: str, state: AgentState | None = None) -> AgentResult:
        """同步跑完整个 loop（CLI / 测试用）。"""

    def run_stream(
        self,
        *,
        config: AgentRunConfig,
        user_input: str,
        state: AgentState | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """
        流式事件，供 SSE 消费。

        AgentStreamEvent 类型：
          - {"type": "step_started", "index": N}
          - {"type": "tool_call", "call": {...}}
          - {"type": "tool_result", "result": {...}}
          - {"type": "answer_delta", "text": "..."}   # 仅 final step
          - {"type": "state_updated", "working": {...}}
          - {"type": "done", "result": AgentResult}
        """
```

### 3.7 LLM Tool Calling 扩展（`src/agent/llm/tool_calling.py`）

现有 `LLMClient` 只有 `complete` / `stream_complete`，需扩展 **不破坏旧接口**：

```python
@dataclass(frozen=True)
class LLMToolRequest:
    model: str
    messages: list[AgentMessage]
    tools: list[dict[str, Any]]
    temperature: float = 0.2
    tool_choice: str | dict = "auto"    # "auto" | "none" | {"type":"function","function":{"name":"..."}}


@dataclass(frozen=True)
class LLMToolResponse:
    content: str
    tool_calls: list[ToolCall]
    finish_reason: str
    raw: dict[str, Any]


class ToolCallingLLMClient(Protocol):
    def complete_with_tools(self, request: LLMToolRequest) -> LLMToolResponse: ...
    def stream_final_with_tools(self, request: LLMToolRequest) -> Iterator[str | LLMToolResponse]:
        """
        多步：非 final 用 complete_with_tools；
        最后一步若只需文本输出，可 tool_choice=none 并 stream content。
        """
```

**实现策略**：在 `openai_compatible.py` 增加 `tools` 字段透传；短期可用 **structured JSON 模拟 tool call**（模型输出 `{"tool_calls":[...]}`）作为 fallback，便于本地小模型调试。

### 3.8 Agent Loop 伪代码

```python
def run(self, config, user_input, state=None) -> AgentResult:
    skill = self.skill_loader.load(config.skill_name)
    registry = self.tool_registry.subset(skill.allowed_tools)
    state = state or empty_state(skill.name)
    state.messages.append(AgentMessage(role="user", content=user_input))

    for step_index in range(config.max_steps):
        step_started = now()
        response = self.llm.complete_with_tools(build_request(state, skill, registry))

        if response.tool_calls:
            state.messages.append(assistant_message_with_tools(response))
            for call in response.tool_calls[: config.max_tool_calls_per_step]:
                result = self.executor.execute(call)
                state.messages.append(tool_message(result))
                record_step(StepKind.TOOL, ...)
            continue

        # 无 tool_calls → 视为 final answer
        state.final_answer = response.content
        state.finished = True
        record_step(StepKind.FINAL, ...)
        return build_result(state, stopped_reason="final")

    return build_result(state, stopped_reason="max_steps")
```

---

## 4. Interview 专用 Tools 设计

### 4.1 Vault Tools

#### `search_notes`

```json
{
  "name": "search_notes",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "scope_paths": {"type": "array", "items": {"type": "string"}},
      "top_k": {"type": "integer", "default": 5}
    },
    "required": ["query"]
  }
}
```

**实现**：`RAGManager.hybrid_search` + scope 过滤；返回 `{hits: [{path, title, snippet, score}]}`。

**替代**：预灌整个 `ContextBuilder.build()`。

#### `read_note`

```json
{
  "name": "read_note",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {"type": "string"},
      "heading": {"type": "string"},
      "max_chars": {"type": "integer", "default": 4000}
    },
    "required": ["path"]
  }
}
```

**实现**：读 vault 文件；可选按 heading 截取。**必须**在成功时 append `working.notes_read_this_turn`。

#### `grep_vault`

精确查证；包装现有 `grep_search`。Coach skill 可用。

### 4.2 Interview State Tools

> **Tool 三分法（已落地，见 §15）**
>
> | 类型 | 交付方式 | 本节目的 |
> |------|----------|----------|
> | **Precondition** | App 注入 `runtime_context`（每轮 user turn JSON） | 默认 state/plan/scope/profile counts — **非** LLM 每轮必调 |
> | **On-demand Read** | LLM 可选调用 | 刷新或 debug 时读 state/plan |
> | **Action** | LLM 必须 tool 才能改 state | `advance_layer`、`select_topic` |
>
> Mock 面试：**禁止** routine 调用 `get_interview_state` / `list_plan_topics`；trace 中 `derived_metrics.routine_state_fetch` 应为 0。

#### `get_interview_state`（Optional refresh / debug）

**非每轮必调。** 正常 turn 从注入的 `runtime_context` 读取 state。仅在 runtime 缺失、矛盾或 action 结果 ambiguous 时刷新。

返回服务端权威 state（非前端 heuristic）：

```json
{
  "source": "server",
  "topic_phase": "active",
  "current_topic": "MCP 协议",
  "current_layer_index": 1,
  "current_layer_name": "调用链路与通信机制",
  "follow_up_count": 2,
  "should_consider_layer_transition": false,
  "at_last_layer": false
}
```

#### `advance_layer`（Action — state mutate）

```json
{
  "name": "advance_layer",
  "parameters": {
    "type": "object",
    "properties": {
      "reason": {"type": "string"},
      "force": {"type": "boolean", "default": false}
    },
    "required": ["reason"]
  }
}
```

**规则（代码 enforcement）**：

- `follow_up_count >= 2` 或 `force=true`（Director 场景）才允许 advance
- advance 后：`current_layer_index += 1`，`follow_up_count = 0`
- 返回新 state

**替代**：前端 `detectLayerTransition` 英文 regex（agent_v2 + `source=server` 时前端不再写 layer）。

到达最后一层后，`topic_phase` 自动变为 `closing`，允许收束话术或 `select_topic` 换 track。

#### `select_topic`（Action — state mutate）

Mock 首轮：**用户** UI/API 选 track（`topic_phase=awaiting_selection` 时不跑 interviewer agent）。中途换 topic：用户 API 或 interviewer `select_topic`。

#### `record_signal`

```json
{
  "name": "record_signal",
  "parameters": {
    "type": "object",
    "properties": {
      "signal_type": {"enum": ["possible_weak_point", "possible_partial", "possible_improvement"]},
      "point": {"type": "string"},
      "topic": {"type": "string"},
      "category": {"enum": ["knowledge_gap", "answer_structure", "communication", "thinking_pattern"]},
      "scope_suggestion": {"enum": ["domain", "universal"]},
      "evidence": {"type": "string"},
      "confidence": {"enum": ["low", "medium", "high"]}
    },
    "required": ["signal_type", "point", "evidence"]
  }
}
```

**关键**：`context_note_paths` 由 **`working.notes_read_this_turn`** 自动附加，LLM 不传。

#### `list_plan_topics`（Optional refresh / debug）

返回 `InterviewPlan` 的 topics + coverage。**非** Mock 开场 discovery 工具；track 列表已在 `runtime_context.compact_plan` 与 UI 中。

### 4.3 Profile / Memory Tools

#### `recall_profile`

```json
{
  "name": "recall_profile",
  "parameters": {
    "type": "object",
    "properties": {
      "topic": {"type": "string"},
      "include_due_only": {"type": "boolean", "default": false},
      "limit": {"type": "integer", "default": 4}
    }
  }
}
```

**实现**：调用 `split_weak_points_for_current` + due 排序；返回结构化 JSON，**不再** `render_candidate_profile_context` 整段塞 prompt。

```json
{
  "topic": "MCP 协议",
  "domain_weak_points": [{"point": "...", "due": true, "category": "..."}],
  "universal_habits": [{"point": "听题不仔细..."}],
  "strengths": [{"point": "..."}]
}
```

#### `write_observation_draft`（Curator / end session）

Session 内草稿；`end session` 时 commit 到 `interview_profile.json`（沿用 `apply_profile_observations`）。

---

## 5. 应用层封装

### 5.1 `InterviewInterviewerApp`（`src/agent/apps/interview_interviewer.py`）

```python
class InterviewInterviewerApp:
    """单轮 interviewer turn 的编排入口。"""

    def __init__(self, runtime: AgentRuntime, deps: InterviewDeps): ...

    def run_turn_stream(
        self,
        *,
        request: InterviewTurnRequest,
    ) -> Iterator[AgentStreamEvent]:
        """
        InterviewTurnRequest:
          - query: str
          - session_id: str | None
          - scope: ScopeSpec
          - interview_plan: InterviewPlan | None
          - interview_state: InterviewSessionState
          - chat_history: list[dict]
        """
```

**每轮执行前**：

1. 从 `interview_state` 初始化 `WorkingMemory`
2. 构建 user message：用户回答 + 极短上下文（**不含**全量 notes）
3. System prompt = `skills/interviewer/SKILL.md` + plan 摘要（非全文）

**每轮执行后**：

1.  Persist `interview_state` 到 session（服务端）
2.  Persist `agent_trace` 到 `interview-sessions/.../traces/{turn_id}.json`
3.  Return `notes_read_this_turn` 供 coach 与 profile 使用

### 5.2 `InterviewCoachApp`（turn review）

- Skill: `coach`
- max_steps: 2（`read_note` 补证据 + 输出 JSON debrief）
- 输出 schema 与现有 `generate_session_summary` 对齐
- **禁止** approval 与 interviewer 冲突：coach SKILL 改写，去掉「方向完全正确」

### 5.3 `InterviewCuratorApp`（session end）

两种策略（Phase 3 可选）：

| 策略 | 说明 |
|------|------|
| A. 保留现有 extraction | `interview_profile.update_from_session` + tools 仅做 audit |
| B. Curator agent | skill=curator 读 transcript + signals → 调 `write_observation_draft` → commit |

建议 **Phase 3 用 A**，**Phase 5 迁 B**。

---

## 6. Web API / SSE 变更

### 6.1 Feature Flag

```python
# workspace config 或环境变量
AGENT_V2_INTERVIEW=1
AGENT_V2_COACH=0
```

`stream_study_or_interview` 分支：

```python
if request.chat_mode == "interview" and agent_v2_enabled():
    yield from stream_interview_agent_v2(request, ...)
else:
    yield from stream_study_or_interview_legacy(request, ...)
```

### 6.2 新增 SSE 事件

| 事件 |  payload | 说明 |
|------|----------|------|
| `agent_step` | `{index, kind, tool_calls, latency_ms}` | 每步 trace |
| `tool_result` | `{name, ok, summary, latency_ms}` | 不含大段 note 正文 |
| `state_updated` | `{interview_state}` | 服务端 state |
| `answer_delta` | `{text}` | 同现有 |
| `done` | `{trace_id, stopped_reason, telemetry}` | 同现有，增加 agent 字段 |

### 6.3 前端改动要点

- **删除** `updateInterviewState` 的 layer/topic 推断逻辑（或仅作 fallback 展示）
- **改为** 消费 `state_updated` SSE 驱动 UI
- Debug panel 展示 `agent_trace` 折叠列表（作品集演示用）

### 6.4 Session 文件 schema 增量

`interview-sessions/.../{session_id}.json` 增加：

```json
{
  "agent": {
    "runtime_version": 1,
    "skill": "interviewer",
    "traces": ["traces/turn-0003.json"]
  },
  "interview_state": {
    "source": "server",
    "current_topic": "...",
    "current_layer_index": 1,
    "follow_up_count": 2
  }
}
```

---

## 7. MCP Server（Phase 4）

### 7.1 暴露工具

与 agent tools 同源（`tool_adapter.py`）：

| MCP Tool | 说明 |
|----------|------|
| `search_notes` | vault 语义检索 |
| `read_note` | 读单篇 note |
| `grep_vault` | 精确搜索 |
| `recall_profile` | 面试 profile |
| `list_interview_sessions` | 列出历史 session |

### 7.2 入口

```bash
uv run python -m mcp.server --transport stdio
uv run python -m mcp.server --transport sse --port 8010
```

### 7.3 Dogfooding

Interview Agent 内部 **可选** 通过 MCP Client 调 vault tools（配置开关），面试叙事：

> 「Interviewer 通过 MCP 访问 Obsidian vault，而不是把 notes 预灌进 prompt。」

---

## 8. Agent Eval（Phase 5）

### 8.1 目录

```text
eval/agent_eval/
├── interview_golden.json
├── rubric.md
└── results/
```

### 8.2 Golden case 示例

```json
{
  "case_id": "mcp_tools_call_gap",
  "turns": [
    {
      "user": "MCP 是 Host 和 Server 之间的接口...",
      "expect_tools": ["recall_profile", "read_note"],
      "expect_tool_args": {"read_note": {"path": "个人/面试/agent面试/MCP.md"}},
      "expect_state": {"follow_up_count": 1},
      "rubric": {
        "must_probe_component_boundary": true,
        "must_not_approve": true,
        "exactly_one_question": true
      }
    }
  ]
}
```

### 8.3 指标

| 指标 | 定义 |
|------|------|
| `tool_recall` | 该调 `read_note` 时是否调了 |
| `over_context_rate` | 是否未调 tool 却凭空引用 note 细节 |
| `anchor_accuracy` | `record_signal.context_note_paths` 是否 ⊆ `notes_read_this_turn` |
| `layer_transition_accuracy` | 对比 golden 的 layer index |
| `due_probe_rate` | due weak point 是否被追问 |
| `avg_steps` / `p95_latency` | 性能 |

### 8.4 脚本

```bash
uv run python scripts/agent_eval.py \
  --cases eval/agent_eval/interview_golden.json \
  --out eval/agent_eval/results/
```

---

## 9. 分步迁移计划

### Phase 0：骨架与 Flag（3–5 天）

**目标**：空 runtime 可跑通单 tool loop，零业务风险。

| 任务 | 文件 |
|------|------|
| 创建 `src/agent/schema.py`, `runtime.py`, `tool_registry.py`, `tool_executor.py` | 新增 |
| 实现 `ToolCallingLLMClient` wrapper | `src/agent/llm/tool_calling.py` + 扩展 `openai_compatible.py` |
| 实现 `SkillLoader` | `skills/interviewer/manifest.yaml` 占位 |
| 注册 mock tool `echo` | 测试用 |
| CLI | `scripts/agent_debug.py run --skill interviewer --input "hello"` |
| Feature flag | `AGENT_V2_INTERVIEW=1` 默认（2026-06 已切换）；`0` 回滚 legacy pipeline |

**验收**：

- [ ] `agent_debug.py` 在 1–3 step 内调用 echo tool 并输出 trace JSON
- [ ] trace 含 `tool_calls`, `tool_results`, `latency_ms`

---

### Phase 1：Vault Tools + 按需检索（5–7 天）

**目标**：用 tools 替代 `ContextBuilder` 全量预灌（agent_v2 可选）。

| 任务 | 说明 |
|------|------|
| 实现 `search_notes`, `read_note`, `grep_vault` | 薄封装 `services/rag/` |
| `InterviewInterviewerApp` 骨架 | 还不切主路径 |
| 对比测试 | 同题：pipeline 预灌 vs agent 检索 note 数量、token、回答质量 |

**验收**：

- [ ] 单轮 turn 平均 `read_note` ≤ 3 次
- [ ] `working.notes_read_this_turn` 非空且路径合法
- [ ] prompt chars 比 legacy 降 ≥ 40%（同 scope）

---

### Phase 2：Interview State Tools（5–7 天）✅ 已落地（Step 1–4）

**目标**：服务端 state machine 替代前端 heuristic；state **预注入** runtime，非每轮 `get_interview_state`。

| 任务 | 说明 |
|------|------|
| 实现 `advance_layer`, `select_topic`, `get_interview_state`, `list_plan_topics` | Action + optional refresh |
| `build_interviewer_runtime_context` 每轮注入 | 替代 routine state fetch |
| Session 持久化 `interview_state.source=server` | `interview_sessions.py` + load 时 normalize |
| SSE `state_updated` | 改 `app.py` + 前端 |
| `should_consider_layer_transition` | 注入 runtime；替代 `render_director_note` 前端注入 |

**验收**：

- [x] layer 变更经 `advance_layer` tool（非前端 regex）
- [x] `follow_up_count` 与 session JSON 一致
- [x] agent_v2 时 `isServerInterviewState()` 跳过 `updateInterviewState`
- [x] `derived_metrics.routine_state_fetch == 0` 为常态（见 5-turn golden test）

---

### Phase 3：Profile Memory Tools + Coach Agent（7–10 天）

**目标**：Profile 按需 recall；turn review agent 化；修复 anchor。

| 任务 | 说明 |
|------|------|
| 实现 `recall_profile`, `record_signal` | 对齐 [INTERVIEW_PROFILE_DESIGN.md](./INTERVIEW_PROFILE_DESIGN.md) |
| `context_note_paths` 仅来自 `notes_read_this_turn` | profile P1 落地 |
| 迁移 `skills/coach/SKILL.md` | 从 `SESSION_SUMMARY_SYSTEM_PROMPT` |
| `InterviewCoachApp` + `AGENT_V2_COACH=1` | 替代 `/api/interview/summary` 部分逻辑 |
| 弱化 `render_candidate_profile_context` | 仅 debug 保留 |

**验收**：

- [ ] 「听题不仔细」类 signal → `scope_suggestion=universal` 准确率 ≥ 90%（抽 10 session）
- [ ] 新 weak point 的 `domain_anchor.context_note_paths` 非空（有 read_note 时）
- [ ] 跨 session：同 note 不同 topic 名，domain weak point 仍注入（strong match）

---

### Phase 4?Agent Interview ???? + Coach ???? + Memory Commit Bridge + Agent Eval?????

**??**?????? Interview Agent ???????????????????????????MCP ?????? Phase 4?

| ?? | ???? |
|------|------|
| Interview agent_v2 ??? | `AGENT_V2_INTERVIEW=1` ?????`AGENT_V2_INTERVIEW=0` ??? legacy |
| Topic / layer server state | ?? topic ?? interviewer agent?topic ??? `/select-topic` ? `select_topic` action?final ? `commit_turn()` ?? follow-up |
| Runtime context ?? | state / compact plan / scope / profile counts ?????note/profile ????? tool ?? |
| Coach ???? | `AGENT_V2_COACH=1` ? `/api/interview/summary` ?? `InterviewCoachApp`?? review ??/fallback ?? |
| Memory Commit Bridge | session end ?? `commit_interview_memory()` ???? session signals/drafts???? `update_from_session()` |
| Agent Eval | `scripts/agent_eval.py` + `eval/agent_eval/interview_golden.json`?fake LLM golden suite ??? `eval-results/agent-golden/summary.json` |
| MCP | ???Phase 4 ??? MCP server/client |

**??**?

- [x] ?? interview ??? agent_v2?legacy ??? `AGENT_V2_INTERVIEW=0` ??
- [x] routine turn `derived_metrics.routine_state_fetch == 0` ?? CLI/golden eval ??
- [x] ?? topic ?? `topic_phase=awaiting_selection`???? interviewer agent
- [x] Coach Agent ??? `AGENT_V2_COACH=1` ????????? review schema
- [x] session end ?? `memory_commit` trace?`profile_update.source=commit_bridge`
- [x] `python scripts/agent_eval.py --cases eval/agent_eval/interview_golden.json --llm-mode fake` ?? 7 ? golden cases

---

### Phase 5?MCP + Curator Agent + Cleanup????

**??**?? Phase 4 ?????????????Curator Agent ???? cleanup?

| ?? | ?? |
|------|------|
| Read-only MCP server | ??? vault/profile read tools??? vault?? commit profile |
| Curator skill???? | ?? transcript/signals/drafts ?? commit ?????? profile ????? commit |
| Librarian Web ?????? | ?? answer mode ????? `librarian` skill |
| README / demo cleanup | ??????trace + eval + memory commit + optional MCP |

**??**?

- [ ] Cursor / MCP Inspector ?? read-only `search_notes` + `read_note`
- [ ] Curator Agent ??? trace??????? profile
- [ ] ????5 ?? live demo?trace + eval + memory commit + optional MCP?

---

### Phase 6：Cleanup（3–5 天）

| 任务 | 说明 |
|------|------|
| 标记 `build_interview_messages` deprecated | |
| 精简 `interview.py` | 仅保留 plan/state 纯函数 |
| 删除前端 legacy state 代码 | flag 移除后 |
| 文档 | 更新 `commands.md` interview 段 |

---

## 10. 时间线总览

```text
Week 1   Phase 0 ─────────────────── Runtime 骨架 + agent_debug
Week 2   Phase 1 ─────────────────── Vault tools + 检索对比
Week 3   Phase 2 ─────────────────── State tools + SSE
Week 4–5 Phase 3 ─────────────────── Profile tools + Coach agent
Week 6–7 Phase 4 ─────────────────── 主路径切换 + MCP
Week 8   Phase 5–6 ───────────────── Eval + Cleanup
```

**总计约 8 周**（单人 part-time 可 ×1.5）。

---

## 11. 风险与对策

| 风险 | 对策 |
|------|------|
| 模型 tool calling 不稳定 | JSON fallback；forced tool_choice；主路径用强模型 |
| 延迟上升（多 step） | max_steps=6；检索与 read 并行；coach 异步 |
| 迁移期双轨维护 | feature flag；legacy 6 周内移除 |
| tool 幻觉乱读 note | `read_note` 限制 scope_paths；path 校验 |
| 与 PROFILE P1/P2 冲突 | Phase 3 必须同 release；见 INTERVIEW_PROFILE 发布策略 |

---

## 12. 不做清单（路线 A 边界）

- 不引入 LangGraph / AutoGen 作为核心运行时
- Phase 4 前不做 multi-agent 互聊
- 不做通用「任意任务 Agent」UI；先 Interview + Librarian 两条 app
- 不在 agent loop 内做 embedding 相似度 dedupe（PROFILE P3 仍后置）
- MCP 不做 write vault（只读），写回 Obsidian 留 Phase 6+ / 人工确认

---

## 13. 成功标准（整体）

1. **叙事**：代码库存在可演示的 `AgentRuntime` + tool trace + MCP server。
2. **产品**：Interview 不再预灌 8 篇 note；profile 按需 recall；state 服务端权威。
3. **数据**：`context_note_paths` anchor 准确率 > 95%；跨 session 弱项注入生效。
4. **Eval**：≥ 5 golden cases 自动化；RAG eval 文化延伸到 agent eval。
5. **回滚**：legacy pipeline 一个 flag 可恢复。

---

## 14. 附录：Legacy → Agent 函数对照表

| Legacy | Agent 替代 |
|--------|------------|
| `ContextBuilder.build(scope, interview_context)` | `search_notes` + `read_note` |
| `render_candidate_profile_context()` | `recall_profile` |
| `build_interview_messages(..., candidate_profile_context=...)` | `SkillLoader` + 精简 messages + tools |
| `render_director_note()` | `get_interview_state().should_consider_layer_transition` |
| 前端 `updateInterviewState()` | `advance_layer` tool + `state_updated` SSE |
| `generate_session_summary()` | `InterviewCoachApp` (skill=coach) |
| `interview_profile.update_from_session()` | Phase 5 Curator 或保留 + `record_signal` 审计 |
| turn review `context_note_paths` from context.items | `working.notes_read_this_turn` |

---

## 15. 落地路线图（已定边界版）

> 假设 **Phase 0–1 已完成**（AgentRuntime、Vault tools、InterviewInterviewerApp 骨架）。  
> **范围**：Mock 模拟面试；**Real 真实面试** 等 agent 系统稳定后再接（见 §15.8）。

### 15.1 已定边界（不再讨论）

#### Tool 三分法

| 类型 | 性质 | 示例 | 交付方式 |
|------|------|------|----------|
| **Precondition** | 无决策价值 | topic、layer、follow_up、plan 摘要、scope、profile 可用性计数 | **App 注入 runtime context**；不进 trace 的 agent 决策 |
| **On-demand Read** | 有决策价值 | `search_notes` / `read_note` / `grep_vault` / `recall_profile` | LLM 自主调 tool |
| **Action** | 有副作用 | `advance_layer` / `record_signal` / **`select_topic`** | **必须走 tool 或等价 server API** |

#### 非 Action 的 state tools

| Tool | 定位 |
|------|------|
| `get_interview_state` | **Optional refresh**；禁止每轮 routine 调用 |
| `list_plan_topics(include_sources=true)` | **Plan 展开**；compact plan 已注入时不 routine 调用 |

**禁止注入**：note 全文、profile 弱点正文、RAG pack、grep 结果。

#### Topic 策略（Mock only）

| 阶段 | 行为 |
|------|------|
| 首轮 | `current_topic = null`，`topic_phase = awaiting_selection`；**用户** UI/消息选 track；**不**跑技术题 agent，**不** LLM `select_topic` |
| 选中后 | Server `select_topic(source=user)` → `topic_phase = active` → 才开始 interviewer agent |
| 中途换 topic | **仅两来源**：① 用户手动（API/UI）② 面试官主动（`select_topic` tool）；**无** user_confirmed 中间态 |
| Topic 收束 | `at_last_layer` 或 layer 信号足够 → `topic_phase = closing` → LLM 可 `select_topic` 或用户手动切 |

#### Trace 叙事

- Trace JSON **平级**字段：`runtime_context`（注入快照）+ `steps`（LLM 自主 tool 决策）+ `derived_metrics`
- Demo 话术：「State/plan 是 precondition；trace 里是可解释的检索与 action。」

---

### 15.2 Step 1 — Runtime Context 注入（P0，1–2 天）

**目标**：LLM 不再因看不到 state 而 routine 调 `get_interview_state` / `list_plan_topics`。

| # | 任务 | 文件 |
|---|------|------|
| 1.1 | 新增 `build_interviewer_runtime_context(request, machine, profile_store?)` | `src/agent/apps/interview_interviewer.py` |
| 1.2 | 重构 `build_turn_input()` → `build_turn_user_message()` = runtime context + 短 history + current user message + Task 文案 | 同上 |
| 1.3 | Task 文案改为「runtime authoritative；tools 仅用于 note/profile/state mutation」 | 同上 |
| 1.4 | 注入 `follow_up_count_before_this_turn`（非 commit 后值） | `state.py` snapshot 或 builder |
| 1.5 | 注入 profile **availability**（`profile_available`, `due_review_count_for_topic`, `domain_weak_count_for_topic`） | 调 `build_candidate_profile_debug` 只取 counts |
| 1.6 | 注入 `interview_mode: mock`、`topic_phase` | builder |
| 1.7 | 更新 `skills/interviewer/SKILL.md`：删除每轮必调 state/plan；写 Tool Use 边界 | `skills/interviewer/SKILL.md` |
| 1.8 | `manifest.json`：`output_contract.rules` 去掉 `use_state_tools` | `skills/interviewer/manifest.json` |

**runtime context 最小字段**：

```yaml
interview_mode: mock
topic_phase: awaiting_selection | active | closing
session_id, current_topic, current_layer_index, current_layer, next_layer
follow_up_count_before_this_turn, should_consider_layer_transition, at_last_layer
last_assistant_question
plan: compact topics + coverage (+ suggested_order)
scope: type, value, allowed_note_count
profile_availability: available, due_count, domain_weak_count
```

**验收**：

- [ ] Mock 跑 5 turn：`get_interview_state` 调用 0 次为常态；偶发 refresh ≤ 1 次/ session 可接受
- [ ] Prompt 不含 note/profile 全文
- [ ] `working.extra["interview_state"]` 与注入 context 一致

---

### 15.3 Step 2 — Topic 状态机 + `select_topic`（P0，2–3 天）

**目标**：Mock 首轮用户选 topic；中途换 topic 可审计、可复现。

| # | 任务 | 文件 |
|---|------|------|
| 2.1 | `InterviewState` 增加 `topic_phase`, `topic_selection_source` | `src/agent/interview/state.py` |
| 2.2 | `initialize_interview_state()`：**不再**默认 `plan.topics[0]`；`current_topic=null`, `topic_phase=awaiting_selection` | 同上 |
| 2.3 | 实现 `InterviewStateMachine.select_topic(name, *, reason, source)` | 同上 |
| 2.4 | 行为：重置 `layer_index=0`, `follow_up_count=0`, `sub_points_touched=[]`；写 `transition_history` type=`select_topic` | 同上 |
| 2.5 | 实现 tool `select_topic`（Action，`side_effect=mutate`） | `src/agent/tools/interview/select_topic.py` |
| 2.6 | Mock SKILL：`topic_phase=awaiting_selection` 时 **禁止** `select_topic`（等用户选） | `SKILL.md` |
| 2.7 | Mock SKILL：`topic_phase=closing` 或收束话术后，面试官可 `select_topic` | `SKILL.md` |
| 2.8 | API `POST .../sessions/{id}/select-topic` `{ topic, source: "user" }` | `app.py` + `interview_sessions.py` |
| 2.9 | UI plan 按钮 → 调 API，不再填 `I want to start with...` | `app.py` CHAT_HTML |
| 2.10 | `awaiting_selection` 时：**跳过** interviewer agent 或只返回固定「请选择方向」+ plan UI | `InterviewInterviewerApp` / `app.py` |

**验收**：

- [ ] 新 session：`current_topic` 为 null，profile 不按 plan[0] 注入错误 topic
- [ ] 用户点「RAG 系统」→ state 立即更新 → 下一轮 runtime context 正确
- [ ] 中途 LLM `select_topic` 可在 trace 看到 reason + transition
- [ ] 用户手动切 topic 与 tool 切 topic 走同一 `select_topic()` 实现

---

### 15.4 Step 3 — Trace 与 derived_metrics（P1，1–2 天）

**目标**：作品集可演示「precondition vs agent decision」。

| # | 任务 | 文件 |
|---|------|------|
| 3.1 | Trace 写入 `runtime_context`（turn 开始前完整注入文本或结构化 dict） | `trace/recorder.py` 或 `rewrite_trace_interview_metadata` |
| 3.2 | 增加 `derived_metrics` | 同上 |
| 3.3 | 指标：`routine_state_fetch`, `notes_read`, `profile_recalled`, `layer_advanced`, `topic_selected`, `over_search` | 同上 |
| 3.4 | Debug panel / session trace 展示 derived_metrics | `app.py` 可选 |

**derived_metrics 规则示例**：

| 字段 | 计算 |
|------|------|
| `routine_state_fetch` | step1 且无 prior action 即调 `get_interview_state` |
| `over_search` | `search_notes` ≥ 3 且无 `read_note` |
| `anchor_ready` | `record_signal` 时 `notes_read_this_turn` 非空 |

**验收**：

- [ ] 任意 turn trace 可一眼区分 injection vs tool steps
- [ ] `scripts/agent_debug.py` 输出含 `derived_metrics`

---

### 15.5 Step 4 — Action 工具与 Profile 对齐（P1，2–3 天）

**目标**：state mutation 与 profile anchor 闭环。

| # | 任务 | 说明 |
|---|------|------|
| 4.1 | `advance_layer` manifest：`side_effect=mutate`（修正当前误标 read/none） | `advance_layer.py` |
| 4.2 | `record_signal`：`context_note_paths` **仅**来自 `working.notes_read_this_turn` | 已有方向，验收 anchor |
| 4.3 | `recall_profile` 保持按需；runtime 只注入 counts | 不回归 preload 全文 |
| 4.4 | `commit_turn()` 时机不变；文档/SKILL 写明 follow_up 语义 | SKILL + runtime context |
| 4.5 | 去掉 agent_v2 路径下前端 `inferTopic` / `updateInterviewState` 对 topic/layer 的写入 | `app.py`；改消费 `state_updated` SSE |

**验收**：

- [ ] `record_signal` 的 paths ⊆ 本轮 `read_note` paths
- [ ] Layer 切换只出现在 `advance_layer` tool result 或 trace，不靠英文 regex

---

### 15.6 Step 5 — 主路径切换与回归（P1，2 天）

| # | 任务 |
|---|------|
| 5.1 | `AGENT_V2_INTERVIEW=1` 默认（Mock + runtime injection + select_topic） |
| 5.2 | Legacy pipeline 保留 flag 回滚 |
| 5.3 | 更新 `tests/test_interview_state_tools.py`：不再假设必调 `get_interview_state` |
| 5.4 | 更新 `scripts/agent_debug.py` 示例：带 runtime context fixture |
| 5.5 | 同步修订本文 §4、§9 旧表述（state tools 非每轮必调） |

**验收**：

- [ ] 完整 Mock session（选 topic → 3 layer → 换 topic）trace 可读
- [ ] 现有 pytest 绿

---

### 15.7 Step 6 及以后（稳定后再做）

| 步骤 | 内容 | 前置 |
|------|------|------|
| **6** | Coach agent 化 + `build_coach_runtime_context` | Step 1–5 稳定 |
| **7** | Agent eval golden cases + `derived_metrics` 门禁 | Step 3 |
| **8** | MCP Server 暴露 vault/profile tools | Step 5 |
| **9** | **Real 面试模式**：`interview_mode=real`、首轮 LLM `select_topic`、隐藏 track 菜单 | Step 5 + eval 基线 |
| **10** | Curator agent / Obsidian writeback | 可选 |

---

### 15.8 Real 模式（仅记录，暂不实施）

| 项 | 与 Mock 差异 |
|----|--------------|
| 首轮 | LLM 可 `recall_profile` + `select_topic` 定 opening |
| UI | 不展示 track 按钮菜单（或弱化） |
| 中途换 topic | 以 interviewer `select_topic` 为主；用户 manual 保留作 override |
| 启用条件 | Mock trace 质量稳定、`routine_state_fetch≈0`、eval 基线存在 |

---

### 15.9 建议执行顺序（一览）

```text
[Done] Phase 0 — Runtime 骨架
[Done] Phase 1 — Vault tools 按需检索
  ↓
Step 1 — Runtime context 注入 + SKILL 修正     ← 立刻做，解 get_state 误调
  ↓
Step 2 — topic_phase + select_topic + UI API   ← Mock 首轮/中途换 topic
  ↓
Step 3 — trace runtime_context + metrics       ← 作品集叙事
  ↓
Step 4 — Action 修正 + profile anchor          ← 与 INTERVIEW_PROFILE P1 对齐
  ↓
Step 5 — 主路径切换 + 测试
  ↓
Step 6+ — Coach / Eval / MCP / Real（后置）
```

---

### 15.10 单轮 Mock turn 理想链路（验收用例）

```text
1. [Precondition] App 注入 runtime context（含 topic_phase=active, follow_up_count_before=2）
2. [Optional Read] LLM → recall_profile（若 due_count>0 且要针对性 probe）
3. [Optional Read] LLM → read_note / search_notes（若要核对用户答案）
4. [Optional Action] LLM → advance_layer（层信号足够）
5. [Optional Action] LLM → record_signal（暴露弱项）
6. [Final] 一句点弱项 +  exactly one question

不应出现：routine get_interview_state / list_plan_topics
不应出现：未 read_note 却引用 note 细节
换 topic 时：select_topic(source=interviewer) 或 用户 API，且 layer 重置
```

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-06-20 | 初稿：路线 A 目录结构、AgentRuntime 接口、Interview 迁移 Phase 0–6 |
| 2026-06-20 | §15：已定边界版落地路线图（Tool 三分法、Mock topic、trace、Real 后置） |
| 2026-06-20 | skills/interviewer & coach v2：回填 legacy prompt P0–P2 行为细则 |
| 2026-06-20 | P0–P2 补完：session normalize、`topic_phase=closing`、runtime  enrichment、5-turn golden test、§4/§9 修订 |
