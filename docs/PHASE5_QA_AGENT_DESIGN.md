# Phase 5：QA / Librarian Agent 化落地方案

> **目标**：把 legacy `AgentAnswerPipeline`（router → 并行召回 → 单次生成）迁移为 agent-native 的 `librarian` skill —— 一个 **scope 驱动 + bounded tool loop + 证据策略** 的自适应问答 agent。
>
> **范围**：仅 QA / 笔记问答。审计（audit）与整理（organize）**不**进入本 Phase，保留为 workflow（见 §9）。
>
> **设计原则**：少量高层配置 + 原子工具 + loop 自适应；不把 legacy 的分支/k 值/按钮搬进 runtime。
>
> 相关代码：`src/services/rag/agent_answer.py`（legacy）、`skills/librarian/`（已存在骨架）、`src/agent/tools/vault/`。
> 关联文档：[agent_plans.md](./agent_plans.md)、[INTERVIEW_PROFILE_DESIGN.md](./INTERVIEW_PROFILE_DESIGN.md)。

---

## 0. 已定决策（本方案前提）

| # | 决策 | 说明 |
|---|------|------|
| D1 | **单一 effort 维度**，A–E 不做成代码分支 | A–E 仅作 skill prompt 里的"入口偏好"描述；执行路径是同一个 librarian loop，由 effort budget 调步数/工具白名单 |
| D2 | **第一版不做独立分类器** | gating 先验由 scope 确定性推导；loop 内首步自行判断是否检索。仅当 eval 显示系统性过/欠检索时再评估引入 Adaptive-RAG 式分类器（见 §7 后置项） |
| D3 | **证据校验先做 prompt 策略** | 不单开 verify pass；CRAG 式"证据不足→预算内升级→声明不足"写进 SKILL。eval 不达标再抽离独立 verifier |
| D4 | **organize 不排期** | 放在所有 Phase 之后，作为独立议题；本 Phase 不接入 agent |

---

## 1. 现状基线

### 1.1 Legacy QA 管线

```text
query
  -> LLMIntentRouter.route()        # gating：NOTES / NOTES_ONLINE / REGEX_SEARCH_FILES
  -> 并行召回 (notes hybrid / rg / bm25 / online)
  -> build_agent_context()          # 打包 + citation [N/R/B/W]
  -> 单次 LLM 生成
```

- gating 已存在（router），但输出是 command 而非 effort level。
- k 值（`notes_top_k=5`, `dense_top_k=50`, `bm25_top_k=8`, `rrf_k=60` 等）焊死在 `AgentAnswerConfig`，**不外露**——符合"底层参数测试期调好"原则，迁移后保留这一形态。

### 1.2 已有 agent 资产

- **`skills/librarian/`** 已存在：`max_steps=6`、`allowed_tools=[grep_vault, read_note, search_notes]`、prompt 已含"工具分流 + 证据不足声明"雏形。**QA agent 骨架已在**。
- **原子工具已按 scope guard 实现**：
  - `search_notes`：hybrid 检索，`filter_items_by_scope`，k 值内置。
  - `grep_vault`：精确/正则，`filter_items_by_scope`。
  - `read_note`：永远返回 `content`；`section_id` 跳节，`offset` 节内续读；`truncated=true` 时附带 `sections` map。
- **scope guard 链路就绪**：`ToolExecutionContext.scope_note_paths` + `guards.py`。

### 1.3 缺口（本 Phase 要补）

| 缺口 | 性质 |
|------|------|
| `list_notes` 枚举工具 | L3 探索建候选 map 必需；当前无 |
| `online_search` 工具 | 当前 online 是 router 分支 + `OnlineSearchClient`，未暴露为 agent 工具 |
| scope → effort budget 映射 | 当前 librarian 固定 `max_steps=6`，无 scope 自适应 |
| 证据不足升级策略 | SKILL 仅"声明不足"，缺"预算内升级"动作 |
| web answer 模式接 librarian | 当前 answer 模式仍走 legacy pipeline |
| QA golden eval | 无覆盖 L0–L3 的回归用例 |

---

## 2. 目标架构

```text
query + scope + online_flag + strict_evidence
  │
  ├─ (确定性，无 LLM) ScopeRouter：scope → effort budget + 工具白名单
  │
  └─ librarian loop  (复用 AgentRuntime，max_steps = budget)
       工具：search_notes / grep_vault / read_note / list_notes(新) / online_search(新)
       skill 内置：
         - A–E 入口偏好（描述，非分支）
         - 是否检索的自判断（D2：替代独立分类器）
         - 证据不足 → 预算内升级 → 声明不足（D3）
       │
       -> final answer + citation 渲染
```

**关键点**：A–E / L0–L3 **不是 5 个代码分支**，而是同一 loop 在不同 effort budget 下的自适应表现。budget 决定步数与工具白名单上界，loop 决定实际深浅。

---

## 3. Effort 维度（取代 A–E 分支）

单一维度 `effort_budget`，由 scope 确定性推导，online 为正交开关。

| scope | 确定性先验 | 默认 effort | max_steps | 工具白名单（上界） |
|-------|-----------|-------------|-----------|--------------------|
| `selected_notes` | 用户已圈定笔记 → 直接读 + 综合 | **L2** | 4 | `read_note`, `grep_vault`（跳过 `search_notes`/`list_notes`） |
| `folder` / `topic` | 范围内 RAG，可升探索 | **L1**（可升 L3） | 6 | `search_notes`, `grep_vault`, `read_note`, `list_notes` |
| `search` | 意图开放 | **L1 起**，loop 自定深浅 | 6 | 全部本地工具 |
| online 开关 = on | 正交 | 任意 level 叠加 | +1 | 追加 `online_search` |

**effort level 语义（写进 SKILL 作为入口偏好，对应你的 A–E / L0–L3）**：

| Level | 旧路径名 | 行为偏好 | 典型问题 |
|-------|----------|----------|----------|
| L0 | A simple_direct | 不调工具直接答 | 通用解释、非笔记依赖 |
| L1 | B scoped_rag / D exact_lookup | 1 次 `search_notes` 或 `grep_vault` + 0–1 篇 `read_note` | 普通 QA、精确查找 |
| L2 | C selected_notes_synthesis | 读 selected / top 候选笔记 → 汇总 | "综合这几篇" |
| L3 | E exploratory_research | `list_notes`/`grep`/`search` 建候选 map → 多篇 `read_note` → 证据检查 → 汇总 | 跨笔记研究/复杂整理 |

- D（exact_lookup）合并进 L1，表现为"grep 优先"的工具选择，不单列。
- budget 给的是**上界**；loop 可在更低 level 提前收束（L3 budget 下若首检索已足够即停）。

---

## 4. 工具补全（原子化）

### 4.1 `list_notes`（新）

**用途**：枚举 scope 内笔记结构，供 L3 建候选 map。ContextBuilder "scope 枚举"职责的原子化。

```json
{
  "name": "list_notes",
  "description": "List markdown note paths within the current vault scope, optionally filtered by a path/name substring.",
  "parameters": {
    "type": "object",
    "properties": {
      "filter": {"type": "string", "description": "optional path or filename substring filter"},
      "limit": {"type": "integer"}
    },
    "required": []
  },
  "side_effect": "none"
}
```

**实现要点**：
- 基于 `ctx.vault_root.rglob("*.md")`，再过 `filter_items_by_scope(..., ctx.scope_note_paths)`。
- `limit` 内置上限（建议 200），返回 `paths` + `total` + `truncated`。
- 只返回路径与标题，**不返回正文**（正文走 `read_note`，保持原子）。
- 严格 scope guard：`search` scope 才是全库，folder/topic/selected 受限。

### 4.2 `online_search`（新）

**用途**：把 legacy 的 online 分支暴露为 agent 工具，由 online 开关 gate。

```json
{
  "name": "online_search",
  "description": "Search the public web for information not in the vault. Only available when the user enabled online for this query.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "top_k": {"type": "integer"}
    },
    "required": ["query"]
  },
  "side_effect": "none"
}
```

**实现要点**：
- 复用 `OnlineSearchClient.search(query, top_k)` → `OnlineSearchResponse`。
- `ToolExecutionContext` 新增字段 `online_search_client: Any | None = None`；online 关闭时**不注册**该工具（白名单层 gate），而非运行期拒绝——避免 agent 反复尝试。
- 返回 `results`（title/url/snippet）+ `enabled` + `message`。
- citation 用 `W{n}`，与本地 `N/R/B` 区分。

### 4.3 明确**不做成工具**

| 候选 | 决策 | 理由 |
|------|------|------|
| `plan_read_notes` | ❌ | 规划是推理步骤，不是工具。做成工具 = 用 tool call 做 prompt stuffing，重蹈 `get_interview_state` routine 覆辙。agent 在 loop 内自行规划读哪几篇 |
| evidence verifier | ❌（v1） | D3：先做 SKILL 策略；eval 不达标再抽离 |
| query classifier | ❌（v1） | D2：scope 先验替代；后置 |
| ContextBuilder 整包 | ❌ | 拆成 `list_notes` + `read_note` + `search_notes`，不一键打包 |

### 4.4 `read_note` / `search_notes` / `grep_vault`

- `read_note` 保持原子读取职责，但增加可选 `reason` 参数，用于记录为什么选择这篇 note。该字段不影响读取逻辑，只进入 tool result / trace / 前端展示。
- `search_notes` / `grep_vault` 保持现状。

示例：

```json
{
  "path": "个人/面试/agent面试/memory.md",
  "reason": "directly supports memory types and lifecycle"
}
```

---

## 5. SKILL 设计（`skills/librarian/SKILL.md` 重写）

在现有骨架上扩写，**仍是薄行为层**，不堆配置。核心新增段落：

### 5.1 Effort & Scope Boundary
- runtime context 注入 `scope`（type/value/allowed_note_count）、`scope_index`、`effort_level`、`online_enabled`、`strict_evidence`。
- 说明：scope 是权威边界；不要枚举 scope 外笔记；effort_level 是预算上界，可提前收束。
- `scope_index` 是轻量目录索引：小 scope（≤30 篇）完整注入 path/title；大 scope 只注入计数与提示，要求 agent 用 `list_notes` / `search_notes` 获取候选。
- `scope_index` 只用于候选选择，不作为内容证据引用。

### 5.1.1 Strict Evidence Constraint
- 默认不改变生成姿态：允许当前自然的合理延展。
- 当 UI 勾选「仅依据资料」时，runtime 注入 `strict_evidence=true` 和 `# Strict Evidence Constraint`。
- strict 下只回答当前 scope 与 tool observations 直接支持的内容；资料不支持的概念要说明不足，不从相邻概念推导完整架构。
- strict 是「保守知识助手」姿态，不是证据审计报告。回答要保持自然、面向用户；除非用户要求证据分析，否则不使用「直接支持 / 推断 / 缺失」这类审计标签。

### 5.2 Retrieval Decision（替代独立分类器，D2）
- 首步自判断"是否需要检索"：通用常识问题（L0）可直接答，不调工具。
- 依赖笔记事实时才检索；`selected_notes` scope 跳过 `search_notes` 直接 `read_note`。
- 精确串（命令/报错/API/文件名）优先 `grep_vault`；概念/解释/对比优先 `search_notes`。

### 5.3 Evidence Policy（CRAG 式，D3）
- 检索后自检证据是否足以支撑结论。
- **不足时的升级顺序（预算内）**：① 换检索词/换工具再试一次 → ② 读更多候选笔记 → ③ 若 online 开启，调 `online_search` → ④ 仍不足：明确声明"依据不足"，不编造。
- 升级次数受 effort budget 限制；预算耗尽即停并声明。
- 禁止把通用知识伪装成笔记内容（沿用 legacy 规则）。
- 未 `read_note` 成功不得声称完整阅读 scope；不过强声明覆盖范围。
- 检索返回 0 或证据已足够时，停止搜索并综合作答，不要用剩余 step 做验证性搜索。
- 默认模式允许合理延展，但用自然语言轻标注，例如「结合这些笔记，可以理解为...」；不要把每个回答拆成审计式证据分层。

### 5.4 Output / Citation
- 先给直接答案，再补解释。
- 按用户问题组织答案，不复现笔记标题/表格/章节顺序；用问答式自然总结，不做资料整理。
- 推断内容自然带过即可，不做证据分层展示。
- 仅引用 tool observation 出现过的来源；本地 `[N/R/B]`、网络 `[W]`。
- 未 `read_note` 成功不得声称读过某篇。
- 中文回答（除术语/命令/代码）。
- 不输出内部检索准备语（如「我已经收集了足够材料」）。
- 不声称「完整阅读所有内容」，除非确实完成候选枚举并阅读了支撑该声明的相关笔记。
- 默认不用 Markdown 表格；用户明确要求对比表格时再使用。优先 bullets / 小节。

### 5.5 manifest 调整
- `allowed_tools`：`search_notes`, `grep_vault`, `read_note`, `list_notes`（online 关闭时不含 `online_search`）。
- `max_steps`：由 ScopeRouter 按 effort 注入覆盖（默认 6）。
- `output_contract`: `grounded_answer`（保持）。

---

## 6. App 层接线

### 6.1 ScopeRouter（确定性，无 LLM）
- 输入：`scope_type` / `scope_value` / `online_flag`。
- 输出：`effort_level`、`max_steps`、`tool_whitelist`。
- 纯查表（§3），不调模型——这是 D2 的核心：用确定性先验替代分类器。
- 同步构建 `scope_index`：小 scope 直接注入完整目录，大 scope 注入 incomplete hint。

### 6.2 web answer 接 librarian
- `web/app.py` answer 模式新增 agent_v2 分支（feature flag，如 `AGENT_V2_QA=1`），对照保留 legacy `AgentAnswerPipeline`。
- 复用 interview 已建的 SSE 事件（`agent_step` / `tool_result` / `answer_delta` / `answer` / `done`）。
- RAG manager 走已有 lazy / prewarm 机制：`search` scope 即时构建，folder/selected 惰性 + 后台预热。
- online 开关→是否注册 `online_search`（白名单 gate）。
- UI 增加「仅依据资料」开关 → `strict_evidence`。默认关闭，避免影响当前自然回答质量。

### 6.3 配置最小化（验证原则）
保留的高层旋钮**仅**：`scope`（folder/topic/selected/search）、`online` on/off、（可选未来）`effort` 手动覆盖。
焊死在工具/Router 内：所有 k 值、rrf_k、model、温度、max_chars、step 预算映射。

---

## 7. 分步落地

| Step | 内容 | 风险 | 验收 |
|------|------|------|------|
| **S1** | 新增 `list_notes`、`online_search` 工具（纯读 + scope guard）；`ToolExecutionContext` 加 `online_search_client` | 低 | 单测：scope 过滤、limit 截断、online 关闭不注册 |
| **S2** | 重写 `skills/librarian/SKILL.md`（§5）+ manifest | 低 | prompt review；L0 不调工具、L3 会枚举 |
| **S3** | ScopeRouter（scope→effort/budget/whitelist），确定性 | 低 | 单测：四类 scope 映射正确 |
| **S4** | web answer 接 librarian（feature flag，对照 legacy） | 中 | 手测四类 scope + online 开关；SSE 正常 |
| **S5** | QA golden eval：覆盖 L0–L3 + online + 证据不足场景 | 中 | over/under-retrieval 指标；citation 命中率；无幻觉笔记 |
| **S6** | （条件触发）按 eval 结果决定是否引入分类器 / verifier | — | 仅在 S5 不达标时启动 |

**顺序**：S1 → S2/S3 可并行 → S4 → S5 →（条件）S6。

### 后置项（明确不在 v1）
- **独立 query classifier**（D2）：触发条件——S5 eval 显示 agent 系统性过检索（简单问题也 L3）或欠检索（复杂问题停在 L1）。届时评估 Adaptive-RAG 式轻量分类器，权衡 +1 LLM 延迟 hop。
- **独立 evidence verifier**（D3）：触发条件——S5 显示 citation 质量差 / 幻觉率高，prompt 策略压不住。届时抽离为 CRAG 式 retrieval-grade pass。
- **organize agent 化**（D4）：不排期，放在所有 Phase 之后单独立项。

---

## 8. Eval 指标（S5）

| 指标 | 含义 | 目标 |
|------|------|------|
| `routine_over_retrieval` | L0 类问题却调了检索的比例 | 低 |
| `under_retrieval` | 需笔记事实却没检索/没读 note | 低 |
| `citation_grounding` | 答案引用的来源确在 tool observation 中 | 高 |
| `note_hallucination` | 声称读过/引用了未成功 read 的笔记 | 0 |
| `evidence_insufficient_honesty` | 证据不足时是否如实声明 | 高 |
| `steps_per_level` | 各 effort level 实际步数分布 | 符合预算 |
| `online_gated` | online 关闭时是否完全不触发 online_search | 100% gated |
| `strict_evidence_respected` | strict 下是否拒绝无直接证据的推断 | 高 |
| `read_reason_quality` | `read_note.reason` 是否说明了选择依据 | 高 |

golden 用例覆盖：
- L0：通用概念解释（不依赖笔记）
- L1：单点笔记问答 + 精确串查找（grep）
- L2：selected_notes 综合
- L3：跨笔记研究 / "整理某主题"
- online on/off 各一组
- 证据不足（库内确实没有）一组

---

## 9. 边界与不做清单

- **audit / organize 保留 workflow**：写副作用 + 自带 review loop，本 Phase 不接 agent；organize 不排期（D4）。
- **ContextBuilder**：QA 路径退役，由原子工具替代；legacy interview / 非 agent 路径继续使用，待相关路径下线再删。
- **legacy `AgentAnswerPipeline`**：feature flag 保留对照，eval 稳定后再下线。
- **不引入**：LangGraph、plan_read_notes 工具、v1 分类器、v1 verifier、外露 k 值配置。

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-06-21 | 初稿：QA Librarian agent 化方案；单一 effort 维度（A–E 降为入口偏好）；分类器/verifier 后置；organize 不排期 |
| 2026-06-21 | 补充：`strict_evidence` 单向约束、`scope_index` 预注入、`read_note.reason`、表达契约与 eval 指标 |
