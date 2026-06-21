# Interview Profile 弱项模型设计

本文档描述面试模式 Candidate Profile 中 weak point 的字段契约、写入/去重/注入三条链路，以及分阶段落地顺序。**设计已定，实现按 P0–P5 推进。**

相关代码：`src/services/workflows/interview_profile.py`、`src/services/workflows/interview.py`。

---

## 背景与问题

当前 weak point 用 `topic` 字符串同时承担两种职责：

1. **领域归属**（这条弱项属于哪个知识域）
2. **注入过滤**（当前 session 是否把这条弱项交给面试官 LLM）

这导致两类失败：

| 问题 | 表现 |
|------|------|
| Topic 字符串漂移 | 同一知识域在不同 session 里 LLM 产出不同 topic 名（如「RAG 基础」vs「向量检索与召回」），domain 弱项注入时 `topic == current_topic` 匹配失败，弱项失联 |
| Universal 弱项误绑 topic | 跨 topic 习惯类弱项（如「听题不仔细」「对比类缺乏结构化展开」）被标在首次出现的 topic 下，换 topic 后面试时不再注入 |

写入时做 topic 标准化或维护别名表不是长期有效方案——LLM 每次生成的 topic 名不可穷举。

参考 MemCoach 的两阶段提取与 SM-2 调度，但在注入层保留 richer guidance（planned_layer、行为策略、due priority），仅将**过滤键**从 topic 字符串改为 `scope` + `domain_anchor`。

---

## 设计目标

1. **Universal 弱项**：任何 topic / session 下持续注入（带上限）。
2. **Domain 弱项**：仅在与当前 session 知识域相关时注入，不依赖 topic 字符串精确匹配。
3. **Topic 降级**：`topic` 仅作首次发现时的可读标签（provenance），不参与过滤。
4. **Session 内 profile 冻结**：不在 turn 中更新 profile；**Agent V2（方案 B）** 下长期画像的唯一写入入口是 **session 结束单层 extractor**（见下文「证据链（方案 B）」）。
5. **去重稳定优先**：P0–P2 以 scope-aware 确定性匹配为主；Stage 2 LLM reconcile 后置（P3），待 profile 数据量增大后再引入。

---

## 证据链（方案 B，Agent V2）

Coach 不再承担 profile 写入决策。每轮 coach 只产出给用户看的 turn review；profile 的提取、去噪、去重、合并、SR 更新全部在 session 结束时一次性完成。

```text
每轮（Coach，纯 turn-review）
  输入：previous question + user answer + interviewer follow-up（分区）
  输出：question_requires / coach_note / covered / gaps /
        thinking_framework / expression_example
  profile_signals → 始终 []（schema 兼容字段，不写入 profile）

Session 结束（单层 extractor，主路径）
  输入：
    - session transcript（问答原文）
    - turn reviews（coach 反馈作辅助证据，含 question_requires / gaps / coach_note）
    - turn_review.context_note_paths（turn 级 provenance，非 profile_signals）
    - existing profile（compact）
    - interview_plan + interview_state（layer / topic 锚点）
  输出：observations → apply_profile_observations() → interview_profile.json

降级（extractor LLM 失败时）
  - 优先从 turn_review.gaps + coach_note 合成 weak_point observations
  - 仍无信号时，commit_bridge 可消费 session 级 record_signal / observation_draft（若其他 agent 写入）
```

**与旧设计的差异**

| 维度 | Legacy / 旧文档 | 方案 B（当前 Agent V2） |
|------|-----------------|-------------------------|
| 每轮 profile 证据 | `turn_review.profile_signals` + `record_signal` | **无**；coach 不 recall、不 record |
| Session 结束输入 | signals 为主 + transcript | **transcript + turn reviews 为主**；signals 仅作 commit_bridge 补充 |
| `profile_signal_count` in trace | 有意义的审计指标 | Agent V2 coach 恒为 0，**不代表 coach 未发现 gap** |

Legacy pipeline（`interview.py` 内旧 prompt）仍可能产出 `profile_signals`；Agent V2 coach（`skills/coach` + `interview_coach.py`）强制清空。新 session 以方案 B 为准。

---

## Anchor 是什么？

**Anchor（域锚点）** 不是弱项内容本身，而是写入 weak point 那一刻系统记录的 **「域身份快照」**，用于回答：

> 这条 domain 弱项属于哪个知识域？下次面试聊另一个 topic 时，还要不要把它交给面试官？

| 概念 | 字段 | 职责 |
|------|------|------|
| 弱项内容 | `point` | 用户差在哪；表述会变，靠 text sim 合并 |
| 可读标签 | `topic` | 人读时属于哪块；**不参与过滤** |
| Session 出处 | `source_note_paths` | 哪次面试、选了哪些笔记；太宽，不能当域键 |
| **域锚点** | `domain_anchor` | 注入 / 去重 / scope 升级用的域身份 |
| 注入范围 | `scope` | `universal` 全局；`domain` 仅域相关时注入 |

第一版 anchor 形态为 **`domain_anchor` 三元组**（代码写入，LLM 不产出）：

```json
{
  "plan_topic": "RAG 系统",
  "context_note_paths": ["个人/面试/agent面试/RAG/rag基础概念.md"],
  "scope_path": "个人/面试/agent面试/RAG"
}
```

| 子字段 | 含义 | 匹配角色 |
|--------|------|----------|
| `context_note_paths` | 弱项暴露时**本轮**实际注入的 note paths | **最优锚源**：仅当数量 ≤3 时作为强锚；过宽时降级到 TopicCard fallback |
| `plan_topic` | 写入时 `interview_state.current_topic` | **辅键**：同 folder 下 note 集合略有差异时的 fallback |
| `scope_path` | session 的 folder/tag `source_value`，或 note 父目录 | **辅键**：同一知识族前缀匹配 |

`scope=universal` 时 `domain_anchor` 为空对象或省略。

**暂不落地 LLM 匹配字段**（如 `domain_key`）。若三元组效果不足，再对比评估 LLM 约束 enum 等方案。

---

## 字段契约（Schema v2）

在现有 weak_point 上新增/调整如下字段。

| 字段 | 类型 | 谁写 | 职责 |
|------|------|------|------|
| `scope` | `"domain"` \| `"universal"` | Session-end extractor LLM | 注入范围 |
| `category` | `"knowledge_gap"` \| `"answer_structure"` \| `"communication"` \| `"thinking_pattern"` | Session-end extractor LLM | 弱项性质；影响追问策略、`planned_layer` 语义、scope 自动升级白名单 |
| `domain_anchor` | `object` | **代码**（`new_weak_point()`） | Domain 匹配主锚；见上文三元组 |
| `topic` | `string`（保留） | Session-end extractor LLM | 首次发现时的可读标签；**不参与注入过滤** |
| `planned_layer` | `string`（保留） | Session-end extractor LLM / session state | domain：InterviewPlan coverage layer；universal：probe hint |
| `source_note_paths` | `string[]`（保留） | **代码**（`new_weak_point()`） | Session 级 provenance：用户本次选中的全部笔记路径 |
| `sr` | `object`（保留） | 代码（SM-2） | 间隔重复参数；见「Due reviews」 |

### `scope` 分类规则（写入 session-end extractor prompt）

| 观察 | scope | category 示例 |
|------|-------|---------------|
| 具体知识点、机制、组件缺口 | `domain` | `knowledge_gap` |
| 答题结构、对比展开、概念到组件映射 | 首次仅在一域出现 → `domain`；已在多域复现 → `universal` | `answer_structure` / `thinking_pattern` |
| 听题、表达、沟通、答非所问 | `universal` | `communication` |

**scope 迁移（非 LLM UPDATE）**：`scope` 默认 immutable，但系统可在以下情况将 `domain` → `universal`：

- **P4 数据迁移**：历史 behavioral 弱项批量升级。
- **P2 复现规则**（代码）：见下文「domain → universal 自动升级」。
- 迁移时 `domain_anchor` 保留原值（审计用），注入改走 universal 路径。

禁止 LLM 在普通 `UPDATE` / `IMPROVE` 中修改 `scope`。

### `domain_anchor` vs `source_note_paths`

| 字段 | 写入时机 | 内容 | 用途 |
|------|----------|------|------|
| `source_note_paths` | `new_weak_point()` | `session.context.source_note_paths[:12]`（本次面试用户选中的**全部**笔记） | 审计 / provenance |
| `domain_anchor` | `new_weak_point()` | 见下文 `resolve_domain_anchor()` | **Domain 匹配键** |

**`domain_anchor` 是系统级字段，LLM 不产出、不修改。**

### Turn 级数据补全（P1 前置）

`context_note_paths` 优先来自 **turn 级实际 context**，不能从 plan 顺序猜测；但只有足够窄（≤3 条）时才作为强锚。P1 须在 **turn_review 记录本身** 补写（不依赖 `profile_signals`）：

| 补写位置 | 字段 | 来源 |
|----------|------|------|
| `turn_review` | `context_note_paths` | 本轮 scope 内实际可用的 note paths（Agent V2：与 coach review 一并 persist） |
| `turn_review.feedback` | `question_requires`、`gaps`、`coach_note` | Coach turn review（方案 B 下 extractor 的辅助证据，非最终 truth） |
| `turn_review` | `profile_signals` | **Legacy 兼容**；Agent V2 coach 恒为 `[]`，不作为 extractor 主输入 |

Session 结束 extraction 时，extractor 通过 transcript + turn review 列表关联每轮 evidence；`domain_anchor` 仍由代码从 session / turn 的 `context_note_paths` 写入。

### 写入逻辑

```text
new_weak_point(observation, session):
  if observation.scope == "universal":
    domain_anchor = {}
  else:
    topic_card = resolve_topic_card(
      plan=session.interview_plan,
      current_topic=session.interview_state.current_topic
               or observation.topic,
    )
    domain_anchor = resolve_domain_anchor(
      topic_card=topic_card,
      session=session,
      observation=observation,
    )
```

```text
resolve_domain_anchor(topic_card, session, observation):
  plan_topic = session.interview_state.current_topic or observation.topic or ""
  scope_path = resolve_scope_path(session)   # folder/tag source_value 或 note 父目录

  turn_context_paths = observation.context_note_paths   # 来自 turn_review.context_note_paths 或 observation 自带 paths
  if turn_context_paths and len(turn_context_paths) <= 3:
    context_note_paths = list(turn_context_paths)
  else:
    # turn context 过宽时不直接作为域锚，避免退化成 session 全量 paths
    context_note_paths = []

  if not context_note_paths and topic_card:
    # TopicCard fallback：小 TopicCard 全取，大 TopicCard deterministic 截断
    card_paths = topic_card.source_note_paths
    context_note_paths = list(card_paths) if len(card_paths) <= 3 else list(card_paths[:3])

  if not context_note_paths and not plan_topic and not scope_path:
    return {}   # 无 anchor，见降级规则

  return {
    plan_topic,
    context_note_paths,
    scope_path,
  }
```

说明：

- 优先使用足够窄的 turn 级 `context_note_paths`（≤3 条）；超过 3 条视为过宽，不直接作为域锚。
- turn context 过宽或缺失时，fallback 到 TopicCard paths（≤3 全取，>3 deterministic 取前 3）。
- `plan_topic` 与 `scope_path` 始终从系统态写入，不依赖 LLM 猜域。

### `resolve_topic_card` fallback

| 情况 | 行为 |
|------|------|
| `interview_state.current_topic` 命中 plan 中某 TopicCard | 使用该 card |
| 仅 `observation.topic` 与 plan 某 topic 名 fuzzy match | 使用匹配到的 card |
| plan 仅一个 topic | 使用唯一 card |
| 均无法匹配 | `domain_anchor` 仅含可解析的 `plan_topic` / `scope_path`；`context_note_paths` 为空时见降级规则 |

### 无 anchor 降级规则（domain 弱项）

`domain_anchor.context_note_paths` 为空且无法构成有效三元组时，**没有强域锚点**：

| 链路 | 行为 |
|------|------|
| **注入** | 默认**不跨 session 注入**；该弱项仅在本 session 内有效，或作为 legacy / low-confidence 保留 |
| **去重** | 仅当 text similarity **很高**（≥ 0.85）**且** `category` 相同才允许合并；否则 ADD |
| **禁止** | 不得用弱 text sim 把不同领域的相似表述合并；不得跨 domain 注入 |

无 anchor 时不写 embedding fallback——embedding 未实现前不作为已定能力。

### domain → universal 自动升级（P2）

在 `apply_profile_observations` 中，每次 ADD / UPDATE 后运行系统规则（**非 LLM UPDATE**）：

**白名单**：仅 `category ∈ { answer_structure, thinking_pattern, communication }` 可自动升级。  
**禁止**：`knowledge_gap` **永不**自动升 universal（「RAG 说不清」与「MCP 说不清」是不同域缺口）。

**触发条件**（与另一条 weak_point 比对，全部满足）：

1. 双方 `scope=domain`（或较早一条升级后合并）
2. 同 `category`，且在白名单内
3. text sim ≥ 0.85（或 containment 等价）
4. `domain_anchor.context_note_paths` **无 overlap**（确认真在不同域）
5. `scope_path` 不同，或 `plan_topic` fuzzy < 0.6

**动作**：

- 将较早条目 `scope` 升级为 `universal`
- `domain_anchor` 保留（审计）；可选记录 `also_seen_in: [{ plan_topic, context_note_paths, scope_path }]`
- 合并 `times_seen` / evidence 到一条 universal 条目（避免 profile 膨胀）

### 域 relevance 匹配（注入 / 去重共用）

```text
domain_relevant(weak, current_topic_card, session) -> strong | medium | none

  strong:
    weak.domain_anchor.context_note_paths
      ∩ current_topic_card.source_note_paths ≠ ∅

  medium:
    weak.domain_anchor.scope_path 与 current scope_path 前缀相同
    AND fuzzy(weak.domain_anchor.plan_topic, current_topic.name) ≥ 0.8

  none: 以上皆否
```

| 场景 | 注入 | 去重合并 |
|------|------|----------|
| `scope=universal` | 始终 | 仅与 universal 比 text sim |
| domain + **strong** | 注入 | text sim + strong 优先 |
| domain + **medium** | 注入 | text sim + medium + 同 category |
| domain + **none** | 不跨 session 注入 | 仅极高 text sim + 同 category |
| 无 `domain_anchor` | 不跨 session 注入 | 仅极高 text sim + 同 category |

### 字段 immutability（UPDATE 时）

| 字段 | CREATE | UPDATE / IMPROVE |
|------|--------|------------------|
| `scope` | LLM | **默认不变**；允许 P2 复现规则 / P4 迁移将 `domain` → `universal`；**禁止 LLM 改 scope** |
| `category` | LLM | P3 前一般不自动改 |
| `domain_anchor` | 代码 | **不变** |
| `topic` | LLM | **不变** |
| `source_note_paths` | 代码 | **不变** |

Stage 2 prompt（P3）须明确：`UPDATE` / `IMPROVE` 不得修改 `scope`、`domain_anchor`、`source_note_paths`、`topic`。

### Normalize 默认值（兼容旧 profile）

```text
scope              → "domain"（若缺失）
category           → "knowledge_gap"（若缺失）
domain_anchor      → {}（若缺失）
```

旧 profile 若仅有 `anchor_note_paths`（已废弃字段），normalize 时可迁移为：

```text
domain_anchor.context_note_paths = anchor_note_paths
domain_anchor.plan_topic = topic
domain_anchor.scope_path = ""   # 无法反推则留空
```

旧数据中 behavioral 弱项在 P4 迁移前会按 `domain` 行为处理——见发布策略。

---

## Due reviews

**Due review** = `sr.next_review <= today` 且 `improved == false` 的 weak point。

| 类型 | 注入规则 |
|------|----------|
| `universal` + due | 必注入，排前，带 SM-2 语义 |
| `domain` + due + relevant（strong/medium） | 注入，标 high priority |
| `domain` + due + 不 relevant | 不注入 |

注入文案继续隐藏 raw 数字背后的「评分」含义，仅给面试官 private guidance。

---

## 三条链路

### 链路 1：创建（写入）

**触发点（方案 B）**：Session 结束 **单层 extractor**（`extract_profile_observations` → `apply_profile_observations`）。Coach 每轮不写 profile。

**Extractor LLM 输入**：transcript、turn reviews（含 `question_requires` / `gaps` / `coach_note`）、existing profile compact、session context、interview plan。

**Extractor LLM 产出**：`scope`、`category`、`point`、`topic`、`planned_layer`、`evidence` 等。**不含** `domain_anchor` 或 LLM 匹配字段。

**代码产出**：`domain_anchor`、`source_note_paths`、`sr` 初始值、`source_session_ids`。

**Legacy 补充路径**：`commit_interview_memory` 可将 session 级 `record_signal` / observation draft 合成 synthetic review 交给同一 extractor；Agent V2 coach 不使用此路径。

**不改**：Session 进行中不更新 profile 文件；面试官每轮读同一份冻结 profile。

---

### 链路 2：去重

**现状（P0–P2）**：Stage 1 提取 observations → `apply_profile_observations()` → 确定性 `find_similar_profile_item()`（SequenceMatcher + topic gate）。

**P0–P2 目标**：scope-aware 确定性匹配 + domain_anchor 联合规则 + P2 scope 自动升级；不引入 Stage 2 LLM。

```text
Session-end extractor LLM（Stage 1）
  输入：transcript、turn reviews（feedback + context_note_paths）、existing profile（轻量 compact）
  输出：observations（含 scope、category；不含 domain_anchor）
  注：不再以 turn_review.profile_signals 为主输入；Agent V2 coach 该字段恒为空

Deterministic reconcile（P2 起，主路径）
  scope-aware find_similar_profile_item()
    universal：仅与 universal 条目比 text sim
    domain + strong/medium anchor：text sim + domain_relevant
    domain + none / 无 anchor：text sim ≥ 0.85 且 category 相同才合并

仍无匹配 → ADD
→ 运行 P2 domain→universal 自动升级（category 白名单）
```

**P3 后置 — Stage 2 LLM reconcile（可选增强）**

P0–P2 + deterministic scope-aware 匹配可解决约 80% 问题。Stage 2 引入 JSON 失败、误 UPDATE/IMPROVE、延迟等风险。**P3 明确后置**。

```text
Stage 2 LLM（P3，非 P0–P2 阻塞项）
  输入：numbered existing weak_points（含 scope、category、domain_anchor）
        + 本次 observations + session 上下文

Fallback（Stage 2 JSON 失败或未启用 P3）
  → 同上 Deterministic reconcile
```

**合并边界**：

- `scope=universal` 不与 `scope=domain` 合并。
- `scope=domain`：优先 `domain_relevant=strong` 的条目；其次 medium。
- `scope=domain` + none / 无 anchor：仅高 text sim + 同 category；不跨 session 注入。
- `topic` 字符串不作 hard gate。
- `knowledge_gap` 不因多域出现自动升 universal。

#### Stage 1 vs Stage 2 的 compact 分离（Stage 2 为 P3）

**Stage 1 — `compact_profile_for_extraction`（轻量，P1 起）**

`point`、`topic`、`planned_layer`、`improved`；可选 `scope`、`category`。**不需要** `domain_anchor`。

**Stage 2 — `compact_profile_for_reconcile`（完整，P3）**

```json
{
  "index": 0,
  "point": "...",
  "scope": "domain",
  "category": "knowledge_gap",
  "topic": "MCP 协议",
  "planned_layer": "...",
  "domain_anchor": {
    "plan_topic": "MCP 协议",
    "context_note_paths": ["个人/面试/agent面试/MCP.md"],
    "scope_path": "个人/面试/agent面试"
  },
  "improved": false
}
```

Session 上下文须含 `current_topic`、`current_topic_note_paths`、`scope_path`。

---

### 链路 3：注入（读取）

**入口**：`render_candidate_profile_context()`。

**不再使用**：`topic == current_topic` 过滤 weak points。

```text
A. Universal（scope=universal，未 improved）
   → 始终注入，cap 3–4；due 排前

B. Domain（scope=domain）
   → strong：context_note_paths ∩ current_topic.paths ≠ ∅ → 注入
   → medium：同 scope_path 前缀 + plan_topic fuzzy ≥ 0.8 → 注入
   → none / 无 domain_anchor：不跨 session 注入

C. Due priority → 合并进 A/B

D. topic_mastery → P5 可改为按 domain_anchor 聚合
```

---

## 分阶段落地（P0–P5）

| 阶段 | 内容 | 依赖 | 验收 |
|------|------|------|------|
| **P0** | Schema v2 + `normalize_interview_profile()`；`domain_anchor` 默认值 | 无 | 旧 profile 可读；`anchor_note_paths` 可迁移 |
| **P1** | turn_review 持久化 `context_note_paths` + coach feedback；session-end extractor 输出 `scope`/`category`；`new_weak_point()` 写 `domain_anchor` | P0 | 新 session 弱项三元组完整 |
| **P2** | 注入：scope 分流 + `domain_relevant`；scope-aware 去重；**domain→universal 自动升级**（category 白名单） | P0, P1 | 同 note 不同 topic 名可注入；跨域结构类弱项升 universal |
| **P3** | **后置** Stage 2 LLM reconcile | P0–P2；数据量足够 | 语义相同弱项 UPDATE 而非重复 ADD |
| **P4** | 数据迁移：behavioral → `scope=universal`；补全 `domain_anchor` | P0–P2 | 历史 behavioral 弱项跨 topic 注入 |
| **P5** | `topic_mastery` / strong_points 按 `domain_anchor` 或 scope 聚合 | P2 | 可选 |

**开发节奏**：P0 → P1 与 P2 可并行 → P4 → **P3 后置** → P5。

**P3 启用条件（建议）**：weak_points ≥ 30，或 deterministic 重复 ADD 率明显偏高。

---

## 发布策略与检查清单

### P1 / P2 必须同 release

**P1 已上线、P2 未上线，且已产出 `scope=universal` 弱项** → universal 仍被旧 topic 过滤 → **禁止**。

1. P1 + P2 **同一 release**；或
2. feature flag / `schema_version >= 2`：P2 未开启时不写入新格式弱项。

### 发布前检查

- [ ] turn_review 含 `context_note_paths`（独立于 `profile_signals`）
- [ ] Agent V2 coach：`profile_signals` 恒空；extractor 主路径不依赖 per-turn signals
- [ ] `normalize` 对缺失 `scope` / `domain_anchor` 有默认值
- [ ] `new_weak_point()` 仅代码写 `domain_anchor`；Stage 1 schema 无 anchor 字段
- [ ] `render_candidate_profile_context()` 用 `domain_relevant`，不用 `topic ==`
- [ ] 去重：strong/medium anchor + text sim；无 anchor 高 threshold + category gate
- [ ] P2：`knowledge_gap` 不自动升 universal；白名单三类可升
- [ ] P4 迁移脚本就绪

---

## 验收标准（整体）

1. 「听题不仔细」：`scope=universal`，换 topic 均注入。
2. 「RAG 融合策略只说加权」：下次 session topic 名不同，但 plan 仍含 `rag基础概念.md` 时注入（strong match）。
3. 「对比类缺乏结构化展开」在 MCP 与 RAG 两域独立出现 → P2 自动升 `universal`。
4. 「RAG 说不清」与「MCP 说不清」同为 `knowledge_gap` → **不**升 universal，各域分别注入。
5. 无 `domain_anchor` 的 domain 弱项：不跨 session 注入；不误合并不同域相似表述。
6. 旧 profile 无新字段：normalize 不 crash。

---

## 明确不做

- Topic 名称标准化表 / 别名映射
- LLM 产出匹配字段（`domain_key` 等）；三元组不足时再评估
- 注入时额外 LLM 调用做 relevance 判断
- Embedding 相似度作为注入/去重 fallback（未实现前）
- Universal 弱项绑定伪 topic「通用」
- Session 进行中更新 profile 文件
- `knowledge_gap` 自动 domain → universal

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-06-19 | 初稿：scope/category/anchor 字段契约、三链路、P0–P5 |
| 2026-06-19 | 修订：turn-context 优先；无 anchor 强保护；P3 后置 |
| 2026-06-19 | 定稿：`domain_anchor` 三元组；turn 补数据；P2 复现升级（category 白名单）；暂不落地 LLM 匹配字段 |
| 2026-06-21 | 方案 B：证据链改为 session-end 单层 extractor；coach 不再产 profile_signals；更新 turn 级补数据与三链路描述 |
