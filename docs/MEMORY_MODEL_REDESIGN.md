# Memory Model Redesign (v5 提案)

状态空间重抽象设计稿。**仅设计，未实现**。目标：在不动"中间层"（单 canonical model、单 commit 引擎、Evidence/Memory/Derived 三层分离、candidate 闸门、铁律）的前提下，收敛三个模式在**触发时机、lifecycle 默认、注入**上的特例与过度细分。

前置：[`MEMORY_CHARTER.md`](MEMORY_CHARTER.md)、[`MEMORY_WRITE_CONSOLIDATION.md`](MEMORY_WRITE_CONSOLIDATION.md)、[`MEMORY_SCHEMA_MIGRATION.md`](MEMORY_SCHEMA_MIGRATION.md)、[`MEMORY_INJECTION.md`](MEMORY_INJECTION.md)。

本稿一旦评审通过，将回写 Charter §3 / §5 / §6 / §7，并触发 v4 → v5 迁移。

---

## 0. 问题陈述

当前设计的复杂度分两层：

- **中间层（保留）**：单 model + 单 commit 引擎 + 三层分离 + candidate 闸门 + 铁律。一致、可推理，是系统支柱。
- **边缘层（本稿目标）**：三模式各自发明触发时机、各自的 lifecycle 默认（interview 默认 active、answer 默认 candidate）、各自的注入 render 函数；Belief 下 `category` 四分类 LLM 分不稳；`strong_points` 无行为消费者。

边缘层的特例不是从中间层原则推导出来的，而是逐模式堆积的。本稿用三条原则把它收敛。

---

## 1. 设计原则

### P1. 统一规则，而非统一结果

模式之间的差异（面试更真实暴露 weak、问答噪音大）应体现在**证据权重**里，而不是体现在"两套 lifecycle 默认值"里。一条"凭证据强度决定可信度"的规则，对不同输入自然产生不同结果，不给任何模式增加确认成本。

### P2. 一个状态维度，只有"下游会分支"才配存在

每个字段都要能回答"谁消费它、消费时是否真的 if/else 分支"。不分支的维度要么塌成粗粒度，要么降级为不参与控制流的元数据。

### P3. LLM 分类不稳的东西，绝不当分支键

误分类一个**不参与控制流的描述 tag** 没有代价；误分类一个**分支键**会直接改变 agent 行为。凡 LLM 判别不稳的细分（如思维 vs 沟通 vs 答题结构），要么塌成可靠二分，要么存为 free-text tag。

---

## 2. 统一实体：MemoryItem（逻辑统一，物理双 list）

把现有 `beliefs[]` 与 `procedures[]` 两套并行类型，统一为**逻辑上的单一 MemoryItem**，用 `about` 轴区分。两者本就共享 lifecycle / evidence_refs / source_kinds / 合并逻辑，逻辑合一去重。

**实现形态（已定）**：物理上保留两个 list —— `learner_items[]`（原 beliefs）与 `assistant_items[]`（原 procedures）。`about` 由所在 list 隐含决定，不在条目内重复存。理由：

- 注入、复习、UI 大多按 about 分别取用，分表读取无需先过滤。
- 迁移最小：v4 的 `beliefs[]`/`procedures[]` 直接改名 + 字段投影，不需要合并再拆。
- 共享逻辑（lifecycle 流转、evidence 合并、评分）写成**与 list 无关的纯函数**，对两个 list 复用，避免"双表 = 双份逻辑"。

```text
MemoryItem
  id
  about:      learner | assistant        # learner=对用户的判断(原 Belief)；assistant=交互偏好(原 Procedure)
  kind:       point | confusion_pair      # 记录形状（仅 about=learner 用 confusion_pair）
  facet:      knowledge | behavior        # 取代原 4-category（见 §4）
  scope:      universal | domain
  lifecycle:  candidate | active | archived
  schedule:   null | { sr... }            # 仅 about=learner & facet=knowledge & 可复习 才有
  point / left / right / distinction       # 按 kind 取用
  steps / item_key                         # 仅 about=assistant（原 procedure_key/steps）
  domain_anchor                            # 仅 scope=domain
  source_kinds[]                           # 评分输入，非到处特判
  evidence_refs[]                          # 注入只取最近一条
  tags[]                                   # 自由描述，**不参与控制流**（容纳 LLM 不稳细分）
  first_seen / last_seen / times_seen
  created_revision
```

> 共享逻辑与 list 解耦：`promote/merge/score(item, ...)` 等只接收 item，不关心来自哪个 list；调用方按 about 选 list 后传入。

---

## 3. 维度审计（留/砍依据）

| 维度 | 消费者（真实代码） | 是否真分支 | 结论 |
|---|---|---|---|
| `lifecycle` | 注入闸门 `is_injectable_weak_point`、`/memory` UI | 是 | 保留，核心 |
| `scope` | 注入相关性过滤 `domain_relevance_for_current` | 是 | 保留 |
| `kind` | 渲染 `_format_belief_for_prompt`、合并键 `_confusion_key` | 是 | 保留 |
| `about` (learner/assistant) | 注入分不同 section（背景 belief vs 交互偏好） | 是 | 保留为一条轴 |
| `facet` (knowledge/behavior) | `review_practice` 的 knowledge vs strategy 分支 | 仅 2 路 | 4-category 塌成 2 |
| `source_kind` | 升 active 评分 | 是（作为评分输入） | 保留为评分输入 |
| `schedule` | review 调度 | 是 | 保留，内嵌 |
| `commitment` | 审计 / lifecycle 变更 | 否（只读日志） | 保留，成本低 |
| `strong_points` | **无行为消费者** | 否 | 砍（见 §6） |
| `tags` | 无（描述用） | 否（**设计上禁止分支**） | 新增，容纳不稳细分 |

---

## 4. facet：四类塌成两类

现状 `category = knowledge_gap | answer_structure | thinking_pattern | communication`。但下游唯一真实分支是 `review_practice` 的 **knowledge vs strategy** 两路（`STRATEGY_CATEGORIES`）。后三者边界模糊、LLM 分不稳。

```text
facet = knowledge | behavior

knowledge  ← knowledge_gap
behavior   ← answer_structure | thinking_pattern | communication
```

- `facet` 是分支键：决定是否可进入卡片复习（knowledge）vs 作为答题策略约束（behavior）。
- 原四分类的语义如需保留，存进 `tags`（如 `tag: "answer_structure"`），**任何逻辑不得读 tags 做分支**。误分类只影响展示，不改行为（P3）。

### 4.1 塌缩前必须清掉的两处隐式分支（代码审计结果）

经代码审计，`behavior` 三子类当前**有两处真实下游分支**，必须先收敛，否则塌缩会丢行为：

| 位置 | 现状分支 | 收敛方案 |
|---|---|---|
| `review_practice.py` fallback 出题（`_fallback_*`，仅 LLM 未产出 question_blocks 时触发） | `answer_structure`/`thinking_pattern`/`communication` 各一套兜底 prompt 文案，`block_type` 也分三种 | 三套兜底模板**合并为单一 behavior 兜底模板**，`block_type=behavior`。主路径（LLM 出题）本就按 weak point 原文定制，兜底是 rare path，合并损失可接受 |
| `interview_profile.py:normalize_scope` | `category == "communication"` → 默认 `scope=universal`；其余 → `domain` | 改为 `facet=behavior` 默认 `scope=domain`（多数情形），`scope_suggestion=universal` 时由 LLM/评分显式覆盖。communication 失去"自动 universal"默认，属可接受的小行为变化 |

> 结论：`facet` 二分**够用**，但落地顺序是「先把这两处分支改掉 → 再删 4-category」。在两处清掉前不得塌缩。

---

## 5. 证据评分 → lifecycle（取代模式特判）

删除"interview 默认 active / answer 默认 candidate"双轨。新 observation 的 lifecycle 由**统一评分**决定：

```text
score = w_source + w_confidence + w_corroboration + w_explicit

w_source:        面试 turn_review(3) > 面试 transcript(2) > 问答 transcript(1) > 复查 card(0，card 只走 schedule)
w_confidence:    high(2) | medium(1) | low(→不入库)
w_corroboration: 多源/重复命中 merge 键 (+2) | 单源 (0)
w_explicit:      用户明确自陈"不懂/不会 X" (+2) | 否 (0)

lifecycle:
  score >= T_active   → active
  否则                → candidate
  low confidence       → 不入库（仅记 extraction 诊断）
```

`T_active` 标定（已定）：设到"面试 turn_review + medium 即可过线"，使**面试行为与今天一致**（自动升 active，不加确认成本）；而"问答单源 medium"过不了线，停在 candidate。

按上表权重，该标定等价于 `T_active = 4`：
- 面试 turn_review(3) + medium(1) = 4 → 过线 ✓
- 问答 transcript(1) + medium(1) = 2 → 不过线，停 candidate ✓
- 问答 transcript(1) + high(2) + explicit(2) = 5 → 过线（高置信自陈）✓
- 任意 + 多源 corroboration(+2) 可把单源 candidate 推过线 ✓

`T_active` 与四项权重为可调参数，初值如上；将来按 §10 outcome 指标微调。

效果对照（与现行铁律一致，但来自一条规则而非特判）：

| 输入 | 今天的特判 | 本稿评分结果 |
|---|---|---|
| 面试 session-end weak | 默认 active（legacy） | 高 w_source → active |
| 问答普通 weak | candidate | 低分 → candidate |
| 问答高置信自陈 | high → active | w_explicit+confidence → active |
| 面试 candidate + 问答同 weak | 多源 merge → active | w_corroboration → active |
| 复查新发现 | candidate | card 不评分正文，仅 candidate propose |
| 用户确认 | active | 视为 explicit 拉满 / 直接 user_commit |

> 用户确认（`user_commit`）仍是升 active 的一条路径，但不是强制闸门。面试不因"统一"而需要确认。

---

## 6. strong_points：冻结

`strong_points` 当前没有"改变 agent 行为"的消费者（无注入 reader 用它塑造追问），只服务能力画像，而现阶段不做画像/成长分析。

- 停止写入新 strong_point。
- **原地冻结，不迁出**（已定）：`strong_points[]` 仍留在 `learner_model.json` 顶层，只读保留既有数据，不抽到独立文件。理由：不值得为一个停写的结构新增文件与读写路径；evidence 本就在 transcript，将来需要"成长视图"时从 evidence 重建即可。
- v5 envelope 保留 `strong_points[]` 字段（见 §9）。

---

## 7. 统一触发：封口 + 增量 checkpoint

三模式共用同一套"段封口（seal）→ 增量提取"机制，差异只在"边界由谁定义"。取代现在 interview/answer/review 各写一套触发。

### 7.1 封口条件

```text
seal(segment) when:
  - explicit_boundary   显式：结束面试 / 新建问答 / 复习对话结束
  - idle_timeout        段内最后一条 message 距今 >= 60min（已定）
  - activity_sweep      每次 POST API 请求时顺带扫描 idle 且未提取的段（已定）
```

触发方式（已定）：
- `idle_timeout` 阈值 = **60 分钟**。
- `activity_sweep` **不做独立定时器/心跳**，挂在**每次 POST API 请求**上：进入任意写类请求时，顺带检查是否有 idle 超时且未封口的段，有则封口提取。无后台进程、无新调度组件。
- 关标签页 / 刷新 / 崩溃：**不直接触发**，只靠已有 message 持久化 + 下次任意 POST 时的 `activity_sweep` 兜底。

> 代价：纯只读会话（一直只 GET、从不 POST）不会触发 sweep，那段 memory 会等到下次任何 POST 才结算。可接受——本系统的核心交互（面试/问答/复习）都是 POST。

### 7.2 增量 checkpoint（Evidence 侧，非 canonical）

checkpoint 从"session 级提取过/没提取过"升级为**增量进度**：

```json
{
  "memory_extraction": {
    "status": "completed",
    "trigger": "explicit_boundary | idle_timeout | activity_sweep",
    "last_message_id": "msg_xxx",
    "last_message_index": 8,
    "evidence_hash": "...",
    "commit_revision": 12,
    "updated_at": "ISO8601"
  }
}
```

- 提取输入 = `last_message_id` 之后的新 message（增量），不是整段重跑。
- `last_message_id` 为稳定边界；`last_message_index` 仅调试/兼容。
- 找不到 `last_message_id`（极少）→ 保守全量重跑，带旧 `evidence_hash` 去重，记 warning。
- 增量片段不足 1 user + 1 assistant → 不提取。
- 幂等：同一 evidence_hash 不重复提交。

### 7.3 触发与评分的衔接

封口后产出 Observation[]，仍走 §5 评分决定 lifecycle。即"何时提取"（§7）与"提取出来算多可信"（§5）是两个解耦的问题。

---

## 8. 注入收敛：一个函数 + per-reader 配置表

现状 `render_librarian_memory_context` / `render_interviewer_memory_context` 已分叉，再加 reviewer 即第三份，行为会漂移。收成**单一注入函数 + 配置表**：

```text
render_memory_context(model, reader, scope) :
  config = READER_CONFIG[reader]
  选 active MemoryItem（about=learner）按 scope 过滤 + 评分排序
  截到 config.belief_budget；due 在列表内打标
  选 active MemoryItem（about=assistant）截到 config.procedure_budget
  derived blurb 若 config.derived
  commitments 若 config.commitment
  按 config.boundary_text 收尾
```

| reader | belief | procedure | due | derived | commitment |
|---|---|---|---|---|---|
| interviewer | ≤5 | ≤2 | 合并入 belief | 1 段 | 是 |
| reviewer | ≤5 | — | 合并入 belief | 1 段 | 是 |
| librarian | ≤3 | ≤2 | ≤2 标记 | 1 段(短) | — |
| coach | 0 | 0 | — | — | — |

预算数字仍是初值；§10 的 outcome 指标用于将来校准，不在本稿拍定。

---

## 9. v4 → v5 迁移

| v4 | v5 |
|---|---|
| `beliefs[]` | `learner_items[]`（list 改名 + 字段投影，about 隐含） |
| `procedures[]` | `assistant_items[]`（list 改名 + 字段投影，about 隐含） |
| `category=knowledge_gap` | `facet=knowledge` |
| `category=answer_structure\|thinking_pattern\|communication` | `facet=behavior` + `tags+=[原值]` |
| `strong_points[]` | **原地保留**，只读冻结，envelope 仍含此字段（不迁出、不并入 items） |
| `lifecycle/scope/kind/sr/evidence_refs/domain_anchor` | 原样 |
| 模式默认 lifecycle | 删除；迁移时按现状保留既有 lifecycle，新写入走 §5 评分 |

v5 顶层 envelope（相对 v4 的变化）：

```json
{
  "schema_version": 5,
  "learner_items": [],      // 原 beliefs
  "assistant_items": [],    // 原 procedures
  "strong_points": [],      // 冻结，只读
  "commitments": [],
  "derived": { },
  "legacy": { }
}
```

- 迁移一次性：v4 读入 → 投影 v5 → 写 `learner_model.json`（`schema_version: 5`）。
- 既有条目的 lifecycle 不重判（避免迁移翻动用户已有 active/archived）；评分仅作用于**迁移后的新 observation**。

---

## 10. 配套：先加一个 outcome 指标

P5 的 metrics 数的是"库存"，不是"注入是否改变了 agent 行为"。在加任何新 memory 功能前，建议先加一个 outcome 信号，用于将来校准预算与评分阈值：

```text
injected_belief_hit:   本轮面试注入的 belief，是否被实际 probe 命中
candidate_confirm_rate: candidate 后续被用户确认 vs 否认 的比例
mis_merge_flag:        用户否认"这两条不是一回事"→ 误合并计数
```

无 outcome 反馈，复杂度只能增不能减。

---

## 11. 非目标 / 暂不做

- `dormant` 四态：仍延后。
- strong_points 重建为画像：延后到有"成长视图"需求。
- 预算数字横向实验：等 §10 指标落地后再做。

---

## 12. 评审决定（已确认）

| # | 问题 | 决定 |
|---|---|---|
| 1 | MemoryItem 实现形态 | **物理双 list**（`learner_items`/`assistant_items`），逻辑统一、共享逻辑做成 list 无关纯函数（§2） |
| 2 | `T_active` 与权重标定 | **接受**"面试 turn_review+medium 即过线"，等价 `T_active=4`（§5） |
| 3 | `facet` 二分是否够用 | **够用**，但审计发现 `behavior` 下有两处隐式分支（fallback 出题、scope 默认），塌缩前必须先清掉（§4.1） |
| 4 | strong_points 去留 | **原地冻结，不迁出**（§6/§9） |
| 5 | idle 阈值 / sweep 频率 | **60min + 每次 POST API 扫描，不做定时心跳**（§7.1） |

落地顺序提醒（来自决定 #3）：必须「先收敛 §4.1 两处分支 → 再删 4-category → 再做 v5 迁移」。

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-06-26 | v0.1：状态空间重抽象提案（MemoryItem、facet 二分、证据评分、统一触发、注入收敛、strong_points 冻结） |
| 2026-06-26 | v0.2：评审决定回填 —— 物理双 list、T_active=4 标定、strong_points 原地冻结、idle 60min+POST sweep；审计出 §4.1 两处 behavior 隐式分支（fallback 出题 / scope 默认），明确塌缩前置顺序 |
