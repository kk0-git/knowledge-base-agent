# 记忆系统架构（v5）

2026-06-27

## 整体数据流



```
对话结束 → Extractor → Bridge → Commit → learner_model.json
                                              ↓
Agent 开口前 ← Injection ← 读取 ←────────────┘
                    ↑
/web/memory 页面 ← 人工审核（confirm/deny）
```

三条路径：**写**（对话→记忆）、**读**（记忆→注入 agent prompt）、**审**（人在 UI 确认/否决候选项）。

------

### 1. 提取：Extractor

每次面试 session 结束、或问答 session 归档时，触发一次提取。提取输入是完整对话 transcript + turn reviews（面试模式）或 assistant citations（问答模式），输出是 observations 列表。

LLM 被要求从对话中识别三类东西，每一类对应不同的存储目标：

**weak_point** — "用户哪里薄弱"。一条 weak_point 包含：`point`（一句话描述，如"无法解释 RRF 中 k 值的选择理由"）、`category`（提取时仍用旧标签 knowledge_gap / thinking_pattern 等，但写入前会转成 facet）、`scope_suggestion`（domain 还是 universal）、`evidence`（对话中的原文摘录）、`confidence`（high / medium / low）。confidence=low 的直接丢弃，不进后续流程。

**confusion_pair** — "用户把两个概念搞混了"。额外字段 `left`、`right`、`distinction`（如 left="语义搜索" right="关键词搜索" distinction="分不清两者的适用场景"）。存储和合并方式跟 weak_point 不同——用无序 tuple (left, right) 做合并键，区分"新混淆"和"已知混淆的再次暴露"。

**procedure** — "助手应该怎么跟用户交互"。字段 `procedure_key`（如 `answer_format.conclusion_first`）、`title`（"先给结论再解释"）、`steps`。来自对话中的交互偏好暴露，或面试 coach 输出的 communication suggestions。

**improvement** — "之前某条弱项这次表现好了"。不产生新条目，而是找到已有 weak_point 并标记 partial improvement（触发 SM-2 的 mark_weak_point_partial，间隔打折但不重置）。

**partial** — 介于 fail 和 pass 之间的中间状态。SM-2 间隔调整但不改变 ease_factor，不认为"真的会了"。

提取的结果还不是最终存储格式——下一步进 Bridge 转换。

------

### 2. 格式转换：Bridge

`observations_from_profile_extractor` 和 `observations_from_answer_extractor` 把 LLM 的松散输出转成 commit 引擎能消费的标准 observation dict。

转换做了几件事：

**scope 判定**。提取时 LLM 给出了 `scope_suggestion`（domain/universal），Bridge 这里最终拍板：如果 `scope_suggestion == "universal"` 或弱项本身 `scope == "universal"` → universal；否则 → domain。domain 弱项还需要解析 `domain_anchor`——这个话题关联了哪些笔记路径（`source_note_paths`）、属于哪个 plan_topic。

**domain_anchor 解析**。对于问答模式，anchor 从 assistant 的 citation paths 里解析——助手引用了哪些笔记，这些笔记的父目录就是 domain。对于面试模式，anchor 来自 session context 的 `source_note_paths`。

**explicit 标记透传**。问答模式有一条规则匹配的 fallback：如果消息里检测到"不懂""不会""不清楚""没搞懂"——这是一个 explicit admission，observation 带上 `explicit=True`，在后续评分里值 +2 分。

**procedure 从 communication suggestions 生成**。面试 coach 输出的 communication.suggestions 每条自动变成一条 procedure observation——`procedure_key` 用标题的规范化形式（如 `assistant_preference.结论先行`），默认 scope=universal。

**improvement → belief_id 查找**。`find_improvement_target` 在 profile view 的 weak_points 列表里找到匹配的已有弱项，把它的 id 填入 observation 的 `belief_id`，后续 commit 阶段直接找到对应条目更新。

**转换产出的标准 observation 格式**：每条 observation 有 `op`（propose_belief / propose_procedure / schedule_pass / schedule_retry / improvement / user_commit）、`source_kind`（interview / answer / review / user）、`confidence`，以及对应类型的专有字段。

------

### 3. 写入与打分：Commit

`commit_observations(model, observations)` 是所有记忆写入的唯一入口。

#### 3.1 入口过滤

`confidence == "low"` 的 observation 直接丢弃，计入 `filtered_low_count`，不进入任何后续步骤。这条规则确保低质量提取不污染记忆库。

#### 3.2 证据评分（Evidence Scoring）

每条 `propose_belief` observation 在决定 lifecycle 前，先过一个统一打分函数 `_score_evidence`：



```
score = w_source + w_confidence + w_corroboration + w_explicit

w_source:
  interview = 3   （面试环境更真实暴露弱项）
  answer    = 1   （问答环境噪音大）
  review    = 0   （复习卡片发现的不算分，只走 schedule）
  user      = 3   （用户手动确认等于最高权重）

w_confidence:
  high   = 2
  medium = 1

w_corroboration:
  找到已有相同弱项 = +2
  新发现           = 0

w_explicit:
  用户自承"我不懂X" = +2
  否               = 0

score >= T_active(4) → lifecycle = "active"
否则                  → lifecycle = "candidate"
```

关键设计：**同一条规则，不同输入自然产生不同结果**。面试 observation（source=3）+ 中等置信（confidence=1）= 4 过线 → active。问答 observation（source=1）+ 中等置信（confidence=1）= 2 不过线 → candidate。但如果问答是用户自承"我不懂"（explicit=+2），1+1+2=4 过线。如果同一条弱项在面试出现过、问答又命中（corroboration=+2），1+1+2=4 也过线。

**review 和 user 操作走快捷路径**：它们的 `target_lifecycle` 直接决定 lifecycle——因为 review 的 source 分值是 0，靠评分永远过不了线；user 操作是人的明确意图，不需要打分。

#### 3.3 合并（Merge）

新 observation 被构建成一条 preliminary belief，然后调用 `_find_merge_target` 查是否已有"同一个弱项"。

**标准 belief 合并键**：`scope` 相同 + `facet` 相同 + point 文本相似度 ≥ 0.72 + domain_anchor 兼容（共享 source_note_paths 或 scope_path 前缀匹配或 topic 文本相似 ≥ 0.8）。

**confusion_pair 合并键**：用无序 tuple `(left.lower(), right.lower())` 精确匹配——"Docker vs K8s"和"K8s vs Docker"命中同一个键。

**命中了** → `_merge_into_belief`：times_seen+1，追加 source_kinds / source_session_ids / evidence_refs。如果 target 是 candidate 而 new 是 active → 提升为 active。distinction 文本更新为最新版本。

**没命中** → 新增一行到 `learner_items`。

#### 3.4 冲突取代（Contradiction）

如果 observation 带了 `contradicts_belief_id`，且 new belief 的 lifecycle 是 active：

- 找到被指向的旧 belief
- 旧 belief → lifecycle 改为 `archived`
- 记一条 commitment：`action="superseded_by"`，记录谁取代了谁

candidate 级别的 belief 没有资格推翻 active——这隐含在 `new_belief["lifecycle"] == "active"` 的前置条件里。

#### 3.5 improvement 写路径

`_commit_improvement`：找到目标 belief（必须是 active），更新 SM-2 schedule（repetitions+1，标记 pass），同时设置 `improved=True` 和 `improved_at` 时间戳，记一条 commitment `action="improved"`。

**improved 标记的控制权**：按 MEMORY_WRITE_CONSOLIDATION 的设计，只有 interview session-end extractor 产出的 improvement observation 可以设置 `improved=true`。复习卡片只写 schedule（schedule_pass/schedule_retry），不改 improved。

#### 3.6 user_commit 写路径

`_commit_user_commit` 处理人在 /memory UI 的操作：

- `confirm_candidate` → candidate 升 active
- `deny_belief` → 任意状态（除 archived）→ archived
- `restore_archived` → archived 恢复到 active
- `confirm_procedure` / `set_procedure` → candidate procedure 升 active
- `deny_procedure` → procedure → archived

每条 user_commit 都记一条 commitment，带 note 和 at 时间戳——这是后续 outcome 指标的数据源。

#### 3.7 写后处理

commit 结束后：

- `canonical_revision` +1
- `updated_at` 更新为当前 UTC 时间
- `derived.stale = True` — 标记派生缓存失效，下次注入前会重建

------

### 4. 存储模型：learner_model.json



```json
{
  "schema_version": 5,
  "canonical_revision": 42,
  "learner_items": [ ... ],
  "assistant_items": [ ... ],
  "strong_points": [ ... ],
  "commitments": [ ... ],
  "derived": { ... },
  "legacy": { ... }
}
```

#### learner_items

每一条描述用户的一个薄弱点。核心字段：

- **lifecycle**: `candidate | active | archived`。决定是否注入 prompt 和是否出现在复习队列。
- **facet**: `knowledge | behavior`。knowledge 进 SM-2 复习调度；behavior 作为答题策略约束挂在复习卡片上，不进 SM-2。
- **scope**: `domain | universal`。universal 弱项无论当前话题是什么都会被注入；domain 弱项只在相关话题下出现。
- **kind**: `standard | confusion_pair`。confusion_pair 有额外的 left/right/distinction 字段，用无序 tuple 做合并键。
- **point**: 一句话描述。
- **domain_anchor**: `{topic, scope_path, source_note_paths, evidence_terms}` — 当 scope=domain 时，定义"这个弱项属于哪个领域"。用于注入时的相关性过滤（`domain_relevance_for_current`）。
- **sr**: SM-2 调度字段 — `{interval_days, ease_factor, repetitions, last_reviewed, last_outcome, next_review}`。仅 facet=knowledge 的条目有实际值。
- **evidence_refs**: 证据链 — 每条 `{source_kind, session_id, turn_id, at, summary}`。注入时只取最新一条的 summary 显示。
- **source_kinds**: 去重后的来源类型列表 — 如 `["interview", "answer"]` 表示在面试和问答中都暴露过。
- **tags**: 自由标签列表，不参与控制流。存原 v4 category 值（如 `"thinking_pattern"`）供展示用。
- **improved**: bool — 只有 interview extractor 可以设为 true。improved=true 的 belief 不注入（已经会了）。

#### assistant_items

每一条描述一个交互偏好。字段比 learner_items 简单：`procedure_key`、`title`、`description`、`steps`、`lifecycle`、`scope`。没有 facet、sr、kind——偏好不需要复习调度。

#### strong_points

冻结列表。只读保留，不再写入新条目。UI 仍可展示。

#### commitments

操作审计日志。每条记录 `{action, belief_id, target_id, at, note}`。action 类型：confirm_candidate、deny_belief、restore_archived、superseded_by、improved、confirm_procedure、deny_procedure。这是 outcome 指标的数据源。

#### derived

派生缓存。`stale` 标记 → 下次注入前触发 `rebuild_derived` 重建 domains 列表和 inject_blurbs。domains 是"每个 topic 下有多少 active 和 due 弱项"的汇总，inject_blurbs 是注入时用的简短领域摘要文本。

------

### 5. 注入：Injection

Agent 开口前，调用 `render_memory_context(model, reader, scope)` 生成一段 markdown 文本，拼到 system prompt 的末尾。

五个 reader 角色各有配置：

| reader      | beliefs | procedures | due        | derived   | commitment | 格式       |
| ----------- | ------- | ---------- | ---------- | --------- | ---------- | ---------- |
| interviewer | ≤5      | ≤2         | 合并进列表 | 1段+标题  | 是         | probe-rich |
| reviewer    | ≤5      | 0          | 合并进列表 | 1段+标题  | 是         | probe-rich |
| librarian   | ≤3      | ≤2         | 最多标2个  | 1段无标题 | 否         | 简洁       |
| coach       | 0       | 0          | —          | —         | —          | —          |

**注入流程**：

1. 从 learner_items 中筛出 `lifecycle=active and !improved` 的条目
2. 按 scope 过滤：universal 全进；domain 调用 `domain_relevance_for_current` 判断与当前话题的相关性（strong/medium/weak 都通过，"none" 丢弃）
3. 排序：due 的排前面，同 priority 按 ease_factor 升序（越不熟的越靠前）
4. 按 config.belief_budget 截断
5. 渲染：probe-rich 格式每条弱项带 probe hint、layer、latest evidence；简洁格式只显示 point + [due] 标记
6. 拼接 derived blurb → beliefs → commitments → procedures → Memory Use Boundary

**probe hint**：帮助 interviewer 决定怎么探这个弱项。`knowledge` facet → "probe definition and underlying mechanism"；`behavior` facet → "probe how they organize, frame, and express the answer"。弱项有显式 `probe_hint` 字段时用自己的。

**Memory Use Boundary**：注入文本末尾的约束——告诉 agent 不要让这些笔记主导对话，不要让用户察觉"你在按清单检查我的弱项"。

------

### 6. 更新路径总览

| 触发事件               | 提取方式                                | 写入类型                                              | lifecycle 决定                        | schedule 影响           |
| ---------------------- | --------------------------------------- | ----------------------------------------------------- | ------------------------------------- | ----------------------- |
| 面试 session 结束      | LLM extractor                           | weak_point / confusion_pair / procedure / improvement | evidence scoring                      | improvement → SM-2 pass |
| 问答 session 归档      | LLM extractor + rules fallback          | weak_point / confusion_pair / procedure               | evidence scoring（explicit+2 可过线） | 无                      |
| 复习卡片练习           | 前端 → commit_review_outcome            | schedule_pass / schedule_retry                        | 不改 lifecycle                        | SM-2 pass/fail          |
| 复习对话中提出新候选项 | commit_review_suggestions_as_candidates | propose_belief (force_new_candidate=True)             | 固定 candidate                        | 无                      |
| /memory 页面确认       | 前端 → user_commit                      | user_commit                                           | 直接改（人的明确意图）                | 无                      |

------

### 7. SM-2 调度：独立于 lifecycle

SM-2 只管"什么时候该复习"，不管"这弱项是不是真的"。两个维度的操作互不干扰：

- `update_weak_point`（答题失败）：ease_factor -= 0.15，最低 1.3；repetitions -= 1，最低 0；interval 重置为 1 天
- `mark_weak_point_partial`（部分回忆）：interval 打折（1→2, <3→3），ease 不变
- `mark_weak_point_improved`（答对）：repetitions += 1，ease_factor += 0.15；rep≤1 → interval=1，rep=2 → interval=7，更高 → min(60, max(14, interval × ease))

`next_review = today + interval_days`。复习队列按 `next_review <= today` 筛选，按 ease_factor 升序排列——越不熟的越先练。