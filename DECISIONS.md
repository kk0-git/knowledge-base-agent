# DECISIONS.md

量化实验 + 工程选型记录。面试用迭代历程讲故事，按组件查附录定位具体决策。

## 迭代历程

从 53 篇个人 Obsidian 笔记出发，目标是深入理解 RAG 管线每一环。

### 1. Dense baseline → 发现 chunk 质量是天花板

搭最简 pipeline：HeadingChunker → BGE-M3 → cosine → top-K。写 18 条 eval + Hit@K/MRR。

**实验**：旧大盘 chunk vs 新小段 chunk。8 改善、2 退化。退化**原因**是部分笔记内容跨段落分散，切小后无独立 heading，embedding 被相邻段落稀释。

**结论**：chunk 策略决定检索上限。同时发现 **note 级标注掩盖 chunk 级质量**——命中同一篇笔记就当对，不管打中定义段还是无关段。评测集应该引入chunk级。

→ 见附录 [Chunking](#chunking)

### 2. Dense 漏术语 → 引入 BM25

"FastAPI dependency_overrides" 在向量空间中召回差——技术术语（API 名、命令、缩写）是 dense 的短板。

**实验**：jieba vs 自定义 tokenizer。jieba 把 `dependency_overrides` 切成无意义碎片。

**结论**：选了 ASCII 正则整体保留 + 中文 2-gram。heading_path 在 BM25 文本中重复两次加权。

→ 见附录 [BM25](#bm25)

### 3. Dense + BM25 两路结果 → RRF 融合

用 RRF 融合后，评测出现 "单笔记霸屏" 警告：Top 5 全来自同一篇笔记。

**验证**：逐 chunk 核验实际内容——五个 chunk 对应五个不同的子主题，笔记本身覆盖全面。eval 标注的 参考经验.md 实际讲后端技术栈，不回答 "架构组件"。这是 **eval 标注错误，不是检索问题**。

**结论**：不做 MMR。Note 边界 ≠ 内容边界，用文件路径做多样性分组是把组织偶然性当成了检索质量问题。

→ 见附录 [RRF](#rrf)、[多样性](#多样性为什么不做-mmr)

### 4. Reranker 选型：小模型反而胜出

**实验**：bge-reranker-v2-m3（本地 568M）vs Qwen3-Reranker（API 4B）。

| | bge | Qwen3 |
|---|---|---|
| Hit@1 | 1.0 | 0.944 |
| "进程是什么" Rank 1 | 进程.md 定义段 (0.875) | 操作系统.md 大纲 (0.597) |

Qwen3 分数压缩在 0.54-0.60，无法区分 "包含关键词的大纲" 和 "定义本身"。

**原因**：bge 在 hard negative 训练中学会了 "回答 vs 提到" 的区分。Qwen3 通用 LLM 语义关联太强，判断标准是话题相关性而非答案精确性。

所以大模型优势在于复杂语义理解，对于定义类查询小模型更集中。

→ 见附录 [Reranker](#reranker)

### 5. CPU 嵌入瓶颈 → API + 增量索引

引入外部笔记后，全量重建 index 需 25 分钟（568M 参数 × 纯 CPU）。

- 全量重建 → 硅基流动免费 API（FP32 不变，秒级）
- 日常变更 → content hash 检测，只重嵌修改文件
- Config fingerprint → embedding_model + chunker_params 变更自动全量重建

→ 见附录 [Embedding](#embedding)、[增量索引](#增量索引)

### 6. 引入 PDF → 转换工具对比

需要异构数据验证 pipeline。同一份 PDF 三份转换对比。

原 PDF 文字渲染完全正确（"管理""并发"），但 PyMuPDF 提取后错误。这与 PDF 源文件质量无关——PyMuPDF 读的是 PDF 内部字体编码层（glyph index），需要通过 ToUnicode CMap 表映射回 Unicode。中文 PDF 的 CMap 经常被简化或缺失，映射出错导致 "管理" 被解释为 "操理"。MinerU 用 PaddleOCR 从页面图像识别像素，不看字体编码，不被错误映射影响。

|               | pymupdf                               | pymupdf4llm   | **MinerU**              |
| ------------- | ------------------------------------- | ------------- | ----------------------- |
| **字体编码**  | glyph→Unicode 映射错误，"管理"→"操理" | 同左          | **OCR 看像素，不受影响** |
| **页面分隔**  | `## Page N` 硬切正文                  | 无，段落连续  | 无，段落连续            |
| **章节标题**  | 混在正文中                            | `##` 但层级乱 | **`##` 层级清晰**        |
| **代码块**    | 内联纯文本，行号混入                  | 同左          | **```c 围栏** ✓          |

**总结**：PDF 显示和文本提取走两条路径。显示直接读字体形状数据（glyph #12847 在宋体里确实是 "管" 的曲线）→ 你看到正确汉字，全过程不需要 Unicode。文本提取才需要 ToUnicode CMap 做 glyph→Unicode 映射——这张表错了，显示正确的东西被提取成乱码。  
PyMuPDF 需要把 glyph index 映射回 Unicode——这个映射表（ToUnicode CMap）在中文 PDF 中经常不准确或被简化过。如果pdf生成器没有按标准写，或映射错了，"管理" 就被解释成 "操理"。MinerU 的 PaddleOCR 从页面图像识别像素，绕开了整条编码路径。  
**结论**：MinerU 胜出，作为离线批处理，转好 MD 放入 vault。

→ 见附录 [PDF → MD](#pdf--md)

### 7. PDF转换成功后跑增量索引
Qwen3-rerank 在当前 eval 上排序收益最大；local bge-reranker 不适合作为默认 reranker。

### 8. 测试：

#### 8.1 Rerank movement 分析

**目的**：不只看 Hit@K，而是观察 reranker 把哪些 chunk 上移/下移，判断它是在纠正排序，还是把语义相关但不能直接回答问题的 chunk 推上来。

**实验**：对比 hybrid 与 Qwen3-Reranker 的排序变化，生成 movement 报告。

**结论**：
- Qwen3 对复杂语义关联更敏感，但对定义类 query 容易过度联想。
- 例如 "进程是什么" 这类定义查询，Qwen3 会把教材中大纲式、背景式内容排到更前，而不是稳定选择定义段。
- 所以 Qwen3-rerank 适合作为可选模式，不作为默认策略；默认仍优先 hybrid，必要时再开启 rerank。

#### 8.2 RRF k 值实验

**目的**：验证论文常用 `k=60` 是否适合当前数据，而不是直接套默认值。

**实验结果**：

| k | Hit@1 | Hit@3 | Hit@5 | MRR |
|---:|---:|---:|---:|---:|
| 0 | 0.9259 | 0.9630 | 1.0000 | 0.9537 |
| 10 | 0.8519 | 0.9259 | 1.0000 | 0.9037 |
| 20+ | 0.8519 | 0.9259 | 0.9630 | 0.8963 |

**观察**：
- 当前小规模 eval 上 `k=0` 指标最高。
- 但 `k>=10` 后指标基本平坦，说明当前数据对 k 不敏感，且样本量不足以支撑改默认值。
- `k=0` 会强烈奖励 rank1，容易过拟合当前 27 条 query。

**结论**：继续使用 `k=60`。理由不是它在当前 eval 最优，而是它更稳健、符合 RRF 常用设定，且当前数据规模不足以证明需要偏离默认值。

#### 8.3 Chunker

**阶段一：代码块原子化**

**问题**：chunker 原先只按 heading/空行/列表边界切分，可能切进 fenced code block；另外代码块内部的 `#` 可能被误判为 heading。

**改动**：
- heading 切分时，代码块内部不识别 `#` 为标题。
- 二次切分时，如果候选切分点落在 ``` fenced code block 内部，优先切到代码块结束后；放不下则切到代码块开始前；代码块本身过大才允许切开，并标记 `oversized_code_block`。

**诊断结果**：

| 指标 | 修改前 | 修改后 |
|---|---:|---:|
| code boundary issues | 321 | 173 |
| non-overlap 硬边界 | 141 | 12 |
| 涉及笔记数 | 44 | 25 |

**结论**：代码块原子化有效，保留。

**阶段二：低价值 chunk 合并**

**问题**：个人笔记中存在大量分类标题，例如：

```markdown
### 报错
#### 环境
1. 环境未切换
```

这种写法在 Markdown/Obsidian 中是合规的，但 `### 报错` 如果单独作为 chunk，没有足够语义回答问题，会污染 top results。

同时还发现孤立 fence-only chunk，例如 chunk 文本只包含 `` ``` ``。

**改动**：
- `heading-only chunk -> 合并到后一个正常 chunk`
- `fence-only chunk -> 合并到前一个 chunk`
- 合并后重算 `chunk_id / start_line / end_line / char_count`

**统计结果**：

| 指标 | 合并前 | 合并后 |
|---|---:|---:|
| chunk_count | 3654 | 3368 |
| below_min_chunk_chars | 1268 | 969 |
| 0-100 字符 chunk | 487 | 180 |
| personal_note below_min | 894 | 622 |
| code boundary issues | 173 | 171 |

**评测结果**：

Hybrid 主指标完全持平：

| 指标 | code-atomic | low-value-merge |
|---|---:|---:|
| MRR | 0.9630 | 0.9630 |
| Hit@1 | 0.9259 | 0.9259 |
| Hit@3 | 1.0000 | 1.0000 |
| Hit@20 | 1.0000 | 1.0000 |
| avg_unique_notes@20 | 8.2963 | 8.3333 |
| avg_max_chunks_per_note@20 | 8.1111 | 8.0370 |

Qwen3 rerank 也基本持平：

| 指标 | code-atomic | low-value-merge |
|---|---:|---:|
| MRR | 0.9671 | 0.9671 |
| Hit@1 | 0.9630 | 0.9630 |
| Hit@20 | 1.0000 | 1.0000 |
| avg_unique_notes@20 | 7.8889 | 7.9259 |
| avg_max_chunks_per_note@20 | 8.3704 | 8.3333 |

逐 query 对比：
- Hybrid changed_count = 0
- Qwen3 changed_count = 0

Claude 相关 query 中，原先的孤立短 chunk：

```text
mini-claude-code启动记录.md:37-37
mini-claude-code启动记录.md:45-45
```

合并后变为：

```text
mini-claude-code启动记录.md:37-45
```

**结论**：低价值 chunk 合并没有损伤召回，且减少了短 chunk 噪音，保留。

#### 8.4 Overlap

当前暂不优先做 overlap sweep。

理由：
- overlap 只影响 overlong section 的二次切分，不影响所有 chunk。
- 代码块原子化和低价值 chunk 合并已经解决了更明确的结构问题。
- 当前 Hit@1/MRR 已较高，盲调 overlap 的收益不明确。

后续如果要做，建议只做轻量实验：

```text
overlap = 0 / 100 / 200 / 300
```

同时记录：
- Hit@K / MRR
- chunk_count
- avg_unique_notes@20
- avg_max_chunks_per_note@20
- index size
### 9. 待做
- chunk大小阈值的确定：比如 target=300/600/900/1200/1500，每个配置跑一次 compare，看 Hit@1 随 chunk 大小的变化曲线。
- 向量数据库的选择：索引算法
- 文档去重：对内容几乎一样的笔记，当前已经有content/exact hash
    - Exact 去重：内容 hash 相同的文件跳过
    - Near 去重：chunk embedding 的 cosine similarity > 0.95 的视为重复

### 10. 暂缓

- **Query 改写**：18 条 eval 都是完整中文，无口语化/歧义 —— 做了再议
- **Metadata 过滤**：tag 不全 + 200 chunks 无需过滤 —— 等异构来源够多再做
- **Chunk 级 eval**：note 级已满分，切细了没区分度 —— 先加 hard case
- 加入图片/表格的处理，pdf中的图和表，纯文本提取会丢失
    - 现状：没有图片密集的数据源，也没有"图上画了什么"类的检索需求。
    - 工程方案：表格：MinerU 其实支持表格识别（--enable_table），输出 HTML/Markdown 表格，你的 eval case 里还没涉及表格查询，所以没遇到问题；图片：用多模态模型（GPT-4V、Qwen-VL）对图片生成描述文字，把描述当 chunk 嵌入。
---

## 附录：按组件决策速查

### Chunking

**二次切分策略**：
- 最大 1500 chars，从后往前扫描半行 → 空行 → 列表边界 → 75% 位置兜底
- Overlap=200 chars，以行为维度，取 ≥200 chars 的完整行复制到下一 chunk 开头
- 代码块不特殊处理，视为普通文本

**收益**：每个 chunk 自带 heading 上下文，embedding 能捕捉主题归属，小节级精确匹配提升。但对内容分散/无独立 heading 的段落反而退化——大盘 chunk 有"无意中的平均语义优势"。

**暴露的问题**：note 级 eval 掩盖了 chunk 质量差异，需要 chunk 级指标。

**chunk_id 格式**：`note_path:start_line-end_line`

### BM25

- **分词**：ASCII_TOKEN_RE (`[a-zA-Z][a-zA-Z0-9_./:+-]*` + `\d+` + 中文序列) + 中文 2-gram + ≤8 字短语保留整体
- **heading 权重**：heading_path 出现 2 次
- **索引**：正向索引（forward index），`chunk_id → {term: tf}`，`term → df`，`document_lengths`，`avg_dl`
- **公式**：Okapi BM25，k1=1.5, b=0.75，Robertson-Sparck Jones IDF
- **为什么不调参数**：BM25 和 Dense 错误模式正交，RRF 自然互补，调参收益 < 融合收益

### RRF

- **k=60**：论文默认值，当前数据上验证过无需调整
- **无检索器权重**：不做 dense 和 bm25 的权重分配
- **去重**：同 chunk_id 在 RRF 时分数累加，`setdefault` 保底一份 chunk 对象
- **BM25 零结果**：空列表不报错，等价于纯 dense

### 多样性（为什么不做 MMR）

- Single-note 多 chunk 命中 ≠ 检索失败。这是因为笔记本身内容集中
- Note 边界是文件组织概念，不是内容边界
- Eval 的 `expected_note_coverage` 在某些 case 下是标注问题（参考经验.md 不回答架构组件）
- 需要多种语义角度时（exploratory query），可以用 MMR 选择不相似的 chunk

### Reranker

| 决策项 | 选择 | 原因 |
|---|---|---|
| 模型 | bge-reranker-v2-m3 (local) | 当前规模下精准度 > 大模型语义关联能力 |
| 架构 | Cross-encoder | Bi-encoder 粗排 + Cross-encoder 精排 = 标准 RAG 管线 |
| 截断 | max_length=512 | Cross-encoder 注意力矩阵 O((q+d)²)，长文本在 CPU 不可行 |
| 候选池 | RRF top-50 → reranker → top-K | 50 个候选在精度和成本间平衡 |
| 为什么不用 Qwen3 | 分数压缩严重，定义类查询区分度不如 bge | Qwen3 赢在复杂语义理解，当前场景未触及 |

### Embedding

- **相似度**：L2 归一化后内积 = cosine similarity
- **为什么 cosine**：BGE-M3 训练 loss 基于 cosine，换欧氏距离相当于换尺子量另一套标准造的空间
- **索引算法**：无（numpy 矩阵乘法），200 chunks 下暴力检索 < 1ms
- **API vs 本地**：全量重建用硅基流动免费 API（秒级），增量嵌入本地 CPU（30s 可接受）

### PDF → MD

| 决策项 | 选择 | 原因 |
|---|---|---|
| 解析器 | MinerU | 中文 OCR 最优，代码块输出 ```c 格式，heading 层级清晰 |
| 为什么不选 PyMuPDF | 依赖 PDF 字体编码层（ToUnicode CMap），中文 glyph→Unicode 映射经常出错 | MinerU OCR 看像素，不受此影响 |
| 集成方式 | 离线批处理 | 不在 RAG pipeline 内，转完 MD 后放入 vault |
| 成本 | 需 GPU | 个人轻薄本不可行，需租云 GPU 或使用 Agent 云端版 |

### Eval

- **框架**：Hit@K + MRR + note 级 diversity 指标
- **已知局限**：note 级标注掩盖 chunk 质量差异，需要新增 chunk 级标注
- **compare 脚本**：一键跑 4-5 种策略，自动输出对比表 + per-query delta
- **Eval 质量比算法更重要**：发现 参考经验.md 标注问题后删除 → 避免了错误的优化方向

### 增量索引

- **变更检测**：content hash（md5，包一层函数不绑死算法），索引存 `files` 元数据
- **Config fingerprint**：embedding_model + chunker_params + embedding_text_version 存入索引 → 配置变更自动全量重建
- **组件分工**：Vector Store 增量 upsert，BM25 每次全量重建（1s 可接受）
- **改名/移动**：不管。操作频率极低，误判成本（白嵌一篇 30s）不值得检测逻辑的复杂度
- **md5 替代方案**：xxhash 更快但不急（53 篇 10ms vs 1ms），函数封装好后续可换

### Query 改写（已讨论，未落地）

- 策略分三步：LLM query_plan → hybrid-rewrite → HyDE（只参与 dense）
- 当前不需做的理由：18 个 eval query 已是完整中文，无口语化/歧义/多子问题
- 真正需要的场景：用户输入不规范、术语不一致、多跳推理
- 需要专门的 rewrite eval，不能用当前 18 条规范 query 判断价值

### Metadata 过滤（已讨论，暂缓）

- 当前不做：tag 覆盖不全 → 过滤有漏切风险；200 chunks 下性能收益为零
- 需要的数据基础：PDF/网页等异构来源的 metadata 需要解析器提前治理
- 过滤做成建议式而非强制式：用户决定是否过滤，机器只给提示
