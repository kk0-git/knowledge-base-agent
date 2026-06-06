# RAG Compare Dependency-Aware

## Config

- index: `rag-index\bge-m3-v2.json`
- bm25_index: `rag-index\bge-m3-v2.bm25.json`
- eval: `./eval/rag_eval.json`
- model: `BAAI/bge-m3`
- top_k: `1`
- hit_ks: `1`
- requested strategies: `bm25,dashscope-rerank`
- successful strategies: `bm25`
- baseline: `bm25`

## Failed Strategies

| strategy | stage | dependencies | error |
|---|---|---|---|
| dashscope-rerank | dependency | dashscope_reranker, vector_store, bm25_index, embedder | dashscope_reranker: ValueError: DASHSCOPE_API_KEY is required. Set it as an environment variable or pass api_key= to DashScopeReranker(). |

## Metrics

| strategy | mrr | hit@1 | avg_expected_note_coverage@1 | avg_unique_notes@1 | avg_max_chunks_per_note@1 |
|---|---:|---:|---:|---:|---:|
| bm25 | 0.8889 | 0.8889 | 0.7824 | 1.0000 | 1.0000 |

## Per Query

| query | bm25 rank | bm25 cov@1 | bm25 top |
|---|---|---|---|
| 进程是什么 | None | 0.0000 | 个人/面试/agent面试/RAG/rag各阶段记录.md:34-70 |
| 进程和线程有什么区别 | 1 | 1.0000 | 个人/面试/基础/进程.md:57-63 |
| PCB 进程控制块包含什么信息 | 1 | 1.0000 | 个人/面试/基础/进程.md:22-28 |
| FastAPI CORS 跨域怎么配置 | 1 | 1.0000 | 全栈ai/技术栈/fastapi.md:146-206 |
| FastAPI dependency_overrides 怎么替换依赖做测试 | 1 | 1.0000 | 全栈ai/技术栈/fastapi.md:113-154 |
| SQLAlchemy async 和 PostgreSQL 怎么配合 | 1 | 1.0000 | 全栈ai/技术栈/postgreSQL.md:7-76 |
| Redis Stream 任务队列怎么消费事件 | None | 0.0000 | 全栈ai/技术栈/全栈路径.md:203-317 |
| Claude Code 代理怎么配置 | 1 | 1.0000 | 软件与系统安装配置/软件安装与配置/ClaudeCode安装.md:1-10 |
| Claude API key 在哪里配置和测试 | 1 | 0.3333 | 软件与系统安装配置/软件安装与配置/ClaudeCode安装.md:11-25 |
| DNS 查询过程递归和迭代有什么区别 | 1 | 1.0000 | 个人/面试/基础/DNS查询.md:2-9 |
| HTTP 强缓存和协商缓存有什么区别 | 1 | 1.0000 | 个人/面试/基础/HTTP缓存.md:1-10 |
| git push 失败怎么设置代理 | 1 | 1.0000 | 软件与工具使用/git/git命令.md:1-38 |
| Docker 怎么启动容器和查看日志 | 1 | 1.0000 | 软件与工具使用/docker/使用命令.md:54-87 |
| Linux 进程管理和系统调用 | 1 | 1.0000 | 个人/面试/基础/进程.md:1-21 |
| 从输入URL到页面渲染，DNS解析、TCP握手和HTTP缓存各在哪个阶段起作用 | 1 | 0.2500 | 个人/面试/基础/计网.md:28-50 |
| 构建LLM Agent需要考虑哪些架构组件和推理范式 | 1 | 1.0000 | 个人/面试/agent面试/agent-architecture.md:12-17 |
| git push 怎么配置 SSH 密钥和私人令牌免密认证 | 1 | 1.0000 | 软件与工具使用/git/gitee/记录.md:1-28 |
| 面试可能会问哪些计算机网络和操作系统的问题 | 1 | 0.5000 | 个人/面试/基础/操作系统.md:1-8 |

## Changes Vs Baseline
