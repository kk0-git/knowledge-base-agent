# RAG Compare Dependency-Aware

## Config

- index: `rag-index\mixed-siliconflow-bge-m3-code-atomic.json`
- bm25_index: `rag-index\mixed-siliconflow-bge-m3-code-atomic.bm25.json`
- eval: `./eval/rag_eval.json`
- model: `BAAI/bge-m3`
- embedding_provider: `openai_compatible`
- embed_batch_size: `32`
- max_seq_length: `None`
- top_k: `20`
- hit_ks: `1,3,5,10,20`
- requested strategies: `hybrid,local-rerank,dashscope-rerank`
- successful strategies: `hybrid,dashscope-rerank`
- baseline: `hybrid`

## Failed Strategies

| strategy | stage | dependencies | error |
|---|---|---|---|
| local-rerank | evaluation | vector_store, bm25_index, embedder, local_reranker | RuntimeError: Embedding request failed: <urlopen error [Errno 11001] getaddrinfo failed> |

## Metrics

| strategy | mrr | hit@1 | hit@3 | hit@5 | hit@10 | hit@20 | avg_expected_note_coverage@1 | avg_expected_note_coverage@3 | avg_expected_note_coverage@5 | avg_expected_note_coverage@10 | avg_expected_note_coverage@20 | avg_unique_notes@1 | avg_max_chunks_per_note@1 | avg_unique_notes@3 | avg_max_chunks_per_note@3 | avg_unique_notes@5 | avg_max_chunks_per_note@5 | avg_unique_notes@10 | avg_max_chunks_per_note@10 | avg_unique_notes@20 | avg_max_chunks_per_note@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 0.9630 | 0.9259 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.7994 | 0.8858 | 0.9630 | 0.9907 | 1.0000 | 1.0000 | 1.0000 | 1.8889 | 2.1111 | 2.7037 | 3.0370 | 4.3704 | 5.1852 | 8.2963 | 8.1111 |
| dashscope-rerank | 0.9671 | 0.9630 | 0.9630 | 0.9630 | 1.0000 | 1.0000 | 0.8179 | 0.9259 | 0.9537 | 0.9907 | 1.0000 | 1.0000 | 1.0000 | 2.0000 | 2.0000 | 2.6296 | 3.0741 | 4.7407 | 4.9630 | 7.8889 | 8.3704 |

## Per Query

| query | hybrid rank | hybrid cov@20 | hybrid top | dashscope-rerank rank | dashscope-rerank cov@20 | dashscope-rerank top |
|---|---|---|---|---|---|---|
| 进程是什么 | 2 | 1.0000 | imported_docs/textbooks-mineru-agent/操作系统导论/04.md:89-106 | 9 | 1.0000 | imported_docs/textbooks-mineru-agent/操作系统导论/04.md:36-47 |
| 进程和线程有什么区别 | 1 | 1.0000 | 个人/面试/基础/进程.md:57-63 | 1 | 1.0000 | 个人/面试/基础/进程.md:57-63 |
| PCB 进程控制块包含什么信息 | 1 | 1.0000 | 个人/面试/基础/进程.md:22-28 | 1 | 1.0000 | 个人/面试/基础/进程.md:22-28 |
| FastAPI CORS 跨域怎么配置 | 1 | 1.0000 | docs/fastapi/tutorial/cors.md:35-67 | 1 | 1.0000 | docs/fastapi/tutorial/cors.md:35-67 |
| FastAPI dependency_overrides 怎么替换依赖做测试 | 1 | 1.0000 | docs/fastapi/advanced/testing-dependencies.md:23-53 | 1 | 1.0000 | docs/fastapi/advanced/testing-dependencies.md:23-53 |
| SQLAlchemy async 和 PostgreSQL 怎么配合 | 1 | 1.0000 | 全栈ai/技术栈/postgreSQL.md:1-76 | 1 | 1.0000 | 全栈ai/技术栈/postgreSQL.md:1-76 |
| Redis Stream 任务队列怎么消费事件 | 1 | 1.0000 | 全栈ai/技术栈/redis.md:253-306 | 1 | 1.0000 | 全栈ai/技术栈/postgreSQL.md:205-336 |
| Claude Code 代理怎么配置 | 1 | 1.0000 | 软件与系统安装配置/软件安装与配置/ClaudeCode安装.md:1-25 | 1 | 1.0000 | 软件与系统安装配置/软件安装与配置/ClaudeCode安装.md:1-25 |
| Claude API key 在哪里配置和测试 | 1 | 1.0000 | 软件与系统安装配置/软件安装与配置/ClaudeCode安装.md:1-25 | 1 | 1.0000 | 个人/常用信息集合/各平台api keys记录.md:1-19 |
| DNS 查询过程递归和迭代有什么区别 | 1 | 1.0000 | 个人/面试/基础/DNS查询.md:2-9 | 1 | 1.0000 | 个人/面试/基础/DNS查询.md:2-9 |
| HTTP 强缓存和协商缓存有什么区别 | 1 | 1.0000 | 个人/面试/基础/HTTP缓存.md:1-10 | 1 | 1.0000 | 个人/面试/基础/HTTP缓存.md:11-23 |
| git push 失败怎么设置代理 | 1 | 1.0000 | 软件与工具使用/git/git命令.md:1-38 | 1 | 1.0000 | 软件与工具使用/git/git命令.md:1-38 |
| Docker 怎么启动容器和查看日志 | 1 | 1.0000 | 软件与工具使用/docker/使用命令.md:1-87 | 1 | 1.0000 | 软件与工具使用/docker/使用命令.md:1-87 |
| Linux 进程管理和系统调用 | 1 | 1.0000 | 个人/面试/基础/进程.md:1-21 | 1 | 1.0000 | 个人/面试/基础/进程.md:1-21 |
| 从输入URL到页面渲染，DNS解析、TCP握手和HTTP缓存各在哪个阶段起作用 | 1 | 1.0000 | 个人/面试/基础/计网.md:28-50 | 1 | 1.0000 | 个人/面试/基础/计网.md:28-50 |
| 构建LLM Agent需要考虑哪些架构组件和推理范式 | 1 | 1.0000 | 个人/面试/agent面试/agent-architecture.md:12-17 | 1 | 1.0000 | 个人/面试/agent面试/agent-architecture.md:12-17 |
| git push 怎么配置 SSH 密钥和私人令牌免密认证 | 1 | 1.0000 | 软件与工具使用/git/gitee/记录.md:1-44 | 1 | 1.0000 | 软件与工具使用/git/gitee/记录.md:1-44 |
| 面试可能会问哪些计算机网络和操作系统的问题 | 2 | 1.0000 | imported_docs/textbooks-mineru-agent/操作系统导论/02.md:364-371 | 1 | 1.0000 | 个人/面试/基础/操作系统.md:1-8 |
| 操作系统为什么被称为资源管理器 | 1 | 1.0000 | imported_docs/textbooks-mineru-agent/操作系统导论/02.md:28-39 | 1 | 1.0000 | imported_docs/textbooks-mineru-agent/操作系统导论/02.md:28-39 |
| 虚拟化 CPU 是什么意思，为什么一个 CPU 看起来能同时运行多个程序 | 1 | 1.0000 | imported_docs/textbooks-mineru-agent/操作系统导论/02.md:40-109 | 1 | 1.0000 | imported_docs/textbooks-mineru-agent/操作系统导论/02.md:40-109 |
| 进程是什么，进程的机器状态包括哪些内容 | 1 | 1.0000 | imported_docs/textbooks-mineru-agent/操作系统导论/04.md:36-47 | 1 | 1.0000 | imported_docs/textbooks-mineru-agent/操作系统导论/04.md:36-47 |
| fork 和 exec 分别做什么，为什么 shell 会用它们启动程序 | 1 | 1.0000 | imported_docs/textbooks-mineru-agent/操作系统导论/05.md:159-170 | 1 | 1.0000 | imported_docs/textbooks-mineru-agent/操作系统导论/05.md:175-198 |
| 受限直接执行是什么，操作系统如何限制用户程序的权限 | 1 | 1.0000 | imported_docs/textbooks-mineru-agent/操作系统导论/06.md:26-37 | 1 | 1.0000 | imported_docs/textbooks-mineru-agent/操作系统导论/06.md:26-37 |
| 并发问题为什么会出现，多个线程递增计数器为什么会得到错误结果 | 1 | 1.0000 | imported_docs/textbooks-mineru-agent/操作系统导论/02.md:177-223 | 1 | 1.0000 | imported_docs/textbooks-mineru-agent/操作系统导论/02.md:224-245 |
| 文件系统如何保证数据持久性，日志和写时复制解决什么问题 | 1 | 1.0000 | imported_docs/textbooks-mineru-agent/操作系统导论/02.md:278-287 | 1 | 1.0000 | imported_docs/textbooks-mineru-agent/操作系统导论/02.md:278-287 |
| 彩票调度和步长调度如何实现比例份额调度 | 1 | 1.0000 | imported_docs/textbooks-mineru-agent/操作系统导论/09.md:151-156 | 1 | 1.0000 | imported_docs/textbooks-mineru-agent/操作系统导论/09.md:151-156 |
| 多处理器调度为什么要考虑缓存亲和度，单队列和多队列调度有什么取舍 | 1 | 1.0000 | imported_docs/textbooks-mineru-agent/操作系统导论/10.md:88-91 | 1 | 1.0000 | imported_docs/textbooks-mineru-agent/操作系统导论/10.md:192-195 |

## Changes Vs Baseline

### dashscope-rerank

| query | coverage delta | top changed | top result |
|---|---:|---|---|
| Redis Stream 任务队列怎么消费事件 | 0.0000 | True | 全栈ai/技术栈/postgreSQL.md:205-336 |
| Claude API key 在哪里配置和测试 | 0.0000 | True | 个人/常用信息集合/各平台api keys记录.md:1-19 |
| 面试可能会问哪些计算机网络和操作系统的问题 | 0.0000 | True | 个人/面试/基础/操作系统.md:1-8 |
