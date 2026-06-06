# RAG Compare

## Config

- index: `rag-index\bge-m3-v2.json`
- eval: `./eval/rag_eval.json`
- model: `BAAI/bge-m3`
- top_k: `1`
- hit_ks: `1`
- strategies: `dashscope-rerank`
- baseline: `dashscope-rerank`

## Metrics

| strategy | mrr | hit@1 | avg_expected_note_coverage@1 | avg_unique_notes@1 | avg_max_chunks_per_note@1 |
|---|---:|---:|---:|---:|---:|
| dashscope-rerank | 0.9444 | 0.9444 | 0.8102 | 1.0000 | 1.0000 |

## Per Query

| query | dashscope-rerank rank | dashscope-rerank cov@1 | dashscope-rerank top |
|---|---|---|---|
| 进程是什么 | None | 0.0000 | 个人/面试/基础/操作系统.md:1-8 |
| 进程和线程有什么区别 | 1 | 1.0000 | 个人/面试/基础/进程.md:57-63 |
| PCB 进程控制块包含什么信息 | 1 | 1.0000 | 个人/面试/基础/进程.md:22-28 |
| FastAPI CORS 跨域怎么配置 | 1 | 1.0000 | 全栈ai/技术栈/fastapi.md:146-206 |
| FastAPI dependency_overrides 怎么替换依赖做测试 | 1 | 1.0000 | 全栈ai/技术栈/fastapi.md:113-154 |
| SQLAlchemy async 和 PostgreSQL 怎么配合 | 1 | 1.0000 | 全栈ai/技术栈/postgreSQL.md:7-76 |
| Redis Stream 任务队列怎么消费事件 | 1 | 0.5000 | 全栈ai/技术栈/postgreSQL.md:205-334 |
| Claude Code 代理怎么配置 | 1 | 1.0000 | 软件与系统安装配置/软件安装与配置/ClaudeCode安装.md:11-25 |
| Claude API key 在哪里配置和测试 | 1 | 0.3333 | 个人/常用信息集合/各平台api keys记录.md:1-19 |
| DNS 查询过程递归和迭代有什么区别 | 1 | 1.0000 | 个人/面试/基础/DNS查询.md:2-9 |
| HTTP 强缓存和协商缓存有什么区别 | 1 | 1.0000 | 个人/面试/基础/HTTP缓存.md:11-23 |
| git push 失败怎么设置代理 | 1 | 1.0000 | 软件与工具使用/git/git命令.md:1-38 |
| Docker 怎么启动容器和查看日志 | 1 | 1.0000 | 软件与工具使用/docker/使用命令.md:54-87 |
| Linux 进程管理和系统调用 | 1 | 1.0000 | 个人/面试/基础/进程.md:1-21 |
| 从输入URL到页面渲染，DNS解析、TCP握手和HTTP缓存各在哪个阶段起作用 | 1 | 0.2500 | 个人/面试/基础/计网.md:28-50 |
| 构建LLM Agent需要考虑哪些架构组件和推理范式 | 1 | 1.0000 | 个人/面试/agent面试/agent-architecture.md:12-17 |
| git push 怎么配置 SSH 密钥和私人令牌免密认证 | 1 | 1.0000 | 软件与工具使用/git/gitee/记录.md:1-28 |
| 面试可能会问哪些计算机网络和操作系统的问题 | 1 | 0.5000 | 个人/面试/基础/操作系统.md:1-8 |

## Changes Vs Baseline
