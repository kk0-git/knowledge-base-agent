# RAG Compare

## Config

- index: `rag-index\bge-m3-v2.json`
- eval: `./eval/rag_eval.json`
- model: `BAAI/bge-m3`
- top_k: `20`
- hit_ks: `1,3,5,10,20`
- strategies: `dense,bm25,hybrid,local-rerank,dashscope-rerank`
- baseline: `hybrid`

## Metrics

| strategy | mrr | hit@1 | hit@3 | hit@5 | hit@10 | hit@20 | avg_expected_note_coverage@1 | avg_expected_note_coverage@3 | avg_expected_note_coverage@5 | avg_expected_note_coverage@10 | avg_expected_note_coverage@20 | avg_unique_notes@1 | avg_max_chunks_per_note@1 | avg_unique_notes@3 | avg_max_chunks_per_note@3 | avg_unique_notes@5 | avg_max_chunks_per_note@5 | avg_unique_notes@10 | avg_max_chunks_per_note@10 | avg_unique_notes@20 | avg_max_chunks_per_note@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8657 | 0.9491 | 0.9676 | 0.9861 | 1.0000 | 1.0000 | 1.0000 | 1.7222 | 2.2778 | 2.7778 | 3.0556 | 4.5000 | 5.0556 | 7.7778 | 7.4444 |
| bm25 | 0.9444 | 0.8889 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.7824 | 0.8843 | 0.9120 | 0.9583 | 0.9583 | 1.0000 | 1.0000 | 2.1111 | 1.8889 | 3.1111 | 2.6111 | 5.2222 | 4.0000 | 8.7778 | 6.4444 |
| hybrid | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8657 | 0.9537 | 0.9861 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.9444 | 2.0556 | 2.8889 | 2.8889 | 4.8889 | 5.0556 | 8.0556 | 7.3333 |
| local-rerank | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8657 | 0.9120 | 0.9537 | 0.9676 | 1.0000 | 1.0000 | 1.0000 | 1.9444 | 2.0556 | 2.6667 | 3.0556 | 4.3333 | 4.9444 | 8.2778 | 7.0000 |
| dashscope-rerank | 0.9722 | 0.9444 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8102 | 0.9444 | 0.9861 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.7222 | 2.2778 | 2.7222 | 3.0556 | 4.6667 | 4.8333 | 8.1111 | 6.9444 |

## Per Query

| query | dense rank | dense cov@20 | dense top | bm25 rank | bm25 cov@20 | bm25 top | hybrid rank | hybrid cov@20 | hybrid top | local-rerank rank | local-rerank cov@20 | local-rerank top | dashscope-rerank rank | dashscope-rerank cov@20 | dashscope-rerank top |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 进程是什么 | 1 | 1.0000 | 个人/面试/基础/进程.md:1-21 | 2 | 1.0000 | 个人/面试/agent面试/RAG/rag各阶段记录.md:34-70 | 1 | 1.0000 | 个人/面试/基础/进程.md:57-63 | 1 | 1.0000 | 个人/面试/基础/进程.md:1-21 | 2 | 1.0000 | 个人/面试/基础/操作系统.md:1-8 |
| 进程和线程有什么区别 | 1 | 1.0000 | 个人/面试/基础/进程.md:57-63 | 1 | 1.0000 | 个人/面试/基础/进程.md:57-63 | 1 | 1.0000 | 个人/面试/基础/进程.md:57-63 | 1 | 1.0000 | 个人/面试/基础/进程.md:57-63 | 1 | 1.0000 | 个人/面试/基础/进程.md:57-63 |
| PCB 进程控制块包含什么信息 | 1 | 1.0000 | 个人/面试/基础/进程.md:22-28 | 1 | 1.0000 | 个人/面试/基础/进程.md:22-28 | 1 | 1.0000 | 个人/面试/基础/进程.md:22-28 | 1 | 1.0000 | 个人/面试/基础/进程.md:22-28 | 1 | 1.0000 | 个人/面试/基础/进程.md:22-28 |
| FastAPI CORS 跨域怎么配置 | 1 | 1.0000 | 全栈ai/技术栈/fastapi.md:146-206 | 1 | 1.0000 | 全栈ai/技术栈/fastapi.md:146-206 | 1 | 1.0000 | 全栈ai/技术栈/fastapi.md:146-206 | 1 | 1.0000 | 全栈ai/技术栈/fastapi.md:146-206 | 1 | 1.0000 | 全栈ai/技术栈/fastapi.md:146-206 |
| FastAPI dependency_overrides 怎么替换依赖做测试 | 1 | 1.0000 | 全栈ai/技术栈/fastapi.md:113-154 | 1 | 1.0000 | 全栈ai/技术栈/fastapi.md:113-154 | 1 | 1.0000 | 全栈ai/技术栈/fastapi.md:113-154 | 1 | 1.0000 | 全栈ai/技术栈/fastapi.md:113-154 | 1 | 1.0000 | 全栈ai/技术栈/fastapi.md:113-154 |
| SQLAlchemy async 和 PostgreSQL 怎么配合 | 1 | 1.0000 | 全栈ai/技术栈/postgreSQL.md:7-76 | 1 | 1.0000 | 全栈ai/技术栈/postgreSQL.md:7-76 | 1 | 1.0000 | 全栈ai/技术栈/postgreSQL.md:7-76 | 1 | 1.0000 | 全栈ai/技术栈/postgreSQL.md:7-76 | 1 | 1.0000 | 全栈ai/技术栈/postgreSQL.md:7-76 |
| Redis Stream 任务队列怎么消费事件 | 1 | 1.0000 | 全栈ai/技术栈/redis.md:253-306 | 2 | 1.0000 | 全栈ai/技术栈/全栈路径.md:203-317 | 1 | 1.0000 | 全栈ai/技术栈/redis.md:253-306 | 1 | 1.0000 | 全栈ai/技术栈/postgreSQL.md:94-230 | 1 | 1.0000 | 全栈ai/技术栈/postgreSQL.md:205-334 |
| Claude Code 代理怎么配置 | 1 | 1.0000 | 软件与系统安装配置/软件安装与配置/ClaudeCode安装.md:11-25 | 1 | 1.0000 | 软件与系统安装配置/软件安装与配置/ClaudeCode安装.md:1-10 | 1 | 1.0000 | 软件与系统安装配置/软件安装与配置/ClaudeCode安装.md:1-10 | 1 | 1.0000 | 软件与系统安装配置/软件安装与配置/ClaudeCode安装.md:1-10 | 1 | 1.0000 | 软件与系统安装配置/软件安装与配置/ClaudeCode安装.md:11-25 |
| Claude API key 在哪里配置和测试 | 1 | 1.0000 | 软件与系统安装配置/软件安装与配置/ClaudeCode安装.md:11-25 | 1 | 1.0000 | 软件与系统安装配置/软件安装与配置/ClaudeCode安装.md:11-25 | 1 | 1.0000 | 软件与系统安装配置/软件安装与配置/ClaudeCode安装.md:11-25 | 1 | 1.0000 | 软件与系统安装配置/软件安装与配置/ClaudeCode安装.md:11-25 | 1 | 1.0000 | 个人/常用信息集合/各平台api keys记录.md:1-19 |
| DNS 查询过程递归和迭代有什么区别 | 1 | 1.0000 | 个人/面试/基础/DNS查询.md:2-9 | 1 | 1.0000 | 个人/面试/基础/DNS查询.md:2-9 | 1 | 1.0000 | 个人/面试/基础/DNS查询.md:2-9 | 1 | 1.0000 | 个人/面试/基础/DNS查询.md:2-9 | 1 | 1.0000 | 个人/面试/基础/DNS查询.md:2-9 |
| HTTP 强缓存和协商缓存有什么区别 | 1 | 1.0000 | 个人/面试/基础/HTTP缓存.md:1-10 | 1 | 1.0000 | 个人/面试/基础/HTTP缓存.md:1-10 | 1 | 1.0000 | 个人/面试/基础/HTTP缓存.md:1-10 | 1 | 1.0000 | 个人/面试/基础/HTTP缓存.md:1-10 | 1 | 1.0000 | 个人/面试/基础/HTTP缓存.md:11-23 |
| git push 失败怎么设置代理 | 1 | 1.0000 | 软件与工具使用/git/git命令.md:1-38 | 1 | 1.0000 | 软件与工具使用/git/git命令.md:1-38 | 1 | 1.0000 | 软件与工具使用/git/git命令.md:1-38 | 1 | 1.0000 | 软件与工具使用/git/git命令.md:1-38 | 1 | 1.0000 | 软件与工具使用/git/git命令.md:1-38 |
| Docker 怎么启动容器和查看日志 | 1 | 1.0000 | 软件与工具使用/docker/使用命令.md:54-87 | 1 | 1.0000 | 软件与工具使用/docker/使用命令.md:54-87 | 1 | 1.0000 | 软件与工具使用/docker/使用命令.md:54-87 | 1 | 1.0000 | 软件与工具使用/docker/使用命令.md:54-87 | 1 | 1.0000 | 软件与工具使用/docker/使用命令.md:54-87 |
| Linux 进程管理和系统调用 | 1 | 1.0000 | 个人/面试/基础/进程.md:44-49 | 1 | 1.0000 | 个人/面试/基础/进程.md:1-21 | 1 | 1.0000 | 个人/面试/基础/进程.md:1-21 | 1 | 1.0000 | 个人/面试/基础/进程.md:1-21 | 1 | 1.0000 | 个人/面试/基础/进程.md:1-21 |
| 从输入URL到页面渲染，DNS解析、TCP握手和HTTP缓存各在哪个阶段起作用 | 1 | 1.0000 | 个人/面试/基础/计网.md:28-50 | 1 | 0.7500 | 个人/面试/基础/计网.md:28-50 | 1 | 1.0000 | 个人/面试/基础/计网.md:28-50 | 1 | 1.0000 | 个人/面试/基础/计网.md:28-50 | 1 | 1.0000 | 个人/面试/基础/计网.md:28-50 |
| 构建LLM Agent需要考虑哪些架构组件和推理范式 | 1 | 1.0000 | 个人/面试/agent面试/agent-architecture.md:12-17 | 1 | 1.0000 | 个人/面试/agent面试/agent-architecture.md:12-17 | 1 | 1.0000 | 个人/面试/agent面试/agent-architecture.md:12-17 | 1 | 1.0000 | 个人/面试/agent面试/agent-architecture.md:12-17 | 1 | 1.0000 | 个人/面试/agent面试/agent-architecture.md:12-17 |
| git push 怎么配置 SSH 密钥和私人令牌免密认证 | 1 | 1.0000 | 软件与工具使用/git/gitee/记录.md:1-28 | 1 | 1.0000 | 软件与工具使用/git/gitee/记录.md:1-28 | 1 | 1.0000 | 软件与工具使用/git/gitee/记录.md:1-28 | 1 | 1.0000 | 软件与工具使用/git/gitee/记录.md:1-28 | 1 | 1.0000 | 软件与工具使用/git/gitee/记录.md:1-28 |
| 面试可能会问哪些计算机网络和操作系统的问题 | 1 | 1.0000 | 个人/面试/基础/操作系统.md:1-8 | 1 | 0.5000 | 个人/面试/基础/操作系统.md:1-8 | 1 | 1.0000 | 个人/面试/基础/操作系统.md:1-8 | 1 | 1.0000 | 个人/面试/基础/操作系统.md:1-8 | 1 | 1.0000 | 个人/面试/基础/操作系统.md:1-8 |

## Changes Vs Baseline

### dense

No top result or coverage changes.

### bm25

| query | coverage delta | top changed | top result |
|---|---:|---|---|
| 进程是什么 | 0.0000 | True | 个人/面试/agent面试/RAG/rag各阶段记录.md:34-70 |
| Redis Stream 任务队列怎么消费事件 | 0.0000 | True | 全栈ai/技术栈/全栈路径.md:203-317 |
| 从输入URL到页面渲染，DNS解析、TCP握手和HTTP缓存各在哪个阶段起作用 | -0.2500 | False | 个人/面试/基础/计网.md:28-50 |
| 面试可能会问哪些计算机网络和操作系统的问题 | -0.5000 | False | 个人/面试/基础/操作系统.md:1-8 |

### local-rerank

| query | coverage delta | top changed | top result |
|---|---:|---|---|
| Redis Stream 任务队列怎么消费事件 | 0.0000 | True | 全栈ai/技术栈/postgreSQL.md:94-230 |

### dashscope-rerank

| query | coverage delta | top changed | top result |
|---|---:|---|---|
| 进程是什么 | 0.0000 | True | 个人/面试/基础/操作系统.md:1-8 |
| Redis Stream 任务队列怎么消费事件 | 0.0000 | True | 全栈ai/技术栈/postgreSQL.md:205-334 |
| Claude API key 在哪里配置和测试 | 0.0000 | True | 个人/常用信息集合/各平台api keys记录.md:1-19 |
