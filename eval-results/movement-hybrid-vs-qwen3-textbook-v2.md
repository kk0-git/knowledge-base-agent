# Rerank Movement Report

## Summary

- Queries: 27
- Improved: 1
- Declined: 1
- Unchanged: 25
- Avg movement: -0.185
- Max improvement: 1
- Max decline: -6

## By Query Type

| group | queries | improved | declined | unchanged | avg movement |
| --- | ---: | ---: | ---: | ---: | ---: |
| architecture | 1 | 0 | 0 | 1 | 0.000 |
| broad_review | 1 | 1 | 0 | 0 | 1.000 |
| causal_reasoning | 1 | 0 | 0 | 1 | 0.000 |
| command | 1 | 0 | 0 | 1 | 0.000 |
| comparison | 4 | 0 | 0 | 4 | 0.000 |
| configuration | 4 | 0 | 0 | 4 | 0.000 |
| cross_document | 1 | 0 | 0 | 1 | 0.000 |
| definition | 5 | 0 | 1 | 4 | -1.200 |
| detail_lookup | 1 | 0 | 0 | 1 | 0.000 |
| integration | 2 | 0 | 0 | 2 | 0.000 |
| mechanism | 4 | 0 | 0 | 4 | 0.000 |
| testing | 1 | 0 | 0 | 1 | 0.000 |
| troubleshooting | 1 | 0 | 0 | 1 | 0.000 |

## By Source Type

| group | queries | improved | declined | unchanged | avg movement |
| --- | ---: | ---: | ---: | ---: | ---: |
| mixed_note | 1 | 0 | 0 | 1 | 0.000 |
| mixed_source | 2 | 0 | 0 | 2 | 0.000 |
| personal_note | 11 | 1 | 1 | 9 | -0.455 |
| textbook_pdf | 9 | 0 | 0 | 9 | 0.000 |
| tool_note | 4 | 0 | 0 | 4 | 0.000 |

## Largest Improvements Top 8

| status | movement | before | after | query_type | source_type | query | before top1 | after top1 |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| improved | 1 | 2 | 1 | broad_review | personal_note | 面试可能会问哪些计算机网络和操作系统的问题 | imported_docs/textbooks-mineru-agent/操作系统导论/02.md | 个人/面试/基础/操作系统.md |

## Largest Declines Top 8

| status | movement | before | after | query_type | source_type | query | before top1 | after top1 |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| declined | -6 | 2 | 8 | definition | personal_note | 进程是什么 | imported_docs/textbooks-mineru-agent/操作系统导论/04.md | imported_docs/textbooks-mineru-agent/操作系统导论/04.md |

## All Movements

| status | movement | before | after | query_type | source_type | query | before top1 | after top1 |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| unchanged | 0 | 1 | 1 | configuration | mixed_note | Claude API key 在哪里配置和测试 | 软件与系统安装配置/软件安装与配置/ClaudeCode安装.md | 个人/常用信息集合/各平台api keys记录.md |
| unchanged | 0 | 1 | 1 | configuration | tool_note | Claude Code 代理怎么配置 | 软件与系统安装配置/软件安装与配置/ClaudeCode安装.md | 软件与系统安装配置/软件安装与配置/ClaudeCode安装.md |
| unchanged | 0 | 1 | 1 | comparison | personal_note | DNS 查询过程递归和迭代有什么区别 | 个人/面试/基础/DNS查询.md | 个人/面试/基础/DNS查询.md |
| unchanged | 0 | 1 | 1 | command | tool_note | Docker 怎么启动容器和查看日志 | 软件与工具使用/docker/使用命令.md | 软件与工具使用/docker/使用命令.md |
| unchanged | 0 | 1 | 1 | configuration | mixed_source | FastAPI CORS 跨域怎么配置 | docs/fastapi/tutorial/cors.md | docs/fastapi/tutorial/cors.md |
| unchanged | 0 | 1 | 1 | testing | mixed_source | FastAPI dependency_overrides 怎么替换依赖做测试 | docs/fastapi/advanced/testing-dependencies.md | docs/fastapi/advanced/testing-dependencies.md |
| unchanged | 0 | 1 | 1 | comparison | personal_note | HTTP 强缓存和协商缓存有什么区别 | 个人/面试/基础/HTTP缓存.md | 个人/面试/基础/HTTP缓存.md |
| unchanged | 0 | 1 | 1 | definition | personal_note | Linux 进程管理和系统调用 | 个人/面试/基础/进程.md | 个人/面试/基础/进程.md |
| unchanged | 0 | 1 | 1 | detail_lookup | personal_note | PCB 进程控制块包含什么信息 | 个人/面试/基础/进程.md | 个人/面试/基础/进程.md |
| unchanged | 0 | 1 | 1 | integration | personal_note | Redis Stream 任务队列怎么消费事件 | 全栈ai/技术栈/redis.md | 全栈ai/技术栈/postgreSQL.md |
| unchanged | 0 | 1 | 1 | integration | personal_note | SQLAlchemy async 和 PostgreSQL 怎么配合 | 全栈ai/技术栈/postgreSQL.md | 全栈ai/技术栈/postgreSQL.md |
| unchanged | 0 | 1 | 1 | mechanism | textbook_pdf | fork 和 exec 分别做什么，为什么 shell 会用它们启动程序 | imported_docs/textbooks-mineru-agent/操作系统导论/05.md | imported_docs/textbooks-mineru-agent/操作系统导论/05.md |
| unchanged | 0 | 1 | 1 | troubleshooting | tool_note | git push 失败怎么设置代理 | 软件与工具使用/git/git命令.md | 软件与工具使用/git/git命令.md |
| unchanged | 0 | 1 | 1 | configuration | tool_note | git push 怎么配置 SSH 密钥和私人令牌免密认证 | 软件与工具使用/git/gitee/记录.md | 软件与工具使用/git/gitee/记录.md |
| unchanged | 0 | 1 | 1 | cross_document | personal_note | 从输入URL到页面渲染，DNS解析、TCP握手和HTTP缓存各在哪个阶段起作用 | 个人/面试/基础/计网.md | 个人/面试/基础/计网.md |
| unchanged | 0 | 1 | 1 | mechanism | textbook_pdf | 受限直接执行是什么，操作系统如何限制用户程序的权限 | imported_docs/textbooks-mineru-agent/操作系统导论/06.md | imported_docs/textbooks-mineru-agent/操作系统导论/06.md |
| unchanged | 0 | 1 | 1 | comparison | textbook_pdf | 多处理器调度为什么要考虑缓存亲和度，单队列和多队列调度有什么取舍 | imported_docs/textbooks-mineru-agent/操作系统导论/10.md | imported_docs/textbooks-mineru-agent/操作系统导论/10.md |
| unchanged | 0 | 1 | 1 | causal_reasoning | textbook_pdf | 并发问题为什么会出现，多个线程递增计数器为什么会得到错误结果 | imported_docs/textbooks-mineru-agent/操作系统导论/02.md | imported_docs/textbooks-mineru-agent/操作系统导论/02.md |
| unchanged | 0 | 1 | 1 | mechanism | textbook_pdf | 彩票调度和步长调度如何实现比例份额调度 | imported_docs/textbooks-mineru-agent/操作系统导论/09.md | imported_docs/textbooks-mineru-agent/操作系统导论/09.md |
| unchanged | 0 | 1 | 1 | definition | textbook_pdf | 操作系统为什么被称为资源管理器 | imported_docs/textbooks-mineru-agent/操作系统导论/02.md | imported_docs/textbooks-mineru-agent/操作系统导论/02.md |
| unchanged | 0 | 1 | 1 | mechanism | textbook_pdf | 文件系统如何保证数据持久性，日志和写时复制解决什么问题 | imported_docs/textbooks-mineru-agent/操作系统导论/02.md | imported_docs/textbooks-mineru-agent/操作系统导论/02.md |
| unchanged | 0 | 1 | 1 | architecture | personal_note | 构建LLM Agent需要考虑哪些架构组件和推理范式 | 个人/面试/agent面试/agent-architecture.md | 个人/面试/agent面试/agent-architecture.md |
| unchanged | 0 | 1 | 1 | definition | textbook_pdf | 虚拟化 CPU 是什么意思，为什么一个 CPU 看起来能同时运行多个程序 | imported_docs/textbooks-mineru-agent/操作系统导论/02.md | imported_docs/textbooks-mineru-agent/操作系统导论/02.md |
| unchanged | 0 | 1 | 1 | comparison | personal_note | 进程和线程有什么区别 | 个人/面试/基础/进程.md | 个人/面试/基础/进程.md |
| declined | -6 | 2 | 8 | definition | personal_note | 进程是什么 | imported_docs/textbooks-mineru-agent/操作系统导论/04.md | imported_docs/textbooks-mineru-agent/操作系统导论/04.md |
| unchanged | 0 | 1 | 1 | definition | textbook_pdf | 进程是什么，进程的机器状态包括哪些内容 | imported_docs/textbooks-mineru-agent/操作系统导论/04.md | imported_docs/textbooks-mineru-agent/操作系统导论/04.md |
| improved | 1 | 2 | 1 | broad_review | personal_note | 面试可能会问哪些计算机网络和操作系统的问题 | imported_docs/textbooks-mineru-agent/操作系统导论/02.md | 个人/面试/基础/操作系统.md |

