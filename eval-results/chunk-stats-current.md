# Chunk Statistics

## Summary

- note_count: 213
- chunk_count: 3368
- avg_chunks_per_note: 15.812
- avg_chunk_chars: 517.94
- median_chunk_chars: 315.5
- max_chunk_chars: 5538
- split_chunk_count: 1212
- overlap_candidate_count: 1224
- notes_with_possible_code_boundary_issue: 25

## By Source Type

| source_type | notes | chunks | avg chunks/note | split chunks | overlap candidates | code boundary issues |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| official_docs | 125 | 1164 | 9.31 | 44 | 48 | 9 |
| paper_or_paper_md | 7 | 444 | 63.43 | 427 | 427 | 0 |
| personal_note | 75 | 1624 | 21.65 | 698 | 706 | 14 |
| textbook_pdf | 6 | 136 | 22.67 | 43 | 43 | 2 |

## Chunk Length Distribution

### Thresholds

| metric | count | pct |
| --- | ---: | ---: |
| below_min_chunk_chars | 969 | 28.77% |
| below_target_chunk_chars | 2615 | 77.64% |
| over_target_chunk_chars | 752 | 22.33% |
| over_max_chunk_chars | 49 | 1.45% |

### Buckets

| bucket | count | pct |
| --- | ---: | ---: |
| 0-100 | 180 | 5.34% |
| 101-200 | 790 | 23.46% |
| 201-500 | 1196 | 35.51% |
| 501-900 | 450 | 13.36% |
| 901-1500 | 703 | 20.87% |
| 1501-2600 | 40 | 1.19% |
| 2601+ | 9 | 0.27% |

## Chunk Length Distribution By Source

| source | chunks | avg chars | median chars | min | max | below min | over max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| official_docs | 1164 | 388.18 | 297.0 | 34 | 3006 | 326 (28.01%) | 5 (0.43%) |
| paper_or_paper_md | 444 | 1080.51 | 1110.0 | 137 | 1490 | 7 (1.58%) | 0 (0.00%) |
| personal_note | 1624 | 429.06 | 218.0 | 11 | 5538 | 622 (38.30%) | 22 (1.35%) |
| textbook_pdf | 136 | 853.32 | 742.5 | 53 | 3028 | 14 (10.29%) | 22 (16.18%) |

### official_docs Buckets

| bucket | count | pct |
| --- | ---: | ---: |
| 0-100 | 67 | 5.76% |
| 101-200 | 260 | 22.34% |
| 201-500 | 552 | 47.42% |
| 501-900 | 213 | 18.30% |
| 901-1500 | 67 | 5.76% |
| 1501-2600 | 4 | 0.34% |
| 2601+ | 1 | 0.09% |

### paper_or_paper_md Buckets

| bucket | count | pct |
| --- | ---: | ---: |
| 0-100 | 0 | 0.00% |
| 101-200 | 7 | 1.58% |
| 201-500 | 7 | 1.58% |
| 501-900 | 50 | 11.26% |
| 901-1500 | 380 | 85.59% |
| 1501-2600 | 0 | 0.00% |
| 2601+ | 0 | 0.00% |

### personal_note Buckets

| bucket | count | pct |
| --- | ---: | ---: |
| 0-100 | 109 | 6.71% |
| 101-200 | 513 | 31.59% |
| 201-500 | 595 | 36.64% |
| 501-900 | 166 | 10.22% |
| 901-1500 | 219 | 13.49% |
| 1501-2600 | 16 | 0.99% |
| 2601+ | 6 | 0.37% |

### textbook_pdf Buckets

| bucket | count | pct |
| --- | ---: | ---: |
| 0-100 | 4 | 2.94% |
| 101-200 | 10 | 7.35% |
| 201-500 | 42 | 30.88% |
| 501-900 | 21 | 15.44% |
| 901-1500 | 37 | 27.21% |
| 1501-2600 | 20 | 14.71% |
| 2601+ | 2 | 1.47% |

## Split Reasons

- overlong_section: 1212
- oversized_code_block: 12
- section_end: 2144

## Top Notes By Chunk Count Top 15

| note | source | chunks | sections | max chars | split chunks | overlap candidates | code issue |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | personal_note | 355 | 25 | 5538 | 327 | 331 | True |
| courses/js/Servlet-JSP-课堂笔记.md | personal_note | 355 | 25 | 5538 | 327 | 331 | True |
| courses/期末复习/网安详细知识点.md | personal_note | 151 | 220 | 1337 | 0 | 0 | False |
| courses/网安课程复习/网安详细知识点.md | personal_note | 151 | 220 | 1337 | 0 | 0 | False |
| papers/indexing/RAPTOR_Sarthi_2024.md | paper_or_paper_md | 90 | 2 | 1474 | 88 | 88 | False |
| papers/rag/RAG_Lewis_2020.md | paper_or_paper_md | 84 | 2 | 1488 | 82 | 82 | False |
| papers/indexing/HNSW_Malkov_2016.md | paper_or_paper_md | 79 | 2 | 1451 | 77 | 77 | False |
| papers/retrieval/ColBERT_Khattab_2020.md | paper_or_paper_md | 74 | 3 | 1452 | 71 | 71 | False |
| papers/rag/DPR_Karpukhin_2020.md | paper_or_paper_md | 64 | 7 | 1490 | 61 | 61 | False |
| courses/js/js.md | personal_note | 59 | 82 | 1300 | 0 | 0 | False |
| courses/js/前端基础/js.md | personal_note | 59 | 82 | 1300 | 0 | 0 | False |
| papers/indexing/FAISS_Johnson_2023.md | paper_or_paper_md | 50 | 2 | 1429 | 48 | 48 | False |
| docs/fastapi/deployment/docker.md | official_docs | 37 | 37 | 1320 | 0 | 0 | False |
| docs/fastapi/index.md | official_docs | 37 | 32 | 1483 | 6 | 6 | False |
| courses/期末复习/网络安全与应用期末考试.md | personal_note | 34 | 40 | 1036 | 2 | 2 | False |

## Top Notes By Max Chunk Length Top 15

| note | source | chunks | sections | max chars | split chunks | overlap candidates | code issue |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | personal_note | 355 | 25 | 5538 | 327 | 331 | True |
| courses/js/Servlet-JSP-课堂笔记.md | personal_note | 355 | 25 | 5538 | 327 | 331 | True |
| imported_docs/textbooks-mineru-agent/操作系统导论/10.md | textbook_pdf | 28 | 12 | 3028 | 16 | 16 | False |
| docs/fastapi/fastapi-cli.md | official_docs | 23 | 5 | 3006 | 18 | 18 | True |
| imported_docs/textbooks-mineru-agent/操作系统导论/04.md | textbook_pdf | 22 | 14 | 2837 | 9 | 9 | True |
| imported_docs/textbooks-mineru-agent/操作系统导论/06.md | textbook_pdf | 29 | 21 | 1734 | 9 | 9 | False |
| imported_docs/textbooks-mineru-agent/操作系统导论/09.md | textbook_pdf | 18 | 14 | 1714 | 4 | 4 | False |
| docs/fastapi/deployment/manually.md | official_docs | 7 | 7 | 1671 | 0 | 1 | True |
| docs/fastapi/deployment/server-workers.md | official_docs | 9 | 5 | 1657 | 3 | 4 | True |
| docs/fastapi/tutorial/first-steps.md | official_docs | 26 | 27 | 1624 | 0 | 1 | True |
| docs/fastapi/tutorial/index.md | official_docs | 5 | 4 | 1608 | 0 | 1 | True |
| courses/react/Notes.md | personal_note | 5 | 2 | 1498 | 3 | 3 | True |
| courses/react/笔记.md | personal_note | 5 | 2 | 1498 | 3 | 3 | True |
| docs/fastapi/tutorial/metadata.md | official_docs | 12 | 10 | 1494 | 2 | 2 | False |
| 全栈ai/技术栈/postgreSQL.md | personal_note | 6 | 1 | 1492 | 5 | 5 | True |

## Shortest Chunks Top 15

| note | heading | chars | split_reason | lines |
| --- | --- | ---: | --- | --- |
| 全栈ai/项目/Agent开发学习路径/参考经验.md |  | 11 | section_end | 1-1 |
| courses/js/js.md |  | 14 | section_end | 1-1 |
| 软件与系统安装配置/Linux系统/Linux.md |  | 14 | section_end | 1-1 |
| courses/python/python数据分析.md |  | 15 | section_end | 1-1 |
| courses/期末复习/py.md |  | 15 | section_end | 1-1 |
| 全栈ai/技术栈/fastapi.md |  | 19 | section_end | 1-1 |
| courses/js/js.md | 2. 入门 > 2.2 数据类型 > 2.2.2 字符串 | 26 | section_end | 35-38 |
| courses/js/前端基础/js.md | 2. 入门 > 2.2 数据类型 > 2.2.2 字符串 | 26 | section_end | 36-39 |
| 全栈ai/项目/Agent开发学习路径/学习资源.md | 拖拽流 | 28 | section_end | 21-23 |
| courses/js/jsp.md |  | 29 | section_end | 1-2 |
| courses/js/js.md | 8. Web API > 8.2 DOM 概念 > DOM节点 | 30 | section_end | 808-810 |
| courses/js/前端基础/js.md | 8. Web API > 8.2 [[Dom]] 概念 > DOM节点 | 30 | section_end | 809-811 |
| courses/js/js.md | 2. 入门 > 2.2 数据类型 > 2.2.3 布尔值 | 31 | section_end | 39-42 |
| courses/js/前端基础/js.md | 2. 入门 > 2.2 数据类型 > 2.2.3 布尔值 | 31 | section_end | 40-43 |
| courses/js/JavaWeb后端/jsp.md |  | 32 | section_end | 1-2 |

## Longest Chunks Top 15

| note | heading | chars | split_reason | lines |
| --- | --- | ---: | --- | --- |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | Servlet > ServletContext | 5538 | overlong_section | 1245-1415 |
| courses/js/Servlet-JSP-课堂笔记.md | Servlet > ServletContext | 5538 | overlong_section | 1255-1425 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | Servlet > 关于B/S结构系统的会话机制（session机制） | 3226 | overlong_section | 2282-2377 |
| courses/js/Servlet-JSP-课堂笔记.md | Servlet > 关于B/S结构系统的会话机制（session机制） | 3226 | overlong_section | 2292-2387 |
| imported_docs/textbooks-mineru-agent/操作系统导论/10.md | 10.4单队列调度 | 3028 | overlong_section | 109-120 |
| docs/fastapi/fastapi-cli.md | FastAPI CLI { #fastapi-cli } | 3006 | overlong_section | 3-45 |
| imported_docs/textbooks-mineru-agent/操作系统导论/04.md | 4.4进程状态 | 2837 | overlong_section | 110-118 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | Filter过滤器 | 2632 | overlong_section | 2942-3047 |
| courses/js/Servlet-JSP-课堂笔记.md | Filter过滤器 | 2632 | overlong_section | 2952-3057 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | Servlet > 开发一个带有Servlet（Java小程序）的webapp（重点） | 2259 | overlong_section | 322-387 |
| courses/js/Servlet-JSP-课堂笔记.md | Servlet > 开发一个带有Servlet（Java小程序）的webapp（重点） | 2259 | overlong_section | 332-397 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | JSP | 1969 | overlong_section | 2525-2598 |
| courses/js/Servlet-JSP-课堂笔记.md | JSP | 1969 | overlong_section | 2535-2608 |
| imported_docs/textbooks-mineru-agent/操作系统导论/04.md | 4.4进程状态 | 1876 | overlong_section | 118-122 |
| imported_docs/textbooks-mineru-agent/操作系统导论/06.md | 补充：为什么系统调用看起来像过程调用 | 1734 | overlong_section | 78-80 |

## Code Boundary Issues

| note | code range | boundary line | kinds | overlap | previous chunk | next chunk |
| --- | --- | ---: | --- | --- | --- | --- |
| courses/js/Dom.md | 66-74 | 71 | chunk_start | True | courses/js/Dom.md:50-77 lines 50-77 reason=overlong_section chars=873 | courses/js/Dom.md:71-108 lines 71-108 reason=section_end chars=1001 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 330-387 | 379 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:322-387 lines 322-387 reason=overlong_section chars=2259 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:379-427 lines 379-427 reason=section_end chars=1481 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 491-508 | 505 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:484-518 lines 484-518 reason=overlong_section chars=1272 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:505-567 lines 505-567 reason=overlong_section chars=1375 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 532-594 | 568 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:505-567 lines 505-567 reason=overlong_section chars=1375 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:562-605 lines 562-605 reason=overlong_section chars=1374 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 532-594 | 562 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:505-567 lines 505-567 reason=overlong_section chars=1375 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:562-605 lines 562-605 reason=overlong_section chars=1374 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 532-594 | 593 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:562-605 lines 562-605 reason=overlong_section chars=1374 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:593-629 lines 593-629 reason=overlong_section chars=1077 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 602-606 | 606 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:562-605 lines 562-605 reason=overlong_section chars=1374 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:593-629 lines 593-629 reason=overlong_section chars=1077 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 628-630 | 630 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:593-629 lines 593-629 reason=overlong_section chars=1077 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:623-671 lines 623-671 reason=overlong_section chars=1457 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 740-780 | 777 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:731-780 lines 731-780 reason=overlong_section chars=1540 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:777-783 lines 777-783 reason=section_end chars=271 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 811-825 | 821 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:784-825 lines 784-825 reason=overlong_section chars=1479 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:821-862 lines 821-862 reason=overlong_section chars=1375 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 850-874 | 863 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:821-862 lines 821-862 reason=overlong_section chars=1375 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:859-886 lines 859-886 reason=overlong_section chars=1151 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 850-874 | 859 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:821-862 lines 821-862 reason=overlong_section chars=1375 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:859-886 lines 859-886 reason=overlong_section chars=1151 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 884-975 | 887 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:859-886 lines 859-886 reason=overlong_section chars=1151 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:880-925 lines 880-925 reason=oversized_code_block chars=1504 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 884-975 | 926 | chunk_end_next_line,chunk_start | False | courses/js/JavaWeb后端/Servlet-JSP-参考.md:880-925 lines 880-925 reason=oversized_code_block chars=1504 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:926-954 lines 926-954 reason=overlong_section chars=813 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 884-975 | 955 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:926-954 lines 926-954 reason=overlong_section chars=813 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:946-974 lines 946-974 reason=overlong_section chars=1359 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 884-975 | 946 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:926-954 lines 926-954 reason=overlong_section chars=813 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:946-974 lines 946-974 reason=overlong_section chars=1359 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 884-975 | 975 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:972-974 lines 972-974 reason=overlong_section chars=227 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:972-1006 lines 972-1006 reason=overlong_section chars=1574 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 884-975 | 972 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:972-974 lines 972-974 reason=overlong_section chars=227 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:972-1006 lines 972-1006 reason=overlong_section chars=1574 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 1006-1073 | 1007 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:972-1006 lines 972-1006 reason=overlong_section chars=1574 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1004-1069 lines 1004-1069 reason=oversized_code_block chars=1424 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 1006-1073 | 1070 | chunk_end_next_line,chunk_start | False | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1004-1069 lines 1004-1069 reason=oversized_code_block chars=1424 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1070-1122 lines 1070-1122 reason=oversized_code_block chars=1521 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 1125-1253 | 1152 | chunk_end_next_line,chunk_start | False | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1123-1151 lines 1123-1151 reason=oversized_code_block chars=1496 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1152-1184 lines 1152-1184 reason=overlong_section chars=1480 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 1125-1253 | 1185 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1152-1184 lines 1152-1184 reason=overlong_section chars=1480 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1178-1214 lines 1178-1214 reason=overlong_section chars=707 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 1125-1253 | 1178 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1152-1184 lines 1152-1184 reason=overlong_section chars=1480 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1178-1214 lines 1178-1214 reason=overlong_section chars=707 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 1125-1253 | 1215 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1178-1214 lines 1178-1214 reason=overlong_section chars=707 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1205-1246 lines 1205-1246 reason=overlong_section chars=1480 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 1125-1253 | 1205 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1178-1214 lines 1178-1214 reason=overlong_section chars=707 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1205-1246 lines 1205-1246 reason=overlong_section chars=1480 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 1125-1253 | 1247 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1205-1246 lines 1205-1246 reason=overlong_section chars=1480 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1239-1252 lines 1239-1252 reason=overlong_section chars=416 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 1125-1253 | 1239 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1205-1246 lines 1205-1246 reason=overlong_section chars=1480 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1239-1252 lines 1239-1252 reason=overlong_section chars=416 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 1125-1253 | 1253 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1245-1252 lines 1245-1252 reason=overlong_section chars=194 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1245-1415 lines 1245-1415 reason=overlong_section chars=5538 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 1125-1253 | 1245 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1245-1252 lines 1245-1252 reason=overlong_section chars=194 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1245-1415 lines 1245-1415 reason=overlong_section chars=5538 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 1415-1448 | 1416 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1245-1415 lines 1245-1415 reason=overlong_section chars=5538 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1406-1464 lines 1406-1464 reason=overlong_section chars=1379 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 1501-1532 | 1527 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1496-1536 lines 1496-1536 reason=overlong_section chars=1193 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1527-1559 lines 1527-1559 reason=section_end chars=890 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 1655-1693 | 1689 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1651-1693 lines 1651-1693 reason=overlong_section chars=1547 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1689-1725 lines 1689-1725 reason=overlong_section chars=1064 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 1707-1726 | 1726 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1689-1725 lines 1689-1725 reason=overlong_section chars=1064 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1720-1757 lines 1720-1757 reason=overlong_section chars=1186 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 1707-1726 | 1720 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1689-1725 lines 1689-1725 reason=overlong_section chars=1064 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1720-1757 lines 1720-1757 reason=overlong_section chars=1186 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 1751-1792 | 1758 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1720-1757 lines 1720-1757 reason=overlong_section chars=1186 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1749-1793 lines 1749-1793 reason=section_end chars=1502 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 1816-1861 | 1853 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1811-1863 lines 1811-1863 reason=overlong_section chars=1314 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1853-1894 lines 1853-1894 reason=overlong_section chars=1358 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 1875-1895 | 1895 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1853-1894 lines 1853-1894 reason=overlong_section chars=1358 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1889-1931 lines 1889-1931 reason=overlong_section chars=1331 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 1875-1895 | 1889 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1853-1894 lines 1853-1894 reason=overlong_section chars=1358 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1889-1931 lines 1889-1931 reason=overlong_section chars=1331 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 1922-1932 | 1932 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1889-1931 lines 1889-1931 reason=overlong_section chars=1331 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:1920-1971 lines 1920-1971 reason=overlong_section chars=1361 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 2021-2043 | 2037 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2001-2045 lines 2001-2045 reason=overlong_section chars=1328 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2037-2091 lines 2037-2091 reason=overlong_section chars=1346 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 2058-2098 | 2092 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2037-2091 lines 2037-2091 reason=overlong_section chars=1346 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2087-2131 lines 2087-2131 reason=overlong_section chars=1447 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 2058-2098 | 2087 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2037-2091 lines 2037-2091 reason=overlong_section chars=1346 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2087-2131 lines 2087-2131 reason=overlong_section chars=1447 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 2108-2173 | 2132 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2087-2131 lines 2087-2131 reason=overlong_section chars=1447 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2127-2172 lines 2127-2172 reason=overlong_section chars=1138 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 2108-2173 | 2127 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2087-2131 lines 2087-2131 reason=overlong_section chars=1447 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2127-2172 lines 2127-2172 reason=overlong_section chars=1138 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 2108-2173 | 2173 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2127-2172 lines 2127-2172 reason=overlong_section chars=1138 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2162-2187 lines 2162-2187 reason=section_end chars=649 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 2108-2173 | 2162 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2127-2172 lines 2127-2172 reason=overlong_section chars=1138 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2162-2187 lines 2162-2187 reason=section_end chars=649 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 2292-2377 | 2369 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2282-2377 lines 2282-2377 reason=overlong_section chars=3226 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2369-2396 lines 2369-2396 reason=section_end chars=670 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 2531-2598 | 2594 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2525-2598 lines 2525-2598 reason=overlong_section chars=1969 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2594-2649 lines 2594-2649 reason=overlong_section chars=1482 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 2628-2714 | 2650 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2594-2649 lines 2594-2649 reason=overlong_section chars=1482 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2642-2699 lines 2642-2699 reason=overlong_section chars=1437 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 2628-2714 | 2642 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2594-2649 lines 2594-2649 reason=overlong_section chars=1482 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2642-2699 lines 2642-2699 reason=overlong_section chars=1437 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 2628-2714 | 2700 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2642-2699 lines 2642-2699 reason=overlong_section chars=1437 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2694-2713 lines 2694-2713 reason=overlong_section chars=701 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 2628-2714 | 2694 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2642-2699 lines 2642-2699 reason=overlong_section chars=1437 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2694-2713 lines 2694-2713 reason=overlong_section chars=701 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 2628-2714 | 2714 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2694-2713 lines 2694-2713 reason=overlong_section chars=701 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2707-2764 lines 2707-2764 reason=overlong_section chars=1491 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 2628-2714 | 2707 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2694-2713 lines 2694-2713 reason=overlong_section chars=701 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2707-2764 lines 2707-2764 reason=overlong_section chars=1491 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 2757-2827 | 2765 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2707-2764 lines 2707-2764 reason=overlong_section chars=1491 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2759-2809 lines 2759-2809 reason=overlong_section chars=1468 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 2757-2827 | 2759 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2707-2764 lines 2707-2764 reason=overlong_section chars=1491 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2759-2809 lines 2759-2809 reason=overlong_section chars=1468 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 2757-2827 | 2810 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2759-2809 lines 2759-2809 reason=overlong_section chars=1468 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2799-2842 lines 2799-2842 reason=overlong_section chars=1114 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 2757-2827 | 2799 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2759-2809 lines 2759-2809 reason=overlong_section chars=1468 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2799-2842 lines 2799-2842 reason=overlong_section chars=1114 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 2840-2863 | 2843 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2799-2842 lines 2799-2842 reason=overlong_section chars=1114 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2835-2876 lines 2835-2876 reason=overlong_section chars=1260 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 2948-3047 | 3040 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:2942-3047 lines 2942-3047 reason=overlong_section chars=2632 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:3040-3087 lines 3040-3087 reason=overlong_section chars=1435 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 3055-3118 | 3088 | chunk_end_next_line | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:3040-3087 lines 3040-3087 reason=overlong_section chars=1435 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:3076-3118 lines 3076-3118 reason=section_end chars=874 |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | 3055-3118 | 3076 | chunk_start | True | courses/js/JavaWeb后端/Servlet-JSP-参考.md:3040-3087 lines 3040-3087 reason=overlong_section chars=1435 | courses/js/JavaWeb后端/Servlet-JSP-参考.md:3076-3118 lines 3076-3118 reason=section_end chars=874 |
| courses/js/JavaWeb后端/Servlet.md | 311-336 | 328 | chunk_start | True | courses/js/JavaWeb后端/Servlet.md:292-336 lines 292-336 reason=overlong_section chars=1223 | courses/js/JavaWeb后端/Servlet.md:328-365 lines 328-365 reason=section_end chars=1012 |
| courses/js/Servlet-JSP-课堂笔记.md | 340-397 | 389 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:332-397 lines 332-397 reason=overlong_section chars=2259 | courses/js/Servlet-JSP-课堂笔记.md:389-437 lines 389-437 reason=section_end chars=1481 |
| courses/js/Servlet-JSP-课堂笔记.md | 501-518 | 515 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:494-528 lines 494-528 reason=overlong_section chars=1272 | courses/js/Servlet-JSP-课堂笔记.md:515-577 lines 515-577 reason=overlong_section chars=1375 |
| courses/js/Servlet-JSP-课堂笔记.md | 542-604 | 578 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:515-577 lines 515-577 reason=overlong_section chars=1375 | courses/js/Servlet-JSP-课堂笔记.md:572-615 lines 572-615 reason=overlong_section chars=1374 |
| courses/js/Servlet-JSP-课堂笔记.md | 542-604 | 572 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:515-577 lines 515-577 reason=overlong_section chars=1375 | courses/js/Servlet-JSP-课堂笔记.md:572-615 lines 572-615 reason=overlong_section chars=1374 |
| courses/js/Servlet-JSP-课堂笔记.md | 542-604 | 603 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:572-615 lines 572-615 reason=overlong_section chars=1374 | courses/js/Servlet-JSP-课堂笔记.md:603-639 lines 603-639 reason=overlong_section chars=1077 |
| courses/js/Servlet-JSP-课堂笔记.md | 612-616 | 616 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:572-615 lines 572-615 reason=overlong_section chars=1374 | courses/js/Servlet-JSP-课堂笔记.md:603-639 lines 603-639 reason=overlong_section chars=1077 |
| courses/js/Servlet-JSP-课堂笔记.md | 638-640 | 640 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:603-639 lines 603-639 reason=overlong_section chars=1077 | courses/js/Servlet-JSP-课堂笔记.md:633-681 lines 633-681 reason=overlong_section chars=1457 |
| courses/js/Servlet-JSP-课堂笔记.md | 750-790 | 787 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:741-790 lines 741-790 reason=overlong_section chars=1540 | courses/js/Servlet-JSP-课堂笔记.md:787-793 lines 787-793 reason=section_end chars=271 |
| courses/js/Servlet-JSP-课堂笔记.md | 821-835 | 831 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:794-835 lines 794-835 reason=overlong_section chars=1479 | courses/js/Servlet-JSP-课堂笔记.md:831-872 lines 831-872 reason=overlong_section chars=1375 |
| courses/js/Servlet-JSP-课堂笔记.md | 860-884 | 873 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:831-872 lines 831-872 reason=overlong_section chars=1375 | courses/js/Servlet-JSP-课堂笔记.md:869-896 lines 869-896 reason=overlong_section chars=1151 |
| courses/js/Servlet-JSP-课堂笔记.md | 860-884 | 869 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:831-872 lines 831-872 reason=overlong_section chars=1375 | courses/js/Servlet-JSP-课堂笔记.md:869-896 lines 869-896 reason=overlong_section chars=1151 |
| courses/js/Servlet-JSP-课堂笔记.md | 894-985 | 897 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:869-896 lines 869-896 reason=overlong_section chars=1151 | courses/js/Servlet-JSP-课堂笔记.md:890-935 lines 890-935 reason=oversized_code_block chars=1504 |
| courses/js/Servlet-JSP-课堂笔记.md | 894-985 | 936 | chunk_end_next_line,chunk_start | False | courses/js/Servlet-JSP-课堂笔记.md:890-935 lines 890-935 reason=oversized_code_block chars=1504 | courses/js/Servlet-JSP-课堂笔记.md:936-964 lines 936-964 reason=overlong_section chars=813 |
| courses/js/Servlet-JSP-课堂笔记.md | 894-985 | 965 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:936-964 lines 936-964 reason=overlong_section chars=813 | courses/js/Servlet-JSP-课堂笔记.md:956-984 lines 956-984 reason=overlong_section chars=1359 |
| courses/js/Servlet-JSP-课堂笔记.md | 894-985 | 956 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:936-964 lines 936-964 reason=overlong_section chars=813 | courses/js/Servlet-JSP-课堂笔记.md:956-984 lines 956-984 reason=overlong_section chars=1359 |
| courses/js/Servlet-JSP-课堂笔记.md | 894-985 | 985 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:982-984 lines 982-984 reason=overlong_section chars=227 | courses/js/Servlet-JSP-课堂笔记.md:982-1016 lines 982-1016 reason=overlong_section chars=1574 |
| courses/js/Servlet-JSP-课堂笔记.md | 894-985 | 982 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:982-984 lines 982-984 reason=overlong_section chars=227 | courses/js/Servlet-JSP-课堂笔记.md:982-1016 lines 982-1016 reason=overlong_section chars=1574 |
| courses/js/Servlet-JSP-课堂笔记.md | 1016-1083 | 1017 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:982-1016 lines 982-1016 reason=overlong_section chars=1574 | courses/js/Servlet-JSP-课堂笔记.md:1014-1079 lines 1014-1079 reason=oversized_code_block chars=1424 |
| courses/js/Servlet-JSP-课堂笔记.md | 1016-1083 | 1080 | chunk_end_next_line,chunk_start | False | courses/js/Servlet-JSP-课堂笔记.md:1014-1079 lines 1014-1079 reason=oversized_code_block chars=1424 | courses/js/Servlet-JSP-课堂笔记.md:1080-1132 lines 1080-1132 reason=oversized_code_block chars=1521 |
| courses/js/Servlet-JSP-课堂笔记.md | 1135-1263 | 1162 | chunk_end_next_line,chunk_start | False | courses/js/Servlet-JSP-课堂笔记.md:1133-1161 lines 1133-1161 reason=oversized_code_block chars=1496 | courses/js/Servlet-JSP-课堂笔记.md:1162-1194 lines 1162-1194 reason=overlong_section chars=1480 |
| courses/js/Servlet-JSP-课堂笔记.md | 1135-1263 | 1195 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:1162-1194 lines 1162-1194 reason=overlong_section chars=1480 | courses/js/Servlet-JSP-课堂笔记.md:1188-1224 lines 1188-1224 reason=overlong_section chars=707 |
| courses/js/Servlet-JSP-课堂笔记.md | 1135-1263 | 1188 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:1162-1194 lines 1162-1194 reason=overlong_section chars=1480 | courses/js/Servlet-JSP-课堂笔记.md:1188-1224 lines 1188-1224 reason=overlong_section chars=707 |
| courses/js/Servlet-JSP-课堂笔记.md | 1135-1263 | 1225 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:1188-1224 lines 1188-1224 reason=overlong_section chars=707 | courses/js/Servlet-JSP-课堂笔记.md:1215-1256 lines 1215-1256 reason=overlong_section chars=1480 |
| courses/js/Servlet-JSP-课堂笔记.md | 1135-1263 | 1215 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:1188-1224 lines 1188-1224 reason=overlong_section chars=707 | courses/js/Servlet-JSP-课堂笔记.md:1215-1256 lines 1215-1256 reason=overlong_section chars=1480 |
| courses/js/Servlet-JSP-课堂笔记.md | 1135-1263 | 1257 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:1215-1256 lines 1215-1256 reason=overlong_section chars=1480 | courses/js/Servlet-JSP-课堂笔记.md:1249-1262 lines 1249-1262 reason=overlong_section chars=416 |
| courses/js/Servlet-JSP-课堂笔记.md | 1135-1263 | 1249 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:1215-1256 lines 1215-1256 reason=overlong_section chars=1480 | courses/js/Servlet-JSP-课堂笔记.md:1249-1262 lines 1249-1262 reason=overlong_section chars=416 |
| courses/js/Servlet-JSP-课堂笔记.md | 1135-1263 | 1263 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:1255-1262 lines 1255-1262 reason=overlong_section chars=194 | courses/js/Servlet-JSP-课堂笔记.md:1255-1425 lines 1255-1425 reason=overlong_section chars=5538 |
| courses/js/Servlet-JSP-课堂笔记.md | 1135-1263 | 1255 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:1255-1262 lines 1255-1262 reason=overlong_section chars=194 | courses/js/Servlet-JSP-课堂笔记.md:1255-1425 lines 1255-1425 reason=overlong_section chars=5538 |
| courses/js/Servlet-JSP-课堂笔记.md | 1425-1458 | 1426 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:1255-1425 lines 1255-1425 reason=overlong_section chars=5538 | courses/js/Servlet-JSP-课堂笔记.md:1416-1474 lines 1416-1474 reason=overlong_section chars=1379 |
| courses/js/Servlet-JSP-课堂笔记.md | 1511-1542 | 1537 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:1506-1546 lines 1506-1546 reason=overlong_section chars=1193 | courses/js/Servlet-JSP-课堂笔记.md:1537-1569 lines 1537-1569 reason=section_end chars=890 |
| courses/js/Servlet-JSP-课堂笔记.md | 1665-1703 | 1699 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:1661-1703 lines 1661-1703 reason=overlong_section chars=1547 | courses/js/Servlet-JSP-课堂笔记.md:1699-1735 lines 1699-1735 reason=overlong_section chars=1064 |
| courses/js/Servlet-JSP-课堂笔记.md | 1717-1736 | 1736 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:1699-1735 lines 1699-1735 reason=overlong_section chars=1064 | courses/js/Servlet-JSP-课堂笔记.md:1730-1767 lines 1730-1767 reason=overlong_section chars=1186 |
| courses/js/Servlet-JSP-课堂笔记.md | 1717-1736 | 1730 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:1699-1735 lines 1699-1735 reason=overlong_section chars=1064 | courses/js/Servlet-JSP-课堂笔记.md:1730-1767 lines 1730-1767 reason=overlong_section chars=1186 |
| courses/js/Servlet-JSP-课堂笔记.md | 1761-1802 | 1768 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:1730-1767 lines 1730-1767 reason=overlong_section chars=1186 | courses/js/Servlet-JSP-课堂笔记.md:1759-1803 lines 1759-1803 reason=section_end chars=1502 |
| courses/js/Servlet-JSP-课堂笔记.md | 1826-1871 | 1863 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:1821-1873 lines 1821-1873 reason=overlong_section chars=1314 | courses/js/Servlet-JSP-课堂笔记.md:1863-1904 lines 1863-1904 reason=overlong_section chars=1358 |
| courses/js/Servlet-JSP-课堂笔记.md | 1885-1905 | 1905 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:1863-1904 lines 1863-1904 reason=overlong_section chars=1358 | courses/js/Servlet-JSP-课堂笔记.md:1899-1941 lines 1899-1941 reason=overlong_section chars=1331 |
| courses/js/Servlet-JSP-课堂笔记.md | 1885-1905 | 1899 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:1863-1904 lines 1863-1904 reason=overlong_section chars=1358 | courses/js/Servlet-JSP-课堂笔记.md:1899-1941 lines 1899-1941 reason=overlong_section chars=1331 |
| courses/js/Servlet-JSP-课堂笔记.md | 1932-1942 | 1942 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:1899-1941 lines 1899-1941 reason=overlong_section chars=1331 | courses/js/Servlet-JSP-课堂笔记.md:1930-1981 lines 1930-1981 reason=overlong_section chars=1361 |
| courses/js/Servlet-JSP-课堂笔记.md | 2031-2053 | 2047 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:2011-2055 lines 2011-2055 reason=overlong_section chars=1328 | courses/js/Servlet-JSP-课堂笔记.md:2047-2101 lines 2047-2101 reason=overlong_section chars=1346 |
| courses/js/Servlet-JSP-课堂笔记.md | 2068-2108 | 2102 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:2047-2101 lines 2047-2101 reason=overlong_section chars=1346 | courses/js/Servlet-JSP-课堂笔记.md:2097-2141 lines 2097-2141 reason=overlong_section chars=1447 |
| courses/js/Servlet-JSP-课堂笔记.md | 2068-2108 | 2097 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:2047-2101 lines 2047-2101 reason=overlong_section chars=1346 | courses/js/Servlet-JSP-课堂笔记.md:2097-2141 lines 2097-2141 reason=overlong_section chars=1447 |
| courses/js/Servlet-JSP-课堂笔记.md | 2118-2183 | 2142 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:2097-2141 lines 2097-2141 reason=overlong_section chars=1447 | courses/js/Servlet-JSP-课堂笔记.md:2137-2182 lines 2137-2182 reason=overlong_section chars=1138 |
| courses/js/Servlet-JSP-课堂笔记.md | 2118-2183 | 2137 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:2097-2141 lines 2097-2141 reason=overlong_section chars=1447 | courses/js/Servlet-JSP-课堂笔记.md:2137-2182 lines 2137-2182 reason=overlong_section chars=1138 |
| courses/js/Servlet-JSP-课堂笔记.md | 2118-2183 | 2183 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:2137-2182 lines 2137-2182 reason=overlong_section chars=1138 | courses/js/Servlet-JSP-课堂笔记.md:2172-2197 lines 2172-2197 reason=section_end chars=649 |
| courses/js/Servlet-JSP-课堂笔记.md | 2118-2183 | 2172 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:2137-2182 lines 2137-2182 reason=overlong_section chars=1138 | courses/js/Servlet-JSP-课堂笔记.md:2172-2197 lines 2172-2197 reason=section_end chars=649 |
| courses/js/Servlet-JSP-课堂笔记.md | 2302-2387 | 2379 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:2292-2387 lines 2292-2387 reason=overlong_section chars=3226 | courses/js/Servlet-JSP-课堂笔记.md:2379-2406 lines 2379-2406 reason=section_end chars=670 |
| courses/js/Servlet-JSP-课堂笔记.md | 2541-2608 | 2604 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:2535-2608 lines 2535-2608 reason=overlong_section chars=1969 | courses/js/Servlet-JSP-课堂笔记.md:2604-2659 lines 2604-2659 reason=overlong_section chars=1482 |
| courses/js/Servlet-JSP-课堂笔记.md | 2638-2724 | 2660 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:2604-2659 lines 2604-2659 reason=overlong_section chars=1482 | courses/js/Servlet-JSP-课堂笔记.md:2652-2709 lines 2652-2709 reason=overlong_section chars=1437 |
| courses/js/Servlet-JSP-课堂笔记.md | 2638-2724 | 2652 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:2604-2659 lines 2604-2659 reason=overlong_section chars=1482 | courses/js/Servlet-JSP-课堂笔记.md:2652-2709 lines 2652-2709 reason=overlong_section chars=1437 |
| courses/js/Servlet-JSP-课堂笔记.md | 2638-2724 | 2710 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:2652-2709 lines 2652-2709 reason=overlong_section chars=1437 | courses/js/Servlet-JSP-课堂笔记.md:2704-2723 lines 2704-2723 reason=overlong_section chars=701 |
| courses/js/Servlet-JSP-课堂笔记.md | 2638-2724 | 2704 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:2652-2709 lines 2652-2709 reason=overlong_section chars=1437 | courses/js/Servlet-JSP-课堂笔记.md:2704-2723 lines 2704-2723 reason=overlong_section chars=701 |
| courses/js/Servlet-JSP-课堂笔记.md | 2638-2724 | 2724 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:2704-2723 lines 2704-2723 reason=overlong_section chars=701 | courses/js/Servlet-JSP-课堂笔记.md:2717-2774 lines 2717-2774 reason=overlong_section chars=1491 |
| courses/js/Servlet-JSP-课堂笔记.md | 2638-2724 | 2717 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:2704-2723 lines 2704-2723 reason=overlong_section chars=701 | courses/js/Servlet-JSP-课堂笔记.md:2717-2774 lines 2717-2774 reason=overlong_section chars=1491 |
| courses/js/Servlet-JSP-课堂笔记.md | 2767-2837 | 2775 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:2717-2774 lines 2717-2774 reason=overlong_section chars=1491 | courses/js/Servlet-JSP-课堂笔记.md:2769-2819 lines 2769-2819 reason=overlong_section chars=1468 |
| courses/js/Servlet-JSP-课堂笔记.md | 2767-2837 | 2769 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:2717-2774 lines 2717-2774 reason=overlong_section chars=1491 | courses/js/Servlet-JSP-课堂笔记.md:2769-2819 lines 2769-2819 reason=overlong_section chars=1468 |
| courses/js/Servlet-JSP-课堂笔记.md | 2767-2837 | 2820 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:2769-2819 lines 2769-2819 reason=overlong_section chars=1468 | courses/js/Servlet-JSP-课堂笔记.md:2809-2852 lines 2809-2852 reason=overlong_section chars=1114 |
| courses/js/Servlet-JSP-课堂笔记.md | 2767-2837 | 2809 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:2769-2819 lines 2769-2819 reason=overlong_section chars=1468 | courses/js/Servlet-JSP-课堂笔记.md:2809-2852 lines 2809-2852 reason=overlong_section chars=1114 |
| courses/js/Servlet-JSP-课堂笔记.md | 2850-2873 | 2853 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:2809-2852 lines 2809-2852 reason=overlong_section chars=1114 | courses/js/Servlet-JSP-课堂笔记.md:2845-2886 lines 2845-2886 reason=overlong_section chars=1260 |
| courses/js/Servlet-JSP-课堂笔记.md | 2958-3057 | 3050 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:2952-3057 lines 2952-3057 reason=overlong_section chars=2632 | courses/js/Servlet-JSP-课堂笔记.md:3050-3097 lines 3050-3097 reason=overlong_section chars=1435 |
| courses/js/Servlet-JSP-课堂笔记.md | 3065-3128 | 3098 | chunk_end_next_line | True | courses/js/Servlet-JSP-课堂笔记.md:3050-3097 lines 3050-3097 reason=overlong_section chars=1435 | courses/js/Servlet-JSP-课堂笔记.md:3086-3128 lines 3086-3128 reason=section_end chars=874 |
| courses/js/Servlet-JSP-课堂笔记.md | 3065-3128 | 3086 | chunk_start | True | courses/js/Servlet-JSP-课堂笔记.md:3050-3097 lines 3050-3097 reason=overlong_section chars=1435 | courses/js/Servlet-JSP-课堂笔记.md:3086-3128 lines 3086-3128 reason=section_end chars=874 |
| courses/js/Servlet.md | 309-334 | 326 | chunk_start | True | courses/js/Servlet.md:290-334 lines 290-334 reason=overlong_section chars=1223 | courses/js/Servlet.md:326-363 lines 326-363 reason=section_end chars=1012 |
| courses/js/前端基础/Dom.md | 68-76 | 73 | chunk_start | True | courses/js/前端基础/Dom.md:52-79 lines 52-79 reason=overlong_section chars=873 | courses/js/前端基础/Dom.md:73-110 lines 73-110 reason=section_end chars=1001 |
| courses/react/Notes.md | 53-74 | 69 | chunk_start | True | courses/react/Notes.md:1-76 lines 1-76 reason=overlong_section chars=1498 | courses/react/Notes.md:69-90 lines 69-90 reason=section_end chars=382 |
| courses/react/Notes.md | 149-165 | 154 | chunk_start | True | courses/react/Notes.md:91-167 lines 91-167 reason=overlong_section chars=1327 | courses/react/Notes.md:154-210 lines 154-210 reason=overlong_section chars=1385 |
| courses/react/Notes.md | 200-215 | 211 | chunk_end_next_line | True | courses/react/Notes.md:154-210 lines 154-210 reason=overlong_section chars=1385 | courses/react/Notes.md:203-224 lines 203-224 reason=section_end chars=598 |
| courses/react/Notes.md | 200-215 | 203 | chunk_start | True | courses/react/Notes.md:154-210 lines 154-210 reason=overlong_section chars=1385 | courses/react/Notes.md:203-224 lines 203-224 reason=section_end chars=598 |
| courses/react/笔记.md | 53-74 | 69 | chunk_start | True | courses/react/笔记.md:1-76 lines 1-76 reason=overlong_section chars=1498 | courses/react/笔记.md:69-90 lines 69-90 reason=section_end chars=382 |
| courses/react/笔记.md | 149-165 | 154 | chunk_start | True | courses/react/笔记.md:91-167 lines 91-167 reason=overlong_section chars=1327 | courses/react/笔记.md:154-210 lines 154-210 reason=overlong_section chars=1385 |
| courses/react/笔记.md | 200-215 | 211 | chunk_end_next_line | True | courses/react/笔记.md:154-210 lines 154-210 reason=overlong_section chars=1385 | courses/react/笔记.md:203-224 lines 203-224 reason=section_end chars=598 |
| courses/react/笔记.md | 200-215 | 203 | chunk_start | True | courses/react/笔记.md:154-210 lines 154-210 reason=overlong_section chars=1385 | courses/react/笔记.md:203-224 lines 203-224 reason=section_end chars=598 |
| courses/期末复习/spark.md | 50-67 | 53 | chunk_start | True | courses/期末复习/spark.md:1-67 lines 1-67 reason=overlong_section chars=1183 | courses/期末复习/spark.md:53-83 lines 53-83 reason=section_end chars=818 |
| courses/期末复习/spark.md | 121-126 | 124 | chunk_start | True | courses/期末复习/spark.md:84-132 lines 84-132 reason=overlong_section chars=1394 | courses/期末复习/spark.md:124-146 lines 124-146 reason=overlong_section chars=536 |
| courses/期末复习/spark.md | 134-147 | 147 | chunk_end_next_line | True | courses/期末复习/spark.md:124-146 lines 124-146 reason=overlong_section chars=536 | courses/期末复习/spark.md:137-175 lines 137-175 reason=overlong_section chars=1351 |
| courses/期末复习/spark.md | 134-147 | 137 | chunk_start | True | courses/期末复习/spark.md:124-146 lines 124-146 reason=overlong_section chars=536 | courses/期末复习/spark.md:137-175 lines 137-175 reason=overlong_section chars=1351 |
| courses/期末复习/spark.md | 175-181 | 176 | chunk_end_next_line | True | courses/期末复习/spark.md:137-175 lines 137-175 reason=overlong_section chars=1351 | courses/期末复习/spark.md:169-190 lines 169-190 reason=section_end chars=637 |
| courses/期末复习/网安提纲.md | 364-384 | 373 | chunk_start | True | courses/期末复习/网安提纲.md:340-385 lines 340-385 reason=overlong_section chars=1007 | courses/期末复习/网安提纲.md:373-412 lines 373-412 reason=section_end chars=714 |
| courses/网安课程复习/网安提纲.md | 364-384 | 373 | chunk_start | True | courses/网安课程复习/网安提纲.md:340-385 lines 340-385 reason=overlong_section chars=1007 | courses/网安课程复习/网安提纲.md:373-412 lines 373-412 reason=section_end chars=714 |
| docs/fastapi/advanced/additional-responses.md | 54-89 | 81 | chunk_start | True | docs/fastapi/advanced/additional-responses.md:47-91 lines 47-91 reason=overlong_section chars=1171 | docs/fastapi/advanced/additional-responses.md:81-123 lines 81-123 reason=overlong_section chars=1066 |
| docs/fastapi/advanced/additional-responses.md | 93-170 | 124 | chunk_end_next_line | True | docs/fastapi/advanced/additional-responses.md:81-123 lines 81-123 reason=overlong_section chars=1066 | docs/fastapi/advanced/additional-responses.md:118-150 lines 118-150 reason=overlong_section chars=1051 |
| docs/fastapi/advanced/additional-responses.md | 93-170 | 118 | chunk_start | True | docs/fastapi/advanced/additional-responses.md:81-123 lines 81-123 reason=overlong_section chars=1066 | docs/fastapi/advanced/additional-responses.md:118-150 lines 118-150 reason=overlong_section chars=1051 |
| docs/fastapi/advanced/additional-responses.md | 93-170 | 151 | chunk_end_next_line | True | docs/fastapi/advanced/additional-responses.md:118-150 lines 118-150 reason=overlong_section chars=1051 | docs/fastapi/advanced/additional-responses.md:145-171 lines 145-171 reason=section_end chars=735 |
| docs/fastapi/advanced/additional-responses.md | 93-170 | 145 | chunk_start | True | docs/fastapi/advanced/additional-responses.md:118-150 lines 118-150 reason=overlong_section chars=1051 | docs/fastapi/advanced/additional-responses.md:145-171 lines 145-171 reason=section_end chars=735 |
| docs/fastapi/advanced/settings.md | 249-288 | 285 | chunk_start | True | docs/fastapi/advanced/settings.md:233-292 lines 233-292 reason=overlong_section chars=1404 | docs/fastapi/advanced/settings.md:285-295 lines 285-295 reason=section_end chars=356 |
| docs/fastapi/deployment/manually.md | 9-37 | 29 | chunk_end_next_line,chunk_start | False | docs/fastapi/deployment/manually.md:1-28 lines 1-28 reason=oversized_code_block chars=1671 | docs/fastapi/deployment/manually.md:29-44 lines 29-44 reason=section_end chars=739 |
| docs/fastapi/deployment/server-workers.md | 38-76 | 58 | chunk_end_next_line,chunk_start | False | docs/fastapi/deployment/server-workers.md:28-57 lines 28-57 reason=oversized_code_block chars=1657 | docs/fastapi/deployment/server-workers.md:58-67 lines 58-67 reason=overlong_section chars=1100 |
| docs/fastapi/deployment/server-workers.md | 38-76 | 68 | chunk_end_next_line | True | docs/fastapi/deployment/server-workers.md:58-67 lines 58-67 reason=overlong_section chars=1100 | docs/fastapi/deployment/server-workers.md:66-88 lines 66-88 reason=overlong_section chars=1387 |
| docs/fastapi/deployment/server-workers.md | 38-76 | 66 | chunk_start | True | docs/fastapi/deployment/server-workers.md:58-67 lines 58-67 reason=overlong_section chars=1100 | docs/fastapi/deployment/server-workers.md:66-88 lines 66-88 reason=overlong_section chars=1387 |
| docs/fastapi/deployment/server-workers.md | 38-76 | 75 | chunk_start | True | docs/fastapi/deployment/server-workers.md:66-88 lines 66-88 reason=overlong_section chars=1387 | docs/fastapi/deployment/server-workers.md:75-103 lines 75-103 reason=overlong_section chars=1443 |
| docs/fastapi/deployment/server-workers.md | 88-104 | 89 | chunk_end_next_line | True | docs/fastapi/deployment/server-workers.md:66-88 lines 66-88 reason=overlong_section chars=1387 | docs/fastapi/deployment/server-workers.md:75-103 lines 75-103 reason=overlong_section chars=1443 |
| docs/fastapi/deployment/server-workers.md | 88-104 | 104 | chunk_end_next_line | True | docs/fastapi/deployment/server-workers.md:75-103 lines 75-103 reason=overlong_section chars=1443 | docs/fastapi/deployment/server-workers.md:101-113 lines 101-113 reason=section_end chars=407 |
| docs/fastapi/deployment/server-workers.md | 88-104 | 101 | chunk_start | True | docs/fastapi/deployment/server-workers.md:75-103 lines 75-103 reason=overlong_section chars=1443 | docs/fastapi/deployment/server-workers.md:101-113 lines 101-113 reason=section_end chars=407 |
| docs/fastapi/environment-variables.md | 138-150 | 139 | chunk_start | True | docs/fastapi/environment-variables.md:53-154 lines 53-154 reason=overlong_section chars=1439 | docs/fastapi/environment-variables.md:139-159 lines 139-159 reason=section_end chars=290 |
| docs/fastapi/fastapi-cli.md | 11-45 | 43 | chunk_start | True | docs/fastapi/fastapi-cli.md:3-45 lines 3-45 reason=overlong_section chars=3006 | docs/fastapi/fastapi-cli.md:43-60 lines 43-60 reason=section_end chars=497 |
| docs/fastapi/tutorial/first-steps.md | 13-47 | 33 | chunk_end_next_line,chunk_start | False | docs/fastapi/tutorial/first-steps.md:1-32 lines 1-32 reason=oversized_code_block chars=1624 | docs/fastapi/tutorial/first-steps.md:33-58 lines 33-58 reason=section_end chars=1416 |
| docs/fastapi/tutorial/index.md | 17-51 | 37 | chunk_end_next_line,chunk_start | False | docs/fastapi/tutorial/index.md:9-36 lines 9-36 reason=oversized_code_block chars=1608 | docs/fastapi/tutorial/index.md:37-60 lines 37-60 reason=section_end chars=1367 |
| docs/fastapi/virtual-environments.md | 693-695 | 694 | chunk_start | True | docs/fastapi/virtual-environments.md:600-709 lines 600-709 reason=overlong_section chars=1428 | docs/fastapi/virtual-environments.md:694-736 lines 694-736 reason=section_end chars=717 |
| imported_docs/textbooks-mineru-agent/操作系统导论/04.md | 150-164 | 163 | chunk_start | True | imported_docs/textbooks-mineru-agent/操作系统导论/04.md:124-172 lines 124-172 reason=overlong_section chars=1412 | imported_docs/textbooks-mineru-agent/操作系统导论/04.md:163-175 lines 163-175 reason=section_end chars=533 |
| imported_docs/textbooks-mineru-agent/操作系统导论/05.md | 128-164 | 159 | chunk_start | True | imported_docs/textbooks-mineru-agent/操作系统导论/05.md:124-167 lines 124-167 reason=overlong_section chars=1259 | imported_docs/textbooks-mineru-agent/操作系统导论/05.md:159-170 lines 159-170 reason=section_end chars=464 |
| 全栈ai/技术栈/fastapi.md | 70-91 | 75 | chunk_start | True | 全栈ai/技术栈/fastapi.md:2-91 lines 2-91 reason=overlong_section chars=1439 | 全栈ai/技术栈/fastapi.md:75-138 lines 75-138 reason=overlong_section chars=1360 |
| 全栈ai/技术栈/fastapi.md | 138-143 | 139 | chunk_end_next_line | True | 全栈ai/技术栈/fastapi.md:75-138 lines 75-138 reason=overlong_section chars=1360 | 全栈ai/技术栈/fastapi.md:130-168 lines 130-168 reason=overlong_section chars=1088 |
| 全栈ai/技术栈/fastapi.md | 154-167 | 161 | chunk_start | True | 全栈ai/技术栈/fastapi.md:130-168 lines 130-168 reason=overlong_section chars=1088 | 全栈ai/技术栈/fastapi.md:161-232 lines 161-232 reason=section_end chars=1480 |
| 全栈ai/技术栈/postgreSQL.md | 202-206 | 205 | chunk_start | True | 全栈ai/技术栈/postgreSQL.md:94-230 lines 94-230 reason=overlong_section chars=1491 | 全栈ai/技术栈/postgreSQL.md:205-336 lines 205-336 reason=overlong_section chars=1396 |
| 全栈ai/技术栈/postgreSQL.md | 323-332 | 324 | chunk_start | True | 全栈ai/技术栈/postgreSQL.md:205-336 lines 205-336 reason=overlong_section chars=1396 | 全栈ai/技术栈/postgreSQL.md:324-383 lines 324-383 reason=overlong_section chars=1191 |
| 全栈ai/技术栈/postgreSQL.md | 336-343 | 337 | chunk_end_next_line | True | 全栈ai/技术栈/postgreSQL.md:205-336 lines 205-336 reason=overlong_section chars=1396 | 全栈ai/技术栈/postgreSQL.md:324-383 lines 324-383 reason=overlong_section chars=1191 |
| 全栈ai/技术栈/postgreSQL.md | 383-400 | 384 | chunk_end_next_line | True | 全栈ai/技术栈/postgreSQL.md:324-383 lines 324-383 reason=overlong_section chars=1191 | 全栈ai/技术栈/postgreSQL.md:378-400 lines 378-400 reason=section_end chars=627 |
| 全栈ai/技术栈/全栈路径.md | 382-389 | 383 | chunk_start | True | 全栈ai/技术栈/全栈路径.md:301-400 lines 301-400 reason=overlong_section chars=1460 | 全栈ai/技术栈/全栈路径.md:383-435 lines 383-435 reason=overlong_section chars=552 |
| 全栈ai/技术栈/全栈路径.md | 426-436 | 436 | chunk_end_next_line | True | 全栈ai/技术栈/全栈路径.md:383-435 lines 383-435 reason=overlong_section chars=552 | 全栈ai/技术栈/全栈路径.md:416-524 lines 416-524 reason=overlong_section chars=1492 |

## Notes With Possible Code Boundary Issues

| note | source | chunks | sections | max chars | split chunks | overlap candidates | code issue |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| courses/js/Dom.md | personal_note | 6 | 5 | 1001 | 1 | 1 | True |
| courses/js/JavaWeb后端/Servlet-JSP-参考.md | personal_note | 355 | 25 | 5538 | 327 | 331 | True |
| courses/js/JavaWeb后端/Servlet.md | personal_note | 24 | 27 | 1223 | 1 | 1 | True |
| courses/js/Servlet-JSP-课堂笔记.md | personal_note | 355 | 25 | 5538 | 327 | 331 | True |
| courses/js/Servlet.md | personal_note | 23 | 26 | 1223 | 1 | 1 | True |
| courses/js/前端基础/Dom.md | personal_note | 6 | 5 | 1001 | 1 | 1 | True |
| courses/react/Notes.md | personal_note | 5 | 2 | 1498 | 3 | 3 | True |
| courses/react/笔记.md | personal_note | 5 | 2 | 1498 | 3 | 3 | True |
| courses/期末复习/spark.md | personal_note | 10 | 6 | 1394 | 4 | 4 | True |
| courses/期末复习/网安提纲.md | personal_note | 31 | 32 | 1219 | 2 | 2 | True |
| courses/网安课程复习/网安提纲.md | personal_note | 31 | 32 | 1219 | 2 | 2 | True |
| docs/fastapi/advanced/additional-responses.md | official_docs | 10 | 6 | 1171 | 4 | 4 | True |
| docs/fastapi/advanced/settings.md | official_docs | 19 | 18 | 1404 | 1 | 1 | True |
| docs/fastapi/deployment/manually.md | official_docs | 7 | 7 | 1671 | 0 | 1 | True |
| docs/fastapi/deployment/server-workers.md | official_docs | 9 | 5 | 1657 | 3 | 4 | True |
| docs/fastapi/environment-variables.md | official_docs | 8 | 7 | 1439 | 1 | 1 | True |
| docs/fastapi/fastapi-cli.md | official_docs | 23 | 5 | 3006 | 18 | 18 | True |
| docs/fastapi/tutorial/first-steps.md | official_docs | 26 | 27 | 1624 | 0 | 1 | True |
| docs/fastapi/tutorial/index.md | official_docs | 5 | 4 | 1608 | 0 | 1 | True |
| docs/fastapi/virtual-environments.md | official_docs | 25 | 23 | 1428 | 2 | 2 | True |
| imported_docs/textbooks-mineru-agent/操作系统导论/04.md | textbook_pdf | 22 | 14 | 2837 | 9 | 9 | True |
| imported_docs/textbooks-mineru-agent/操作系统导论/05.md | textbook_pdf | 18 | 14 | 1490 | 4 | 4 | True |
| 全栈ai/技术栈/fastapi.md | personal_note | 5 | 2 | 1480 | 3 | 3 | True |
| 全栈ai/技术栈/postgreSQL.md | personal_note | 6 | 1 | 1492 | 5 | 5 | True |
| 全栈ai/技术栈/全栈路径.md | personal_note | 21 | 16 | 1492 | 5 | 5 | True |

