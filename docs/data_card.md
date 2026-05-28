# Data Card

## 数据组成

本项目使用两类评测材料：

| 数据集 | 路径 | 数量 | 用途 |
| --- | --- | ---: | --- |
| 合成/样例评测集 | `data/eval/hvac_eval.jsonl` | 108 | 规模化回归评测，覆盖文档问答、时序查询、异常诊断、策略建议和多步任务 |
| 真实文档子集 | `data/eval/real_eval.jsonl` | 50 | 基于真实公开 PDF 和 BEAR rollout 的手写评测，用于展示真实知识库检索、干扰文档区分和工具链表现 |

合成/样例评测集用于稳定回归和 baseline 对照；真实文档子集用于验证真实公开文档上传、解析、FAISS 索引、BGE embedding、RRF 融合检索和 tool agent 行为。

## 真实公开文档

以下 7 篇 PDF 通过项目上传接口进入知识库，链路为：上传 API -> 解析 -> SQLite metadata/chunks -> FAISS index -> `/ask` refresh。

| 文件名 | 来源 URL | 类型 | 用途 | 许可/可用性说明 |
| --- | --- | --- | --- | --- |
| `doe_best_practice_guide_data_center_design_2024.pdf` | https://www.energy.gov/sites/default/files/2024-07/best-practice-guide-data-center-design.pdf | 指南/白皮书 | 数据中心节能设计、冷却系统、空气管理、PUE | 美国 DOE/NREL 公开发布材料，适合作为公开技术参考 |
| `ashrae_tc99_power_equipment_thermal_guidelines_2016.pdf` | https://resourcecenter.ashrae.org/File%20Library/Technical%20Resources/Bookstore/ASHRAE_TC0909_Power_White_Paper_22_June_2016_REVISED.pdf | ASHRAE TC 9.9 白皮书 | 数据中心热环境、电力设备热边界、A2/A3/A4 温度等级 | ASHRAE Resource Center 公开可访问白皮书 |
| `uptime_annual_outage_analysis_2024_exec_summary.pdf` | https://datacenter.uptimeinstitute.com/rs/711-RIA-145/images/2024.Resiliency.Survey.ExecSum.pdf | 行业报告摘要 | outage 趋势、故障原因、成本、后果和 resilience | Uptime Institute 公开 executive summary |
| `google_ml_applications_data_center_optimization.pdf` | https://research.google.com/pubs/archive/42542.pdf | Google 公开论文/白皮书 | PUE 优化、运行数据建模、机器学习辅助数据中心优化 | Google Research 公开 PDF |
| `ocp_cooling_efficiency_platform_power_telemetry_2024.pdf` | https://www.opencompute.org/documents/ocp-wp-dcf-improve-data-center-cooling-facility-efficiency-through-platform-power-telemetryr1-0-final-update-pdf | OCP 白皮书 | 平台功耗遥测、IT 负载与冷却设施效率协同 | Open Compute Project 公开白皮书 |
| `ashrae_standard_55_2023_fact_sheet.pdf` | https://www.ashrae.org/file%20library/about/government%20affairs/advocacy%20toolkit/virtual%20packet/standard-55-fact-sheet.pdf | 标准事实表 | 热舒适、室内环境边界、Standard 55 与 TC9.9 区分 | ASHRAE 公开 fact sheet |
| `bear_physics_principled_building_environment_arxiv_2211_14744.pdf` | https://arxiv.org/pdf/2211.14744 | 论文 | BEAR 仿真环境、building control、reinforcement learning 基准 | arXiv 公开论文 |

## 合成 Demo 文档

`data/documents/` 中的 Markdown 文档是项目自带 demo 知识库，用于 108 条合成/样例评测集的可复现回归。它们不是外部真实数据中心文档，主要服务于：

- 稳定覆盖 route、tool、policy boundary、citation/context 等指标；
- 避免真实 PDF 解析和外部文档变动影响基础 CI 回归；
- 为 LangGraph workflow、planner、agent executor 提供小规模可控测试语料。

## BEAR Rollout 数据

当前时序数据来自：

```text
data/bear_processed/bear_rollout.csv
```

关键事实：

- 2016 行；
- 6 个 zone；
- 每个 zone 336 个时间步；
- 时间范围为 `2026-01-01 00:00:00+00:00` 到 `2026-01-14 23:00:00+00:00`；
- 来自 BEAR HVAC 仿真/导出流程，不是生产数据中心遥测。

注意：当前 CSV 中 `PUE`、`humidity`、`it_load` 没有可用观测值。策略或问答不能编造这些字段。

## 评测边界

`data/eval/hvac_eval.jsonl`：

- 108 条；
- 适合做规模化 regression baseline；
- 默认与 `data/documents/` 的 demo 文档 ID 对齐；
- 主要证明 workflow、tool selection、policy boundary、baseline 对照稳定性。

`data/eval/real_eval.jsonl`：

- 50 条；
- 基于真实公开 PDF、当前 BEAR rollout 和项目工具输出手写；
- 包含 30 条文档问答、7 条时序查询、6 条异常诊断、7 条策略建议；
- 文档题包含基础召回、语义整合、相似文档干扰、跨文档整合和文档外边界题；
- 适合展示真实知识库检索能力和系统边界；
- `required_documents` 当前绑定上传后生成的 `document_id`，该 ID 是随机 UUID，不是内容 hash。

后续若要跨机器稳定复现真实子集，应优先改造 citation metric，使 `required_documents` 支持按 `filename` 或 `file_hash` 匹配，而不是只匹配 `citation.source_id`。

## 主要实验结果

真实公开文档 + 50 条真实手写子集，BGE-small-zh + FAISS：

| Mode | Citation / Context | Tool Select | Evidence | Correctness Proxy |
| --- | ---: | ---: | ---: | ---: |
| `rag_dense` | 0.562 | 0.000 | 1.000 | 0.148 |
| `rag_hybrid` BM25 | 0.781 | 0.000 | 0.760 | 0.191 |
| `hybrid_rrf` | 0.812 | 0.000 | 1.000 | 0.205 |
| `rag_tool_agent` | 0.562 | 0.850 | 1.000 | 0.703 |
| `langgraph_tool_agent` | 0.562 | 1.000 | 1.000 | 0.727 |
| `react_agent` | 0.562 | 0.900 | 1.000 | 0.713 |

重构后的真实子集不是为了让系统拿满分，而是为了暴露边界：`hybrid_rrf` 保持最高检索召回，但 citation/context 为 0.812，说明相似文档干扰和多文档题已经拉开区分度。

合成/样例 108 条，BGE-small-zh + FAISS，隔离真实知识库后在 demo 文档上运行：

| Mode | Citation / Context | Tool Select | Evidence | Correctness Proxy |
| --- | ---: | ---: | ---: | ---: |
| `rag_dense` | 0.708 | 0.000 | 1.000 | 0.432 |
| `rag_hybrid` BM25 | 0.523 | 0.000 | 0.620 | 0.344 |
| `hybrid_rrf` | 0.708 | 0.000 | 1.000 | 0.454 |
| `rag_tool_agent` | 0.338 | 0.882 | 0.917 | 0.541 |
| `langgraph_tool_agent` | 0.338 | 0.882 | 0.917 | 0.541 |
| `react_agent` | 0.338 | 0.956 | 0.917 | 0.582 |

## 解析质量观察

- 大多数 PDF 文本可被 `pypdf` 抽取，足以进行第一轮 RAG 和 eval。
- ASHRAE TC9.9、DOE、OCP 文档含目录、封面、表格和页眉页脚，会带来少量 chunk 噪声。
- Google PDF 存在不可见空格/排版空格。
- BEAR 论文为双栏论文，可能有轻微行序混杂。
- Uptime 摘要含版权页眉，控制台需要 UTF-8 输出。

后续优化方向：

- 去页眉页脚；
- 跳过封面和免责声明；
- 清理特殊空格；
- 对表格页进行更稳的 chunk 标注；
- 将真实文档 `document_id` 改为 file-hash 派生或在 metric 中支持 file-hash 匹配。
