# 第一阶段交接说明

> 这份文档用于后续新会话 handoff：说明第一阶段已经完成什么、关键文件在哪里、如何验证，以及下一阶段建议做什么。

**目标：** 完成 DataCenter-HVAC Copilot 的第一阶段项目启动，让后续可以扩展到完整的 BEAR-based RAG + Agent + 工具调用 + 评测系统，而不需要重写基础结构。

**架构：** 共享 schema 放在 `src/core/`，BEAR 标准化逻辑放在 `src/ingestion/`，确定性时序分析工具放在 `src/tools/`，policy adapter 放在 `src/policies/`，文档加载、chunk、关键词检索、BM25-style hybrid 检索、metadata-aware 轻量 reranker wrapper 和 extractive RAG baseline 放在 `src/retrieval/`，deterministic router 和 baseline orchestrator 放在 `src/agent/`，评测集读取、指标和 baseline runner 放在 `src/evaluation/`，FastAPI 服务雏形放在 `src/api/`，Streamlit demo 放在 `app/`。

**技术栈：** Python 3.10+、pandas、numpy、pydantic、matplotlib、pytest。

**第二阶段起步更新：** API、orchestrator 和 Streamlit demo 已能展示当前 demo 轨迹数据源；评测 runner 已提供 `llm_only`、`rag_keyword`、`rag_hybrid`、`rag_hybrid_rerank`、`rag`、`rag_tool_agent` 多组 baseline comparison summary 和按任务类型分组指标；`data/eval/hvac_eval.jsonl` 已扩展到 49 条样例，且全部样例包含人工维护的 `expected_keywords`，代表性样例包含 `must_include` / `must_not_include` 质量代理标注；demo RAG 已支持加载 `data/documents/` 下的多篇 UTF-8 Markdown/TXT 文档，并补充了相似主题内部文档、长噪声/短目标文档对和 metadata-aware reranking 压力文档，用于早期检索压力测试；Streamlit demo 已扩展为 Copilot / 评测摘要双页，支持展示 route、tools、citations、tool results、metric summary、时序趋势图和 eval metrics。

---

## 任务完成情况

### 任务 1：项目基础结构

**相关文件：**

- `pyproject.toml`
- `.env.example`
- `.gitignore`
- `README.md`
- `docs/system_design.md`
- `docs/stage_1_handoff.md`

**完成内容：**

- [x] 创建推荐项目目录结构。
- [x] 初始化 Python 项目配置和 pytest 配置。
- [x] 添加基础依赖建议，不引入无关大型框架。
- [x] 添加环境变量模板。
- [x] 写明 BEAR 仿真数据边界和后续扩展方向。

### 任务 2：BEAR schema 和标准化

**相关文件：**

- `src/core/schemas.py`
- `src/ingestion/bear_schema.py`
- `tests/test_bear_schema.py`

**完成内容：**

- [x] 定义字段来源枚举：`native`、`derived`、`optional_derived`、`optional_synthetic`。
- [x] 定义标准化 BEAR 轨迹字段。
- [x] 实现 `normalize_bear_trajectory`。
- [x] 保证 `pue`、`humidity`、`it_load`、`chiller_power` 不会被默认视为 BEAR 原生字段。
- [x] 对缺失 optional 字段保留空值，不编造数据。

### 任务 3：第一批时序分析工具

**相关文件：**

- `src/tools/timeseries.py`
- `tests/test_timeseries_tools.py`

**完成内容：**

- [x] 实现 `query_metric`。
- [x] 实现 `compare_period`。
- [x] 实现 `detect_anomaly`。
- [x] 实现 `compute_energy_breakdown`。
- [x] 实现 `plot_metric_trend`。
- [x] 使用小型 mock BEAR-like 数据完成单元测试。

### 任务 4：policy 接口骨架

**相关文件：**

- `src/policies/base.py`
- `src/policies/rule_based.py`
- `src/policies/mpc_like.py`
- `src/policies/diffusion_adapter.py`
- `src/policies/offline_replay.py`
- `tests/test_policies.py`

**完成内容：**

- [x] 定义共享 `PolicyResult` 结构。
- [x] 实现 `rule_based_policy` fallback。
- [x] 实现 `mpc_like_policy` placeholder。
- [x] 实现 `diffusion_policy_adapter` 接口边界。
- [x] diffusion adapter 未配置真实后端时显式失败，不伪造 DiffFNO / Guided-DiffFNO 效果。
- [x] 实现 `offline_replay_policy`，从已保存 JSON 实验结果读取 policy 输出。

### 任务 5：最小评测样例

**相关文件：**

- `data/eval/hvac_eval.jsonl`

**完成内容：**

- [x] 添加 49 条 JSONL 样例。
- [x] 覆盖文档问答、时序查询、异常诊断和策略建议。
- [x] 每条样例包含 `question`、`task_type`、`gold_answer`、`required_tools`、`required_documents`、`expected_output_format` 和人工维护的 `expected_keywords`；代表性样例补充 `must_include` / `must_not_include` 质量代理标注。

### 任务 6：文档入库与最小检索 baseline

**相关文件：**

- `src/retrieval/schemas.py`
- `src/retrieval/loader.py`
- `src/retrieval/chunking.py`
- `src/retrieval/retriever.py`
- `data/documents/sample_hvac_guidance.md`
- `tests/test_retrieval_pipeline.py`

**完成内容：**

- [x] 定义 `DocumentMetadata`、`SourceDocument`、`DocumentChunk`。
- [x] 每个 chunk 保留 `source_id`、`title`、`source_path`、`published_at`、`category`、`section`、`chunk_id` 等 citation metadata。
- [x] 实现 UTF-8 Markdown / text 文档加载。
- [x] 实现基础 word-based chunker。
- [x] 实现可替换的 `KeywordRetriever` top-k baseline。
- [x] 实现轻量 `HybridRetriever`，使用 BM25-style 长度归一化并标记 `retrieval_mode = hybrid_bm25`。
- [x] 实现轻量 `RerankingRetriever` wrapper，基于候选检索结果做短语命中、查询覆盖率、citation metadata 命中和原始分数重排，并标记 `retrieval_mode = rerank_keyword_overlap`。
- [x] 添加 13 篇项目内部可追溯 Markdown 样例文档，其中包含相似主题的 containment / setpoint tradeoff 压力测试文档、long-noise / short-target 检索压力文档，以及 supply air reset、sensor drift、return air delta-t 等 metadata-aware reranking 压力文档。
- [x] demo RAG 从 `data/documents/` 加载全部 `.md` / `.txt` 文档。
- [x] 添加检索链路单元测试。

### 任务 7：RAG 回答 baseline 与最小评测

**相关文件：**

- `src/retrieval/rag.py`
- `src/evaluation/dataset.py`
- `src/evaluation/metrics.py`
- `tests/test_rag_baseline.py`
- `tests/test_evaluation.py`

**完成内容：**

- [x] 定义 `RAGAnswer`，包含 `question`、`answer`、`citations`、`retrieved_contexts`。
- [x] 实现 `ExtractiveRAGPipeline`，从检索片段直接生成带引用的 baseline 回答。
- [x] 检索不到证据时返回明确的不确定说明。
- [x] 实现 `EvalRecord` 和 `load_eval_dataset`。
- [x] 实现 `citation_hit_rate`。
- [x] 实现 `context_recall`，检查 required document 是否出现在 top-k retrieved contexts 中。
- [x] 实现 `expected_keyword_coverage`，基于人工维护关键词衡量回答覆盖。
- [x] 实现 `lexical_answer_coverage`，作为不依赖 LLM judge 的弱监督回答覆盖指标。
- [x] 实现 `answer_correctness_proxy` 和 `faithfulness_proxy`，基于代表性样例的 `must_include` / `must_not_include` 做本地确定性质量代理评估。
- [x] 实现 `tool_selection_accuracy`。
- [x] 实现 `tool_execution_success_rate`。
- [x] 实现 `evidence_coverage`。
- [x] 添加 RAG baseline 和 evaluation 单元测试。

### 任务 8：工具调用路由雏形与 baseline runner

**相关文件：**

- `src/agent/router.py`
- `src/agent/orchestrator.py`
- `src/evaluation/runner.py`
- `scripts/run_eval.py`
- `tests/test_agent_orchestrator.py`
- `tests/test_baseline_runner.py`

**完成内容：**

- [x] 定义 deterministic router，支持 `document_qa`、`timeseries_query`、`anomaly_diagnosis`、`policy_recommendation`。
- [x] 实现 `BaselineOrchestrator`，串联 RAG baseline、时序工具和规则 policy fallback。
- [x] 实现 `run_baseline_eval`，读取 eval JSONL 并生成 predictions。
- [x] 实现 `run_baseline_comparison`，对比 `llm_only`、`rag_keyword`、`rag_hybrid`、`rag_hybrid_rerank`、`rag`、`rag_tool_agent` 多组 baseline。
- [x] 实现 `save_predictions_jsonl`，保存 UTF-8 JSONL 预测结果。
- [x] 添加 `scripts/run_eval.py` 作为轻量命令入口，并输出 `baseline_comparison.json` 和 `docs/experiment_report.md`。
- [x] `baseline_comparison.json` 同时保存整体 `summary` 和 `by_task_type` 指标；`docs/experiment_report.md` 渲染全局 baseline 表和按任务类型表。
- [x] 保持当前 runner 使用 demo 文档和 mock trajectory，不伪装成真实 BEAR 或真实数据中心生产数据。

### 任务 9：FastAPI 服务雏形

**相关文件：**

- `src/api/schemas.py`
- `src/api/demo_factory.py`
- `src/api/app.py`
- `tests/test_api_app.py`

**完成内容：**

- [x] 添加 FastAPI 和 uvicorn 依赖。
- [x] 定义 `/health` 接口。
- [x] 定义 `/ask` 接口，调用当前 `BaselineOrchestrator`。
- [x] 定义 `/eval/run` 接口，调用 baseline eval runner。
- [x] `/health` 和 `/ask` 返回只读 `data_source`，标明当前轨迹来自 processed CSV、BEAR sample CSV 还是 mock。
- [x] 抽出 `build_demo_orchestrator`，供 API 和脚本复用。
- [x] 添加 API 单元测试。

### 任务 10：Streamlit demo 雏形

**相关文件：**

- `app/api_client.py`
- `app/streamlit_app.py`
- `tests/test_streamlit_client.py`

**完成内容：**

- [x] 添加 Streamlit 依赖。
- [x] 实现 API client，调用 `/ask` 并处理非 200 响应。
- [x] 实现轻量 Streamlit 页面。
- [x] 页面支持输入问题、选择任务类型和配置 API 地址。
- [x] 页面展示 answer、route、tools、citations、tool_results、retrieved_contexts。
- [x] 页面展示当前 API 返回的数据源标签和路径。
- [x] 页面拆分为 Copilot 和评测摘要两个 tab。
- [x] Copilot tab 将时序工具 summary / records 渲染为表格和折线图。
- [x] 评测摘要 tab 调用 `/eval/run`，展示 metric cards、指标表和 prediction preview。
- [x] 添加 API client 单元测试。

### 任务 11：真实 BEAR 仓库接入雏形

**相关文件：**

- `src/ingestion/bear_adapter.py`
- `scripts/export_bear_data.py`
- `tests/test_bear_adapter.py`

**完成内容：**

- [x] 确认 BEAR 仓库为 `https://github.com/chz056/BEAR.git`。
- [x] 基于真实 `BuildingEnvReal.reset()` / `step()` / `statelist` / `actionlist` 接口设计 adapter。
- [x] 定义 BEAR state 布局：`zone_temperature(n)`、`outdoor_temp(1)`、`solar_irradiance/GHI(n)`、`ground_temp(1)`、`occupancy_power(n)`。
- [x] 实现 `parse_bear_state`。
- [x] 实现 `export_bear_rollout`，导出标准化 trajectory DataFrame。
- [x] 实现 `build_bear_env` 和 `require_bear`。
- [x] 添加 `scripts/export_bear_data.py`，从外部 BEAR clone 导出 rollout CSV。
- [x] 未安装或未指定 BEAR 时给出清晰错误提示。
- [x] 保持 `pue`、`humidity`、`it_load`、`chiller_power` 为 optional，不由 BEAR adapter 编造。

### 任务 12：processed trajectory 优先加载

**相关文件：**

- `src/ingestion/processed_loader.py`
- `src/api/demo_factory.py`
- `tests/test_bear_processed_loader.py`

**完成内容：**

- [x] 实现 `load_processed_bear_trajectory`，读取标准 CSV 并复用 BEAR schema 标准化。
- [x] 让 `build_demo_orchestrator` 优先读取 `data/bear_processed/bear_rollout.csv`。
- [x] 读取不到 processed CSV 时保留 mock trajectory fallback。
- [x] 让 demo factory 可注入 `project_root`，方便测试和后续部署挂载。
- [x] 为 processed CSV 数据源附加 `data_source.kind = processed_csv` 和实际路径。
- [x] 添加 processed trajectory 加载测试。

### 任务 13：BEAR sample CSV 接入 demo fallback

**相关文件：**

- `src/ingestion/bear_sample_loader.py`
- `src/api/demo_factory.py`
- `tests/test_bear_sample_loader.py`

**完成内容：**

- [x] 读取仓库内 `BEAR/BEAR/Data/Exercise2A-mytest.csv`。
- [x] 将 BEAR sample CSV 转成长表标准 trajectory。
- [x] 让 demo factory 在 processed CSV 不存在时优先使用 BEAR sample CSV。
- [x] 最后再 fallback 到 mock trajectory。
- [x] 为 BEAR sample CSV 和 mock fallback 附加只读数据源元数据。
- [x] 添加 BEAR sample loader 测试。

---

## 文件职责总览

- `pyproject.toml`：Python 包元数据、运行依赖、开发依赖和 pytest 配置。
- `.env.example`：本地环境变量模板，包含数据路径以及后续模型、检索配置占位。
- `.gitignore`：忽略 Python 缓存、本地环境、构建产物、原始和处理后的 BEAR 数据。
- `README.md`：项目目标、环境配置、测试命令和目录结构说明。
- `docs/system_design.md`：系统边界、第一阶段架构、BEAR 数据约束和后续扩展方向。
- `docs/experiment_report.md`：由 `scripts/run_eval.py` 生成的 Markdown 实验报告，包含全局 baseline 指标表、按任务类型指标表和数据边界说明。
- `src/core/schemas.py`：共享字段来源枚举、字段定义和统计摘要 helper。
- `src/ingestion/bear_schema.py`：标准 BEAR 轨迹字段定义和 normalize 函数。
- `src/ingestion/bear_adapter.py`：基于 `chz056/BEAR` 的 rollout adapter 和导出逻辑。
- `src/ingestion/processed_loader.py`：读取 `data/bear_processed/` 下的标准 CSV trajectory。
- `src/ingestion/bear_sample_loader.py`：读取仓库内 BEAR 样例 CSV 并转成长表 trajectory。
- `src/tools/timeseries.py`：第一批确定性时序分析工具。
- `src/policies/base.py`：共享 `PolicyResult` 数据模型和 state helper。
- `src/policies/rule_based.py`：规则策略 fallback。
- `src/policies/mpc_like.py`：MPC-like policy 接口占位。
- `src/policies/diffusion_adapter.py`：DiffFNO / Guided-DiffFNO adapter 边界；未配置真实后端时显式失败。
- `src/policies/offline_replay.py`：从已保存实验结果中读取 policy 输出的 offline replay 工具。
- `src/retrieval/schemas.py`：文档、chunk 和 citation metadata 数据结构。
- `src/retrieval/loader.py`：UTF-8 Markdown / text 文档加载。
- `src/retrieval/chunking.py`：基础文档 chunker。
- `src/retrieval/retriever.py`：轻量关键词 top-k 检索 baseline、BM25-style hybrid 检索 baseline 和 metadata-aware lightweight reranker wrapper。
- `src/retrieval/rag.py`：extractive RAG baseline，返回回答、引用和检索上下文。
- `src/agent/router.py`：确定性任务路由。
- `src/agent/orchestrator.py`：baseline orchestrator，串联 RAG、时序工具和 policy fallback，并在输出中携带 `data_source`。
- `src/evaluation/dataset.py`：读取 eval JSONL 并验证记录结构。
- `src/evaluation/metrics.py`：citation hit rate、context recall、expected keyword coverage、lexical answer coverage、tool selection accuracy、tool execution success rate、evidence coverage、answer correctness proxy 和 faithfulness proxy 指标。
- `src/evaluation/runner.py`：baseline eval runner、多组 baseline comparison、按任务类型指标聚合和预测结果保存。
- `src/api/schemas.py`：API 请求和响应数据结构。
- `src/api/demo_factory.py`：构建 demo orchestrator，从 `data/documents/` 加载多文档 RAG 语料，并按 processed CSV、BEAR sample CSV、mock 的顺序加载轨迹并记录数据源。
- `src/api/app.py`：FastAPI app，提供 `/health`、`/ask`、`/eval/run`，并返回数据源元信息。
- `app/api_client.py`：Streamlit demo 使用的 API client，封装 `/ask` 和 `/eval/run`。
- `app/streamlit_app.py`：轻量展示页面，显示回答、路由、工具结果、当前数据源、时序趋势和评测摘要。
- `scripts/run_eval.py`：运行 baseline eval，写出 baseline comparison summary / by-task-type JSON，并生成 Markdown 实验报告。
- `scripts/export_bear_data.py`：从外部 BEAR 仓库导出标准化 rollout CSV。
- `data/documents/*.md`：用于开发和测试的项目内部 HVAC / 控制 / 评测 / 数据边界样例文档，以及相似主题和长噪声/短目标检索压力测试文档。
- `data/eval/hvac_eval.jsonl`：49 条评测 JSONL 样例，全部包含人工维护的 `expected_keywords`，代表性样例包含 `must_include` / `must_not_include` 质量代理标注。
- `tests/test_bear_schema.py`：BEAR 字段来源和标准化测试。
- `tests/test_timeseries_tools.py`：时序查询、周期对比、异常检测、能耗拆分和趋势数据测试。
- `tests/test_policies.py`：policy fallback、diffusion adapter 和 offline replay 测试。
- `tests/test_retrieval_pipeline.py`：文档加载、chunk citation、关键词检索和 hybrid 检索测试。
- `tests/test_rag_baseline.py`：extractive RAG answer schema 和不确定回答测试。
- `tests/test_evaluation.py`：eval JSONL loader 和最小指标测试。
- `tests/test_agent_orchestrator.py`：router 和 baseline orchestrator 测试。
- `tests/test_baseline_runner.py`：baseline eval runner、comparison、预测 JSONL 和报告生成入口测试。
- `tests/test_experiment_report.py`：实验报告 Markdown 渲染和 UTF-8 保存测试。
- `tests/test_api_app.py`：FastAPI 服务接口测试。
- `tests/test_streamlit_client.py`：Streamlit API client 测试。
- `tests/test_bear_adapter.py`：BEAR state 解析、rollout 导出和缺失 BEAR 错误提示测试。
- `tests/test_bear_processed_loader.py`：processed CSV 加载和 demo 优先读取测试。
- `tests/test_bear_sample_loader.py`：BEAR sample CSV 长表转换和 demo fallback 测试。

## 如何运行测试

```bash
python -m pytest
```

当前验证结果：

```text
64 passed
```

最近一次 Streamlit smoke test 使用 `streamlit run app/streamlit_app.py --server.headless true --server.port 8502` 启动后，HTTP 访问 `http://127.0.0.1:8502` 返回 200；FastAPI smoke test 中 `/health` 返回 `ok`，`/ask` 的时序查询返回 route=`timeseries_query` 且 tools=`query_metric`。

当前 `scripts/run_eval.py` 会生成 49 条样例的 baseline comparison，并在 `baseline_comparison.json` 中保存整体 `summary` 和 `by_task_type`。最新报告中 `rag_keyword` 的 `citation_hit_rate` / `context_recall` 为 0.519，`rag_hybrid` 为 0.593，说明长噪声/短目标压力样例继续体现 BM25-style hybrid 检索优势。`rag_hybrid_rerank` 的 citation/context 指标提升到 0.630，说明 metadata-aware 轻量重排已能在当前压力样例中拉开与 `rag_hybrid` 的差异。`rag_tool_agent` 的 `tool_selection_accuracy` 和 `tool_execution_success_rate` 为 1.000，`evidence_coverage` 为 0.878，`answer_correctness_proxy` 为 0.417，`faithfulness_proxy` 为 0.333。按任务类型表显示，工具类任务的 tool selection / execution 已达 1.000，文档问答主要受 citation/context 和回答覆盖率限制；质量代理指标是本地确定性弱指标，不等价于完整人工评审或 LLM judge。

## 本阶段尚未实现

- PDF / 网页文档解析。
- FAISS / Qdrant 向量检索。
- 更强的 neural / cross-encoder / LLM reranker，以及更多能检验 reranker 的真实领域压力样例。
- 真实 LLM 回答生成器。
- LangGraph Agent 工作流。
- 更完整的 Streamlit 运行日志、案例 walkthrough 和截图素材。
- 真实 DiffFNO / Guided-DiffFNO 推理。
- 100 条以上完整评测集。
- 更完整的人工 correctness / faithfulness 标注和可选 LLM judge 报告表格。

## 下一步建议

建议下一阶段做“文档库扩展和检索差异验证”：

- [x] 在现有相似主题文档基础上增加近义/噪声压力问题，观察到 `rag_keyword` 与 `rag_hybrid` 的 context/citation 指标差异。
- [x] 接入轻量 reranker wrapper，并纳入 `rag_hybrid_rerank` baseline comparison。
- [x] 继续增加更多真实领域风格的近义问题，避免当前差异只来自人工构造的长噪声样例。
- [x] 继续设计能区分 `rag_hybrid` 与 `rag_hybrid_rerank` 的真实领域压力样例，或替换为 cross-encoder / LLM reranker。
- [x] 增加基于代表性人工轻标注的 correctness / faithfulness proxy，避免只依赖 extractive coverage。
- [ ] 后续将评测集扩展到 100 条以上，并补充更细的回答质量人工标注。
- [x] 增加按任务类型分组的 baseline 指标表，便于说明文档问答、时序工具、异常诊断和策略建议各自瓶颈。
- [x] 开始增强 Streamlit 图表和评测摘要展示，把 route、tool call、citations、data_source 和 metric summary 做成更适合演示的视图。
- [ ] 继续补充 Streamlit 运行日志时间线、典型案例 walkthrough 和 README 截图说明。
