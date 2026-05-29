# Optimization Log

本日志用于记录后续把 DataCenter-HVAC Copilot 打磨成 Agent 算法岗简历项目的每个优化模块。每完成一个模块后更新一次，不等全部完成后补写。

## 2026-05-29 Baseline Review

### 优化目标

建立当前项目的客观基线：明确已有能力、简历含金量、与成熟 Agent 项目的差距，以及后续优化优先级。

### 实施过程

- 读取项目目录、依赖配置、README、系统设计、评测报告、data card、真实数据评测记录和核心代码。
- 核对 Agent 编排、工具执行、检索、知识库、记忆、评测、API 与 Streamlit 入口。
- 运行 `pytest --collect-only -q` 收集测试规模，确认当前测试集可被 pytest 收集。
- 统计 Python 文件数量和代码行数，作为后续工程规模基线。

### 遇到的问题

- 终端默认编码读取中文 README 时出现乱码，改用 `Get-Content -Encoding UTF8` 重新读取。
- 仓库中已有 `_review/*` 删除状态，属于进入本轮分析前的工作区状态，本轮没有恢复或改动这些文件。
- `Get-ChildItem -Recurse -File -Include *.py -Path src,app,scripts,tests | Measure-Object -Line` 只能统计文件对象，不是代码行数；随后改为读取文件内容后统计。

### 指标对比

优化前基线：

- Python 文件：123 个。
- Python 代码与测试总行数：17,653 行。
- pytest 可收集测试：44 个测试文件、322 个测试用例。
- 当前主要评测数据：108 条合成/样例评测集、50 条真实公开文档子集。
- 当前真实公开文档知识库：7 篇 PDF、340 chunks。

### 结论

本次为评估基线建立，不涉及功能优化。项目已有 RAG + Tool Agent + LangGraph workflow + memory + persistent knowledge base + evaluation 的完整雏形，但仍需要围绕真实规划能力、工具协议、记忆评测、安全泛化和工程验证做简历向增强。

## 2026-05-29 Cross-Encoder Reranker

### 优化目标

新增真正的 cross-encoder 二阶段精排能力，把现有 `hybrid_rrf` 从“召回融合”扩展为“召回 + pairwise rerank”的高质量检索 baseline。目标是让项目能像成熟 RAG/Agent 系统一样区分 first-stage retrieval 和 second-stage reranking，并可评估排序质量与延迟成本。

### 实施过程

- 新增 `src/retrieval/cross_encoder.py`，包含 `CrossEncoderRerankingRetriever`、可注入的 scorer 协议，以及基于 `sentence_transformers.CrossEncoder` 的真实 scorer。
- `CrossEncoderRerankingRetriever` 先调用 base retriever 获取候选，再对 `(query, document_text)` pair 打分，输出 `cross_encoder_score`、`cross_encoder_model`、`base_score`、`base_retrieval_mode` 和 `candidate_rank`。
- 在 `src/evaluation/runner.py` 新增 `hybrid_rrf_cross_encoder` baseline：先用 BM25 + dense RRF 召回，再用 cross-encoder 精排。
- 在 `scripts/run_eval.py` 新增 `--enable-cross-encoder-rerank` 和 `--cross-encoder-model`，默认关闭，显式开启才加载模型，避免默认 CI 或本地 smoke test 触发模型下载。
- 更新 `src/evaluation/report.py` 和 `README.md`，在实验报告中记录 `cross_encoder_model` 并解释二阶段精排和 latency tradeoff。
- 根据后续要求，将 `scripts/run_eval.py` 调整为默认启用 `hybrid_rrf_cross_encoder`，保留 `--disable-cross-encoder-rerank` 用于快速 smoke test 或无模型环境；旧的 `--enable-cross-encoder-rerank` 保留为兼容参数。

### 遇到的问题

- 按 TDD 先写测试后，第一次失败为 `ModuleNotFoundError: No module named 'src.retrieval.cross_encoder'`，说明新模块尚未实现。
- 接入 runner 后，测试失败为 `run_baseline_comparison() got an unexpected keyword argument 'cross_encoder_scorer'`，随后补充显式 scorer 注入点，保证测试不依赖真实模型下载。
- 接入 CLI 后，测试失败为脚本不识别 `--enable-cross-encoder-rerank` 与 `--cross-encoder-model`，随后补充 argparse 参数。
- 改成默认开启时，测试先失败于默认 comparison 中缺少 `hybrid_rrf_cross_encoder`，以及脚本不识别 `--disable-cross-encoder-rerank`；随后补充默认启用逻辑和关闭参数。
- CLI 测试需要避免下载真实 cross-encoder 模型，因此增加测试专用环境变量 `HVAC_COPILOT_TEST_FAKE_CROSS_ENCODER=1`，只在脚本子进程测试中使用确定性 scorer。

### 指标对比

本模块先完成工程接入与可测 baseline，尚未重跑完整真实 BGE + FAISS + cross-encoder 指标。

新增可观测字段：

- baseline mode：`hybrid_rrf_cross_encoder`。
- rerank metadata：`cross_encoder_score`、`cross_encoder_model`、`base_score`、`base_retrieval_mode`、`candidate_rank`。
- latency metric：`retrieval_average_latency_seconds`。

已验证的行为：

- fake scorer 能把 base retriever 排在后面的目标 chunk 提升到第一位。
- `run_baseline_comparison(..., cross_encoder_scorer=...)` 会输出 `hybrid_rrf_cross_encoder`。
- `scripts/run_eval.py --cross-encoder-model fake-cross-encoder` 默认生成 `hybrid_rrf_cross_encoder` comparison JSON 和报告配置字段。
- `scripts/run_eval.py --disable-cross-encoder-rerank` 可以显式关闭该 baseline。

### 结论

达到预期的工程接入目标：项目现在默认把 cross-encoder reranker 纳入离线评测 baseline，同时保留显式关闭参数。后续需要在本机模型可用时运行真实 `BAAI/bge-reranker-base` 或其他中文/多语 reranker，对 108 条样例集和 50 条真实子集补充指标，判断 MRR/nDCG 收益是否抵消延迟成本。

## 2026-05-29 ToolSpec 与 HVAC 高频工具扩展

### 优化目标

把第二模块从“新增几个函数”升级为可规划、可校验、可执行、可测试的工具协议层。目标是让项目更接近成熟 Agent 项目的工具体系：工具有统一 spec，planner 只能选择白名单工具，executor 能按计划步骤真实执行，工具输出结构化证据而不是自由文本。

### 实施过程

- 新增 `src/tools/registry.py`，定义 `ToolSpec`，统一记录工具名、route、描述、输入输出 schema、风险等级、默认 metric、关键词和 policy 边界。
- 在原有 `query_metric`、`compare_period`、`plot_metric_trend`、`compute_energy_breakdown`、`detect_anomaly` 基础上，新增五个高频 HVAC 运维工具：
  - `data_quality_check`：检查必需字段、空值、重复时间戳和时间间隔缺口。
  - `comfort_risk_assessment`：按舒适上下限统计冷热越界、风险等级和最差 zone。
  - `zone_hotspot_rank`：按最大值、均值和舒适性违规数排序热点 zone。
  - `control_action_audit`：检查控制动作突变和稳定性。
  - `cooling_efficiency_summary`：汇总冷却/风机/HVAC 功率与舒适性风险的关系。
- 将 `ALLOWED_STEP_TOOLS` 改为从 ToolSpec 派生，并把 planner 最大步数从 3 提升到 5。
- 为 deterministic planner 增加高优先级意图识别，避免 `control` 被误路由到 policy，或 `zones` 被误路由成普通 metric 查询。
- 扩展 `AgentTaskExecutor`，让 LangGraph 的 `PlanStep.tool` 可以驱动新增工具执行，而不是只执行老的默认工具。
- 更新 README，说明 ToolSpec 工具协议、HVAC 高频工具集和 1-5 步受控 planner。

### 遇到的问题

- 首轮新增工具函数测试通过后，LangGraph 端到端测试失败：计划步骤指定 `data_quality_check` 时实际仍执行 `query_metric`，指定 `comfort_risk_assessment` / `control_action_audit` 时实际仍执行 `detect_anomaly` / `query_metric`。原因是 executor 的工具选择白名单没有同步扩展。
- planner 的旧关键词规则过宽：`control_action` 问题会因为 `control` 同时触发 `policy_recommendation`，`overheating comfort risk across zones` 会因为 `zones` 同时触发 `timeseries_query`。随后增加 focused tool step 优先级，让明确工具意图先于宽泛 route 关键词。
- 仓库里已有 `_review/*` 删除状态，仍保持不触碰。

### 指标对比

优化前：

- ToolSpec registry：无。
- planner 最大步数：3。
- 明确定义的时序/诊断工具：5 个。
- 新增五类高频工具无法被 LangGraph 计划步骤端到端执行。

优化后：

- ToolSpec registry：12 个工具 spec。
- planner 最大步数：5。
- 明确定义的时序/诊断工具：10 个。
- 新增测试覆盖：
  - 5 个工具函数单测。
  - 2 个 ToolSpec registry 单测。
  - 4 个 deterministic planner 工具选择单测。
  - 2 个 LangGraph 端到端工具执行单测。
- 验证命令：`pytest tests/test_timeseries_tools.py tests/test_tool_registry.py tests/test_route_planner.py tests/test_agent_orchestrator.py -q`，结果 49 passed。
- 静态检查：`ruff check src/tools/timeseries.py src/tools/registry.py src/agent/planner.py src/agent/executor.py tests/test_timeseries_tools.py tests/test_tool_registry.py tests/test_route_planner.py tests/test_agent_orchestrator.py`，结果通过。

### 结论

达到预期。项目现在不是简单堆工具，而是具备了统一工具协议、规划白名单、确定性兜底选择、LangGraph 步骤执行和回归测试闭环。后续可以继续增强工具参数抽取、工具失败恢复、工具调用 trace 展示和离线 tool-selection eval，但本模块已经能作为简历中“Agent 工具系统工程化”的有力亮点。

## 2026-05-29 Tool Schema、Permission Gate 与真实 Cross-Encoder 评测

### 优化目标

把工具系统从“有 ToolSpec 描述”推进到更接近主流 Agent runtime 的执行层：工具输入使用 Pydantic/JSON Schema 可校验，执行前经过 permission gate，每次工具调用记录结构化 trace，planner prompt 从 ToolSpec 自动生成，并补一次真实 cross-encoder 指标运行。

### 实施过程

- 将 `ToolSpec.input_schema/output_schema` 升级为 Pydantic model 派生的 schema，保留原有 `input_schema` / `output_schema` 兼容属性，同时新增 `input_json_schema` / `output_json_schema`。
- 新增 `validate_tool_input(tool_name, payload)`，executor 在调用工具函数前先校验参数；无效参数会返回结构化错误结果，不进入实际工具函数。
- 新增 `build_planner_tool_prompt()`，由 registry 自动生成 route/tool/risk/input 摘要；`src/agent/planner.py` 的 system prompt 改为引用该生成结果，减少 prompt 与工具注册表漂移。
- 在 `AgentTaskExecutor` 中加入统一 `_execute_tool_call()` 包装，记录 `tool_call_id`、`tool_name`、`input`、`output`、`status`、`duration_ms`、`error`、`risk_level`、`permission_decision`、`audit_required`。
- 增加 permission gate 语义：`read_only` 自动 allow；`advisory` 自动执行但标记需要审计；`control_boundary` 通过 policy boundary 执行并记录 `permission_decision=policy_boundary`。
- LangGraph evidence aggregator 现在会合并 `tool_calls`，evaluation prediction 也保留 `tool_calls`，便于后续做 tool-call-level eval。
- 运行真实 cross-encoder 评测命令：
  `python scripts/run_eval.py --dense-provider sentence-transformers --dense-backend faiss --dense-model BAAI/bge-small-zh-v1.5 --cross-encoder-model BAAI/bge-reranker-base --output data/eval/cross_encoder_real_predictions.jsonl --comparison-output data/eval/cross_encoder_real_comparison.json --report-output docs/cross_encoder_real_report.md --human-review-sample-output data/eval/cross_encoder_real_human_review_sample.jsonl --human-review-annotations-output data/eval/cross_encoder_real_human_review_annotations.jsonl`

### 遇到的问题

- 第一次真实模型评测时网络中断，Hugging Face `HEAD` 请求报 `[WinError 10054]`，导致 `SentenceTransformer` 加载失败。网络恢复后重跑成功。
- 上一次中断留下了一个 fake cross-encoder 评测进程；重跑真实评测前通过进程命令行确认并停止该残留进程，保留正在运行的 API 服务。
- `PolicyResult.recommended_action` 在项目中实际是 `list[float]`，最初 `PolicyRunnerOutput` 写成 `str` 不匹配，随后调整为 `list[float]`。
- 默认 108 条样例集的检索排序指标全部为 0，说明这批数据主要测 answer/tool proxy，不适合作为 cross-encoder 排序收益 benchmark。该结果必须如实记录，不能包装成 reranker 提升。

### 指标对比

工具系统指标：

- ToolSpec registry：12 个工具均有 Pydantic input/output model。
- 新增 tool-call trace 字段：`tool_call_id/input/output/status/duration_ms/error/risk_level/permission_decision/audit_required`。
- 新增测试覆盖：
  - JSON Schema 暴露与输入校验。
  - 无效工具参数执行前拦截。
  - planner prompt 从 registry 自动包含工具。
  - `read_only`、`advisory`、`control_boundary` 三类风险等级的执行记录。

真实 cross-encoder 评测结果（108 条默认样例集）：

- `hybrid_rrf`：MRR@10 = 0.000，nDCG@10 = 0.000，Recall@10 = 0.000。
- `hybrid_rrf_cross_encoder`：MRR@10 = 0.000，nDCG@10 = 0.000，Recall@10 = 0.000。
- `hybrid_rrf_cross_encoder` 平均检索延迟：0.16099 秒。
- 结论：真实模型链路和 latency 记录已跑通，但当前默认样例集无法证明 reranker 排序收益。需要改用 50 条真实公开文档子集或构造含 `required_documents` / `required_contexts` 的检索排序集来评估收益。

### 结论

本模块达到了 Agent 工具 runtime 工程化目标：工具 schema、权限决策、执行 trace、planner prompt 生成和真实模型评测链路都已接通。cross-encoder 的真实运行没有显示排序收益，但原因是评测集不适合衡量检索排序，而不是 reranker 实现未运行。后续应优先补一个面向文档检索的真实 ranking eval，再判断是否把 cross-encoder 结果写进简历主指标。

## 2026-05-29 RAG 知识库扩容到 30 篇权威文档

### 优化目标

把 persistent RAG 知识库从少量示例文档扩充为更像真实 Agent 项目的领域语料库。目标不是单纯堆 PDF，而是补齐数据中心 HVAC 场景里高频且权威的知识面：ASHRAE 热/能效标准、DOE/FEMP 数据中心能效和计量、LBNL 空气管理、液冷、服务器遥测、能效行动清单与运维评估。

### 实施过程

- 联网检索并下载 30 篇候选 PDF，来源集中在 ASHRAE、U.S. DOE/FEMP、LBNL Data Centers Center of Expertise。
- 先用文件头和 `pypdf` 验证候选文档：30 个候选均为 `%PDF-` 文件，均可打开并抽取文本；按 SHA256 检查没有和现有 7 篇上传文档完全重复。
- 从 30 篇候选中筛选 23 篇入正式 persistent KB，使总数从 7 篇补到 30 篇；偏 overview、训练课件或案例性较强的 7 篇保留在 `data/knowledge_candidates`，不进入正式索引。
- 入库时为新文档增加 `source_url`、`authority`、`topic`、`curated_for` 元数据，便于后续做 source-aware eval、引用展示和数据集追踪。
- 为避免逐篇上传时反复重建 FAISS，采用批量写入 SQLite/chunks 后统一调用一次 `_reindex_unlocked()` 的方式重建索引。
- 新增 `docs/knowledge_corpus_manifest.md`，记录新增文档、来源 URL、主题标签和未入库候选。

### 遇到的问题

- 第一次批量脚本读取 `download_manifest.json` 失败，原因是该文件带 UTF-8 BOM，Python 普通 `utf-8` 解码报 `Unexpected UTF-8 BOM`。改用 `utf-8-sig` 后解决。
- 本地环境没有 `PyMuPDF/fitz`，但项目 parser 本身会 fallback 到 `pypdf`；因此验证阶段改用 `pypdf`，和项目可用解析路径保持一致。
- 部分 LBNL PDF 标记为 encrypted，但 `pypdf` 仍能打开并抽取文本，实际入库和 chunking 没有失败。
- 当前已有 7 篇老文档缺少 `authority/topic/source_url` 元数据，本次没有回填，避免把扩容任务变成历史数据迁移。后续如要做严格 source-aware eval，建议统一补齐旧文档元数据。

### 指标对比

优化前：

- persistent KB 文档数：7。
- persistent KB chunk 数：340。
- 正式知识库主题覆盖：已有 ASHRAE TC 9.9、DOE best-practice、OCP、Uptime、Google ML、BEAR 相关文档，但空气管理、液冷、计量、服务器遥测和能效行动清单覆盖不足。

优化后：

- persistent KB 文档数：30。
- persistent KB chunk 数：1,682。
- 新增正式文档：23 篇。
- 候选下载池：30 篇 PDF，其中 7 篇保留为候选不入正式索引。
- 索引后端：FAISS + `sentence-transformers` + `BAAI/bge-small-zh-v1.5`。
- 抽样检索验证：
  - ASHRAE thermal query 命中 `ashrae_tc99_power_equipment_thermal_guidelines_2016.pdf` 和 `ashrae_tc99_thermal_guidelines_refcard_2021.pdf`。
  - air-management query 命中 `lbnl_air_management_tool_user_manual_2023.pdf`、`lbnl_air_management_small_data_centers_2016.pdf`、`lbnl_data_center_air_management_report_2006.pdf`。
  - liquid-cooling query 命中 `doe_thermosyphon_hybrid_cooling_water_efficiency_2019.pdf`、`lbnl_liquid_cooling_new_horizons_2019.pdf`、`lbnl_ashrae_liquid_cooling_guidelines_hpc_2011.pdf`。
  - server-telemetry query 命中 `lbnl_accessing_onboard_server_data_2021.pdf` 和 `lbnl_guidance_zombie_servers_2024.pdf`。

### 结论

达到预期。项目现在拥有一个更像真实垂直领域 Agent 的 persistent RAG 语料库，而不是只有少量示例文件。下一步最值得做的是基于这 30 篇文档生成真实检索评测集，把 `required_documents` 对齐到 UUID/file_hash/source_url/filename alias，解决当前 cross-encoder 指标为 0 的评测口径问题。

## 2026-05-29 检索评测 Source Alias 对齐与 Persistent Ranking Eval

### 优化目标

修复真实 cross-encoder 评测中 retrieval 指标全 0 的口径问题。目标是让评测系统能像成熟 RAG/Agent 项目一样，把同一文档的多种身份标识视为同一个 source：`document_id/source_id`、`file_hash`、原始 filename、上传后的 `doc_uuid_filename`、`source_path` 和 `source_url`。同时保留 demo eval 的可复现性，并为当前 30 篇 persistent KB 生成真实 ranking eval 数据集。

### 实施过程

- 在 `src/evaluation/metrics.py` 中新增 source alias 归一化逻辑，从 citation、retrieved context 和 metadata 中收集 `source_id/document_id/chunk_id/file_hash/filename/title/source_path/source_url/url`。
- 对上传文件名做特殊处理：`doc_<uuid>_original.pdf` 会额外生成 `original.pdf` 和无扩展名 alias，避免 persistent KB 的 UUID 文件名和 eval 标签错位。
- 将 `citation_hit_rate`、`context_recall`、`retrieval_recall@k`、`MRR@k`、`nDCG@k` 改为基于 alias set 匹配，而不是只比较 `source_id` 字符串。
- 给 `build_demo_orchestrator()` / `build_rag_pipeline()` 增加 `use_persistent_knowledge` 参数；`scripts/run_eval.py` 增加 `--disable-persistent-knowledge`，用于旧 demo eval 强制使用 `data/documents` 逻辑 source ID。
- 新增 `scripts/generate_persistent_ranking_eval.py`，从 `data/knowledge/knowledge.db` 读取 30 篇 indexed 文档，生成 `data/eval/persistent_knowledge_ranking_eval.jsonl`。
- 生成的 ranking eval 每条记录以当前 KB 的 `document_id` 作为 `required_documents`，并在 `gold_answer` 中记录 `document_id/file_hash/filename/source_url`，方便后续人工核查和 source-aware eval。

### 遇到的问题

- 原先全 0 并不是 reranker 没有返回上下文，而是 `required_documents` 使用 demo 逻辑 ID，persistent KB 检索返回 `doc_<uuid>`，两者没有交集。
- 新增 CLI 测试时，最初把具体 demo 文档排序断言写死为 `rack_delta_t_short_note`。实际禁用 persistent KB 后确实返回 demo source ID，但该 query 在 demo 文档中命中了相邻主题笔记；因此测试收窄为验证“不再返回 `doc_` UUID source”，避免把排序行为误写成配置开关测试。
- 真实 cross-encoder 评测耗时较长，并且会加载 Hugging Face 模型；为测试保留 `HVAC_COPILOT_TEST_FAKE_CROSS_ENCODER=1` 的确定性 scorer，只在最终指标运行中使用 `BAAI/bge-reranker-base`。

### 指标对比

修复前：

- 默认 108 条 eval 上 `hybrid_rrf` 与 `hybrid_rrf_cross_encoder` 的 MRR@10、nDCG@10、Recall@10 全为 0。
- 诊断发现 prediction 有 citations/retrieved_contexts，但 required document IDs 与 retrieved source IDs 交集为空。

修复后，30 条 persistent ranking eval + `BAAI/bge-small-zh-v1.5` + FAISS + 真实 `BAAI/bge-reranker-base`：

- `rag_keyword`：Recall@10 = 0.400，MRR@10 = 0.261，nDCG@10 = 0.295。
- `rag_dense`：Recall@10 = 0.867，MRR@10 = 0.505，nDCG@10 = 0.592。
- `hybrid_rrf`：Recall@10 = 0.733，MRR@10 = 0.378，nDCG@10 = 0.466。
- `hybrid_rrf_cross_encoder`：Recall@10 = 0.900，MRR@10 = 0.717，nDCG@10 = 0.763，平均检索延迟 = 0.230 秒。
- `rag_tool_agent`：Recall@10 = 0.500，MRR@10 = 0.422，nDCG@10 = 0.442。

新增验证：

- 红灯测试先确认 alias 不支持时会失败，再实现转绿。
- `pytest tests/test_evaluation.py tests/test_generate_persistent_ranking_eval.py -q`：24 passed。
- `pytest tests/test_baseline_runner.py::test_run_eval_script_can_force_demo_documents_without_persistent_knowledge tests/test_baseline_runner.py::test_run_eval_script_can_disable_default_cross_encoder_rerank tests/test_baseline_runner.py::test_run_eval_script_enables_cross_encoder_rerank_by_default_without_downloading_model -q`：3 passed。
- `ruff check src/evaluation/metrics.py src/api/demo_factory.py scripts/run_eval.py scripts/generate_persistent_ranking_eval.py tests/test_evaluation.py tests/test_baseline_runner.py tests/test_generate_persistent_ranking_eval.py`：通过。

### 结论

达到预期。此前“指标全 0”的根因已修复，项目现在具备 source alias-aware retrieval metrics、demo/persistent KB 明确切换、以及面向 30 篇真实文档的 persistent ranking eval。真实 cross-encoder 在新 ranking eval 上体现了明显排序收益，可以作为后续简历项目里的有效检索指标，但仍建议补人工审核样本，避免只用文件名导向问题高估 reranker。
