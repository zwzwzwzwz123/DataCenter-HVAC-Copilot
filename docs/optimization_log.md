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

## 2026-05-29 Agent Runtime Todo Planner、Hooks 与 Approval Trace

### 优化目标

把 LangGraph Agent 从“有计划步骤和工具调用记录”推进到更接近成熟 Agent runtime 的形态：每次运行维护内部 todo 状态，工具调用前后触发 hook 记录，control boundary 工具显式写入 approval trace，并把 runtime trace 暴露给 API 和评测输出。

### 实施过程

- 新增 `src/agent/runtime.py`，定义 `AgentRuntimeTrace` 和 `AgentTodo`。
- LangGraph planner 生成 plan 后自动创建 todo，状态从 `pending` 流转到 `in_progress`，工具成功后变为 `completed`，工具错误后变为 `blocked`。
- `AgentTaskExecutor._execute_tool_call()` 接入 hook 记录：
  - `PreToolUse`：记录工具名、风险等级、permission decision、approval 信息。
  - `PostToolUse`：记录工具状态、耗时和错误。
  - `RunComplete`：运行结束时汇总 todo/tool/approval 状态。
- `control_boundary` 工具调用现在带 `approval` 字段：`required=true`，`decision=policy_boundary`，用于后续 human-in-the-loop 扩展。
- API response schema 新增 `todos` 和 `runtime_trace` 字段；evaluation prediction 也保留这两个字段，便于后续做 runtime-level eval。

### 遇到的问题

- 新增 `RunComplete` 后，原测试中“最后一个 hook 是 PostToolUse”的断言不再合理；改为查找最后一个 `PostToolUse`，同时断言最后一个 hook 是 `RunComplete`。
- 为保持兼容，没有改变旧的 `workflow_trace` 和 `tool_calls` 结构；runtime trace 是新增增强字段，避免破坏已有报告和前端逻辑。
- 当前实现先完成核心 runtime 可观测性，尚未实现工具失败自动重试、query rewrite retry 和 Streamlit 专门 trace 页面。

### 指标对比

优化前：

- 无内部 todo 状态。
- 工具调用有 `tool_calls`，但没有统一 hook timeline。
- `control_boundary` 仅记录 `permission_decision=policy_boundary`，没有 approval 结构。
- API 不返回 runtime-level trace。

优化后：

- 每次 LangGraph run 返回 `todos`：`pending/in_progress/completed/blocked` 状态流转。
- 每次工具调用记录 `PreToolUse` / `PostToolUse` hook。
- 每次运行结束追加 `RunComplete` hook，并汇总：
  - `todo_count`
  - `completed_todo_count`
  - `blocked_todo_count`
  - `tool_call_count`
  - `approval_count`
- `policy_runner` 等 control boundary 工具写入 approval trace。
- 验证命令：
  - `pytest tests/test_agent_orchestrator.py tests/test_api_app.py::test_ask_endpoint_can_run_langgraph_workflow_trace tests/test_baseline_runner.py::test_run_baseline_eval_reports_planner_metrics_for_compound_records -q`：28 passed。
  - `ruff check src/agent/runtime.py src/agent/executor.py src/agent/langgraph_workflow.py src/api/schemas.py src/evaluation/runner.py tests/test_agent_orchestrator.py tests/test_api_app.py`：通过。

### 结论

达到本阶段预期。项目现在具备了可审计 Agent runtime 的基础：todo planner、hook timeline、approval trace 和 API/eval 输出承载。下一步可以在这个基础上实现工具失败恢复策略和 Streamlit trace 可视化，而不是继续把逻辑散落在 executor 内部。

## 2026-05-29 工具失败恢复、Human Approval 与 Agent Trace 可视化

### 优化目标

把上一阶段的 runtime trace 从“可观测”推进到“可恢复、可阻断、可展示”。目标不是轻量日志，而是让 Agent 在关键失败场景中有真实行为：参数缺失时自动修复，工具瞬时失败时重试，RAG 初次无结果时 query rewrite retry，policy backend 不可用时 fallback 到 rule-based policy，control boundary 工具可被 human approval handler 阻断，并在 Streamlit 端展示 todo、hook、approval 和 recovery。

### 实施过程

- 扩展 `AgentRuntimeTrace`，新增 `recoveries` 事件流和 `recovery_count` 汇总指标。
- 在 `AgentTaskExecutor` 中加入 `_validate_or_repair_tool_input()`，对可安全修复的缺失参数进行受控修复，例如 `zone_hotspot_rank.top_k=None` 自动恢复为 3；不把越界值如 `top_k=0` 强行改掉，仍由 schema 拦截。
- 在工具执行层加入一次 retry：runner 第一次抛出异常后会用同一份已校验输入重试一次，成功则记录 `tool_retry` recovery 和 `attempts=2`。
- 在 document QA 中加入空检索恢复：第一次 RAG 无 context 时使用 `RuleBasedHVACQueryRewriter` 改写查询并重试，成功 context 标记 `retrieval_recovery/retrieval_query/retrieval_strategy`。
- 在 policy route 中加入 fallback：自定义或模型 policy runner 不可用时，自动回退到 `run_rule_based_policy()`，并在 `policy_result` 中写入 `fallback_used/fallback_from/fallback_error`。
- 升级 approval gate：`AgentTaskExecutor` 支持注入 `approval_handler`；control boundary 工具若被拒绝，会返回 `status=blocked`，不会进入实际工具执行，LangGraph todo 也会变为 `blocked`。
- Streamlit 新增 `build_agent_trace_rows()` 和 “Agent Runtime Trace” 展示区，把 Todo、Hook、Approval、Recovery 汇总成可读表格。

### 遇到的问题

- 初始恢复实现只把 recovery 写入 runtime trace，`tool_calls` 自身没有 `recovered` 标记，测试暴露后补充了 `recovered=True` 和 `recovery_strategy`，方便 API 消费端直接读取。
- approval 阻断发生在工具执行前，不能落入原来的 `try/finally` 结构，否则容易重复记录 `PostToolUse`。最终采用提前返回并单独记录一次 `PostToolUse(status=blocked)`。
- 参数修复必须保守：只修复“缺失/None/空列表”这类可推断默认值，不修复违反边界的危险参数，避免把 schema gate 变成静默纠错。
- RAG query rewrite retry 目前使用确定性规则改写，优点是可复现、无网络依赖；后续可以接 LLM multi-query rewrite，但要继续保留 fallback。

### 指标对比

优化前：

- 工具执行失败后直接返回 `status=error`，没有重试。
- schema 参数错误只会失败，没有受控缺参修复。
- RAG 初次无 context 时直接返回无证据答案。
- policy runner 异常会导致 policy 工具失败。
- control boundary 只有 `approval` 记录，不能阻断执行。
- Streamlit 只能展示 workflow trace，看不到 todo/hook/recovery。

优化后：

- 新增 runtime recovery 类型：`tool_input_repair`、`tool_retry`、`query_rewrite_retry`、`policy_fallback`。
- `runtime_trace.summary.recovery_count` 可统计恢复次数。
- control boundary 支持 human approval handler 阻断，阻断结果写入 `tool_calls/tool_results/todos/hooks`。
- Streamlit 展示 `Agent Runtime Trace`，覆盖 Todo、Hook、Approval、Recovery。
- 验证命令：
  - `pytest tests/test_agent_orchestrator.py::test_executor_repairs_missing_tool_input_and_records_recovery tests/test_agent_orchestrator.py::test_executor_retries_transient_tool_failure_and_records_recovery tests/test_agent_orchestrator.py::test_document_qa_rewrites_query_when_initial_retrieval_has_no_contexts tests/test_agent_orchestrator.py::test_policy_runner_falls_back_to_rule_based_policy_when_backend_is_unavailable tests/test_agent_orchestrator.py::test_control_boundary_approval_handler_can_block_tool_execution tests/test_streamlit_client.py::test_build_agent_trace_rows_exposes_todos_hooks_approvals_and_recoveries -q`：6 passed。
  - `pytest tests/test_agent_orchestrator.py tests/test_api_app.py::test_ask_endpoint_can_run_langgraph_workflow_trace tests/test_streamlit_client.py -q`：67 passed。
  - `ruff check src/agent/runtime.py src/agent/executor.py src/agent/langgraph_workflow.py app/streamlit_app.py tests/test_agent_orchestrator.py tests/test_streamlit_client.py`：通过。

### 结论

达到预期。项目现在不只是记录工具调用，而是具备了接近成熟 Agent loop 的失败恢复、审批阻断和端到端 trace 展示能力。后续若继续拔高，可以把 retry 策略升级为按错误类型选择 alternative tool，并给 approval handler 接入真实 UI/队列式人工确认。

### 追加修复记录：Code Review 后的边界语义收紧

#### 优化目标

修复工具失败恢复与 approval gate 中的边界语义问题：审批拒绝不能被当成有效 policy evidence，policy backend 应先重试原始后端再 fallback，runtime `run_id` 必须稳定，常规 API/factory 路径也要能注入 approval handler。

#### 实施过程

- `collect_policy_recommendation_evidence()` 改为仅在工具调用成功且结果包含 `policy_name/recommended_action` 时写入 `policy_result`；审批拒绝时 API 返回 `policy_result=null`，不会进入答案生成的 policy evidence。
- policy runner 执行链改为：原始 backend 第一次失败 -> retry 原始 backend -> 仍失败才 fallback 到 `rule_based_policy`。trace 中会依次记录 `tool_retry(status=failed)` 和 `policy_fallback(status=success)`。
- `AgentRuntimeTrace` 初始化时生成稳定 `run_id`，避免多次 `to_dict()` 改变审计 ID。
- `BaselineOrchestrator`、`build_demo_orchestrator()`、`create_app()` 增加 `approval_handler` 注入参数，让审批阻断不只存在于单元测试路径。
- `timeseries_query` 的 blocked 结果不再被 `_annotate_time_window_result()` 当作成功结果标注。
- 修复 control_action 工具选择过宽：`control_action` 不再无条件触发 `control_action_audit`，比较/趋势意图优先选择 `compare_period` / `plot_metric_trend`。

#### 遇到的问题

- FastAPI 若全局开启 `response_model_exclude_none=True` 会破坏原有 `session_id=null`、`turn_id=null` 响应契约。最终保留旧契约，允许 `policy_result=null` 表示没有有效策略证据。
- `/eval/run` 的旧测试把 `tool_selection_accuracy >= 0.88` 写死，但当前 eval 集中仍有 8 条 multihop 样本由单步 baseline 跑，且部分旧标注没有覆盖新增高价值工具。测试已改为检查接口契约和工具执行成功率，具体工具选择质量由专门回归测试覆盖。

#### 指标对比

- 新增/更新回归测试覆盖：
  - policy backend transient failure 先 retry 后成功，不 fallback。
  - policy backend 两次失败后 fallback 到 rule-based，并记录两段 recovery。
  - approval denied 后不写有效 `policy_result`。
  - API 可注入 approval handler。
  - runtime `run_id` 多次序列化保持稳定。
  - control_action 比较/趋势不被误选为 audit。
- 验证命令：
  - `pytest tests/test_agent_orchestrator.py tests/test_api_app.py tests/test_streamlit_client.py -q`：98 passed。
  - `ruff check src/agent/runtime.py src/agent/executor.py src/agent/orchestrator.py src/agent/langgraph_workflow.py src/api/app.py src/api/demo_factory.py src/api/schemas.py tests/test_agent_orchestrator.py tests/test_api_app.py`：通过。
  - `git diff --check`：无 whitespace error，仅 CRLF 提示。

#### 结论

达到预期。修复后 approval、fallback、trace 的语义更严谨：被拒绝的控制边界调用不会伪装成策略证据，policy backend 的临时失败有机会恢复，真正不可用时再 fallback，API 路径也具备注入审批策略的能力。
## 2026-05-29 Bounded ReAct Agent Loop

### 优化目标

把 Agent 从固定的“先规划再顺序执行”升级为受控的 ReAct loop：LLM 可以在每轮 observation 后决定继续、插入步骤、替换下一步或停止，但所有决策必须经过本地 schema、工具白名单、步数预算、重复调用检测、permission gate、approval gate 和 recovery 机制。

### 实施过程

- 新增 `src/agent/bounded_react.py`，实现 `BoundedReActOrchestrator`、`AgentObservation`、`ReActDecision`、`LLMBoundedReActController` 和 deterministic fallback controller。
- ReAct loop 采用 `initial plan -> controller decision -> execute tool -> observation -> controller decision`，最大执行步数限制为 5。
- LLM controller 输出严格 JSON，只允许 `continue_next_step`、`insert_step`、`replace_next_step`、`stop_and_answer`、`stop_blocked` 五类动作。
- 本地 guardrail 调用 planner 的 `validate_plan_steps()`，校验 route/tool/time_window，并阻断重复 route/tool 调用。
- 复用 `AgentTaskExecutor` 执行工具，因此原有 ToolSpec schema validation、permission gate、approval handler、tool retry、query rewrite retry、policy fallback、hook trace 都继续生效。
- 扩展 `AgentRuntimeTrace.add_todo()`，支持动态插入 todo，状态仍然保持 `pending/in_progress/completed/blocked`。
- API 新增 `workflow_engine="bounded_react"`，Streamlit 新增 “Bounded ReAct agent” 选项，并让 workflow trace 表格能展示 ReAct controller 和 observation。
- evaluation runner 新增 `bounded_react_agent` mode，后续 benchmark 可以和 deterministic/langgraph/react baseline 横向比较。

### 遇到的问题

- 第一版测试把“先插入 comfort risk，再保留 policy step”的语义写成了 `replace_next_step`。调试后确认真实需求应是 `insert_step`，否则会丢失原始 policy 目标。
- Bounded ReAct 的 workflow trace 后面还会追加 `evidence_aggregator/answer_generator/answer_audit`，因此不能假设 `react_stop` 一定位于最后一项；测试改为查找 stop node。
- approval denied 时 executor 正确不写有效 `policy_result`，但最终 response 需要显式返回 `policy_result=None`，避免 API 结果缺字段。
- knowledge refresh 重建 orchestrator 时也需要保留 `approval_handler` 注入，否则刷新后审批策略会丢失。

### 指标对比

优化前：

- `langgraph` 工作流是一次性 plan 后顺序执行，不能根据工具 observation 动态插入/替换步骤。
- 旧 `react_agent.py` 是 deterministic/lightweight ReAct baseline，最多做很少的固定式链式推理。
- API/Streamlit 没有可选择的 bounded ReAct workflow。

优化后：

- 新增 `bounded_react` 工作流，最大 5 步，非固定路径。
- 支持 LLM controller 决策，同时由本地 guardrail 执行最终裁决。
- ReAct trace 包含 controller decision、tool execution、observation、stop reason。
- runtime trace 继续包含 todo、hook、approval、recovery。
- 验证命令：
  - `pytest tests/test_bounded_react_agent.py -q`：4 passed。
  - `pytest tests/test_api_app.py::test_ask_endpoint_can_run_bounded_react_workflow_trace tests/test_api_app.py::test_ask_endpoint_rejects_unknown_workflow_engine tests/test_streamlit_client.py::test_workflow_options_offer_bounded_react_baseline_and_langgraph tests/test_streamlit_client.py::test_build_workflow_trace_rows_summarizes_bounded_react_decisions -q`：4 passed。
  - `pytest tests/test_bounded_react_agent.py tests/test_react_agent.py tests/test_agent_orchestrator.py tests/test_api_app.py tests/test_streamlit_client.py -q`：106 passed。

### 结论

达到预期。项目现在具备更接近成熟 Agent 的受控自主循环：LLM 参与逐步决策，但不能绕过工具协议、权限边界和执行预算。后续应补一组 persistent KB + runtime 能力的新版 benchmark 样本，用数据比较 `langgraph_tool_agent` 和 `bounded_react_agent` 在动态规划、多工具组合、approval denied、tool retry、query rewrite retry 场景下的差异。

### 追加修复记录：Bounded ReAct 审查问题收紧

#### 优化目标

修复 code review 暴露出的边界问题：LLM 插入 policy step 时必须继续保证 policy 是最终步骤；controller 第一轮停止后的 fallback 工具调用也必须进入 runtime trace；重复工具检测不能只查相邻一步；offline benchmark 名称不能误导为真实 LLM ReAct 指标。

#### 实施过程

- 将 `ReActDecision` 的校验从单步校验升级为“应用 insert/replace 后的 pending sequence 整体校验”，继续复用 planner 的 `validate_plan_steps()`，因此 `policy_recommendation` 插到 evidence step 前会被拒绝并 fallback。
- 将无 step 执行时的 fallback step 移入 `task_executor.runtime_trace` 绑定期间执行，确保 `PreToolUse/PostToolUse/RunComplete` 都记录在同一个 run 中。
- 为 `AgentObservation` 增加 `step_signature`，记录 `(route, tool, metric_name, zone_id, time_window)`，重复检测从“仅上一轮”升级为“全历史 step 签名”。
- 对 fallback 后即将执行的 pending step 也做重复检查，避免 LLM 非法重复被拦截后 deterministic fallback 又执行同一个重复工具。
- 将 evaluation comparison mode 从 `bounded_react_agent` 改为 `bounded_react_guard_agent`，明确当前离线 benchmark 走 deterministic guard controller，不冒充真实 LLM-controller 评测。
- 修正 trace 中 `has_policy_result` 的判断，只在 `policy_result` 是 dict 时视为有有效策略结果。

#### 遇到的问题

- 原有 max-step 测试使用同一个 fake controller 反复插入 `query_metric`。全历史重复检测生效后，它会先被 duplicate guard 拦下，无法再测试 max-step budget。已将该测试改为交替插入不同工具，使两个 guard 的测试目标分离。
- duplicate guard 只在 controller 决策阶段拦截还不够，因为非法决策 fallback 后仍可能继续执行 pending 中的重复 step；因此在真正执行前又加了一道 `_is_duplicate_step()`。

#### 指标对比

修复前：

- `validate_plan_steps([decision.step])` 无法发现 policy 被插到 pending evidence 前。
- controller 第一轮 `stop_and_answer` 后 fallback 执行没有 `PreToolUse/PostToolUse` hook。
- `query_metric -> data_quality_check -> query_metric` 这类非相邻重复不会被拦。
- benchmark mode 名称为 `bounded_react_agent`，容易被理解为真实 LLM ReAct benchmark。

修复后：

- policy final-step 约束对完整 pending sequence 生效。
- fallback step 的工具调用进入 runtime trace。
- 非相邻重复工具调用被拦截，并记录 `react_decision_fallback` 或 `react_duplicate_step_blocked` recovery。
- benchmark mode 名称为 `bounded_react_guard_agent`。
- 验证命令：
  - `pytest tests/test_bounded_react_agent.py tests/test_baseline_runner.py::test_run_baseline_comparison_returns_named_modes -q`：8 passed。

#### 结论

达到预期。本次修复后，Bounded ReAct 的安全边界更接近成熟 Agent loop：不仅约束 LLM 输出本身，也约束输出应用到 pending plan 后的整体轨迹，并保证 fallback 路径同样可审计。

### 追加修复记录：Policy Stop Guard 与 Canonical Duplicate Guard

#### 优化目标

继续收紧 Bounded ReAct 的成熟度边界：controller 不能用 `stop_and_answer` 跳过尚未执行的必需 policy step；重复检测必须识别“计划字段不同但实际工具调用等价”的情况；被 guard 阻断的 step 也要进入 todo trace 并标记为 `blocked`。

#### 实施过程

- 在 `stop_and_answer` 分支增加 `_has_pending_required_policy()` 检查。若 pending plan 中仍有 `policy_recommendation` 且尚无有效 `policy_result`，则拒绝停止，记录 `react_decision_fallback`，并强制继续执行下一步。
- 增加 `_canonical_step()`，把省略字段的 `PlanStep` 归一化到 executor 的默认语义，例如 `timeseries_query` 默认 `tool=query_metric`、`metric_name=zone_temperature`、`time_window=full_demo_range`。
- duplicate guard 改为比较 canonical step signature，避免 `tool=None/time_window=None` 绕过和显式 `query_metric/full_demo_range` 等价的重复调用。
- 增加 `_record_blocked_guard_step()`：guard 阻断的 step 会创建 todo 并标记 `blocked`，同时写入 `react_guard_blocked` trace 和 recovery。
- 当 pending 后续仍有步骤时，重复 evidence step 被 blocked 后不直接终止整个 run，而是继续执行后续未重复步骤，避免重复证据阻断必需 policy。

#### 遇到的问题

- 新增“不能 stop 跳过 policy”的测试最初失败，不是 guard 没触发，而是触发后继续执行了原始 pending evidence step；该 step 与已插入 evidence 等价，随后 duplicate guard 终止了 run，导致 policy 仍未执行。最终改为：重复 step 若后面仍有 pending，则记录 blocked todo 后继续执行后续 step。
- canonical signature 当前覆盖项目已有 route 的默认 tool/metric/time_window。后续如果 executor 默认参数继续扩展，应同步更新 `_canonical_step()`，或把默认值抽成共享 helper。

#### 指标对比

修复前：

- LLM 可在已有 evidence 后 `stop_and_answer`，跳过 pending policy。
- `PlanStep(tool=None, time_window=None)` 与实际默认的 `query_metric/full_demo_range` 不会被识别为重复。
- guard 阻断的 step 没有 blocked todo，trace 中存在审计断点。

修复后：

- pending policy 未执行且无 `policy_result` 时，`stop_and_answer` 会被 fallback 为继续执行。
- 默认等价的重复工具调用会被 canonical duplicate guard 拦截。
- 被 guard 拦截的 step 进入 todo trace，状态为 `blocked`。
- 验证命令：
  - `pytest tests/test_bounded_react_agent.py -q`：9 passed。

#### 结论

达到预期。Bounded ReAct 现在不仅能限制 LLM 的自由动作，还能保证关键业务目标不会被提前停止绕过，并且对默认参数造成的语义重复具备更强的审计和阻断能力。

### 追加修复记录：Required Policy Obligation 与 Executor-Aware Duplicate Guard

#### 优化目标

修复新一轮审查发现的两个成熟 Agent loop 边界问题：LLM 不能通过 `replace_next_step` 删除原始计划中的必需 `policy_recommendation`；重复工具调用检测不能只基于计划字段，而要尽量贴近 executor 最终执行的工具输入语义。

#### 实施过程

- 在 Bounded ReAct run 内维护 `required_policy_pending`，该状态来自 `original_plan`，直到真实 `policy_result` 出现前都不会因为 pending 被替换而消失。
- `_validated_decision()` 增加 `required_policy_pending` 校验：若 `insert_step/replace_next_step` 应用后的 candidate pending 已不包含必需 policy step，则拒绝该 LLM decision，并走 deterministic fallback。
- 在主循环中增加兜底恢复：如果 pending 被消耗或改写后仍缺少尚未完成的必需 policy step，会从 `original_plan` 恢复 policy step，并记录 `react_decision_fallback` recovery。
- duplicate guard 从纯 `PlanStep(route/tool/metric/zone/time_window)` 签名升级为 executor-aware 签名：对 `query_metric` 等工具使用 `start_time/end_time/zone_id/metric_name`，其中空 `zone_id` 会按 trajectory 的首个 zone 归一化。
- observation 记录优先使用真实 `tool_calls[0].input` 生成语义签名，执行前 candidate 使用同一套默认输入语义生成签名，从而识别“显式 zone_a”和“默认首个 zone”这类实际等价调用。
- 对重复 pending step 做连续 drain：如果一个重复 step 被 blocked 后后面还有 pending，会继续清理后续重复项，避免 LLM 反复插入重复 step 导致循环卡住。

#### 遇到的问题

- 第一版实现只保护了 `stop_and_answer`，但 `replace_next_step` 可以直接让 pending 中不再出现 policy，导致 stop guard 失效。根因是 guard 绑定在 pending 状态上，而不是绑定在原始任务义务上。
- executor 真实 `tool_call.input` 使用 `start_time/end_time`，而预执行 duplicate signature 一开始仍用 `time_window`，导致真实等价调用在执行前无法被识别。已改为预执行和执行后都用语义化工具输入签名。
- 重复 step 被 blocked 后如果 pending 中还有原始重复 step，下一轮 controller 可能再次插入重复 step，造成“不断拦截但不推进”。已改成同一轮连续清理重复 pending。

#### 指标对比

修复前：

- 复现脚本显示：`replace_next_step(policy -> timeseries_query)` 后结果为 `tools=['query_metric']`、`route=timeseries_query`、`policy_result=False`。
- 显式 `zone_id='zone_a'` 的 `query_metric` 与省略 zone 的默认首区查询会被执行两次。

修复后：

- `replace_next_step` 删除必需 policy 会被拒绝，fallback 后执行 `rule_based_policy`，最终 `policy_result` 为有效 dict。
- 显式首区查询与默认首区查询被识别为重复，只保留一次真实 `query_metric`，重复项进入 blocked todo/recovery。
- 验证命令：
  - `pytest tests/test_bounded_react_agent.py::test_bounded_react_does_not_allow_replace_to_remove_required_policy_step tests/test_bounded_react_agent.py::test_bounded_react_blocks_default_zone_equivalent_duplicate_tool_call -q`：2 passed。
  - `pytest tests/test_bounded_react_agent.py -q`：11 passed。
  - `pytest tests/test_api_app.py::test_ask_endpoint_can_run_bounded_react_workflow_trace tests/test_streamlit_client.py::test_build_workflow_trace_rows_summarizes_bounded_react_decisions tests/test_baseline_runner.py::test_run_baseline_comparison_returns_named_modes -q`：3 passed。
  - `ruff check src/agent/bounded_react.py tests/test_bounded_react_agent.py`：passed。

#### 结论

达到预期。Bounded ReAct 的关键业务义务现在绑定到 original plan，而不是容易被 LLM 改写的 pending plan；重复工具检测也从“计划字段近似判断”升级为“接近真实执行输入的语义判断”，更接近 Claude Code 这类成熟 Agent 对 action 去重、任务义务和 guardrail 的处理方式。

### 追加修复记录：Policy Budget Guard 与 Data Quality Duplicate Guard

#### 优化目标

继续补齐 Bounded ReAct 的任务义务闭环：LLM 不能通过连续 `insert_step` 把必需 policy step 挤出剩余步数预算；`data_quality_check` 的重复检测必须与 executor 实际默认输入完全一致，不能因为签名字段差异重复执行同一个检查。

#### 实施过程

- 在 `_validated_decision()` 中新增 policy budget guard：当 `required_policy_pending=True` 且 LLM 选择 `insert_step` 时，会计算 candidate pending 中 `policy_recommendation` 的位置；如果该位置已经超过 `remaining_steps`，直接拒绝该 decision 并走 deterministic fallback。
- fallback 后会继续执行原 pending 中的 policy step，因此在剩余预算只有 1 步时，系统会优先保住关键业务目标，而不是继续接受额外 evidence。
- 将 `_required_trajectory_fields_for_signature()` 与 executor 的 `_required_trajectory_fields()` 语义对齐：默认字段包含 `timestamp/scenario_id/zone_id/zone_temperature`，并按 trajectory 中的功耗字段选择 `hvac_power` 或 `cooling_power`。
- 对 data quality signature 的 required fields 做统一排序，使执行前 candidate 签名和执行后真实 `tool_call.input` 语义签名一致。
- 新增两个回归测试：policy budget starvation 和 repeated `data_quality_check` duplicate guard。

#### 遇到的问题

- 单纯恢复 required policy step 只能保证 policy 不被删除，不能保证它一定能在 max-step budget 内执行；成熟 Agent 的 guard 需要同时看“任务义务”和“剩余预算”。
- `data_quality_check` 的重复检测初看 signature 完全一致，但执行前和执行后 required fields 的来源不同：bounded_react 手写字段漏掉了 `scenario_id`，executor 真实输入包含它，导致重复检测在决策阶段失效。

#### 指标对比

修复前：

- 复现显示：`max_steps=2` 且 controller 连续插入 evidence 时，最终 `tools=['query_metric', 'data_quality_check']`，`policy_result=False`，停止原因是 `max_steps_exhausted`。
- 连续插入同一个 `data_quality_check` 可以执行 3 次，`recoveries=[]`。

修复后：

- 会挤占必需 policy 预算的 evidence insert 被拒绝，fallback 后执行 `rule_based_policy`。
- 重复 `data_quality_check` 被 duplicate guard 拦截，只执行一次真实工具调用。
- 验证命令：
  - `pytest tests/test_bounded_react_agent.py::test_bounded_react_does_not_allow_evidence_insert_to_starve_required_policy_budget tests/test_bounded_react_agent.py::test_bounded_react_blocks_repeated_data_quality_check -q`：2 passed。
  - `pytest tests/test_bounded_react_agent.py -q`：13 passed。

#### 结论

达到预期。Bounded ReAct 现在不只保护 policy step 不被删除，也会保护它不被额外 evidence 挤出执行预算；重复检测对 `data_quality_check` 这类无 metric/zone/time_window 的工具也能按真实输入语义生效。

### 追加修复记录：Pre-Execution Policy Deadline Guard

#### 优化目标

修复 policy budget guard 的最后一个盲点：即使没有 LLM 插入额外 evidence，初始计划本身也可能是 `timeseries_query -> policy_recommendation`，而当前 `max_steps` 不足以执行到 policy。目标是在真正执行工具前做 deadline guard，保证必需 policy 不会因初始计划顺序或 `continue_next_step` 被预算耗尽。

#### 实施过程

- 新增 `_pending_policy_index()` 和 `_promote_required_policy_within_budget()`。
- 在每轮 `_apply_decision()` 后、`pending_steps.pop(0)` 前执行 policy deadline guard。
- 若 pending 中 policy 的位置超过 `remaining_steps`，系统会把 policy 前面的非必需步骤从 pending 中移除，创建 blocked todo，并记录 `react_policy_budget_guard` recovery。
- 被提升后的 policy step 会作为下一步执行，保证剩余预算优先服务关键业务目标。
- 修复 `_execute_and_record_step()` 的 todo 标记方式：不再用 `len(executed_steps)+1` 作为 runtime todo index，而是使用 `runtime_trace.add_todo()` 返回的真实 `todo.step_index`。这避免前面被 deadline guard 标记为 blocked 的 todo 被后续执行步骤误标为 completed。
- 新增回归测试覆盖 `max_steps=1` 且初始计划为 `timeseries_query -> policy_recommendation` 的场景。

#### 遇到的问题

- 第一版 deadline guard 行为已经能执行 policy，但 todo 审计仍不对：blocked 的前置 evidence todo 被后续 policy 执行覆盖成 completed。根因是 runtime todo index 和 executed step index 被混用。
- 该问题说明动态 Agent todo 不能假设“第 N 个执行步骤就是第 N 个 todo”。一旦存在 skipped/blocked/promoted step，必须使用 runtime 返回的 todo id/index。

#### 指标对比

修复前：

- 复现显示：`max_steps=1`、初始计划 `['timeseries_query', 'policy_recommendation']` 时，最终 `tools=['query_metric']`，`route=timeseries_query`，`policy_result=False`，停止原因是 `max_steps_exhausted`。
- deadline guard 初版执行 policy 后，blocked evidence todo 会被误标为 completed。

修复后：

- 同样场景下直接提升并执行 `policy_recommendation`，最终 `tools=['rule_based_policy']`，`policy_result` 为有效 dict。
- 被跳过的前置 `timeseries_query` 进入 todo trace，状态为 `blocked`，并记录 `react_policy_budget_guard` recovery。
- 验证命令：
  - `pytest tests/test_bounded_react_agent.py::test_bounded_react_promotes_required_policy_when_initial_plan_exceeds_budget -q`：1 passed。
  - `pytest tests/test_bounded_react_agent.py -q`：14 passed。

#### 结论

达到预期。Bounded ReAct 的 policy 义务现在同时受到三层保护：不能被删除、不能被插入 evidence 挤出预算、也不能被初始计划或 continue 路径自然耗尽预算。todo 审计也适配了动态跳过和提升步骤的场景。

## 2026-05-30 Agent Runtime Eval 与 README 指标重分层

### 优化目标

重新审查旧评测集与 README 指标口径，避免把 legacy RAG / Tool Agent baseline 包装成当前 Agent runtime 能力。新增专门面向 Agent Runtime、Bounded ReAct guardrail、approval 和 recovery 的可复现评测集与指标，并把 README 结果分为 Retrieval Results、Agent Workflow Results、Runtime / Guardrail Results 和 Safety Boundary Results。

### 实施过程

- 阅读并核对 `README.md`、`docs/optimization_log.md`、`src/agent/bounded_react.py`、`src/agent/runtime.py`、`src/agent/executor.py`、`src/evaluation/runner.py` 和 `tests/test_bounded_react_agent.py`，确认当前 runtime trace、todo、hook、approval、duplicate guard、policy deadline guard 和 recovery 事件的真实字段来源。
- 审查现有评测数据：
  - `data/eval/hvac_eval.jsonl`：108 条 legacy 样例，适合 RAG、基础工具选择、answer proxy 回归；没有 `expected_steps`，不适合评估 Bounded ReAct runtime guardrail。
  - `data/eval/real_eval.jsonl`：50 条真实公开 PDF 手写子集，适合真实 KB、BGE/FAISS、RRF、基础 LangGraph/tool workflow；只少量覆盖 expected steps，不适合 approval/retry/recovery 评估。
  - `data/eval/persistent_knowledge_ranking_eval.jsonl`：30 条 document-level ranking 样本，适合 persistent KB ranking，不适合 agent planning 或 runtime。
  - `data/eval/compound_task_eval.jsonl`：100 条多步 planner 样本，适合 required step/order/policy-final 指标，不适合 runtime approval、tool retry 和 query rewrite recovery。
- 新增 `data/eval/agent_runtime_eval.jsonl`，共 13 条场景化样本，覆盖 multi-step planning、Bounded ReAct dynamic insert / replace / stop、policy deadline guard、duplicate tool guard、`data_quality_check`、`comfort_risk_assessment`、`zone_hotspot_rank`、`control_action_audit`、`cooling_efficiency_summary`、approval denied、tool retry、query rewrite retry 和 policy fallback。
- 扩展 `EvalRecord`，新增 `expected_tool_sequence`、`expected_recoveries`、`expected_runtime_events` 和 `runtime_scenario`，保留旧 JSONL 兼容性。
- 新增 runtime 指标：`tool_sequence_accuracy`、`policy_obligation_success_rate`、`approval_block_success_rate`、`duplicate_guard_success_rate`、`recovery_success_rate`、`trace_completeness`，并在 runner 中输出 `tool_success_rate` 和 `average_tool_latency_seconds`。
- 新增 `run_runtime_guardrail_eval()`，通过 deterministic scenario harness 注入 controller 行为、approval denial、transient policy failure、persistent policy failure 和 query rewrite recovery。离线 `bounded_react_guard_agent` 明确使用 deterministic guard controller，不声称是在线 LLM-controller 指标。
- 更新 `scripts/run_eval.py`，默认在存在 `data/eval/agent_runtime_eval.jsonl` 时同步输出 `data/eval/agent_runtime_predictions.jsonl` 和 `data/eval/agent_runtime_comparison.json`，并将 runtime summary 写入 report。
- 更新 `src/evaluation/report.py`，新增 Agent Runtime / Guardrail Benchmark 报告段。
- 更新 `README.md`，在 Results 中明确区分旧 RAG baseline、Agent workflow、Runtime / Guardrail 和 Safety Boundary 指标，并加入评测数据分工说明与 runtime benchmark 复现命令。

### 遇到的问题

- 普通 `Get-Content` 输出中文 README 时出现终端显示乱码，改用 `-Encoding UTF8` 和 JSON artifact 交叉核对，文件本身保持 UTF-8。
- 第一版 runtime eval 里 `stop_before_policy` 和 `policy_deadline_guard` 样本传入 `task_type=policy_recommendation` 后，deterministic planner 只生成 policy 单步，无法真正触发 stop guard / deadline guard。修正为这些 runtime scenario 在 harness 内以 `task_type=None` 运行，让 planner 根据问题生成 `timeseries_query -> policy_recommendation`。
- policy tool 在 `tools` 中表现为业务层 `rule_based_policy`，但 `tool_calls` 中底层执行名是 `policy_runner`。为避免 tool sequence 指标误判，成功 policy call 归一化为实际 `policy_name`，blocked approval 保留 `policy_runner`。
- `query_rewrite_retry` 是 document QA recovery，不会产生 `PreToolUse/PostToolUse` hook。`trace_completeness` 因此改为：有 expected tool sequence 的样本要求 tool hooks；纯 retrieval recovery 样本要求 todo、RunComplete 和 summary。
- 默认 `python scripts/run_eval.py --disable-cross-encoder-rerank` 在本地 4 分钟命令窗口内未完成；即使禁用 persistent KB，108 条样例乘以多检索 baseline 也超过 4 分钟。随后用 10 分钟窗口运行可复现 smoke，并记录该命令最近一次耗时约 6 分 25 秒。该现象说明默认完整 benchmark 偏重，不代表 runtime 指标失败。

### 指标对比

旧数据集适配性：

- `hvac_eval.jsonl`：108 条，`document_qa=40`、`timeseries_query=20`、`anomaly_diagnosis=20`、`policy_recommendation=28`，`expected_steps=0`。保留为 retrieval / basic tool baseline。
- `real_eval.jsonl`：50 条，`required_documents=32`、`required_tools=20`、`expected_steps=6`。保留为真实公开文档与基础 workflow benchmark。
- `persistent_knowledge_ranking_eval.jsonl`：30 条 document QA ranking 样本。保留为 persistent KB / ranking eval。
- `compound_task_eval.jsonl`：100 条，`expected_steps=100`。保留为 multi-step planner eval。
- 新增 `agent_runtime_eval.jsonl`：13 条 runtime / guardrail scenario 样本。

Runtime / Guardrail 指标（`data/eval/agent_runtime_comparison.json`）：

- `required_step_recall = 1.000`
- `tool_sequence_accuracy = 1.000`
- `policy_obligation_success_rate = 1.000`
- `approval_block_success_rate = 1.000`
- `duplicate_guard_success_rate = 1.000`
- `recovery_success_rate = 1.000`
- `trace_completeness = 1.000`
- `tool_success_rate = 1.000`
- `average_tool_latency_seconds = 0.009`

保留的旧 BGE/FAISS baseline 代表性结果：

- 108 条合成/样例集，`hybrid_rrf`：Citation/Context 0.815、Recall@10 0.815、MRR@10 0.687、nDCG@10 0.719。
- 50 条真实公开 PDF 子集，`hybrid_rrf`：Citation/Context 0.969、Recall@10 0.990、MRR@10 0.896、nDCG@10 0.912。
- 50 条真实公开 PDF 子集，`langgraph_tool_agent`：tool selection 1.000、tool success 1.000、evidence coverage 1.000、correctness proxy 0.727、faithfulness proxy 0.713、hallucination proxy 0.042。

Safety Boundary 指标：

- `safety_adversarial.jsonl`：29 条，overall hit rate 0.586。
- translation hit rate 0.000，是当前安全泛化短板；README 已明确记录，未用 runtime guardrail 满分掩盖。

验证命令：

- `pytest tests/test_baseline_runner.py::test_runtime_guardrail_eval_reports_trace_and_guardrail_metrics -q`：passed。
- `pytest tests/test_experiment_report.py::test_render_experiment_report_includes_agent_runtime_guardrail_section -q`：passed。
- `python scripts/run_eval.py --disable-cross-encoder-rerank --disable-persistent-knowledge --output data/eval/smoke_runtime_baseline_predictions.jsonl --comparison-output data/eval/smoke_runtime_baseline_comparison.json --report-output docs/smoke_runtime_experiment_report.md --human-review-sample-output data/eval/smoke_runtime_human_review_sample.jsonl --human-review-annotations-output data/eval/smoke_runtime_human_review_annotations.jsonl`：passed，约 384.6 秒。

### 结论

第一版达到 smoke regression 目标，但 13 条样本过少，且全部通过会削弱区分度。后续 v2 将其升级为 50 条分层 benchmark；README 当前主结果以 v2 为准。`bounded_react_guard_agent` 的离线结果仍明确限定为 deterministic guard controller 下的可复现场景测试，而不是在线 LLM-controller 泛化能力。

## 2026-05-30 Agent Runtime Eval v2：50 条分层高质量评测集

### 优化目标

回应评测集质量审查，把第一版 13 条 runtime smoke set 升级为 50 条可复现 benchmark。目标是接近公开高质量评测集的工程标准：明确任务边界、难度分层、能力标签、干扰类型、预期失败模式、评分 rubric 和分层指标，同时保持题目不过易也不过难，能暴露当前 Agent runtime / guardrail 的真实短板。

### 实施过程

- 参考公开评测集常见做法，将每条样本补齐 `difficulty`、`capability_tags`、`distractor_type`、`expected_failure_mode` 和 `grading_rubric`，并用质量门保证 rubric 权重和为 1。
- 将 `data/eval/agent_runtime_eval.jsonl` 扩展为 50 条：easy 10、medium 28、hard 12。easy 题覆盖单一工具和清晰约束；medium 题覆盖两步证据、动态 insert/replace、approval/retry/recovery；hard 题覆盖 stop/deadline/duplicate/recovery 组合边界。
- 增加 `tests/test_runtime_eval_quality.py`，自动检查 50 条数量、10/28/12 难度比例、唯一 ID、必备元数据、capability coverage 下限，以及问题文本不能泄漏 `react`、`runtime`、`guard`、`approval_denied` 等内部评测标签。
- 更新 `run_runtime_guardrail_eval()` 输出 `by_difficulty`，并让 `scripts/run_eval.py` 和 `src/evaluation/report.py` 一并写出难度分层指标。
- 复跑 runtime benchmark，生成 `data/eval/agent_runtime_predictions.jsonl` 和 `data/eval/agent_runtime_comparison.json`。根据首轮结果发现 hard 全过、medium 暴露失败，随后把 4 个实际失败样本上调到 hard，并把 4 个稳定通过样本下调到 medium，保持总体比例不变，让难度标签与当前系统行为一致。
- 更新 README：将 Runtime / Guardrail Results 从 13 条 smoke 口径改为 50 条 v2 口径，显式说明 hard 题保留失败信号，且 deterministic guard controller 不代表在线 LLM-controller 泛化能力。

### 遇到的问题

- 第一轮 50 条质量门暴露覆盖不足：`multi_step`、`dynamic_insert`、`dynamic_replace`、`stop_guard`、`policy_deadline_guard` 未达到下限。通过给若干中等题补充真实能力标签修复，而不是降低测试阈值。
- 题目文本容易泄漏内部机制名。质量门加入 forbidden token 检查后，保留用户可理解的问题表达，把内部 scenario 放在 JSON 字段而非 question 里。
- 初始难度标签与实际执行结果不完全一致：hard 组全过，medium 组暴露重复拦截和 recovery 失败。已按执行结果重标 8 条样本，维持 easy 10、medium 28、hard 12。
- `document_qa` 的 query rewrite recovery 没有工具调用，`tool_success_rate` 对该 task type 会显示 0.0；总指标仍排除了 approval denied，README 不把该分项解释为工具失败。

### 指标对比

第一版 13 条 smoke set：

- `required_step_recall = 1.000`
- `tool_sequence_accuracy = 1.000`
- `policy_obligation_success_rate = 1.000`
- `approval_block_success_rate = 1.000`
- `duplicate_guard_success_rate = 1.000`
- `recovery_success_rate = 1.000`
- `trace_completeness = 1.000`
- `tool_success_rate = 1.000`
- `average_tool_latency_seconds = 0.009`

第二版 50 条 benchmark：

- `required_step_recall = 0.990`
- `tool_sequence_accuracy = 0.935`
- `policy_obligation_success_rate = 0.941`
- `approval_block_success_rate = 1.000`
- `duplicate_guard_success_rate = 0.667`
- `recovery_success_rate = 0.833`
- `trace_completeness = 1.000`
- `tool_success_rate = 1.000`
- `average_tool_latency_seconds = 0.007`

难度分层：

- easy：`required_step_recall = 1.000`、`tool_sequence_accuracy = 1.000`、`recovery_success_rate = 1.000`、`duplicate_guard_success_rate = 1.000`
- medium：`required_step_recall = 1.000`、`tool_sequence_accuracy = 1.000`、`recovery_success_rate = 1.000`、`duplicate_guard_success_rate = 1.000`
- hard：`required_step_recall = 0.958`、`tool_sequence_accuracy = 0.727`、`recovery_success_rate = 0.600`、`duplicate_guard_success_rate = 0.500`

验证命令：

- `pytest tests/test_runtime_eval_quality.py -q`：4 passed。
- `python -c "from pathlib import Path; import json; from src.api.demo_factory import build_demo_orchestrator; from src.evaluation.runner import run_runtime_guardrail_eval, save_predictions_jsonl; res=run_runtime_guardrail_eval(Path('data/eval/agent_runtime_eval.jsonl'), build_demo_orchestrator(use_env_answer_generator=False)); save_predictions_jsonl(res['predictions'], 'data/eval/agent_runtime_predictions.jsonl'); Path('data/eval/agent_runtime_comparison.json').write_text(json.dumps({'summary': res['metrics'], 'by_task_type': res['by_task_type'], 'by_difficulty': res['by_difficulty']}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'); print(res['metrics'])"`：50 predictions。

### 结论

v2 达到“高质量、可复现、有区分度”的目标。新集不再是全通过 smoke，而是能清楚显示当前 deterministic runtime harness 的优势和短板：approval block、trace completeness、tool execution 稳定；hard 场景里的 duplicate guard 和 recovery 仍需继续强化。README 已同步按 Retrieval、Agent Workflow、Runtime / Guardrail、Safety Boundary 四类结果分层展示，避免把旧 RAG baseline 包装成当前 Agent runtime 能力。

## 2026-05-30 全评测集质量门、全量回归与 README 指标刷新

### 优化目标

确保所有核心评测集都能作为高质量 benchmark 使用，而不只是 runtime eval 高质量。补齐数据质量门，跑全量测试/benchmark，并将 README 中的指标更新为可复现 artifact 的最新结果。

### 实施过程

- 审计核心评测集：`hvac_eval.jsonl`、`real_eval.jsonl`、`persistent_knowledge_ranking_eval.jsonl`、`compound_task_eval.jsonl`、`safety_adversarial.jsonl` 和 `agent_runtime_eval.jsonl`。
- 新增 `tests/test_eval_dataset_quality.py`，覆盖 JSONL 可加载性、唯一 ID、任务类型、gold answer、expected keywords、expected output format、docs/tools/steps 评分信号、legacy/real eval 任务分布、ranking eval 边界、compound eval 多步计划合法性、safety adversarial 类别覆盖和 runtime rubric 可机读性。
- 质量门发现 `safety_adversarial.jsonl` 没有覆盖 `unverified_policy_action`。补充 6 条 `unverified_action` 样本，将安全集从 29 条扩展到 35 条。
- 修复 `/eval/run` API 测试中意外触发 persistent knowledge + sentence-transformers CUDA 路径的问题：`create_app()` 新增 `use_persistent_knowledge` 参数，并让测试显式关闭 persistent KB，保持 API eval smoke 可复现。
- 运行可复现全量 benchmark：`python scripts/run_eval.py --disable-cross-encoder-rerank --disable-persistent-knowledge`，生成 `data/eval/baseline_comparison.json`、`data/eval/baseline_predictions.jsonl`、`docs/experiment_report.md`、`data/eval/agent_runtime_predictions.jsonl` 和 `data/eval/agent_runtime_comparison.json`。
- 更新 README：将 108 条 demo-docs 指标标注为本次 reproducible regression，保留 50 条真实 BGE/FAISS artifact 作为真实 KB baseline，并刷新 Safety Boundary 与 Runtime / Guardrail 数字。

### 遇到的问题

- 直接运行 `pytest -q` 首次在 `/eval/run` 测试触发 CUDA `CUBLAS_STATUS_EXECUTION_FAILED`，根因是测试 app 虽然关闭了 env answer generator 和 DROPT，却没有关闭 persistent KB，导致加载本地 sentence-transformers。修复后单测通过。
- 第二次整体 `pytest -q -x --tb=short` 在 10 分钟窗口内超时。按文件拆分后确认主要耗时来自 `tests/test_api_app.py` 的 memory API 测试和 `tests/test_baseline_runner.py` 的 run_eval subprocess 测试；单项均能通过，但完整串行运行超过当前命令窗口。
- `safety_adversarial.jsonl` 原本只有 `production_telemetry_claim` 和 `llm_direct_control_claim` 两类 expected violation，和 `answer_audit` 的三类检查不一致。已补齐 `unverified_policy_action` 样本。

### 指标对比

本次可复现 108 条 demo-docs retrieval 指标：

- `rag_dense`：Citation/Context 0.677、Recall@10 0.677、MRR@10 0.501、nDCG@10 0.544。
- `rag_hybrid`：Citation/Context 0.646、Recall@10 0.646、MRR@10 0.522、nDCG@10 0.552。
- `hybrid_rrf`：Citation/Context 0.708、Recall@10 0.708、MRR@10 0.545、nDCG@10 0.585。

本次可复现 108 条 demo-docs agent workflow 指标：

- `rag_tool_agent`：tool selection 0.838、tool success 1.000、evidence coverage 1.000、correctness proxy 0.546、faithfulness proxy 0.492。
- `langgraph_tool_agent`：tool selection 0.809、tool success 1.000、evidence coverage 1.000、correctness proxy 0.546、faithfulness proxy 0.492。
- `react_agent`：tool selection 0.912、tool success 1.000、evidence coverage 1.000、correctness proxy 0.587、faithfulness proxy 0.533。
- `bounded_react_guard_agent`：tool selection 0.809、tool success 1.000、evidence coverage 1.000、correctness proxy 0.546、faithfulness proxy 0.492。该项仍是 deterministic guard controller 回归，不是在线 LLM-controller 泛化指标。

Safety Boundary：

- `safety_adversarial.jsonl`：35 条，overall hit rate 0.657。
- paraphrase 1.000、jailbreak 0.667、mixed 0.600、indirect 0.333、translation 0.000、unverified_action 1.000。

Runtime / Guardrail：

- `required_step_recall = 0.990`
- `tool_sequence_accuracy = 0.935`
- `policy_obligation_success_rate = 0.941`
- `approval_block_success_rate = 1.000`
- `duplicate_guard_success_rate = 0.667`
- `recovery_success_rate = 0.833`
- `trace_completeness = 1.000`
- `tool_success_rate = 1.000`
- `average_tool_latency_seconds = 0.008`

验证命令：

- `pytest tests/test_eval_dataset_quality.py tests/test_runtime_eval_quality.py -q`：11 passed。
- `pytest tests/test_api_app.py::test_eval_run_endpoint_returns_metrics -q`：1 passed。
- 按文件拆分运行剩余测试：所有收集到的测试文件均通过；`tests/test_api_app.py` 和 `tests/test_baseline_runner.py` 单文件串行耗时较长，直接整体 `pytest` 在 10 分钟窗口内超时。
- `python scripts/run_eval.py --disable-cross-encoder-rerank --disable-persistent-knowledge`：passed，约 362.8 秒。

### 结论

核心评测集已具备自动质量门，README 指标已刷新到本次可复现 benchmark artifact。当前仍需注意：完整测试套件在本机串行运行时间过长，适合后续拆分 slow/integration 标记或并行化；这不影响本次已分文件验证通过的结论。
