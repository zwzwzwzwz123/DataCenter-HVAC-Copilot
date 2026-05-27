# 系统设计

## 项目定位

本项目遵循 `DataCenter-HVAC-Copilot-Project-Spec.md` 中定义的 B 路线：RAG + Agent + 工具调用 + 评测。主要数据源是 BEAR HVAC 仿真轨迹。项目叙事可以面向数据中心冷却优化，但不能把 BEAR 仿真轨迹描述成真实数据中心生产数据。

## 当前阶段架构

当前阶段优先搭建可测试、可扩展的后端基础：

1. `src/core/` 定义共享 schema、字段来源和结果格式。
2. `src/ingestion/` 将 BEAR-like 轨迹记录标准化成统一表格格式，并提供 `chz056/BEAR` 的 `BuildingEnvReal` rollout adapter。
3. `src/tools/` 暴露确定性的时序分析工具函数。
4. `src/policies/` 定义 policy adapter 边界，包含 rule-based、MPC-like、offline replay、diffusion 边界和可选 DROPT Guided-DiffFNO checkpoint 推理适配器，避免把未配置或输入不足的模型伪装成可用能力。
5. `src/retrieval/` 提供文档 schema、UTF-8 文档加载、chunk、轻量关键词检索、BM25-style hybrid 检索 baseline、metadata-aware lightweight reranker wrapper 和 extractive RAG baseline。
6. `src/agent/` 提供 deterministic router、baseline orchestrator、LangGraph workflow、可插拔 intent classifier、evidence-grounded answer generator、answer safety audit，以及可选 DeepSeek/Ollama 解释生成适配器。
7. `src/evaluation/` 提供 eval JSONL 读取、最小指标计算、轻量回答质量代理指标和 baseline runner。
8. `src/api/` 提供 FastAPI 服务雏形，暴露 `/health`、`/ask` 和 `/eval/run`。
9. `app/` 提供 Streamlit demo，采用专业深色控制台布局，调用 API 并展示 route、tools、answer、citations、tool results、时序趋势、状态卡片和评测摘要。
10. `data/eval/hvac_eval.jsonl` 保存 108 条评测样例，按文档问答 40、时序查询 20、异常诊断 20、策略建议 28 覆盖四类任务，全部包含人工维护的 `expected_keywords`，代表性样例包含 `must_include` / `must_not_include` 质量代理标注。

第二阶段起步补充了两个可复现能力：

1. API、orchestrator 和 Streamlit demo 会显式返回/展示当前轨迹数据源标签和路径，取值为 `processed_csv`、`bear_sample_csv` 或 `mock`，避免把 BEAR 或 mock 轨迹误表述成真实生产数据。
2. `src/evaluation/runner.py` 提供 `run_baseline_comparison`，对同一评测集运行 `llm_only`、`rag_keyword`、`rag_hybrid`、`rag_hybrid_rerank`、`rag`、`rag_tool_agent` 多组可替换 baseline，并输出 citation hit rate、context recall、expected keyword coverage、lexical answer coverage、tool selection accuracy、tool execution success rate、evidence coverage、answer correctness proxy 与 faithfulness proxy 的整体 summary 和 `by_task_type` 分组指标。
3. `src/evaluation/report.py` 将 comparison summary 和按任务类型指标渲染为 `docs/experiment_report.md`，形成可展示的 Markdown 实验表格和数据边界说明。
4. `app/streamlit_app.py` 提供 Copilot / 评测摘要双页：Copilot 页采用 Mission Control + Grounded Answer 的深色控制台结构，展示 route、tools、citations、retrieved contexts、tool results、data_source、answer generator、evidence 状态卡片和 execution timeline，并将时序 summary / records 渲染为表格和折线图；评测摘要页调用 `/eval/run`，将指标分为 Retrieval、Answer、Tool 和 Quality Proxy 组展示，并在预测预览中展示 citation/tool/audit evidence 标记和 answer length。
5. `src/agent/answer_generator.py` 提供确定性证据约束回答生成器，`src/agent/deepseek_generator.py` 在配置 `DEEPSEEK_API_KEY` 后可调用 DeepSeek 生成最终解释；DeepSeek 只读取检索上下文、引用、工具结果、policy result 和数据源，不直接生成控制动作。
6. `src/agent/answer_audit.py` 对最终回答做确定性安全审计，检查是否把 BEAR 表述为真实生产遥测、是否声称 LLM 直接生成/写回控制动作、以及策略回答中是否出现 policy 工具未返回的动作。

当前 108 条评测集上的 baseline summary 显示：

- `llm_only` 没有引用、检索上下文和工具证据，各项指标均为 0。
- `rag_keyword` 的 citation/context 指标为 0.554，`rag_hybrid` 为 0.585，长噪声/短目标和领域近义压力样例继续体现 BM25-style hybrid 检索优势。
- `rag_dense` 已作为 dense retrieval baseline 纳入 comparison；默认使用 deterministic hash embedding，保证不安装 FAISS 或 sentence-transformers 时仍可复现。真实 FAISS dense retrieval 通过 `pip install -e ".[dev,dense]"` 启用，FAISS 本身不需要 API。
- `rag_hybrid_rerank` 的 citation/context 指标为 0.600，metadata-aware 轻量重排仍能在 108 条样例中拉开与 `rag_hybrid` 的差异。
- `rag_rewrite`、`rag_hyde`、`rag_hyde_rerank` 已作为 Query Rewrite / HyDE baseline 纳入 comparison。当前 `rag_rewrite` citation/context 为 0.646，说明 deterministic query expansion 在 HVAC/BEAR 领域样例中有效；template HyDE 指标较低，提示模板假想文档可能引入查询漂移，后续可替换为 DeepSeek/Ollama HyDE generator 再评估。
- `rag_tool_agent` 在当前确定性路由样例上完成工具选择与执行，tool selection / execution 均为 1.000，并将 evidence coverage 提升到 0.910。
- 代表性样例已加入 `must_include` / `must_not_include` 标注；最新 `rag_tool_agent` 的 `expected_keyword_coverage` 为 0.618，`lexical_answer_coverage` 为 0.285，`answer_correctness_proxy` 为 0.547，`faithfulness_proxy` 为 0.465。这是本地确定性代理指标，不等价于完整人工评审或 LLM judge。
- 按任务类型表显示，工具类任务的 tool selection / execution 已达到 1.000，文档问答仍主要受 citation/context 和回答覆盖率限制，异常诊断与策略建议的自然语言覆盖仍需更强生成器或人工/LLM judge 指标进一步评估。
- `expected_keyword_coverage` 使用人工维护的 `expected_keywords`，比直接对 `gold_answer` 做 token 覆盖更适合中文样例；`answer_correctness_proxy` 和 `faithfulness_proxy` 使用代表性样例上的人工轻标注，`lexical_answer_coverage` 仍保留为备用弱监督指标。

## 数据契约

BEAR 标准字段按来源分为：

- `native`：直接来自 BEAR 导出或环境状态。
- `derived`：可从 BEAR 轨迹中可重复计算得到。
- `optional_derived`：可选派生字段，缺少时不能编造。
- `optional_synthetic`：可选合成字段，只能在明确说明生成方式后使用。

`pue`、`humidity`、`it_load`、`chiller_power` 等字段不能默认视为 BEAR 原生字段。除非后续 BEAR 导出映射能够证明这些字段真实存在，否则它们只能作为 optional、derived 或 synthetic 字段处理。

## BEAR 接入边界

当前 BEAR 接入基于开源仓库 `https://github.com/chz056/BEAR.git` 的真实接口：

- `ParameterGenerator(building, weather, location, root=...)` 创建环境参数。
- `BuildingEnvReal(parameter)` 创建环境。
- `env.reset()` 返回初始 state。
- `env.step(action)` 返回 `(state, reward, terminated, truncated, info)`。
- `env.statelist` 保存 step 前 state。
- `env.actionlist` 保存动作，BEAR 内部会将 action 乘以 `maxpower` 后记录。

BEAR state 布局为：

```text
[zone_temperature(n), outdoor_temp(1), solar_irradiance_or_ghi(n), ground_temp(1), occupancy_power(n)]
```

本项目将其映射为标准 trajectory 字段：`zone_temperature`、`outdoor_temp`、`solar_irradiance`、`ground_temp`、`internal_load`、`control_action`、`reward`、`comfort_violation`。`pue`、`humidity`、`it_load`、`chiller_power` 仍保持 optional，不由 adapter 编造。

## 外部依赖放置方式

BEAR 仓库已放在主项目根目录下的 `BEAR/`。这样它会和主项目一起出现在仓库里，但仍然只作为外部依赖和数据源，不承担主项目核心逻辑。后续如果切换到 conda 环境，只需要在那个环境里安装 BEAR 依赖并复用现有 adapter 和导出脚本。

当前 demo 的轨迹数据优先级是：

1. `data/bear_processed/bear_rollout.csv`
2. `BEAR/BEAR/Data/Exercise2A-mytest.csv`
3. mock trajectory

该选择结果会作为只读 `data_source` 元数据返回给 API 和 demo；它只说明当前演示/评测使用的数据来源，不代表真实数据中心生产遥测。

## Agent 边界

Agent 后续负责：

- 判断用户任务类型。
- 路由到 RAG、时序工具或 policy 工具。
- 整合证据。
- 生成解释性回答。

Agent 不负责直接训练模型，也不直接向环境写入控制动作。控制建议必须来自规则策略、MPC-like policy、DROPT / DiffFNO / Guided-DiffFNO adapter 或 offline replay 等工具。即使启用 checkpoint policy，LLM 也只解释 `policy_result`，不能自行生成或修改控制动作。

`src/policies/dropt_adapter.py` 可加载本地 `models/dropt/policy_best_fno_guided.pth`，并重建与 DROPT 训练脚本一致的 Guided-DiffFNO 推理骨架。该适配器需要显式 20 维 BEAR state vector，布局为 `[zone_temperature(6), outdoor_temp(1), solar_irradiance(6), ground_temp(1), internal_load(6)]`；当 checkpoint 缺失或 state 不完整时会退回 `rule_based_policy` 并在 `notes` 中说明。交互式 `/ask` 默认启用 DROPT policy backend；`/eval/run` 和 `scripts/run_eval.py` 仍显式使用 rule-based policy，避免改变 deterministic baseline 指标口径。

可选 LLM 生成器只负责最终解释生成。它必须基于 `retrieved_contexts`、`citations`、`tool_results`、`policy_result` 和 `data_source`，不能把 BEAR 表述为真实生产遥测，也不能发明新的控制动作。DeepSeek API 未配置或调用失败时，系统回退到 `deterministic_grounded` 生成器。

`/ask` 可在配置 DeepSeek 后使用真实 LLM 解释生成；`scripts/run_eval.py` 和 `/eval/run` 默认关闭 env-driven LLM 生成器，使用 deterministic generator，以避免评测触发批量 API 调用并保持指标可复现。

环境变量可来自 shell 或项目根目录 `.env`。`.env` 加载器不会覆盖 shell 中已有变量，便于本地 demo 与 CI/测试环境分别控制是否启用 DeepSeek。

每个 `/ask` 响应会返回 `answer_audit`。该字段用于演示和调试输出边界，不替代人工审查或完整 LLM judge。

LLM judge adapter 仅作为可选评测辅助，默认关闭。`scripts/run_eval.py` 不带参数时只输出 deterministic metrics；显式传入 `--enable-llm-judge` 时才会额外生成 `llm_judge_correctness` 和 `llm_judge_faithfulness`。当前内置 `deterministic` provider 用于 smoke test 和接口占位，不替代人工评审。

人工评测校准集用于弥补 deterministic proxy 与 optional LLM judge 的可信度边界。`scripts/run_eval.py` 会生成 `data/eval/human_review_sample.jsonl` 和 `data/eval/human_review_annotations.jsonl`；后者只能由人工填写 correctness、faithfulness 和 safety boundary。未填写前，报告仅显示 `pending_human_review`，不能把 proxy 或 judge 结果表述为人工评审。

## LangGraph Agent Workflow

当前 LangGraph 路径已经从单纯 wrapper 升级为可插拔 workflow。交互式 `/ask` 和 Streamlit 默认使用 `workflow_engine=langgraph`；`planner` 节点使用 `src/agent/planner.py` 中的 route planner 生成 1 到 3 个受控步骤。`LANGGRAPH_PLANNER_PROVIDER=auto` 时，如果配置了 `DEEPSEEK_API_KEY` 会优先使用 DeepSeek LLM route planner，否则使用 deterministic planner；LLM plan 输出非法、调用失败或未配置 key 时会回退 deterministic planner，并在 `workflow_trace` 中记录 `planner`、`confidence` 和 `fallback_used`。`/eval/run` 和 `scripts/run_eval.py` 仍保持 deterministic generator、rule-based policy 与可复现评测口径。

Baseline orchestrator 与 LangGraph orchestrator 共享 `AgentTaskExecutor`，因此工具执行、RAG 检索、policy 调用和 answer audit 不再通过 LangGraph 调用 baseline 私有方法完成。交互式 demo 默认展示 route planner trace 和 DROPT policy backend；离线评测中的 `langgraph_tool_agent` 仍与 deterministic baseline 对齐，这是为了保证可复现。

Intent routing 评测由 `scripts/run_intent_eval.py` 单独输出 `data/eval/intent_routing_comparison.json`。该评测不传入 gold `task_type`，直接让 classifier 从问题文本判断 route，并报告 accuracy、fallback rate、按任务类型分组和 confusion matrix。默认 rule-based classifier 在当前 108 条样例上 accuracy 为 0.640；DeepSeek 或 Ollama/Qwen 可作为同一脚本中的 LLM routing backend 进行横向对比。

## 后续扩展方向

后续阶段可以继续加入：

- Qdrant 向量数据库服务化检索。
- 更强的 neural / cross-encoder / LLM reranker，以及更多能检验 reranker 的真实领域压力样例。
- DeepSeek/Ollama HyDE generator 与 template HyDE 的对比实验。
- 更强的真实 LLM 回答生成器评测、提示词审计和输出约束。
- 更完整的 DROPT / Guided-DiffFNO offline replay 指标、state 导出脚本和 policy 对比报告。
- 更完整的 Streamlit 运行日志、案例 walkthrough 和截图素材。
- 扩展 LLM-only、RAG、RAG + Tool Agent 三组 baseline 的指标和样本规模。
- 更完整的人工 correctness / faithfulness 标注或可选 LLM judge adapter。
