# Demo Walkthrough

这份 walkthrough 用于面试或项目展示。目标是用 5 到 8 分钟说明 DataCenter-HVAC Copilot 不是普通 ChatPDF，而是一个围绕 BEAR HVAC 仿真轨迹的 RAG + Tool Agent + Evaluation 系统。

## 演示前准备

启动 API：

```bash
uvicorn src.api.app:app --reload
```

启动 Streamlit：

```bash
streamlit run app/streamlit_app.py
```

如果本地 `.env` 配置了 `DEEPSEEK_API_KEY`，Copilot tab 的 `Generator` 会显示 `deepseek:<model>`。评测页仍默认使用 deterministic generator，避免批量 API 调用影响复现。

当前 Streamlit 首页是专业深色控制台布局：左侧 `Mission Control` 负责选择 walkthrough、任务类型和问题，右侧 `Grounded Answer` 展示回答、Route / Tools / Generator / Evidence / Audit / Data Source 状态卡片，以及后续的结构化证据面板。演示时可以先说明这是一个 HVAC 仿真 Agent 工作台，而不是普通聊天框。

## 讲解主线

一句话介绍：

> 这个项目把 BEAR HVAC 仿真轨迹作为可控代理场景，结合文档检索、时序工具、策略工具边界和可复现评测，回答数据中心冷却优化类问题。

必须强调的边界：

- BEAR 是 HVAC 物理仿真 / 可控代理场景，不是真实数据中心生产遥测。
- LLM / Agent 只负责任务路由、证据整合和解释生成。
- 控制建议必须来自规则策略、MPC-like、DiffFNO / Guided-DiffFNO adapter 或 offline replay 等工具。
- LLM 不直接生成或写回控制动作。

## Case 1: BEAR 数据边界

在 Copilot tab 选择 `BEAR 数据边界`。

观察点：

- Route 应为 `document_qa`。
- Citations / Retrieved Contexts 应出现数据边界相关文档。
- Answer 中应说明 BEAR 是仿真轨迹或可控代理场景。
- Execution Timeline 的 Data Boundary 应再次提示不能表述为真实生产遥测。
- Safety Audit 应显示审计通过；若出现生产遥测误述，会标记 `production_telemetry_claim`。

讲解重点：

> 这个案例展示系统的防跑偏约束。即使叙事面向数据中心冷却优化，证据层也必须保持 BEAR 仿真数据边界，避免把代理场景包装成生产遥测。

## Case 2: 温度时序查询

在 Copilot tab 选择 `温度时序查询`。

观察点：

- Route 应为 `timeseries_query`。
- Tools 应包含 `query_metric`。
- Tool Results 应返回 `summary`，包括 `count`、`mean`、`min`、`max` 等字段。
- 页面应展示 Metric Summary 表格；若有 records / series，则展示趋势表或折线图。

讲解重点：

> 这个案例展示项目不是纯文档问答。问题涉及最近窗口、区域和指标最大值时，orchestrator 会路由到时序工具，答案基于结构化工具结果，而不是让 LLM 自己猜数值。

## Case 3: 策略建议边界

在 Copilot tab 选择 `策略建议边界`。

观察点：

- Route 应为 `policy_recommendation`。
- Tools 应包含 `rule_based_policy`。
- Tool Results / policy result 应包含 `policy_name`、`recommended_action` 或 notes。
- Answer 应说明控制动作来自策略工具，LLM 只解释工具结果。
- Safety Audit 应显示审计通过；如果答案声称 LLM 直接生成动作，或出现 policy 工具未返回的动作，会标记 violation。

讲解重点：

> 这个案例展示 Agent 与控制策略的分工。LLM 不是控制器；它只解释 policy adapter 的结构化输出。真实 DiffFNO / Guided-DiffFNO 尚未接入时，不能伪造模型效果。

## Evaluation Tab

打开 `评测摘要` tab，运行默认评测。

观察点：

- Metrics 按 Retrieval、Answer、Tool、Quality Proxy 分组。
- `rag_tool_agent` 的工具选择和工具执行在当前确定性样例上为 1.000。
- `evidence_coverage` 高于不使用工具的 baseline。
- Quality Proxy 是本地弱标注代理指标，不等价于人工评审或 LLM judge。

讲解重点：

> 项目不是只展示单个 demo，而是有 100 条 JSONL 评测集和多 baseline comparison。报告由 `scripts/run_eval.py` 生成，保证评测结果可复现。

## 常见追问回答

为什么不用 LLM 直接控制？

> 因为控制动作属于高风险决策，必须来自可验证策略工具、MPC-like、DiffFNO adapter 或 offline replay。LLM 只做证据整合和解释。

DeepSeek 在这里做什么？

> DeepSeek 只负责 final answer generation。输入被限制为 retrieved contexts、citations、tool results、policy result 和 data_source。没有证据时应说明证据不足。

Safety Audit 在这里做什么？

> Safety Audit 是 deterministic guardrail，用于检查最终 answer 是否踩到项目边界，例如把 BEAR 说成真实生产遥测、声称 LLM 直接写回控制动作，或在策略回答中出现 policy 工具未返回的动作。

为什么还没上 LangGraph / FAISS？

> 当前阶段优先完成可复现闭环：RAG、工具、策略边界、API、UI 和评测。LangGraph 和 FAISS/Qdrant 是后续可替换增强项，不影响当前核心能力证明。

评测指标怎么理解？

> citation/context 指标衡量检索证据，tool selection/execution 衡量工具路由，evidence coverage 衡量回答是否携带证据，answer_correctness_proxy 和 faithfulness_proxy 是基于 must_include / must_not_include 的本地弱监督代理指标。

为什么 LLM judge 默认关闭？

> 因为 LLM judge 会引入成本、速度和可复现性问题。默认报告使用 deterministic metrics；只有显式开启时才作为语义质量辅助参考。

## 一分钟简历表达

> 构建 DataCenter-HVAC Copilot：基于 BEAR HVAC 仿真轨迹，设计 RAG + Tool Agent + Evaluation 系统，支持文档问答、时序查询、异常诊断和策略建议。实现 UTF-8 多文档检索、BM25-style hybrid retrieval、metadata-aware reranking、DeepSeek evidence-grounded answer generation、时序工具、policy adapter 边界、FastAPI/Streamlit demo 和 100 条评测集。通过 baseline comparison 验证检索、工具调用、证据覆盖和回答质量代理指标。
