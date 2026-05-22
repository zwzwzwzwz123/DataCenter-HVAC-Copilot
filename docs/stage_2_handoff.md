# Stage 2 Handoff

## 2026-05-21 LangGraph Trace Panel

- `/ask` 新增 `workflow_engine` 参数，支持 `deterministic` 和 `langgraph` 两种运行路径。
- `langgraph` 路径会返回 `workflow_engine=langgraph` 与真实 `workflow_trace`，不再只停留在代码内部。
- Streamlit Copilot tab 新增 workflow 选择器和 `LangGraph Workflow Trace` 面板，可展示 step / node / route / tools / evidence / audit。
- 这一步让 Stage 5 的 LangGraph 能力在面试 demo 中可见，后续截图可以直接截该面板。

> 本文档记录 DataCenter-HVAC Copilot 从“简历可展示基础版”进入 Stage 2 后的进展、当前状态和下一步任务。Stage 2 重点是补 RAG 技术硬度和 Agent 深度：先真实 embedding / FAISS，再 LangGraph。

## Stage 2 目标

1. **RAG 技术硬度**：把 `rag_dense` 从默认 deterministic hash embedding 的可复现占位，推进到真实 `sentence-transformers + FAISS` 指标对比。
2. **Agent 深度**：保留 deterministic router 作为 baseline，再新增 LangGraph workflow，用 state graph 展示 intent、retrieval/tool、evidence aggregation、answer 和 audit 的编排过程。
3. **简历表述边界**：完成前不提前写“基于 LangGraph”或“真实 FAISS/BGE 指标”；完成后用实验报告中的数字更新 README 和简历描述。

## 已完成的 Stage 2 前置工作

- 使用你的 DROPT 源码仓库中的完整 BEAR 环境，生成了 14 天逐小时、6 zone 的 `data/bear_processed/bear_rollout.csv`。
- 当前 demo/API 优先加载 `processed_csv`，轨迹规模为 2016 行、19 列。
- 人工评测从当前阻塞项调整为可选增强项；默认报告强调 deterministic metrics + quality proxy，LLM judge 只能表述为 LLM-as-Judge。
- 添加 Dockerfile / docker-compose，并让 Streamlit 支持 `HVAC_COPILOT_API_BASE_URL`，便于容器间连接。
- 新增 `docs/resume_readiness_plan.md`，记录当前简历可展示边界和后续顺序。

## 第 4 阶段：RAG 技术硬度

| 顺序 | 任务 | 状态 | 验收标准 |
|---|---|---|---|
| 13 | 启用 `sentence-transformers` + FAISS | 已完成 | 已用 `BAAI/bge-small-zh-v1.5` 本地生成 embedding，并用 FAISS IndexFlatIP 检索 |
| 14 | 增加真实 dense baseline | 已完成 | 报告运行配置区分 `dense_provider=sentence-transformers`、`dense_backend=faiss`、`dense_model=BAAI/bge-small-zh-v1.5` |
| 15 | 重跑 keyword / dense / hybrid / rerank 对比 | 已完成 | `rag_dense` 的 citation/context 为 0.692，高于 `rag_keyword` 0.554、`rag_hybrid` 0.585、`rag_hybrid_rerank` 0.600 |
| 16 | 更新实验报告和 README 指标 | 已完成 | `docs/experiment_report.md` 和 README 使用真实报告指标 |

当前相关入口：

```bash
pip install -e ".[dev,dense]"
python scripts/run_eval.py --dense-provider sentence-transformers --dense-backend faiss
```

本轮实际运行命令：

```bash
python scripts/run_eval.py --dense-provider sentence-transformers --dense-backend faiss --dense-model BAAI/bge-small-zh-v1.5
```

最新整体指标：

| baseline | citation/context | expected_keyword_coverage | answer_correctness_proxy | faithfulness_proxy |
|---|---:|---:|---:|---:|
| rag_keyword | 0.554 | 0.372 | 0.475 | 0.418 |
| rag_dense (BGE + FAISS) | 0.692 | 0.528 | 0.654 | 0.566 |
| rag_hybrid | 0.585 | 0.382 | 0.478 | 0.396 |
| rag_hybrid_rerank | 0.600 | 0.398 | 0.500 | 0.412 |
| rag_tool_agent | 0.385 | 0.618 | 0.547 | 0.465 |

阶段结论：真实 BGE + FAISS 在当前文档问答和策略相关样例中明显提升 citation/context 与 quality proxy；`rag_tool_agent` 的优势仍主要体现在工具选择、工具执行和证据覆盖，而不是纯检索召回。

## 第 5 阶段：Agent 深度

| 顺序 | 任务 | 状态 | 验收标准 |
|---|---|---|---|
| 17 | 保留 deterministic router 作为 baseline | 已完成 | 不破坏现有可复现评测 |
| 18 | 新增 LangGraph workflow | 已完成 | 节点包括 intent_classifier、retrieval/timeseries/anomaly/policy tool、evidence_aggregator、answer_audit |
| 19 | 对比 deterministic vs LangGraph routing | 已完成 | `langgraph_tool_agent` 已进入 baseline comparison，指标与 deterministic `rag_tool_agent` 对齐 |
| 20 | README 展示工作流图 | 已完成 | README 增加 LangGraph workflow 图，可讲 state、node、edge、fallback |

最新 LangGraph 对比指标：

| baseline | tool_selection_accuracy | tool_execution_success_rate | evidence_coverage | answer_correctness_proxy | faithfulness_proxy |
|---|---:|---:|---:|---:|---:|
| rag_tool_agent | 1.000 | 1.000 | 0.910 | 0.547 | 0.465 |
| langgraph_tool_agent | 1.000 | 1.000 | 0.910 | 0.547 | 0.465 |

阶段结论：当前 LangGraph 版本刻意保持与 deterministic baseline 一致的工具行为和指标，用于展示 StateGraph 编排、workflow trace 和未来可替换节点，而不是为了改变可复现评测口径。

## 当前不能提前写的内容

- 不能写“基于 LangGraph 的 Agent 工作流”，直到 Stage 5 完成。
- 不能写“人工评测已完成”，除非 `human_review_annotations.jsonl` 由人工填写并重跑报告。
- 不能写“真实数据中心生产遥测”，当前数据是 BEAR HVAC 仿真 rollout。

## 当前可以写的内容

- 可以写“接入 `BAAI/bge-small-zh-v1.5` + FAISS 真实 dense retrieval，并在 100 条评测集上与 keyword / hybrid / rerank baseline 对比”。
- 可以写“真实 dense retrieval 的 citation/context 指标为 0.692，高于 keyword 的 0.554、hybrid 的 0.585 和 hybrid+rerank 的 0.600”。
- 可以写“`rag_tool_agent` 的优势主要体现在工具选择、工具执行和结构化证据覆盖，而非纯检索召回”。
- 可以写“新增 LangGraph StateGraph workflow，保留 deterministic router 作为 baseline，节点包括 intent、retrieval/tool、evidence aggregation 和 audit；`langgraph_tool_agent` 与 deterministic baseline 指标对齐，保证可复现性”。

## 下一步建议

第 4 和第 5 阶段已完成。下一步建议补 README / demo 截图和架构图，把 Stage 2 的 RAG 指标与 LangGraph workflow 可视化展示出来。
