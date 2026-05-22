# 实验报告

## 数据与边界

当前评测集包含 100 条样例，覆盖文档问答、时序查询、异常诊断和策略建议。轨迹数据来自 BEAR 仿真轨迹、BEAR 样例 CSV 或 mock fallback，不能表述为真实数据中心生产遥测。
其中 100 条样例包含人工维护的 expected_keywords，用于计算中文回答要点覆盖率。

## 运行配置

- dense_provider: `sentence-transformers`
- dense_backend: `faiss`
- dense_model: `BAAI/bge-small-zh-v1.5`

## Baseline 对比

| baseline | citation_hit_rate | context_recall | expected_keyword_coverage | lexical_answer_coverage | tool_selection_accuracy | tool_execution_success_rate | evidence_coverage | answer_correctness_proxy | faithfulness_proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_only | 0.000 | 0.000 | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 | 0.022 | 0.022 |
| rag_keyword | 0.554 | 0.554 | 0.372 | 0.173 | 0.000 | 0.000 | 0.590 | 0.475 | 0.418 |
| rag_dense | 0.692 | 0.692 | 0.528 | 0.240 | 0.000 | 0.000 | 1.000 | 0.654 | 0.566 |
| rag_hybrid | 0.585 | 0.585 | 0.382 | 0.182 | 0.000 | 0.000 | 0.590 | 0.478 | 0.396 |
| rag_hybrid_rerank | 0.600 | 0.600 | 0.398 | 0.189 | 0.000 | 0.000 | 0.590 | 0.500 | 0.412 |
| rag_rewrite | 0.646 | 0.646 | 0.584 | 0.279 | 0.000 | 0.000 | 1.000 | 0.601 | 0.500 |
| rag_hyde | 0.246 | 0.246 | 0.182 | 0.112 | 0.000 | 0.000 | 1.000 | 0.233 | 0.214 |
| rag_hyde_rerank | 0.338 | 0.338 | 0.195 | 0.128 | 0.000 | 0.000 | 1.000 | 0.242 | 0.204 |
| rag | 0.585 | 0.585 | 0.382 | 0.182 | 0.000 | 0.000 | 0.590 | 0.478 | 0.396 |
| rag_tool_agent | 0.385 | 0.385 | 0.618 | 0.283 | 1.000 | 1.000 | 0.910 | 0.547 | 0.465 |
| langgraph_tool_agent | 0.385 | 0.385 | 0.618 | 0.283 | 1.000 | 1.000 | 0.910 | 0.547 | 0.465 |

## 按任务类型指标

| baseline | task_type | citation_hit_rate | context_recall | expected_keyword_coverage | lexical_answer_coverage | tool_selection_accuracy | tool_execution_success_rate | evidence_coverage | answer_correctness_proxy | faithfulness_proxy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_only | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| llm_only | document_qa | 0.000 | 0.000 | 0.017 | 0.000 | 0.000 | 0.000 | 0.000 | 0.028 | 0.028 |
| llm_only | policy_recommendation | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | 0.031 |
| llm_only | timeseries_query | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_keyword | anomaly_diagnosis | 0.333 | 0.333 | 0.167 | 0.092 | 0.000 | 0.000 | 0.300 | 0.183 | 0.183 |
| rag_keyword | document_qa | 0.600 | 0.600 | 0.596 | 0.234 | 0.000 | 0.000 | 0.775 | 0.681 | 0.569 |
| rag_keyword | policy_recommendation | 0.615 | 0.615 | 0.300 | 0.190 | 0.000 | 0.000 | 0.550 | 0.438 | 0.417 |
| rag_keyword | timeseries_query | 0.000 | 0.000 | 0.200 | 0.117 | 0.000 | 0.000 | 0.550 | 0.000 | 0.000 |
| rag_dense | anomaly_diagnosis | 0.750 | 0.750 | 0.567 | 0.268 | 0.000 | 0.000 | 1.000 | 0.750 | 0.750 |
| rag_dense | document_qa | 0.600 | 0.600 | 0.704 | 0.260 | 0.000 | 0.000 | 1.000 | 0.750 | 0.556 |
| rag_dense | policy_recommendation | 0.923 | 0.923 | 0.417 | 0.292 | 0.000 | 0.000 | 1.000 | 0.573 | 0.573 |
| rag_dense | timeseries_query | 0.000 | 0.000 | 0.250 | 0.118 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| rag_hybrid | anomaly_diagnosis | 0.333 | 0.333 | 0.167 | 0.092 | 0.000 | 0.000 | 0.300 | 0.183 | 0.183 |
| rag_hybrid | document_qa | 0.625 | 0.625 | 0.604 | 0.246 | 0.000 | 0.000 | 0.775 | 0.667 | 0.528 |
| rag_hybrid | policy_recommendation | 0.692 | 0.692 | 0.300 | 0.200 | 0.000 | 0.000 | 0.550 | 0.438 | 0.375 |
| rag_hybrid | timeseries_query | 0.000 | 0.000 | 0.233 | 0.124 | 0.000 | 0.000 | 0.550 | 0.167 | 0.167 |
| rag_hybrid_rerank | anomaly_diagnosis | 0.417 | 0.417 | 0.183 | 0.114 | 0.000 | 0.000 | 0.300 | 0.233 | 0.233 |
| rag_hybrid_rerank | document_qa | 0.625 | 0.625 | 0.637 | 0.255 | 0.000 | 0.000 | 0.775 | 0.694 | 0.542 |
| rag_hybrid_rerank | policy_recommendation | 0.692 | 0.692 | 0.300 | 0.200 | 0.000 | 0.000 | 0.550 | 0.438 | 0.375 |
| rag_hybrid_rerank | timeseries_query | 0.000 | 0.000 | 0.233 | 0.124 | 0.000 | 0.000 | 0.550 | 0.167 | 0.167 |
| rag_rewrite | anomaly_diagnosis | 0.500 | 0.500 | 0.583 | 0.309 | 0.000 | 0.000 | 1.000 | 0.450 | 0.450 |
| rag_rewrite | document_qa | 0.675 | 0.675 | 0.677 | 0.260 | 0.000 | 0.000 | 1.000 | 0.722 | 0.542 |
| rag_rewrite | policy_recommendation | 0.692 | 0.692 | 0.450 | 0.286 | 0.000 | 0.000 | 1.000 | 0.542 | 0.479 |
| rag_rewrite | timeseries_query | 0.000 | 0.000 | 0.533 | 0.279 | 0.000 | 0.000 | 1.000 | 0.444 | 0.444 |
| rag_hyde | anomaly_diagnosis | 0.000 | 0.000 | 0.117 | 0.060 | 0.000 | 0.000 | 1.000 | 0.217 | 0.217 |
| rag_hyde | document_qa | 0.300 | 0.300 | 0.237 | 0.112 | 0.000 | 0.000 | 1.000 | 0.222 | 0.181 |
| rag_hyde | policy_recommendation | 0.308 | 0.308 | 0.267 | 0.243 | 0.000 | 0.000 | 1.000 | 0.281 | 0.281 |
| rag_hyde | timeseries_query | 0.000 | 0.000 | 0.050 | 0.032 | 0.000 | 0.000 | 1.000 | 0.111 | 0.111 |
| rag_hyde_rerank | anomaly_diagnosis | 0.000 | 0.000 | 0.017 | 0.018 | 0.000 | 0.000 | 1.000 | 0.033 | 0.033 |
| rag_hyde_rerank | document_qa | 0.425 | 0.425 | 0.287 | 0.156 | 0.000 | 0.000 | 1.000 | 0.264 | 0.222 |
| rag_hyde_rerank | policy_recommendation | 0.385 | 0.385 | 0.367 | 0.286 | 0.000 | 0.000 | 1.000 | 0.365 | 0.302 |
| rag_hyde_rerank | timeseries_query | 0.000 | 0.000 | 0.017 | 0.023 | 0.000 | 0.000 | 1.000 | 0.111 | 0.111 |
| rag | anomaly_diagnosis | 0.333 | 0.333 | 0.167 | 0.092 | 0.000 | 0.000 | 0.300 | 0.183 | 0.183 |
| rag | document_qa | 0.625 | 0.625 | 0.604 | 0.246 | 0.000 | 0.000 | 0.775 | 0.667 | 0.528 |
| rag | policy_recommendation | 0.692 | 0.692 | 0.300 | 0.200 | 0.000 | 0.000 | 0.550 | 0.438 | 0.375 |
| rag | timeseries_query | 0.000 | 0.000 | 0.233 | 0.124 | 0.000 | 0.000 | 0.550 | 0.167 | 0.167 |
| rag_tool_agent | anomaly_diagnosis | 0.000 | 0.000 | 0.533 | 0.242 | 1.000 | 1.000 | 1.000 | 0.367 | 0.367 |
| rag_tool_agent | document_qa | 0.625 | 0.625 | 0.629 | 0.269 | 0.000 | 0.000 | 0.775 | 0.597 | 0.444 |
| rag_tool_agent | policy_recommendation | 0.000 | 0.000 | 0.517 | 0.254 | 1.000 | 1.000 | 1.000 | 0.531 | 0.490 |
| rag_tool_agent | timeseries_query | 0.000 | 0.000 | 0.783 | 0.381 | 1.000 | 1.000 | 1.000 | 0.833 | 0.833 |
| langgraph_tool_agent | anomaly_diagnosis | 0.000 | 0.000 | 0.533 | 0.242 | 1.000 | 1.000 | 1.000 | 0.367 | 0.367 |
| langgraph_tool_agent | document_qa | 0.625 | 0.625 | 0.629 | 0.269 | 0.000 | 0.000 | 0.775 | 0.597 | 0.444 |
| langgraph_tool_agent | policy_recommendation | 0.000 | 0.000 | 0.517 | 0.254 | 1.000 | 1.000 | 1.000 | 0.531 | 0.490 |
| langgraph_tool_agent | timeseries_query | 0.000 | 0.000 | 0.783 | 0.381 | 1.000 | 1.000 | 1.000 | 0.833 | 0.833 |

## Human Calibration

人工校准集用于核对 deterministic proxy 和 optional LLM judge 的可信度；不会把 deterministic proxy 或 LLM judge 当作人工评审。

| sample_count | labeled_count | pending_count | mean_correctness | mean_faithfulness | safety_pass_rate | status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 24 | 0 | 24 | null | null | null | pending_human_review |

## 当前结论

- `llm_only` 不使用检索证据或工具结果，作为最低可复现基线。
- `rag_dense` 使用真实 sentence-transformers embedding + FAISS 本地向量索引；该运行可用于面试中说明真实语义检索 baseline，但仍需结合 hybrid/rerank 指标判断中文 HVAC 场景效果。
- `rag_keyword` 与 `rag_hybrid` 用于比较轻量检索方案；`rag_hybrid` 在 citation/context 指标上优于 `rag_keyword`，说明 BM25-style 长度归一化在当前压力样例中有效。
- `rag_hybrid_rerank` 在当前评测中进一步提升 citation/context 指标，可作为后续替换为 cross-encoder 或 LLM reranker 的接口基线。
- Query Rewrite / HyDE 已作为 deterministic query expansion baseline 纳入对比；当前 context_recall 最高的是 `rag_rewrite`，可用于评估 raw query、rewrite 和 template HyDE 在 HVAC/BEAR 领域检索中的收益，再决定是否替换为 DeepSeek/Ollama HyDE generator。
- `rag_tool_agent` 在当前确定性路由样例上体现工具选择、工具执行和证据覆盖优势。
- `langgraph_tool_agent` 保留与 deterministic `rag_tool_agent` 一致的工具行为和指标，用于展示 StateGraph 编排、workflow trace 和可选 DeepSeek/Ollama LLM intent classifier，而不是改变当前可复现评测口径。
- `scripts/run_intent_eval.py` 单独评测 intent routing accuracy；默认 rule-based classifier 在当前 100 条样例上 accuracy 为 0.640，并输出 `data/eval/intent_routing_comparison.json` 作为 keyword vs LLM routing 对比入口。
