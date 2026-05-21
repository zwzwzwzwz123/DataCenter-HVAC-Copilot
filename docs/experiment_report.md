# 实验报告

## 数据与边界

当前评测集包含 100 条样例，覆盖文档问答、时序查询、异常诊断和策略建议。轨迹数据来自 BEAR 仿真轨迹、BEAR 样例 CSV 或 mock fallback，不能表述为真实数据中心生产遥测。
其中 100 条样例包含人工维护的 expected_keywords，用于计算中文回答要点覆盖率。

## Baseline 对比

| baseline | citation_hit_rate | context_recall | expected_keyword_coverage | lexical_answer_coverage | tool_selection_accuracy | tool_execution_success_rate | evidence_coverage | answer_correctness_proxy | faithfulness_proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_only | 0.000 | 0.000 | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 | 0.022 | 0.022 |
| rag_keyword | 0.554 | 0.554 | 0.372 | 0.173 | 0.000 | 0.000 | 0.590 | 0.475 | 0.418 |
| rag_dense | 0.477 | 0.477 | 0.368 | 0.171 | 0.000 | 0.000 | 0.930 | 0.469 | 0.418 |
| rag_hybrid | 0.585 | 0.585 | 0.382 | 0.182 | 0.000 | 0.000 | 0.590 | 0.478 | 0.396 |
| rag_hybrid_rerank | 0.600 | 0.600 | 0.398 | 0.189 | 0.000 | 0.000 | 0.590 | 0.500 | 0.412 |
| rag | 0.585 | 0.585 | 0.382 | 0.182 | 0.000 | 0.000 | 0.590 | 0.478 | 0.396 |
| rag_tool_agent | 0.385 | 0.385 | 0.618 | 0.285 | 1.000 | 1.000 | 0.910 | 0.547 | 0.465 |

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
| rag_dense | anomaly_diagnosis | 0.500 | 0.500 | 0.317 | 0.160 | 0.000 | 0.000 | 0.950 | 0.417 | 0.417 |
| rag_dense | document_qa | 0.500 | 0.500 | 0.521 | 0.204 | 0.000 | 0.000 | 0.900 | 0.597 | 0.486 |
| rag_dense | policy_recommendation | 0.385 | 0.385 | 0.250 | 0.185 | 0.000 | 0.000 | 0.950 | 0.365 | 0.365 |
| rag_dense | timeseries_query | 0.000 | 0.000 | 0.233 | 0.102 | 0.000 | 0.000 | 0.950 | 0.167 | 0.167 |
| rag_hybrid | anomaly_diagnosis | 0.333 | 0.333 | 0.167 | 0.092 | 0.000 | 0.000 | 0.300 | 0.183 | 0.183 |
| rag_hybrid | document_qa | 0.625 | 0.625 | 0.604 | 0.246 | 0.000 | 0.000 | 0.775 | 0.667 | 0.528 |
| rag_hybrid | policy_recommendation | 0.692 | 0.692 | 0.300 | 0.200 | 0.000 | 0.000 | 0.550 | 0.438 | 0.375 |
| rag_hybrid | timeseries_query | 0.000 | 0.000 | 0.233 | 0.124 | 0.000 | 0.000 | 0.550 | 0.167 | 0.167 |
| rag_hybrid_rerank | anomaly_diagnosis | 0.417 | 0.417 | 0.183 | 0.114 | 0.000 | 0.000 | 0.300 | 0.233 | 0.233 |
| rag_hybrid_rerank | document_qa | 0.625 | 0.625 | 0.637 | 0.255 | 0.000 | 0.000 | 0.775 | 0.694 | 0.542 |
| rag_hybrid_rerank | policy_recommendation | 0.692 | 0.692 | 0.300 | 0.200 | 0.000 | 0.000 | 0.550 | 0.438 | 0.375 |
| rag_hybrid_rerank | timeseries_query | 0.000 | 0.000 | 0.233 | 0.124 | 0.000 | 0.000 | 0.550 | 0.167 | 0.167 |
| rag | anomaly_diagnosis | 0.333 | 0.333 | 0.167 | 0.092 | 0.000 | 0.000 | 0.300 | 0.183 | 0.183 |
| rag | document_qa | 0.625 | 0.625 | 0.604 | 0.246 | 0.000 | 0.000 | 0.775 | 0.667 | 0.528 |
| rag | policy_recommendation | 0.692 | 0.692 | 0.300 | 0.200 | 0.000 | 0.000 | 0.550 | 0.438 | 0.375 |
| rag | timeseries_query | 0.000 | 0.000 | 0.233 | 0.124 | 0.000 | 0.000 | 0.550 | 0.167 | 0.167 |
| rag_tool_agent | anomaly_diagnosis | 0.000 | 0.000 | 0.533 | 0.242 | 1.000 | 1.000 | 1.000 | 0.367 | 0.367 |
| rag_tool_agent | document_qa | 0.625 | 0.625 | 0.629 | 0.271 | 0.000 | 0.000 | 0.775 | 0.597 | 0.444 |
| rag_tool_agent | policy_recommendation | 0.000 | 0.000 | 0.517 | 0.260 | 1.000 | 1.000 | 1.000 | 0.531 | 0.490 |
| rag_tool_agent | timeseries_query | 0.000 | 0.000 | 0.783 | 0.381 | 1.000 | 1.000 | 1.000 | 0.833 | 0.833 |

## Human Calibration

人工校准集用于核对 deterministic proxy 和 optional LLM judge 的可信度；不会把 deterministic proxy 或 LLM judge 当作人工评审。

| sample_count | labeled_count | pending_count | mean_correctness | mean_faithfulness | safety_pass_rate | status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 24 | 0 | 24 | null | null | null | pending_human_review |

## 当前结论

- `llm_only` 不使用检索证据或工具结果，作为最低可复现基线。
- `rag_dense` 使用 deterministic hash embedding 作为默认 dense retrieval baseline；真实 FAISS + sentence-transformers 作为可选增强，避免默认评测依赖模型下载或外部 API。
- `rag_keyword` 与 `rag_hybrid` 用于比较轻量检索方案；`rag_hybrid` 在 citation/context 指标上优于 `rag_keyword`，说明 BM25-style 长度归一化在当前压力样例中有效。
- `rag_hybrid_rerank` 在当前评测中进一步提升 citation/context 指标，可作为后续替换为 cross-encoder 或 LLM reranker 的接口基线。
- `rag_tool_agent` 在当前确定性路由样例上体现工具选择、工具执行和证据覆盖优势。
