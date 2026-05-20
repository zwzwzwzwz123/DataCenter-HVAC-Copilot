# 实验报告

## 数据与边界

当前评测集包含 49 条样例，覆盖文档问答、时序查询、异常诊断和策略建议。轨迹数据来自 BEAR 仿真轨迹、BEAR 样例 CSV 或 mock fallback，不能表述为真实数据中心生产遥测。
其中 49 条样例包含人工维护的 expected_keywords，用于计算中文回答要点覆盖率。

## Baseline 对比

| baseline | citation_hit_rate | context_recall | expected_keyword_coverage | lexical_answer_coverage | tool_selection_accuracy | tool_execution_success_rate | evidence_coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_only | 0.000 | 0.000 | 0.014 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_keyword | 0.519 | 0.519 | 0.303 | 0.135 | 0.000 | 0.000 | 0.510 |
| rag_hybrid | 0.593 | 0.593 | 0.330 | 0.153 | 0.000 | 0.000 | 0.510 |
| rag_hybrid_rerank | 0.630 | 0.630 | 0.344 | 0.157 | 0.000 | 0.000 | 0.510 |
| rag | 0.593 | 0.593 | 0.330 | 0.153 | 0.000 | 0.000 | 0.510 |
| rag_tool_agent | 0.481 | 0.481 | 0.316 | 0.119 | 1.000 | 1.000 | 0.878 |

## 按任务类型指标

| baseline | task_type | citation_hit_rate | context_recall | expected_keyword_coverage | lexical_answer_coverage | tool_selection_accuracy | tool_execution_success_rate | evidence_coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_only | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| llm_only | document_qa | 0.000 | 0.000 | 0.029 | 0.000 | 0.000 | 0.000 | 0.000 |
| llm_only | policy_recommendation | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| llm_only | timeseries_query | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_keyword | anomaly_diagnosis | 1.000 | 1.000 | 0.185 | 0.100 | 0.000 | 0.000 | 0.222 |
| rag_keyword | document_qa | 0.478 | 0.478 | 0.486 | 0.190 | 0.000 | 0.000 | 0.739 |
| rag_keyword | policy_recommendation | 0.500 | 0.500 | 0.148 | 0.120 | 0.000 | 0.000 | 0.333 |
| rag_keyword | timeseries_query | 0.000 | 0.000 | 0.083 | 0.032 | 0.000 | 0.000 | 0.375 |
| rag_hybrid | anomaly_diagnosis | 1.000 | 1.000 | 0.185 | 0.100 | 0.000 | 0.000 | 0.222 |
| rag_hybrid | document_qa | 0.565 | 0.565 | 0.543 | 0.229 | 0.000 | 0.000 | 0.739 |
| rag_hybrid | policy_recommendation | 0.500 | 0.500 | 0.148 | 0.120 | 0.000 | 0.000 | 0.333 |
| rag_hybrid | timeseries_query | 0.000 | 0.000 | 0.083 | 0.032 | 0.000 | 0.000 | 0.375 |
| rag_hybrid_rerank | anomaly_diagnosis | 1.000 | 1.000 | 0.185 | 0.100 | 0.000 | 0.000 | 0.222 |
| rag_hybrid_rerank | document_qa | 0.609 | 0.609 | 0.572 | 0.237 | 0.000 | 0.000 | 0.739 |
| rag_hybrid_rerank | policy_recommendation | 0.500 | 0.500 | 0.148 | 0.120 | 0.000 | 0.000 | 0.333 |
| rag_hybrid_rerank | timeseries_query | 0.000 | 0.000 | 0.083 | 0.032 | 0.000 | 0.000 | 0.375 |
| rag | anomaly_diagnosis | 1.000 | 1.000 | 0.185 | 0.100 | 0.000 | 0.000 | 0.222 |
| rag | document_qa | 0.565 | 0.565 | 0.543 | 0.229 | 0.000 | 0.000 | 0.739 |
| rag | policy_recommendation | 0.500 | 0.500 | 0.148 | 0.120 | 0.000 | 0.000 | 0.333 |
| rag | timeseries_query | 0.000 | 0.000 | 0.083 | 0.032 | 0.000 | 0.000 | 0.375 |
| rag_tool_agent | anomaly_diagnosis | 0.000 | 0.000 | 0.111 | 0.011 | 1.000 | 1.000 | 1.000 |
| rag_tool_agent | document_qa | 0.565 | 0.565 | 0.543 | 0.229 | 0.000 | 0.000 | 0.739 |
| rag_tool_agent | policy_recommendation | 0.000 | 0.000 | 0.037 | 0.025 | 1.000 | 1.000 | 1.000 |
| rag_tool_agent | timeseries_query | 0.000 | 0.000 | 0.208 | 0.032 | 1.000 | 1.000 | 1.000 |

## 当前结论

- `llm_only` 不使用检索证据或工具结果，作为最低可复现基线。
- `rag_keyword` 与 `rag_hybrid` 用于比较轻量检索方案；`rag_hybrid` 在 citation/context 指标上优于 `rag_keyword`，说明 BM25-style 长度归一化在当前压力样例中有效。
- `rag_hybrid_rerank` 在当前评测中进一步提升 citation/context 指标，可作为后续替换为 cross-encoder 或 LLM reranker 的接口基线。
- `rag_tool_agent` 在当前确定性路由样例上体现工具选择、工具执行和证据覆盖优势。
