# 实验报告

## 数据与边界

当前评测集包含 37 条样例，覆盖文档问答、时序查询、异常诊断和策略建议。轨迹数据来自 BEAR 仿真轨迹、BEAR 样例 CSV 或 mock fallback，不能表述为真实数据中心生产遥测。
其中 37 条样例包含人工维护的 expected_keywords，用于计算中文回答要点覆盖率。

## Baseline 对比

| baseline | citation_hit_rate | context_recall | expected_keyword_coverage | lexical_answer_coverage | tool_selection_accuracy | tool_execution_success_rate | evidence_coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_only | 0.000 | 0.000 | 0.018 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_keyword | 0.533 | 0.533 | 0.176 | 0.078 | 0.000 | 0.000 | 0.405 |
| rag_hybrid | 0.600 | 0.600 | 0.194 | 0.094 | 0.000 | 0.000 | 0.405 |
| rag_hybrid_rerank | 0.600 | 0.600 | 0.194 | 0.094 | 0.000 | 0.000 | 0.405 |
| rag | 0.600 | 0.600 | 0.194 | 0.094 | 0.000 | 0.000 | 0.405 |
| rag_tool_agent | 0.600 | 0.600 | 0.230 | 0.081 | 1.000 | 1.000 | 0.865 |

## 按任务类型指标

| baseline | task_type | citation_hit_rate | context_recall | expected_keyword_coverage | lexical_answer_coverage | tool_selection_accuracy | tool_execution_success_rate | evidence_coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_only | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| llm_only | document_qa | 0.000 | 0.000 | 0.044 | 0.000 | 0.000 | 0.000 | 0.000 |
| llm_only | policy_recommendation | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| llm_only | timeseries_query | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_keyword | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_keyword | document_qa | 0.533 | 0.533 | 0.322 | 0.130 | 0.000 | 0.000 | 0.667 |
| rag_keyword | policy_recommendation | 0.000 | 0.000 | 0.143 | 0.119 | 0.000 | 0.000 | 0.286 |
| rag_keyword | timeseries_query | 0.000 | 0.000 | 0.083 | 0.014 | 0.000 | 0.000 | 0.375 |
| rag_hybrid | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_hybrid | document_qa | 0.600 | 0.600 | 0.367 | 0.169 | 0.000 | 0.000 | 0.667 |
| rag_hybrid | policy_recommendation | 0.000 | 0.000 | 0.143 | 0.119 | 0.000 | 0.000 | 0.286 |
| rag_hybrid | timeseries_query | 0.000 | 0.000 | 0.083 | 0.014 | 0.000 | 0.000 | 0.375 |
| rag_hybrid_rerank | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_hybrid_rerank | document_qa | 0.600 | 0.600 | 0.367 | 0.169 | 0.000 | 0.000 | 0.667 |
| rag_hybrid_rerank | policy_recommendation | 0.000 | 0.000 | 0.143 | 0.119 | 0.000 | 0.000 | 0.286 |
| rag_hybrid_rerank | timeseries_query | 0.000 | 0.000 | 0.083 | 0.014 | 0.000 | 0.000 | 0.375 |
| rag | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag | document_qa | 0.600 | 0.600 | 0.367 | 0.169 | 0.000 | 0.000 | 0.667 |
| rag | policy_recommendation | 0.000 | 0.000 | 0.143 | 0.119 | 0.000 | 0.000 | 0.286 |
| rag | timeseries_query | 0.000 | 0.000 | 0.083 | 0.014 | 0.000 | 0.000 | 0.375 |
| rag_tool_agent | anomaly_diagnosis | 0.000 | 0.000 | 0.143 | 0.000 | 1.000 | 1.000 | 1.000 |
| rag_tool_agent | document_qa | 0.600 | 0.600 | 0.367 | 0.169 | 0.000 | 0.000 | 0.667 |
| rag_tool_agent | policy_recommendation | 0.000 | 0.000 | 0.048 | 0.032 | 1.000 | 1.000 | 1.000 |
| rag_tool_agent | timeseries_query | 0.000 | 0.000 | 0.208 | 0.032 | 1.000 | 1.000 | 1.000 |

## 当前结论

- `llm_only` 不使用检索证据或工具结果，作为最低可复现基线。
- `rag_keyword` 与 `rag_hybrid` 用于比较轻量检索方案；`rag_hybrid` 在 citation/context 指标上优于 `rag_keyword`，说明 BM25-style 长度归一化在当前压力样例中有效。
- `rag_hybrid_rerank` 已纳入对比表；当前指标与 `rag_hybrid` 持平，说明轻量重排接口已具备，但还需要更强重排策略或更多重排压力样例。
- `rag_tool_agent` 在当前确定性路由样例上体现工具选择、工具执行和证据覆盖优势。
