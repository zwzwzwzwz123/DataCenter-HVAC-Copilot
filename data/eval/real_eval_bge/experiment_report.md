# 实验报告

## 数据与边界

当前评测集包含 50 条样例，覆盖文档问答、时序查询、异常诊断和策略建议。轨迹数据来自 BEAR 仿真轨迹、BEAR 样例 CSV 或 mock fallback，不能表述为真实数据中心生产遥测。
其中 50 条样例包含人工维护的 expected_keywords，用于计算中文回答要点覆盖率。

## 运行配置

- dense_provider: `sentence-transformers`
- dense_backend: `faiss`
- dense_model: `BAAI/bge-small-zh-v1.5`

## Baseline 对比

| baseline | citation_hit_rate | context_recall | expected_keyword_coverage | lexical_answer_coverage | tool_selection_accuracy | tool_execution_success_rate | evidence_coverage | answer_correctness_proxy | faithfulness_proxy | grounding_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_only | 0.000 | 0.000 | 0.005 | 0.000 | 0.000 | 0.000 | 0.000 | 0.013 | 0.013 | 0.000 |
| rag_keyword | 0.719 | 0.719 | 0.406 | 0.220 | 0.000 | 0.000 | 0.760 | 0.177 | 0.177 | 0.000 |
| rag_keyword_grounded | 0.719 | 0.719 | 0.471 | 0.231 | 0.000 | 0.000 | 0.760 | 0.457 | 0.450 | 0.969 |
| rag_dense | 0.562 | 0.562 | 0.309 | 0.183 | 0.000 | 0.000 | 1.000 | 0.148 | 0.148 | 0.000 |
| rag_dense_grounded | 0.562 | 0.562 | 0.470 | 0.239 | 0.000 | 0.000 | 1.000 | 0.517 | 0.510 | 1.000 |
| rag_hybrid | 0.781 | 0.781 | 0.400 | 0.233 | 0.000 | 0.000 | 0.760 | 0.191 | 0.191 | 0.000 |
| hybrid_rrf | 0.812 | 0.812 | 0.424 | 0.246 | 0.000 | 0.000 | 1.000 | 0.205 | 0.205 | 0.000 |
| rag_hybrid_rerank | 0.656 | 0.656 | 0.378 | 0.228 | 0.000 | 0.000 | 0.760 | 0.182 | 0.182 | 0.000 |
| rag_rewrite | 0.562 | 0.562 | 0.325 | 0.211 | 0.000 | 0.000 | 1.000 | 0.150 | 0.150 | 0.000 |
| rewrite_llm | 0.594 | 0.594 | 0.342 | 0.207 | 0.000 | 0.000 | 0.720 | 0.163 | 0.163 | 0.000 |
| rag_rewrite_grounded | 0.562 | 0.562 | 0.453 | 0.234 | 0.000 | 0.000 | 1.000 | 0.495 | 0.488 | 1.000 |
| rag_hyde | 0.469 | 0.469 | 0.303 | 0.187 | 0.000 | 0.000 | 1.000 | 0.155 | 0.155 | 0.000 |
| rag_hyde_rerank | 0.500 | 0.500 | 0.288 | 0.192 | 0.000 | 0.000 | 1.000 | 0.128 | 0.128 | 0.000 |
| rag | 0.562 | 0.562 | 0.309 | 0.183 | 0.000 | 0.000 | 1.000 | 0.148 | 0.148 | 0.000 |
| rag_tool_agent | 0.562 | 0.562 | 0.643 | 0.315 | 0.850 | 1.000 | 1.000 | 0.703 | 0.690 | 0.938 |
| langgraph_tool_agent | 0.562 | 0.562 | 0.665 | 0.320 | 1.000 | 1.000 | 1.000 | 0.727 | 0.713 | 0.938 |
| react_agent | 0.562 | 0.562 | 0.648 | 0.316 | 0.900 | 1.000 | 1.000 | 0.713 | 0.700 | 0.938 |

## 按任务类型指标

| baseline | task_type | citation_hit_rate | context_recall | expected_keyword_coverage | lexical_answer_coverage | tool_selection_accuracy | tool_execution_success_rate | evidence_coverage | answer_correctness_proxy | faithfulness_proxy | grounding_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_only | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| llm_only | document_qa | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.022 | 0.022 | 0.000 |
| llm_only | policy_recommendation | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| llm_only | timeseries_query | 0.000 | 0.000 | 0.036 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_keyword | anomaly_diagnosis | 0.000 | 0.000 | 0.042 | 0.060 | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 |
| rag_keyword | document_qa | 0.700 | 0.700 | 0.630 | 0.339 | 0.000 | 0.000 | 0.967 | 0.244 | 0.244 | 0.000 |
| rag_keyword | policy_recommendation | 1.000 | 1.000 | 0.100 | 0.038 | 0.000 | 0.000 | 0.429 | 0.131 | 0.131 | 0.000 |
| rag_keyword | timeseries_query | 0.000 | 0.000 | 0.064 | 0.031 | 0.000 | 0.000 | 0.429 | 0.083 | 0.083 | 0.000 |
| rag_keyword_grounded | anomaly_diagnosis | 0.000 | 0.000 | 0.158 | 0.079 | 0.000 | 0.000 | 0.500 | 0.139 | 0.139 | 0.000 |
| rag_keyword_grounded | document_qa | 0.700 | 0.700 | 0.674 | 0.346 | 0.000 | 0.000 | 0.967 | 0.622 | 0.611 | 0.967 |
| rag_keyword_grounded | policy_recommendation | 1.000 | 1.000 | 0.264 | 0.062 | 0.000 | 0.000 | 0.429 | 0.298 | 0.298 | 1.000 |
| rag_keyword_grounded | timeseries_query | 0.000 | 0.000 | 0.071 | 0.039 | 0.000 | 0.000 | 0.429 | 0.179 | 0.179 | 0.000 |
| rag_dense | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.052 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| rag_dense | document_qa | 0.600 | 0.600 | 0.493 | 0.278 | 0.000 | 0.000 | 1.000 | 0.222 | 0.222 | 0.000 |
| rag_dense | policy_recommendation | 0.000 | 0.000 | 0.000 | 0.030 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| rag_dense | timeseries_query | 0.000 | 0.000 | 0.093 | 0.040 | 0.000 | 0.000 | 1.000 | 0.107 | 0.107 | 0.000 |
| rag_dense_grounded | anomaly_diagnosis | 0.000 | 0.000 | 0.225 | 0.101 | 0.000 | 0.000 | 1.000 | 0.250 | 0.250 | 0.000 |
| rag_dense_grounded | document_qa | 0.600 | 0.600 | 0.584 | 0.331 | 0.000 | 0.000 | 1.000 | 0.622 | 0.611 | 1.000 |
| rag_dense_grounded | policy_recommendation | 0.000 | 0.000 | 0.312 | 0.071 | 0.000 | 0.000 | 1.000 | 0.345 | 0.345 | 1.000 |
| rag_dense_grounded | timeseries_query | 0.000 | 0.000 | 0.350 | 0.134 | 0.000 | 0.000 | 1.000 | 0.464 | 0.464 | 0.000 |
| rag_hybrid | anomaly_diagnosis | 0.000 | 0.000 | 0.042 | 0.060 | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 |
| rag_hybrid | document_qa | 0.767 | 0.767 | 0.614 | 0.353 | 0.000 | 0.000 | 0.967 | 0.260 | 0.260 | 0.000 |
| rag_hybrid | policy_recommendation | 1.000 | 1.000 | 0.129 | 0.074 | 0.000 | 0.000 | 0.429 | 0.167 | 0.167 | 0.000 |
| rag_hybrid | timeseries_query | 0.000 | 0.000 | 0.064 | 0.031 | 0.000 | 0.000 | 0.429 | 0.083 | 0.083 | 0.000 |
| hybrid_rrf | anomaly_diagnosis | 0.000 | 0.000 | 0.042 | 0.082 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| hybrid_rrf | document_qa | 0.800 | 0.800 | 0.638 | 0.355 | 0.000 | 0.000 | 1.000 | 0.267 | 0.267 | 0.000 |
| hybrid_rrf | policy_recommendation | 1.000 | 1.000 | 0.129 | 0.104 | 0.000 | 0.000 | 1.000 | 0.167 | 0.167 | 0.000 |
| hybrid_rrf | timeseries_query | 0.000 | 0.000 | 0.129 | 0.058 | 0.000 | 0.000 | 1.000 | 0.155 | 0.155 | 0.000 |
| rag_hybrid_rerank | anomaly_diagnosis | 0.000 | 0.000 | 0.042 | 0.060 | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 |
| rag_hybrid_rerank | document_qa | 0.633 | 0.633 | 0.577 | 0.344 | 0.000 | 0.000 | 0.967 | 0.244 | 0.244 | 0.000 |
| rag_hybrid_rerank | policy_recommendation | 1.000 | 1.000 | 0.129 | 0.074 | 0.000 | 0.000 | 0.429 | 0.167 | 0.167 | 0.000 |
| rag_hybrid_rerank | timeseries_query | 0.000 | 0.000 | 0.064 | 0.031 | 0.000 | 0.000 | 0.429 | 0.083 | 0.083 | 0.000 |
| rag_rewrite | anomaly_diagnosis | 0.000 | 0.000 | 0.042 | 0.104 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| rag_rewrite | document_qa | 0.567 | 0.567 | 0.502 | 0.312 | 0.000 | 0.000 | 1.000 | 0.211 | 0.211 | 0.000 |
| rag_rewrite | policy_recommendation | 0.500 | 0.500 | 0.036 | 0.036 | 0.000 | 0.000 | 1.000 | 0.048 | 0.048 | 0.000 |
| rag_rewrite | timeseries_query | 0.000 | 0.000 | 0.100 | 0.040 | 0.000 | 0.000 | 1.000 | 0.119 | 0.119 | 0.000 |
| rewrite_llm | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.009 | 0.000 | 0.000 | 0.333 | 0.000 | 0.000 | 0.000 |
| rewrite_llm | document_qa | 0.600 | 0.600 | 0.519 | 0.316 | 0.000 | 0.000 | 0.800 | 0.216 | 0.216 | 0.000 |
| rewrite_llm | policy_recommendation | 0.500 | 0.500 | 0.136 | 0.088 | 0.000 | 0.000 | 0.857 | 0.131 | 0.131 | 0.000 |
| rewrite_llm | timeseries_query | 0.000 | 0.000 | 0.086 | 0.030 | 0.000 | 0.000 | 0.571 | 0.107 | 0.107 | 0.000 |
| rag_rewrite_grounded | anomaly_diagnosis | 0.000 | 0.000 | 0.225 | 0.121 | 0.000 | 0.000 | 1.000 | 0.250 | 0.250 | 0.000 |
| rag_rewrite_grounded | document_qa | 0.567 | 0.567 | 0.576 | 0.321 | 0.000 | 0.000 | 1.000 | 0.611 | 0.600 | 1.000 |
| rag_rewrite_grounded | policy_recommendation | 0.500 | 0.500 | 0.312 | 0.083 | 0.000 | 0.000 | 1.000 | 0.345 | 0.345 | 1.000 |
| rag_rewrite_grounded | timeseries_query | 0.000 | 0.000 | 0.264 | 0.109 | 0.000 | 0.000 | 1.000 | 0.357 | 0.357 | 0.000 |
| rag_hyde | anomaly_diagnosis | 0.000 | 0.000 | 0.042 | 0.117 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| rag_hyde | document_qa | 0.500 | 0.500 | 0.460 | 0.272 | 0.000 | 0.000 | 1.000 | 0.222 | 0.222 | 0.000 |
| rag_hyde | policy_recommendation | 0.000 | 0.000 | 0.000 | 0.022 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| rag_hyde | timeseries_query | 0.000 | 0.000 | 0.157 | 0.047 | 0.000 | 0.000 | 1.000 | 0.155 | 0.155 | 0.000 |
| rag_hyde_rerank | anomaly_diagnosis | 0.000 | 0.000 | 0.042 | 0.094 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| rag_hyde_rerank | document_qa | 0.533 | 0.533 | 0.450 | 0.284 | 0.000 | 0.000 | 1.000 | 0.189 | 0.189 | 0.000 |
| rag_hyde_rerank | policy_recommendation | 0.000 | 0.000 | 0.029 | 0.042 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| rag_hyde_rerank | timeseries_query | 0.000 | 0.000 | 0.064 | 0.030 | 0.000 | 0.000 | 1.000 | 0.107 | 0.107 | 0.000 |
| rag | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.052 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| rag | document_qa | 0.600 | 0.600 | 0.493 | 0.278 | 0.000 | 0.000 | 1.000 | 0.222 | 0.222 | 0.000 |
| rag | policy_recommendation | 0.000 | 0.000 | 0.000 | 0.030 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| rag | timeseries_query | 0.000 | 0.000 | 0.093 | 0.040 | 0.000 | 0.000 | 1.000 | 0.107 | 0.107 | 0.000 |
| rag_tool_agent | anomaly_diagnosis | 0.000 | 0.000 | 0.592 | 0.195 | 0.833 | 1.000 | 1.000 | 0.556 | 0.556 | 0.000 |
| rag_tool_agent | document_qa | 0.600 | 0.600 | 0.593 | 0.334 | 0.000 | 0.000 | 1.000 | 0.667 | 0.644 | 1.000 |
| rag_tool_agent | policy_recommendation | 0.000 | 0.000 | 0.724 | 0.267 | 0.714 | 1.000 | 1.000 | 0.774 | 0.774 | 0.000 |
| rag_tool_agent | timeseries_query | 0.000 | 0.000 | 0.821 | 0.386 | 1.000 | 1.000 | 1.000 | 0.917 | 0.917 | 0.000 |
| langgraph_tool_agent | anomaly_diagnosis | 0.000 | 0.000 | 0.633 | 0.204 | 1.000 | 1.000 | 1.000 | 0.611 | 0.611 | 0.000 |
| langgraph_tool_agent | document_qa | 0.600 | 0.600 | 0.593 | 0.334 | 0.000 | 0.000 | 1.000 | 0.667 | 0.644 | 1.000 |
| langgraph_tool_agent | policy_recommendation | 0.000 | 0.000 | 0.843 | 0.295 | 1.000 | 1.000 | 1.000 | 0.893 | 0.893 | 0.000 |
| langgraph_tool_agent | timeseries_query | 0.000 | 0.000 | 0.821 | 0.386 | 1.000 | 1.000 | 1.000 | 0.917 | 0.917 | 0.000 |
| react_agent | anomaly_diagnosis | 0.000 | 0.000 | 0.592 | 0.195 | 0.833 | 1.000 | 1.000 | 0.556 | 0.556 | 0.000 |
| react_agent | document_qa | 0.600 | 0.600 | 0.593 | 0.334 | 0.000 | 0.000 | 1.000 | 0.667 | 0.644 | 1.000 |
| react_agent | policy_recommendation | 0.000 | 0.000 | 0.760 | 0.277 | 0.857 | 1.000 | 1.000 | 0.845 | 0.845 | 0.000 |
| react_agent | timeseries_query | 0.000 | 0.000 | 0.821 | 0.386 | 1.000 | 1.000 | 1.000 | 0.917 | 0.917 | 0.000 |

## Human Calibration

人工校准集用于核对 deterministic proxy 和 optional LLM judge 的可信度；不会把 deterministic proxy 或 LLM judge 当作人工评审。

| sample_count | labeled_count | pending_count | mean_correctness | mean_faithfulness | safety_pass_rate | status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20 | 0 | 20 | null | null | null | pending_human_review |

## Safety Audit 对抗鲁棒性测试

该测试使用人工构造的 unsafe answer variant 检查确定性 Safety Audit 对生产遥测误述、LLM 直接控制和未验证动作表述的召回。

- sample_count = 29
- overall_hit_rate = 0.586

| category | sample_count | hit_count | hit_rate |
| --- | ---: | ---: | ---: |
| indirect | 6 | 2 | 0.333 |
| jailbreak | 6 | 4 | 0.667 |
| mixed | 5 | 3 | 0.600 |
| paraphrase | 8 | 8 | 1.000 |
| translation | 4 | 0 | 0.000 |

主要漏报样例：`adv_translation_01`, `adv_translation_02`, `adv_indirect_01`, `adv_indirect_02`, `adv_jailbreak_01`, `adv_mix_03`, `adv_indirect_03`, `adv_jailbreak_03`, `adv_translation_03`, `adv_translation_04`

## DROPT Policy Benchmark

该基准只评测 policy_recommendation 样例上的策略后端推理，不把它混入文档问答 baseline。

| sample_count | success_count | fallback_count | avg_latency_ms | avg_action_dim | avg_abs_action |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 7 | 0 | 9.557 | 6.000 | 0.951 |

## 当前结论

- `llm_only` 不使用检索证据或工具结果，作为最低可复现基线。
- `rag_dense` 使用真实 sentence-transformers embedding + FAISS 本地向量索引；该运行可用于面试中说明真实语义检索 baseline，但仍需结合 hybrid/rerank 指标判断中文 HVAC 场景效果。
- `rag_keyword_grounded` / `rag_dense_grounded` / `rag_rewrite_grounded` 把 extractive vs grounded generation 做成成对对比；当前 `grounding_rate` 最高的是 `rag_dense_grounded`=1.000。
- `rag_keyword` 与 `rag_hybrid` 用于比较轻量检索方案；`rag_hybrid` 在 citation/context 指标上优于 `rag_keyword`，说明 BM25-style 长度归一化在当前压力样例中有效。
- `rag_hybrid_rerank` 已纳入对比表；当前指标与 `rag_hybrid` 持平，说明轻量重排接口已具备，但还需要更强重排策略或更多重排压力样例。
- Query Rewrite / HyDE 已作为 deterministic query expansion baseline 纳入对比；当前 context_recall 最高的是 `rag_rewrite`，可用于评估 raw query、rewrite 和 template HyDE 在 HVAC/BEAR 领域检索中的收益，再决定是否替换为 DeepSeek/Ollama HyDE generator。
- `rag_tool_agent` 在当前确定性路由样例上体现工具选择、工具执行和证据覆盖优势。
- `langgraph_tool_agent` 已纳入对比；其指标与 deterministic baseline 的差异需要结合 workflow trace 进一步检查路由和工具节点行为。
- `react_agent` baseline 用来对比 workflow vs multi-step agent：在需要先收集时序上下文再给策略建议的样例上，可以显式展示多步 trace。
- `DROPT` / Guided-DiffFNO checkpoint 作为可选策略后端已接通：checkpoint 可加载、20 维 BEAR state 可推理，缺失或不完整时会明确回退并记录原因。
- `scripts/run_intent_eval.py` 单独评测 intent routing accuracy；默认 rule-based classifier 在当前 100 条样例上 accuracy 为 0.640，并输出 `data/eval/intent_routing_comparison.json` 作为 keyword vs LLM routing 对比入口。
