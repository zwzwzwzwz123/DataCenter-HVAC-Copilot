# 实验报告

## 数据与边界

当前评测集包含 108 条样例，覆盖文档问答、时序查询、异常诊断和策略建议。轨迹数据来自 BEAR 仿真轨迹、BEAR 样例 CSV 或 mock fallback，不能表述为真实数据中心生产遥测。
其中 108 条样例包含人工维护的 expected_keywords，用于计算中文回答要点覆盖率。

## 运行配置

- dense_provider: `deterministic`
- dense_backend: `memory`
- dense_model: `default`

## Baseline 对比

| baseline | citation_hit_rate | context_recall | expected_keyword_coverage | lexical_answer_coverage | tool_selection_accuracy | tool_execution_success_rate | evidence_coverage | answer_correctness_proxy | faithfulness_proxy | grounding_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_only | 0.000 | 0.000 | 0.006 | 0.000 | 0.000 | 0.000 | 0.000 | 0.019 | 0.019 | 0.000 |
| rag_keyword | 0.492 | 0.492 | 0.258 | 0.129 | 0.000 | 0.000 | 0.620 | 0.333 | 0.333 | 0.000 |
| rag_keyword_grounded | 0.492 | 0.492 | 0.353 | 0.165 | 0.000 | 0.000 | 0.620 | 0.352 | 0.314 | 0.708 |
| rag_dense | 0.508 | 0.508 | 0.261 | 0.122 | 0.000 | 0.000 | 1.000 | 0.273 | 0.268 | 0.000 |
| rag_dense_grounded | 0.508 | 0.508 | 0.409 | 0.164 | 0.000 | 0.000 | 1.000 | 0.377 | 0.322 | 1.000 |
| rag_hybrid | 0.523 | 0.523 | 0.295 | 0.138 | 0.000 | 0.000 | 0.620 | 0.344 | 0.328 | 0.000 |
| hybrid_rrf | 0.569 | 0.569 | 0.292 | 0.140 | 0.000 | 0.000 | 1.000 | 0.350 | 0.328 | 0.000 |
| rag_hybrid_rerank | 0.615 | 0.615 | 0.301 | 0.143 | 0.000 | 0.000 | 0.620 | 0.347 | 0.325 | 0.000 |
| rag_rewrite | 0.569 | 0.569 | 0.316 | 0.150 | 0.000 | 0.000 | 1.000 | 0.311 | 0.311 | 0.000 |
| rag_rewrite_grounded | 0.569 | 0.569 | 0.465 | 0.206 | 0.000 | 0.000 | 1.000 | 0.410 | 0.372 | 1.000 |
| rag_hyde | 0.246 | 0.246 | 0.156 | 0.114 | 0.000 | 0.000 | 1.000 | 0.158 | 0.142 | 0.000 |
| rag_hyde_rerank | 0.338 | 0.338 | 0.140 | 0.098 | 0.000 | 0.000 | 1.000 | 0.131 | 0.126 | 0.000 |
| rag | 0.523 | 0.523 | 0.295 | 0.138 | 0.000 | 0.000 | 0.620 | 0.344 | 0.328 | 0.000 |
| rag_tool_agent | 0.338 | 0.338 | 0.628 | 0.288 | 0.882 | 1.000 | 0.917 | 0.541 | 0.486 | 0.477 |
| langgraph_tool_agent | 0.338 | 0.338 | 0.628 | 0.288 | 0.882 | 1.000 | 0.917 | 0.541 | 0.486 | 0.477 |
| react_agent | 0.338 | 0.338 | 0.644 | 0.294 | 0.956 | 1.000 | 0.917 | 0.582 | 0.527 | 0.477 |

## 按任务类型指标

| baseline | task_type | citation_hit_rate | context_recall | expected_keyword_coverage | lexical_answer_coverage | tool_selection_accuracy | tool_execution_success_rate | evidence_coverage | answer_correctness_proxy | faithfulness_proxy | grounding_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_only | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| llm_only | document_qa | 0.000 | 0.000 | 0.017 | 0.000 | 0.000 | 0.000 | 0.000 | 0.028 | 0.028 | 0.000 |
| llm_only | policy_recommendation | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.021 | 0.021 | 0.000 |
| llm_only | timeseries_query | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_keyword | anomaly_diagnosis | 0.250 | 0.250 | 0.100 | 0.044 | 0.000 | 0.000 | 0.300 | 0.133 | 0.133 | 0.000 |
| rag_keyword | document_qa | 0.550 | 0.550 | 0.429 | 0.155 | 0.000 | 0.000 | 0.775 | 0.542 | 0.542 | 0.000 |
| rag_keyword | policy_recommendation | 0.538 | 0.538 | 0.202 | 0.185 | 0.000 | 0.000 | 0.679 | 0.250 | 0.250 | 0.000 |
| rag_keyword | timeseries_query | 0.000 | 0.000 | 0.150 | 0.081 | 0.000 | 0.000 | 0.550 | 0.000 | 0.000 | 0.000 |
| rag_keyword_grounded | anomaly_diagnosis | 0.250 | 0.250 | 0.117 | 0.044 | 0.000 | 0.000 | 0.300 | 0.100 | 0.100 | 0.500 |
| rag_keyword_grounded | document_qa | 0.550 | 0.550 | 0.537 | 0.217 | 0.000 | 0.000 | 0.775 | 0.569 | 0.514 | 0.775 |
| rag_keyword_grounded | policy_recommendation | 0.538 | 0.538 | 0.333 | 0.238 | 0.000 | 0.000 | 0.679 | 0.264 | 0.222 | 0.692 |
| rag_keyword_grounded | timeseries_query | 0.000 | 0.000 | 0.250 | 0.081 | 0.000 | 0.000 | 0.550 | 0.167 | 0.167 | 0.000 |
| rag_dense | anomaly_diagnosis | 0.583 | 0.583 | 0.167 | 0.089 | 0.000 | 0.000 | 1.000 | 0.150 | 0.150 | 0.000 |
| rag_dense | document_qa | 0.500 | 0.500 | 0.429 | 0.134 | 0.000 | 0.000 | 1.000 | 0.444 | 0.444 | 0.000 |
| rag_dense | policy_recommendation | 0.462 | 0.462 | 0.167 | 0.167 | 0.000 | 0.000 | 1.000 | 0.174 | 0.160 | 0.000 |
| rag_dense | timeseries_query | 0.000 | 0.000 | 0.150 | 0.070 | 0.000 | 0.000 | 1.000 | 0.111 | 0.111 | 0.000 |
| rag_dense_grounded | anomaly_diagnosis | 0.583 | 0.583 | 0.250 | 0.097 | 0.000 | 0.000 | 1.000 | 0.283 | 0.283 | 1.000 |
| rag_dense_grounded | document_qa | 0.500 | 0.500 | 0.588 | 0.215 | 0.000 | 0.000 | 1.000 | 0.542 | 0.458 | 1.000 |
| rag_dense_grounded | policy_recommendation | 0.462 | 0.462 | 0.333 | 0.202 | 0.000 | 0.000 | 1.000 | 0.264 | 0.208 | 1.000 |
| rag_dense_grounded | timeseries_query | 0.000 | 0.000 | 0.317 | 0.076 | 0.000 | 0.000 | 1.000 | 0.278 | 0.278 | 0.000 |
| rag_hybrid | anomaly_diagnosis | 0.333 | 0.333 | 0.117 | 0.069 | 0.000 | 0.000 | 0.300 | 0.133 | 0.133 | 0.000 |
| rag_hybrid | document_qa | 0.550 | 0.550 | 0.529 | 0.170 | 0.000 | 0.000 | 0.775 | 0.569 | 0.528 | 0.000 |
| rag_hybrid | policy_recommendation | 0.615 | 0.615 | 0.190 | 0.178 | 0.000 | 0.000 | 0.679 | 0.250 | 0.250 | 0.000 |
| rag_hybrid | timeseries_query | 0.000 | 0.000 | 0.150 | 0.087 | 0.000 | 0.000 | 0.550 | 0.000 | 0.000 | 0.000 |
| hybrid_rrf | anomaly_diagnosis | 0.500 | 0.500 | 0.200 | 0.103 | 0.000 | 0.000 | 1.000 | 0.233 | 0.233 | 0.000 |
| hybrid_rrf | document_qa | 0.575 | 0.575 | 0.471 | 0.150 | 0.000 | 0.000 | 1.000 | 0.528 | 0.486 | 0.000 |
| hybrid_rrf | policy_recommendation | 0.615 | 0.615 | 0.190 | 0.192 | 0.000 | 0.000 | 1.000 | 0.250 | 0.236 | 0.000 |
| hybrid_rrf | timeseries_query | 0.000 | 0.000 | 0.167 | 0.083 | 0.000 | 0.000 | 1.000 | 0.111 | 0.111 | 0.000 |
| rag_hybrid_rerank | anomaly_diagnosis | 0.417 | 0.417 | 0.100 | 0.063 | 0.000 | 0.000 | 0.300 | 0.083 | 0.083 | 0.000 |
| rag_hybrid_rerank | document_qa | 0.650 | 0.650 | 0.546 | 0.180 | 0.000 | 0.000 | 0.775 | 0.597 | 0.556 | 0.000 |
| rag_hybrid_rerank | policy_recommendation | 0.692 | 0.692 | 0.202 | 0.187 | 0.000 | 0.000 | 0.679 | 0.250 | 0.236 | 0.000 |
| rag_hybrid_rerank | timeseries_query | 0.000 | 0.000 | 0.150 | 0.087 | 0.000 | 0.000 | 0.550 | 0.000 | 0.000 | 0.000 |
| rag_rewrite | anomaly_diagnosis | 0.500 | 0.500 | 0.133 | 0.087 | 0.000 | 0.000 | 1.000 | 0.100 | 0.100 | 0.000 |
| rag_rewrite | document_qa | 0.625 | 0.625 | 0.412 | 0.150 | 0.000 | 0.000 | 1.000 | 0.431 | 0.431 | 0.000 |
| rag_rewrite | policy_recommendation | 0.462 | 0.462 | 0.262 | 0.149 | 0.000 | 0.000 | 1.000 | 0.264 | 0.264 | 0.000 |
| rag_rewrite | timeseries_query | 0.000 | 0.000 | 0.383 | 0.212 | 0.000 | 0.000 | 1.000 | 0.444 | 0.444 | 0.000 |
| rag_rewrite_grounded | anomaly_diagnosis | 0.500 | 0.500 | 0.250 | 0.095 | 0.000 | 0.000 | 1.000 | 0.283 | 0.283 | 1.000 |
| rag_rewrite_grounded | document_qa | 0.625 | 0.625 | 0.579 | 0.240 | 0.000 | 0.000 | 1.000 | 0.542 | 0.486 | 1.000 |
| rag_rewrite_grounded | policy_recommendation | 0.462 | 0.462 | 0.381 | 0.203 | 0.000 | 0.000 | 1.000 | 0.326 | 0.285 | 1.000 |
| rag_rewrite_grounded | timeseries_query | 0.000 | 0.000 | 0.567 | 0.249 | 0.000 | 0.000 | 1.000 | 0.444 | 0.444 | 0.000 |
| rag_hyde | anomaly_diagnosis | 0.000 | 0.000 | 0.083 | 0.046 | 0.000 | 0.000 | 1.000 | 0.133 | 0.133 | 0.000 |
| rag_hyde | document_qa | 0.300 | 0.300 | 0.196 | 0.113 | 0.000 | 0.000 | 1.000 | 0.167 | 0.125 | 0.000 |
| rag_hyde | policy_recommendation | 0.308 | 0.308 | 0.226 | 0.229 | 0.000 | 0.000 | 1.000 | 0.167 | 0.167 | 0.000 |
| rag_hyde | timeseries_query | 0.000 | 0.000 | 0.050 | 0.023 | 0.000 | 0.000 | 1.000 | 0.111 | 0.111 | 0.000 |
| rag_hyde_rerank | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.006 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| rag_hyde_rerank | document_qa | 0.425 | 0.425 | 0.229 | 0.111 | 0.000 | 0.000 | 1.000 | 0.153 | 0.139 | 0.000 |
| rag_hyde_rerank | policy_recommendation | 0.385 | 0.385 | 0.190 | 0.209 | 0.000 | 0.000 | 1.000 | 0.167 | 0.167 | 0.000 |
| rag_hyde_rerank | timeseries_query | 0.000 | 0.000 | 0.033 | 0.007 | 0.000 | 0.000 | 1.000 | 0.111 | 0.111 | 0.000 |
| rag | anomaly_diagnosis | 0.333 | 0.333 | 0.117 | 0.069 | 0.000 | 0.000 | 0.300 | 0.133 | 0.133 | 0.000 |
| rag | document_qa | 0.550 | 0.550 | 0.529 | 0.170 | 0.000 | 0.000 | 0.775 | 0.569 | 0.528 | 0.000 |
| rag | policy_recommendation | 0.615 | 0.615 | 0.190 | 0.178 | 0.000 | 0.000 | 0.679 | 0.250 | 0.250 | 0.000 |
| rag | timeseries_query | 0.000 | 0.000 | 0.150 | 0.087 | 0.000 | 0.000 | 0.550 | 0.000 | 0.000 | 0.000 |
| rag_tool_agent | anomaly_diagnosis | 0.000 | 0.000 | 0.533 | 0.242 | 1.000 | 1.000 | 1.000 | 0.367 | 0.367 | 0.000 |
| rag_tool_agent | document_qa | 0.550 | 0.550 | 0.646 | 0.248 | 0.000 | 0.000 | 0.775 | 0.597 | 0.486 | 0.775 |
| rag_tool_agent | policy_recommendation | 0.000 | 0.000 | 0.560 | 0.312 | 0.714 | 1.000 | 1.000 | 0.521 | 0.493 | 0.000 |
| rag_tool_agent | timeseries_query | 0.000 | 0.000 | 0.783 | 0.381 | 1.000 | 1.000 | 1.000 | 0.833 | 0.833 | 0.000 |
| langgraph_tool_agent | anomaly_diagnosis | 0.000 | 0.000 | 0.533 | 0.242 | 1.000 | 1.000 | 1.000 | 0.367 | 0.367 | 0.000 |
| langgraph_tool_agent | document_qa | 0.550 | 0.550 | 0.646 | 0.248 | 0.000 | 0.000 | 0.775 | 0.597 | 0.486 | 0.775 |
| langgraph_tool_agent | policy_recommendation | 0.000 | 0.000 | 0.560 | 0.312 | 0.714 | 1.000 | 1.000 | 0.521 | 0.493 | 0.000 |
| langgraph_tool_agent | timeseries_query | 0.000 | 0.000 | 0.783 | 0.381 | 1.000 | 1.000 | 1.000 | 0.833 | 0.833 | 0.000 |
| react_agent | anomaly_diagnosis | 0.000 | 0.000 | 0.533 | 0.242 | 1.000 | 1.000 | 1.000 | 0.367 | 0.367 | 0.000 |
| react_agent | document_qa | 0.550 | 0.550 | 0.646 | 0.248 | 0.000 | 0.000 | 0.775 | 0.597 | 0.486 | 0.775 |
| react_agent | policy_recommendation | 0.000 | 0.000 | 0.619 | 0.333 | 0.893 | 1.000 | 1.000 | 0.625 | 0.597 | 0.000 |
| react_agent | timeseries_query | 0.000 | 0.000 | 0.783 | 0.381 | 1.000 | 1.000 | 1.000 | 0.833 | 0.833 | 0.000 |

## Human Calibration

人工校准集用于核对 deterministic proxy 和 optional LLM judge 的可信度；不会把 deterministic proxy 或 LLM judge 当作人工评审。

| sample_count | labeled_count | pending_count | mean_correctness | mean_faithfulness | safety_pass_rate | status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 24 | 0 | 24 | null | null | null | pending_human_review |

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
| 28 | 28 | 0 | 6.282 | 6.000 | 0.951 |

## 当前结论

- `llm_only` 不使用检索证据或工具结果，作为最低可复现基线。
- `rag_dense` 使用 deterministic hash embedding 作为默认 dense retrieval baseline；真实 FAISS + sentence-transformers 作为可选增强，避免默认评测依赖模型下载或外部 API。
- `rag_keyword_grounded` / `rag_dense_grounded` / `rag_rewrite_grounded` 把 extractive vs grounded generation 做成成对对比；当前 `grounding_rate` 最高的是 `rag_dense_grounded`=1.000。
- `rag_keyword` 与 `rag_hybrid` 用于比较轻量检索方案；`rag_hybrid` 在 citation/context 指标上优于 `rag_keyword`，说明 BM25-style 长度归一化在当前压力样例中有效。
- `rag_hybrid_rerank` 在当前评测中进一步提升 citation/context 指标，可作为后续替换为 cross-encoder 或 LLM reranker 的接口基线。
- Query Rewrite / HyDE 已作为 deterministic query expansion baseline 纳入对比；当前 context_recall 最高的是 `rag_rewrite`，可用于评估 raw query、rewrite 和 template HyDE 在 HVAC/BEAR 领域检索中的收益，再决定是否替换为 DeepSeek/Ollama HyDE generator。
- `rag_tool_agent` 在当前确定性路由样例上体现工具选择、工具执行和证据覆盖优势。
- `langgraph_tool_agent` 保留与 deterministic `rag_tool_agent` 一致的工具行为和指标，用于展示 StateGraph 编排、workflow trace 和可选 DeepSeek LLM route planner，而不是改变当前可复现评测口径。
- `react_agent` baseline 用于对比 single-step workflow vs deterministic multi-step planner；新增 multi-hop policy 样例后，policy 子集 tool_selection_accuracy 从 `0.714` 提升到 `0.893`，answer_correctness_proxy 从 `0.521` 提升到 `0.625`。
- `DROPT` / Guided-DiffFNO checkpoint 作为可选策略后端已接通：checkpoint 可加载、20 维 BEAR state 可推理，缺失或不完整时会明确回退并记录原因。
- `scripts/run_intent_eval.py` 单独评测 intent routing accuracy；默认 rule-based classifier 在当前 100 条样例上 accuracy 为 0.640，并输出 `data/eval/intent_routing_comparison.json` 作为 keyword vs LLM routing 对比入口。
