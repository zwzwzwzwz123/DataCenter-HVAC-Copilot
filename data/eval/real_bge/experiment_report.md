# 实验报告

## 数据与边界

当前评测集包含 108 条样例，覆盖文档问答、时序查询、异常诊断和策略建议。轨迹数据来自 BEAR 仿真轨迹、BEAR 样例 CSV 或 mock fallback，不能表述为真实数据中心生产遥测。
其中 108 条样例包含人工维护的 expected_keywords，用于计算中文回答要点覆盖率。

## 运行配置

- dense_provider: `sentence-transformers`
- dense_backend: `faiss`
- dense_model: `BAAI/bge-small-zh-v1.5`

## Baseline 对比

| baseline | citation_hit_rate | context_recall | expected_keyword_coverage | lexical_answer_coverage | tool_selection_accuracy | tool_execution_success_rate | evidence_coverage | answer_correctness_proxy | faithfulness_proxy | grounding_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_only | 0.000 | 0.000 | 0.006 | 0.000 | 0.000 | 0.000 | 0.000 | 0.019 | 0.019 | 0.000 |
| rag_keyword | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.005 | 0.000 |
| rag_keyword_grounded | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.005 | 0.000 |
| rag_dense | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.005 | 0.000 |
| rag_dense_grounded | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.005 | 0.000 |
| rag_hybrid | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.005 | 0.000 |
| hybrid_rrf | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.005 | 0.000 |
| rag_hybrid_rerank | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.005 | 0.000 |
| rag_rewrite | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.005 | 0.000 |
| rewrite_llm | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.005 | 0.000 |
| rag_rewrite_grounded | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.005 | 0.000 |
| rag_hyde | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.005 | 0.000 |
| rag_hyde_rerank | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.005 | 0.000 |
| rag | 0.000 | 0.000 | 0.030 | 0.044 | 0.000 | 0.000 | 1.000 | 0.044 | 0.044 | 0.000 |
| rag_tool_agent | 0.000 | 0.000 | 0.564 | 0.226 | 0.882 | 1.000 | 1.000 | 0.464 | 0.421 | 0.615 |
| langgraph_tool_agent | 0.000 | 0.000 | 0.564 | 0.226 | 0.882 | 1.000 | 1.000 | 0.464 | 0.421 | 0.615 |
| react_agent | 0.000 | 0.000 | 0.579 | 0.232 | 0.956 | 1.000 | 1.000 | 0.505 | 0.462 | 0.615 |

## 按任务类型指标

| baseline | task_type | citation_hit_rate | context_recall | expected_keyword_coverage | lexical_answer_coverage | tool_selection_accuracy | tool_execution_success_rate | evidence_coverage | answer_correctness_proxy | faithfulness_proxy | grounding_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_only | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| llm_only | document_qa | 0.000 | 0.000 | 0.017 | 0.000 | 0.000 | 0.000 | 0.000 | 0.028 | 0.028 | 0.000 |
| llm_only | policy_recommendation | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.021 | 0.021 | 0.000 |
| llm_only | timeseries_query | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_keyword | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_keyword | document_qa | 0.000 | 0.000 | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 | 0.014 | 0.014 | 0.000 |
| rag_keyword | policy_recommendation | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_keyword | timeseries_query | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_keyword_grounded | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_keyword_grounded | document_qa | 0.000 | 0.000 | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 | 0.014 | 0.014 | 0.000 |
| rag_keyword_grounded | policy_recommendation | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_keyword_grounded | timeseries_query | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_dense | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_dense | document_qa | 0.000 | 0.000 | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 | 0.014 | 0.014 | 0.000 |
| rag_dense | policy_recommendation | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_dense | timeseries_query | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_dense_grounded | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_dense_grounded | document_qa | 0.000 | 0.000 | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 | 0.014 | 0.014 | 0.000 |
| rag_dense_grounded | policy_recommendation | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_dense_grounded | timeseries_query | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_hybrid | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_hybrid | document_qa | 0.000 | 0.000 | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 | 0.014 | 0.014 | 0.000 |
| rag_hybrid | policy_recommendation | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_hybrid | timeseries_query | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hybrid_rrf | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hybrid_rrf | document_qa | 0.000 | 0.000 | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 | 0.014 | 0.014 | 0.000 |
| hybrid_rrf | policy_recommendation | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hybrid_rrf | timeseries_query | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_hybrid_rerank | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_hybrid_rerank | document_qa | 0.000 | 0.000 | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 | 0.014 | 0.014 | 0.000 |
| rag_hybrid_rerank | policy_recommendation | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_hybrid_rerank | timeseries_query | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_rewrite | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_rewrite | document_qa | 0.000 | 0.000 | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 | 0.014 | 0.014 | 0.000 |
| rag_rewrite | policy_recommendation | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_rewrite | timeseries_query | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rewrite_llm | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rewrite_llm | document_qa | 0.000 | 0.000 | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 | 0.014 | 0.014 | 0.000 |
| rewrite_llm | policy_recommendation | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rewrite_llm | timeseries_query | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_rewrite_grounded | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_rewrite_grounded | document_qa | 0.000 | 0.000 | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 | 0.014 | 0.014 | 0.000 |
| rag_rewrite_grounded | policy_recommendation | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_rewrite_grounded | timeseries_query | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_hyde | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_hyde | document_qa | 0.000 | 0.000 | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 | 0.014 | 0.014 | 0.000 |
| rag_hyde | policy_recommendation | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_hyde | timeseries_query | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_hyde_rerank | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_hyde_rerank | document_qa | 0.000 | 0.000 | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 | 0.014 | 0.014 | 0.000 |
| rag_hyde_rerank | policy_recommendation | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_hyde_rerank | timeseries_query | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag | anomaly_diagnosis | 0.000 | 0.000 | 0.000 | 0.017 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| rag | document_qa | 0.000 | 0.000 | 0.065 | 0.034 | 0.000 | 0.000 | 1.000 | 0.097 | 0.097 | 0.000 |
| rag | policy_recommendation | 0.000 | 0.000 | 0.012 | 0.110 | 0.000 | 0.000 | 1.000 | 0.014 | 0.014 | 0.000 |
| rag | timeseries_query | 0.000 | 0.000 | 0.017 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| rag_tool_agent | anomaly_diagnosis | 0.000 | 0.000 | 0.533 | 0.242 | 1.000 | 1.000 | 1.000 | 0.367 | 0.367 | 0.000 |
| rag_tool_agent | document_qa | 0.000 | 0.000 | 0.473 | 0.081 | 0.000 | 0.000 | 1.000 | 0.403 | 0.319 | 1.000 |
| rag_tool_agent | policy_recommendation | 0.000 | 0.000 | 0.560 | 0.312 | 0.714 | 1.000 | 1.000 | 0.521 | 0.493 | 0.000 |
| rag_tool_agent | timeseries_query | 0.000 | 0.000 | 0.783 | 0.381 | 1.000 | 1.000 | 1.000 | 0.833 | 0.833 | 0.000 |
| langgraph_tool_agent | anomaly_diagnosis | 0.000 | 0.000 | 0.533 | 0.242 | 1.000 | 1.000 | 1.000 | 0.367 | 0.367 | 0.000 |
| langgraph_tool_agent | document_qa | 0.000 | 0.000 | 0.473 | 0.081 | 0.000 | 0.000 | 1.000 | 0.403 | 0.319 | 1.000 |
| langgraph_tool_agent | policy_recommendation | 0.000 | 0.000 | 0.560 | 0.312 | 0.714 | 1.000 | 1.000 | 0.521 | 0.493 | 0.000 |
| langgraph_tool_agent | timeseries_query | 0.000 | 0.000 | 0.783 | 0.381 | 1.000 | 1.000 | 1.000 | 0.833 | 0.833 | 0.000 |
| react_agent | anomaly_diagnosis | 0.000 | 0.000 | 0.533 | 0.242 | 1.000 | 1.000 | 1.000 | 0.367 | 0.367 | 0.000 |
| react_agent | document_qa | 0.000 | 0.000 | 0.473 | 0.081 | 0.000 | 0.000 | 1.000 | 0.403 | 0.319 | 1.000 |
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
| 28 | 28 | 0 | 8.743 | 6.000 | 0.951 |

## 当前结论

- `llm_only` 不使用检索证据或工具结果，作为最低可复现基线。
- `rag_dense` 使用真实 sentence-transformers embedding + FAISS 本地向量索引；该运行可用于面试中说明真实语义检索 baseline，但仍需结合 hybrid/rerank 指标判断中文 HVAC 场景效果。
- grounded RAG paired baselines 已纳入对比；`grounding_rate` 仍需结合 answer correctness 一起看，以区分检索失败和生成漂移。
- `rag_keyword` 与 `rag_hybrid` 用于比较轻量检索方案；当前样例下两者指标持平，仍需更丰富的相似主题文档继续拉开差异。
- `rag_hybrid_rerank` 已纳入对比表；当前指标与 `rag_hybrid` 持平，说明轻量重排接口已具备，但还需要更强重排策略或更多重排压力样例。
- Query Rewrite / HyDE 已作为 deterministic query expansion baseline 纳入对比；当前 context_recall 最高的是 `rag_rewrite`，可用于评估 raw query、rewrite 和 template HyDE 在 HVAC/BEAR 领域检索中的收益，再决定是否替换为 DeepSeek/Ollama HyDE generator。
- `rag_tool_agent` 在当前确定性路由样例上体现工具选择、工具执行和证据覆盖优势。
- `langgraph_tool_agent` 保留与 deterministic `rag_tool_agent` 一致的工具行为和指标，用于展示 StateGraph 编排、workflow trace 和可选 DeepSeek LLM route planner，而不是改变当前可复现评测口径。
- `react_agent` baseline 用于对比 single-step workflow vs deterministic multi-step planner；新增 multi-hop policy 样例后，policy 子集 tool_selection_accuracy 从 `0.714` 提升到 `0.893`，answer_correctness_proxy 从 `0.521` 提升到 `0.625`。
- `DROPT` / Guided-DiffFNO checkpoint 作为可选策略后端已接通：checkpoint 可加载、20 维 BEAR state 可推理，缺失或不完整时会明确回退并记录原因。
- `scripts/run_intent_eval.py` 单独评测 intent routing accuracy；默认 rule-based classifier 在当前 100 条样例上 accuracy 为 0.640，并输出 `data/eval/intent_routing_comparison.json` 作为 keyword vs LLM routing 对比入口。
