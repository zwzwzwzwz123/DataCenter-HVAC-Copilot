# 实验报告

## 数据与边界

当前评测集包含 30 条样例，覆盖文档问答、时序查询、异常诊断和策略建议。轨迹数据来自 BEAR 仿真轨迹、BEAR 样例 CSV 或 mock fallback，不能表述为真实数据中心生产遥测。
其中 30 条样例包含人工维护的 expected_keywords，用于计算中文回答要点覆盖率。

## 运行配置

- dense_provider: `sentence-transformers`
- dense_backend: `faiss`
- dense_model: `BAAI/bge-small-zh-v1.5`
- cross_encoder_model: `BAAI/bge-reranker-base`

## Baseline 对比

| baseline | citation_hit_rate | context_recall | retrieval_recall@1 | retrieval_recall@3 | retrieval_recall@5 | retrieval_recall@10 | retrieval_mrr@10 | retrieval_ndcg@10 | expected_keyword_coverage | lexical_answer_coverage | tool_selection_accuracy | tool_execution_success_rate | evidence_coverage | answer_correctness_proxy | faithfulness_proxy | hallucination_proxy_rate | grounding_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_only | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_keyword | 0.400 | 0.400 | 0.200 | 0.300 | 0.400 | 0.400 | 0.261 | 0.295 | 0.691 | 0.187 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_keyword_grounded | 0.400 | 0.400 | 0.200 | 0.300 | 0.400 | 0.400 | 0.261 | 0.295 | 0.978 | 0.202 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| rag_dense | 0.867 | 0.867 | 0.367 | 0.567 | 0.733 | 0.867 | 0.505 | 0.592 | 0.864 | 0.325 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_dense_grounded | 0.867 | 0.867 | 0.367 | 0.567 | 0.733 | 0.867 | 0.505 | 0.592 | 0.983 | 0.251 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| rag_hybrid | 0.400 | 0.400 | 0.233 | 0.267 | 0.333 | 0.400 | 0.276 | 0.305 | 0.754 | 0.362 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hybrid_rrf | 0.733 | 0.733 | 0.167 | 0.500 | 0.633 | 0.733 | 0.378 | 0.466 | 0.843 | 0.343 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_hybrid_rerank | 0.400 | 0.400 | 0.267 | 0.333 | 0.367 | 0.400 | 0.307 | 0.329 | 0.754 | 0.362 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_rewrite | 0.300 | 0.300 | 0.100 | 0.267 | 0.300 | 0.300 | 0.184 | 0.214 | 0.508 | 0.226 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rewrite_llm | 0.867 | 0.867 | 0.533 | 0.700 | 0.833 | 0.867 | 0.638 | 0.693 | 0.904 | 0.366 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_rewrite_grounded | 0.300 | 0.300 | 0.100 | 0.267 | 0.300 | 0.300 | 0.184 | 0.214 | 0.978 | 0.189 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| rag_hyde | 0.267 | 0.267 | 0.100 | 0.200 | 0.267 | 0.267 | 0.159 | 0.186 | 0.647 | 0.272 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_hyde_rerank | 0.267 | 0.267 | 0.100 | 0.200 | 0.267 | 0.267 | 0.159 | 0.186 | 0.647 | 0.272 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag | 0.867 | 0.867 | 0.367 | 0.567 | 0.733 | 0.867 | 0.505 | 0.592 | 0.864 | 0.325 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hybrid_rrf_cross_encoder | 0.900 | 0.900 | 0.600 | 0.833 | 0.900 | 0.900 | 0.717 | 0.763 | 0.916 | 0.362 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_tool_agent | 0.500 | 0.500 | 0.367 | 0.500 | 0.500 | 0.500 | 0.422 | 0.442 | 0.983 | 0.236 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| langgraph_tool_agent | 0.500 | 0.500 | 0.367 | 0.500 | 0.500 | 0.500 | 0.422 | 0.442 | 0.983 | 0.236 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| react_agent | 0.500 | 0.500 | 0.367 | 0.500 | 0.500 | 0.500 | 0.422 | 0.442 | 0.983 | 0.236 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |

## 按任务类型指标

| baseline | task_type | citation_hit_rate | context_recall | retrieval_recall@1 | retrieval_recall@3 | retrieval_recall@5 | retrieval_recall@10 | retrieval_mrr@10 | retrieval_ndcg@10 | expected_keyword_coverage | lexical_answer_coverage | tool_selection_accuracy | tool_execution_success_rate | evidence_coverage | answer_correctness_proxy | faithfulness_proxy | hallucination_proxy_rate | grounding_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_only | document_qa | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_keyword | document_qa | 0.400 | 0.400 | 0.200 | 0.300 | 0.400 | 0.400 | 0.261 | 0.295 | 0.691 | 0.187 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_keyword_grounded | document_qa | 0.400 | 0.400 | 0.200 | 0.300 | 0.400 | 0.400 | 0.261 | 0.295 | 0.978 | 0.202 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| rag_dense | document_qa | 0.867 | 0.867 | 0.367 | 0.567 | 0.733 | 0.867 | 0.505 | 0.592 | 0.864 | 0.325 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_dense_grounded | document_qa | 0.867 | 0.867 | 0.367 | 0.567 | 0.733 | 0.867 | 0.505 | 0.592 | 0.983 | 0.251 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| rag_hybrid | document_qa | 0.400 | 0.400 | 0.233 | 0.267 | 0.333 | 0.400 | 0.276 | 0.305 | 0.754 | 0.362 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hybrid_rrf | document_qa | 0.733 | 0.733 | 0.167 | 0.500 | 0.633 | 0.733 | 0.378 | 0.466 | 0.843 | 0.343 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_hybrid_rerank | document_qa | 0.400 | 0.400 | 0.267 | 0.333 | 0.367 | 0.400 | 0.307 | 0.329 | 0.754 | 0.362 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_rewrite | document_qa | 0.300 | 0.300 | 0.100 | 0.267 | 0.300 | 0.300 | 0.184 | 0.214 | 0.508 | 0.226 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rewrite_llm | document_qa | 0.867 | 0.867 | 0.533 | 0.700 | 0.833 | 0.867 | 0.638 | 0.693 | 0.904 | 0.366 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_rewrite_grounded | document_qa | 0.300 | 0.300 | 0.100 | 0.267 | 0.300 | 0.300 | 0.184 | 0.214 | 0.978 | 0.189 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| rag_hyde | document_qa | 0.267 | 0.267 | 0.100 | 0.200 | 0.267 | 0.267 | 0.159 | 0.186 | 0.647 | 0.272 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_hyde_rerank | document_qa | 0.267 | 0.267 | 0.100 | 0.200 | 0.267 | 0.267 | 0.159 | 0.186 | 0.647 | 0.272 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag | document_qa | 0.867 | 0.867 | 0.367 | 0.567 | 0.733 | 0.867 | 0.505 | 0.592 | 0.864 | 0.325 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hybrid_rrf_cross_encoder | document_qa | 0.900 | 0.900 | 0.600 | 0.833 | 0.900 | 0.900 | 0.717 | 0.763 | 0.916 | 0.362 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rag_tool_agent | document_qa | 0.500 | 0.500 | 0.367 | 0.500 | 0.500 | 0.500 | 0.422 | 0.442 | 0.983 | 0.236 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| langgraph_tool_agent | document_qa | 0.500 | 0.500 | 0.367 | 0.500 | 0.500 | 0.500 | 0.422 | 0.442 | 0.983 | 0.236 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| react_agent | document_qa | 0.500 | 0.500 | 0.367 | 0.500 | 0.500 | 0.500 | 0.422 | 0.442 | 0.983 | 0.236 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |

## Human Calibration

人工校准集用于核对 deterministic proxy 和 optional LLM judge 的可信度；不会把 deterministic proxy 或 LLM judge 当作人工评审。

| sample_count | labeled_count | pending_count | mean_correctness | mean_faithfulness | safety_pass_rate | status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 6 | 0 | 6 | null | null | null | pending_human_review |

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
| 0 | 0 | 0 | 0.000 | 0.000 | 0.000 |

## 当前结论

- `llm_only` 不使用检索证据或工具结果，作为最低可复现基线。
- `rag_dense` 使用真实 sentence-transformers embedding + FAISS 本地向量索引；该运行可用于面试中说明真实语义检索 baseline，但仍需结合 hybrid/rerank 指标判断中文 HVAC 场景效果。
- `rag_keyword_grounded` / `rag_dense_grounded` / `rag_rewrite_grounded` 把 extractive vs grounded generation 做成成对对比；当前 `grounding_rate` 最高的是 `rag_keyword_grounded`=1.000。
- `rag_keyword` 与 `rag_hybrid` 用于比较轻量检索方案；当前样例下两者指标持平，仍需更丰富的相似主题文档继续拉开差异。
- `hybrid_rrf_cross_encoder` 使用 BM25 + dense RRF 召回候选，再用 cross-encoder 对 query-document pair 做二阶段精排；当前相对 `hybrid_rrf` 是提升，需要结合 retrieval latency 判断排序质量与推理成本。
- Query Rewrite / HyDE 已作为 deterministic query expansion baseline 纳入对比；当前 context_recall 最高的是 `rag_rewrite`，可用于评估 raw query、rewrite 和 template HyDE 在 HVAC/BEAR 领域检索中的收益，再决定是否替换为 DeepSeek/Ollama HyDE generator。
- `rag_tool_agent` 在当前确定性路由样例上体现工具选择、工具执行和证据覆盖优势。
- `langgraph_tool_agent` 保留与 deterministic `rag_tool_agent` 一致的工具行为和指标，用于展示 StateGraph 编排、workflow trace 和可选 DeepSeek LLM route planner，而不是改变当前可复现评测口径。
- `react_agent` baseline 用来对比 workflow vs multi-step agent：在需要先收集时序上下文再给策略建议的样例上，可以显式展示多步 trace。
- `DROPT` / Guided-DiffFNO checkpoint 作为可选策略后端已接通：checkpoint 可加载、20 维 BEAR state 可推理，缺失或不完整时会明确回退并记录原因。
- `scripts/run_intent_eval.py` 单独评测 intent routing accuracy；默认 rule-based classifier 在当前 100 条样例上 accuracy 为 0.640，并输出 `data/eval/intent_routing_comparison.json` 作为 keyword vs LLM routing 对比入口。
