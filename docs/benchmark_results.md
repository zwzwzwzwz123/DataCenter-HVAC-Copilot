# Benchmark Results

本文件记录项目的评测结果。所有数字来自实际评测产物，可复现命令见 README「复现评测」一节。

数据来源：
- 检索 / Agent 回答：`data/eval/real_eval_true_model_full/baseline_comparison.json`
- Runtime / Guardrail：`data/eval/agent_runtime_comparison.json`

> **指标性质说明（重要）。** 下表分两类。**排序类指标**（citation、recall、MRR、nDCG、tool selection/success、required_step_recall 等）是确定性可复现的硬指标。**proxy 类指标**（`answer_correctness_proxy`、`faithfulness_proxy`、`grounding_rate`）是基于关键词/字符串匹配的近似，不是人工评审也不是 LLM judge，只用于相对比较，不应作为绝对回答质量。`grounding_rate` 在本轮 DeepSeek 自然语言回答下未稳定触发模板，故为 0.0，不代表实际未 grounded。

---

## 1. 检索排序（50 条真实文档 true-model）

配置：7 篇公开 PDF（340 chunks）、BGE-small-zh + FAISS embedding、BGE cross-encoder reranker。

| Mode | Citation/Context | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rag_dense` | 0.531 | 0.333 | 0.464 | 0.542 | 0.635 | 0.491 | 0.496 |
| `rag_hybrid` (BM25 lexical) | 0.688 | 0.583 | 0.750 | 0.766 | 0.766 | 0.760 | 0.722 |
| `hybrid_rrf` (BM25+dense RRF) | 0.719 | 0.490 | 0.714 | 0.776 | 0.786 | 0.701 | 0.694 |
| **`hybrid_rrf_cross_encoder`** | **0.781** | **0.615** | **0.839** | **0.854** | **0.854** | **0.797** | **0.791** |

要点：两阶段检索（RRF 召回 + cross-encoder 精排）在 citation、Recall@3/5/10、MRR、nDCG 上全面最优。cross-encoder 精排额外引入约 0.19s/query 延迟（`retrieval_average_latency_seconds`）。

---

## 2. Agent 回答质量（50 条真实文档 true-model）

| Mode | Tool Select | Tool Success | Correctness (proxy) | Faithfulness (proxy) | Hallucination (proxy) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rag_tool_agent` | 0.800 | 1.000 | 0.707 | 0.693 | 0.042 |
| `langgraph_tool_agent` | 0.800 | 0.950 | 0.658 | 0.638 | 0.042 |
| `react_agent` | 0.850 | 1.000 | 0.674 | 0.661 | 0.042 |
| `bounded_react_guard_agent` | 0.800 | 0.950 | 0.658 | 0.645 | 0.042 |
| `bounded_react_llm_batch_agent` | 0.750 | 0.950 | 0.672 | 0.659 | 0.042 |

要点：`react_agent` 工具选择最高（0.850）；`rag_tool_agent` 回答 proxy 最高。Correctness/Faithfulness 为 proxy，仅用于横向比较。`hallucination_proxy_rate` 是基于 `must_not_include` 的边界违规率（0.042），不是完整幻觉率。

---

## 3. Runtime / Guardrail（50 条场景集）

场景集含难度分层（easy 10 / medium 28 / hard 12）、干扰项和注入失败模式，通过确定性 runtime harness 评估。

**总体：**

| 指标 | 数值 |
| --- | ---: |
| required_step_recall | 0.990 |
| tool_sequence_accuracy | 0.935 |
| policy_obligation_success_rate | 0.941 |
| approval_block_success_rate | 1.000 |
| duplicate_guard_success_rate | 0.667 |
| recovery_success_rate | 0.833 |
| trace_completeness | 1.000 |
| tool_success_rate | 1.000 |
| average_tool_latency_seconds | 0.007 |

**按难度（保留 hard 失败信号）：**

| Difficulty | required_step_recall | tool_sequence_accuracy | duplicate_guard | recovery |
| --- | ---: | ---: | ---: | ---: |
| easy | 1.000 | 1.000 | 1.000 | 1.000 |
| medium | 1.000 | 1.000 | 1.000 | 1.000 |
| hard | 0.958 | 0.727 | 0.500 | 0.600 |

要点：approval block 与 trace 完整性满分；主要短板在 hard 难度的重复工具拦截（0.500）和 recovery（0.600），刻意保留以避免做成满分演示。

---

## 4. 参考指标（口径不同，勿与上表并列）

以下指标来自不同评测集或不同环境，仅供参考，不能与 50 条 true-model 主结果直接比较：

| 指标 | 数值 | 来源 / 口径 |
| --- | ---: | --- |
| 108 条合成集 `hybrid_rrf` Citation/Context | 0.708 | demo docs，非真实语义 embedding |
| rule-based intent accuracy | 0.640 | 100-sample artifact，与主评测集未对齐 |
| safety adversarial overall hit rate | 0.657 | 确定性边界检查，translation 类 0.000 为已知短板 |
| DROPT policy benchmark success | 28 / 28 | 研究型 policy adapter |

---

*最后核对：所有第 1–3 节数字已与对应 JSON 产物逐项核对一致。更新评测后需重新核对本文件。*
