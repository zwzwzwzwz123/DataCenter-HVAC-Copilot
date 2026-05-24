# Compound Task LLM Planner Evaluation

Generated at: `2026-05-24T14:12:16.790970+00:00`

Eval dataset: `data/eval/compound_task_eval.jsonl`  
Record count: `100`

Planner: `LLMRoutePlanner` via DeepSeek `deepseek-v4-flash`  
Fallback count: `0`

Answer generation and policy backend were kept deterministic/rule-based so this run isolates route planning behavior.

## Planner Metrics

| Metric | Value |
|---|---:|
| planned_step_accuracy | 0.780 |
| planned_step_order_accuracy | 0.780 |
| required_step_recall | 0.937 |
| policy_final_step_rate | 1.000 |

## Supporting Metrics

| Metric | Value |
|---|---:|
| tool_selection_accuracy | 0.340 |
| tool_execution_success_rate | 1.000 |
| evidence_coverage | 1.000 |
| expected_keyword_coverage | 0.806 |
| grounding_rate | 0.657 |

## Interpretation

The LLM planner substantially improves compound-step planning on the 100-record constrained compound eval set. The policy-final constraint is fully preserved. Remaining errors are mostly tool-level and exact-step mismatches, not final-policy ordering failures.
