# Tier 1 Optimization Report

> Date: 2026-05-22
>
> Scope: Tier 1 A/B/C/D from `optimization_roadmap.md`, refreshed after `tier1_progress_review.md`.

## Executive Summary

Tier 1 is now interview-ready at the engineering-evidence level: the code paths exist, the baseline runner measures them, and `docs/experiment_report.md` has been regenerated from the current implementation.

The main remaining caveats are explicit rather than hidden: Human Calibration is still pending, Safety Audit translation attacks still miss, and the changes have not yet been split into git commits.

## A. Adversarial Safety Audit

Implemented a JSONL adversarial dataset and deterministic evaluator for answer safety audit robustness.

- Dataset: `data/eval/safety_adversarial.jsonl`
- Evaluator: `src/evaluation/safety_adversarial.py`
- Report section: `Safety Audit 对抗鲁棒性测试`

Current result:

| metric | value |
| --- | ---: |
| sample_count | 29 |
| overall_hit_rate | 0.586 |
| paraphrase_hit_rate | 1.000 |
| translation_hit_rate | 0.000 |

Interview story: the audit is useful because it exposes a real limitation, not because it claims perfect safety. Chinese paraphrases are covered; English translation attacks bypass the deterministic Chinese phrase dictionary.

## B. Grounded RAG

Implemented grounded generation as a separate pipeline and added paired extractive-vs-grounded baselines.

Paired baselines:

- `rag_keyword` vs `rag_keyword_grounded`
- `rag_dense` vs `rag_dense_grounded`
- `rag_rewrite` vs `rag_rewrite_grounded`

The latest report uses real dense retrieval:

- dense_provider: `sentence-transformers`
- dense_backend: `faiss`
- dense_model: `BAAI/bge-small-zh-v1.5`

Current result:

| baseline | citation_hit_rate | grounding_rate |
| --- | ---: | ---: |
| rag_keyword | 0.554 | 0.000 |
| rag_keyword_grounded | 0.554 | 0.708 |
| rag_dense | 0.692 | 0.000 |
| rag_dense_grounded | 0.692 | 1.000 |
| rag_rewrite | 0.646 | 0.000 |
| rag_rewrite_grounded | 0.646 | 1.000 |

Interview story: retrieval quality and generation grounding are measured separately. BGE+FAISS improves citation recall, while grounded variants make answer citations explicit and measurable.

## C. DROPT / Guided-DiffFNO Policy Backend

The DROPT adapter is not a stub. It loads the local `models/dropt/policy_best_fno_guided.pth` checkpoint and runs a Guided-DiffFNO policy on an explicit 20-dimensional BEAR state vector.

Added an independent policy benchmark section so this can be discussed as measured integration, not only code integration.

Current result:

| metric | value |
| --- | ---: |
| policy_sample_count | 28 |
| checkpoint_success_count | 28 |
| fallback_count | 0 |
| avg_latency_ms | 6.555 |
| avg_action_dim | 6.000 |
| avg_abs_action | 0.951 |

Interview story: LLM does not generate control actions. It routes, gathers evidence, and explains the result of a policy tool. The policy action itself comes from a checkpoint-backed adapter or an explicit fallback.

## D. Deterministic ReAct Multi-Step Baseline

Added a deterministic multi-step planner and 8 multi-hop policy evaluation records.

- Agent: `src/agent/react_agent.py`
- Multi-hop records: `multihop_001` through `multihop_008`
- Total eval set size: 108 records

Current policy subset result:

| baseline | tool_selection_accuracy | answer_correctness_proxy |
| --- | ---: | ---: |
| langgraph_tool_agent | 0.714 | 0.521 |
| react_agent | 0.893 | 0.625 |

Interview story: this is deliberately framed as a deterministic multi-step baseline, not a full LLM-driven ReAct agent. Its value is proving that multi-hop policy questions need an evidence-gathering step before the policy step.

## Verification

Latest targeted verification:

```text
40 passed
```

Latest evaluation command:

```bash
python scripts/run_eval.py --dense-provider sentence-transformers --dense-backend faiss --dense-model BAAI/bge-small-zh-v1.5
```

Generated artifacts:

- `data/eval/baseline_comparison.json`
- `docs/experiment_report.md`
- `data/eval/human_review_sample.jsonl`

## Remaining Work

- Split the current working tree into meaningful git commits.
- Complete 24 human calibration labels in `data/eval/human_review_annotations.jsonl`.
- Decide whether to improve Safety Audit translation coverage or keep it as a known limitation story.
- Add CI if this repo will be shown as a GitHub project rather than only a local interview demo.
