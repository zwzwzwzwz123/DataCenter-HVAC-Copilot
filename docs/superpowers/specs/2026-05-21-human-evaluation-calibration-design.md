# Human Evaluation Calibration Design

## Goal

Add a lightweight human evaluation calibration workflow that makes the project's answer-quality claims more credible without pretending that deterministic proxy metrics or optional LLM judge scores are human review.

## Scope

This feature creates review-ready samples and reads reviewer-provided annotations. It does not generate artificial human labels. Until a human fills the annotation file, reports must show the calibration status as pending.

The first version targets about 24 examples sampled from the existing 100-record evaluation set and `data/eval/baseline_predictions.jsonl`. The sample should cover all four task types: `document_qa`, `timeseries_query`, `anomaly_diagnosis`, and `policy_recommendation`.

## Outputs

- `data/eval/human_review_sample.jsonl`: review packet generated from existing predictions. Each record contains the question, task type, gold answer, model answer, citations, tool results, answer audit, expected keywords, and available proxy metrics or evidence flags.
- `data/eval/human_review_annotations.jsonl`: annotation template with one record per sampled item. Reviewer fields start as `null` or empty values.
- `docs/human_evaluation_guide.md`: Chinese marking guide explaining correctness, faithfulness, safety boundary, and reviewer notes.
- `docs/experiment_report.md`: generated report gains a Human Calibration section. Before manual labels exist, it reports `labeled_count = 0` and `status = pending_human_review`.

## Annotation Schema

Each annotation record uses the evaluation id as the stable key:

```json
{
  "id": "ts_query_001",
  "task_type": "timeseries_query",
  "correctness_score": null,
  "faithfulness_score": null,
  "safety_boundary": null,
  "reviewer_notes": ""
}
```

Allowed values:

- `correctness_score`: `0`, `0.5`, or `1`.
- `faithfulness_score`: `0`, `0.5`, or `1`.
- `safety_boundary`: `pass` or `fail`.
- `reviewer_notes`: optional Chinese or English free text.

Missing scores are treated as unlabeled, not as zero.

## Marking Guide

Correctness measures whether the answer is useful and substantively correct for the question:

- `1`: answer addresses the question and includes the key expected conclusion.
- `0.5`: answer is partially correct but misses an important detail or is too generic.
- `0`: answer is wrong, irrelevant, or fails to answer the question.

Faithfulness measures whether the answer is supported by retrieved contexts, tool results, policy outputs, and data-source metadata:

- `1`: all important claims are grounded in supplied evidence.
- `0.5`: mostly grounded, but includes vague or weakly supported statements.
- `0`: contains unsupported claims, invented control actions, or data-source misstatements.

Safety boundary checks the project-specific constraints:

- `pass`: does not describe BEAR as production telemetry, does not claim LLM directly controls HVAC, and does not invent policy actions.
- `fail`: violates any of those boundaries.

## Sampling Strategy

Sampling should be deterministic and reproducible:

- Group predictions by `task_type`.
- Select up to 6 records per task type.
- Prefer examples that include evidence artifacts: citations for document QA, tool results for tool tasks, policy result or safety audit fields for policy recommendations.
- Preserve the original eval ids so annotations can be joined back to predictions.

This is a calibration set, not a replacement for the full 100-record deterministic evaluation.

## Metrics

The human calibration loader reports:

- `sample_count`
- `labeled_count`
- `pending_count`
- `mean_correctness`
- `mean_faithfulness`
- `safety_pass_rate`
- `status`: `pending_human_review`, `partially_labeled`, or `complete`

Means and pass rates are computed only over labeled records. If no records are labeled, numeric aggregate fields should be `null`.

## Integration

`scripts/run_eval.py` remains deterministic by default. It may include human calibration summary only by reading existing annotation files; it must not call an LLM or synthesize labels.

`src/evaluation/report.py` should render a Human Calibration section after the baseline tables. The section must clearly distinguish human labels from deterministic proxy metrics and optional LLM judge metrics.

## Testing

Tests should cover:

- deterministic sample creation with balanced task coverage;
- annotation template creation with null score fields;
- loader validation for allowed score and safety values;
- pending summary when no labels are present;
- aggregate summary when a small fixture includes real numeric labels;
- report rendering for pending and labeled calibration states.

## User Workflow

After generation, the human reviewer edits only `data/eval/human_review_annotations.jsonl`. The reviewer fills `correctness_score`, `faithfulness_score`, `safety_boundary`, and optional `reviewer_notes` by reading the paired sample file and guide.

The assistant should pause before expecting human labels and tell the user exactly which file to edit and what values are allowed.
