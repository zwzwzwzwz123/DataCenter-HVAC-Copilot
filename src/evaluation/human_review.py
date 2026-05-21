from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.dataset import EvalRecord


TASK_ORDER = [
    "anomaly_diagnosis",
    "document_qa",
    "policy_recommendation",
    "timeseries_query",
]
ALLOWED_SCORES = {0, 0.5, 1}
ALLOWED_SAFETY = {"pass", "fail"}


def create_human_review_sample(
    records: list[EvalRecord],
    prediction_map: dict[str, dict[str, Any]],
    *,
    per_task: int = 6,
) -> list[dict[str, Any]]:
    sample: list[dict[str, Any]] = []
    for task_type in TASK_ORDER:
        task_records = [record for record in records if record.task_type == task_type]
        selected = sorted(task_records, key=lambda record: record.id)[:per_task]
        for record in selected:
            prediction = prediction_map.get(record.id, {})
            sample.append(
                {
                    "id": record.id,
                    "task_type": record.task_type,
                    "question": record.question,
                    "gold_answer": record.gold_answer,
                    "answer": prediction.get("answer", ""),
                    "citations": prediction.get("citations", []),
                    "retrieved_contexts": prediction.get("retrieved_contexts", []),
                    "tool_results": prediction.get("tool_results", []),
                    "answer_audit": prediction.get("answer_audit", {}),
                    "expected_keywords": record.expected_keywords,
                    "required_tools": record.required_tools,
                    "required_documents": record.required_documents,
                }
            )
    return sample


def create_annotation_template(sample: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": item["id"],
            "task_type": item["task_type"],
            "correctness_score": None,
            "faithfulness_score": None,
            "safety_boundary": None,
            "reviewer_notes": "",
        }
        for item in sample
    ]


def load_human_annotations(path: str | Path) -> list[dict[str, Any]]:
    annotation_path = Path(path)
    if not annotation_path.exists():
        return []
    annotations: list[dict[str, Any]] = []
    for line_number, line in enumerate(annotation_path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        _validate_annotation(item, line_number)
        annotations.append(item)
    return annotations


def human_calibration_summary(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [
        item
        for item in annotations
        if item.get("correctness_score") is not None
        and item.get("faithfulness_score") is not None
        and item.get("safety_boundary") is not None
    ]
    sample_count = len(annotations)
    labeled_count = len(labeled)
    pending_count = sample_count - labeled_count
    if labeled_count == 0:
        return {
            "sample_count": sample_count,
            "labeled_count": 0,
            "pending_count": pending_count,
            "mean_correctness": None,
            "mean_faithfulness": None,
            "safety_pass_rate": None,
            "status": "pending_human_review",
        }
    status = "complete" if pending_count == 0 else "partially_labeled"
    return {
        "sample_count": sample_count,
        "labeled_count": labeled_count,
        "pending_count": pending_count,
        "mean_correctness": sum(float(item["correctness_score"]) for item in labeled)
        / labeled_count,
        "mean_faithfulness": sum(float(item["faithfulness_score"]) for item in labeled)
        / labeled_count,
        "safety_pass_rate": sum(1 for item in labeled if item["safety_boundary"] == "pass")
        / labeled_count,
        "status": status,
    }


def _validate_annotation(item: dict[str, Any], line_number: int) -> None:
    for score_name in ["correctness_score", "faithfulness_score"]:
        value = item.get(score_name)
        if value is not None and value not in ALLOWED_SCORES:
            raise ValueError(
                f"Invalid {score_name} on line {line_number}: expected 0, 0.5, 1, or null."
            )
    safety = item.get("safety_boundary")
    if safety is not None and safety not in ALLOWED_SAFETY:
        raise ValueError(
            f"Invalid safety_boundary on line {line_number}: expected pass, fail, or null."
        )


def save_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
