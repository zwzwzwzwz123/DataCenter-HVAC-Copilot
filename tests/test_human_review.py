from pathlib import Path

import pytest

from src.evaluation.dataset import EvalRecord
from src.evaluation.human_review import (
    create_annotation_template,
    create_human_review_sample,
    human_calibration_summary,
    load_human_annotations,
)


def _record(record_id: str, task_type: str) -> EvalRecord:
    return EvalRecord(
        id=record_id,
        question=f"Question {record_id}",
        task_type=task_type,
        gold_answer=f"Gold {record_id}",
        required_tools=["query_metric"] if task_type != "document_qa" else [],
        required_documents=["doc_a"] if task_type == "document_qa" else [],
        expected_keywords=["keyword"],
        expected_output_format="answer_with_evidence",
    )


def _prediction(record_id: str, task_type: str) -> dict:
    return {
        "id": record_id,
        "question": f"Question {record_id}",
        "task_type": task_type,
        "answer": f"Answer {record_id}",
        "citations": [{"source_id": "doc_a"}] if task_type == "document_qa" else [],
        "retrieved_contexts": [{"source_id": "doc_a", "text": "context"}],
        "tool_results": [{"tool_name": "query_metric", "status": "success"}]
        if task_type != "document_qa"
        else [],
        "answer_audit": {"passed": True, "flags": []},
    }


def test_create_human_review_sample_balances_task_types() -> None:
    records = []
    predictions = {}
    for task_type in [
        "document_qa",
        "timeseries_query",
        "anomaly_diagnosis",
        "policy_recommendation",
    ]:
        for index in range(8):
            record_id = f"{task_type}_{index:03d}"
            records.append(_record(record_id, task_type))
            predictions[record_id] = _prediction(record_id, task_type)

    sample = create_human_review_sample(records, predictions, per_task=6)

    assert len(sample) == 24
    counts = {}
    for item in sample:
        counts[item["task_type"]] = counts.get(item["task_type"], 0) + 1
        assert item["gold_answer"].startswith("Gold")
        assert item["answer"].startswith("Answer")
        assert "answer_audit" in item
        assert "expected_keywords" in item
    assert counts == {
        "anomaly_diagnosis": 6,
        "document_qa": 6,
        "policy_recommendation": 6,
        "timeseries_query": 6,
    }


def test_create_annotation_template_leaves_human_scores_empty() -> None:
    sample = [
        {"id": "doc_001", "task_type": "document_qa"},
        {"id": "ts_001", "task_type": "timeseries_query"},
    ]

    template = create_annotation_template(sample)

    assert template == [
        {
            "id": "doc_001",
            "task_type": "document_qa",
            "correctness_score": None,
            "faithfulness_score": None,
            "safety_boundary": None,
            "reviewer_notes": "",
        },
        {
            "id": "ts_001",
            "task_type": "timeseries_query",
            "correctness_score": None,
            "faithfulness_score": None,
            "safety_boundary": None,
            "reviewer_notes": "",
        },
    ]


def test_human_calibration_summary_reports_pending_when_unlabeled() -> None:
    annotations = [
        {
            "id": "doc_001",
            "task_type": "document_qa",
            "correctness_score": None,
            "faithfulness_score": None,
            "safety_boundary": None,
            "reviewer_notes": "",
        }
    ]

    summary = human_calibration_summary(annotations)

    assert summary == {
        "sample_count": 1,
        "labeled_count": 0,
        "pending_count": 1,
        "mean_correctness": None,
        "mean_faithfulness": None,
        "safety_pass_rate": None,
        "status": "pending_human_review",
    }


def test_human_calibration_summary_uses_only_labeled_records() -> None:
    annotations = [
        {
            "id": "doc_001",
            "task_type": "document_qa",
            "correctness_score": 1,
            "faithfulness_score": 0.5,
            "safety_boundary": "pass",
            "reviewer_notes": "good",
        },
        {
            "id": "ts_001",
            "task_type": "timeseries_query",
            "correctness_score": 0.5,
            "faithfulness_score": 1,
            "safety_boundary": "fail",
            "reviewer_notes": "boundary issue",
        },
        {
            "id": "policy_001",
            "task_type": "policy_recommendation",
            "correctness_score": None,
            "faithfulness_score": None,
            "safety_boundary": None,
            "reviewer_notes": "",
        },
    ]

    summary = human_calibration_summary(annotations)

    assert summary["sample_count"] == 3
    assert summary["labeled_count"] == 2
    assert summary["pending_count"] == 1
    assert summary["mean_correctness"] == 0.75
    assert summary["mean_faithfulness"] == 0.75
    assert summary["safety_pass_rate"] == 0.5
    assert summary["status"] == "partially_labeled"


def test_load_human_annotations_rejects_invalid_scores(tmp_path: Path) -> None:
    path = tmp_path / "annotations.jsonl"
    path.write_text(
        '{"id":"doc_001","task_type":"document_qa","correctness_score":0.7,'
        '"faithfulness_score":1,"safety_boundary":"pass","reviewer_notes":""}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="correctness_score"):
        load_human_annotations(path)


def test_load_human_annotations_rejects_invalid_safety_value(tmp_path: Path) -> None:
    path = tmp_path / "annotations.jsonl"
    path.write_text(
        '{"id":"doc_001","task_type":"document_qa","correctness_score":1,'
        '"faithfulness_score":1,"safety_boundary":"maybe","reviewer_notes":""}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="safety_boundary"):
        load_human_annotations(path)
