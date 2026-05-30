from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.evaluation.dataset import EvalRecord, load_eval_dataset
from src.evaluation.safety_adversarial import load_safety_adversarial_dataset


CORE_EVAL_PATHS = [
    Path("data/eval/hvac_eval.jsonl"),
    Path("data/eval/real_eval.jsonl"),
    Path("data/eval/persistent_knowledge_ranking_eval.jsonl"),
    Path("data/eval/compound_task_eval.jsonl"),
    Path("data/eval/agent_runtime_eval.jsonl"),
]

SUPPORTED_TASK_TYPES = {
    "document_qa",
    "timeseries_query",
    "anomaly_diagnosis",
    "policy_recommendation",
}

SUPPORTED_STEPS = SUPPORTED_TASK_TYPES
SUPPORTED_SAFETY_VIOLATIONS = {
    "production_telemetry_claim",
    "llm_direct_control_claim",
    "unverified_policy_action",
}


def test_core_eval_jsonl_files_are_loadable_and_have_unique_ids() -> None:
    for path in CORE_EVAL_PATHS:
        records = load_eval_dataset(path)
        ids = [record.id for record in records]

        assert records, f"{path} is empty"
        assert len(ids) == len(set(ids)), f"{path} has duplicate ids"
        assert all(record.task_type in SUPPORTED_TASK_TYPES for record in records)


def test_core_eval_records_have_grading_signal() -> None:
    for path in CORE_EVAL_PATHS:
        for record in load_eval_dataset(path):
            assert record.question.strip(), record.id
            assert len(record.gold_answer.strip()) >= 20, record.id
            assert record.expected_keywords, record.id
            assert record.expected_output_format.strip(), record.id
            assert (
                record.required_documents
                or record.required_tools
                or record.expected_steps
                or record.expected_tool_sequence
            ), record.id


def test_legacy_and_real_eval_have_balanced_task_coverage() -> None:
    expected = {
        Path("data/eval/hvac_eval.jsonl"): {
            "document_qa": 40,
            "timeseries_query": 20,
            "anomaly_diagnosis": 20,
            "policy_recommendation": 28,
        },
        Path("data/eval/real_eval.jsonl"): {
            "document_qa": 30,
            "timeseries_query": 7,
            "anomaly_diagnosis": 6,
            "policy_recommendation": 7,
        },
    }

    for path, task_counts in expected.items():
        records = load_eval_dataset(path)
        assert Counter(record.task_type for record in records) == task_counts
        assert sum(bool(record.required_documents) for record in records) >= 30
        assert sum(bool(record.required_tools) for record in records) >= 20


def test_ranking_eval_is_document_ranking_focused() -> None:
    records = load_eval_dataset("data/eval/persistent_knowledge_ranking_eval.jsonl")

    assert len(records) == 30
    assert {record.task_type for record in records} == {"document_qa"}
    assert all(record.required_documents for record in records)
    assert all(not record.required_tools for record in records)
    assert all(not record.expected_steps for record in records)


def test_compound_eval_has_valid_multi_step_plans() -> None:
    records = load_eval_dataset("data/eval/compound_task_eval.jsonl")

    assert len(records) == 100
    assert all(len(record.expected_steps) >= 2 for record in records)
    for record in records:
        assert record.task_type == record.expected_steps[-1], record.id
        assert all(step in SUPPORTED_STEPS for step in record.expected_steps), record.id
        assert len(record.expected_steps) == len(set(record.expected_steps)), record.id
        if "policy_recommendation" in record.expected_steps:
            assert record.expected_steps[-1] == "policy_recommendation", record.id


def test_safety_adversarial_eval_has_boundary_coverage() -> None:
    records = load_safety_adversarial_dataset("data/eval/safety_adversarial.jsonl")
    category_counts = Counter(record.category for record in records)
    violation_counts = Counter(record.expected_violation for record in records)

    assert len(records) >= 25
    assert len({record.id for record in records}) == len(records)
    assert category_counts == {
        "paraphrase": 8,
        "indirect": 6,
        "jailbreak": 6,
        "mixed": 5,
        "translation": 4,
        "unverified_action": 6,
    }
    assert set(violation_counts) == SUPPORTED_SAFETY_VIOLATIONS
    assert all(record.question.strip() and record.answer.strip() for record in records)


def test_runtime_eval_grading_rubrics_are_machine_readable() -> None:
    records = _raw_jsonl_records(Path("data/eval/agent_runtime_eval.jsonl"))

    for record in records:
        rubric = record["grading_rubric"]
        assert isinstance(rubric, dict), record["id"]
        assert abs(sum(float(value) for value in rubric.values()) - 1.0) < 1e-9, record["id"]
        assert record["difficulty"] in {"easy", "medium", "hard"}, record["id"]
        assert record["capability_tags"], record["id"]
        assert record["distractor_type"], record["id"]
        assert record["expected_failure_mode"], record["id"]


def _raw_jsonl_records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
