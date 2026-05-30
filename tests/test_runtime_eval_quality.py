from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


RUNTIME_EVAL_PATH = Path("data/eval/agent_runtime_eval.jsonl")


def _records() -> list[dict]:
    return [
        json.loads(line)
        for line in RUNTIME_EVAL_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_agent_runtime_eval_has_planned_size_and_difficulty_mix() -> None:
    records = _records()

    assert len(records) == 50
    assert Counter(record["difficulty"] for record in records) == {
        "easy": 10,
        "medium": 28,
        "hard": 12,
    }


def test_agent_runtime_eval_records_have_benchmark_metadata() -> None:
    records = _records()

    ids = [record["id"] for record in records]
    assert len(ids) == len(set(ids))
    for record in records:
        assert record["capability_tags"]
        assert record["distractor_type"]
        assert record["expected_failure_mode"]
        rubric = record["grading_rubric"]
        assert set(rubric)
        assert abs(sum(float(value) for value in rubric.values()) - 1.0) < 1e-9


def test_agent_runtime_eval_has_capability_coverage() -> None:
    records = _records()
    tags = Counter(tag for record in records for tag in record["capability_tags"])

    expected_minimums = {
        "multi_step": 12,
        "dynamic_insert": 4,
        "dynamic_replace": 4,
        "stop_guard": 4,
        "policy_deadline_guard": 4,
        "duplicate_guard": 6,
        "approval_denied": 4,
        "tool_retry": 3,
        "query_rewrite_retry": 3,
        "policy_fallback": 3,
        "data_quality_check": 5,
        "comfort_risk_assessment": 5,
        "zone_hotspot_rank": 3,
        "control_action_audit": 3,
        "cooling_efficiency_summary": 3,
        "safety_boundary": 5,
    }
    for tag, minimum in expected_minimums.items():
        assert tags[tag] >= minimum, f"{tag} coverage below {minimum}: {tags[tag]}"


def test_agent_runtime_eval_questions_do_not_leak_guardrail_answers() -> None:
    forbidden = [
        "react",
        "guard",
        "runtime",
        "tool_retry",
        "policy_fallback",
        "query_rewrite_retry",
        "approval_denied",
        "duplicate guard",
        "deadline guard",
    ]

    for record in _records():
        question = record["question"].lower()
        assert not any(token in question for token in forbidden), record["id"]
