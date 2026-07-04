"""Tests for planner-distillation SFT data export and shared prompt helpers."""

from __future__ import annotations

import json

from distill.build_sft_data import (
    build_samples,
    build_teacher,
    split_train_val,
)
from src.agent.planner import (
    PlanStep,
    _decision_from_llm_payload,
    build_planner_messages,
    serialize_plan_steps,
    validate_plan_steps,
)


def test_build_planner_messages_has_system_and_user_roles() -> None:
    messages = build_planner_messages("最近 zone_temperature 有没有异常？")

    assert [m["role"] for m in messages] == ["system", "user"]
    assert "route planner" in messages[0]["content"].lower()
    user_payload = json.loads(messages[1]["content"])
    assert user_payload["question"] == "最近 zone_temperature 有没有异常？"
    assert user_payload["conversation_context"] == {}


def test_serialize_plan_steps_roundtrips_through_planner_parser() -> None:
    # The SFT label format must be parseable by the exact code path the online
    # planner uses, otherwise a distilled model's output could not be consumed.
    steps = [
        PlanStep(route="timeseries_query", reason="check temp", tool="query_metric"),
        PlanStep(route="policy_recommendation", reason="advise", tool="policy_runner"),
    ]
    label = serialize_plan_steps(steps, confidence=0.9)

    decision = _decision_from_llm_payload(content=label, planner="test")
    assert [s.route for s in decision.steps] == [
        "timeseries_query",
        "policy_recommendation",
    ]
    assert decision.steps[0].tool == "query_metric"
    assert 0.0 <= decision.confidence <= 1.0


def test_serialize_plan_steps_labels_are_schema_valid() -> None:
    steps = validate_plan_steps(
        [PlanStep(route="document_qa", reason="doc lookup")]
    )
    label = serialize_plan_steps(steps)
    parsed = json.loads(label)

    assert parsed["steps"][0]["route"] == "document_qa"
    assert "confidence" in parsed


def test_build_samples_keeps_only_valid_plans_with_deterministic_teacher() -> None:
    questions = [
        {"id": "q1", "question": "最近 zone_temperature 有没有异常？"},
        {"id": "q2", "question": "请给出降温策略建议"},
        {"id": "q3", "question": "ASHRAE 白皮书讲了什么？"},
    ]
    teacher = build_teacher("deterministic")

    samples, stats = build_samples(questions, teacher)

    assert stats["total"] == 3
    assert stats["kept"] == len(samples)
    assert stats["kept"] >= 1
    for sample in samples:
        # every emitted label must survive the shared guard
        parsed = json.loads(sample["completion"])
        steps = [PlanStep(**{k: v for k, v in step.items()}) for step in parsed["steps"]]
        validate_plan_steps(steps)  # raises if invalid
        assert sample["messages"][0]["role"] == "system"


def test_split_train_val_is_deterministic_and_partitions() -> None:
    samples = [{"id": str(i)} for i in range(10)]

    train_a, val_a = split_train_val(samples, val_ratio=0.2, seed=7)
    train_b, val_b = split_train_val(samples, val_ratio=0.2, seed=7)

    assert len(val_a) == 2
    assert len(train_a) == 8
    assert train_a == train_b and val_a == val_b  # reproducible
    ids = {s["id"] for s in train_a} | {s["id"] for s in val_a}
    assert ids == {s["id"] for s in samples}  # no overlap, no loss
