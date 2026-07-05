"""Tests for hand-labeled (gold) SFT data conversion and its guard."""

from __future__ import annotations

import json

from distill.build_gold_sft import _step_from_dict, load_gold
from src.agent.planner import PlanStep, validate_plan_steps


def _write(tmp_path, rows):
    path = tmp_path / "gold.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    return str(path)


def test_step_from_dict_rejects_unknown_fields() -> None:
    try:
        _step_from_dict({"route": "document_qa", "bogus": 1})
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown field")


def test_load_gold_keeps_valid_and_drops_invalid(tmp_path) -> None:
    rows = [
        {"id": "ok1", "question": "查一下温度", "steps": [{"route": "timeseries_query", "reason": "r", "tool": "query_metric"}]},
        # invalid: time_window in days is not supported by the guard
        {"id": "bad1", "question": "过去一周温度", "steps": [{"route": "timeseries_query", "reason": "r", "tool": "query_metric", "time_window": "last_7_days"}]},
        # invalid: empty question
        {"id": "bad2", "question": "", "steps": [{"route": "document_qa", "reason": "r"}]},
    ]
    samples, stats = load_gold(_write(tmp_path, rows))

    assert stats["total"] == 3
    assert stats["kept"] == 1
    assert stats["invalid"] == 2
    assert samples[0]["teacher"] == "hand_labeled"


def test_load_gold_completion_roundtrips_and_is_valid(tmp_path) -> None:
    rows = [
        {
            "id": "c1",
            "question": "先查温度再给策略",
            "steps": [
                {"route": "timeseries_query", "reason": "查温度", "tool": "query_metric", "metric_name": "zone_temperature"},
                {"route": "policy_recommendation", "reason": "给策略", "tool": "policy_runner"},
            ],
        }
    ]
    samples, _ = load_gold(_write(tmp_path, rows))
    parsed = json.loads(samples[0]["completion"])

    # policy step must remain last, and the whole plan must pass the guard
    assert parsed["steps"][-1]["route"] == "policy_recommendation"
    steps = [PlanStep(**{k: v for k, v in s.items()}) for s in parsed["steps"]]
    validate_plan_steps(steps)


def test_shipped_gold_dataset_is_fully_valid() -> None:
    # The checked-in gold_labeled.jsonl must stay 100% guard-valid so training
    # data never silently degrades.
    samples, stats = load_gold("distill/data/gold_labeled.jsonl")

    assert stats["total"] == 700
    assert stats["invalid"] == 0
    assert stats["kept"] == 700
