from __future__ import annotations

import pytest

from src.evaluation.policy_benchmark import summarize_policy_benchmark
from src.policies.base import PolicyResult


def test_summarize_policy_benchmark_reports_latency_and_action_distribution() -> None:
    rows = [
        {
            "id": "policy_001",
            "latency_ms": 12.0,
            "result": PolicyResult(
                policy_name="dropt_guided_diffno_checkpoint",
                input_state_id="state_1",
                recommended_action=[0.1, -0.2],
                mean_action_change=0.15,
                notes="ok",
            ),
        },
        {
            "id": "policy_002",
            "latency_ms": 18.0,
            "result": PolicyResult(
                policy_name="dropt_checkpoint_fallback",
                input_state_id="state_2",
                recommended_action=[0.0, 0.0],
                mean_action_change=0.0,
                baseline="rule_based",
                notes="fallback",
            ),
        },
    ]

    summary = summarize_policy_benchmark(rows)

    assert summary["sample_count"] == 2
    assert summary["success_count"] == 1
    assert summary["fallback_count"] == 1
    assert summary["avg_latency_ms"] == 15.0
    assert summary["avg_action_dim"] == 2.0
    assert summary["avg_abs_action"] == pytest.approx(0.075)
