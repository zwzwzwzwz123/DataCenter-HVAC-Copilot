from __future__ import annotations

from statistics import mean
from time import perf_counter
from typing import Any, Callable

from src.evaluation.dataset import EvalRecord
from src.policies.base import PolicyResult


def run_policy_benchmark(
    records: list[EvalRecord],
    state_provider: Callable[[EvalRecord], dict[str, Any]],
    policy_runner: Callable[[dict[str, Any]], PolicyResult],
) -> dict[str, object]:
    rows = []
    for record in records:
        if record.task_type != "policy_recommendation":
            continue
        state = state_provider(record)
        start = perf_counter()
        result = policy_runner(state)
        latency_ms = (perf_counter() - start) * 1000
        rows.append(
            {
                "id": record.id,
                "latency_ms": latency_ms,
                "result": result,
            }
        )
    return summarize_policy_benchmark(rows)


def summarize_policy_benchmark(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {
            "sample_count": 0,
            "success_count": 0,
            "fallback_count": 0,
            "avg_latency_ms": 0.0,
            "avg_action_dim": 0.0,
            "avg_abs_action": 0.0,
        }

    results = [row["result"] for row in rows if isinstance(row.get("result"), PolicyResult)]
    latencies = [float(row.get("latency_ms", 0.0)) for row in rows]
    actions = [result.recommended_action for result in results]
    flat_actions = [abs(value) for action in actions for value in action]
    success_count = sum(
        1 for result in results if result.policy_name == "dropt_guided_diffno_checkpoint"
    )
    fallback_count = sum(1 for result in results if "fallback" in result.policy_name)
    return {
        "sample_count": len(rows),
        "success_count": success_count,
        "fallback_count": fallback_count,
        "avg_latency_ms": mean(latencies),
        "avg_action_dim": mean([len(action) for action in actions]) if actions else 0.0,
        "avg_abs_action": mean(flat_actions) if flat_actions else 0.0,
    }
