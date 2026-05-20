from __future__ import annotations

from src.policies.base import PolicyResult, current_action_from, state_id_from


def run_rule_based_policy(state: dict) -> PolicyResult:
    """Small deterministic policy for interface testing and fallback demos."""

    current_action = current_action_from(state)
    temperature = float(state.get("zone_temperature", 0.0))
    upper_bound = float(state.get("comfort_upper_bound", 26.0))
    lower_bound = float(state.get("comfort_lower_bound", 22.0))

    if temperature > upper_bound:
        recommended = [-0.1 for _ in current_action]
        note = "Rule-based fallback increased cooling because temperature exceeded comfort upper bound."
    elif temperature < lower_bound:
        recommended = [0.1 for _ in current_action]
        note = "Rule-based fallback relaxed cooling because temperature was below comfort lower bound."
    else:
        recommended = [0.0 for _ in current_action]
        note = "Rule-based fallback kept action neutral because temperature was within comfort bounds."

    mean_change = sum(abs(value) for value in recommended) / len(recommended)
    return PolicyResult(
        policy_name="rule_based",
        input_state_id=state_id_from(state),
        recommended_action=recommended,
        estimated_comfort_violations=1.0 if temperature > upper_bound or temperature < lower_bound else 0.0,
        mean_action_change=mean_change,
        notes=note,
    )

