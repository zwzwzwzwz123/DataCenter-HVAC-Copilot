from __future__ import annotations

from src.policies.base import PolicyResult, current_action_from, state_id_from


def run_mpc_like_policy(state: dict, horizon: int = 6) -> PolicyResult:
    """Simplified MPC-like adapter that returns a structured estimate.

    This is not a full optimizer. It is a deterministic placeholder with a stable
    interface for later replacement by a real MPC or simulator-backed policy.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    current_action = current_action_from(state)
    temperature = float(state.get("zone_temperature", 0.0))
    upper_bound = float(state.get("comfort_upper_bound", 26.0))
    hvac_power = float(state.get("hvac_power", 0.0))

    if temperature > upper_bound:
        recommended = [-0.05 for _ in current_action]
        comfort_risk = min(1.0, (temperature - upper_bound) / 5.0)
        energy_factor = 1.05
    else:
        recommended = [-0.02 for _ in current_action]
        comfort_risk = 0.0
        energy_factor = 0.95

    mean_change = sum(abs(value) for value in recommended) / len(recommended)
    return PolicyResult(
        policy_name="mpc_like",
        input_state_id=state_id_from(state),
        recommended_action=recommended,
        estimated_energy=hvac_power * horizon * energy_factor,
        estimated_comfort_violations=comfort_risk,
        mean_action_change=mean_change,
        baseline="current_policy",
        notes="Deterministic MPC-like placeholder; replace with simulator-backed policy later.",
    )

