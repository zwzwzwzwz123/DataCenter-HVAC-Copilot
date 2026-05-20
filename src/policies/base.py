from __future__ import annotations

from pydantic import BaseModel, Field


class PolicyResult(BaseModel):
    policy_name: str
    input_state_id: str
    recommended_action: list[float]
    estimated_energy: float | None = None
    estimated_comfort_violations: float | None = None
    mean_action_change: float | None = None
    baseline: str | None = None
    notes: str = Field(default="")


def state_id_from(state: dict) -> str:
    return str(state.get("state_id") or state.get("input_state_id") or "unknown_state")


def current_action_from(state: dict) -> list[float]:
    action = state.get("current_action", [0.0])
    return [float(value) for value in action]

