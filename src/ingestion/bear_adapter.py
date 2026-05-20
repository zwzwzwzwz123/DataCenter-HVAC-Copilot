from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.ingestion.bear_schema import normalize_bear_trajectory


BEAR_IMPORT_HINT = (
    "BEAR is not installed or importable. Clone https://github.com/chz056/BEAR.git "
    "outside this project, install its requirements, and pass --bear-root to the export script."
)


@dataclass(frozen=True)
class BearStateLayout:
    roomnum: int

    @property
    def expected_length(self) -> int:
        return 3 * self.roomnum + 2


def require_bear(package_root: str | Path | None = None) -> tuple[type, Callable]:
    if package_root is not None:
        root = Path(package_root)
        if not root.exists():
            raise ImportError(BEAR_IMPORT_HINT)
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    try:
        from BEAR.Env.env_building import BuildingEnvReal
        from BEAR.Utils.utils_building import ParameterGenerator
    except Exception as exc:
        raise ImportError(BEAR_IMPORT_HINT) from exc
    return BuildingEnvReal, ParameterGenerator


def parse_bear_state(state: np.ndarray, layout: BearStateLayout) -> dict[str, Any]:
    values = np.asarray(state, dtype=float).reshape(-1)
    if values.size != layout.expected_length:
        raise ValueError(
            f"Expected BEAR state length {layout.expected_length} for roomnum={layout.roomnum}, got {values.size}."
        )

    n = layout.roomnum
    return {
        "zone_temperature": values[:n].tolist(),
        "outdoor_temp": float(values[n]),
        "solar_irradiance": values[n + 1 : n + 1 + n].tolist(),
        "ground_temp": float(values[n + 1 + n]),
        "internal_load": values[n + 2 + n : n + 2 + 2 * n].tolist(),
    }


def export_bear_rollout(
    env: Any,
    scenario_id: str,
    num_steps: int,
    start_time: str,
    time_resolution_seconds: int,
    action_policy: Callable[[Any], np.ndarray] | None = None,
) -> pd.DataFrame:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")

    initial_state, _ = env.reset()
    layout = BearStateLayout(roomnum=int(env.roomnum))
    timestamps = pd.date_range(
        start=pd.to_datetime(start_time, utc=True),
        periods=num_steps,
        freq=f"{time_resolution_seconds}s",
    )
    rows: list[dict[str, Any]] = []
    state = initial_state

    for step_index in range(num_steps):
        action = action_policy(state) if action_policy is not None else env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)

        stored_state = env.statelist[-1] if getattr(env, "statelist", None) else state
        stored_action = env.actionlist[-1] if getattr(env, "actionlist", None) else action
        parsed = parse_bear_state(stored_state, layout)

        for zone_index in range(layout.roomnum):
            zone_id = f"zone_{zone_index}"
            rows.append(
                {
                    "timestamp": timestamps[step_index],
                    "scenario_id": scenario_id,
                    "zone_id": zone_id,
                    "zone_temperature": parsed["zone_temperature"][zone_index],
                    "outdoor_temp": parsed["outdoor_temp"],
                    "solar_irradiance": parsed["solar_irradiance"][zone_index],
                    "ground_temp": parsed["ground_temp"],
                    "internal_load": parsed["internal_load"][zone_index],
                    "control_action": float(np.asarray(stored_action).reshape(-1)[zone_index]),
                    "reward": float(reward),
                    "comfort_violation": bool(_comfort_violation(parsed["zone_temperature"][zone_index])),
                }
            )

        state = next_state
        if terminated or truncated:
            break

    frame = normalize_bear_trajectory(pd.DataFrame(rows))
    frame.attrs["source"] = "BEAR BuildingEnvReal rollout"
    frame.attrs["scenario_id"] = scenario_id
    frame.attrs["time_resolution_seconds"] = time_resolution_seconds
    return frame


def build_bear_env(
    bear_root: str | Path | None = None,
    building: str = "OfficeSmall",
    weather: str = "Hot_Dry",
    location: str = "Tucson",
) -> Any:
    BuildingEnvReal, ParameterGenerator = require_bear(bear_root)
    root = _bear_data_root(bear_root)
    parameter = ParameterGenerator(building, weather, location, root=root)
    return BuildingEnvReal(parameter)


def _bear_data_root(bear_root: str | Path | None) -> str:
    if bear_root is None:
        return "BEAR/Data/"
    return str(Path(bear_root) / "BEAR" / "Data") + "/"


def _comfort_violation(zone_temperature: float, lower: float = 20.0, upper: float = 26.0) -> bool:
    return zone_temperature < lower or zone_temperature > upper

