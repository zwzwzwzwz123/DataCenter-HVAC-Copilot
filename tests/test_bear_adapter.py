import numpy as np
import pandas as pd
import pytest
import subprocess
import sys

from src.ingestion.bear_adapter import (
    BEAR_IMPORT_HINT,
    BearStateLayout,
    export_bear_rollout,
    parse_bear_state,
    require_bear,
)


class FakeActionSpace:
    def __init__(self) -> None:
        self.actions = [np.array([-0.1, -0.2]), np.array([0.0, -0.1])]
        self.index = 0

    def sample(self):
        action = self.actions[self.index]
        self.index += 1
        return action


class FakeBearEnv:
    roomnum = 2
    maxpower = 8000

    def __init__(self) -> None:
        self.action_space = FakeActionSpace()
        self.statelist = []
        self.actionlist = []
        self._states = [
            np.array([23.0, 24.0, 31.0, 0.4, 0.4, 18.0, 0.12, 0.12]),
            np.array([23.5, 24.2, 32.0, 0.5, 0.5, 18.0, 0.13, 0.13]),
        ]
        self._step_index = 0

    def reset(self):
        return self._states[0], {}

    def step(self, action):
        current_state = self._states[self._step_index]
        self.statelist.append(current_state)
        self.actionlist.append(action * self.maxpower)
        self._step_index += 1
        next_state = self._states[min(self._step_index, len(self._states) - 1)]
        return next_state, -1.5, False, False, {"zone_temperature": next_state[:2]}


def test_parse_bear_state_uses_real_bear_layout():
    layout = BearStateLayout(roomnum=2)
    parsed = parse_bear_state(
        np.array([23.0, 24.0, 31.0, 0.4, 0.5, 18.0, 0.12, 0.13]),
        layout,
    )

    assert parsed["zone_temperature"] == [23.0, 24.0]
    assert parsed["outdoor_temp"] == 31.0
    assert parsed["solar_irradiance"] == [0.4, 0.5]
    assert parsed["ground_temp"] == 18.0
    assert parsed["internal_load"] == [0.12, 0.13]


def test_export_bear_rollout_returns_standardized_rows():
    frame = export_bear_rollout(
        env=FakeBearEnv(),
        scenario_id="fake_bear_rollout",
        num_steps=2,
        start_time="2026-01-01T00:00:00Z",
        time_resolution_seconds=3600,
    )

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 4
    assert set(frame["zone_id"]) == {"zone_0", "zone_1"}
    assert frame.attrs["source"] == "BEAR BuildingEnvReal rollout"
    assert frame.loc[0, "control_action"] == -800.0
    assert frame.loc[1, "control_action"] == -1600.0
    assert frame["pue"].isna().all()
    assert frame["humidity"].isna().all()


def test_require_bear_raises_clear_error_when_package_missing():
    with pytest.raises(ImportError, match=BEAR_IMPORT_HINT):
        require_bear(package_root="definitely_missing_bear_path")


def test_export_bear_script_reports_missing_bear_root():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/export_bear_data.py",
            "--bear-root",
            "definitely_missing_bear_path",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode != 0
    assert "BEAR is not installed or importable" in completed.stderr
