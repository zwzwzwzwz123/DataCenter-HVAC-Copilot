from pathlib import Path

import pandas as pd

from src.api.demo_factory import build_demo_orchestrator
from src.ingestion.processed_loader import load_processed_bear_trajectory


def test_load_processed_bear_trajectory_normalizes_standard_csv(tmp_path: Path):
    csv_path = tmp_path / "bear_rollout.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
            "scenario_id": ["episode_001", "episode_001"],
            "zone_id": ["zone_0", "zone_1"],
            "zone_temperature": [23.0, 24.0],
            "outdoor_temp": [31.0, 31.0],
            "solar_irradiance": [0.4, 0.5],
            "ground_temp": [18.0, 18.0],
            "internal_load": [0.12, 0.13],
            "control_action": [-800.0, -900.0],
            "reward": [-1.0, -1.1],
            "comfort_violation": [False, True],
        }
    ).to_csv(csv_path, index=False)

    frame = load_processed_bear_trajectory(csv_path)

    assert len(frame) == 2
    assert list(frame["zone_id"]) == ["zone_0", "zone_1"]
    assert frame.attrs["source"] == "processed_bear_csv"
    assert frame["pue"].isna().all()


def test_build_demo_orchestrator_prefers_processed_bear_csv(tmp_path: Path, monkeypatch):
    documents_dir = tmp_path / "data" / "documents"
    documents_dir.mkdir(parents=True)
    (documents_dir / "sample_hvac_guidance.md").write_text(
        "# Sample HVAC Guidance\n\nProcessed BEAR data should be preferred when present.",
        encoding="utf-8",
    )
    processed_dir = tmp_path / "data" / "bear_processed"
    processed_dir.mkdir(parents=True)
    csv_path = processed_dir / "bear_rollout.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z"],
            "scenario_id": ["episode_processed"],
            "zone_id": ["zone_0"],
            "zone_temperature": [22.0],
            "outdoor_temp": [30.0],
            "solar_irradiance": [0.2],
            "ground_temp": [18.0],
            "internal_load": [0.1],
            "control_action": [-500.0],
            "reward": [-0.5],
            "comfort_violation": [False],
        }
    ).to_csv(csv_path, index=False)

    orchestrator = build_demo_orchestrator(project_root=tmp_path)

    assert orchestrator.trajectory.attrs["source"] == "processed_bear_csv"
    assert orchestrator.data_source["kind"] == "processed_csv"
    assert orchestrator.data_source["path"] == "data/bear_processed/bear_rollout.csv"
    assert orchestrator.trajectory.iloc[0]["scenario_id"] == "episode_processed"
