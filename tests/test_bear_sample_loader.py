from pathlib import Path

import pandas as pd

from src.api.demo_factory import build_demo_orchestrator
from src.ingestion.bear_sample_loader import load_bear_sample_timeseries


def _write_sample_csv(path: Path) -> None:
    path.write_text(
        "Date/Time,Environment:Site Outdoor Air Drybulb Temperature [C](Hourly),"
        "SOUTH PERIMETER LIGHTS 1:Lights Electricity Rate [W](Hourly),"
        "EAST PERIMETER LIGHTS 1:Lights Electricity Rate [W](Hourly),"
        "NORTH PERIMETER LIGHTS 1:Lights Electricity Rate [W](Hourly),"
        "WEST PERIMETER LIGHTS 1:Lights Electricity Rate [W](Hourly),"
        "CORE LIGHTS 1:Lights Electricity Rate [W](Hourly),"
        "PLENUM:Zone Air Temperature [C](Hourly),"
        "SOUTH PERIMETER:Zone Air Temperature [C](Hourly),"
        "EAST PERIMETER:Zone Air Temperature [C](Hourly),"
        "NORTH PERIMETER:Zone Air Temperature [C](Hourly),"
        "WEST PERIMETER:Zone Air Temperature [C](Hourly),"
        "CORE:Zone Air Temperature [C](Hourly),"
        "Whole Building:Facility Total HVAC Electricity Demand Rate [W](Hourly),"
        "SOUTH ZONE UNITARY HEATING COIL:Heating Coil Electricity Rate [W](Hourly),"
        "EAST ZONE UNITARY HEATING COIL:Heating Coil Electricity Rate [W](Hourly),"
        "WEST ZONE UNITARY HEATING COIL:Heating Coil Electricity Rate [W](Hourly),"
        "CORE UNITARY HEATING COIL:Heating Coil Electricity Rate [W](Hourly),"
        "NORTH ZONE UNITARY HEATING COIL:Heating Coil Electricity Rate [W](Hourly),"
        "SOUTH ZONE UNITARY COOLING COIL:Cooling Coil Electricity Rate [W](Hourly),"
        "EAST ZONE UNITARY COOLING COIL:Cooling Coil Electricity Rate [W](Hourly),"
        "WEST ZONE UNITARY COOLING COIL:Cooling Coil Electricity Rate [W](Hourly),"
        "CORE UNITARY COOLING COIL:Cooling Coil Electricity Rate [W](Hourly),"
        "NORTH ZONE UNITARY COOLING COIL:Cooling Coil Electricity Rate [W](Hourly)\n"
        "01/01  01:00:00,-12.2,1,2,3,4,5,15.0,18.0,19.0,20.0,21.0,22.0,100,10,11,12,13,14,2,3,4,5,6\n"
        "01/01  02:00:00,-11.7,2,3,4,5,6,15.5,18.5,19.5,20.5,21.5,22.5,110,11,12,13,14,15,3,4,5,6,7\n",
        encoding="utf-8",
    )


def test_load_bear_sample_timeseries_creates_long_standard_frame(tmp_path: Path):
    csv_path = tmp_path / "Exercise2A-mytest.csv"
    _write_sample_csv(csv_path)

    frame = load_bear_sample_timeseries(csv_path, scenario_id="sample_rollout")

    assert len(frame) == 12
    assert frame.attrs["source"] == "BEAR sample timeseries csv"
    assert set(frame["zone_id"]) == {
        "plenum",
        "south_perimeter",
        "east_perimeter",
        "north_perimeter",
        "west_perimeter",
        "core",
    }
    assert frame.loc[frame["zone_id"] == "south_perimeter", "control_action"].iloc[0] == 2.0
    assert frame["pue"].isna().all()


def test_build_demo_orchestrator_uses_bear_sample_csv_when_processed_missing(tmp_path: Path):
    documents_dir = tmp_path / "data" / "documents"
    documents_dir.mkdir(parents=True)
    (documents_dir / "sample_hvac_guidance.md").write_text(
        "# Sample HVAC Guidance\n\nBEAR sample data should be used when processed data is missing.",
        encoding="utf-8",
    )
    bear_data_dir = tmp_path / "BEAR" / "BEAR" / "Data"
    bear_data_dir.mkdir(parents=True)
    _write_sample_csv(bear_data_dir / "Exercise2A-mytest.csv")

    orchestrator = build_demo_orchestrator(project_root=tmp_path)

    assert orchestrator.trajectory.attrs["source"] == "BEAR sample timeseries csv"
    assert orchestrator.data_source["kind"] == "bear_sample_csv"
    assert orchestrator.data_source["path"] == str(bear_data_dir / "Exercise2A-mytest.csv")
    assert len(orchestrator.trajectory) == 12
