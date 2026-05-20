import pandas as pd
import pytest

from src.ingestion.bear_schema import (
    BEAR_FIELD_SPECS,
    FieldProvenance,
    normalize_bear_trajectory,
)


def test_optional_fields_are_not_marked_native():
    assert BEAR_FIELD_SPECS["pue"].provenance is FieldProvenance.OPTIONAL_DERIVED
    assert BEAR_FIELD_SPECS["humidity"].provenance is FieldProvenance.OPTIONAL_SYNTHETIC
    assert BEAR_FIELD_SPECS["it_load"].provenance is FieldProvenance.OPTIONAL_SYNTHETIC
    assert BEAR_FIELD_SPECS["chiller_power"].provenance is FieldProvenance.OPTIONAL_DERIVED


def test_normalize_bear_trajectory_preserves_missing_optional_fields():
    raw = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00", "2026-01-01T01:00:00"],
            "scenario_id": ["episode_001", "episode_001"],
            "zone_id": ["zone_a", "zone_a"],
            "zone_temperature": [24.0, 25.5],
            "outdoor_temp": [31.0, 32.0],
            "internal_load": [10.0, 11.0],
            "control_action": ["[0.1, 0.2]", "[0.0, 0.1]"],
            "reward": [-1.0, -1.2],
            "comfort_violation": [False, True],
        }
    )

    normalized = normalize_bear_trajectory(raw)

    assert list(normalized["timestamp"]) == list(pd.to_datetime(raw["timestamp"], utc=True))
    assert "pue" in normalized.columns
    assert normalized["pue"].isna().all()
    assert "humidity" in normalized.columns
    assert normalized["humidity"].isna().all()
    assert normalized.attrs["field_provenance"]["pue"] == "optional_derived"


def test_normalize_bear_trajectory_requires_core_fields():
    raw = pd.DataFrame({"timestamp": ["2026-01-01T00:00:00"]})

    with pytest.raises(ValueError, match="Missing required BEAR trajectory fields"):
        normalize_bear_trajectory(raw)

