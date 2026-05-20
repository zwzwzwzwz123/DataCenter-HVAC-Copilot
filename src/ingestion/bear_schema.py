from __future__ import annotations

import pandas as pd

from src.core.schemas import FieldProvenance, FieldSpec


BEAR_FIELD_SPECS: dict[str, FieldSpec] = {
    "timestamp": FieldSpec(
        "timestamp",
        "datetime64[ns, UTC]",
        FieldProvenance.NATIVE,
        True,
        "Trajectory timestamp normalized to UTC.",
    ),
    "scenario_id": FieldSpec(
        "scenario_id",
        "string",
        FieldProvenance.NATIVE,
        True,
        "BEAR episode, scenario, or rollout identifier.",
    ),
    "zone_id": FieldSpec(
        "zone_id",
        "string",
        FieldProvenance.NATIVE,
        True,
        "Thermal zone identifier.",
    ),
    "zone_temperature": FieldSpec(
        "zone_temperature",
        "float",
        FieldProvenance.NATIVE,
        True,
        "Zone air temperature from BEAR trajectory state.",
    ),
    "outdoor_temp": FieldSpec(
        "outdoor_temp",
        "float",
        FieldProvenance.NATIVE,
        True,
        "Outdoor temperature from BEAR weather or environment state.",
    ),
    "solar_irradiance": FieldSpec(
        "solar_irradiance",
        "float",
        FieldProvenance.NATIVE,
        False,
        "Solar irradiance when exported by the BEAR environment.",
    ),
    "ground_temp": FieldSpec(
        "ground_temp",
        "float",
        FieldProvenance.NATIVE,
        False,
        "Ground temperature when exported by the BEAR environment.",
    ),
    "internal_load": FieldSpec(
        "internal_load",
        "float",
        FieldProvenance.NATIVE,
        True,
        "Internal heat load from BEAR or a direct trajectory mapping.",
    ),
    "humidity": FieldSpec(
        "humidity",
        "float",
        FieldProvenance.OPTIONAL_SYNTHETIC,
        False,
        "Optional humidity feature; synthetic unless a BEAR mapping is documented.",
    ),
    "it_load": FieldSpec(
        "it_load",
        "float",
        FieldProvenance.OPTIONAL_SYNTHETIC,
        False,
        "Optional data-center IT load feature; synthetic unless explicitly mapped.",
    ),
    "cooling_power": FieldSpec(
        "cooling_power",
        "float",
        FieldProvenance.OPTIONAL_DERIVED,
        False,
        "Optional cooling power; derived only when a reproducible mapping exists.",
    ),
    "fan_power": FieldSpec(
        "fan_power",
        "float",
        FieldProvenance.OPTIONAL_DERIVED,
        False,
        "Optional fan power; derived only when a reproducible mapping exists.",
    ),
    "chiller_power": FieldSpec(
        "chiller_power",
        "float",
        FieldProvenance.OPTIONAL_DERIVED,
        False,
        "Optional chiller power; not assumed to be native BEAR output.",
    ),
    "hvac_power": FieldSpec(
        "hvac_power",
        "float",
        FieldProvenance.OPTIONAL_DERIVED,
        False,
        "Aggregate HVAC power or energy proxy when equipment-level split is unavailable.",
    ),
    "pue": FieldSpec(
        "pue",
        "float",
        FieldProvenance.OPTIONAL_DERIVED,
        False,
        "Optional PUE-like metric; must have an explicit calculation method.",
    ),
    "control_action": FieldSpec(
        "control_action",
        "object",
        FieldProvenance.NATIVE,
        True,
        "Control action emitted by the BEAR rollout policy.",
    ),
    "reward": FieldSpec(
        "reward",
        "float",
        FieldProvenance.NATIVE,
        True,
        "Reward from the BEAR environment or direct trajectory export.",
    ),
    "alarm_flag": FieldSpec(
        "alarm_flag",
        "bool",
        FieldProvenance.DERIVED,
        False,
        "Derived alarm flag from reproducible threshold or event rules.",
    ),
    "comfort_violation": FieldSpec(
        "comfort_violation",
        "bool",
        FieldProvenance.NATIVE,
        True,
        "Comfort violation flag from BEAR or a direct comfort-bound calculation.",
    ),
}


def normalize_bear_trajectory(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize BEAR-like trajectory data without inventing optional metrics."""

    required = [name for name, spec in BEAR_FIELD_SPECS.items() if spec.required]
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError(f"Missing required BEAR trajectory fields: {missing}")

    normalized = raw.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)

    for name in BEAR_FIELD_SPECS:
        if name not in normalized.columns:
            normalized[name] = pd.NA

    ordered_columns = list(BEAR_FIELD_SPECS)
    extra_columns = [col for col in normalized.columns if col not in BEAR_FIELD_SPECS]
    normalized = normalized[ordered_columns + extra_columns]
    normalized.attrs["field_provenance"] = {
        name: spec.provenance.value for name, spec in BEAR_FIELD_SPECS.items()
    }
    return normalized

