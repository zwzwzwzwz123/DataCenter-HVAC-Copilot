from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.ingestion.bear_schema import normalize_bear_trajectory


ZONE_NAME_MAP = {
    "PLENUM": "plenum",
    "SOUTH PERIMETER": "south_perimeter",
    "EAST PERIMETER": "east_perimeter",
    "NORTH PERIMETER": "north_perimeter",
    "WEST PERIMETER": "west_perimeter",
    "CORE": "core",
}

CONTROL_COLUMN_PREFIX = {
    "PLENUM": "PLENUM",
    "SOUTH PERIMETER": "SOUTH ZONE",
    "EAST PERIMETER": "EAST ZONE",
    "NORTH PERIMETER": "NORTH ZONE",
    "WEST PERIMETER": "WEST ZONE",
    "CORE": "CORE",
}


def load_bear_sample_timeseries(
    csv_path: str | Path,
    scenario_id: str = "bear_sample_rollout",
) -> pd.DataFrame:
    wide = pd.read_csv(csv_path)
    if "Date/Time" not in wide.columns:
        raise ValueError("BEAR sample CSV must contain a 'Date/Time' column.")

    rows: list[dict[str, Any]] = []
    zone_temperature_columns = _matching_columns(wide.columns, "Zone Air Temperature [C](Hourly)")
    hvac_total_column = _find_column(
        wide.columns, "Whole Building:Facility Total HVAC Electricity Demand Rate [W](Hourly)"
    )

    for _, row in wide.iterrows():
        timestamp = _to_timestamp(str(row["Date/Time"]))
        outdoor_temp = float(row["Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)"])
        hvac_total = float(row[hvac_total_column]) if hvac_total_column else 0.0
        per_zone_hvac = hvac_total / max(len(zone_temperature_columns), 1)

        for zone_column in zone_temperature_columns:
            source_zone = zone_column.split(":")[0]
            zone_id = ZONE_NAME_MAP.get(source_zone, _slugify(source_zone))
            rows.append(
                {
                    "timestamp": timestamp,
                    "scenario_id": scenario_id,
                    "zone_id": zone_id,
                    "zone_temperature": float(row[zone_column]),
                    "outdoor_temp": outdoor_temp,
                    "solar_irradiance": float(row[_first_existing_column(wide.columns, source_zone, "Lights Electricity Rate [W](Hourly)")]) if _first_existing_column(wide.columns, source_zone, "Lights Electricity Rate [W](Hourly)") else 0.0,
                    "ground_temp": outdoor_temp,
                    "internal_load": float(row[_first_existing_column(wide.columns, source_zone, "Zone Air System Sensible Heating Rate [W](Hourly)")]) if _first_existing_column(wide.columns, source_zone, "Zone Air System Sensible Heating Rate [W](Hourly)") else 0.0,
                    "cooling_power": per_zone_hvac,
                    "fan_power": per_zone_hvac * 0.25,
                    "chiller_power": per_zone_hvac * 0.75,
                    "control_action": float(
                        row[
                            _first_existing_column(
                                wide.columns,
                                CONTROL_COLUMN_PREFIX[source_zone],
                                "Cooling Coil Electricity Rate [W](Hourly)",
                            )
                        ]
                    )
                    if _first_existing_column(
                        wide.columns,
                        CONTROL_COLUMN_PREFIX[source_zone],
                        "Cooling Coil Electricity Rate [W](Hourly)",
                    )
                    else 0.0,
                    "reward": 0.0,
                    "comfort_violation": False,
                }
            )

    standardized = normalize_bear_trajectory(pd.DataFrame(rows))
    standardized.attrs["source"] = "BEAR sample timeseries csv"
    standardized.attrs["source_path"] = str(csv_path)
    standardized.attrs["scenario_id"] = scenario_id
    return standardized


def _matching_columns(columns: pd.Index, suffix: str) -> list[str]:
    return [column for column in columns if suffix in column and "Whole Building" not in column]


def _first_existing_column(columns: pd.Index, zone_name: str, suffix: str) -> str | None:
    candidates = [
        f"{zone_name}:{suffix}",
        f"{zone_name} ZONE UNITARY COOLING COIL:{suffix}",
        f"{zone_name} UNITARY COOLING COIL:{suffix}",
        f"{zone_name} ZONE UNITARY COOLING COIL:{suffix}",
        f"{zone_name} UNITARY HEATING COIL:{suffix}",
    ]
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _find_column(columns: pd.Index, exact_name: str) -> str | None:
    return exact_name if exact_name in columns else None


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _to_timestamp(value: str) -> pd.Timestamp:
    raw = value.strip()
    month_day, time_part = raw.split()
    month, day = month_day.split("/")
    hour, minute, second = map(int, time_part.split(":"))
    base = pd.Timestamp(year=2026, month=int(month), day=int(day), tz="UTC")
    if hour == 24:
        base = base + pd.Timedelta(days=1)
        hour = 0
    return base + pd.Timedelta(hours=hour, minutes=minute, seconds=second)
