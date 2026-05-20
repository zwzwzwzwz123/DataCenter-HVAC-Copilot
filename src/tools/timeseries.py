from __future__ import annotations

from typing import Any

import pandas as pd

from src.core.schemas import make_summary


def _to_utc(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True)


def _filtered_frame(
    data: pd.DataFrame,
    metric_name: str,
    start_time: Any,
    end_time: Any,
    zone_id: str | None = None,
) -> pd.DataFrame:
    if "timestamp" not in data.columns:
        raise ValueError("Trajectory data must include a timestamp column.")
    if metric_name not in data.columns:
        raise ValueError(f"Metric '{metric_name}' is not present in trajectory data.")

    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    mask = (frame["timestamp"] >= _to_utc(start_time)) & (
        frame["timestamp"] <= _to_utc(end_time)
    )
    if zone_id is not None:
        if "zone_id" not in frame.columns:
            raise ValueError("zone_id filter requested, but trajectory data has no zone_id column.")
        mask &= frame["zone_id"] == zone_id
    return frame.loc[mask].sort_values("timestamp")


def _records(frame: pd.DataFrame, metric_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        row_data = row._asdict()
        records.append(
            {
                "timestamp": row_data["timestamp"].isoformat(),
                "zone_id": row_data.get("zone_id"),
                "value": float(row_data[metric_name]),
            }
        )
    return records


def query_metric(
    data: pd.DataFrame,
    metric_name: str,
    start_time: Any,
    end_time: Any,
    zone_id: str | None = None,
) -> dict[str, Any]:
    frame = _filtered_frame(data, metric_name, start_time, end_time, zone_id)
    values = [float(value) for value in frame[metric_name].dropna().tolist()]
    return {
        "tool_name": "query_metric",
        "metric_name": metric_name,
        "zone_id": zone_id,
        "start_time": _to_utc(start_time).isoformat(),
        "end_time": _to_utc(end_time).isoformat(),
        "summary": make_summary(values),
        "records": _records(frame, metric_name),
    }


def compare_period(
    data: pd.DataFrame,
    metric_name: str,
    period_a: tuple[Any, Any],
    period_b: tuple[Any, Any],
    zone_id: str | None = None,
) -> dict[str, Any]:
    result_a = query_metric(data, metric_name, period_a[0], period_a[1], zone_id)
    result_b = query_metric(data, metric_name, period_b[0], period_b[1], zone_id)
    mean_a = result_a["summary"]["mean"]
    mean_b = result_b["summary"]["mean"]
    delta = None if mean_a is None or mean_b is None else mean_b - mean_a
    percent_change = None
    if mean_a not in (None, 0) and delta is not None:
        percent_change = (delta / mean_a) * 100.0

    return {
        "tool_name": "compare_period",
        "metric_name": metric_name,
        "zone_id": zone_id,
        "period_a": result_a["summary"],
        "period_b": result_b["summary"],
        "delta_mean": delta,
        "percent_change_mean": percent_change,
    }


def detect_anomaly(
    data: pd.DataFrame,
    metric_name: str,
    window_size: int,
    threshold: float,
    zone_id: str | None = None,
) -> dict[str, Any]:
    if window_size < 2:
        raise ValueError("window_size must be at least 2.")

    frame = _filtered_frame(
        data,
        metric_name,
        data["timestamp"].min(),
        data["timestamp"].max(),
        zone_id,
    )
    values = frame[metric_name].astype(float).reset_index(drop=True)
    anomalies: list[dict[str, Any]] = []
    rows = frame.reset_index(drop=True)

    for index in range(window_size, len(values)):
        baseline = values.iloc[index - window_size : index]
        mean = float(baseline.mean())
        std = float(baseline.std(ddof=0))
        value = float(values.iloc[index])
        if std == 0:
            score = abs(value - mean)
        else:
            score = abs(value - mean) / std
        if score > threshold:
            anomalies.append(
                {
                    "timestamp": rows.loc[index, "timestamp"].isoformat(),
                    "zone_id": rows.loc[index].get("zone_id"),
                    "value": value,
                    "baseline_mean": mean,
                    "score": score,
                }
            )

    return {
        "tool_name": "detect_anomaly",
        "metric_name": metric_name,
        "zone_id": zone_id,
        "window_size": window_size,
        "threshold": threshold,
        "anomalies": anomalies,
    }


def compute_energy_breakdown(
    data: pd.DataFrame,
    start_time: Any,
    end_time: Any,
) -> dict[str, Any]:
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame[
        (frame["timestamp"] >= _to_utc(start_time))
        & (frame["timestamp"] <= _to_utc(end_time))
    ]

    preferred_components = ["cooling_power", "fan_power", "chiller_power"]
    components = {
        name: float(frame[name].dropna().sum())
        for name in preferred_components
        if name in frame.columns and frame[name].notna().any()
    }
    notes: list[str] = []

    if not components and "hvac_power" in frame.columns and frame["hvac_power"].notna().any():
        components["hvac_power"] = float(frame["hvac_power"].dropna().sum())
        notes.append("Used aggregate hvac_power because equipment-level fields were unavailable.")

    return {
        "tool_name": "compute_energy_breakdown",
        "start_time": _to_utc(start_time).isoformat(),
        "end_time": _to_utc(end_time).isoformat(),
        "components": components,
        "total": float(sum(components.values())),
        "notes": notes,
    }


def plot_metric_trend(
    data: pd.DataFrame,
    metric_name: str,
    start_time: Any,
    end_time: Any,
    zone_id: str | None = None,
) -> dict[str, Any]:
    result = query_metric(data, metric_name, start_time, end_time, zone_id)
    return {
        "tool_name": "plot_metric_trend",
        "chart_type": "line",
        "metric_name": metric_name,
        "zone_id": zone_id,
        "x_axis": "timestamp",
        "y_axis": metric_name,
        "series": result["records"],
    }

