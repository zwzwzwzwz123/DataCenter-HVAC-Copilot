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


def data_quality_check(
    data: pd.DataFrame,
    *,
    required_fields: list[str] | None = None,
    expected_frequency: str | None = None,
) -> dict[str, Any]:
    required = required_fields or [
        "timestamp",
        "scenario_id",
        "zone_id",
        "zone_temperature",
    ]
    missing_fields = [field for field in required if field not in data.columns]
    present_required = [field for field in required if field in data.columns]
    null_counts = {
        field: int(data[field].isna().sum())
        for field in present_required
        if data[field].isna().any()
    }
    duplicate_timestamp_count = 0
    time_gap_count = 0
    if "timestamp" in data.columns and not data.empty:
        frame = data.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        group_columns = ["zone_id"] if "zone_id" in frame.columns else []
        duplicate_timestamp_count = int(
            frame.duplicated(subset=[*group_columns, "timestamp"]).sum()
        )
        if expected_frequency:
            expected_delta = pd.Timedelta(expected_frequency)
            ordered_timestamps = frame["timestamp"].drop_duplicates().sort_values()
            global_deltas = ordered_timestamps.diff().dropna()
            time_gap_count += int((global_deltas > expected_delta).sum())
            groups = frame.groupby(group_columns) if group_columns else [(None, frame)]
            for _, group in groups:
                ordered = group.sort_values("timestamp")
                deltas = ordered["timestamp"].diff().dropna()
                time_gap_count += int((deltas > expected_delta).sum())

    issue_count = (
        len(missing_fields)
        + len(null_counts)
        + duplicate_timestamp_count
        + time_gap_count
    )
    quality_score = max(0.0, 1.0 - min(issue_count, 10) / 10.0)
    status = "pass" if issue_count == 0 else "warning"
    if missing_fields or quality_score < 0.7:
        status = "fail"
    return {
        "tool_name": "data_quality_check",
        "row_count": int(len(data)),
        "required_fields": required,
        "missing_fields": missing_fields,
        "null_counts": null_counts,
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "time_gap_count": time_gap_count,
        "quality_score": quality_score,
        "status": status,
    }


def comfort_risk_assessment(
    data: pd.DataFrame,
    *,
    temperature_metric: str = "zone_temperature",
    comfort_lower_bound: float = 22.0,
    comfort_upper_bound: float = 26.0,
) -> dict[str, Any]:
    if temperature_metric not in data.columns:
        raise ValueError(f"Metric '{temperature_metric}' is not present in trajectory data.")
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    values = frame[temperature_metric].astype(float)
    low_mask = values < comfort_lower_bound
    high_mask = values > comfort_upper_bound
    violation_mask = low_mask | high_mask
    violation_count = int(violation_mask.sum())
    max_temperature = float(values.max()) if not values.empty else None
    min_temperature = float(values.min()) if not values.empty else None

    zone_risks: list[dict[str, Any]] = []
    if "zone_id" in frame.columns:
        for zone_id, group in frame.groupby("zone_id"):
            group_values = group[temperature_metric].astype(float)
            group_violation = (
                (group_values < comfort_lower_bound)
                | (group_values > comfort_upper_bound)
            )
            zone_risks.append(
                {
                    "zone_id": str(zone_id),
                    "violation_count": int(group_violation.sum()),
                    "max_temperature": float(group_values.max()),
                    "min_temperature": float(group_values.min()),
                }
            )
        zone_risks.sort(
            key=lambda item: (
                -int(item["violation_count"]),
                -float(item["max_temperature"]),
                str(item["zone_id"]),
            )
        )
    worst_zone_id = zone_risks[0]["zone_id"] if zone_risks else None
    risk_level = "low"
    if violation_count > 0:
        risk_level = "medium"
    if max_temperature is not None and max_temperature > comfort_upper_bound + 2:
        risk_level = "high"
    return {
        "tool_name": "comfort_risk_assessment",
        "temperature_metric": temperature_metric,
        "comfort_lower_bound": comfort_lower_bound,
        "comfort_upper_bound": comfort_upper_bound,
        "violation_count": violation_count,
        "high_violation_count": int(high_mask.sum()),
        "low_violation_count": int(low_mask.sum()),
        "risk_level": risk_level,
        "worst_zone_id": worst_zone_id,
        "max_temperature": max_temperature,
        "min_temperature": min_temperature,
        "zone_risks": zone_risks,
    }


def zone_hotspot_rank(
    data: pd.DataFrame,
    *,
    metric_name: str = "zone_temperature",
    top_k: int = 3,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if "zone_id" not in data.columns:
        raise ValueError("zone_hotspot_rank requires a zone_id column.")
    if metric_name not in data.columns:
        raise ValueError(f"Metric '{metric_name}' is not present in trajectory data.")
    ranked: list[dict[str, Any]] = []
    for zone_id, group in data.groupby("zone_id"):
        values = group[metric_name].astype(float)
        item = {
            "zone_id": str(zone_id),
            "mean": float(values.mean()),
            "max": float(values.max()),
            "min": float(values.min()),
            "count": int(values.count()),
            "comfort_violation_count": int(group.get("comfort_violation", pd.Series(dtype=bool)).sum()),
        }
        ranked.append(item)
    ranked.sort(
        key=lambda item: (
            -float(item["max"]),
            -float(item["mean"]),
            str(item["zone_id"]),
        )
    )
    return {
        "tool_name": "zone_hotspot_rank",
        "metric_name": metric_name,
        "top_k": top_k,
        "ranked_zones": ranked[:top_k],
    }


def control_action_audit(
    data: pd.DataFrame,
    *,
    action_metric: str = "control_action",
    change_threshold: float = 0.5,
) -> dict[str, Any]:
    if action_metric not in data.columns:
        raise ValueError(f"Metric '{action_metric}' is not present in trajectory data.")
    if change_threshold < 0:
        raise ValueError("change_threshold must be non-negative.")
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    group_columns = ["zone_id"] if "zone_id" in frame.columns else []
    groups = frame.groupby("zone_id") if group_columns else [(None, frame)]
    events: list[dict[str, Any]] = []
    max_abs_change = 0.0
    for zone_id, group in groups:
        ordered = group.sort_values("timestamp").reset_index(drop=True)
        values = ordered[action_metric].astype(float)
        changes = values.diff()
        for index, change in changes.dropna().items():
            abs_change = abs(float(change))
            max_abs_change = max(max_abs_change, abs_change)
            if abs_change > change_threshold:
                events.append(
                    {
                        "timestamp": ordered.loc[index, "timestamp"].isoformat(),
                        "zone_id": str(zone_id) if zone_id is not None else None,
                        "previous_action": float(values.iloc[index - 1]),
                        "current_action": float(values.iloc[index]),
                        "abs_change": abs_change,
                    }
                )
    return {
        "tool_name": "control_action_audit",
        "action_metric": action_metric,
        "change_threshold": change_threshold,
        "large_change_count": len(events),
        "max_abs_change": max_abs_change,
        "stability_status": "unstable" if events else "stable",
        "events": events,
    }


def cooling_efficiency_summary(
    data: pd.DataFrame,
    *,
    power_metrics: list[str] | None = None,
    temperature_metric: str = "zone_temperature",
    comfort_upper_bound: float = 26.0,
) -> dict[str, Any]:
    metrics = power_metrics or ["cooling_power", "fan_power", "chiller_power", "hvac_power"]
    available_metrics = [
        metric for metric in metrics if metric in data.columns and data[metric].notna().any()
    ]
    components = {
        metric: float(data[metric].dropna().sum())
        for metric in available_metrics
    }
    total_power = float(sum(components.values()))
    comfort_violation_count = 0
    max_temperature = None
    if temperature_metric in data.columns:
        temperatures = data[temperature_metric].astype(float)
        comfort_violation_count = int((temperatures > comfort_upper_bound).sum())
        max_temperature = float(temperatures.max())
    power_per_violation = (
        None
        if comfort_violation_count == 0
        else total_power / comfort_violation_count
    )
    efficiency_status = "normal"
    if total_power > 0 and comfort_violation_count > 0:
        efficiency_status = "needs_attention"
    if max_temperature is not None and max_temperature > comfort_upper_bound + 2:
        efficiency_status = "critical"
    return {
        "tool_name": "cooling_efficiency_summary",
        "power_metrics": available_metrics,
        "components": components,
        "total_power": total_power,
        "temperature_metric": temperature_metric,
        "comfort_upper_bound": comfort_upper_bound,
        "comfort_violation_count": comfort_violation_count,
        "max_temperature": max_temperature,
        "power_per_violation": power_per_violation,
        "efficiency_status": efficiency_status,
    }

