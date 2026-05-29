import pandas as pd

from src.tools.timeseries import (
    compare_period,
    compute_energy_breakdown,
    control_action_audit,
    cooling_efficiency_summary,
    data_quality_check,
    detect_anomaly,
    comfort_risk_assessment,
    plot_metric_trend,
    query_metric,
    zone_hotspot_rank,
)


def mock_trajectory():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC"),
            "scenario_id": ["episode_001"] * 6,
            "zone_id": ["zone_a", "zone_a", "zone_a", "zone_b", "zone_b", "zone_b"],
            "zone_temperature": [23.0, 24.0, 30.0, 22.0, 22.5, 23.0],
            "cooling_power": [100.0, 110.0, 180.0, 90.0, 95.0, 100.0],
            "fan_power": [20.0, 21.0, 30.0, 18.0, 18.5, 19.0],
            "hvac_power": [120.0, 131.0, 210.0, 108.0, 113.5, 119.0],
            "control_action": [0.2, 0.2, 0.9, 0.1, 0.1, 0.15],
            "comfort_violation": [False, False, True, False, False, False],
        }
    )


def test_query_metric_filters_time_and_zone():
    result = query_metric(
        mock_trajectory(),
        metric_name="zone_temperature",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T02:00:00Z",
        zone_id="zone_a",
    )

    assert result["metric_name"] == "zone_temperature"
    assert result["summary"]["count"] == 3
    assert result["summary"]["max"] == 30.0
    assert result["records"][0]["zone_id"] == "zone_a"


def test_compare_period_returns_delta_and_percent_change():
    result = compare_period(
        mock_trajectory(),
        metric_name="cooling_power",
        period_a=("2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"),
        period_b=("2026-01-01T02:00:00Z", "2026-01-01T02:00:00Z"),
        zone_id="zone_a",
    )

    assert result["period_a"]["mean"] == 105.0
    assert result["period_b"]["mean"] == 180.0
    assert result["delta_mean"] == 75.0
    assert round(result["percent_change_mean"], 2) == 71.43


def test_detect_anomaly_finds_threshold_breach():
    result = detect_anomaly(
        mock_trajectory(),
        metric_name="zone_temperature",
        window_size=2,
        threshold=2.0,
        zone_id="zone_a",
    )

    assert result["metric_name"] == "zone_temperature"
    assert len(result["anomalies"]) == 1
    assert result["anomalies"][0]["value"] == 30.0


def test_compute_energy_breakdown_uses_available_power_fields():
    result = compute_energy_breakdown(
        mock_trajectory(),
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T02:00:00Z",
    )

    assert result["components"]["cooling_power"] == 390.0
    assert result["components"]["fan_power"] == 71.0
    assert result["total"] == 461.0
    assert result["notes"] == []


def test_plot_metric_trend_returns_serializable_series():
    result = plot_metric_trend(
        mock_trajectory(),
        metric_name="zone_temperature",
        start_time="2026-01-01T03:00:00Z",
        end_time="2026-01-01T05:00:00Z",
        zone_id="zone_b",
    )

    assert result["chart_type"] == "line"
    assert result["series"][0]["value"] == 22.0
    assert result["series"][-1]["value"] == 23.0


def test_data_quality_check_reports_missing_fields_nulls_and_time_gaps():
    data = mock_trajectory().copy()
    data.loc[1, "zone_temperature"] = None
    data = data.drop(columns=["fan_power"])
    data = data.drop(index=2).reset_index(drop=True)

    result = data_quality_check(
        data,
        required_fields=["timestamp", "zone_id", "zone_temperature", "fan_power"],
        expected_frequency="1h",
    )

    assert result["tool_name"] == "data_quality_check"
    assert result["row_count"] == 5
    assert "fan_power" in result["missing_fields"]
    assert result["null_counts"]["zone_temperature"] == 1
    assert result["time_gap_count"] == 1
    assert result["quality_score"] < 1.0
    assert result["status"] in {"warning", "fail"}


def test_comfort_risk_assessment_summarizes_zone_boundary_risk():
    result = comfort_risk_assessment(
        mock_trajectory(),
        temperature_metric="zone_temperature",
        comfort_lower_bound=22.0,
        comfort_upper_bound=26.0,
    )

    assert result["tool_name"] == "comfort_risk_assessment"
    assert result["risk_level"] == "high"
    assert result["violation_count"] == 1
    assert result["worst_zone_id"] == "zone_a"
    assert result["max_temperature"] == 30.0
    assert result["zone_risks"][0]["zone_id"] == "zone_a"


def test_zone_hotspot_rank_orders_zones_by_metric_and_violations():
    result = zone_hotspot_rank(
        mock_trajectory(),
        metric_name="zone_temperature",
        top_k=2,
    )

    assert result["tool_name"] == "zone_hotspot_rank"
    assert [item["zone_id"] for item in result["ranked_zones"]] == ["zone_a", "zone_b"]
    assert result["ranked_zones"][0]["max"] == 30.0
    assert result["ranked_zones"][0]["comfort_violation_count"] == 1


def test_control_action_audit_detects_large_action_changes():
    result = control_action_audit(
        mock_trajectory(),
        action_metric="control_action",
        change_threshold=0.5,
    )

    assert result["tool_name"] == "control_action_audit"
    assert result["large_change_count"] == 1
    assert result["max_abs_change"] == 0.7
    assert result["stability_status"] == "unstable"
    assert result["events"][0]["zone_id"] == "zone_a"


def test_cooling_efficiency_summary_flags_high_power_with_risk():
    result = cooling_efficiency_summary(
        mock_trajectory(),
        power_metrics=["cooling_power", "fan_power"],
        temperature_metric="zone_temperature",
        comfort_upper_bound=26.0,
    )

    assert result["tool_name"] == "cooling_efficiency_summary"
    assert result["total_power"] == 801.5
    assert result["comfort_violation_count"] == 1
    assert result["power_per_violation"] == 801.5
    assert result["efficiency_status"] in {"needs_attention", "critical"}

