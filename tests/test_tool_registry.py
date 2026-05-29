from src.tools.registry import (
    TOOL_REGISTRY,
    build_planner_tool_prompt,
    validate_tool_input,
    tools_for_route,
)


def test_tool_registry_exposes_specs_for_core_hvac_tools():
    expected_tools = {
        "query_metric",
        "compare_period",
        "plot_metric_trend",
        "compute_energy_breakdown",
        "detect_anomaly",
        "data_quality_check",
        "comfort_risk_assessment",
        "zone_hotspot_rank",
        "control_action_audit",
        "cooling_efficiency_summary",
        "policy_runner",
        "rag_retrieval",
    }

    assert expected_tools.issubset(set(TOOL_REGISTRY))
    assert TOOL_REGISTRY["data_quality_check"].risk_level == "read_only"
    assert TOOL_REGISTRY["comfort_risk_assessment"].route == "anomaly_diagnosis"
    assert "required_fields" in TOOL_REGISTRY["data_quality_check"].input_schema
    assert "risk_level" in TOOL_REGISTRY["comfort_risk_assessment"].output_schema
    assert TOOL_REGISTRY["policy_runner"].risk_level == "control_boundary"


def test_tools_for_route_lists_hvac_operations_without_policy_runner():
    timeseries_tools = {tool.name for tool in tools_for_route("timeseries_query")}
    policy_tools = {tool.name for tool in tools_for_route("policy_recommendation")}

    assert "data_quality_check" in timeseries_tools
    assert "zone_hotspot_rank" in timeseries_tools
    assert "cooling_efficiency_summary" in timeseries_tools
    assert "policy_runner" not in timeseries_tools
    assert "policy_runner" in policy_tools


def test_tool_specs_expose_json_schema_and_validate_inputs():
    spec = TOOL_REGISTRY["zone_hotspot_rank"]

    assert spec.input_json_schema["type"] == "object"
    assert spec.output_json_schema["type"] == "object"
    assert "metric_name" in spec.input_json_schema["properties"]

    validated = validate_tool_input(
        "zone_hotspot_rank",
        {"metric_name": "zone_temperature", "top_k": "2"},
    )

    assert validated == {"metric_name": "zone_temperature", "top_k": 2}


def test_tool_input_validation_rejects_bad_parameters():
    try:
        validate_tool_input("zone_hotspot_rank", {"metric_name": "zone_temperature", "top_k": 0})
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("Expected invalid top_k to be rejected.")


def test_planner_prompt_is_generated_from_tool_specs():
    prompt = build_planner_tool_prompt()

    assert "Allowed routes are exactly" in prompt
    assert "data_quality_check" in prompt
    assert "control_action_audit" in prompt
    assert "policy_runner" in prompt
