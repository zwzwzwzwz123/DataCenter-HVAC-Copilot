from __future__ import annotations

import json

from src.agent.planner import (
    DeterministicRoutePlanner,
    LLMRoutePlanner,
    build_route_planner_from_env,
    _is_supported_time_window,
    _normalize_time_window,
    _system_prompt,
)
from src.tools.registry import TOOL_REGISTRY


class FakePlannerTransport:
    def __init__(self, response: dict | None = None, error: Exception | None = None) -> None:
        self.response = response or {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "steps": [
                                    {
                                        "route": "timeseries_query",
                                        "reason": "Get the current trajectory evidence first.",
                                    },
                                    {
                                        "route": "policy_recommendation",
                                        "reason": "Use policy tool only after evidence is gathered.",
                                    },
                                ],
                                "confidence": 0.9,
                            }
                        )
                    }
                }
            ]
        }
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, url: str, headers: dict[str, str], body: bytes, timeout: float) -> dict:
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "payload": json.loads(body.decode("utf-8")),
                "timeout": timeout,
            }
        )
        if self.error:
            raise self.error
        return self.response


def test_deterministic_planner_preserves_explicit_task_type_as_single_step() -> None:
    planner = DeterministicRoutePlanner()

    decision = planner.plan("This text should not matter.", task_type="timeseries_query")

    assert [step.route for step in decision.steps] == ["timeseries_query"]
    assert decision.steps[0].tool == "query_metric"
    assert decision.steps[0].metric_name == "zone_temperature"
    assert decision.steps[0].zone_id is None
    assert decision.steps[0].time_window == "full_demo_range"
    assert decision.planner == "deterministic"
    assert decision.fallback_used is False
    assert decision.confidence == 1.0


def test_deterministic_planner_creates_bounded_multi_step_plan_with_policy_last() -> None:
    planner = DeterministicRoutePlanner()

    decision = planner.plan(
        "Check the zone_temperature trend, diagnose any anomaly, then recommend a control policy."
    )

    assert [step.route for step in decision.steps] == [
        "timeseries_query",
        "anomaly_diagnosis",
        "policy_recommendation",
    ]
    assert decision.steps[0].tool == "plot_metric_trend"
    assert decision.steps[0].metric_name == "zone_temperature"
    assert decision.steps[0].time_window == "full_demo_range"
    assert decision.steps[1].tool == "detect_anomaly"
    assert decision.steps[1].metric_name == "zone_temperature"
    assert decision.steps[2].tool == "policy_runner"
    assert len(decision.steps) <= 3


def test_deterministic_planner_selects_data_quality_tool() -> None:
    planner = DeterministicRoutePlanner()

    decision = planner.plan("Check data quality, missing fields, and timestamp gaps.")

    assert [step.route for step in decision.steps] == ["timeseries_query"]
    assert decision.steps[0].tool == "data_quality_check"


def test_deterministic_planner_selects_zone_hotspot_tool() -> None:
    planner = DeterministicRoutePlanner()

    decision = planner.plan("Which zone is the hottest hotspot by zone_temperature?")

    assert [step.route for step in decision.steps] == ["timeseries_query"]
    assert decision.steps[0].tool == "zone_hotspot_rank"


def test_deterministic_planner_selects_control_action_audit_tool() -> None:
    planner = DeterministicRoutePlanner()

    decision = planner.plan("Audit whether control_action is oscillating or changing too fast.")

    assert [step.route for step in decision.steps] == ["timeseries_query"]
    assert decision.steps[0].tool == "control_action_audit"
    assert decision.steps[0].metric_name == "control_action"


def test_deterministic_planner_selects_comfort_risk_tool() -> None:
    planner = DeterministicRoutePlanner()

    decision = planner.plan("Assess overheating comfort risk across zones.")

    assert [step.route for step in decision.steps] == ["anomaly_diagnosis"]
    assert decision.steps[0].tool == "comfort_risk_assessment"


def test_deterministic_planner_handles_chinese_multi_intent_questions() -> None:
    planner = DeterministicRoutePlanner()

    decision = planner.plan("最近温度异常升高，并给出控制建议")

    assert [step.route for step in decision.steps] == [
        "timeseries_query",
        "anomaly_diagnosis",
        "policy_recommendation",
    ]


def test_llm_planner_parses_controlled_steps() -> None:
    transport = FakePlannerTransport()
    planner = LLMRoutePlanner(
        provider="deepseek",
        api_key="test-key",
        base_url="https://example.deepseek.test",
        model="planner-test",
        transport=transport,
    )

    decision = planner.plan("Should we adjust the control policy based on latest zone metrics?")

    assert [step.route for step in decision.steps] == [
        "timeseries_query",
        "policy_recommendation",
    ]
    assert decision.steps[0].tool == "query_metric"
    assert decision.steps[0].metric_name == "zone_temperature"
    assert decision.steps[0].time_window == "full_demo_range"
    assert decision.planner == "llm:deepseek:planner-test"
    assert decision.confidence == 0.9
    assert decision.fallback_used is False
    assert transport.calls[0]["url"] == "https://example.deepseek.test/chat/completions"
    assert "policy_recommendation" in transport.calls[0]["payload"]["messages"][0]["content"]


def test_llm_planner_accepts_structured_step_parameters() -> None:
    transport = FakePlannerTransport(
        response={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "steps": [
                                    {
                                        "route": "timeseries_query",
                                        "reason": "Use explicit metric evidence.",
                                        "tool": "compare_period",
                                        "metric_name": "fan_power",
                                        "zone_id": "zone_a",
                                        "time_window": "full_demo_range",
                                    }
                                ],
                                "confidence": 0.82,
                            }
                        )
                    }
                }
            ]
        }
    )
    planner = LLMRoutePlanner(
        provider="deepseek",
        api_key="test-key",
        base_url="https://example.deepseek.test",
        model="planner-test",
        transport=transport,
    )

    decision = planner.plan("Compare fan power for zone_a.")

    assert decision.steps[0].tool == "compare_period"
    assert decision.steps[0].metric_name == "fan_power"
    assert decision.steps[0].zone_id == "zone_a"
    assert decision.steps[0].time_window == "full_demo_range"


def test_llm_planner_falls_back_when_step_tool_is_not_allowed_for_route() -> None:
    transport = FakePlannerTransport(
        response={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "steps": [
                                    {
                                        "route": "timeseries_query",
                                        "reason": "Bad tool choice.",
                                        "tool": "policy_runner",
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
    )
    planner = LLMRoutePlanner(
        provider="deepseek",
        api_key="test-key",
        base_url="https://example.deepseek.test",
        model="planner-test",
        transport=transport,
    )

    decision = planner.plan("Check the latest temperature.")

    assert decision.planner == "deterministic"
    assert decision.fallback_used is True
    assert decision.steps[0].tool == "query_metric"
    assert "unsupported tool" in decision.steps[0].reason


def test_llm_planner_falls_back_when_time_window_is_not_allowed() -> None:
    transport = FakePlannerTransport(
        response={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "steps": [
                                    {
                                        "route": "timeseries_query",
                                        "reason": "Bad time window shape.",
                                        "tool": "query_metric",
                                        "metric_name": "zone_temperature",
                                        "time_window": "last_24",
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
    )
    planner = LLMRoutePlanner(
        provider="deepseek",
        api_key="test-key",
        base_url="https://example.deepseek.test",
        model="planner-test",
        transport=transport,
    )

    decision = planner.plan("Check the latest temperature.")

    assert decision.planner == "deterministic"
    assert decision.fallback_used is True
    assert decision.steps[0].time_window == "full_demo_range"
    assert "unsupported time_window" in decision.steps[0].reason


def test_llm_planner_falls_back_when_policy_step_is_not_last() -> None:
    transport = FakePlannerTransport(
        response={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "steps": [
                                    {"route": "policy_recommendation", "reason": "too early"},
                                    {"route": "timeseries_query", "reason": "late evidence"},
                                ]
                            }
                        )
                    }
                }
            ]
        }
    )
    planner = LLMRoutePlanner(
        provider="deepseek",
        api_key="test-key",
        base_url="https://example.deepseek.test",
        model="planner-test",
        transport=transport,
    )

    decision = planner.plan("Recommend a control policy after checking the latest metrics.")

    assert [step.route for step in decision.steps] == [
        "timeseries_query",
        "policy_recommendation",
    ]
    assert decision.planner == "deterministic"
    assert decision.fallback_used is True
    assert "policy_recommendation must be the final step" in decision.steps[0].reason


def test_build_route_planner_from_env_defaults_to_deterministic_without_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LANGGRAPH_PLANNER_PROVIDER", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    planner = build_route_planner_from_env(project_root=tmp_path)
    decision = planner.plan("Recommend a control policy.")

    assert decision.planner == "deterministic"
    assert [step.route for step in decision.steps] == ["policy_recommendation"]


def test_llm_planner_prompt_includes_conversation_context() -> None:
    transport = FakePlannerTransport()
    planner = LLMRoutePlanner(
        provider="deepseek",
        api_key="test-key",
        base_url="https://example.deepseek.test",
        model="planner-test",
        transport=transport,
    )
    context = {
        "stable_context": {"boundary_summary": "BEAR is simulation only."},
        "recent_turns": [{"question": "previous", "answer": "zone_a peaked"}],
        "relevant_memory": [{"text": "zone_a peaked at 30 C"}],
    }

    planner.plan("What about that zone?", conversation_context=context)

    user_payload = json.loads(transport.calls[0]["payload"]["messages"][1]["content"])
    assert user_payload["conversation_context"] == context
    assert "current fresh evidence" in transport.calls[0]["payload"]["messages"][0]["content"]


def test_system_prompt_includes_tools_from_registry() -> None:
    prompt = _system_prompt()

    for tool_name in ["data_quality_check", "comfort_risk_assessment", "policy_runner"]:
        assert tool_name in TOOL_REGISTRY
        assert tool_name in prompt


def test_normalize_time_window_passes_through_canonical_values() -> None:
    for value in ["last_24_hours", "last_30_minutes", "full_demo_range", "latest", "recent"]:
        assert _normalize_time_window(value) == value


def test_normalize_time_window_folds_natural_language_into_hours() -> None:
    # Days/weeks/months fold into hours (matching the gold vocabulary, which
    # never uses larger units) so train/inference formats stay consistent.
    assert _normalize_time_window("past 7 days") == "last_168_hours"
    assert _normalize_time_window("last 2 hours") == "last_2_hours"
    assert _normalize_time_window("last 30 minutes") == "last_30_minutes"
    assert _normalize_time_window("last month") == "last_720_hours"
    assert _normalize_time_window("last 3 months") == "last_2160_hours"
    assert _normalize_time_window("last two weeks") == "last_336_hours"
    assert _normalize_time_window("today") == "last_24_hours"
    assert _normalize_time_window("all data") == "full_demo_range"


def test_normalize_time_window_handles_compact_and_range_forms() -> None:
    assert _normalize_time_window("7d") == "last_168_hours"
    assert _normalize_time_window("last_24h") == "last_24_hours"
    assert _normalize_time_window("last_48h") == "last_48_hours"
    assert _normalize_time_window("now-12h to now") == "last_12_hours"


def test_normalize_time_window_leaves_unmappable_values_for_the_guard() -> None:
    # Episode IDs, structured dicts, and truly ambiguous phrases must NOT be
    # coerced into a valid window. They pass through unchanged so the guard
    # still rejects them (forcing a deterministic fallback) rather than being
    # silently rescued into a wrong value.
    assert _normalize_time_window(None) is None
    for value in ["episode_001", "garbage nonsense", "last_24", "weekends vs weekdays"]:
        result = _normalize_time_window(value)
        assert result == value
        assert not _is_supported_time_window(result)


def test_llm_planner_normalizes_natural_language_time_window() -> None:
    # A window the model naturally emits ("past 7 days") used to be rejected by
    # the guard and forced a deterministic fallback; it should now normalize and
    # keep the LLM plan.
    transport = FakePlannerTransport(
        response={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "steps": [
                                    {
                                        "route": "timeseries_query",
                                        "reason": "Look at the recent trend.",
                                        "tool": "query_metric",
                                        "metric_name": "zone_temperature",
                                        "time_window": "past 7 days",
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
    )
    planner = LLMRoutePlanner(
        provider="deepseek",
        api_key="test-key",
        base_url="https://example.deepseek.test",
        model="planner-test",
        transport=transport,
    )

    decision = planner.plan("Show the zone_temperature trend over the past week.")

    assert decision.fallback_used is False
    assert decision.planner == "llm:deepseek:planner-test"
    assert decision.steps[0].time_window == "last_168_hours"

