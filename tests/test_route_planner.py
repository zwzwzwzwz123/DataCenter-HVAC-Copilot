from __future__ import annotations

import json

from src.agent.planner import (
    DeterministicRoutePlanner,
    LLMRoutePlanner,
    build_route_planner_from_env,
)


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
