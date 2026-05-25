import subprocess
import sys
from pathlib import Path

import pytest

from app.api_client import ApiClientError, ask_api, run_eval_api
from app.streamlit_app import (
    CONSOLE_CSS,
    DEMO_WALKTHROUGHS,
    MOUSE_GLOW_SCRIPT,
    WORKFLOW_OPTIONS,
    build_sidebar_config_groups,
    build_workflow_trace_rows,
    get_dashboard_copy,
    get_default_api_base_url,
    build_status_cards,
    build_execution_timeline,
    build_safety_audit_rows,
    build_prediction_preview,
    group_eval_metrics,
    render_thinking_indicator_html,
    render_status_grid_html,
    render_sidebar_config_group_html,
)


class FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = "error text"

    def json(self) -> dict:
        return self._payload


class FakeHttpClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.last_url = None
        self.last_json = None
        self.last_timeout = None

    def post(self, url: str, json: dict, timeout: float) -> FakeResponse:
        self.last_url = url
        self.last_json = json
        self.last_timeout = timeout
        return self.response


def test_ask_api_posts_question_and_task_type():
    http_client = FakeHttpClient(
        FakeResponse(
            200,
            {
                "question": "q",
                "route": "document_qa",
                "answer": "a",
                "tools": [],
                "citations": [],
                "retrieved_contexts": [],
                "tool_results": [],
            },
        )
    )

    result = ask_api(
        "http://localhost:8000",
        question="q",
        task_type="document_qa",
        http_client=http_client,
    )

    assert http_client.last_url == "http://localhost:8000/ask"
    assert http_client.last_json == {
        "question": "q",
        "task_type": "document_qa",
        "workflow_engine": "langgraph",
    }
    assert result["answer"] == "a"


def test_ask_api_posts_selected_workflow_engine():
    http_client = FakeHttpClient(FakeResponse(200, {"answer": "a"}))

    ask_api(
        "http://localhost:8000",
        question="q",
        task_type=None,
        workflow_engine="langgraph",
        http_client=http_client,
    )

    assert http_client.last_json == {
        "question": "q",
        "task_type": None,
        "workflow_engine": "langgraph",
    }


def test_ask_api_posts_session_id_and_memory_flag():
    http_client = FakeHttpClient(FakeResponse(200, {"answer": "a"}))

    ask_api(
        "http://localhost:8000",
        question="q",
        session_id="session-1",
        memory_enabled=True,
        http_client=http_client,
    )

    assert http_client.last_json == {
        "question": "q",
        "task_type": None,
        "workflow_engine": "langgraph",
        "session_id": "session-1",
        "memory_enabled": True,
    }


def test_ask_api_raises_on_non_200_response():
    http_client = FakeHttpClient(FakeResponse(500, {"detail": "broken"}))

    with pytest.raises(ApiClientError, match="API request failed"):
        ask_api("http://localhost:8000", question="q", http_client=http_client)


def test_run_eval_api_posts_eval_path():
    http_client = FakeHttpClient(
        FakeResponse(
            200,
            {
                "metrics": {"citation_hit_rate": 0.6},
                "predictions": [{"id": "doc_qa_001"}],
            },
        )
    )

    result = run_eval_api(
        "http://localhost:8000/",
        eval_path="data/eval/hvac_eval.jsonl",
        http_client=http_client,
    )

    assert http_client.last_url == "http://localhost:8000/eval/run"
    assert http_client.last_json == {"eval_path": "data/eval/hvac_eval.jsonl"}
    assert result["metrics"]["citation_hit_rate"] == 0.6


def test_default_api_base_url_can_be_overridden_for_docker(monkeypatch):
    monkeypatch.setenv("HVAC_COPILOT_API_BASE_URL", "http://api:8000")

    assert get_default_api_base_url() == "http://api:8000"


def test_streamlit_script_imports_when_run_by_path(tmp_path):
    script_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    command = [
        sys.executable,
        "-c",
        (
            "import runpy; "
            f"runpy.run_path(r'{script_path}', run_name='streamlit_smoke')"
        ),
    ]

    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_group_eval_metrics_splits_quality_proxy_metrics():
    grouped = group_eval_metrics(
        {
            "citation_hit_rate": 0.6,
            "context_recall": 0.7,
            "expected_keyword_coverage": 0.5,
            "lexical_answer_coverage": 0.2,
            "tool_selection_accuracy": 1.0,
            "tool_execution_success_rate": 1.0,
            "evidence_coverage": 0.8,
            "planned_step_accuracy": 0.9,
            "planned_step_order_accuracy": 0.8,
            "required_step_recall": 0.95,
            "policy_final_step_rate": 1.0,
            "answer_correctness_proxy": 0.4,
            "faithfulness_proxy": 0.3,
        }
    )

    assert [name for name, _ in grouped["Retrieval"]] == [
        "citation_hit_rate",
        "context_recall",
    ]
    assert [name for name, _ in grouped["Quality Proxy"]] == [
        "answer_correctness_proxy",
        "faithfulness_proxy",
    ]
    assert [name for name, _ in grouped["Planner"]] == [
        "planned_step_accuracy",
        "planned_step_order_accuracy",
        "required_step_recall",
        "policy_final_step_rate",
    ]


def test_group_eval_metrics_omits_planner_group_when_metrics_absent():
    grouped = group_eval_metrics(
        {
            "citation_hit_rate": 0.6,
            "context_recall": 0.7,
            "tool_selection_accuracy": 1.0,
        }
    )

    assert "Planner" not in grouped


def test_build_prediction_preview_adds_evidence_flags_and_answer_length():
    preview = build_prediction_preview(
        [
            {
                "id": "doc_qa_001",
                "task_type": "document_qa",
                "route": "rag",
                "tools": [],
                "answer": "带引用的回答",
                "citations": [{"source_id": "doc"}],
                "tool_results": [],
                "answer_audit": {"passed": True, "violations": []},
            },
            {
                "id": "ts_query_001",
                "task_type": "timeseries_query",
                "route": "timeseries_query",
                "tools": ["query_metric"],
                "answer": "",
                "citations": [],
                "tool_results": [{"summary": {"count": 3}}],
                "answer_audit": {
                    "passed": False,
                    "violations": ["production_telemetry_claim"],
                },
            },
        ]
    )

    assert preview[0]["has_citation"] is True
    assert preview[0]["has_tool_result"] is False
    assert preview[0]["answer_length"] == len("带引用的回答")
    assert preview[0]["audit_passed"] is True
    assert preview[1]["has_citation"] is False
    assert preview[1]["has_tool_result"] is True
    assert preview[1]["audit_violations"] == "production_telemetry_claim"


def test_demo_walkthroughs_cover_core_routes():
    task_types = {case["task_type"] for case in DEMO_WALKTHROUGHS}

    assert {"document_qa", "timeseries_query", "policy_recommendation"}.issubset(
        task_types
    )
    assert all(case["question"] for case in DEMO_WALKTHROUGHS)
    assert all(case["why"] for case in DEMO_WALKTHROUGHS)


def test_workflow_options_offer_baseline_and_langgraph():
    assert list(WORKFLOW_OPTIONS.values())[0] == "langgraph"
    assert WORKFLOW_OPTIONS["LangGraph workflow"] == "langgraph"
    assert WORKFLOW_OPTIONS["Deterministic baseline"] == "deterministic"


def test_build_sidebar_config_groups_summarizes_system_setup():
    groups = build_sidebar_config_groups(
        api_base_url="http://localhost:8000",
        workflow_label="LangGraph workflow",
        last_result={
            "answer_generator": "deepseek:deepseek-chat",
            "data_source": {"kind": "processed_csv"},
            "workflow_trace": [{"planner": "llm:deepseek:deepseek-chat"}],
            "tools": ["dropt_guided_diffno_checkpoint"],
        },
    )

    assert [group["title"] for group in groups] == [
        "Connection",
        "Model Runtime",
        "Data & Policy",
        "Evaluation",
    ]
    assert groups[0]["items"][0] == {
        "label": "Endpoint",
        "value": "localhost:8000",
        "tone": "neutral",
    }
    assert {"label": "Workflow", "value": "LangGraph workflow", "tone": "success"} in groups[1]["items"]
    assert {"label": "Generator", "value": "deepseek:deepseek-chat", "tone": "success"} in groups[1]["items"]
    assert {"label": "Data", "value": "processed_csv", "tone": "success"} in groups[2]["items"]


def test_render_sidebar_config_group_html_uses_compact_config_blocks():
    html = render_sidebar_config_group_html(
        {
            "title": "Model Runtime",
            "items": [
                {"label": "Workflow", "value": "LangGraph workflow", "tone": "success"},
                {"label": "Planner", "value": "deterministic", "tone": "warning"},
            ],
        }
    )

    assert 'class="sidebar-config-group"' in html
    assert 'class="sidebar-config-title"' in html
    assert 'class="config-dot success"' in html
    assert 'class="config-dot warning"' in html
    assert "LangGraph workflow" in html


def test_render_thinking_indicator_html_shows_model_working_state():
    html = render_thinking_indicator_html("Analyzing evidence")

    assert 'class="thinking-panel"' in html
    assert 'class="thinking-orbit"' in html
    assert "Analyzing evidence" in html
    assert "模型正在组织证据" in html


def test_build_workflow_trace_rows_summarizes_langgraph_nodes():
    rows = build_workflow_trace_rows(
        {
            "workflow_engine": "langgraph",
            "workflow_trace": [
                {
                    "node": "planner",
                    "route": "policy_recommendation",
                    "planner": "llm:deepseek:planner-test",
                    "planned_steps": ["timeseries_query", "policy_recommendation"],
                    "planned_step_specs": [
                        {
                            "route": "timeseries_query",
                            "tool": "query_metric",
                            "metric_name": "zone_temperature",
                            "time_window": "full_demo_range",
                        },
                        {
                            "route": "policy_recommendation",
                            "tool": "policy_runner",
                        },
                    ],
                    "confidence": 0.88,
                    "fallback_used": False,
                    "tools": [],
                    "citation_count": 0,
                    "tool_result_count": 0,
                    "audit_passed": None,
                },
                {
                    "node": "answer_generator",
                    "route": "policy_recommendation",
                    "answer_generator": "deterministic_grounded",
                    "tools": [],
                    "citation_count": 0,
                    "tool_result_count": 1,
                    "audit_passed": None,
                },
                {
                    "node": "answer_audit",
                    "route": "policy_recommendation",
                    "tools": ["rule_based_policy"],
                    "citation_count": 0,
                    "tool_result_count": 1,
                    "audit_passed": True,
                },
            ],
        }
    )

    assert rows == [
        {
            "step": 1,
            "node": "planner",
            "route": "policy_recommendation",
            "classifier": "llm:deepseek:planner-test",
            "fallback": "no",
            "tools": "query_metric, policy_runner",
            "evidence": "0 citations / 0 tool results",
            "audit": "n/a",
        },
        {
            "step": 2,
            "node": "answer_generator",
            "route": "policy_recommendation",
            "classifier": "n/a",
            "fallback": "n/a",
            "tools": "none",
            "evidence": "0 citations / 1 tool results",
            "audit": "n/a",
        },
        {
            "step": 3,
            "node": "answer_audit",
            "route": "policy_recommendation",
            "classifier": "n/a",
            "fallback": "n/a",
            "tools": "rule_based_policy",
            "evidence": "0 citations / 1 tool results",
            "audit": "passed",
        },
    ]


def test_dashboard_copy_uses_control_console_positioning():
    copy = get_dashboard_copy()

    assert copy["title"] == "DataCenter-HVAC Copilot"
    assert "RAG + Tool Agent" in copy["subtitle"]
    assert "HVAC 仿真" in copy["boundary"]
    assert "LLM 不直接控制" in copy["boundary"]


def test_build_status_cards_summarizes_runtime_evidence():
    cards = build_status_cards(
        {
            "route": "policy_recommendation",
            "tools": ["rule_based_policy"],
            "answer_generator": "deterministic_grounded",
            "data_source": {"kind": "bear_sample_csv"},
            "answer_audit": {"passed": True},
            "citations": [{"source_id": "doc"}],
            "tool_results": [{"policy_name": "rule_based"}],
        }
    )

    assert [card["label"] for card in cards] == [
        "Route",
        "Policy / Tools",
        "Generator",
        "Evidence",
        "Audit",
        "Data Source",
    ]
    assert cards[0]["value"] == "policy_recommendation"
    assert cards[1]["value"] == "rule_based_policy"
    assert cards[3]["value"] == "1 citations / 1 tool results"
    assert cards[4]["value"] == "passed"
    assert cards[5]["value"] == "bear_sample_csv"


def test_render_status_grid_html_does_not_emit_indented_code_blocks():
    html = render_status_grid_html(
        [
            {"label": "Route", "value": "waiting", "hint": "ready"},
            {"label": "Tools", "value": "standby", "hint": "ready"},
        ]
    )

    assert '<div class="status-grid">' in html
    assert "\n    <div" not in html
    assert html.count('class="status-card"') == 2


def test_console_css_keeps_minimal_saas_visual_language():
    forbidden_fragments = [
        "text-transform: uppercase",
        "font-weight: 700",
        "font-weight: 800",
    ]

    for fragment in forbidden_fragments:
        assert fragment not in CONSOLE_CSS

    assert CONSOLE_CSS.count("linear-gradient") == 3
    assert CONSOLE_CSS.count("radial-gradient") == 5
    assert "box-shadow: var(--shadow-card);" in CONSOLE_CSS


def test_console_css_removes_default_streamlit_top_gap():
    assert 'header[data-testid="stHeader"]' in CONSOLE_CSS
    assert 'div[data-testid="stDecoration"]' in CONSOLE_CSS
    assert "padding-top: 0.35rem;" in CONSOLE_CSS


def test_console_css_uses_compact_hero_layout():
    expected_fragments = [
        "padding-top: 0.35rem;",
        "margin-bottom: 1.15rem;",
        "font-size: clamp(1.95rem, 3.1vw, 2.75rem);",
        "margin: 0 0 0.52rem 0;",
        "padding: 0.56rem 0.82rem;",
        "margin-bottom: 1.35rem;",
    ]

    for fragment in expected_fragments:
        assert fragment in CONSOLE_CSS


def test_console_css_uses_light_premium_visual_theme():
    expected_fragments = [
        "--bg:           #f7f7f5;",
        "--bg-panel:     #ffffff;",
        "--accent:       #10a37f;",
        "--shadow-card:",
        "background: var(--bg-panel);",
        "border: 1px solid var(--border-panel);",
        "box-shadow: var(--shadow-card);",
    ]

    for fragment in expected_fragments:
        assert fragment in CONSOLE_CSS


def test_console_css_uses_refined_product_typography():
    expected_fragments = [
        "Noto+Sans+SC:wght@400;500;600",
        "Playfair+Display:wght@500;600",
        "Space+Grotesk:wght@400;500;600",
        "--font-sans:",
        "--font-display:",
        "--font-accent:",
        "--font-numeric:",
        "--font-mono:",
        "font-family: var(--font-sans) !important;",
        "font-family: var(--font-display) !important;",
        "font-family: var(--font-accent) !important;",
        "font-family: var(--font-numeric) !important;",
        "line-height: 1.68;",
        "letter-spacing: -0.035em;",
    ]

    for fragment in expected_fragments:
        assert fragment in CONSOLE_CSS


def test_console_css_adds_subtle_ambient_background_motion():
    expected_fragments = [
        "--mouse-x:",
        "--mouse-y:",
        ".stApp::before",
        ".stApp::after",
        "animation: ambient-breathe",
        "@keyframes ambient-breathe",
        "@media (prefers-reduced-motion: reduce)",
    ]

    for fragment in expected_fragments:
        assert fragment in CONSOLE_CSS


def test_console_css_adds_claude_like_thinking_indicator():
    expected_fragments = [
        ".thinking-panel",
        ".thinking-orbit",
        "thinking-spin 1.08s",
        "thinking-pulse 2.4s",
        "@keyframes thinking-spin",
        "@keyframes thinking-pulse",
        "running indicator",
        'div[data-testid="stStatusWidget"]',
        'div[data-testid="stToolbar"]',
    ]

    for fragment in expected_fragments:
        assert fragment in CONSOLE_CSS


def test_mouse_glow_script_updates_streamlit_app_css_variables():
    expected_fragments = [
        "mousemove",
        "--mouse-x",
        "--mouse-y",
        "requestAnimationFrame",
        ".stApp",
    ]

    for fragment in expected_fragments:
        assert fragment in MOUSE_GLOW_SCRIPT


def test_console_css_spaces_stacked_cards_in_empty_state():
    assert "margin-bottom: 1.25rem;" in CONSOLE_CSS


def test_console_css_gives_primary_buttons_visible_boundaries():
    expected_fragments = [
        "button[data-testid=\"stBaseButton-primary\"]",
        "background: rgba(255, 255, 255, 0.72);",
        "color: var(--text);",
        "border: 0;",
        "box-shadow: inset 0 0 0 1px var(--border-panel), var(--shadow-card);",
        "background: var(--accent-soft);",
        "color: #0f6f58;",
    ]

    for fragment in expected_fragments:
        assert fragment in CONSOLE_CSS


def test_console_css_prevents_button_bottom_clipping():
    expected_fragments = [
        ".stButton {",
        "margin-bottom: 1.1rem;",
        "height: 3rem;",
        "box-sizing: border-box;",
        "display: inline-flex;",
        "align-items: center;",
        "justify-content: center;",
    ]

    for fragment in expected_fragments:
        assert fragment in CONSOLE_CSS


def test_build_execution_timeline_summarizes_route_tools_generator_and_data_source():
    timeline = build_execution_timeline(
        {
            "route": "timeseries_query",
            "route_reason": "explicit task type",
            "tools": ["query_metric"],
            "citations": [{"source_id": "doc"}],
            "retrieved_contexts": [{"source_id": "doc"}],
            "tool_results": [{"summary": {"max": 30.0}}],
            "answer_generator": "deepseek:deepseek-v4-flash",
            "data_source": {
                "kind": "bear_sample_csv",
                "path": "BEAR/BEAR/Data/Exercise2A-mytest.csv",
            },
        }
    )

    assert [item["stage"] for item in timeline] == [
        "Route",
        "Retrieval",
        "Tool Call",
        "Answer Generator",
        "Data Boundary",
    ]
    assert timeline[0]["status"] == "timeseries_query"
    assert "query_metric" in timeline[2]["detail"]
    assert "deepseek:deepseek-v4-flash" in timeline[3]["detail"]
    assert "HVAC 仿真" in timeline[4]["detail"]


def test_build_safety_audit_rows_exposes_passed_and_violations():
    rows = build_safety_audit_rows(
        {
            "answer_audit": {
                "passed": False,
                "violations": ["production_telemetry_claim"],
                "checks": [
                    "production_telemetry_claim",
                    "llm_direct_control_claim",
                ],
            }
        }
    )

    assert rows[0]["check"] == "production_telemetry_claim"
    assert rows[0]["status"] == "violation"
    assert rows[1]["status"] == "passed"
