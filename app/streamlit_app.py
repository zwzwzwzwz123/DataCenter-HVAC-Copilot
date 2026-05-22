from __future__ import annotations

from html import escape
import os
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api_client import ApiClientError, ask_api, run_eval_api


TASK_OPTIONS = {
    "自动判断": None,
    "文档问答": "document_qa",
    "时序查询": "timeseries_query",
    "异常诊断": "anomaly_diagnosis",
    "策略建议": "policy_recommendation",
}

WORKFLOW_OPTIONS = {
    "LangGraph workflow": "langgraph",
    "Deterministic baseline": "deterministic",
}

DEMO_WALKTHROUGHS = [
    {
        "label": "BEAR 数据边界",
        "task_type": "document_qa",
        "question": "BEAR 轨迹在本项目中应如何表述，为什么不能说成真实生产数据？",
        "why": "展示 RAG 引用、数据边界和防幻觉约束。",
    },
    {
        "label": "温度时序查询",
        "task_type": "timeseries_query",
        "question": "episode_001 中 zone_a 在最近 3 小时的温度最大值是多少？",
        "why": "展示 router 如何选择时序工具并返回结构化 metric summary。",
    },
    {
        "label": "策略建议边界",
        "task_type": "policy_recommendation",
        "question": "如果当前温度超过舒适上限，是否应该调整控制策略？",
        "why": "展示控制建议来自 policy 工具，LLM 只负责解释。",
    },
]

METRIC_GROUPS = {
    "Retrieval": ["citation_hit_rate", "context_recall"],
    "Answer": ["expected_keyword_coverage", "lexical_answer_coverage"],
    "Tool": [
        "tool_selection_accuracy",
        "tool_execution_success_rate",
        "evidence_coverage",
    ],
    "Quality Proxy": ["answer_correctness_proxy", "faithfulness_proxy"],
}


def get_default_api_base_url() -> str:
    return os.environ.get("HVAC_COPILOT_API_BASE_URL", "http://localhost:8000")

DASHBOARD_COPY = {
    "title": "DataCenter-HVAC Copilot",
    "subtitle": "RAG + Tool Agent control console for BEAR HVAC simulation evidence.",
    "boundary": "BEAR 仅作为 HVAC 仿真 / 可控代理场景；LLM 不直接控制，策略动作只来自 policy tool。",
}

CONSOLE_CSS = """
<style>
    :root {
        --console-bg: #0b1117;
        --console-panel: #101a24;
        --console-panel-2: #142230;
        --console-border: #263847;
        --console-text: #e6edf3;
        --console-muted: #94a3b8;
        --console-cyan: #37d5ff;
        --console-green: #68e391;
        --console-orange: #ffb454;
        --console-red: #ff5f6d;
    }

    .stApp {
        background:
            radial-gradient(circle at 18% 12%, rgba(55, 213, 255, 0.10), transparent 30%),
            linear-gradient(135deg, #091017 0%, #0b1117 46%, #111827 100%);
        color: var(--console-text);
    }

    section[data-testid="stSidebar"] {
        background: #091017;
        border-right: 1px solid var(--console-border);
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--console-muted);
    }

    .block-container {
        max-width: 1420px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4, p, label, span, div {
        letter-spacing: 0;
    }

    div[data-testid="stTabs"] button {
        color: var(--console-muted);
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--console-green);
        border-bottom-color: var(--console-green);
    }

    .stButton > button {
        width: 100%;
        border: 1px solid rgba(104, 227, 145, 0.45);
        background: linear-gradient(90deg, #1db16a, #37d5ff);
        color: #061015;
        font-weight: 700;
        border-radius: 8px;
        min-height: 2.75rem;
    }

    .stTextArea textarea,
    .stTextInput input,
    div[data-baseweb="select"] > div {
        background-color: #0e1822;
        border: 1px solid var(--console-border);
        color: var(--console-text);
        border-radius: 8px;
    }

    .console-hero {
        border: 1px solid var(--console-border);
        border-radius: 8px;
        background: linear-gradient(135deg, rgba(16, 26, 36, 0.98), rgba(20, 34, 48, 0.92));
        padding: 1.4rem 1.5rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.25);
    }

    .console-kicker {
        color: var(--console-green);
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .console-title {
        color: var(--console-text);
        font-size: clamp(2rem, 4vw, 3.5rem);
        line-height: 1;
        font-weight: 800;
        margin: 0;
    }

    .console-subtitle {
        color: var(--console-muted);
        font-size: 0.98rem;
        margin-top: 0.75rem;
        margin-bottom: 0;
        max-width: 920px;
    }

    .boundary-strip {
        border-left: 4px solid var(--console-orange);
        background: rgba(255, 180, 84, 0.10);
        color: #ffd89b;
        padding: 0.75rem 0.9rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-size: 0.92rem;
    }

    .panel {
        border: 1px solid var(--console-border);
        background: rgba(16, 26, 36, 0.88);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .panel-title {
        color: var(--console-text);
        font-weight: 750;
        font-size: 0.95rem;
        margin-bottom: 0.75rem;
    }

    .answer-panel {
        border: 1px solid rgba(104, 227, 145, 0.30);
        background: rgba(9, 16, 23, 0.74);
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }

    .answer-panel h3 {
        color: var(--console-green);
        margin-top: 0;
        margin-bottom: 0.75rem;
        font-size: 1rem;
    }

    .status-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.9rem 0 1rem;
    }

    .status-card {
        border: 1px solid var(--console-border);
        background: rgba(20, 34, 48, 0.86);
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
        min-height: 88px;
    }

    .status-card .label {
        color: var(--console-muted);
        font-size: 0.72rem;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .status-card .value {
        color: var(--console-text);
        font-size: 1.05rem;
        font-weight: 760;
        overflow-wrap: anywhere;
    }

    .status-card .hint {
        color: var(--console-muted);
        font-size: 0.74rem;
        margin-top: 0.25rem;
        overflow-wrap: anywhere;
    }

    .section-label {
        color: var(--console-cyan);
        font-size: 0.82rem;
        font-weight: 760;
        text-transform: uppercase;
        margin: 1rem 0 0.45rem;
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--console-border);
        border-radius: 8px;
        background: rgba(16, 26, 36, 0.70);
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--console-border);
        border-radius: 8px;
    }

    code {
        color: var(--console-green);
        background: rgba(104, 227, 145, 0.10);
        border-radius: 4px;
    }

    @media (max-width: 900px) {
        .status-grid {
            grid-template-columns: 1fr;
        }
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
</style>
"""


def get_dashboard_copy() -> dict[str, str]:
    return DASHBOARD_COPY.copy()


def build_status_cards(result: dict) -> list[dict[str, str]]:
    tools = result.get("tools", [])
    citations = result.get("citations", [])
    tool_results = result.get("tool_results", [])
    audit = result.get("answer_audit") or {}
    data_source = result.get("data_source") or {}
    return [
        {
            "label": "Route",
            "value": str(result.get("route", "unknown")),
            "hint": str(result.get("route_reason") or "deterministic router"),
        },
        {
            "label": "Policy / Tools",
            "value": ", ".join(tools) if tools else "none",
            "hint": "工具输出动作，LLM 只解释证据",
        },
        {
            "label": "Generator",
            "value": str(result.get("answer_generator", "unknown")),
            "hint": "evidence-grounded final answer",
        },
        {
            "label": "Evidence",
            "value": f"{len(citations)} citations / {len(tool_results)} tool results",
            "hint": "引用与结构化工具结果",
        },
        {
            "label": "Audit",
            "value": "passed" if audit.get("passed") else "review",
            "hint": ", ".join(audit.get("violations", [])) or "boundary checks clear",
        },
        {
            "label": "Data Source",
            "value": str(data_source.get("kind", "unknown")),
            "hint": str(data_source.get("path", "")),
        },
    ]


def group_eval_metrics(metrics: dict) -> dict[str, list[tuple[str, float]]]:
    grouped: dict[str, list[tuple[str, float]]] = {}
    for group_name, metric_names in METRIC_GROUPS.items():
        values = []
        for metric_name in metric_names:
            if metric_name in metrics:
                values.append((metric_name, float(metrics[metric_name])))
        if values:
            grouped[group_name] = values
    return grouped


def build_prediction_preview(predictions: list[dict]) -> list[dict]:
    preview = []
    for prediction in predictions:
        citations = prediction.get("citations", [])
        tool_results = prediction.get("tool_results", [])
        answer_audit = prediction.get("answer_audit") or {}
        answer = str(prediction.get("answer") or "")
        preview.append(
            {
                "id": prediction.get("id"),
                "task_type": prediction.get("task_type"),
                "route": prediction.get("route"),
                "tools": ", ".join(prediction.get("tools", [])),
                "citation_count": len(citations),
                "tool_result_count": len(tool_results),
                "has_citation": bool(citations),
                "has_tool_result": bool(tool_results),
                "answer_length": len(answer),
                "audit_passed": answer_audit.get("passed"),
                "audit_violations": ", ".join(answer_audit.get("violations", [])),
            }
        )
    return preview


def build_execution_timeline(result: dict) -> list[dict]:
    citations = result.get("citations", [])
    contexts = result.get("retrieved_contexts", [])
    tools = result.get("tools", [])
    tool_results = result.get("tool_results", [])
    data_source = result.get("data_source", {})
    return [
        {
            "stage": "Route",
            "status": result.get("route", "unknown"),
            "detail": result.get("route_reason") or "No route reason returned.",
        },
        {
            "stage": "Retrieval",
            "status": f"{len(citations)} citations / {len(contexts)} contexts",
            "detail": _summarize_citations(citations),
        },
        {
            "stage": "Tool Call",
            "status": f"{len(tool_results)} results",
            "detail": ", ".join(tools) if tools else "No tool call required.",
        },
        {
            "stage": "Answer Generator",
            "status": result.get("answer_generator", "unknown"),
            "detail": f"Final answer generated by {result.get('answer_generator', 'unknown')}.",
        },
        {
            "stage": "Data Boundary",
            "status": data_source.get("kind", "unknown"),
            "detail": (
                f"{data_source.get('path', '')} | 当前轨迹仅用于 HVAC 仿真 / 可控代理场景，"
                "不能表述为真实数据中心生产遥测。"
            ),
        },
    ]


def build_workflow_trace_rows(result: dict) -> list[dict]:
    rows = []
    for index, item in enumerate(result.get("workflow_trace", []), start=1):
        tools = item.get("tools") or []
        citation_count = int(item.get("citation_count", 0) or 0)
        tool_result_count = int(item.get("tool_result_count", 0) or 0)
        audit_value = item.get("audit_passed", item.get("passed"))
        rows.append(
            {
                "step": index,
                "node": item.get("node", "unknown"),
                "route": item.get("route", result.get("route", "unknown")),
                "classifier": str(item.get("classifier", "n/a")),
                "fallback": _format_fallback_status(item.get("fallback_used")),
                "tools": ", ".join(str(tool) for tool in tools) if tools else "none",
                "evidence": f"{citation_count} citations / {tool_result_count} tool results",
                "audit": _format_audit_status(audit_value),
            }
        )
    return rows


def _format_audit_status(value: object) -> str:
    if value is True:
        return "passed"
    if value is False:
        return "review"
    return "n/a"


def _format_fallback_status(value: object) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "n/a"


def build_safety_audit_rows(result: dict) -> list[dict]:
    audit = result.get("answer_audit") or {}
    violations = set(audit.get("violations", []))
    checks = audit.get("checks", [])
    return [
        {
            "check": check,
            "status": "violation" if check in violations else "passed",
        }
        for check in checks
    ]


def _summarize_citations(citations: list[dict]) -> str:
    if not citations:
        return "No document citation returned."
    source_ids = [str(citation.get("source_id", "unknown")) for citation in citations[:3]]
    return ", ".join(source_ids)


def _render_console_shell() -> None:
    copy = get_dashboard_copy()
    st.markdown(CONSOLE_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="console-hero">
            <div class="console-kicker">HVAC Simulation Agent Console</div>
            <h1 class="console-title">{escape(copy["title"])}</h1>
            <p class="console-subtitle">{escape(copy["subtitle"])}</p>
        </div>
        <div class="boundary-strip">{escape(copy["boundary"])}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_panel_start(title: str) -> None:
    st.markdown(
        f'<div class="panel-title">{escape(title)}</div>',
        unsafe_allow_html=True,
    )


def _render_status_grid(cards: list[dict[str, str]]) -> None:
    st.markdown(render_status_grid_html(cards), unsafe_allow_html=True)


def render_status_grid_html(cards: list[dict[str, str]]) -> str:
    card_html = []
    for card in cards:
        card_html.append(
            '<div class="status-card">'
            f'<div class="label">{escape(card["label"])}</div>'
            f'<div class="value">{escape(card["value"])}</div>'
            f'<div class="hint">{escape(card["hint"])}</div>'
            "</div>"
        )
    return '<div class="status-grid">' + "".join(card_html) + "</div>"


def _render_section_label(label: str) -> None:
    st.markdown(
        f'<div class="section-label">{escape(label)}</div>',
        unsafe_allow_html=True,
    )


def _render_answer_panel(answer: str) -> None:
    st.markdown(
        f"""
        <div class="answer-panel">
            <h3>Grounded Answer</h3>
            <div>{escape(answer).replace(chr(10), "<br>")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="DataCenter-HVAC Copilot", layout="wide")
    _render_console_shell()

    api_base_url = st.sidebar.text_input("API 地址", value=get_default_api_base_url())
    tab_ask, tab_eval = st.tabs(["Copilot", "评测摘要"])
    with tab_ask:
        _render_ask_tab(api_base_url)
    with tab_eval:
        _render_eval_tab(api_base_url)


def _render_ask_tab(api_base_url: str) -> None:
    control_col, result_col = st.columns([0.34, 0.66], gap="large")
    with control_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        _render_panel_start("Mission Control")
        walkthrough_labels = ["自定义问题"] + [case["label"] for case in DEMO_WALKTHROUGHS]
        walkthrough_label = st.selectbox("典型案例", walkthrough_labels)
        selected_case = next(
            (case for case in DEMO_WALKTHROUGHS if case["label"] == walkthrough_label),
            None,
        )
        if selected_case:
            st.caption(selected_case["why"])

        default_task_type = selected_case["task_type"] if selected_case else None
        default_task_label = next(
            label for label, value in TASK_OPTIONS.items() if value == default_task_type
        )
        task_label = st.selectbox(
            "任务类型",
            list(TASK_OPTIONS.keys()),
            index=list(TASK_OPTIONS.keys()).index(default_task_label),
        )
        workflow_label = st.selectbox(
            "Workflow",
            list(WORKFLOW_OPTIONS.keys()),
            index=0,
        )
        question = st.text_area(
            "问题",
            value=(
                selected_case["question"]
                if selected_case
                else "episode_001 中 zone_a 在最近 3 小时的温度最大值是多少？"
            ),
            height=210,
        )
        run_clicked = st.button("运行分析", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    with result_col:
        if not run_clicked:
            _render_empty_state()
            return
        if not question.strip():
            st.warning("请输入问题。")
            return
        try:
            result = ask_api(
                api_base_url=api_base_url,
                question=question.strip(),
                task_type=TASK_OPTIONS[task_label],
                workflow_engine=WORKFLOW_OPTIONS[workflow_label],
            )
        except ApiClientError as exc:
            st.error(str(exc))
            return

        _render_result(result)


def _render_eval_tab(api_base_url: str) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    _render_panel_start("Evaluation Bench")
    eval_path = st.text_input("评测集路径", value="data/eval/hvac_eval.jsonl")
    if st.button("运行评测", type="primary"):
        try:
            result = run_eval_api(api_base_url=api_base_url, eval_path=eval_path)
        except ApiClientError as exc:
            st.error(str(exc))
            return
        _render_eval_result(result)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_empty_state() -> None:
    st.markdown(
        """
        <div class="answer-panel">
            <h3>Ready</h3>
            <div>
                选择一个 walkthrough 或输入自定义问题，然后运行分析。推荐先看
                <code>BEAR 数据边界</code>、<code>温度时序查询</code> 和
                <code>策略建议边界</code> 三个案例。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_status_grid(
        [
            {"label": "Route", "value": "waiting", "hint": "默认进入 LangGraph workflow"},
            {"label": "Tools", "value": "standby", "hint": "按问题类型调用 RAG / timeseries / policy"},
            {"label": "Audit", "value": "armed", "hint": "回答会检查生产遥测和 LLM 控制误述"},
        ]
    )


def _render_result(result: dict) -> None:
    _render_answer_panel(result.get("answer", ""))
    _render_status_grid(build_status_cards(result))

    _render_section_label("Structured Evidence")
    _render_tool_summary(result.get("tool_results", []))

    with st.expander("Execution Timeline", expanded=True):
        st.dataframe(pd.DataFrame(build_execution_timeline(result)), use_container_width=True)

    trace_rows = build_workflow_trace_rows(result)
    if trace_rows:
        with st.expander("LangGraph Workflow Trace", expanded=True):
            st.caption(
                "StateGraph nodes executed by the selected workflow, including route, tool, evidence, and audit state."
            )
            st.dataframe(pd.DataFrame(trace_rows), use_container_width=True)

    audit = result.get("answer_audit")
    if audit:
        with st.expander("Safety Audit", expanded=True):
            st.metric("Audit", "passed" if audit.get("passed") else "review")
            st.dataframe(
                pd.DataFrame(build_safety_audit_rows(result)),
                use_container_width=True,
            )

    with st.expander("Citations", expanded=True):
        citations = result.get("citations", [])
        if citations:
            st.json(citations)
        else:
            st.caption("No citations returned.")

    with st.expander("Tool Results", expanded=True):
        tool_results = result.get("tool_results", [])
        if tool_results:
            st.json(tool_results)
        else:
            st.caption("No tool results returned.")

    with st.expander("Retrieved Contexts"):
        contexts = result.get("retrieved_contexts", [])
        if contexts:
            st.json(contexts)
        else:
            st.caption("No retrieved contexts returned.")


def _render_tool_summary(tool_results: list[dict]) -> None:
    for result in tool_results:
        summary = result.get("summary")
        if isinstance(summary, dict):
            _render_section_label("Metric Summary")
            summary_frame = pd.DataFrame([summary])
            st.dataframe(summary_frame, use_container_width=True)

        records = result.get("records") or result.get("series")
        if isinstance(records, list) and records:
            records_frame = pd.DataFrame(records)
            _render_section_label("Trend")
            st.dataframe(records_frame, use_container_width=True)
            if {"timestamp", "value"}.issubset(records_frame.columns):
                chart_frame = records_frame.copy()
                chart_frame["timestamp"] = pd.to_datetime(chart_frame["timestamp"])
                st.line_chart(chart_frame, x="timestamp", y="value")

        anomalies = result.get("anomalies")
        if isinstance(anomalies, list) and anomalies:
            _render_section_label("Anomalies")
            st.dataframe(pd.DataFrame(anomalies), use_container_width=True)


def _render_eval_result(result: dict) -> None:
    metrics = result.get("metrics", {})
    if metrics:
        _render_section_label("Metrics")
        grouped_metrics = group_eval_metrics(metrics)
        for group_name, metric_items in grouped_metrics.items():
            st.markdown(f"**{group_name}**")
            columns = st.columns(min(3, len(metric_items)))
            for index, (name, value) in enumerate(metric_items):
                with columns[index % len(columns)]:
                    st.metric(name, f"{value:.3f}")
        st.caption(
            "Quality Proxy 指标来自本地 must_include / must_not_include 弱标注，"
            "不等价于人工评审或 LLM judge。"
        )
        st.dataframe(
            pd.DataFrame(
                [{"metric": name, "value": value} for name, value in metrics.items()]
            ),
            use_container_width=True,
        )

    predictions = result.get("predictions", [])
    if predictions:
        st.subheader("Predictions")
        preview = build_prediction_preview(predictions)
        st.dataframe(pd.DataFrame(preview), use_container_width=True)
        with st.expander("Raw predictions"):
            st.json(predictions)


if __name__ == "__main__":
    main()
