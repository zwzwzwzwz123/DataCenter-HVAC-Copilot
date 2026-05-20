from __future__ import annotations

import pandas as pd
import streamlit as st

from app.api_client import ApiClientError, ask_api, run_eval_api


TASK_OPTIONS = {
    "自动判断": None,
    "文档问答": "document_qa",
    "时序查询": "timeseries_query",
    "异常诊断": "anomaly_diagnosis",
    "策略建议": "policy_recommendation",
}

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
            }
        )
    return preview


def main() -> None:
    st.set_page_config(page_title="DataCenter-HVAC Copilot", layout="wide")
    st.title("DataCenter-HVAC Copilot")

    api_base_url = st.sidebar.text_input("API 地址", value="http://localhost:8000")
    tab_ask, tab_eval = st.tabs(["Copilot", "评测摘要"])
    with tab_ask:
        _render_ask_tab(api_base_url)
    with tab_eval:
        _render_eval_tab(api_base_url)


def _render_ask_tab(api_base_url: str) -> None:
    task_label = st.selectbox("任务类型", list(TASK_OPTIONS.keys()))
    question = st.text_area(
        "问题",
        value="episode_001 中 zone_a 在最近 3 小时的温度最大值是多少？",
        height=120,
    )

    if st.button("运行", type="primary"):
        if not question.strip():
            st.warning("请输入问题。")
            return
        try:
            result = ask_api(
                api_base_url=api_base_url,
                question=question.strip(),
                task_type=TASK_OPTIONS[task_label],
            )
        except ApiClientError as exc:
            st.error(str(exc))
            return

        _render_result(result)


def _render_eval_tab(api_base_url: str) -> None:
    eval_path = st.text_input("评测集路径", value="data/eval/hvac_eval.jsonl")
    if st.button("运行评测", type="primary"):
        try:
            result = run_eval_api(api_base_url=api_base_url, eval_path=eval_path)
        except ApiClientError as exc:
            st.error(str(exc))
            return
        _render_eval_result(result)


def _render_result(result: dict) -> None:
    st.subheader("回答")
    st.write(result.get("answer", ""))

    col_route, col_tools = st.columns(2)
    with col_route:
        st.metric("Route", result.get("route", "unknown"))
    with col_tools:
        tools = result.get("tools", [])
        st.metric("Tools", ", ".join(tools) if tools else "none")

    data_source = result.get("data_source", {})
    if data_source:
        st.caption(
            f"数据源: {data_source.get('kind', 'unknown')} | {data_source.get('path', '')}"
        )

    _render_tool_summary(result.get("tool_results", []))

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
            st.subheader("Metric Summary")
            summary_frame = pd.DataFrame([summary])
            st.dataframe(summary_frame, use_container_width=True)

        records = result.get("records") or result.get("series")
        if isinstance(records, list) and records:
            records_frame = pd.DataFrame(records)
            st.subheader("Trend")
            st.dataframe(records_frame, use_container_width=True)
            if {"timestamp", "value"}.issubset(records_frame.columns):
                chart_frame = records_frame.copy()
                chart_frame["timestamp"] = pd.to_datetime(chart_frame["timestamp"])
                st.line_chart(chart_frame, x="timestamp", y="value")

        anomalies = result.get("anomalies")
        if isinstance(anomalies, list) and anomalies:
            st.subheader("Anomalies")
            st.dataframe(pd.DataFrame(anomalies), use_container_width=True)


def _render_eval_result(result: dict) -> None:
    metrics = result.get("metrics", {})
    if metrics:
        st.subheader("Metrics")
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
