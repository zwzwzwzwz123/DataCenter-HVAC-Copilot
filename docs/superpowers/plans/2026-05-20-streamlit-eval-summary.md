# Streamlit Eval Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the Streamlit evaluation summary tab so it groups metrics by purpose and previews citation/tool evidence fields.

**Architecture:** Add pure helper functions in `app/streamlit_app.py` for metric grouping and prediction preview construction. Render grouped metric sections in the existing eval tab without changing the API response contract.

**Tech Stack:** Python, Streamlit, pandas, pytest.

---

## File Structure

- Modify `app/streamlit_app.py`: add `METRIC_GROUPS`, `group_eval_metrics`, `build_prediction_preview`, and grouped rendering.
- Modify `tests/test_streamlit_client.py`: add tests for helper functions.
- Modify `README.md`, `docs/system_design.md`, `docs/stage_1_handoff.md`: document the demo update.

## Tasks

### Task 1: Metric Grouping Helper

**Files:**
- Modify: `tests/test_streamlit_client.py`
- Modify: `app/streamlit_app.py`

- [ ] **Step 1: Write failing test**

Add:

```python
from app.streamlit_app import group_eval_metrics, build_prediction_preview


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
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python -m pytest tests/test_streamlit_client.py::test_group_eval_metrics_splits_quality_proxy_metrics -v
```

Expected: FAIL because `group_eval_metrics` does not exist.

- [ ] **Step 3: Implement helper**

In `app/streamlit_app.py`, add:

```python
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
```

- [ ] **Step 4: Run focused test**

Run:

```bash
python -m pytest tests/test_streamlit_client.py::test_group_eval_metrics_splits_quality_proxy_metrics -v
```

Expected: PASS.

### Task 2: Prediction Preview Helper

**Files:**
- Modify: `tests/test_streamlit_client.py`
- Modify: `app/streamlit_app.py`

- [ ] **Step 1: Write failing test**

Add:

```python
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
            },
            {
                "id": "ts_query_001",
                "task_type": "timeseries_query",
                "route": "timeseries_query",
                "tools": ["query_metric"],
                "answer": "",
                "citations": [],
                "tool_results": [{"summary": {"count": 3}}],
            },
        ]
    )

    assert preview[0]["has_citation"] is True
    assert preview[0]["has_tool_result"] is False
    assert preview[0]["answer_length"] == len("带引用的回答")
    assert preview[1]["has_citation"] is False
    assert preview[1]["has_tool_result"] is True
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python -m pytest tests/test_streamlit_client.py::test_build_prediction_preview_adds_evidence_flags_and_answer_length -v
```

Expected: FAIL because `build_prediction_preview` does not exist.

- [ ] **Step 3: Implement helper**

Add:

```python
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
```

- [ ] **Step 4: Run Streamlit client tests**

Run:

```bash
python -m pytest tests/test_streamlit_client.py -v
```

Expected: PASS.

### Task 3: Render Grouped Metrics

**Files:**
- Modify: `app/streamlit_app.py`

- [ ] **Step 1: Update rendering**

In `_render_eval_result`, replace the flat metric-card loop with grouped rendering:

```python
grouped_metrics = group_eval_metrics(metrics)
for group_name, metric_items in grouped_metrics.items():
    st.markdown(f"**{group_name}**")
    columns = st.columns(min(3, len(metric_items)))
    for index, (name, value) in enumerate(metric_items):
        with columns[index % len(columns)]:
            st.metric(name, f"{value:.3f}")
```

Keep the dataframe with all metrics below the grouped cards.

Add:

```python
st.caption("Quality Proxy 指标来自本地 must_include / must_not_include 弱标注，不等价于人工评审或 LLM judge。")
```

Use `build_prediction_preview(predictions)` instead of inline preview construction.

- [ ] **Step 2: Run tests**

Run:

```bash
python -m pytest tests/test_streamlit_client.py -v
```

Expected: PASS.

### Task 4: Update Docs and Verify

**Files:**
- Modify: `README.md`
- Modify: `docs/system_design.md`
- Modify: `docs/stage_1_handoff.md`

- [ ] **Step 1: Update docs**

Mention that Streamlit eval summary now groups retrieval, answer, tool, and quality proxy metrics and includes citation/tool/answer preview fields.

- [ ] **Step 2: Final verification**

Run:

```bash
python -m pytest -q
python scripts/run_eval.py
```

Expected: both commands exit 0.

- [ ] **Step 3: Commit**

Run:

```bash
git add README.md docs app tests
git commit -m "feat: improve streamlit eval summary"
```

Expected: commit succeeds.
