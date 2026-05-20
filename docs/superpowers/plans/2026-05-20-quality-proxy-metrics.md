# Quality Proxy Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic answer correctness and faithfulness proxy metrics to the evaluation pipeline and generated report.

**Architecture:** Extend `EvalRecord` with optional annotation fields, implement two local metrics in `src/evaluation/metrics.py`, wire them through `runner.py` and `report.py`, and annotate representative eval records. Keep the implementation dependency-free and compatible with existing JSONL records.

**Tech Stack:** Python 3.12-compatible standard library, Pydantic, pytest, existing evaluation runner/report modules, UTF-8 JSONL/Markdown.

---

## File Structure

- Modify `src/evaluation/dataset.py`: add `must_include` and `must_not_include` fields to `EvalRecord`.
- Modify `src/evaluation/metrics.py`: add `answer_correctness_proxy` and `faithfulness_proxy`.
- Modify `src/evaluation/runner.py`: include new metrics in summary and by-task-type metrics.
- Modify `src/evaluation/report.py`: include new metric columns and report explanation.
- Modify `data/eval/hvac_eval.jsonl`: add annotations to representative records.
- Modify `tests/test_evaluation.py`: cover loader fields and new metrics.
- Modify `tests/test_experiment_report.py`: cover report rendering with new metric columns.
- Modify `README.md`, `docs/system_design.md`, `docs/stage_1_handoff.md`: document new metrics and latest results.
- Regenerate `data/eval/baseline_comparison.json` and `docs/experiment_report.md`.

## Tasks

### Task 1: Dataset Fields

**Files:**
- Modify: `tests/test_evaluation.py`
- Modify: `src/evaluation/dataset.py`

- [ ] **Step 1: Write failing loader test**

Add this test to `tests/test_evaluation.py`:

```python
def test_eval_dataset_loads_quality_proxy_annotations():
    records = load_eval_dataset(Path("data/eval/hvac_eval.jsonl"))
    record = {record.id: record for record in records}["doc_qa_006"]

    assert "BEAR" in record.must_include
    assert "真实数据中心生产遥测" in record.must_not_include
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_evaluation.py::test_eval_dataset_loads_quality_proxy_annotations -v
```

Expected: FAIL because `EvalRecord` does not expose `must_include`.

- [ ] **Step 3: Add fields**

In `src/evaluation/dataset.py`, add:

```python
must_include: list[str] = Field(default_factory=list)
must_not_include: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Annotate `doc_qa_006`**

In `data/eval/hvac_eval.jsonl`, add:

```json
"must_include":["BEAR","仿真","可控代理场景"],"must_not_include":["真实数据中心生产遥测","真实生产遥测"]
```

to `doc_qa_006`.

- [ ] **Step 5: Run loader test**

Run:

```bash
python -m pytest tests/test_evaluation.py::test_eval_dataset_loads_quality_proxy_annotations -v
```

Expected: PASS.

### Task 2: Metrics

**Files:**
- Modify: `tests/test_evaluation.py`
- Modify: `src/evaluation/metrics.py`

- [ ] **Step 1: Write failing metrics tests**

Add imports:

```python
answer_correctness_proxy,
faithfulness_proxy,
```

Add these tests:

```python
def test_answer_correctness_proxy_scores_must_include_matches():
    records = _records_by_id(["doc_qa_006"])
    predictions = {
        "doc_qa_006": {
            "answer": "BEAR 是 HVAC 仿真轨迹，可作为可控代理场景。",
            "citations": [{"source_id": "bear_data_boundary_note"}],
            "tool_results": [],
        }
    }

    assert answer_correctness_proxy(records, predictions) == 1.0


def test_faithfulness_proxy_penalizes_missing_evidence_and_forbidden_terms():
    records = _records_by_id(["doc_qa_006", "policy_002"])
    predictions = {
        "doc_qa_006": {
            "answer": "BEAR 是真实数据中心生产遥测。",
            "citations": [{"source_id": "bear_data_boundary_note"}],
            "tool_results": [],
        },
        "policy_002": {
            "answer": "LLM 不应直接编造控制动作。",
            "citations": [],
            "tool_results": [],
        },
    }

    assert faithfulness_proxy(records, predictions) == 0.25
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_evaluation.py::test_answer_correctness_proxy_scores_must_include_matches tests/test_evaluation.py::test_faithfulness_proxy_penalizes_missing_evidence_and_forbidden_terms -v
```

Expected: FAIL because metric functions do not exist.

- [ ] **Step 3: Implement metrics**

In `src/evaluation/metrics.py`, add:

```python
def answer_correctness_proxy(records: list[EvalRecord], predictions: dict[str, dict]) -> float:
    annotated_records = [record for record in records if record.must_include]
    if not annotated_records:
        return 0.0
    scores = []
    for record in annotated_records:
        answer = str(predictions.get(record.id, {}).get("answer", "")).lower()
        required = [item.lower() for item in record.must_include]
        matches = [item for item in required if item in answer]
        scores.append(len(matches) / len(required))
    return sum(scores) / len(scores)


def faithfulness_proxy(records: list[EvalRecord], predictions: dict[str, dict]) -> float:
    annotated_records = [
        record for record in records if record.must_include or record.must_not_include
    ]
    if not annotated_records:
        return 0.0
    scores = []
    for record in annotated_records:
        predicted = predictions.get(record.id, {})
        answer = str(predicted.get("answer", "")).lower()
        forbidden = [item.lower() for item in record.must_not_include]
        if any(item in answer for item in forbidden):
            scores.append(0.0)
            continue
        score = 1.0
        if record.must_include:
            required = [item.lower() for item in record.must_include]
            matches = [item for item in required if item in answer]
            score *= len(matches) / len(required)
        if (record.required_documents or record.required_tools) and not (
            predicted.get("citations") or predicted.get("tool_results")
        ):
            score = min(score, 0.5)
        scores.append(score)
    return sum(scores) / len(scores)
```

- [ ] **Step 4: Run metrics tests**

Run:

```bash
python -m pytest tests/test_evaluation.py -v
```

Expected: PASS.

### Task 3: Runner and Report Wiring

**Files:**
- Modify: `src/evaluation/runner.py`
- Modify: `src/evaluation/report.py`
- Modify: `tests/test_experiment_report.py`

- [ ] **Step 1: Write failing report test**

In `tests/test_experiment_report.py`, update expected report assertions to check:

```python
assert "answer_correctness_proxy" in markdown
assert "faithfulness_proxy" in markdown
```

- [ ] **Step 2: Run report tests to verify failure if needed**

Run:

```bash
python -m pytest tests/test_experiment_report.py -v
```

Expected: FAIL if report does not include new columns.

- [ ] **Step 3: Wire metrics**

In `src/evaluation/runner.py`, import and include both metrics in `_compute_metrics`.

In `src/evaluation/report.py`, add both metric names to `METRIC_COLUMNS` and update table headings to render columns from `METRIC_COLUMNS`.

- [ ] **Step 4: Run report and runner tests**

Run:

```bash
python -m pytest tests/test_experiment_report.py tests/test_baseline_runner.py -v
```

Expected: PASS.

### Task 4: Annotate Representative Eval Records

**Files:**
- Modify: `data/eval/hvac_eval.jsonl`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Add annotation coverage test**

Add:

```python
def test_eval_dataset_has_quality_proxy_annotations_for_representative_records():
    records = load_eval_dataset(Path("data/eval/hvac_eval.jsonl"))
    annotated = [record for record in records if record.must_include or record.must_not_include]

    assert len(annotated) >= 12
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_evaluation.py::test_eval_dataset_has_quality_proxy_annotations_for_representative_records -v
```

Expected: FAIL until enough records are annotated.

- [ ] **Step 3: Add annotations**

Annotate at least these records:

`doc_qa_006`, `doc_qa_007`, `doc_qa_008`, `policy_001`, `policy_002`, `policy_005`, `doc_qa_012`, `doc_qa_016`, `doc_qa_017`, `doc_qa_018`, `policy_008`, `policy_009`.

- [ ] **Step 4: Run evaluation tests**

Run:

```bash
python -m pytest tests/test_evaluation.py -v
```

Expected: PASS.

### Task 5: Regenerate and Document

**Files:**
- Modify: `README.md`
- Modify: `docs/system_design.md`
- Modify: `docs/stage_1_handoff.md`
- Regenerate: `data/eval/baseline_comparison.json`
- Regenerate: `docs/experiment_report.md`

- [ ] **Step 1: Run full tests**

Run:

```bash
python -m pytest -q
```

Expected: PASS.

- [ ] **Step 2: Regenerate evaluation outputs**

Run:

```bash
python scripts/run_eval.py
```

Expected: exits 0 and generated report includes `answer_correctness_proxy` and `faithfulness_proxy`.

- [ ] **Step 3: Update maintained docs**

Update README, system design, and handoff with the new metrics, annotation count, and latest results from the generated report.

- [ ] **Step 4: Final verification**

Run:

```bash
python -m pytest -q
python scripts/run_eval.py
```

Expected: both commands exit 0.

### Task 6: Commit

**Files:**
- Review all changed files.

- [ ] **Step 1: Inspect status**

Run:

```bash
git status --short
git diff --stat
```

Expected: scoped changes only.

- [ ] **Step 2: Commit implementation**

Run:

```bash
git add README.md docs data src tests
git commit -m "feat: add quality proxy eval metrics"
```

Expected: commit succeeds.
