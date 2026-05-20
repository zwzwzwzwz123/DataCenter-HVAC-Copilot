# Eval 100 Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the evaluation dataset to 100 records with balanced task coverage, more domain notes, and broader quality-proxy annotations.

**Architecture:** Keep the existing JSONL dataset and Markdown document loading pipeline. Add domain notes under `data/documents/`, append eval records to `data/eval/hvac_eval.jsonl`, tighten tests for count/distribution/annotation coverage, regenerate reports, and update maintained docs.

**Tech Stack:** Python, pytest, UTF-8 Markdown, JSONL, existing evaluation runner.

---

## File Structure

- Modify `tests/test_evaluation.py`: assert 100 records, 100 keyword annotations, target task distribution, and at least 40 quality proxy annotations.
- Modify `tests/test_retrieval_pipeline.py`: assert new domain documents are loaded.
- Create six files under `data/documents/`.
- Modify `data/eval/hvac_eval.jsonl`: append records 50-100.
- Regenerate `data/eval/baseline_comparison.json` and `docs/experiment_report.md`.
- Modify `README.md`, `docs/system_design.md`, `docs/stage_1_handoff.md`.

## Tasks

### Task 1: Failing Eval Coverage Tests

**Files:**
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Update count and annotation tests**

Change expected count from 49 to 100:

```python
assert len(records) == 100
assert len(keyword_records) == 100
```

Change quality proxy annotation coverage:

```python
assert len(annotated) >= 40
```

Add distribution assertions:

```python
def test_eval_dataset_task_type_distribution_matches_stage_target():
    records = load_eval_dataset(Path("data/eval/hvac_eval.jsonl"))
    counts = {}
    for record in records:
        counts[record.task_type] = counts.get(record.task_type, 0) + 1

    assert counts == {
        "document_qa": 40,
        "timeseries_query": 20,
        "anomaly_diagnosis": 20,
        "policy_recommendation": 20,
    }
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_evaluation.py::test_eval_dataset_has_curated_keywords_for_representative_records tests/test_evaluation.py::test_eval_dataset_has_quality_proxy_annotations_for_representative_records tests/test_evaluation.py::test_eval_dataset_task_type_distribution_matches_stage_target -v
```

Expected: FAIL because the dataset still has 49 records.

### Task 2: Failing Document Loader Test

**Files:**
- Modify: `tests/test_retrieval_pipeline.py`

- [ ] **Step 1: Add expected source ids**

Extend `test_demo_documents_include_similar_theme_pressure_notes` with:

```python
"economizer_free_cooling_note",
"redundancy_maintenance_alarm_note",
"liquid_air_hybrid_cooling_note",
"sensor_missing_data_quality_note",
"policy_offline_replay_boundary_note",
"timeseries_tool_workflow_note",
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python -m pytest tests/test_retrieval_pipeline.py::test_demo_documents_include_similar_theme_pressure_notes -v
```

Expected: FAIL because new documents are not present.

### Task 3: Add Domain Documents

**Files:**
- Create: `data/documents/economizer_free_cooling_note.md`
- Create: `data/documents/redundancy_maintenance_alarm_note.md`
- Create: `data/documents/liquid_air_hybrid_cooling_note.md`
- Create: `data/documents/sensor_missing_data_quality_note.md`
- Create: `data/documents/policy_offline_replay_boundary_note.md`
- Create: `data/documents/timeseries_tool_workflow_note.md`

- [ ] **Step 1: Create documents**

Each document must be UTF-8 Markdown with an H1 title, 2-3 short sections, and explicit BEAR simulation/proxy boundary wording where relevant.

- [ ] **Step 2: Run loader test**

Run:

```bash
python -m pytest tests/test_retrieval_pipeline.py::test_demo_documents_include_similar_theme_pressure_notes -v
```

Expected: PASS.

### Task 4: Append Eval Records

**Files:**
- Modify: `data/eval/hvac_eval.jsonl`

- [ ] **Step 1: Append 51 records**

Append:

- `doc_qa_024` through `doc_qa_040`
- `ts_query_009` through `ts_query_020`
- `anomaly_010` through `anomaly_020`
- `policy_010` through `policy_020`

Every record must include non-empty `expected_keywords`. At least 28 of the new records should include `must_include` or `must_not_include` so the total annotated count reaches at least 40.

- [ ] **Step 2: Run evaluation tests**

Run:

```bash
python -m pytest tests/test_evaluation.py -v
```

Expected: PASS.

### Task 5: Regenerate Reports and Update Docs

**Files:**
- Regenerate: `data/eval/baseline_comparison.json`
- Regenerate: `docs/experiment_report.md`
- Modify: `README.md`
- Modify: `docs/system_design.md`
- Modify: `docs/stage_1_handoff.md`

- [ ] **Step 1: Run full tests**

Run:

```bash
python -m pytest -q
```

Expected: PASS.

- [ ] **Step 2: Run evaluation script**

Run:

```bash
python scripts/run_eval.py
```

Expected: exits 0 and report says 100 records.

- [ ] **Step 3: Update docs**

Update README/system design/handoff with 100 records, target distribution, at least 40 quality proxy annotations, and latest metrics.

### Task 6: Final Verification and Commit

**Files:**
- Review changed files.

- [ ] **Step 1: Final verification**

Run:

```bash
python -m pytest -q
python scripts/run_eval.py
```

Expected: both commands exit 0.

- [ ] **Step 2: Commit**

Run:

```bash
git add README.md docs data tests
git commit -m "feat: expand eval set to 100 records"
```

Expected: commit succeeds.
