# Eval Retrieval Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the HVAC document/eval corpus and add a deterministic metadata-aware reranker so retrieval experiments better distinguish keyword, hybrid, and reranked baselines.

**Architecture:** Keep the existing lightweight stack. Add domain Markdown notes under `data/documents/`, append JSONL eval records under `data/eval/`, and minimally extend `RerankingRetriever` scoring to include citation metadata tokens from `title`, `section`, and `source_id`.

**Tech Stack:** Python 3.12-compatible standard library, pytest, existing project retrieval/evaluation modules, UTF-8 Markdown/JSONL.

---

## File Structure

- Modify `tests/test_retrieval_pipeline.py`: add a failing test for metadata-aware reranking.
- Modify `src/retrieval/retriever.py`: add metadata score weighting in `RerankingRetriever`.
- Create `data/documents/supply_air_reset_risk_note.md`: domain note for supply air reset risk.
- Create `data/documents/sensor_drift_alarm_boundary_note.md`: domain note for sensor drift and alarm evidence.
- Create `data/documents/return_air_delta_t_operations_note.md`: domain note for return air Delta-T and heat exchange.
- Create `data/documents/cooling_airflow_noise_long_note.md`: long similar-theme noise document for retrieval pressure.
- Modify `data/eval/hvac_eval.jsonl`: append 12 eval samples with `expected_keywords`.
- Modify `README.md`: update eval count, document expansion, and reranker description.
- Modify `docs/system_design.md`: update current architecture and conclusions.
- Modify `docs/stage_1_handoff.md`: update completed items, test expectations, and next steps.
- Regenerate `data/eval/baseline_predictions.jsonl`, `data/eval/baseline_comparison.json`, and `docs/experiment_report.md` via `python scripts/run_eval.py`.

## Tasks

### Task 1: Metadata-Aware Reranker Test

**Files:**
- Modify: `tests/test_retrieval_pipeline.py`

- [ ] **Step 1: Write the failing test**

Add this test after `test_reranking_retriever_promotes_exact_phrase_and_labels_mode`:

```python
def test_reranking_retriever_uses_citation_metadata_for_tie_breaking():
    target_metadata = DocumentMetadata(
        source_id="supply_air_reset_risk_note",
        title="Supply Air Reset Risk Note",
        source_path="memory",
        published_at="2026",
        category="internal_note",
    )
    noise_metadata = DocumentMetadata(
        source_id="cooling_airflow_noise_long_note",
        title="Cooling Airflow Noise Long Note",
        source_path="memory",
        published_at="2026",
        category="internal_note",
    )
    chunks = [
        noise_metadata.to_chunk(
            chunk_id="cooling_airflow_noise_long_note::chunk_0000",
            text=(
                "supply air reset risk comfort violation policy evidence "
                "cooling airflow repeated repeated repeated repeated repeated"
            ),
            section="General Cooling Noise",
            start_word=0,
            end_word=13,
        ),
        target_metadata.to_chunk(
            chunk_id="supply_air_reset_risk_note::chunk_0000",
            text="supply air reset risk comfort violation policy evidence",
            section="Supply Air Reset Risk",
            start_word=0,
            end_word=7,
        ),
    ]

    retriever = RerankingRetriever(
        KeywordRetriever(chunks),
        candidate_k=2,
        base_score_weight=0.0,
        metadata_weight=2.0,
    )
    results = retriever.search("supply air reset risk note", top_k=1)

    assert results[0]["chunk_id"] == "supply_air_reset_risk_note::chunk_0000"
    assert results[0]["metadata_score"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_retrieval_pipeline.py::test_reranking_retriever_uses_citation_metadata_for_tie_breaking -v
```

Expected: FAIL because `RerankingRetriever.__init__()` does not accept `metadata_weight`.

### Task 2: Metadata-Aware Reranker Implementation

**Files:**
- Modify: `src/retrieval/retriever.py`
- Test: `tests/test_retrieval_pipeline.py`

- [ ] **Step 1: Implement minimal metadata scoring**

Change `RerankingRetriever.__init__` signature to include:

```python
metadata_weight: float = 1.0,
```

Set:

```python
self.metadata_weight = metadata_weight
```

In `search`, after `score = self._score(...)`, add:

```python
metadata_score = self._metadata_score(query_tokens, candidate)
score += self.metadata_weight * metadata_score
```

Add to `updated`:

```python
updated["metadata_score"] = metadata_score
```

Add this method to `RerankingRetriever`:

```python
def _metadata_score(self, query_tokens: list[str], candidate: dict) -> float:
    query_token_set = set(query_tokens)
    if not query_token_set:
        return 0.0
    citation = candidate.get("citation") or {}
    metadata_text = " ".join(
        str(citation.get(key, ""))
        for key in ("source_id", "title", "section")
        if citation.get(key) is not None
    )
    metadata_tokens = set(_tokenize(metadata_text))
    return len(query_token_set & metadata_tokens) / len(query_token_set)
```

- [ ] **Step 2: Run focused retrieval tests**

Run:

```bash
python -m pytest tests/test_retrieval_pipeline.py -v
```

Expected: all retrieval pipeline tests pass.

### Task 3: Add Domain Documents

**Files:**
- Create: `data/documents/supply_air_reset_risk_note.md`
- Create: `data/documents/sensor_drift_alarm_boundary_note.md`
- Create: `data/documents/return_air_delta_t_operations_note.md`
- Create: `data/documents/cooling_airflow_noise_long_note.md`
- Modify: `tests/test_retrieval_pipeline.py`

- [ ] **Step 1: Write loader coverage test**

Extend `test_demo_documents_include_similar_theme_pressure_notes` expected set with:

```python
"supply_air_reset_risk_note",
"sensor_drift_alarm_boundary_note",
"return_air_delta_t_operations_note",
"cooling_airflow_noise_long_note",
```

- [ ] **Step 2: Run loader test to verify it fails**

Run:

```bash
python -m pytest tests/test_retrieval_pipeline.py::test_demo_documents_include_similar_theme_pressure_notes -v
```

Expected: FAIL because the new document files do not exist.

- [ ] **Step 3: Create documents**

Create the four Markdown files with Chinese content, clear H1 titles, and explicit BEAR boundary wording where relevant.

- [ ] **Step 4: Run loader test to verify it passes**

Run:

```bash
python -m pytest tests/test_retrieval_pipeline.py::test_demo_documents_include_similar_theme_pressure_notes -v
```

Expected: PASS.

### Task 4: Append Eval Samples

**Files:**
- Modify: `data/eval/hvac_eval.jsonl`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Write eval count and keyword coverage test**

In `tests/test_evaluation.py`, add or update a test that loads `data/eval/hvac_eval.jsonl` and asserts:

```python
records = load_eval_dataset("data/eval/hvac_eval.jsonl")
assert len(records) == 49
assert all(record.expected_keywords for record in records)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_evaluation.py -v
```

Expected: FAIL with record count 37.

- [ ] **Step 3: Append 12 JSONL records**

Append records `doc_qa_016` through `doc_qa_023`, `anomaly_008` through `anomaly_009`, and `policy_008` through `policy_009`. Required documents must include the new source ids where appropriate, and every record must include non-empty `expected_keywords`.

- [ ] **Step 4: Run evaluation tests**

Run:

```bash
python -m pytest tests/test_evaluation.py -v
```

Expected: PASS.

### Task 5: Regenerate Reports and Update Docs

**Files:**
- Modify: `README.md`
- Modify: `docs/system_design.md`
- Modify: `docs/stage_1_handoff.md`
- Regenerate: `data/eval/baseline_predictions.jsonl`
- Regenerate: `data/eval/baseline_comparison.json`
- Regenerate: `docs/experiment_report.md`

- [ ] **Step 1: Run full tests before report regeneration**

Run:

```bash
python -m pytest
```

Expected: PASS.

- [ ] **Step 2: Regenerate evaluation outputs**

Run:

```bash
python scripts/run_eval.py
```

Expected: command exits 0 and writes comparison/report files.

- [ ] **Step 3: Update maintained docs**

Update README, system design, and handoff to reflect 49 eval records, new documents, metadata-aware reranker, and latest metrics from the regenerated report.

- [ ] **Step 4: Run full verification**

Run:

```bash
python -m pytest
python scripts/run_eval.py
```

Expected: both commands exit 0.

### Task 6: Final Review

**Files:**
- Review: all changed files from `git diff --stat`

- [ ] **Step 1: Inspect changed files**

Run:

```bash
git status --short
git diff --stat
```

Expected: changes are limited to tests, retrieval code, docs, eval data, and generated report artifacts.

- [ ] **Step 2: Inspect BEAR boundary wording**

Run:

```bash
Select-String -Path README.md,docs/system_design.md,docs/stage_1_handoff.md,docs/experiment_report.md,data/documents/*.md -Pattern "生产遥测|真实数据中心生产|BEAR|仿真" -Encoding UTF8
```

Expected: any production-data wording is explicitly negated or bounded as simulation/proxy context.

- [ ] **Step 3: Commit implementation**

Run:

```bash
git add README.md docs data src tests app scripts
git commit -m "feat: expand retrieval eval pressure set"
```

Expected: commit succeeds.
