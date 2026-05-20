# DataCenter-HVAC Copilot

DataCenter-HVAC Copilot is a staged Python project for building a RAG + Agent + tool-calling system around BEAR HVAC simulation trajectories. BEAR data is treated as a controllable simulation source for data-center-cooling style analysis, not as real production data center telemetry.

## Current Scope

This first stage builds the project foundation:

- BEAR trajectory schema and field provenance rules
- Time-series analysis tool interfaces
- Policy adapter interfaces for rule-based, MPC-like, diffusion, and offline replay policies
- UTF-8 Markdown/text document loading, citation-preserving chunks, keyword retrieval, a lightweight BM25-style hybrid retrieval baseline, and a metadata-aware lexical reranker wrapper
- Extractive RAG baseline with citations
- A 49-record evaluation JSONL sample with curated expected keywords and representative quality-proxy annotations
- Evaluation dataset loader and citation/tool/evidence plus lightweight correctness/faithfulness proxy metrics
- Deterministic router, baseline orchestrator, and baseline eval runner
- Demo data-source metadata for processed BEAR CSV, BEAR sample CSV, or mock fallback
- A first reproducible comparison summary for LLM-only, keyword RAG, hybrid RAG, hybrid RAG + reranker, and RAG + Tool Agent baselines, including overall and per-task-type metrics
- A Streamlit demo with Copilot and evaluation-summary tabs, route/tool/citation display, metric tables, and trend charts for tool results
- Tests for retrieval, evaluation, API, BEAR ingestion, time-series, and policy behavior

Full vector RAG, LangGraph Agent orchestration, richer production-grade UI polish, and real DiffFNO / Guided-DiffFNO integration are planned for later stages.

## Environment

Recommended conda setup:

```bash
conda create -n hvac-copilot python=3.12
conda activate hvac-copilot
pip install -e ".[dev]"
```

If you already have a suitable Python environment, install only the project dependencies:

```bash
pip install -e ".[dev]"
```

## Run Tests

```bash
python -m pytest
```

Run the current baseline evaluation demo:

```bash
python scripts/run_eval.py
```

The eval script writes:

- `data/eval/baseline_predictions.jsonl`
- `data/eval/baseline_comparison.json`
- `docs/experiment_report.md`

The current evaluation set contains 49 records across document QA, time-series query, anomaly diagnosis, and policy recommendation tasks. All 49 records include curated `expected_keywords`, and representative records include `must_include` / `must_not_include` annotations for deterministic quality proxy metrics. `data/documents/` includes similar-theme internal notes plus long-noise/short-target and metadata-aware reranking pressure notes. The comparison summary currently reports `citation_hit_rate`, `context_recall`, `expected_keyword_coverage`, `lexical_answer_coverage`, `tool_selection_accuracy`, `tool_execution_success_rate`, `evidence_coverage`, `answer_correctness_proxy`, and `faithfulness_proxy` for LLM-only, keyword RAG, hybrid RAG, hybrid RAG + reranker, default RAG, and RAG + Tool Agent modes. `data/eval/baseline_comparison.json` stores both overall `summary` and `by_task_type` metrics, and `docs/experiment_report.md` renders both tables. In the latest run, `rag_hybrid_rerank` improves citation/context metrics to `0.630`, ahead of `rag_hybrid` at `0.593` and `rag_keyword` at `0.519`; `rag_tool_agent` keeps tool selection and execution at `1.000`, with lightweight `answer_correctness_proxy = 0.417` and `faithfulness_proxy = 0.333`.

Export a rollout from the real BEAR repository after cloning it outside this project:

```bash
git clone https://github.com/chz056/BEAR.git ../BEAR
pip install -r ../BEAR/requirements.txt
python scripts/export_bear_data.py --bear-root ../BEAR --num-steps 24 --output data/bear_processed/bear_rollout.csv
```

The exporter uses `BuildingEnvReal.reset()` and `BuildingEnvReal.step(action)` from BEAR. It maps BEAR state layout `[zone temperatures, outdoor temperature, GHI per zone, ground temperature, occupancy power per zone]` into this project's standardized trajectory schema.

The repository also includes the upstream BEAR code under `BEAR/`, along with the sample CSV `BEAR/BEAR/Data/Exercise2A-mytest.csv`. The demo factory uses this order by default:

1. `data/bear_processed/bear_rollout.csv`
2. `BEAR/BEAR/Data/Exercise2A-mytest.csv`
3. built-in mock trajectory

Run the API service:

```bash
uvicorn src.api.app:app --reload
```

Available starter endpoints:

- `GET /health`
- `POST /ask`
- `POST /eval/run`

`/health` and `/ask` include a `data_source` object so the demo can show whether it is using `data/bear_processed/bear_rollout.csv`, the bundled BEAR sample CSV, or the built-in mock trajectory.

Run the Streamlit demo in another terminal after the API is running:

```bash
streamlit run app/streamlit_app.py
```

The Streamlit demo includes:

- a Copilot tab for `/ask`, showing route, tools, citations, retrieved contexts, tool results, and the active trajectory `data_source`
- table and line-chart rendering for time-series tool summaries and records
- an evaluation-summary tab for `/eval/run`, showing metric cards, a metric table, and prediction previews

## Project Layout

```text
docs/                  Project design and handoff notes
data/bear_raw/         Raw BEAR exports, ignored by git
data/bear_processed/   Standardized BEAR trajectories, ignored by git
data/documents/        UTF-8 Markdown/TXT domain notes loaded by the demo RAG
data/eval/             Evaluation JSONL samples and datasets
src/core/              Shared schemas, result objects, and validation helpers
src/ingestion/         BEAR trajectory loading and normalization
src/retrieval/         Document loading, chunking, retrieval, reranking, and RAG baseline
src/tools/             Time-series analysis tools
src/agent/             Deterministic routing and baseline evidence synthesis
src/policies/          Policy adapter interfaces and fallback policies
src/evaluation/        Evaluation runners, metrics, and report rendering
src/api/               FastAPI service
app/                   Streamlit demo
scripts/               Utility scripts
tests/                 Unit tests
```
