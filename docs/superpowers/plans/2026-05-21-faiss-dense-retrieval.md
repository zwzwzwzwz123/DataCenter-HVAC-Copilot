# FAISS Dense Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional dense embedding retrieval and a `rag_dense` baseline while keeping the default project deterministic and free of API/model-download requirements.

**Architecture:** Introduce a small embedding provider interface, a deterministic hash embedding provider for tests/default comparison, a pure-Python dense retriever, and an optional FAISS retriever wrapper. Evaluation adds `rag_dense` to baseline comparison, while docs explain FAISS/Qdrant/API boundaries.

**Tech Stack:** Python standard library, numpy, optional `faiss-cpu`, optional `sentence-transformers`, pytest.

---

## Task 1: Embedding Providers

**Files:**
- Create: `src/retrieval/embeddings.py`
- Test: `tests/test_dense_retrieval.py`

- [ ] Write failing tests for deterministic embedding stability, dimensions, and unit norm.
- [ ] Run `python -m pytest tests/test_dense_retrieval.py -q` and verify import failure.
- [ ] Implement `EmbeddingProvider`, `DeterministicHashEmbeddingProvider`, and optional `SentenceTransformerEmbeddingProvider`.
- [ ] Run test and verify pass.

## Task 2: Pure-Python Dense Retriever

**Files:**
- Create: `src/retrieval/dense.py`
- Test: `tests/test_dense_retrieval.py`

- [ ] Write failing test that dense retriever returns nearest semantic-ish hash vectors and preserves citations.
- [ ] Verify RED.
- [ ] Implement `DenseRetriever` with normalized dot-product scoring and `retrieval_mode="dense_hash"`.
- [ ] Verify GREEN.

## Task 3: Optional FAISS Retriever

**Files:**
- Create: `src/retrieval/faiss_retriever.py`
- Test: `tests/test_dense_retrieval.py`

- [ ] Write failing test that `FaissDenseRetriever` raises a clear `ImportError` when FAISS is unavailable.
- [ ] Verify RED.
- [ ] Implement optional import and `FaissDenseRetriever`; if FAISS exists, build `IndexFlatIP` over normalized vectors and return `retrieval_mode="dense_faiss"`.
- [ ] Verify GREEN.

## Task 4: Baseline Comparison

**Files:**
- Modify: `src/evaluation/runner.py`
- Modify: `tests/test_baseline_runner.py`

- [ ] Write failing test expecting `rag_dense` in comparison modes and summary.
- [ ] Verify RED.
- [ ] Add dense RAG baseline using `DenseRetriever` + `DeterministicHashEmbeddingProvider`.
- [ ] Verify GREEN with baseline runner tests.

## Task 5: Dependencies and Documentation

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `docs/system_design.md`
- Modify: `docs/stage_1_handoff.md`
- Test: `tests/test_readme_doc.py`

- [ ] Add failing README doc assertions for FAISS dense retrieval, optional dense install, and Qdrant roadmap wording.
- [ ] Verify RED.
- [ ] Add optional `dense` extra with `faiss-cpu` and `sentence-transformers`.
- [ ] Update docs with FAISS/Qdrant/API explanation and `rag_dense` baseline.
- [ ] Verify GREEN.

## Task 6: Final Verification

- [ ] Run `python -m pytest tests/test_dense_retrieval.py tests/test_baseline_runner.py tests/test_readme_doc.py -q`.
- [ ] Run `python -m pytest -q`.
- [ ] Run `python scripts/run_eval.py`.
- [ ] Confirm `docs/experiment_report.md` includes `rag_dense`.
- [ ] Summarize dense retrieval result and note that real FAISS requires `pip install -e ".[dev,dense]"`.
