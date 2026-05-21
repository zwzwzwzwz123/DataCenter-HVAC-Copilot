# Evidence-Grounded Answer Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional DeepSeek-backed answer generator with a deterministic evidence-grounded fallback.

**Architecture:** A small generator interface receives question, route, retrieved contexts, citations, tool results, policy results, and data-source metadata. The orchestrator calls this generator after routing/retrieval/tool execution. Demo factory enables DeepSeek only when `DEEPSEEK_API_KEY` exists.

**Tech Stack:** Python 3.10+, pytest, pydantic-compatible dictionaries, standard-library `urllib.request` for DeepSeek-compatible HTTP calls.

---

### Task 1: Deterministic Generator

**Files:**
- Create: `src/agent/answer_generator.py`
- Test: `tests/test_answer_generator.py`

- [ ] Write failing tests for citation preservation, tool evidence, policy-action safety, and BEAR data-source wording.
- [ ] Run `python -m pytest tests/test_answer_generator.py -q` and confirm failures are due to missing module.
- [ ] Implement `AnswerGeneratorInput`, `AnswerGenerator`, and `DeterministicAnswerGenerator`.
- [ ] Run `python -m pytest tests/test_answer_generator.py -q` and confirm pass.

### Task 2: Orchestrator Integration

**Files:**
- Modify: `src/agent/orchestrator.py`
- Test: `tests/test_agent_orchestrator.py`

- [ ] Add failing tests showing orchestrator uses the generator for document, time-series, anomaly, and policy routes.
- [ ] Run targeted tests and confirm expected failures.
- [ ] Inject `answer_generator` into `BaselineOrchestrator` with deterministic default.
- [ ] Run targeted tests and confirm pass.

### Task 3: DeepSeek Optional Adapter

**Files:**
- Create: `src/agent/deepseek_generator.py`
- Modify: `src/api/demo_factory.py`
- Modify: `.env.example`
- Test: `tests/test_deepseek_generator.py`

- [ ] Add failing tests for request payload construction and fallback behavior using fake transport.
- [ ] Run targeted tests and confirm expected failures.
- [ ] Implement standard-library HTTP adapter and environment-driven factory selection.
- [ ] Run targeted tests and confirm pass.

### Task 4: Docs And Evaluation

**Files:**
- Modify: `README.md`
- Modify: `docs/system_design.md`
- Modify: `docs/stage_1_handoff.md`

- [ ] Document the DeepSeek optional setup and safety boundary.
- [ ] Run `python -m pytest -q`.
- [ ] Run `python scripts/run_eval.py`.
- [ ] Review generated `docs/experiment_report.md` and include any metric changes in final notes.
