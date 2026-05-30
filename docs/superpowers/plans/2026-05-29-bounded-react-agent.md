# Bounded ReAct Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an LLM-driven, bounded ReAct workflow that replans after observations while enforcing local guardrails.

**Architecture:** Keep the existing deterministic and LangGraph workflows intact. Add a new `bounded_react` orchestrator that uses the current route planner for the initial plan, a structured LLM controller for per-step decisions, and the existing `AgentTaskExecutor` for tool execution, schema validation, permissions, approvals, retries, fallback, answer generation, and audit.

**Tech Stack:** Python 3.10+, Pydantic, FastAPI, Streamlit, existing DeepSeek-compatible transport, pytest, ruff.

---

### Task 1: Runtime Support

**Files:**
- Modify: `src/agent/runtime.py`
- Test: `tests/test_bounded_react_agent.py`

- [ ] Write a failing test that dynamic ReAct insertions create additional todos and events.
- [ ] Add `AgentRuntimeTrace.add_todo(step)` with stable 1-based step indexes.
- [ ] Keep `mark_todo()` compatible with existing LangGraph behavior.
- [ ] Verify new runtime tests pass.

### Task 2: Bounded ReAct Core

**Files:**
- Create: `src/agent/bounded_react.py`
- Modify: `src/agent/planner.py`
- Test: `tests/test_bounded_react_agent.py`

- [ ] Write failing tests for insert-step, invalid LLM decision fallback, max-step stopping, and approval-denied blocking.
- [ ] Add public `validate_plan_steps()` around the existing planner guardrails.
- [ ] Define `AgentObservation` and `ReActDecision` dataclasses.
- [ ] Implement `LLMBoundedReActController` with strict JSON parsing, action enum validation, route/tool validation, duplicate-call guard, and deterministic fallback.
- [ ] Implement `BoundedReActOrchestrator` loop: initial plan, execute, observe, ask controller, validate, continue/insert/replace/stop, aggregate evidence, generate answer, audit.
- [ ] Store `workflow_trace`, `react_trace`, `runtime_trace`, `todos`, and bounded controller decisions.

### Task 3: API and Streamlit Integration

**Files:**
- Modify: `src/api/app.py`
- Modify: `src/api/schemas.py`
- Modify: `app/streamlit_app.py`
- Test: `tests/test_api_app.py`
- Test: `tests/test_streamlit_client.py`

- [ ] Add API support for `workflow_engine="bounded_react"`.
- [ ] Instantiate the bounded ReAct orchestrator in `create_app()` and refresh it when knowledge is refreshed.
- [ ] Add Streamlit workflow option for Bounded ReAct.
- [ ] Extend workflow trace rows to display ReAct decision nodes cleanly.
- [ ] Verify API and Streamlit tests pass.

### Task 4: Evaluation Runner and Logs

**Files:**
- Modify: `src/evaluation/runner.py`
- Modify: `docs/optimization_log.md`

- [ ] Add a `bounded_react_agent` mode to benchmark comparison.
- [ ] Update the optimization log immediately after implementation with objective, process, issues, metrics, and conclusion.
- [ ] Verify targeted evaluation/import tests pass.

### Task 5: Final Verification

**Files:**
- Run tests and lint only.

- [ ] Run bounded ReAct tests.
- [ ] Run API and Streamlit tests.
- [ ] Run relevant existing agent tests.
- [ ] Run `ruff check`.
- [ ] Summarize verification evidence and remaining benchmark-update work.
