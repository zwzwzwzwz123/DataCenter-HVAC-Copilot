# Conversation Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add session-scoped persistent conversation memory for `/ask` while keeping evaluation runs reproducible and memory-disabled.

**Architecture:** Introduce `src/memory` as the only backend memory boundary: SQLite storage is the source of truth, retrieval is a separate status-bearing subsystem, and `ContextManager` coordinates loading context and saving turns. API and agent layers receive structured `conversation_context` but current RAG/tool/policy evidence remains authoritative.

**Tech Stack:** Python, FastAPI, Pydantic, SQLite, existing dense/FAISS and hybrid retriever utilities, pytest.

---

### Task 1: Memory Storage, Indexing, Retrieval, And Context Core

**Files:**
- Create: `src/memory/__init__.py`
- Create: `src/memory/schemas.py`
- Create: `src/memory/storage.py`
- Create: `src/memory/indexer.py`
- Create: `src/memory/retriever.py`
- Create: `src/memory/context_manager.py`
- Create: `src/memory/budget.py`
- Create: `src/memory/stable_context.py`
- Test: `tests/test_memory_storage.py`
- Test: `tests/test_memory_indexer.py`
- Test: `tests/test_memory_retriever.py`
- Test: `tests/test_memory_context_manager.py`

- [x] **Step 1: Write failing storage tests**
- [x] **Step 2: Run `pytest tests/test_memory_storage.py -q` and verify expected import failure**
- [x] **Step 3: Implement SQLite repository with sessions, turns, chunks, and metadata tables**
- [x] **Step 4: Run storage tests and verify pass**
- [x] **Step 5: Write failing indexer, retriever, and context manager tests**
- [x] **Step 6: Run targeted memory tests and verify expected failures**
- [x] **Step 7: Implement schemas, indexer, retriever backends, budget manager, stable context, and context manager**
- [x] **Step 8: Run targeted memory tests and verify pass**

### Task 2: Agent Integration

**Files:**
- Modify: `src/agent/answer_generator.py`
- Modify: `src/agent/executor.py`
- Modify: `src/agent/orchestrator.py`
- Modify: `src/agent/langgraph_workflow.py`
- Modify: `src/agent/planner.py`
- Test: `tests/test_agent_orchestrator.py`
- Test: `tests/test_answer_generator.py`

- [x] **Step 1: Write failing tests that conversation context reaches planner and answer generator**
- [x] **Step 2: Run targeted agent tests and verify expected failures**
- [x] **Step 3: Add optional `conversation_context` parameters and trace metadata**
- [x] **Step 4: Run targeted agent tests and verify pass**

### Task 3: API, Evaluation Isolation, And Streamlit Compatibility

**Files:**
- Modify: `src/api/schemas.py`
- Modify: `src/api/app.py`
- Modify: `app/api_client.py`
- Modify: `app/streamlit_app.py`
- Modify: `scripts/run_eval.py`
- Test: `tests/test_api_app.py`
- Test: `tests/test_streamlit_client.py`
- Test: `tests/test_compound_eval_script.py`

- [x] **Step 1: Write failing API tests for session creation, second turn, invalid session, memory disabled, and eval isolation**
- [x] **Step 2: Run targeted API tests and verify expected failures**
- [x] **Step 3: Wire `ContextManager` into `/ask`, add response fields, and keep `/eval/run` memory-disabled**
- [x] **Step 4: Add Streamlit client `session_id` / `memory_enabled` compatibility**
- [x] **Step 5: Run targeted API, agent, memory, and Streamlit tests**

### Task 4: Final Verification And Commit Hygiene

**Files:**
- All touched memory/API/agent/UI/test files

- [x] **Step 1: Run `pytest -q`**
- [x] **Step 2: Inspect `git status --short` and confirm `ui_improvement_plan.md` remains untouched**
- [x] **Step 3: Commit in logical chunks if tests pass**
