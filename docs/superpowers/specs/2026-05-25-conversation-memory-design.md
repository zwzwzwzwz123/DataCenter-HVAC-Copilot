# Conversation Memory Design

## Purpose

DataCenter-HVAC Copilot currently treats each `/ask` request as an independent
single-turn task. The goal of this design is to add a persistent multi-turn
conversation memory system that behaves like a data-center analysis log: each
analysis session records questions, routes, retrieved evidence, tool results,
policy outputs, workflow traces, safety audits, and final answers.

This is not a lightweight chat-history patch. Memory retrieval is a first-class
backend subsystem with durable storage, indexing, dense retrieval by default,
context-budget enforcement, and explicit evidence boundaries.

## Scope

In scope:

- Persistent session and turn storage in SQLite.
- `/ask` support for `session_id` and `memory_enabled`.
- Automatic session creation when no `session_id` is provided.
- Full turn logging after each successful `/ask`.
- Memory indexing from conversation turns into searchable memory chunks.
- Default FAISS + dense memory retrieval.
- Configurable hybrid/rerank fallback or alternate backend.
- Context manager that combines recent turns, dense-retrieved memory, stable
  context, and fresh evidence under an explicit context budget.
- Planner and answer-generator access to conversation context.
- Tests for storage, indexing, retrieval, context building, API behavior, and
  evaluation isolation.

Out of scope for the first implementation:

- Project/workspace-level memory.
- Full Streamlit session-list UI.
- Cross-user authentication or authorization.
- Cloud database deployment.
- Human-edited long-term user profiles.

## Architecture

Add a new package:

```text
src/memory/
  __init__.py
  schemas.py
  storage.py
  indexer.py
  retriever.py
  context_manager.py
  budget.py
  stable_context.py
```

Responsibilities:

- `schemas.py`: typed memory models such as `ConversationSession`,
  `ConversationTurn`, `MemoryDocument`, `MemoryChunk`,
  `ConversationContext`, and `MemoryStatus`.
- `storage.py`: SQLite repository for sessions, turns, chunks, and index
  metadata.
- `indexer.py`: converts structured turn results into memory documents and
  chunks.
- `retriever.py`: memory retrieval backends. Default backend is FAISS + dense;
  hybrid/rerank is available as an alternate backend and explicit fallback.
- `context_manager.py`: high-level orchestration for session creation, context
  loading, turn saving, and post-turn indexing.
- `budget.py`: deterministic context-budget policy.
- `stable_context.py`: stable system boundary text and version metadata.

The existing application layers use this package through `ContextManager`
instead of directly reading or writing memory tables.

## Storage

Default database path:

```text
data/memory/conversations.db
```

Config:

```text
HVAC_COPILOT_MEMORY_DB_PATH=data/memory/conversations.db
HVAC_COPILOT_MEMORY_ENABLED=true
HVAC_COPILOT_MEMORY_RETRIEVER=faiss_dense
HVAC_COPILOT_MEMORY_ALLOW_FALLBACK=false
HVAC_COPILOT_MEMORY_EMBEDDING_PROVIDER=sentence-transformers
HVAC_COPILOT_MEMORY_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

Tables:

```text
conversation_sessions
  session_id TEXT PRIMARY KEY
  title TEXT NOT NULL
  created_at TEXT NOT NULL
  updated_at TEXT NOT NULL
  summary TEXT NOT NULL DEFAULT ''
  metadata_json TEXT NOT NULL DEFAULT '{}'

conversation_turns
  turn_id TEXT PRIMARY KEY
  session_id TEXT NOT NULL
  turn_index INTEGER NOT NULL
  question TEXT NOT NULL
  answer TEXT NOT NULL
  route TEXT NOT NULL
  tools_json TEXT NOT NULL DEFAULT '[]'
  citations_json TEXT NOT NULL DEFAULT '[]'
  retrieved_contexts_json TEXT NOT NULL DEFAULT '[]'
  tool_results_json TEXT NOT NULL DEFAULT '[]'
  policy_result_json TEXT NOT NULL DEFAULT '{}'
  workflow_trace_json TEXT NOT NULL DEFAULT '[]'
  answer_audit_json TEXT NOT NULL DEFAULT '{}'
  data_source_json TEXT NOT NULL DEFAULT '{}'
  memory_context_json TEXT NOT NULL DEFAULT '{}'
  created_at TEXT NOT NULL
  FOREIGN KEY(session_id) REFERENCES conversation_sessions(session_id)

memory_chunks
  chunk_id TEXT PRIMARY KEY
  session_id TEXT NOT NULL
  turn_id TEXT NOT NULL
  chunk_index INTEGER NOT NULL
  text TEXT NOT NULL
  metadata_json TEXT NOT NULL DEFAULT '{}'
  embedding_status TEXT NOT NULL DEFAULT 'pending'
  created_at TEXT NOT NULL
  FOREIGN KEY(session_id) REFERENCES conversation_sessions(session_id)
  FOREIGN KEY(turn_id) REFERENCES conversation_turns(turn_id)

memory_index_metadata
  index_id TEXT PRIMARY KEY
  backend TEXT NOT NULL
  embedding_provider TEXT NOT NULL
  embedding_model TEXT NOT NULL
  index_path TEXT NOT NULL
  updated_at TEXT NOT NULL
  metadata_json TEXT NOT NULL DEFAULT '{}'
```

SQLite remains the source of truth. FAISS index files are derived artifacts and
can be rebuilt from `memory_chunks`.

## Memory Indexing

After each successful `/ask`, the context manager saves the full turn and asks
the indexer to create one or more memory chunks.

Each memory chunk text should be structured, concise, and retrieval-friendly:

```text
Question: ...
Route: ...
Tools: ...
Answer summary: ...
Policy result: ...
Citation source ids: ...
Tool result summary: ...
Data boundary: ...
```

The indexer must not store fabricated interpretations. It may summarize long
JSON fields deterministically, but complete structured payloads stay in
`conversation_turns`.

## Retrieval

Default backend: `faiss_dense`.

Default dense behavior:

- Use the existing dense retrieval architecture where possible.
- Prefer `SentenceTransformerEmbeddingProvider` with
  `BAAI/bge-small-zh-v1.5` for Chinese/English HVAC queries.
- Build or update a FAISS index over `memory_chunks`.
- Retrieve top relevant memory chunks for the current question.

Alternate backends:

- `hybrid`: reuse the existing BM25-style `HybridRetriever` pattern over memory
  chunks.
- `hybrid_rerank`: hybrid retrieval wrapped with the existing lexical reranker.
- `dense_memory`: in-memory dense retrieval, useful for tests and small
  sessions.

Fallback policy:

- Default runtime does not silently downgrade from FAISS/Dense. If FAISS or the
  embedding provider is unavailable, return a clear `memory_status` with
  `available=false` and an install/configuration hint.
- If `HVAC_COPILOT_MEMORY_ALLOW_FALLBACK=true`, the system may use
  `hybrid_rerank` and mark the fallback in `memory_status`.
- Tests may use deterministic embeddings or explicit non-FAISS backends to avoid
  model downloads.

## Context Construction

`ContextManager.load_context(session_id, question)` returns a structured
`ConversationContext`.

Inputs:

- Current question.
- Session summary.
- Recent turns for pronoun and reference resolution.
- Dense-retrieved memory chunks.
- Stable context reference.

Output shape:

```python
{
    "session_id": "...",
    "summary": "...",
    "recent_turns": [...],
    "relevant_memory": [...],
    "reusable_evidence_refs": [...],
    "stable_context": {
        "version": "...",
        "boundary_summary": "..."
    },
    "budget": {
        "max_chars": 6000,
        "used_chars": 0,
        "truncated": false
    }
}
```

Context priority:

1. Stable boundary reference.
2. Current user question.
3. Recent turns for reference resolution.
4. Retrieved memory chunks.
5. Current fresh evidence from RAG, tools, and policy.
6. Safety audit.

For final answer generation, current fresh evidence has the highest authority.
Conversation memory can explain continuity and resolve references, but it must
not replace current tool results or invent new evidence.

## Context Budget

The first implementation uses deterministic character budgets rather than
provider-specific token counting.

Initial budget:

```text
total conversation_context budget: 6000 chars
session_summary: max 1200 chars
recent_turns: max 3 turns, max 700 chars each
relevant_memory: max 5 chunks, max 700 chars each
reusable_evidence_refs: max 1000 chars
```

The budget manager should:

- Prefer newer recent turns for reference resolution.
- Prefer higher-scoring memory chunks for relevant history.
- Preserve route, tools, citation source ids, and policy result summaries.
- Truncate long answers before truncating tool/citation identifiers.
- Mark `budget.truncated=true` when any field is shortened.

## Stable Prefix

Stable context is separated from dynamic conversation context. It contains
project boundaries that rarely change:

- BEAR is an HVAC simulation / controllable proxy scenario, not production
  telemetry.
- LLMs do not directly generate or write back control actions.
- Policy actions must come from policy tools.
- Answers should cite retrieved contexts or tool results when possible.
- Safety audit checks remain active.

First implementation only returns a stable context version and summary to the
agent payload. It does not require provider-level prompt caching. The separation
keeps the architecture ready for future prompt caching or local prefix caching.

## API Changes

`AskRequest`:

```python
class AskRequest(BaseModel):
    question: str
    task_type: str | None = None
    workflow_engine: str = "langgraph"
    session_id: str | None = None
    memory_enabled: bool = True
```

`AskResponse` adds:

```python
session_id: str | None = None
turn_id: str | None = None
memory_status: dict = Field(default_factory=dict)
conversation_context: dict = Field(default_factory=dict)
```

Behavior:

- No `session_id` and `memory_enabled=true`: create a new session.
- Valid `session_id`: load context, run the workflow, save a new turn.
- Unknown `session_id`: return `404`, because silently creating a new session
  can hide client state bugs.
- `memory_enabled=false`: run single-turn behavior and do not read/write memory.
- `/eval/run`: memory remains disabled.

## Agent Integration

`BaselineOrchestrator.run` and `LangGraphOrchestrator.run` should accept an
optional `conversation_context`.

Planner integration:

- Route planners receive conversation context.
- They may use it to resolve references such as "last result", "that zone", or
  "the previous policy".
- LLM planner prompts include stable context summary and relevant memory, under
  budget.
- Deterministic planner can use simple reference hints from context metadata,
  but should not become dependent on LLM behavior.

Executor integration:

- Current route execution still calls RAG, timeseries tools, anomaly tools, and
  policy tools through existing executor methods.
- Historical tool results may be referenced only when the current question is a
  clear follow-up about previous results.
- If historical evidence is reused, the output must mark it as history-derived,
  not fresh tool execution.

Answer generator integration:

- `AnswerGeneratorInput` gains `conversation_context`.
- DeepSeek/Ollama/deterministic generators can mention relevant historical
  context.
- The generator must keep current fresh evidence higher priority than memory.
- The safety audit remains unchanged in purpose and should inspect the final
  answer after memory is included.

Workflow trace:

- Add trace items such as `memory_context_loaded`, `memory_retrieval`, and
  `memory_turn_saved`.
- Include memory backend, retrieved memory count, fallback status, and budget
  truncation status.

## Streamlit Impact

First implementation keeps UI work minimal:

- `app/api_client.py` can send `session_id` and parse `session_id` / `turn_id`.
- Streamlit may store the current `session_id` in `st.session_state`.
- No session list, rename, delete, or history browser in the first UI pass.

## Error Handling

- SQLite path cannot be created: run `/ask` without memory and return
  `memory_status.available=false`.
- Invalid session id: return HTTP 404.
- FAISS or embedding provider unavailable: return a clear memory unavailable
  status unless fallback is explicitly enabled.
- Index corruption: rebuild from `memory_chunks` when possible; otherwise mark
  memory retrieval unavailable and preserve turn logging if SQLite still works.
- Turn save failure after successful answer: return the answer and
  `memory_status.saved=false`; do not mask the core response.

## Evaluation Isolation

Existing deterministic evaluation must remain reproducible:

- `/eval/run` always builds an orchestrator with memory disabled.
- `scripts/run_eval.py` does not create or read memory sessions.
- Memory-specific evaluation can be added as a separate dataset/script later,
  such as `data/eval/memory_followup_eval.jsonl`.

## Testing Plan

Storage tests:

- Initializes SQLite schema.
- Creates sessions.
- Saves turns with monotonically increasing `turn_index`.
- Loads recent turns in order.
- Returns 404-like behavior for unknown session ids at the API layer.

Indexer tests:

- Converts a turn with citations, tools, and policy output into memory chunks.
- Preserves source ids, tool names, route, and data boundary metadata.
- Does not drop complete JSON payloads from `conversation_turns`.

Retriever tests:

- FAISS/Dense backend retrieves relevant memory chunks when dense dependencies
  are available.
- Deterministic or explicit test backend avoids external downloads in CI.
- Hybrid/rerank backend can be selected by config.
- Fallback only occurs when explicitly allowed.

Context manager tests:

- Builds context from summary, recent turns, and retrieved memory.
- Enforces character budget.
- Marks truncation.
- Keeps current question separate from memory.

API tests:

- `/ask` without `session_id` creates a session and saves turn 1.
- `/ask` with returned `session_id` saves turn 2 and includes conversation
  context.
- Invalid `session_id` returns 404.
- `memory_enabled=false` preserves single-turn behavior.
- `/eval/run` does not read or write memory.

Agent tests:

- Conversation context reaches planner and answer generator.
- Follow-up questions can resolve recent references.
- Fresh tool evidence remains higher priority than historical evidence.
- Workflow trace reports memory retrieval and budget status.

## Open Decisions Resolved

- Project/workspace layer: not included.
- Default memory backend: FAISS/Dense.
- Hybrid/rerank: available as configurable backend and explicit fallback.
- Streamlit: minimal compatibility only in first pass.
- Evaluation: memory disabled by default.

