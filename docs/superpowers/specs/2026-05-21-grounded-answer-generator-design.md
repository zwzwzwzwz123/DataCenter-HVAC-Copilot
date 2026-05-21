# Evidence-Grounded Answer Generator Design

## Goal

Add an answer generation layer that can produce more natural Copilot-style answers while staying grounded in retrieved document contexts and deterministic tool results.

## Scope

The generator is an explanation component, not a controller. It may summarize retrieved contexts, citations, tool results, route metadata, and data-source metadata. It must not call BEAR directly, write control actions, or invent policy outputs.

The default path remains deterministic and works without network access or API keys. DeepSeek is optional and enabled only when environment configuration is present.

## Architecture

- `src/agent/answer_generator.py` defines a small generator interface and deterministic fallback.
- `src/agent/deepseek_generator.py` contains an OpenAI-compatible DeepSeek client implemented with the Python standard library.
- `BaselineOrchestrator` accepts an optional generator and uses it to produce final answers after retrieval and tool execution.
- `src/api/demo_factory.py` selects DeepSeek only when `DEEPSEEK_API_KEY` is configured; otherwise it uses deterministic fallback.

## Evidence Rules

- Document QA answers may only use `retrieved_contexts` and `citations`.
- Time-series and anomaly answers must cite `tool_results` and must describe the active `data_source`.
- Policy recommendation answers may only explain policy outputs returned by policy tools. They must not create new control actions.
- If no evidence is available, the generator must state that evidence is insufficient.
- BEAR must be described only as HVAC simulation or a controllable proxy scenario, never as real production data-center telemetry.

## DeepSeek Use

DeepSeek is used only for final explanation generation. The prompt instructs the model to answer in Chinese, use only supplied evidence, preserve citations/tool evidence, and refuse to invent control actions. API configuration is read from environment variables:

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`, default `https://api.deepseek.com`
- `DEEPSEEK_MODEL`, default `deepseek-chat`
- `DEEPSEEK_TIMEOUT_SECONDS`, default `30`

If the API call fails, the system falls back to deterministic generation.

## Testing

Tests verify that deterministic answers include evidence, preserve citations, describe data-source boundaries, avoid production-telemetry wording, and avoid invented policy actions. DeepSeek tests use an injected fake transport so no real network call is required.
