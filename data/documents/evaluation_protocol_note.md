# Evaluation Protocol Note

Source type: project internal note.
Published at: 2026.
Category: evaluation.

Evaluation should compare LLM-only, RAG, and RAG plus tool-agent modes on the same JSONL dataset. Metrics should include citation hit rate, tool selection accuracy, tool execution success rate, and evidence coverage.

The evaluation dataset should keep each record traceable through an id, task type, gold answer, required tools, required documents, and expected output format.
