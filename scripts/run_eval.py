from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.api.demo_factory import build_demo_orchestrator
from src.evaluation.dataset import load_eval_dataset
from src.evaluation.human_review import (
    create_annotation_template,
    create_human_review_sample,
    human_calibration_summary,
    load_human_annotations,
    save_jsonl,
)
from src.evaluation.report import save_experiment_report
from src.evaluation.policy_benchmark import run_policy_benchmark
from src.evaluation.safety_adversarial import (
    evaluate_safety_adversarial_dataset,
    load_safety_adversarial_dataset,
)
from src.evaluation.llm_judge import DeterministicKeywordJudge
from src.retrieval.cross_encoder import SentenceTransformersCrossEncoderScorer
from src.evaluation.runner import (
    run_baseline_comparison,
    run_baseline_eval,
    run_runtime_guardrail_eval,
    save_predictions_jsonl,
)

DEFAULT_EVAL_PATH = Path("data/eval/hvac_eval.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/eval/baseline_predictions.jsonl")
DEFAULT_COMPARISON_OUTPUT_PATH = Path("data/eval/baseline_comparison.json")
DEFAULT_REPORT_OUTPUT_PATH = Path("docs/experiment_report.md")
DEFAULT_REVIEW_SAMPLE_PATH = Path("data/eval/human_review_sample.jsonl")
DEFAULT_REVIEW_ANNOTATIONS_PATH = Path("data/eval/human_review_annotations.jsonl")
DEFAULT_SAFETY_ADVERSARIAL_PATH = Path("data/eval/safety_adversarial.jsonl")
DEFAULT_RUNTIME_EVAL_PATH = Path("data/eval/agent_runtime_eval.jsonl")
DEFAULT_RUNTIME_OUTPUT_PATH = Path("data/eval/agent_runtime_predictions.jsonl")
DEFAULT_RUNTIME_COMPARISON_OUTPUT_PATH = Path("data/eval/agent_runtime_comparison.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the baseline evaluation demo.")
    parser.add_argument(
        "--eval-path",
        default=str(DEFAULT_EVAL_PATH),
        help="Path to the eval JSONL dataset.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path where prediction JSONL should be written.",
    )
    parser.add_argument(
        "--comparison-output",
        default=None,
        help=(
            "Path where LLM-only/RAG/RAG+Tool Agent comparison JSON should be written. "
            "Defaults to the project eval artifact for the full dataset, or next to --output "
            "for custom eval runs."
        ),
    )
    parser.add_argument(
        "--report-output",
        default=None,
        help=(
            "Path where the Markdown experiment report should be written. Defaults to the "
            "project report for the full dataset, or next to --output for custom eval runs."
        ),
    )
    parser.add_argument(
        "--human-review-sample-output",
        default=None,
        help="Path where the human review sample JSONL should be written.",
    )
    parser.add_argument(
        "--human-review-annotations-output",
        default=None,
        help=(
            "Path where the human annotation template JSONL should be written. Existing files "
            "are preserved so reviewer labels are not overwritten."
        ),
    )
    parser.add_argument(
        "--enable-llm-judge",
        action="store_true",
        help="Enable optional LLM judge metrics. Disabled by default for reproducibility.",
    )
    parser.add_argument(
        "--llm-judge-provider",
        default="deterministic",
        choices=["deterministic"],
        help="Judge provider to use when --enable-llm-judge is set.",
    )
    parser.add_argument(
        "--dense-provider",
        default="deterministic",
        choices=["deterministic", "sentence-transformers"],
        help="Dense retrieval embedding provider for rag_dense baseline.",
    )
    parser.add_argument(
        "--dense-backend",
        default="memory",
        choices=["memory", "faiss"],
        help="Dense retrieval backend for rag_dense baseline.",
    )
    parser.add_argument(
        "--dense-model",
        default=None,
        help=(
            "Sentence-transformers model name used when --dense-provider sentence-transformers. "
            "Example: BAAI/bge-small-zh-v1.5"
        ),
    )
    parser.add_argument(
        "--enable-cross-encoder-rerank",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Cross-encoder reranking is enabled by default; "
            "use --disable-cross-encoder-rerank for fast smoke tests."
        ),
    )
    parser.add_argument(
        "--disable-cross-encoder-rerank",
        action="store_true",
        help=(
            "Disable hybrid_rrf_cross_encoder in baseline comparison for fast runs or "
            "environments without the reranker model."
        ),
    )
    parser.add_argument(
        "--cross-encoder-model",
        default="BAAI/bge-reranker-base",
        help="Sentence-transformers CrossEncoder model used when cross-encoder rerank is enabled.",
    )
    parser.add_argument(
        "--disable-persistent-knowledge",
        action="store_true",
        help=(
            "Force the demo markdown documents under data/documents for this eval run, even when "
            "a persistent FAISS knowledge base exists. Useful for legacy eval sets whose "
            "required_documents use demo source IDs."
        ),
    )
    parser.add_argument(
        "--safety-adversarial-path",
        default=str(DEFAULT_SAFETY_ADVERSARIAL_PATH),
        help="Path to the adversarial Safety Audit JSONL dataset.",
    )
    parser.add_argument(
        "--runtime-eval-path",
        default=str(DEFAULT_RUNTIME_EVAL_PATH),
        help="Path to the Agent Runtime / Guardrail JSONL dataset.",
    )
    parser.add_argument(
        "--runtime-output",
        default=str(DEFAULT_RUNTIME_OUTPUT_PATH),
        help="Path where Agent Runtime predictions should be written.",
    )
    parser.add_argument(
        "--runtime-comparison-output",
        default=str(DEFAULT_RUNTIME_COMPARISON_OUTPUT_PATH),
        help="Path where Agent Runtime metrics JSON should be written.",
    )
    args = parser.parse_args()
    eval_path = Path(args.eval_path)
    output_path = Path(args.output)
    comparison_path = _resolve_secondary_output_path(
        explicit_path=args.comparison_output,
        eval_path=eval_path,
        output_path=output_path,
        default_path=DEFAULT_COMPARISON_OUTPUT_PATH,
        fallback_name="baseline_comparison.json",
    )
    report_path = _resolve_secondary_output_path(
        explicit_path=args.report_output,
        eval_path=eval_path,
        output_path=output_path,
        default_path=DEFAULT_REPORT_OUTPUT_PATH,
        fallback_name="experiment_report.md",
    )
    review_sample_path = _resolve_secondary_output_path(
        explicit_path=args.human_review_sample_output,
        eval_path=eval_path,
        output_path=output_path,
        default_path=DEFAULT_REVIEW_SAMPLE_PATH,
        fallback_name="human_review_sample.jsonl",
    )
    review_annotations_path = _resolve_secondary_output_path(
        explicit_path=args.human_review_annotations_output,
        eval_path=eval_path,
        output_path=output_path,
        default_path=DEFAULT_REVIEW_ANNOTATIONS_PATH,
        fallback_name="human_review_annotations.jsonl",
    )

    orchestrator = build_demo_orchestrator(
        use_env_answer_generator=False,
        use_persistent_knowledge=not args.disable_persistent_knowledge,
    )
    llm_judge = _build_llm_judge(args.llm_judge_provider) if args.enable_llm_judge else None
    result = run_baseline_eval(
        eval_path=eval_path,
        orchestrator=orchestrator,
        llm_judge=llm_judge,
    )
    save_predictions_jsonl(result["predictions"], output_path)
    comparison = run_baseline_comparison(
        eval_path,
        orchestrator,
        dense_provider=args.dense_provider,
        dense_backend=args.dense_backend,
        dense_model=args.dense_model,
        cross_encoder_scorer=(
            _build_cross_encoder_scorer(args.cross_encoder_model)
            if not args.disable_cross_encoder_rerank
            else None
        ),
    )
    safety_summary = _load_optional_safety_adversarial_summary(
        Path(args.safety_adversarial_path)
    )
    dropt_summary = _load_optional_dropt_policy_summary(eval_path)
    runtime_summary = _run_optional_runtime_guardrail_eval(
        Path(args.runtime_eval_path),
        Path(args.runtime_output),
        Path(args.runtime_comparison_output),
        orchestrator,
    )
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_payload = {
        "summary": comparison["summary"],
        "by_task_type": comparison["by_task_type"],
    }
    if safety_summary is not None:
        comparison_payload["safety_adversarial"] = safety_summary
    if dropt_summary is not None:
        comparison_payload["dropt_policy_benchmark"] = dropt_summary
    if runtime_summary is not None:
        comparison_payload["agent_runtime_guardrail"] = runtime_summary
    comparison_path.write_text(
        json.dumps(comparison_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    records = load_eval_dataset(eval_path)
    prediction_map = {prediction["id"]: prediction for prediction in result["predictions"]}
    review_sample = create_human_review_sample(records, prediction_map)
    save_jsonl(review_sample, review_sample_path)
    if not review_annotations_path.exists():
        save_jsonl(create_annotation_template(review_sample), review_annotations_path)
    human_summary = human_calibration_summary(load_human_annotations(review_annotations_path))
    save_experiment_report(
        comparison["summary"],
        output_path=report_path,
        eval_record_count=len(records),
        expected_keyword_record_count=sum(1 for record in records if record.expected_keywords),
        by_task_type=comparison["by_task_type"],
        human_calibration=human_summary,
        safety_adversarial=safety_summary,
        agent_runtime_guardrail=runtime_summary,
        dropt_policy_benchmark=dropt_summary,
        dense_provider=args.dense_provider,
        dense_backend=args.dense_backend,
        dense_model=args.dense_model,
        cross_encoder_model=(
            args.cross_encoder_model if not args.disable_cross_encoder_rerank else None
        ),
    )
    print(f"Saved predictions to {output_path}")
    print(f"Saved baseline comparison summary to {comparison_path}")
    print(f"Saved experiment report to {report_path}")
    print(f"Saved human review sample to {review_sample_path}")
    print(f"Human review annotations at {review_annotations_path}")
    if runtime_summary is not None:
        print(f"Saved Agent Runtime predictions to {args.runtime_output}")
        print(f"Saved Agent Runtime comparison to {args.runtime_comparison_output}")
    print(result["metrics"])


def _resolve_secondary_output_path(
    *,
    explicit_path: str | None,
    eval_path: Path,
    output_path: Path,
    default_path: Path,
    fallback_name: str,
) -> Path:
    if explicit_path:
        return Path(explicit_path)
    if eval_path == DEFAULT_EVAL_PATH and output_path == DEFAULT_OUTPUT_PATH:
        return default_path
    return output_path.with_name(fallback_name)


def _build_llm_judge(provider: str):
    if provider == "deterministic":
        return DeterministicKeywordJudge()
    raise ValueError(f"Unsupported LLM judge provider: {provider}")


class _DeterministicTestCrossEncoderScorer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def score(self, query: str, texts: list[str]) -> list[float]:
        query_tokens = set(query.lower().split())
        scores = []
        for text in texts:
            text_tokens = set(text.lower().split())
            scores.append(float(len(query_tokens & text_tokens)))
        return scores


def _build_cross_encoder_scorer(model_name: str):
    if os.getenv("HVAC_COPILOT_TEST_FAKE_CROSS_ENCODER", "").strip() == "1":
        return _DeterministicTestCrossEncoderScorer(model_name)
    return SentenceTransformersCrossEncoderScorer(model_name)


def _load_optional_safety_adversarial_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    return evaluate_safety_adversarial_dataset(load_safety_adversarial_dataset(path))


def _load_optional_dropt_policy_summary(eval_path: Path) -> dict | None:
    checkpoint_path = PROJECT_ROOT / "models" / "dropt" / "policy_best_fno_guided.pth"
    if not checkpoint_path.exists():
        return None
    orchestrator = build_demo_orchestrator(
        project_root=PROJECT_ROOT,
        use_env_answer_generator=False,
        use_dropt_policy=True,
    )
    records = load_eval_dataset(eval_path)

    def latest_state(_record):
        return orchestrator.task_executor.latest_policy_state()

    return run_policy_benchmark(records, latest_state, orchestrator.policy_runner)


def _run_optional_runtime_guardrail_eval(
    eval_path: Path,
    output_path: Path,
    comparison_path: Path,
    orchestrator,
) -> dict | None:
    if not eval_path.exists():
        return None

    result = run_runtime_guardrail_eval(eval_path, orchestrator)
    save_predictions_jsonl(result["predictions"], output_path)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": result["metrics"],
        "by_task_type": result["by_task_type"],
        "by_difficulty": result["by_difficulty"],
    }
    comparison_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "summary": result["metrics"],
        "by_task_type": result["by_task_type"],
        "by_difficulty": result["by_difficulty"],
    }


if __name__ == "__main__":
    main()
