from __future__ import annotations

import argparse
import json
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
from src.evaluation.llm_judge import DeterministicKeywordJudge
from src.evaluation.runner import (
    run_baseline_comparison,
    run_baseline_eval,
    save_predictions_jsonl,
)

DEFAULT_EVAL_PATH = Path("data/eval/hvac_eval.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/eval/baseline_predictions.jsonl")
DEFAULT_COMPARISON_OUTPUT_PATH = Path("data/eval/baseline_comparison.json")
DEFAULT_REPORT_OUTPUT_PATH = Path("docs/experiment_report.md")
DEFAULT_REVIEW_SAMPLE_PATH = Path("data/eval/human_review_sample.jsonl")
DEFAULT_REVIEW_ANNOTATIONS_PATH = Path("data/eval/human_review_annotations.jsonl")


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

    orchestrator = build_demo_orchestrator(use_env_answer_generator=False)
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
    )
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(
        json.dumps(
            {
                "summary": comparison["summary"],
                "by_task_type": comparison["by_task_type"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
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
    )
    print(f"Saved predictions to {output_path}")
    print(f"Saved baseline comparison summary to {comparison_path}")
    print(f"Saved experiment report to {report_path}")
    print(f"Saved human review sample to {review_sample_path}")
    print(f"Human review annotations at {review_annotations_path}")
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


if __name__ == "__main__":
    main()
