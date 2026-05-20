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
from src.evaluation.report import save_experiment_report
from src.evaluation.runner import (
    run_baseline_comparison,
    run_baseline_eval,
    save_predictions_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the baseline evaluation demo.")
    parser.add_argument(
        "--eval-path",
        default="data/eval/hvac_eval.jsonl",
        help="Path to the eval JSONL dataset.",
    )
    parser.add_argument(
        "--output",
        default="data/eval/baseline_predictions.jsonl",
        help="Path where prediction JSONL should be written.",
    )
    parser.add_argument(
        "--comparison-output",
        default="data/eval/baseline_comparison.json",
        help="Path where LLM-only/RAG/RAG+Tool Agent comparison JSON should be written.",
    )
    parser.add_argument(
        "--report-output",
        default="docs/experiment_report.md",
        help="Path where the Markdown experiment report should be written.",
    )
    args = parser.parse_args()

    orchestrator = build_demo_orchestrator()
    result = run_baseline_eval(
        eval_path=Path(args.eval_path),
        orchestrator=orchestrator,
    )
    output_path = Path(args.output)
    save_predictions_jsonl(result["predictions"], output_path)
    comparison = run_baseline_comparison(Path(args.eval_path), orchestrator)
    comparison_path = Path(args.comparison_output)
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
    report_path = Path(args.report_output)
    records = load_eval_dataset(args.eval_path)
    save_experiment_report(
        comparison["summary"],
        output_path=report_path,
        eval_record_count=len(records),
        expected_keyword_record_count=sum(1 for record in records if record.expected_keywords),
        by_task_type=comparison["by_task_type"],
    )
    print(f"Saved predictions to {output_path}")
    print(f"Saved baseline comparison summary to {comparison_path}")
    print(f"Saved experiment report to {report_path}")
    print(result["metrics"])


if __name__ == "__main__":
    main()
