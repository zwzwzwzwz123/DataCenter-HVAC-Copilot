from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.agent.intent_classifier import LLMIntentClassifier, OllamaIntentClassifier, RuleBasedIntentClassifier
from src.core.env import load_env_file
from src.evaluation.dataset import load_eval_dataset
from src.evaluation.intent_routing import evaluate_intent_classifier

DEFAULT_EVAL_PATH = Path("data/eval/hvac_eval.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/eval/intent_routing_comparison.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare keyword and LLM intent routing accuracy.")
    parser.add_argument("--eval-path", default=str(DEFAULT_EVAL_PATH), help="Path to eval JSONL.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path where intent routing comparison JSON should be written.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["rule_based", "deepseek", "ollama"],
        default=["rule_based"],
        help="Intent classifier providers to evaluate.",
    )
    args = parser.parse_args()

    load_env_file(PROJECT_ROOT / ".env")
    records = load_eval_dataset(args.eval_path)
    runs: dict[str, Any] = {}
    for provider in args.providers:
        classifier, skip_reason = _build_classifier(provider)
        if classifier is None:
            runs[provider] = {"status": "skipped_missing_config", "reason": skip_reason}
            continue
        result = evaluate_intent_classifier(records, classifier)
        runs[provider] = {
            "status": "complete",
            "metrics": {
                "total": result["total"],
                "correct": result["correct"],
                "accuracy": result["accuracy"],
                "fallback_rate": result["fallback_rate"],
                "by_task_type": result["by_task_type"],
                "confusion_matrix": result["confusion_matrix"],
            },
            "predictions": result["predictions"],
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "eval_path": str(args.eval_path),
                "providers": args.providers,
                "runs": runs,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved intent routing comparison to {output_path}")


def _build_classifier(provider: str):
    if provider == "rule_based":
        return RuleBasedIntentClassifier(), None
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            return None, "DEEPSEEK_API_KEY is not set"
        return (
            LLMIntentClassifier(
                provider="deepseek",
                api_key=api_key,
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                model=os.getenv("LANGGRAPH_INTENT_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
                timeout_seconds=float(os.getenv("LANGGRAPH_INTENT_TIMEOUT_SECONDS", "20")),
            ),
            None,
        )
    if provider == "ollama":
        return (
            OllamaIntentClassifier(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=os.getenv("LANGGRAPH_INTENT_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:7b")),
                timeout_seconds=float(os.getenv("LANGGRAPH_INTENT_TIMEOUT_SECONDS", "20")),
            ),
            None,
        )
    raise ValueError(f"Unsupported intent provider: {provider}")


if __name__ == "__main__":
    main()
