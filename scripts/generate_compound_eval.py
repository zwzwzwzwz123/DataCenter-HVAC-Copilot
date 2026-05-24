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

from src.core.env import load_env_file
from src.evaluation.compound_task_generator import (
    CompoundTaskGenerator,
    validate_compound_task_candidates,
    write_compound_task_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate compound_task planner-eval records with an LLM and local validation."
    )
    parser.add_argument(
        "--output",
        default="data/eval/compound_task_eval.jsonl",
        help="Path where validated compound_task JSONL records should be written.",
    )
    parser.add_argument("--count", type=int, default=20, help="Requested candidate count.")
    parser.add_argument(
        "--fixture-json",
        default=None,
        help="Optional local JSON fixture with a records list, used for offline validation tests.",
    )
    parser.add_argument("--provider", default="deepseek", choices=["deepseek"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()

    if args.fixture_json:
        payload = json.loads(Path(args.fixture_json).read_text(encoding="utf-8"))
        records = validate_compound_task_candidates(payload.get("records", []))
    else:
        load_env_file(PROJECT_ROOT / ".env")
        import os

        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("DEEPSEEK_API_KEY is required unless --fixture-json is provided.")
        generator = CompoundTaskGenerator(
            provider=args.provider,
            api_key=api_key,
            base_url=args.base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=args.model or os.getenv("LANGGRAPH_PLANNER_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
            timeout_seconds=args.timeout_seconds,
        )
        records = generator.generate(count=args.count)

    output_path = Path(args.output)
    write_compound_task_dataset(records, output_path)
    print(f"Saved {len(records)} compound eval records to {output_path}")


if __name__ == "__main__":
    main()
