"""Export supervised fine-tuning (SFT) data for planner distillation.

This script turns questions from the existing evaluation datasets into
`{messages, completion}` training samples by running the project's own route
planner as the teacher and keeping only plans that pass `validate_plan_steps`.

Key design choices (see distillation_plan.md, stage 1):
- The prompt is built with `build_planner_messages`, the SAME function the
  online `LLMRoutePlanner` uses, so the student model trains on exactly the
  input format it will see at inference time.
- The label is produced by `serialize_plan_steps`, matching the JSON schema the
  online planner parses back in `_decision_from_llm_payload`.
- Only schema-valid plans are kept, so label quality is guaranteed by the same
  guard that protects the live system.

It does not modify any production code and needs no GPU.

Examples
--------
Deterministic teacher (offline, no API), all three seed datasets::

    python -m distill.build_sft_data

DeepSeek teacher (requires DEEPSEEK_API_KEY in .env), single dataset::

    python -m distill.build_sft_data --teacher deepseek \\
        --eval-path data/eval/real_eval.jsonl

Run as a module (``-m``) from the repo root so ``src`` is importable.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.agent.planner import (
    DeterministicRoutePlanner,
    RoutePlanner,
    build_planner_messages,
    build_route_planner_from_env,
    serialize_plan_steps,
    validate_plan_steps,
)
from src.evaluation.dataset import load_eval_dataset

DEFAULT_EVAL_PATHS = [
    "data/eval/real_eval.jsonl",
    "data/eval/hvac_eval.jsonl",
    "data/eval/compound_task_eval.jsonl",
]


def build_teacher(name: str) -> RoutePlanner:
    """Return the teacher planner used to generate labels.

    ``deterministic`` uses the offline rule-based planner (no API, fully
    reproducible). ``deepseek`` uses the env-configured LLM planner, which
    already falls back to deterministic on any error.
    """
    if name == "deterministic":
        return DeterministicRoutePlanner()
    if name == "deepseek":
        return build_route_planner_from_env()
    raise ValueError(f"unknown teacher: {name!r} (expected 'deterministic' or 'deepseek')")


def _dedupe_questions(paths: list[str]) -> list[dict[str, str]]:
    """Load questions from eval datasets, de-duplicated by question text."""
    seen: set[str] = set()
    items: list[dict[str, str]] = []
    for path in paths:
        if not Path(path).exists():
            print(f"[skip] dataset not found: {path}")
            continue
        for record in load_eval_dataset(path):
            question = record.question.strip()
            if not question or question in seen:
                continue
            seen.add(question)
            # task_type is intentionally NOT passed to the planner: we want the
            # student to learn routing from the question alone, mirroring the
            # online path where free-form questions have no eval task_type.
            items.append({"id": record.id, "question": question})
    return items


def _load_augmented_questions(path: str, seen: set[str]) -> list[dict[str, str]]:
    """Load plain {id, question} rows produced by augment_questions.py."""
    items: list[dict[str, str]] = []
    if not Path(path).exists():
        print(f"[skip] augmented questions not found: {path}")
        return items
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        question = str(row.get("question", "")).strip()
        if not question or question in seen:
            continue
        seen.add(question)
        items.append({"id": str(row.get("id", question[:24])), "question": question})
    return items


def build_samples(
    questions: list[dict[str, str]],
    teacher: RoutePlanner,
) -> tuple[list[dict], dict[str, int]]:
    """Run the teacher over questions and keep only schema-valid plans."""
    samples: list[dict] = []
    stats = {"total": 0, "kept": 0, "invalid": 0, "planner_error": 0}
    for item in questions:
        stats["total"] += 1
        question = item["question"]
        try:
            decision = teacher.plan(question)
        except Exception as exc:  # teacher itself failed on this question
            stats["planner_error"] += 1
            print(f"[planner-error] {item['id']}: {exc}")
            continue
        try:
            valid_steps = validate_plan_steps(list(decision.steps))
        except ValueError as exc:  # plan did not pass the shared guard
            stats["invalid"] += 1
            print(f"[invalid-plan] {item['id']}: {exc}")
            continue
        samples.append(
            {
                "id": item["id"],
                "messages": build_planner_messages(question),
                "completion": serialize_plan_steps(valid_steps, decision.confidence),
                "teacher": decision.planner,
            }
        )
        stats["kept"] += 1
    return samples, stats


def split_train_val(
    samples: list[dict],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    val_size = int(len(shuffled) * val_ratio)
    return shuffled[val_size:], shuffled[:val_size]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-path",
        action="append",
        dest="eval_paths",
        help="Eval JSONL to draw questions from (repeatable). "
        "Defaults to real_eval + hvac_eval + compound_task_eval.",
    )
    parser.add_argument(
        "--teacher",
        default="deterministic",
        choices=["deterministic", "deepseek"],
        help="Planner used to generate labels (default: deterministic, offline).",
    )
    parser.add_argument(
        "--questions-path",
        help="Optional JSONL of augmented {id, question} rows "
        "(from augment_questions.py) to merge in alongside the eval seeds.",
    )
    parser.add_argument("--out-dir", default="distill/data")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    eval_paths = args.eval_paths or DEFAULT_EVAL_PATHS
    questions = _dedupe_questions(eval_paths)
    seen_texts = {item["question"] for item in questions}
    if args.questions_path:
        augmented = _load_augmented_questions(args.questions_path, seen_texts)
        questions.extend(augmented)
        print(f"[load] +{len(augmented)} augmented questions from {args.questions_path}")
    print(f"[load] {len(questions)} unique questions total")

    teacher = build_teacher(args.teacher)
    samples, stats = build_samples(questions, teacher)
    print(
        f"[build] kept {stats['kept']}/{stats['total']} "
        f"(invalid={stats['invalid']}, planner_error={stats['planner_error']})"
    )
    if not samples:
        raise SystemExit("no valid samples produced; aborting")

    train, val = split_train_val(samples, args.val_ratio, args.seed)
    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "sft_train.jsonl", train)
    write_jsonl(out_dir / "sft_val.jsonl", val)

    valid_rate = stats["kept"] / stats["total"] if stats["total"] else 0.0
    card = {
        "eval_paths": eval_paths,
        "questions_path": args.questions_path,
        "teacher": args.teacher,
        "unique_questions": len(questions),
        "kept": stats["kept"],
        "invalid": stats["invalid"],
        "planner_error": stats["planner_error"],
        "valid_rate": round(valid_rate, 4),
        "train_size": len(train),
        "val_size": len(val),
        "prompt_source": "src.agent.planner.build_planner_messages",
        "label_source": "src.agent.planner.serialize_plan_steps",
    }
    (out_dir / "sft_data_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[write] {out_dir}/sft_train.jsonl ({len(train)}), sft_val.jsonl ({len(val)})")
    print(f"[write] {out_dir}/sft_data_card.json (valid_rate={valid_rate:.2%})")


if __name__ == "__main__":
    main()
