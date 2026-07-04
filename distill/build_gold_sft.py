"""Convert hand-labeled planner examples into validated SFT data.

Option-2 workflow (hand labeling): a human authors high-quality
``{id, question, steps}`` rows in ``gold_labeled.jsonl`` where ``steps`` is a
list of plan-step dicts. This script validates every label through the SAME
``validate_plan_steps`` guard the live system uses, then emits training rows in
the exact ``{messages, completion}`` format produced by ``build_sft_data.py``.

Any row whose steps do not pass the guard is reported and skipped, so a typo in
a route/tool/time_window can never leak into the training set.

Example
-------
    python -m distill.build_gold_sft
    python -m distill.build_gold_sft --gold distill/data/gold_labeled.jsonl

Run as a module (``-m``) from the repo root so ``src`` is importable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent.planner import (
    PlanStep,
    build_planner_messages,
    serialize_plan_steps,
    validate_plan_steps,
)

# reuse split/write helpers so gold and teacher data share one code path
from distill.build_sft_data import split_train_val, write_jsonl

_STEP_FIELDS = {"route", "reason", "tool", "metric_name", "zone_id", "time_window"}


def _step_from_dict(raw: dict) -> PlanStep:
    unknown = set(raw) - _STEP_FIELDS
    if unknown:
        raise ValueError(f"unknown step fields: {sorted(unknown)}")
    if "route" not in raw:
        raise ValueError("step is missing required 'route'")
    return PlanStep(
        route=raw["route"],
        reason=raw.get("reason", "hand-labeled plan step"),
        tool=raw.get("tool"),
        metric_name=raw.get("metric_name"),
        zone_id=raw.get("zone_id"),
        time_window=raw.get("time_window"),
    )


def load_gold(path: str) -> tuple[list[dict], dict[str, int]]:
    samples: list[dict] = []
    stats = {"total": 0, "kept": 0, "invalid": 0}
    for line_no, line in enumerate(
        Path(path).read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        stats["total"] += 1
        row = json.loads(stripped)
        question = str(row.get("question", "")).strip()
        raw_steps = row.get("steps", [])
        try:
            steps = validate_plan_steps([_step_from_dict(s) for s in raw_steps])
            if not question:
                raise ValueError("empty question")
        except ValueError as exc:
            stats["invalid"] += 1
            print(f"[invalid] line {line_no} id={row.get('id')}: {exc}")
            continue
        samples.append(
            {
                "id": str(row.get("id", f"gold_{line_no:04d}")),
                "messages": build_planner_messages(question),
                "completion": serialize_plan_steps(steps, row.get("confidence", 1.0)),
                "teacher": "hand_labeled",
            }
        )
        stats["kept"] += 1
    return samples, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="distill/data/gold_labeled.jsonl")
    parser.add_argument("--out-dir", default="distill/data")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    samples, stats = load_gold(args.gold)
    print(f"[gold] kept {stats['kept']}/{stats['total']} (invalid={stats['invalid']})")
    if not samples:
        raise SystemExit("no valid gold samples; aborting")

    train, val = split_train_val(samples, args.val_ratio, args.seed)
    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "gold_sft_train.jsonl", train)
    write_jsonl(out_dir / "gold_sft_val.jsonl", val)

    card = {
        "gold_path": args.gold,
        "teacher": "hand_labeled",
        "kept": stats["kept"],
        "invalid": stats["invalid"],
        "train_size": len(train),
        "val_size": len(val),
        "prompt_source": "src.agent.planner.build_planner_messages",
        "label_source": "src.agent.planner.serialize_plan_steps",
    }
    (out_dir / "gold_sft_data_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[write] {out_dir}/gold_sft_train.jsonl ({len(train)}), gold_sft_val.jsonl ({len(val)})")


if __name__ == "__main__":
    main()
