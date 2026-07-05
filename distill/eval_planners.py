"""Stage 4: evaluate planners on the compound-task set and compare them.

Two decoupled steps so the GPU-only part is isolated from pure-Python scoring:

  predict : run ONE planner over the eval set -> predictions.jsonl
            (the distilled planner needs GPU + `.[train]`; deterministic is local)
  score   : take one or more predictions.jsonl -> metrics table
            (pure Python, runs anywhere; reuses src/evaluation/metrics.py)

Predictions use the schema the existing metrics expect:
    {"id": ..., "planned_steps": [{"route": ...}, ...], "planner": ..., "latency_s": ...}

Examples
--------
Deterministic baseline (local, no GPU)::

    python -m distill.eval_planners predict --planner deterministic \\
        --out distill/data/eval_pred_deterministic.jsonl

Distilled model (on GPU box)::

    python -m distill.eval_planners predict --planner distilled \\
        --adapter distill/checkpoints/sft-qwen1.5b \\
        --out distill/data/eval_pred_distilled.jsonl

Compare (anywhere)::

    python -m distill.eval_planners score \\
        distill/data/eval_pred_deterministic.jsonl \\
        distill/data/eval_pred_distilled.jsonl

Run as a module (``-m``) from the repo root so ``src`` is importable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.agent.planner import DeterministicRoutePlanner, _plan_step_to_dict
from src.evaluation.dataset import load_eval_dataset
from src.evaluation.metrics import (
    planned_step_accuracy,
    planned_step_order_accuracy,
    policy_final_step_rate,
    required_step_recall,
)

DEFAULT_EVAL = "data/eval/compound_task_eval.jsonl"


def build_planner(name: str, adapter: str | None, no_quantize: bool, max_new_tokens: int = 256):
    if name == "deterministic":
        return DeterministicRoutePlanner()
    if name == "distilled":
        if not adapter:
            raise SystemExit("--adapter is required for --planner distilled")
        from distill.distilled_planner import DistilledRoutePlanner

        return DistilledRoutePlanner(
            adapter_dir=adapter, quantize=not no_quantize, max_new_tokens=max_new_tokens
        )
    if name == "env":
        # DeepSeek / whatever LANGGRAPH_PLANNER_PROVIDER points at (teacher).
        from src.agent.planner import build_route_planner_from_env

        return build_route_planner_from_env()
    raise SystemExit(f"unknown planner: {name!r}")


def run_predict(args) -> None:
    records = load_eval_dataset(args.eval_path)
    planner = build_planner(args.planner, args.adapter, args.no_quantize, args.max_new_tokens)

    # Fast path: distilled planner supports batched GPU decode, which is far
    # faster than the per-item loop. Latency is reported as wall-clock / n
    # (per-item timing is meaningless when a whole batch decodes together).
    if args.fast and args.planner == "distilled":
        rows = _predict_distilled_batched(planner, records, args.batch_size)
    elif args.concurrency > 1:
        # API planners (env/deepseek) are network-bound and safe to run in
        # parallel — a thread pool cuts wall-clock ~Nx without touching results.
        rows = _predict_concurrent(planner, records, args.concurrency)
    else:
        rows = _predict_sequential(planner, records)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    legal = sum(1 for r in rows if not r["fallback_used"] and r["planned_steps"])
    avg_lat = sum(r["latency_s"] for r in rows) / n if n else 0.0
    print(f"[predict] planner={args.planner} n={n}")
    print(f"[predict] non-fallback plans: {legal}/{n} ({legal / n:.1%})" if n else "")
    print(f"[predict] avg latency: {avg_lat:.3f}s")
    print(f"[write] {out}")


def _row_from_decision(rec_id: str, decision, latency_s: float) -> dict:
    return {
        "id": rec_id,
        "planned_steps": [_plan_step_to_dict(s) for s in decision.steps],
        "planner": decision.planner,
        "fallback_used": decision.fallback_used,
        "latency_s": round(latency_s, 4),
    }


def _predict_sequential(planner, records) -> list[dict]:
    rows = []
    for rec in records:
        t0 = time.perf_counter()
        # Do NOT pass task_type: we want the planner to route from the question
        # alone (compound tasks have no single task_type shortcut anyway).
        decision = planner.plan(rec.question)
        rows.append(_row_from_decision(rec.id, decision, time.perf_counter() - t0))
    return rows


def _predict_concurrent(planner, records, concurrency: int) -> list[dict]:
    """Run an API planner over the eval set with a thread pool.

    For network-bound planners (env/deepseek) each ``plan`` call is a blocking
    HTTP request, so threads overlap the waits and cut wall-clock roughly by
    ``concurrency``. Results are re-sorted into the original record order, so
    output is identical to the sequential path aside from ``latency_s`` (which
    becomes per-call wall time, now overlapped with other calls).
    """
    from concurrent.futures import ThreadPoolExecutor

    def _one(rec):
        t0 = time.perf_counter()
        decision = planner.plan(rec.question)
        return rec.id, _row_from_decision(rec.id, decision, time.perf_counter() - t0)

    done = 0
    by_id: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for rec_id, row in pool.map(_one, records):
            by_id[rec_id] = row
            done += 1
            print(f"[predict] {done}/{len(records)} done", flush=True)
    # Preserve original eval order.
    return [by_id[rec.id] for rec in records]


def _predict_distilled_batched(planner, records, batch_size: int) -> list[dict]:
    questions = [rec.question for rec in records]
    n = len(questions)

    def _progress(done: int, total: int) -> None:
        print(f"[predict] {done}/{total} done", flush=True)

    t0 = time.perf_counter()
    decisions = planner.plan_batch(questions, batch_size=batch_size, progress=_progress)
    per_item = (time.perf_counter() - t0) / n if n else 0.0

    rows = []
    for rec, decision in zip(records, decisions):
        rows.append(
            {
                "id": rec.id,
                "planned_steps": [_plan_step_to_dict(s) for s in decision.steps],
                "planner": decision.planner,
                "fallback_used": decision.fallback_used,
                "latency_s": round(per_item, 4),
            }
        )
    return rows


def _load_predictions(path: str) -> tuple[str, dict[str, dict]]:
    preds: dict[str, dict] = {}
    label = Path(path).stem
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            preds[row["id"]] = row
    return label, preds


def run_score(args) -> None:
    records = load_eval_dataset(args.eval_path)

    def pct(x: float | None) -> str:
        return "  n/a" if x is None else f"{x:.1%}"

    header = ["planner", "step_acc", "order_acc", "req_recall", "policy_final", "non_fb", "avg_lat"]
    table = []
    for path in args.predictions:
        label, preds = _load_predictions(path)
        n = len(preds)
        non_fb = sum(1 for p in preds.values() if not p.get("fallback_used", False))
        lats = [p.get("latency_s", 0.0) for p in preds.values()]
        avg_lat = sum(lats) / n if n else 0.0
        table.append([
            label,
            pct(planned_step_accuracy(records, preds)),
            pct(planned_step_order_accuracy(records, preds)),
            pct(required_step_recall(records, preds)),
            pct(policy_final_step_rate(records, preds)),
            f"{non_fb}/{n}",
            f"{avg_lat:.3f}s",
        ])

    widths = [max(len(str(row[i])) for row in [header] + table) for i in range(len(header))]
    def fmt(row):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))

    print(fmt(header))
    print("  ".join("-" * w for w in widths))
    for row in table:
        print(fmt(row))

    if args.report:
        _write_report(args.report, header, table, args.eval_path, len(records))
        print(f"\n[write] {args.report}")


def _write_report(path: str, header, table, eval_path, n_records) -> None:
    lines = [
        "# Planner Distillation — Stage 4 Comparison",
        "",
        f"Eval set: `{eval_path}` ({n_records} records; step metrics use the "
        f"{sum(1 for _ in table)}-way rows below over records that have `expected_steps`).",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in table:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines += [
        "",
        "**指标说明**：",
        "- `step_acc` planned_step_accuracy：预测 route 集合 == 期望集合的比例",
        "- `order_acc` planned_step_order_accuracy：预测 route 顺序完全一致的比例",
        "- `req_recall` required_step_recall：期望 route 被覆盖的平均比例",
        "- `policy_final`：含 policy 的任务中 policy 步落在最后的比例",
        "- `non_fb`：未走 deterministic fallback 的计划数（越高说明该 planner 自身产出越多有效计划）",
        "- `avg_lat`：单条平均规划延迟",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("predict", help="run a planner -> predictions.jsonl")
    pp.add_argument("--planner", required=True, choices=["deterministic", "distilled", "env"])
    pp.add_argument("--adapter", default=None, help="LoRA adapter dir (for --planner distilled)")
    pp.add_argument("--no-quantize", action="store_true")
    pp.add_argument(
        "--fast",
        action="store_true",
        help="batched GPU decode for --planner distilled (much faster; "
        "latency reported as wall-clock/n)",
    )
    pp.add_argument("--batch-size", type=int, default=16, help="batch size for --fast")
    pp.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="parallel requests for API planners (env/deepseek); network-bound "
        "so >1 cuts wall-clock. Ignored by the distilled --fast path.",
    )
    pp.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="cap on generated tokens; plans are ~130 tokens max so 160 is safe "
        "and cuts decode time on cap-bound batches",
    )
    pp.add_argument("--eval-path", default=DEFAULT_EVAL)
    pp.add_argument("--out", required=True)
    pp.set_defaults(func=run_predict)

    ps = sub.add_parser("score", help="predictions.jsonl... -> metrics table")
    ps.add_argument("predictions", nargs="+", help="one or more predictions.jsonl")
    ps.add_argument("--eval-path", default=DEFAULT_EVAL)
    ps.add_argument("--report", default=None, help="optional markdown report output path")
    ps.set_defaults(func=run_score)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
