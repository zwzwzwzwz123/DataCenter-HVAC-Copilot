"""Augment planner questions for distillation (rule-based, no LLM, no token cost).

This produces *questions only*. Labels are generated separately by the teacher
planner in ``build_sft_data.py`` so there is a single source of truth for label
format. Generating questions needs no model call, so scaling to thousands of
questions is free; the (batched) teacher is only invoked at labeling time.

Two augmentation strategies:

1. Paraphrase seeds — take questions from the existing eval datasets and expand
   each with colloquial prefixes and light synonym swaps, staying in-domain.
2. Template synthesis — combine each tool's real trigger keywords (from the
   ToolSpec registry) with metrics, time windows, and zones to cover routing
   cases the seed set may miss.

Examples
--------
    python -m distill.augment_questions                       # both strategies
    python -m distill.augment_questions --strategy template   # coverage only
    python -m distill.augment_questions --max 2000            # cap output size

Run as a module (``-m``) from the repo root so ``src`` is importable.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

from src.evaluation.dataset import load_eval_dataset

DEFAULT_SEED_PATHS = [
    "data/eval/real_eval.jsonl",
    "data/eval/hvac_eval.jsonl",
    "data/eval/compound_task_eval.jsonl",
]

# Colloquial lead-ins that do not change routing intent.
PREFIXES = [
    "",
    "帮我看看",
    "麻烦分析一下",
    "能不能告诉我",
    "我想知道",
    "请问",
    "Could you check",
    "Please analyze",
]

# Metrics and zones actually present in the BEAR rollout / registry defaults.
METRICS = [
    "zone_temperature",
    "cooling_power",
    "fan_power",
    "outdoor_temp",
    "control_action",
    "internal_load",
]
ZONES = ["", "zone_0", "zone_1", "zone_2"]
TIME_WINDOWS = ["", "最近", "last_24_hours", "full_demo_range"]

# One representative question stem per tool, seeded with that tool's real
# trigger keywords so the deterministic router lands on the intended tool.
# The teacher planner still produces the actual label; these only shape routing.
TOOL_STEMS = {
    "query_metric": "{metric} 的平均值和最大值是多少",
    "compare_period": "{metric} 前后两个时段有没有明显变化对比",
    "plot_metric_trend": "画一下 {metric} 的趋势折线图",
    "compute_energy_breakdown": "{metric} 的能耗构成是怎样的",
    "data_quality_check": "{metric} 数据有没有缺失或字段质量问题",
    "comfort_risk_assessment": "{zone} 舒适度有没有过热越限风险",
    "zone_hotspot_rank": "哪个区域是最热的 hotspot 排名",
    "control_action_audit": "control_action 控制动作有没有震荡抖动",
    "cooling_efficiency_summary": "制冷能效和 power 效率如何",
    "detect_anomaly": "{metric} 最近有没有异常告警",
    "rag_retrieval": "文档里关于{topic}是怎么说的",
    "policy_runner": "针对当前情况给出降温策略建议",
}
DOC_TOPICS = ["ASHRAE 温度上限", "冷通道封闭", "PUE 优化", "机房送风温度", "数据中心节能"]


def load_seed_questions(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        if not Path(path).exists():
            print(f"[skip] seed dataset not found: {path}")
            continue
        for record in load_eval_dataset(path):
            q = record.question.strip()
            if q and q not in seen:
                seen.add(q)
                out.append(q)
    return out


def paraphrase(questions: list[str], rng: random.Random, variants: int) -> list[str]:
    out: list[str] = []
    for q in questions:
        chosen = rng.sample(PREFIXES, k=min(variants, len(PREFIXES)))
        for prefix in chosen:
            if not prefix:
                out.append(q)
            elif prefix[0].isascii():
                out.append(f"{prefix}: {q}")
            else:
                out.append(f"{prefix}{q}")
    return out


def synthesize_from_templates() -> list[str]:
    out: list[str] = []
    for tool, stem in TOOL_STEMS.items():
        if tool == "rag_retrieval":
            for topic in DOC_TOPICS:
                out.append(stem.format(topic=topic))
            continue
        if tool == "policy_runner":
            out.append(stem)
            continue
        needs_metric = "{metric}" in stem
        needs_zone = "{zone}" in stem
        metrics = METRICS if needs_metric else [""]
        zones = ZONES if needs_zone else [""]
        for metric, zone, window in itertools.product(metrics, zones, TIME_WINDOWS):
            text = stem.format(metric=metric, zone=zone or "机房整体").strip()
            if window:
                text = f"{window} {text}" if window[0].isascii() else f"{window}{text}"
            out.append(text)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-path", action="append", dest="seed_paths")
    parser.add_argument(
        "--strategy",
        choices=["both", "paraphrase", "template"],
        default="both",
    )
    parser.add_argument("--variants", type=int, default=4, help="paraphrases per seed")
    parser.add_argument("--max", type=int, default=0, help="cap total questions (0=no cap)")
    parser.add_argument("--out", default="distill/data/questions.jsonl")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seed_paths = args.seed_paths or DEFAULT_SEED_PATHS

    questions: list[str] = []
    if args.strategy in {"both", "paraphrase"}:
        seeds = load_seed_questions(seed_paths)
        para = paraphrase(seeds, rng, args.variants)
        questions.extend(para)
        print(f"[paraphrase] {len(seeds)} seeds -> {len(para)} questions")
    if args.strategy in {"both", "template"}:
        templ = synthesize_from_templates()
        questions.extend(templ)
        print(f"[template] synthesized {len(templ)} questions")

    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique = [q for q in questions if not (q in seen or seen.add(q))]
    rng.shuffle(unique)
    if args.max and len(unique) > args.max:
        unique = unique[: args.max]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for i, q in enumerate(unique):
            handle.write(json.dumps({"id": f"aug_{i:05d}", "question": q}, ensure_ascii=False) + "\n")
    print(f"[write] {out_path} ({len(unique)} unique questions)")


if __name__ == "__main__":
    main()
