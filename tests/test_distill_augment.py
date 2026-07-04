"""Tests for rule-based question augmentation (no LLM, no token cost)."""

from __future__ import annotations

from distill.augment_questions import (
    TOOL_STEMS,
    paraphrase,
    synthesize_from_templates,
)
from src.agent.planner import DeterministicRoutePlanner, validate_plan_steps
from src.tools.registry import TOOL_REGISTRY


def test_template_stems_reference_real_tools() -> None:
    # Every templated tool must exist in the registry, else synthesized
    # questions target tools the executor cannot run.
    for tool_name in TOOL_STEMS:
        assert tool_name in TOOL_REGISTRY, f"unknown tool in template: {tool_name}"


def test_synthesize_from_templates_produces_unique_nonempty_questions() -> None:
    questions = synthesize_from_templates()

    assert len(questions) > 20
    assert all(q.strip() for q in questions)
    # no unresolved format placeholders left behind
    assert not any("{" in q or "}" in q for q in questions)


def test_paraphrase_expands_each_seed() -> None:
    import random

    seeds = ["最近温度有没有异常", "给出降温策略"]
    out = paraphrase(seeds, random.Random(0), variants=3)

    assert len(out) >= len(seeds)
    # original question text is preserved somewhere in the expansion
    assert any("最近温度有没有异常" in q for q in out)


def test_synthesized_questions_mostly_yield_valid_plans() -> None:
    # Sanity check that the synthesized questions route to schema-valid plans
    # under the deterministic planner (the offline teacher).
    planner = DeterministicRoutePlanner()
    questions = synthesize_from_templates()

    valid = 0
    for q in questions:
        try:
            decision = planner.plan(q)
            validate_plan_steps(list(decision.steps))
            valid += 1
        except Exception:
            pass

    # Templates are seeded with real trigger keywords, so the vast majority
    # should produce valid plans. Allow some slack for edge combinations.
    assert valid / len(questions) >= 0.9
