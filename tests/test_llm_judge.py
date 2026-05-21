from __future__ import annotations

from src.evaluation.llm_judge import DeterministicKeywordJudge


def test_deterministic_keyword_judge_scores_expected_keywords_and_faithfulness() -> None:
    judge = DeterministicKeywordJudge()

    result = judge.judge(
        question="为什么不能把 BEAR 说成真实生产数据？",
        answer="BEAR 是 HVAC 仿真轨迹，是可控代理场景，不能说成真实生产数据。",
        gold_answer="应说明 BEAR 是仿真轨迹。",
        expected_keywords=["BEAR", "仿真", "可控代理场景"],
        evidence_texts=["BEAR 是 HVAC 仿真轨迹，可作为可控代理场景。"],
    )

    assert result["judge_name"] == "deterministic_keyword_judge"
    assert result["correctness"] == 1.0
    assert result["faithfulness"] == 1.0
    assert result["notes"]
