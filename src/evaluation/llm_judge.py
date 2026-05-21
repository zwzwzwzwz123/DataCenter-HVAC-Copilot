from __future__ import annotations

from typing import Protocol


class LLMJudge(Protocol):
    def judge(
        self,
        *,
        question: str,
        answer: str,
        gold_answer: str,
        expected_keywords: list[str],
        evidence_texts: list[str],
    ) -> dict[str, float | str]:
        """Return optional semantic quality scores for one prediction."""


class DeterministicKeywordJudge:
    """Local fake judge for tests and deterministic smoke runs."""

    judge_name = "deterministic_keyword_judge"

    def judge(
        self,
        *,
        question: str,
        answer: str,
        gold_answer: str,
        expected_keywords: list[str],
        evidence_texts: list[str],
    ) -> dict[str, float | str]:
        correctness = _coverage(answer, expected_keywords)
        evidence_text = "\n".join(evidence_texts)
        faithfulness = _coverage(evidence_text, expected_keywords)
        return {
            "judge_name": self.judge_name,
            "correctness": correctness,
            "faithfulness": faithfulness,
            "notes": (
                "Deterministic keyword judge for optional adapter tests; "
                "not a replacement for human review."
            ),
        }


def _coverage(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    hits = sum(1 for keyword in keywords if keyword and keyword in text)
    return hits / len(keywords)
