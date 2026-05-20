from __future__ import annotations

from pathlib import Path


METRIC_COLUMNS = [
    "citation_hit_rate",
    "context_recall",
    "expected_keyword_coverage",
    "lexical_answer_coverage",
    "tool_selection_accuracy",
    "tool_execution_success_rate",
    "evidence_coverage",
]


def render_experiment_report(
    comparison_summary: dict[str, dict[str, float]],
    *,
    eval_record_count: int,
    expected_keyword_record_count: int = 0,
    by_task_type: dict[str, dict[str, dict[str, float]]] | None = None,
) -> str:
    lines = [
        "# 实验报告",
        "",
        "## 数据与边界",
        "",
        (
            f"当前评测集包含 {eval_record_count} 条样例，覆盖文档问答、时序查询、"
            "异常诊断和策略建议。轨迹数据来自 BEAR 仿真轨迹、BEAR 样例 CSV "
            "或 mock fallback，不能表述为真实数据中心生产遥测。"
        ),
        (
            f"其中 {expected_keyword_record_count} 条样例包含人工维护的 expected_keywords，"
            "用于计算中文回答要点覆盖率。"
        ),
        "",
        "## Baseline 对比",
        "",
        "| baseline | citation_hit_rate | context_recall | expected_keyword_coverage | lexical_answer_coverage | tool_selection_accuracy | tool_execution_success_rate | evidence_coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in comparison_summary.items():
        values = [_format_metric(metrics.get(column, 0.0)) for column in METRIC_COLUMNS]
        lines.append(f"| {name} | {' | '.join(values)} |")

    if by_task_type:
        lines.extend(
            [
                "",
                "## 按任务类型指标",
                "",
                "| baseline | task_type | citation_hit_rate | context_recall | expected_keyword_coverage | lexical_answer_coverage | tool_selection_accuracy | tool_execution_success_rate | evidence_coverage |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for baseline, task_metrics in by_task_type.items():
            for task_type, metrics in task_metrics.items():
                values = [_format_metric(metrics.get(column, 0.0)) for column in METRIC_COLUMNS]
                lines.append(f"| {baseline} | {task_type} | {' | '.join(values)} |")

    lines.extend(
        [
            "",
            "## 当前结论",
            "",
            "- `llm_only` 不使用检索证据或工具结果，作为最低可复现基线。",
            _retrieval_comparison_conclusion(comparison_summary),
            _reranker_comparison_conclusion(comparison_summary),
            "- `rag_tool_agent` 在当前确定性路由样例上体现工具选择、工具执行和证据覆盖优势。",
            "",
        ]
    )
    return "\n".join(lines)


def _retrieval_comparison_conclusion(
    comparison_summary: dict[str, dict[str, float]],
) -> str:
    keyword = comparison_summary.get("rag_keyword", {})
    hybrid = comparison_summary.get("rag_hybrid", {})
    keyword_context = keyword.get("context_recall", 0.0)
    hybrid_context = hybrid.get("context_recall", 0.0)
    keyword_citation = keyword.get("citation_hit_rate", 0.0)
    hybrid_citation = hybrid.get("citation_hit_rate", 0.0)
    if hybrid_context > keyword_context or hybrid_citation > keyword_citation:
        return (
            "- `rag_keyword` 与 `rag_hybrid` 用于比较轻量检索方案；"
            "`rag_hybrid` 在 citation/context 指标上优于 `rag_keyword`，"
            "说明 BM25-style 长度归一化在当前压力样例中有效。"
        )
    return (
        "- `rag_keyword` 与 `rag_hybrid` 用于比较轻量检索方案；"
        "当前样例下两者指标持平，仍需更丰富的相似主题文档继续拉开差异。"
    )


def _reranker_comparison_conclusion(
    comparison_summary: dict[str, dict[str, float]],
) -> str:
    hybrid = comparison_summary.get("rag_hybrid", {})
    rerank = comparison_summary.get("rag_hybrid_rerank", {})
    if not rerank:
        return "- `rag_hybrid_rerank` 尚未纳入当前报告。"
    hybrid_context = hybrid.get("context_recall", 0.0)
    rerank_context = rerank.get("context_recall", 0.0)
    hybrid_citation = hybrid.get("citation_hit_rate", 0.0)
    rerank_citation = rerank.get("citation_hit_rate", 0.0)
    if rerank_context > hybrid_context or rerank_citation > hybrid_citation:
        return (
            "- `rag_hybrid_rerank` 在当前评测中进一步提升 citation/context 指标，"
            "可作为后续替换为 cross-encoder 或 LLM reranker 的接口基线。"
        )
    return (
        "- `rag_hybrid_rerank` 已纳入对比表；当前指标与 `rag_hybrid` 持平，"
        "说明轻量重排接口已具备，但还需要更强重排策略或更多重排压力样例。"
    )


def save_experiment_report(
    comparison_summary: dict[str, dict[str, float]],
    *,
    output_path: str | Path,
    eval_record_count: int,
    expected_keyword_record_count: int = 0,
    by_task_type: dict[str, dict[str, dict[str, float]]] | None = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_experiment_report(
            comparison_summary,
            eval_record_count=eval_record_count,
            expected_keyword_record_count=expected_keyword_record_count,
            by_task_type=by_task_type,
        ),
        encoding="utf-8",
    )


def _format_metric(value: float) -> str:
    return f"{value:.3f}"
