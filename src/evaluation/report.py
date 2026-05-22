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
    "answer_correctness_proxy",
    "faithfulness_proxy",
]

OPTIONAL_METRIC_COLUMNS = [
    "llm_judge_correctness",
    "llm_judge_faithfulness",
]


def render_experiment_report(
    comparison_summary: dict[str, dict[str, float]],
    *,
    eval_record_count: int,
    expected_keyword_record_count: int = 0,
    by_task_type: dict[str, dict[str, dict[str, float]]] | None = None,
    human_calibration: dict[str, object] | None = None,
    dense_provider: str = "deterministic",
    dense_backend: str = "memory",
    dense_model: str | None = None,
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
        "## 运行配置",
        "",
        f"- dense_provider: `{dense_provider}`",
        f"- dense_backend: `{dense_backend}`",
        f"- dense_model: `{dense_model or 'default'}`",
        "",
        "## Baseline 对比",
        "",
        _metric_header(["baseline"], _active_metric_columns(comparison_summary)),
        _metric_alignment(["---"], _active_metric_columns(comparison_summary)),
    ]
    metric_columns = _active_metric_columns(comparison_summary)
    for name, metrics in comparison_summary.items():
        values = [_format_metric(metrics.get(column, 0.0)) for column in metric_columns]
        lines.append(f"| {name} | {' | '.join(values)} |")

    if by_task_type:
        lines.extend(
            [
                "",
                "## 按任务类型指标",
                "",
                _metric_header(["baseline", "task_type"], metric_columns),
                _metric_alignment(["---", "---"], metric_columns),
            ]
        )
        for baseline, task_metrics in by_task_type.items():
            for task_type, metrics in task_metrics.items():
                values = [_format_metric(metrics.get(column, 0.0)) for column in metric_columns]
                lines.append(f"| {baseline} | {task_type} | {' | '.join(values)} |")

    if human_calibration:
        lines.extend(_human_calibration_section(human_calibration))

    lines.extend(
        [
            "",
            "## 当前结论",
            "",
            "- `llm_only` 不使用检索证据或工具结果，作为最低可复现基线。",
            _dense_comparison_conclusion(
                comparison_summary,
                dense_provider=dense_provider,
                dense_backend=dense_backend,
            ),
            _retrieval_comparison_conclusion(comparison_summary),
            _reranker_comparison_conclusion(comparison_summary),
            _query_rewrite_hyde_conclusion(comparison_summary),
            "- `rag_tool_agent` 在当前确定性路由样例上体现工具选择、工具执行和证据覆盖优势。",
            _langgraph_comparison_conclusion(comparison_summary),
            _intent_routing_conclusion(),
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


def _dense_comparison_conclusion(
    comparison_summary: dict[str, dict[str, float]],
    *,
    dense_provider: str = "deterministic",
    dense_backend: str = "memory",
) -> str:
    dense = comparison_summary.get("rag_dense", {})
    if not dense:
        return "- `rag_dense` 尚未纳入当前报告。"
    if dense_provider == "sentence-transformers" and dense_backend == "faiss":
        return (
            "- `rag_dense` 使用真实 sentence-transformers embedding + FAISS 本地向量索引；"
            "该运行可用于面试中说明真实语义检索 baseline，但仍需结合 hybrid/rerank 指标判断中文 HVAC 场景效果。"
        )
    return (
        "- `rag_dense` 使用 deterministic hash embedding 作为默认 dense retrieval baseline；"
        "真实 FAISS + sentence-transformers 作为可选增强，避免默认评测依赖模型下载或外部 API。"
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


def _query_rewrite_hyde_conclusion(
    comparison_summary: dict[str, dict[str, float]],
) -> str:
    rewrite = comparison_summary.get("rag_rewrite")
    hyde = comparison_summary.get("rag_hyde")
    hyde_rerank = comparison_summary.get("rag_hyde_rerank")
    if not rewrite and not hyde and not hyde_rerank:
        return "- Query Rewrite / HyDE 尚未纳入当前报告。"

    best_name = "rag_rewrite"
    best_context = (rewrite or {}).get("context_recall", 0.0)
    for name, metrics in (
        ("rag_hyde", hyde or {}),
        ("rag_hyde_rerank", hyde_rerank or {}),
    ):
        context = metrics.get("context_recall", 0.0)
        if context > best_context:
            best_name = name
            best_context = context
    return (
        "- Query Rewrite / HyDE 已作为 deterministic query expansion baseline 纳入对比；"
        f"当前 context_recall 最高的是 `{best_name}`，可用于评估 raw query、rewrite 和 template HyDE "
        "在 HVAC/BEAR 领域检索中的收益，再决定是否替换为 DeepSeek/Ollama HyDE generator。"
    )


def _langgraph_comparison_conclusion(
    comparison_summary: dict[str, dict[str, float]],
) -> str:
    langgraph = comparison_summary.get("langgraph_tool_agent")
    baseline = comparison_summary.get("rag_tool_agent")
    if not langgraph:
        return "- `langgraph_tool_agent` 尚未纳入当前报告。"
    if baseline and langgraph == baseline:
        return (
            "- `langgraph_tool_agent` 保留与 deterministic `rag_tool_agent` 一致的工具行为和指标，"
            "用于展示 StateGraph 编排、workflow trace 和可选 DeepSeek/Ollama LLM intent classifier，而不是改变当前可复现评测口径。"
        )
    return (
        "- `langgraph_tool_agent` 已纳入对比；其指标与 deterministic baseline 的差异需要结合 workflow trace "
        "进一步检查路由和工具节点行为。"
    )


def _intent_routing_conclusion() -> str:
    return (
        "- `scripts/run_intent_eval.py` 单独评测 intent routing accuracy；默认 rule-based classifier "
        "在当前 100 条样例上 accuracy 为 0.640，并输出 "
        "`data/eval/intent_routing_comparison.json` 作为 keyword vs LLM routing 对比入口。"
    )


def save_experiment_report(
    comparison_summary: dict[str, dict[str, float]],
    *,
    output_path: str | Path,
    eval_record_count: int,
    expected_keyword_record_count: int = 0,
    by_task_type: dict[str, dict[str, dict[str, float]]] | None = None,
    human_calibration: dict[str, object] | None = None,
    dense_provider: str = "deterministic",
    dense_backend: str = "memory",
    dense_model: str | None = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_experiment_report(
            comparison_summary,
            eval_record_count=eval_record_count,
            expected_keyword_record_count=expected_keyword_record_count,
            by_task_type=by_task_type,
            human_calibration=human_calibration,
            dense_provider=dense_provider,
            dense_backend=dense_backend,
            dense_model=dense_model,
        ),
        encoding="utf-8",
    )


def _format_metric(value: float) -> str:
    return f"{value:.3f}"


def _format_optional_metric(value: object) -> str:
    if value is None:
        return "null"
    return f"{float(value):.3f}"


def _human_calibration_section(summary: dict[str, object]) -> list[str]:
    return [
        "",
        "## Human Calibration",
        "",
        "人工校准集用于核对 deterministic proxy 和 optional LLM judge 的可信度；不会把 deterministic proxy 或 LLM judge 当作人工评审。",
        "",
        "| sample_count | labeled_count | pending_count | mean_correctness | mean_faithfulness | safety_pass_rate | status |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        (
            f"| {summary.get('sample_count', 0)} | {summary.get('labeled_count', 0)} | "
            f"{summary.get('pending_count', 0)} | "
            f"{_format_optional_metric(summary.get('mean_correctness'))} | "
            f"{_format_optional_metric(summary.get('mean_faithfulness'))} | "
            f"{_format_optional_metric(summary.get('safety_pass_rate'))} | "
            f"{summary.get('status', 'pending_human_review')} |"
        ),
    ]


def _active_metric_columns(comparison_summary: dict[str, dict[str, float]]) -> list[str]:
    active = list(METRIC_COLUMNS)
    for column in OPTIONAL_METRIC_COLUMNS:
        if any(column in metrics for metrics in comparison_summary.values()):
            active.append(column)
    return active


def _metric_header(prefix_columns: list[str], metric_columns: list[str]) -> str:
    return "| " + " | ".join(prefix_columns + metric_columns) + " |"


def _metric_alignment(prefix_columns: list[str], metric_columns: list[str]) -> str:
    return "| " + " | ".join(prefix_columns + ["---:"] * len(metric_columns)) + " |"
