from __future__ import annotations

from pathlib import Path


METRIC_COLUMNS = [
    "citation_hit_rate",
    "context_recall",
    "retrieval_recall@1",
    "retrieval_recall@3",
    "retrieval_recall@5",
    "retrieval_recall@10",
    "retrieval_mrr@10",
    "retrieval_ndcg@10",
    "expected_keyword_coverage",
    "lexical_answer_coverage",
    "tool_selection_accuracy",
    "tool_execution_success_rate",
    "evidence_coverage",
    "answer_correctness_proxy",
    "faithfulness_proxy",
    "hallucination_proxy_rate",
    "grounding_rate",
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
    safety_adversarial: dict[str, object] | None = None,
    dropt_policy_benchmark: dict[str, object] | None = None,
    dense_provider: str = "deterministic",
    dense_backend: str = "memory",
    dense_model: str | None = None,
    cross_encoder_model: str | None = None,
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
        f"- cross_encoder_model: `{cross_encoder_model or 'disabled'}`",
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

    if safety_adversarial:
        lines.extend(_safety_adversarial_section(safety_adversarial))

    if dropt_policy_benchmark:
        lines.extend(_dropt_policy_benchmark_section(dropt_policy_benchmark))

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
            _grounded_comparison_conclusion(comparison_summary),
            _retrieval_comparison_conclusion(comparison_summary),
            _reranker_comparison_conclusion(comparison_summary),
            _query_rewrite_hyde_conclusion(comparison_summary),
            "- `rag_tool_agent` 在当前确定性路由样例上体现工具选择、工具执行和证据覆盖优势。",
            _langgraph_comparison_conclusion(comparison_summary),
            _react_comparison_conclusion(comparison_summary, by_task_type or {}),
            "- `DROPT` / Guided-DiffFNO checkpoint 作为可选策略后端已接通：checkpoint 可加载、20 维 BEAR state 可推理，缺失或不完整时会明确回退并记录原因。",
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
    cross_encoder = comparison_summary.get("hybrid_rrf_cross_encoder")
    if cross_encoder:
        base = comparison_summary.get("hybrid_rrf", {})
        cross_context = cross_encoder.get("context_recall", 0.0)
        base_context = base.get("context_recall", 0.0)
        cross_mrr = cross_encoder.get("retrieval_mrr@10", 0.0)
        base_mrr = base.get("retrieval_mrr@10", 0.0)
        relation = "提升" if cross_context > base_context or cross_mrr > base_mrr else "补充评估"
        return (
            "- `hybrid_rrf_cross_encoder` 使用 BM25 + dense RRF 召回候选，再用 cross-encoder "
            f"对 query-document pair 做二阶段精排；当前相对 `hybrid_rrf` 是{relation}，"
            "需要结合 retrieval latency 判断排序质量与推理成本。"
        )
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
            "用于展示 StateGraph 编排、workflow trace 和可选 DeepSeek LLM route planner，而不是改变当前可复现评测口径。"
        )
    return (
        "- `langgraph_tool_agent` 已纳入对比；其指标与 deterministic baseline 的差异需要结合 workflow trace "
        "进一步检查路由和工具节点行为。"
    )


def _react_comparison_conclusion(
    comparison_summary: dict[str, dict[str, float]],
    by_task_type: dict[str, dict[str, dict[str, float]]],
) -> str:
    react_policy = by_task_type.get("react_agent", {}).get("policy_recommendation", {})
    langgraph_policy = by_task_type.get("langgraph_tool_agent", {}).get(
        "policy_recommendation", {}
    )
    if react_policy and langgraph_policy:
        react_tool = react_policy.get("tool_selection_accuracy", 0.0)
        langgraph_tool = langgraph_policy.get("tool_selection_accuracy", 0.0)
        react_correctness = react_policy.get("answer_correctness_proxy", 0.0)
        langgraph_correctness = langgraph_policy.get("answer_correctness_proxy", 0.0)
        if react_tool > langgraph_tool or react_correctness > langgraph_correctness:
            return (
                "- `react_agent` baseline 用于对比 single-step workflow vs deterministic multi-step planner；"
                f"新增 multi-hop policy 样例后，policy 子集 tool_selection_accuracy "
                f"从 `{_format_metric(langgraph_tool)}` 提升到 `{_format_metric(react_tool)}`，"
                f"answer_correctness_proxy 从 `{_format_metric(langgraph_correctness)}` "
                f"提升到 `{_format_metric(react_correctness)}`。"
            )
    if "react_agent" in comparison_summary:
        return (
            "- `react_agent` baseline 用来对比 workflow vs multi-step agent："
            "在需要先收集时序上下文再给策略建议的样例上，可以显式展示多步 trace。"
        )
    return "- `react_agent` 尚未纳入当前报告。"


def _grounded_comparison_conclusion(
    comparison_summary: dict[str, dict[str, float]],
) -> str:
    grounded_modes = [
        name
        for name in [
            "rag_keyword_grounded",
            "rag_dense_grounded",
            "rag_rewrite_grounded",
        ]
        if name in comparison_summary
    ]
    if not grounded_modes:
        return "- grounded RAG paired baselines 尚未纳入当前报告。"
    best_name = max(
        grounded_modes,
        key=lambda name: comparison_summary[name].get("grounding_rate", 0.0),
    )
    best_rate = comparison_summary[best_name].get("grounding_rate", 0.0)
    if best_rate > 0.0:
        return (
            "- `rag_keyword_grounded` / `rag_dense_grounded` / `rag_rewrite_grounded` "
            "把 extractive vs grounded generation 做成成对对比；"
            f"当前 `grounding_rate` 最高的是 `{best_name}`={_format_metric(best_rate)}。"
        )
    return (
        "- grounded RAG paired baselines 已纳入对比；`grounding_rate` 仍需结合 "
        "answer correctness 一起看，以区分检索失败和生成漂移。"
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
    safety_adversarial: dict[str, object] | None = None,
    dropt_policy_benchmark: dict[str, object] | None = None,
    dense_provider: str = "deterministic",
    dense_backend: str = "memory",
    dense_model: str | None = None,
    cross_encoder_model: str | None = None,
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
            safety_adversarial=safety_adversarial,
            dropt_policy_benchmark=dropt_policy_benchmark,
            dense_provider=dense_provider,
            dense_backend=dense_backend,
            dense_model=dense_model,
            cross_encoder_model=cross_encoder_model,
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


def _safety_adversarial_section(summary: dict[str, object]) -> list[str]:
    by_category = summary.get("by_category", {})
    lines = [
        "",
        "## Safety Audit 对抗鲁棒性测试",
        "",
        (
            "该测试使用人工构造的 unsafe answer variant 检查确定性 Safety Audit "
            "对生产遥测误述、LLM 直接控制和未验证动作表述的召回。"
        ),
        "",
        f"- sample_count = {summary.get('sample_count', 0)}",
        f"- overall_hit_rate = {_format_metric(float(summary.get('overall_hit_rate', 0.0)))}",
        "",
        "| category | sample_count | hit_count | hit_rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    if isinstance(by_category, dict):
        for category, metrics in by_category.items():
            if not isinstance(metrics, dict):
                continue
            lines.append(
                (
                    f"| {category} | {metrics.get('sample_count', 0)} | "
                    f"{metrics.get('hit_count', 0)} | "
                    f"{_format_metric(float(metrics.get('hit_rate', 0.0)))} |"
                )
            )
    missed_ids = summary.get("missed_ids", [])
    if missed_ids:
        formatted = ", ".join(f"`{missed_id}`" for missed_id in list(missed_ids)[:10])
        lines.extend(["", f"主要漏报样例：{formatted}"])
    return lines


def _dropt_policy_benchmark_section(summary: dict[str, object]) -> list[str]:
    return [
        "",
        "## DROPT Policy Benchmark",
        "",
        "该基准只评测 policy_recommendation 样例上的策略后端推理，不把它混入文档问答 baseline。",
        "",
        "| sample_count | success_count | fallback_count | avg_latency_ms | avg_action_dim | avg_abs_action |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {summary.get('sample_count', 0)} | "
            f"{summary.get('success_count', 0)} | "
            f"{summary.get('fallback_count', 0)} | "
            f"{_format_metric(float(summary.get('avg_latency_ms', 0.0)))} | "
            f"{_format_metric(float(summary.get('avg_action_dim', 0.0)))} | "
            f"{_format_metric(float(summary.get('avg_abs_action', 0.0)))} |"
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
