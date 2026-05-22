from pathlib import Path

from src.retrieval.chunking import chunk_document
from src.retrieval.loader import load_markdown_document
from src.retrieval.query_rewrite import (
    HyDERAGPipeline,
    RewriteRAGPipeline,
    RuleBasedHVACQueryRewriter,
    TemplateHyDEGenerator,
)
from src.retrieval.rag import ExtractiveRAGPipeline
from src.retrieval.retriever import KeywordRetriever


def test_rule_based_hvac_query_rewriter_adds_metric_and_bear_terms() -> None:
    rewriter = RuleBasedHVACQueryRewriter()

    result = rewriter.rewrite(
        "episode_001 中 zone_a 最近 3 小时温度最大值是多少？",
        task_type="timeseries_query",
    )

    assert result.original_query.startswith("episode_001")
    assert "zone_temperature" in result.rewritten_query
    assert "query_metric" in result.rewritten_query
    assert "BEAR HVAC simulation" in result.rewritten_query
    assert "zone_temperature" in result.added_terms


def test_template_hyde_generator_creates_grounded_domain_document() -> None:
    generator = TemplateHyDEGenerator()

    result = generator.generate(
        "如果当前温度超过舒适上限，是否应该调整控制策略？",
        task_type="policy_recommendation",
    )

    assert result.original_query.startswith("如果当前温度")
    assert result.strategy == "template_hyde"
    assert "Hypothetical evidence document" in result.hypothetical_document
    assert "policy_result" in result.hypothetical_document
    assert "LLM 不直接生成或写回控制动作" in result.hypothetical_document


def test_rewrite_rag_pipeline_uses_rewritten_query_for_retrieval() -> None:
    document = load_markdown_document(
        Path("data/documents/timeseries_tool_workflow_note.md"),
        source_id="timeseries_tool_workflow_note",
        title="Timeseries Tool Workflow",
    )
    chunks = chunk_document(document, chunk_size=45, overlap=5)
    raw_pipeline = ExtractiveRAGPipeline(KeywordRetriever(chunks))
    rewrite_pipeline = RewriteRAGPipeline(
        KeywordRetriever(chunks),
        query_rewriter=RuleBasedHVACQueryRewriter(),
        task_type="timeseries_query",
    )

    raw_answer = raw_pipeline.answer("最近三小时温度最大值", top_k=1)
    rewrite_answer = rewrite_pipeline.answer("最近三小时温度最大值", top_k=1)

    assert raw_answer.retrieved_contexts == []
    assert rewrite_answer.retrieved_contexts
    assert rewrite_answer.retrieved_contexts[0]["retrieval_query_strategy"] == "rule_based_hvac_rewrite"


def test_hyde_rag_pipeline_marks_hypothetical_retrieval_query() -> None:
    document = load_markdown_document(
        Path("data/documents/policy_offline_replay_boundary_note.md"),
        source_id="policy_offline_replay_boundary_note",
        title="Policy Offline Replay Boundary",
    )
    chunks = chunk_document(document, chunk_size=45, overlap=5)
    pipeline = HyDERAGPipeline(
        KeywordRetriever(chunks),
        hyde_generator=TemplateHyDEGenerator(),
        task_type="policy_recommendation",
    )

    answer = pipeline.answer("是否应该调整控制策略？", top_k=1)

    assert answer.retrieved_contexts
    assert answer.retrieved_contexts[0]["retrieval_query_strategy"] == "template_hyde"
    assert "policy_result" in answer.retrieved_contexts[0]["retrieval_query"]
