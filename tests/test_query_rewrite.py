from pathlib import Path

from src.retrieval.chunking import chunk_document
from src.retrieval.loader import load_markdown_document
from src.retrieval.query_rewrite import (
    LLMMultiQueryRAGPipeline,
    LLMMultiQueryRewriter,
    HyDERAGPipeline,
    RewriteRAGPipeline,
    RuleBasedHVACQueryRewriter,
    TemplateHyDEGenerator,
)
from src.retrieval.rag import ExtractiveRAGPipeline
from src.retrieval.retriever import KeywordRetriever


class FakeRewriteTransport:
    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        self.content = content or '["zone temperature max", "query_metric zone_temperature"]'
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, url: str, headers: dict[str, str], body: bytes, timeout: float) -> dict:
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "body": body,
                "timeout": timeout,
            }
        )
        if self.error:
            raise self.error
        return {"choices": [{"message": {"content": self.content}}]}


class StubRetriever:
    def __init__(self) -> None:
        self.chunks = []
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        self.queries.append(query)
        if "alpha" in query:
            return [
                {
                    "chunk_id": "shared",
                    "score": 9.0,
                    "text": "shared from alpha",
                    "citation": {"source_id": "doc_shared"},
                    "retrieval_mode": "stub",
                },
                {
                    "chunk_id": "alpha_only",
                    "score": 8.0,
                    "text": "alpha only",
                    "citation": {"source_id": "doc_alpha"},
                    "retrieval_mode": "stub",
                },
            ][:top_k]
        if "beta" in query:
            return [
                {
                    "chunk_id": "shared",
                    "score": 7.0,
                    "text": "shared from beta",
                    "citation": {"source_id": "doc_shared"},
                    "retrieval_mode": "stub",
                },
                {
                    "chunk_id": "beta_only",
                    "score": 6.0,
                    "text": "beta only",
                    "citation": {"source_id": "doc_beta"},
                    "retrieval_mode": "stub",
                },
            ][:top_k]
        return []


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


def test_llm_multi_query_rewriter_parses_json_array_with_at_most_five_variants() -> None:
    transport = FakeRewriteTransport(
        content='["alpha query", "beta query", "alpha query", "  "]'
    )
    rewriter = LLMMultiQueryRewriter(
        provider="deepseek",
        api_key="test-key",
        base_url="https://example.deepseek.test",
        model="rewrite-test",
        transport=transport,
    )

    result = rewriter.rewrite_queries("original question", task_type="document_qa")

    assert result.queries == ["alpha query", "beta query"]
    assert result.strategy == "llm_multi_query_rewrite"
    assert result.fallback_used is False
    assert transport.calls[0]["url"] == "https://example.deepseek.test/chat/completions"


def test_llm_multi_query_rewriter_falls_back_to_rule_rewrite_on_invalid_payload() -> None:
    rewriter = LLMMultiQueryRewriter(
        provider="deepseek",
        api_key="test-key",
        base_url="https://example.deepseek.test",
        model="rewrite-test",
        transport=FakeRewriteTransport(content='["one", "two", "three", "four", "five", "six"]'),
    )

    result = rewriter.rewrite_queries("最近三小时温度最大值", task_type="timeseries_query")

    assert result.fallback_used is True
    assert result.strategy == "rule_based_hvac_rewrite"
    assert len(result.queries) == 1
    assert "zone_temperature" in result.queries[0]


def test_llm_multi_query_rewriter_falls_back_when_array_items_are_not_strings() -> None:
    rewriter = LLMMultiQueryRewriter(
        provider="deepseek",
        api_key="test-key",
        base_url="https://example.deepseek.test",
        model="rewrite-test",
        transport=FakeRewriteTransport(content='[{"query": "bad wrapper"}]'),
    )

    result = rewriter.rewrite_queries("最近三小时温度最大值", task_type="timeseries_query")

    assert result.fallback_used is True
    assert result.strategy == "rule_based_hvac_rewrite"
    assert "zone_temperature" in result.queries[0]


def test_llm_multi_query_rag_pipeline_fuses_variant_results_with_rrf() -> None:
    retriever = StubRetriever()
    pipeline = LLMMultiQueryRAGPipeline(
        retriever,
        query_rewriter=LLMMultiQueryRewriter(
            provider="deepseek",
            api_key="test-key",
            base_url="https://example.deepseek.test",
            model="rewrite-test",
            transport=FakeRewriteTransport(content='["alpha query", "beta query"]'),
        ),
        candidate_k=2,
    )

    answer = pipeline.answer("original question", top_k=2)

    assert retriever.queries == ["alpha query", "beta query"]
    assert [context["chunk_id"] for context in answer.retrieved_contexts] == [
        "shared",
        "alpha_only",
    ]
    assert answer.retrieved_contexts[0]["retrieval_query_strategy"] == "llm_multi_query_rewrite"
    assert answer.retrieved_contexts[0]["retrieval_queries"] == ["alpha query", "beta query"]


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
