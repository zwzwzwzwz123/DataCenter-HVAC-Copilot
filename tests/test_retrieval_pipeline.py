from pathlib import Path

from src.retrieval.chunking import chunk_document
from src.retrieval.loader import load_markdown_document, load_text_documents
from src.retrieval.retriever import (
    HybridRetriever,
    HybridRRFRetriever,
    KeywordRetriever,
    RerankingRetriever,
    reciprocal_rank_fusion,
)
from src.retrieval.cross_encoder import CrossEncoderRerankingRetriever
from src.retrieval.schemas import DocumentMetadata, SourceDocument


def test_load_markdown_document_preserves_metadata(tmp_path: Path):
    doc_path = tmp_path / "ashrae.md"
    doc_path.write_text(
        "# ASHRAE Thermal Guidance\n\n"
        "Cooling systems should keep thermal conditions within documented limits.\n",
        encoding="utf-8",
    )

    document = load_markdown_document(
        doc_path,
        source_id="ashrae_thermal_guidance",
        title="ASHRAE Thermal Guidance",
        published_at="2021",
        category="standard",
    )

    assert document.metadata.source_id == "ashrae_thermal_guidance"
    assert document.metadata.title == "ASHRAE Thermal Guidance"
    assert document.metadata.source_path.endswith("ashrae.md")
    assert "Cooling systems" in document.text


def test_chunk_document_keeps_citation_metadata():
    document = load_markdown_document(
        Path("data/documents/sample_hvac_guidance.md"),
        source_id="sample_hvac_guidance",
        title="Sample HVAC Guidance",
        published_at="2026",
        category="internal_note",
    )

    chunks = chunk_document(document, chunk_size=22, overlap=5)

    assert len(chunks) > 1
    assert chunks[0].chunk_id == "sample_hvac_guidance::chunk_0000"
    assert chunks[0].metadata.source_id == "sample_hvac_guidance"
    assert chunks[0].metadata.title == "Sample HVAC Guidance"
    assert chunks[0].citation["source_id"] == "sample_hvac_guidance"
    assert chunks[0].citation["chunk_id"] == chunks[0].chunk_id


def test_chunk_document_splits_long_chinese_text_without_spaces():
    document = SourceDocument(
        text="# 冷却策略\n\n" + "回风温度机柜温差告警状态需要联合分析" * 8,
        metadata=DocumentMetadata(
            source_id="cn_manual",
            title="中文冷却手册",
            source_path="cn.md",
        ),
    )

    chunks = chunk_document(document, chunk_size=18, overlap=3)

    assert len(chunks) > 1
    assert all(chunk.text for chunk in chunks)
    assert all(chunk.end_word - chunk.start_word <= 18 for chunk in chunks)
    assert chunks[0].citation["section"] == "冷却策略"


def test_chunk_document_keeps_ascii_tool_tokens_separate_from_chinese_text():
    document = SourceDocument(
        text="当问题询问最大值时优先调用 query_metric，并保留 zone_id 和 summary。",
        metadata=DocumentMetadata(
            source_id="mixed_manual",
            title="Mixed Manual",
            source_path="mixed.md",
        ),
    )

    chunks = chunk_document(document, chunk_size=30, overlap=3)

    assert " query_metric " in chunks[0].text
    assert " zone_id " in chunks[0].text
    assert " summary" in chunks[0].text


def test_keyword_retriever_returns_ranked_chunks_with_citations():
    metadata = DocumentMetadata(
        source_id="cooling_doc",
        title="Cooling Doc",
        source_path="memory",
        published_at="2026",
        category="note",
    )
    chunks = [
        metadata.to_chunk(
            chunk_id="cooling_doc::chunk_0000",
            text="Airflow management reduces hot spots in data center cooling.",
            section="Thermal Operations",
            start_word=0,
            end_word=9,
        ),
        metadata.to_chunk(
            chunk_id="cooling_doc::chunk_0001",
            text="Battery maintenance procedures are unrelated to HVAC airflow.",
            section="Maintenance",
            start_word=10,
            end_word=17,
        ),
    ]

    retriever = KeywordRetriever(chunks)
    results = retriever.search("cooling airflow hot spots", top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "cooling_doc::chunk_0000"
    assert results[0]["score"] > 0
    assert results[0]["citation"]["title"] == "Cooling Doc"


def test_hybrid_retriever_uses_bm25_length_normalization_and_labels_mode():
    metadata = DocumentMetadata(
        source_id="cooling_doc",
        title="Cooling Doc",
        source_path="memory",
        published_at="2026",
        category="note",
    )
    chunks = [
        metadata.to_chunk(
            chunk_id="cooling_doc::chunk_long",
            text=(
                "cooling airflow "
                "battery maintenance lighting envelope occupancy unrelated unrelated unrelated unrelated"
            ),
            section="Mixed",
            start_word=0,
            end_word=12,
        ),
        metadata.to_chunk(
            chunk_id="cooling_doc::chunk_short",
            text="cooling airflow hot spots",
            section="Thermal",
            start_word=13,
            end_word=16,
        ),
    ]

    retriever = HybridRetriever(chunks)
    results = retriever.search("cooling airflow hot spots", top_k=2)

    assert [result["chunk_id"] for result in results] == [
        "cooling_doc::chunk_short",
        "cooling_doc::chunk_long",
    ]
    assert results[0]["retrieval_mode"] == "hybrid_bm25"
    assert results[0]["score"] > results[1]["score"]


def test_reciprocal_rank_fusion_accumulates_rank_scores_and_deduplicates():
    keyword_results = [
        {"chunk_id": "doc_a::chunk_0000", "score": 100.0, "text": "A", "citation": {}},
        {"chunk_id": "doc_b::chunk_0000", "score": 90.0, "text": "B", "citation": {}},
    ]
    dense_results = [
        {"chunk_id": "doc_b::chunk_0000", "score": 0.1, "text": "B dense", "citation": {}},
        {"chunk_id": "doc_c::chunk_0000", "score": 0.9, "text": "C", "citation": {}},
    ]

    fused = reciprocal_rank_fusion([keyword_results, dense_results], k=60, top_k=3)

    assert [result["chunk_id"] for result in fused] == [
        "doc_b::chunk_0000",
        "doc_a::chunk_0000",
        "doc_c::chunk_0000",
    ]
    assert fused[0]["score"] == (1 / 62) + (1 / 61)
    assert fused[0]["rrf_score"] == fused[0]["score"]
    assert fused[0]["source_retrieval_modes"] == []


class FixedRetriever:
    def __init__(self, results: list[dict]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


def test_hybrid_rrf_retriever_uses_top_n_from_bm25_and_dense_then_returns_top_k():
    bm25 = FixedRetriever(
        [
            {
                "chunk_id": "doc_a::chunk_0000",
                "score": 3.0,
                "text": "bm25 a",
                "citation": {},
                "retrieval_mode": "hybrid_bm25",
            },
            {
                "chunk_id": "doc_b::chunk_0000",
                "score": 2.0,
                "text": "bm25 b",
                "citation": {},
                "retrieval_mode": "hybrid_bm25",
            },
        ]
    )
    dense = FixedRetriever(
        [
            {
                "chunk_id": "doc_b::chunk_0000",
                "score": 0.2,
                "text": "dense b",
                "citation": {},
                "retrieval_mode": "dense_hash",
            },
            {
                "chunk_id": "doc_c::chunk_0000",
                "score": 0.9,
                "text": "dense c",
                "citation": {},
                "retrieval_mode": "dense_hash",
            },
        ]
    )

    retriever = HybridRRFRetriever(bm25, dense, candidate_k=2, rrf_k=60)
    results = retriever.search("cooling airflow", top_k=2)

    assert bm25.calls == [("cooling airflow", 2)]
    assert dense.calls == [("cooling airflow", 2)]
    assert [result["chunk_id"] for result in results] == [
        "doc_b::chunk_0000",
        "doc_a::chunk_0000",
    ]
    assert results[0]["retrieval_mode"] == "hybrid_rrf"
    assert results[0]["source_retrieval_modes"] == ["hybrid_bm25", "dense_hash"]


def test_reranking_retriever_promotes_exact_phrase_and_labels_mode():
    metadata = DocumentMetadata(
        source_id="rack_doc",
        title="Rack Doc",
        source_path="memory",
        published_at="2026",
        category="note",
    )
    chunks = [
        metadata.to_chunk(
            chunk_id="rack_doc::noise",
            text="rack delta-t alarm rack delta-t alarm rack delta-t alarm unrelated lighting",
            section="Noise",
            start_word=0,
            end_word=8,
        ),
        metadata.to_chunk(
            chunk_id="rack_doc::target",
            text="rack delta-t return differential alarm evidence uses supply return temperature delta",
            section="Target",
            start_word=9,
            end_word=18,
        ),
    ]

    retriever = RerankingRetriever(KeywordRetriever(chunks), candidate_k=2)
    results = retriever.search("rack delta-t return differential alarm evidence", top_k=1)

    assert results[0]["chunk_id"] == "rack_doc::target"
    assert results[0]["retrieval_mode"] == "rerank_keyword_overlap"
    assert results[0]["base_retrieval_mode"] == "keyword"
    assert results[0]["rerank_score"] > 0


def test_reranking_retriever_uses_citation_metadata_for_tie_breaking():
    target_metadata = DocumentMetadata(
        source_id="supply_air_reset_risk_note",
        title="Supply Air Reset Risk Note",
        source_path="memory",
        published_at="2026",
        category="internal_note",
    )
    noise_metadata = DocumentMetadata(
        source_id="cooling_airflow_noise_long_note",
        title="Cooling Airflow Noise Long Note",
        source_path="memory",
        published_at="2026",
        category="internal_note",
    )
    chunks = [
        noise_metadata.to_chunk(
            chunk_id="cooling_airflow_noise_long_note::chunk_0000",
            text=(
                "supply air reset risk comfort violation policy evidence "
                "cooling airflow repeated repeated repeated repeated repeated"
            ),
            section="General Cooling Noise",
            start_word=0,
            end_word=13,
        ),
        target_metadata.to_chunk(
            chunk_id="supply_air_reset_risk_note::chunk_0000",
            text="supply air reset risk comfort violation policy evidence",
            section="Supply Air Reset Risk",
            start_word=0,
            end_word=7,
        ),
    ]

    retriever = RerankingRetriever(
        KeywordRetriever(chunks),
        candidate_k=2,
        base_score_weight=0.0,
        metadata_weight=2.0,
    )
    results = retriever.search("supply air reset risk note", top_k=1)

    assert results[0]["chunk_id"] == "supply_air_reset_risk_note::chunk_0000"
    assert results[0]["metadata_score"] > 0


class FakeCrossEncoderScorer:
    model_name = "fake-cross-encoder"

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, texts))
        return [self.scores[text] for text in texts]


def test_cross_encoder_reranker_reranks_candidate_pairs_and_preserves_base_metadata():
    base = FixedRetriever(
        [
            {
                "chunk_id": "doc_noise::chunk_0000",
                "score": 9.0,
                "text": "generic cooling airflow paragraph without return differential evidence",
                "citation": {"source_id": "doc_noise", "title": "Noise"},
                "retrieval_mode": "hybrid_rrf",
            },
            {
                "chunk_id": "doc_target::chunk_0000",
                "score": 4.0,
                "text": "rack delta-t return differential alarm evidence from supply return temperatures",
                "citation": {"source_id": "doc_target", "title": "Target"},
                "retrieval_mode": "hybrid_rrf",
            },
            {
                "chunk_id": "doc_other::chunk_0000",
                "score": 3.0,
                "text": "policy boundary note unrelated to rack delta-t diagnosis",
                "citation": {"source_id": "doc_other", "title": "Other"},
                "retrieval_mode": "hybrid_rrf",
            },
        ]
    )
    scorer = FakeCrossEncoderScorer(
        {
            "generic cooling airflow paragraph without return differential evidence": 0.12,
            "rack delta-t return differential alarm evidence from supply return temperatures": 0.91,
            "policy boundary note unrelated to rack delta-t diagnosis": 0.25,
        }
    )

    retriever = CrossEncoderRerankingRetriever(
        base,
        scorer=scorer,
        candidate_k=3,
    )
    results = retriever.search("rack delta-t return differential evidence", top_k=2)

    assert base.calls == [("rack delta-t return differential evidence", 3)]
    assert scorer.calls == [
        (
            "rack delta-t return differential evidence",
            [
                "generic cooling airflow paragraph without return differential evidence",
                "rack delta-t return differential alarm evidence from supply return temperatures",
                "policy boundary note unrelated to rack delta-t diagnosis",
            ],
        )
    ]
    assert [result["chunk_id"] for result in results] == [
        "doc_target::chunk_0000",
        "doc_other::chunk_0000",
    ]
    assert results[0]["retrieval_mode"] == "cross_encoder_rerank"
    assert results[0]["base_retrieval_mode"] == "hybrid_rrf"
    assert results[0]["base_score"] == 4.0
    assert results[0]["cross_encoder_score"] == 0.91
    assert results[0]["cross_encoder_model"] == "fake-cross-encoder"
    assert results[0]["candidate_rank"] == 2


def test_load_text_documents_loads_supported_files(tmp_path: Path):
    (tmp_path / "doc_a.md").write_text("# Cooling\n\nCooling saves energy.", encoding="utf-8")
    (tmp_path / "doc_b.txt").write_text("HVAC alarms need evidence.", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("not,a,document", encoding="utf-8")

    documents = load_text_documents(tmp_path)

    assert [doc.metadata.source_id for doc in documents] == ["doc_a", "doc_b"]


def test_demo_documents_include_similar_theme_pressure_notes():
    documents = load_text_documents(Path("data/documents"))
    source_ids = {document.metadata.source_id for document in documents}

    assert {
        "airflow_containment_operations_note",
        "setpoint_tradeoff_operations_note",
        "supply_air_reset_risk_note",
        "sensor_drift_alarm_boundary_note",
        "return_air_delta_t_operations_note",
        "cooling_airflow_noise_long_note",
        "economizer_free_cooling_note",
        "redundancy_maintenance_alarm_note",
        "liquid_air_hybrid_cooling_note",
        "sensor_missing_data_quality_note",
        "policy_offline_replay_boundary_note",
        "timeseries_tool_workflow_note",
    }.issubset(source_ids)
