from pathlib import Path

from src.retrieval.chunking import chunk_document
from src.retrieval.loader import load_markdown_document, load_text_documents
from src.retrieval.retriever import HybridRetriever, KeywordRetriever, RerankingRetriever
from src.retrieval.schemas import DocumentMetadata


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
