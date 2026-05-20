from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.agent.orchestrator import BaselineOrchestrator
from src.ingestion.bear_sample_loader import load_bear_sample_timeseries
from src.ingestion.processed_loader import load_processed_bear_trajectory
from src.retrieval.chunking import chunk_document
from src.retrieval.loader import load_markdown_document, load_text_documents
from src.retrieval.rag import ExtractiveRAGPipeline
from src.retrieval.retriever import HybridRetriever


def build_demo_orchestrator(project_root: Path | None = None) -> BaselineOrchestrator:
    """Build a demo orchestrator from sample docs and mock trajectory data."""

    project_root = project_root or Path(__file__).resolve().parents[2]
    chunks = []
    for document in _load_demo_documents(project_root):
        chunks.extend(chunk_document(document, chunk_size=45, overlap=5))
    rag = ExtractiveRAGPipeline(HybridRetriever(chunks))
    trajectory = _load_demo_trajectory(project_root)
    return BaselineOrchestrator(
        rag_pipeline=rag,
        trajectory=trajectory,
        data_source=trajectory.attrs["data_source"],
    )


def _load_demo_documents(project_root: Path) -> list:
    documents_dir = project_root / "data" / "documents"
    sample_path = documents_dir / "sample_hvac_guidance.md"
    documents = []
    if sample_path.exists():
        documents.append(
            load_markdown_document(
                sample_path,
                source_id="hvac_energy_reference",
                title="HVAC Energy Reference",
                published_at="2026",
                category="internal_note",
            )
        )
    for document in load_text_documents(documents_dir):
        if document.metadata.source_path == str(sample_path):
            continue
        documents.append(document)
    return documents


def _load_demo_trajectory(project_root: Path | None = None) -> pd.DataFrame:
    project_root = project_root or Path(__file__).resolve().parents[2]
    processed_path = project_root / "data" / "bear_processed" / "bear_rollout.csv"
    if processed_path.exists():
        frame = load_processed_bear_trajectory(processed_path)
        frame.attrs["data_source"] = {
            "kind": "processed_csv",
            "path": str(processed_path),
        }
        return frame
    bear_sample_path = project_root / "BEAR" / "BEAR" / "Data" / "Exercise2A-mytest.csv"
    if bear_sample_path.exists():
        frame = load_bear_sample_timeseries(bear_sample_path)
        frame.attrs["data_source"] = {
            "kind": "bear_sample_csv",
            "path": str(bear_sample_path),
        }
        return frame
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            "scenario_id": ["episode_001"] * 4,
            "zone_id": ["zone_a"] * 4,
            "zone_temperature": [23.0, 24.0, 30.0, 25.0],
            "outdoor_temp": [31.0, 31.5, 32.0, 32.0],
            "solar_irradiance": [0.2, 0.3, 0.4, 0.5],
            "ground_temp": [18.0, 18.0, 18.0, 18.0],
            "internal_load": [0.1, 0.1, 0.12, 0.13],
            "control_action": [-500.0, -500.0, -600.0, -600.0],
            "reward": [-0.5, -0.6, -0.8, -0.7],
            "comfort_violation": [False, False, True, False],
        }
    )
    frame.attrs["source"] = "mock_trajectory"
    frame.attrs["data_source"] = {
        "kind": "mock",
        "path": "built-in demo trajectory",
    }
    return frame
