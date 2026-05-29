from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.knowledge.service import KnowledgeBaseService


DEFAULT_OUTPUT_PATH = Path("data/eval/persistent_knowledge_ranking_eval.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a source-aligned ranking eval dataset from the persistent KB."
    )
    parser.add_argument(
        "--knowledge-dir",
        default="data/knowledge",
        help="Persistent knowledge directory containing knowledge.db and FAISS artifacts.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="JSONL output path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Maximum number of documents to convert into eval records.",
    )
    args = parser.parse_args()

    records = build_records(
        knowledge_dir=Path(args.knowledge_dir),
        limit=args.limit,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(records)} persistent ranking eval records to {output_path}")


def build_records(*, knowledge_dir: Path, limit: int) -> list[dict]:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    service = KnowledgeBaseService(knowledge_dir=knowledge_dir)
    documents = [
        document
        for document in service.list_documents()
        if document.get("status") == "indexed" and document.get("chunk_count", 0) > 0
    ]
    documents.sort(key=lambda document: document["filename"].lower())
    records = []
    for index, document in enumerate(documents[:limit], start=1):
        metadata = document.get("metadata", {})
        filename = document["filename"]
        topic = metadata.get("topic") or _topic_from_filename(filename)
        authority = metadata.get("authority") or _authority_from_filename(filename)
        required_documents = [document["document_id"]]
        gold_aliases = [
            f"document_id:{document['document_id']}",
            f"file_hash:{document['file_hash']}",
            f"filename:{filename}",
        ]
        if metadata.get("source_url"):
            gold_aliases.append(f"source_url:{metadata['source_url']}")
        records.append(
            {
                "id": f"persistent_rank_{index:03d}",
                "question": (
                    f"Retrieve the authoritative document `{filename}` and summarize its "
                    f"relevance to data-center HVAC/copilot decisions. Focus on {topic}."
                ),
                "task_type": "document_qa",
                "gold_answer": (
                    f"Expected source: {filename}. Authority: {authority}. "
                    f"Aliases: {', '.join(gold_aliases)}."
                ),
                "required_tools": [],
                "required_documents": required_documents,
                "expected_keywords": _expected_keywords(filename, topic, authority),
                "expected_output_format": "answer_with_citations",
            }
        )
    return records


def _expected_keywords(filename: str, topic: str, authority: str) -> list[str]:
    keywords = []
    for value in (filename, topic, authority):
        for token in value.replace("_", " ").replace("-", " ").replace(".", " ").split():
            normalized = token.strip().lower()
            if len(normalized) >= 4 and normalized not in keywords:
                keywords.append(normalized)
            if len(keywords) >= 6:
                return keywords
    return keywords or ["data", "center"]


def _topic_from_filename(filename: str) -> str:
    stem = Path(filename).stem.replace("_", " ").replace("-", " ")
    return " ".join(stem.split())


def _authority_from_filename(filename: str) -> str:
    lowered = filename.lower()
    if lowered.startswith("ashrae"):
        return "ASHRAE"
    if lowered.startswith("doe"):
        return "U.S. Department of Energy"
    if lowered.startswith("lbnl"):
        return "LBNL/DOE Data Centers Center of Expertise"
    if lowered.startswith("ocp"):
        return "Open Compute Project"
    if lowered.startswith("uptime"):
        return "Uptime Institute"
    if lowered.startswith("google"):
        return "Google"
    return "Curated data-center HVAC knowledge base"


if __name__ == "__main__":
    main()
