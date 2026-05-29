import json
import subprocess
import sys
from pathlib import Path

from src.evaluation.dataset import load_eval_dataset
from src.knowledge.service import KnowledgeBaseService


def test_generate_persistent_ranking_eval_from_current_knowledge_base(tmp_path: Path):
    output_path = tmp_path / "persistent_ranking_eval.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/generate_persistent_ranking_eval.py",
            "--knowledge-dir",
            "data/knowledge",
            "--output",
            str(output_path),
            "--limit",
            "6",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    records = load_eval_dataset(output_path)
    assert len(records) == 6
    documents = {
        document["document_id"]: document
        for document in KnowledgeBaseService(knowledge_dir="data/knowledge").list_documents()
    }
    for record in records:
        assert record.task_type == "document_qa"
        assert record.required_documents
        required_id = record.required_documents[0]
        assert required_id in documents
        document = documents[required_id]
        assert document["filename"] in record.question
        assert "file_hash:" in record.gold_answer
        assert "filename:" in record.gold_answer

    first_payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert first_payload["expected_output_format"] == "answer_with_citations"
