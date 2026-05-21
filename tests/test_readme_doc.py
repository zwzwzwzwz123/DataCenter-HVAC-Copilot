from pathlib import Path


def test_readme_documents_chinese_project_overview_and_boundaries() -> None:
    content = Path("README.md").read_text(encoding="utf-8")

    assert "项目亮点" in content
    assert "系统架构" in content
    assert "DeepSeek 配置" in content
    assert "不是普通 ChatPDF" in content
    assert "不能伪装成真实数据中心生产遥测" in content
    assert "LLM / Agent 只负责任务路由、证据整合和解释生成" in content
    assert "docs/demo_walkthrough.md" in content
    assert "--enable-llm-judge" in content


def test_readme_mentions_human_evaluation_calibration() -> None:
    content = Path("README.md").read_text(encoding="utf-8")

    assert "人工评测校准" in content
    assert "docs/human_evaluation_guide.md" in content
    assert "human_review_annotations.jsonl" in content


def test_readme_mentions_optional_faiss_dense_retrieval() -> None:
    content = Path("README.md").read_text(encoding="utf-8")

    assert "FAISS dense retrieval" in content
    assert 'pip install -e ".[dev,dense]"' in content
    assert "rag_dense" in content
    assert "Qdrant" in content
    assert "不需要 API" in content
