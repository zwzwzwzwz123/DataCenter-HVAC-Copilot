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

    assert "评测口径" in content
    assert "人工评测是可选增强项" in content
    assert "Human Calibration" in content
    assert "docs/human_evaluation_guide.md" in content
    assert "human_review_annotations.jsonl" in content


def test_readme_mentions_optional_faiss_dense_retrieval() -> None:
    content = Path("README.md").read_text(encoding="utf-8")

    assert "FAISS dense retrieval" in content
    assert 'pip install -e ".[dev,dense]"' in content
    assert "rag_dense" in content
    assert "Qdrant" in content
    assert "不需要 API" in content


def test_readme_documents_docker_compose_demo_startup() -> None:
    content = Path("README.md").read_text(encoding="utf-8")

    assert "Docker 一键启动" in content
    assert "docker compose up --build" in content
    assert "HVAC_COPILOT_API_BASE_URL" in content


def test_docker_compose_files_are_present() -> None:
    dockerfile = Path("Dockerfile")
    compose = Path("docker-compose.yml")

    assert dockerfile.exists()
    assert compose.exists()
    assert "uvicorn" in dockerfile.read_text(encoding="utf-8")
    compose_content = compose.read_text(encoding="utf-8")
    assert "api:" in compose_content
    assert "streamlit:" in compose_content
