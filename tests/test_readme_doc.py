from pathlib import Path


def test_readme_documents_chinese_project_overview_and_boundaries() -> None:
    content = Path("README.md").read_text(encoding="utf-8")

    assert "项目亮点" in content
    assert "系统架构" in content
    assert "LLM 后端配置" in content
    assert "OLLAMA_MODEL" in content
    assert "LANGGRAPH_INTENT_PROVIDER" in content
    assert "LLM intent classifier" in content
    assert "scripts/run_intent_eval.py" in content
    assert "ollama" in content
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


def test_demo_walkthrough_does_not_describe_langgraph_as_future_work() -> None:
    content = Path("docs/demo_walkthrough.md").read_text(encoding="utf-8")

    assert "LangGraph 和 FAISS/Qdrant 是后续可替换增强项" not in content
    assert "为什么还没上 LangGraph / FAISS" not in content


def test_project_progress_evaluation_reflects_langgraph_and_ollama_updates() -> None:
    content = Path("project_progress_evaluation.md").read_text(encoding="utf-8")

    assert "IntentClassifier` 不存在" not in content
    assert "无 `OllamaAnswerGenerator`" not in content
    assert "不能说**\"用 LLM 做意图路由\"" not in content
    assert "src/agent/intent_classifier.py" in content
    assert "src/agent/ollama_generator.py" in content
    assert "scripts/run_intent_eval.py" in content


def test_readme_mentions_optional_faiss_dense_retrieval() -> None:
    content = Path("README.md").read_text(encoding="utf-8")

    assert "FAISS dense retrieval" in content
    assert 'pip install -e ".[dev,dense]"' in content
    assert "rag_dense" in content
    assert "Qdrant" in content
    assert "不需要 API" in content


def test_readme_mentions_query_rewrite_and_hyde_baselines() -> None:
    content = Path("README.md").read_text(encoding="utf-8")

    assert "Query Rewrite / HyDE" in content
    assert "rag_rewrite" in content
    assert "rag_hyde" in content
    assert "rag_hyde_rerank" in content
    assert "deterministic query expansion" in content


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
