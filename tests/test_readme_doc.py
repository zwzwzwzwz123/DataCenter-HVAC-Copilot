from pathlib import Path


def test_readme_documents_chinese_project_overview_and_boundaries() -> None:
    content = Path("README.md").read_text(encoding="utf-8")

    assert "项目亮点" in content
    assert "系统架构" in content
    assert "LLM 后端配置" in content
    assert "OLLAMA_MODEL" in content
    assert "LANGGRAPH_PLANNER_PROVIDER" in content
    assert "LLM route planner" in content
    assert "src/agent/planner.py" in content
    assert "scripts/run_intent_eval.py" in content
    assert "ollama" in content
    assert "不是普通 ChatPDF" in content
    assert "不能伪装成真实数据中心生产遥测" in content
    assert "LLM / Agent 只负责任务路由、证据整合和解释生成" in content
    assert "docs/demo_walkthrough.md" in content
    assert "--enable-llm-judge" in content


def test_readme_langgraph_intro_is_chinese() -> None:
    content = Path("README.md").read_text(encoding="utf-8")

    assert "## LangGraph Trace Demo" not in content
    assert "## LangGraph 工作流追踪演示" in content
    assert "Copilot tab now lets you switch" not in content
    assert "LangGraph 现在使用" in content


def test_readme_describes_langgraph_planner_without_stale_intent_node_claims() -> None:
    content = Path("README.md").read_text(encoding="utf-8")

    assert "LANGGRAPH_PLANNER_PROVIDER" in content
    assert "LLM route planner" in content
    assert "execute_plan_steps" in content
    assert "answer_generator" in content
    assert "collect_*_evidence" in content
    assert "tool / metric_name / zone_id / time_window" in content
    assert "last_N_hours" in content
    assert "非法 `time_window`" in content
    assert "time_window_applied" in content
    assert "没有 `expected_steps`" in content
    assert "compound_task" in content
    assert "planned_step_accuracy" in content
    assert "scripts/generate_compound_eval.py" in content
    assert "compound_task_llm_planner_eval.json" in content
    assert "compound_task_llm_planner_eval.md" in content
    assert "planned_step_accuracy` = 0.780" in content
    assert "LANGGRAPH_INTENT_PROVIDER` 只影响 `workflow_engine=langgraph`" not in content
    assert "LangGraph StateGraph workflow + DeepSeek/Ollama optional LLM intent classifier" not in content
    assert "可选 LLM intent classification" not in content
    assert "LANGGRAPH_INTENT_PROVIDER" not in content
    assert "intent classification 示例" not in content


def test_env_example_uses_langgraph_planner_provider_not_stale_intent_provider() -> None:
    content = Path(".env.example").read_text(encoding="utf-8")

    assert "LANGGRAPH_PLANNER_PROVIDER" in content
    assert "LANGGRAPH_PLANNER_MODEL" in content
    assert "LANGGRAPH_PLANNER_TIMEOUT_SECONDS" in content
    assert "LANGGRAPH_INTENT_PROVIDER" not in content
    assert "intent classification" not in content.lower()


def test_readme_system_architecture_diagram_shows_planner_path() -> None:
    content = Path("README.md").read_text(encoding="utf-8")
    architecture_section = content.split("## 系统架构", maxsplit=1)[1].split(
        "## 数据边界",
        maxsplit=1,
    )[0]

    assert "LangGraph Route Planner" in architecture_section
    assert "collect_*_evidence" in architecture_section
    assert "tool / metric_name / zone_id / time_window" in architecture_section
    assert "Merged Evidence" in architecture_section
    assert "Merged Evidence" in architecture_section.split("answer_generator")[0]
    assert "answer_generator" in architecture_section.split("Answer Safety Audit")[0]
    assert "G --> H[answer_generator]" in architecture_section
    assert "H --> I[answer_audit]" in architecture_section
    assert "Deterministic Router\n  |" not in architecture_section


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
