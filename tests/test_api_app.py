from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.demo_factory import build_demo_orchestrator


def test_health_endpoint_returns_status():
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "datacenter-hvac-copilot"
    assert body["data_source"]["kind"] in {"processed_csv", "bear_sample_csv", "mock"}
    assert body["data_source"]["path"]


def test_ask_endpoint_returns_orchestrator_response():
    client = TestClient(create_app())

    response = client.post(
        "/ask",
        json={
            "question": "episode_001 中 zone_a 在最近 3 小时的温度最大值是多少？",
            "task_type": "timeseries_query",
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["route"] == "timeseries_query"
    assert body["tools"] == ["query_metric"]
    assert body["data_source"]["kind"] in {"processed_csv", "bear_sample_csv", "mock"}
    assert body["answer"]


def test_eval_run_endpoint_returns_metrics():
    client = TestClient(create_app())

    response = client.post("/eval/run", json={"eval_path": "data/eval/hvac_eval.jsonl"})

    body = response.json()
    assert response.status_code == 200
    assert body["metrics"]["tool_selection_accuracy"] == 1.0
    assert len(body["predictions"]) >= 30


def test_demo_orchestrator_loads_all_text_documents_from_documents_dir(tmp_path):
    documents_dir = tmp_path / "data" / "documents"
    documents_dir.mkdir(parents=True)
    (documents_dir / "sample_hvac_guidance.md").write_text(
        "# HVAC Energy Reference\n\nCooling energy depends on setpoints.",
        encoding="utf-8",
    )
    (documents_dir / "thermal_guidance.md").write_text(
        "# Thermal Guidance\n\nHot aisle containment reduces recirculation risk.",
        encoding="utf-8",
    )

    orchestrator = build_demo_orchestrator(project_root=tmp_path)
    answer = orchestrator.rag_pipeline.answer("recirculation containment risk", top_k=1)

    assert answer.citations[0]["source_id"] == "thermal_guidance"
