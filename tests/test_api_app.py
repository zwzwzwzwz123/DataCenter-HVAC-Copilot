from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from src.agent.answer_generator import AnswerGeneratorInput, GeneratedAnswer
from src.api.app import create_app
from src.api.demo_factory import build_demo_orchestrator


class FactorySpyGenerator:
    def generate(self, payload: AnswerGeneratorInput) -> GeneratedAnswer:
        return GeneratedAnswer(answer="factory-spy", generator="factory_spy")


def test_health_endpoint_returns_status():
    client = TestClient(create_app(use_env_answer_generator=False))

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "datacenter-hvac-copilot"
    assert body["data_source"]["kind"] in {"processed_csv", "bear_sample_csv", "mock"}
    assert body["data_source"]["path"]


def test_ask_endpoint_returns_orchestrator_response():
    client = TestClient(create_app(use_env_answer_generator=False))

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
    assert body["answer_generator"]


def test_eval_run_endpoint_returns_metrics():
    client = TestClient(create_app(use_env_answer_generator=False))

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


def test_demo_orchestrator_uses_deterministic_generator_without_deepseek_key(
    tmp_path,
    monkeypatch,
):
    documents_dir = tmp_path / "data" / "documents"
    documents_dir.mkdir(parents=True)
    (documents_dir / "sample_hvac_guidance.md").write_text(
        "# HVAC Energy Reference\n\nCooling energy depends on setpoints.",
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    orchestrator = build_demo_orchestrator(project_root=tmp_path)
    result = orchestrator.run("Cooling energy depends on what?", task_type="document_qa")

    assert result["answer_generator"] == "deterministic_grounded"


def test_demo_orchestrator_uses_env_answer_generator_factory(tmp_path, monkeypatch):
    documents_dir = tmp_path / "data" / "documents"
    documents_dir.mkdir(parents=True)
    (documents_dir / "sample_hvac_guidance.md").write_text(
        "# HVAC Energy Reference\n\nCooling energy depends on setpoints.",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.api.demo_factory.build_answer_generator_from_env",
        lambda **_: FactorySpyGenerator(),
    )

    orchestrator = build_demo_orchestrator(project_root=tmp_path)
    result = orchestrator.run("Cooling energy depends on what?", task_type="document_qa")

    assert result["answer"] == "factory-spy"
    assert result["answer_generator"] == "factory_spy"


def test_demo_orchestrator_can_disable_env_answer_generator(tmp_path, monkeypatch):
    documents_dir = tmp_path / "data" / "documents"
    documents_dir.mkdir(parents=True)
    (documents_dir / "sample_hvac_guidance.md").write_text(
        "# HVAC Energy Reference\n\nCooling energy depends on setpoints.",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.api.demo_factory.build_answer_generator_from_env",
        lambda: FactorySpyGenerator(),
    )

    orchestrator = build_demo_orchestrator(
        project_root=tmp_path,
        use_env_answer_generator=False,
    )
    result = orchestrator.run("Cooling energy depends on what?", task_type="document_qa")

    assert result["answer_generator"] == "deterministic_grounded"


def test_demo_orchestrator_uses_offline_replay_policy_when_file_exists(tmp_path):
    documents_dir = tmp_path / "data" / "documents"
    documents_dir.mkdir(parents=True)
    (documents_dir / "sample_hvac_guidance.md").write_text(
        "# HVAC Energy Reference\n\nCooling energy depends on setpoints.",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "data" / "eval"
    replay_dir.mkdir(parents=True)
    (replay_dir / "offline_policy_replay.json").write_text(
        """
[
  {
    "policy_name": "guided_diffno_offline_replay",
    "input_state_id": "episode_001_latest",
    "recommended_action": [-0.2, -0.1],
    "estimated_energy": 901.3,
    "estimated_comfort_violations": 0.1,
    "mean_action_change": 0.15,
    "baseline": "rule_based",
    "notes": "Values come from offline replay."
  }
]
""",
        encoding="utf-8",
    )

    orchestrator = build_demo_orchestrator(
        project_root=tmp_path,
        use_env_answer_generator=False,
    )
    result = orchestrator.run("请给出策略建议", task_type="policy_recommendation")

    assert result["tools"] == ["guided_diffno_offline_replay"]
    assert result["policy_result"]["estimated_energy"] == 901.3


def test_demo_orchestrator_uses_dropt_checkpoint_when_present(tmp_path):
    documents_dir = tmp_path / "data" / "documents"
    documents_dir.mkdir(parents=True)
    (documents_dir / "sample_hvac_guidance.md").write_text(
        "# HVAC Energy Reference\n\nCooling energy depends on setpoints.",
        encoding="utf-8",
    )
    bear_dir = tmp_path / "data" / "bear_processed"
    bear_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z"] * 6,
            "scenario_id": ["episode_001"] * 6,
            "zone_id": [
                "zone_0",
                "zone_1",
                "zone_2",
                "zone_3",
                "zone_4",
                "zone_5",
            ],
            "zone_temperature": [23.0, 23.5, 24.0, 24.5, 25.0, 25.5],
            "outdoor_temp": [31.0] * 6,
            "solar_irradiance": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "ground_temp": [18.0] * 6,
            "internal_load": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "control_action": [-500.0, -500.0, -500.0, -500.0, -500.0, -500.0],
            "reward": [-0.5] * 6,
            "comfort_violation": [False] * 6,
        }
    ).to_csv(bear_dir / "bear_rollout.csv", index=False)
    checkpoint = Path("policy_best_fno_guided.pth")
    assert checkpoint.exists()
    (tmp_path / "policy_best_fno_guided.pth").write_bytes(checkpoint.read_bytes())

    orchestrator = build_demo_orchestrator(
        project_root=tmp_path,
        use_env_answer_generator=False,
        use_dropt_policy=True,
    )
    result = orchestrator.run("请给出策略建议", task_type="policy_recommendation")

    assert result["tools"] == ["dropt_guided_diffno_checkpoint"]
    assert result["policy_result"]["policy_name"] == "dropt_guided_diffno_checkpoint"
