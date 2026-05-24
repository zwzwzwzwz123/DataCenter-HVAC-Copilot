from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_generate_compound_eval_script_can_run_with_fixture(tmp_path: Path) -> None:
    fixture_path = tmp_path / "llm_candidates.json"
    output_path = tmp_path / "compound_eval.jsonl"
    fixture_path.write_text(
        """
{
  "records": [
    {
      "id": "compound_fixture_001",
      "question": "查询 zone_temperature 趋势，判断是否异常，再给出控制建议",
      "task_type": "policy_recommendation",
      "gold_answer": "先查询时序，再诊断异常，最后调用策略工具。",
      "required_tools": ["query_metric", "detect_anomaly", "rule_based_policy"],
      "required_documents": [],
      "expected_keywords": ["zone_temperature", "异常", "策略"],
      "expected_steps": ["timeseries_query", "anomaly_diagnosis", "policy_recommendation"],
      "expected_output_format": "multi_step_policy_with_tool_evidence"
    }
  ]
}
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/generate_compound_eval.py",
            "--fixture-json",
            str(fixture_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0
    assert output_path.exists()
    assert "Saved 1 compound eval records" in completed.stdout
    assert "expected_steps" in output_path.read_text(encoding="utf-8")
