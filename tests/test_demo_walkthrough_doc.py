from pathlib import Path


def test_demo_walkthrough_documents_core_cases_and_boundaries() -> None:
    content = Path("docs/demo_walkthrough.md").read_text(encoding="utf-8")

    assert "BEAR 数据边界" in content
    assert "温度时序查询" in content
    assert "策略建议边界" in content
    assert "不是普通 ChatPDF" in content
    assert "不是真实数据中心生产遥测" in content
    assert "LLM 不直接生成或写回控制动作" in content
    assert "scripts/run_eval.py" in content
