from __future__ import annotations

import os

from src.core.env import load_env_file


def test_load_env_file_reads_key_value_pairs_without_overriding_existing_values(
    tmp_path,
    monkeypatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# comment",
                "DEEPSEEK_API_KEY=file-key",
                "DEEPSEEK_MODEL='deepseek-test'",
                'DEEPSEEK_BASE_URL="https://example.test"',
                "export DEEPSEEK_TIMEOUT_SECONDS=45",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "existing-key")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_TIMEOUT_SECONDS", raising=False)

    loaded = load_env_file(env_path)

    assert loaded is True
    assert os.environ["DEEPSEEK_API_KEY"] == "existing-key"
    assert os.environ["DEEPSEEK_MODEL"] == "deepseek-test"
    assert os.environ["DEEPSEEK_BASE_URL"] == "https://example.test"
    assert os.environ["DEEPSEEK_TIMEOUT_SECONDS"] == "45"


def test_load_env_file_returns_false_for_missing_file(tmp_path) -> None:
    assert load_env_file(tmp_path / ".env") is False
