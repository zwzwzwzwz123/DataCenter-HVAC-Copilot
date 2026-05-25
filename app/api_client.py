from __future__ import annotations

from typing import Any

import httpx


class ApiClientError(RuntimeError):
    pass


def ask_api(
    api_base_url: str,
    question: str,
    task_type: str | None = None,
    workflow_engine: str = "langgraph",
    session_id: str | None = None,
    memory_enabled: bool | None = None,
    http_client: Any = httpx,
    timeout: float = 30.0,
) -> dict[str, Any]:
    payload = {
        "question": question,
        "task_type": task_type,
        "workflow_engine": workflow_engine,
    }
    if session_id is not None:
        payload["session_id"] = session_id
    if memory_enabled is not None:
        payload["memory_enabled"] = memory_enabled
    response = http_client.post(
        _join_url(api_base_url, "/ask"),
        json=payload,
        timeout=timeout,
    )
    if response.status_code != 200:
        raise ApiClientError(f"API request failed with status {response.status_code}: {response.text}")
    return response.json()


def run_eval_api(
    api_base_url: str,
    eval_path: str = "data/eval/hvac_eval.jsonl",
    http_client: Any = httpx,
    timeout: float = 60.0,
) -> dict[str, Any]:
    payload = {"eval_path": eval_path}
    response = http_client.post(
        _join_url(api_base_url, "/eval/run"),
        json=payload,
        timeout=timeout,
    )
    if response.status_code != 200:
        raise ApiClientError(f"API request failed with status {response.status_code}: {response.text}")
    return response.json()


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"
