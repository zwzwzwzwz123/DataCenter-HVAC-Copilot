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
    _raise_for_bad_status(response)
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
    _raise_for_bad_status(response)
    return response.json()


def list_knowledge_documents_api(
    api_base_url: str,
    http_client: Any = httpx,
    timeout: float = 30.0,
) -> dict[str, Any]:
    response = http_client.get(_join_url(api_base_url, "/knowledge/documents"), timeout=timeout)
    _raise_for_bad_status(response)
    return response.json()


def get_knowledge_status_api(
    api_base_url: str,
    http_client: Any = httpx,
    timeout: float = 30.0,
) -> dict[str, Any]:
    response = http_client.get(_join_url(api_base_url, "/knowledge/status"), timeout=timeout)
    _raise_for_bad_status(response)
    return response.json()


def reindex_knowledge_api(
    api_base_url: str,
    http_client: Any = httpx,
    timeout: float = 60.0,
) -> dict[str, Any]:
    response = http_client.post(_join_url(api_base_url, "/knowledge/reindex"), timeout=timeout)
    _raise_for_bad_status(response)
    return response.json()


def upload_knowledge_document_api(
    api_base_url: str,
    filename: str,
    content: bytes,
    http_client: Any = httpx,
    timeout: float = 120.0,
) -> dict[str, Any]:
    response = http_client.post(
        _join_url(api_base_url, "/knowledge/documents/upload"),
        files={"file": (filename, content)},
        timeout=timeout,
    )
    _raise_for_bad_status(response)
    return response.json()


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _raise_for_bad_status(response: Any) -> None:
    if response.status_code != 200:
        raise ApiClientError(
            f"API request failed with status {response.status_code}: {response.text}"
        )
