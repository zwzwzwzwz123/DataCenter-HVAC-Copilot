import pytest

from app.api_client import ApiClientError, ask_api, run_eval_api
from app.streamlit_app import build_prediction_preview, group_eval_metrics


class FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = "error text"

    def json(self) -> dict:
        return self._payload


class FakeHttpClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.last_url = None
        self.last_json = None
        self.last_timeout = None

    def post(self, url: str, json: dict, timeout: float) -> FakeResponse:
        self.last_url = url
        self.last_json = json
        self.last_timeout = timeout
        return self.response


def test_ask_api_posts_question_and_task_type():
    http_client = FakeHttpClient(
        FakeResponse(
            200,
            {
                "question": "q",
                "route": "document_qa",
                "answer": "a",
                "tools": [],
                "citations": [],
                "retrieved_contexts": [],
                "tool_results": [],
            },
        )
    )

    result = ask_api(
        "http://localhost:8000",
        question="q",
        task_type="document_qa",
        http_client=http_client,
    )

    assert http_client.last_url == "http://localhost:8000/ask"
    assert http_client.last_json == {"question": "q", "task_type": "document_qa"}
    assert result["answer"] == "a"


def test_ask_api_raises_on_non_200_response():
    http_client = FakeHttpClient(FakeResponse(500, {"detail": "broken"}))

    with pytest.raises(ApiClientError, match="API request failed"):
        ask_api("http://localhost:8000", question="q", http_client=http_client)


def test_run_eval_api_posts_eval_path():
    http_client = FakeHttpClient(
        FakeResponse(
            200,
            {
                "metrics": {"citation_hit_rate": 0.6},
                "predictions": [{"id": "doc_qa_001"}],
            },
        )
    )

    result = run_eval_api(
        "http://localhost:8000/",
        eval_path="data/eval/hvac_eval.jsonl",
        http_client=http_client,
    )

    assert http_client.last_url == "http://localhost:8000/eval/run"
    assert http_client.last_json == {"eval_path": "data/eval/hvac_eval.jsonl"}
    assert result["metrics"]["citation_hit_rate"] == 0.6


def test_group_eval_metrics_splits_quality_proxy_metrics():
    grouped = group_eval_metrics(
        {
            "citation_hit_rate": 0.6,
            "context_recall": 0.7,
            "expected_keyword_coverage": 0.5,
            "lexical_answer_coverage": 0.2,
            "tool_selection_accuracy": 1.0,
            "tool_execution_success_rate": 1.0,
            "evidence_coverage": 0.8,
            "answer_correctness_proxy": 0.4,
            "faithfulness_proxy": 0.3,
        }
    )

    assert [name for name, _ in grouped["Retrieval"]] == [
        "citation_hit_rate",
        "context_recall",
    ]
    assert [name for name, _ in grouped["Quality Proxy"]] == [
        "answer_correctness_proxy",
        "faithfulness_proxy",
    ]


def test_build_prediction_preview_adds_evidence_flags_and_answer_length():
    preview = build_prediction_preview(
        [
            {
                "id": "doc_qa_001",
                "task_type": "document_qa",
                "route": "rag",
                "tools": [],
                "answer": "带引用的回答",
                "citations": [{"source_id": "doc"}],
                "tool_results": [],
            },
            {
                "id": "ts_query_001",
                "task_type": "timeseries_query",
                "route": "timeseries_query",
                "tools": ["query_metric"],
                "answer": "",
                "citations": [],
                "tool_results": [{"summary": {"count": 3}}],
            },
        ]
    )

    assert preview[0]["has_citation"] is True
    assert preview[0]["has_tool_result"] is False
    assert preview[0]["answer_length"] == len("带引用的回答")
    assert preview[1]["has_citation"] is False
    assert preview[1]["has_tool_result"] is True
