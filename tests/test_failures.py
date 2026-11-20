"""When things break, the employee gets a plain message, never a hang or a stack trace."""

import time

import httpx
from fastapi.testclient import TestClient

from app import assistant, config, retrieve
from app.main import app
from conftest import broken_llm, fake_llm, headers_for, use_llm

QUESTION = {"question": "Who approves an airport trip?"}


def test_slow_retrieval_returns_try_again_quickly(client, monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_TIMEOUT_SECONDS", 0.2)
    real_search = retrieve.search

    def slow_search(*args, **kwargs):
        time.sleep(1.0)
        return real_search(*args, **kwargs)
    monkeypatch.setattr(retrieve, "search", slow_search)

    started = time.monotonic()
    response = client.post("/ask", json=QUESTION, headers=headers_for("NIM-1001"))
    assert time.monotonic() - started < 0.9  # we stopped waiting, no hang
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert "try again" in body["answer"] and "taking too long" in body["answer"]
    assert body["llm_calls"] == 0


def test_retrieval_crash_returns_try_again(client, monkeypatch):
    def crash(*args, **kwargs):
        raise RuntimeError("chroma exploded: /secret/path line 42")
    monkeypatch.setattr(retrieve, "search", crash)
    response = client.post("/ask", json=QUESTION, headers=headers_for("NIM-1001"))
    assert response.status_code == 503
    assert "chroma" not in response.text and "secret" not in response.text


def test_llm_timeout_on_both_tries_returns_try_again(client, monkeypatch):
    use_llm(monkeypatch, broken_llm(httpx.ReadTimeout("timed out")), broken_llm(httpx.ReadTimeout("timed out")))
    response = client.post("/ask", json=QUESTION, headers=headers_for("NIM-1001"))
    assert response.status_code == 503
    body = response.json()
    assert "try again" in body["answer"] and "transport desk" in body["answer"]
    assert body["llm_calls"] == 2
    assert body["citations"] == []  # no half answer


def test_quota_error_falls_back_to_second_provider(client, monkeypatch):
    quota = httpx.HTTPStatusError("429", request=httpx.Request("POST", "http://x"),
                                  response=httpx.Response(429))
    use_llm(monkeypatch, broken_llm(quota), fake_llm(answer="Fallback worked."))
    body = client.post("/ask", json=QUESTION, headers=headers_for("NIM-1001")).json()
    assert body["status"] == "answered" and body["answer"] == "Fallback worked."


def test_no_second_call_when_deadline_is_used_up(fresh_store, monkeypatch):
    monkeypatch.setattr(config, "ANSWER_DEADLINE_SECONDS", 1.5)
    calls = []

    def slow_then_fail(system, user, timeout):
        calls.append(timeout)
        time.sleep(1.3)  # leaves less than 0.3 s of the deadline
        raise httpx.ReadTimeout("slow")

    result = assistant.answer_question("nimbus", QUESTION["question"], providers=[slow_then_fail, slow_then_fail])
    assert len(calls) == 1          # second call skipped: not enough time left
    assert calls[0] <= 1.5          # the call's own timeout fits inside the deadline
    assert result["status"] == "error"


def test_provider_ignoring_its_timeout_is_cut_off(fresh_store, monkeypatch):
    # Like a reply that trickles in slowly: the HTTP timeout never fires.
    monkeypatch.setattr(config, "ANSWER_DEADLINE_SECONDS", 1.0)

    def stuck(system, user, timeout):
        time.sleep(3)
        return fake_llm()(system, user, timeout)

    started = time.monotonic()
    result = assistant.answer_question("nimbus", QUESTION["question"], providers=[stuck, stuck])
    assert time.monotonic() - started < 1.5   # we stopped waiting at the deadline
    assert result["status"] == "error"


def test_slow_llm_answer_still_under_three_seconds(fresh_store):
    def provider(system, user, timeout):
        assert timeout <= config.LLM_CALL_TIMEOUT_SECONDS
        return fake_llm()(system, user, timeout)
    result = assistant.answer_question("nimbus", QUESTION["question"], providers=[provider])
    assert result["latency_ms"] < 3000


def test_empty_question_gets_plain_message(client):
    for bad in [{"question": ""}, {"question": "   "}, {"question": "x" * 501}, {}, {"question": 42}]:
        response = client.post("/ask", json=bad, headers=headers_for("NIM-1001"))
        assert response.status_code == 422
        assert "type your question" in response.json()["answer"]


def test_not_json_body_gets_plain_message(client):
    response = client.post("/ask", content=b"hello??", headers={**headers_for("NIM-1001"),
                                                               "Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["status"] == "error"


def test_unexpected_crash_shows_no_stack_trace(fresh_store, monkeypatch):
    def bug(*args, **kwargs):
        raise ZeroDivisionError("division by zero in secret_module.py")
    monkeypatch.setattr(assistant, "answer_question", bug)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/ask", json=QUESTION, headers=headers_for("NIM-1001"))
    assert response.status_code == 500
    assert "Traceback" not in response.text and "secret_module" not in response.text
    assert "try again" in response.json()["answer"]


def test_unknown_url_gets_plain_json(client):
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.json()["status"] == "error"


def test_home_page_opens_swagger(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
