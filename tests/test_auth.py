"""Identity comes from the JWT only, and only from a genuine, fresh token."""

import time

import jwt

from app import auth, config
from conftest import fake_llm, headers_for, use_llm


def make_token(**claims):
    base = {"sub": "NIM-1001", "company": "nimbus", "role": "employee", "exp": int(time.time()) + 600}
    base.update(claims)
    return jwt.encode(base, config.JWT_SECRET, algorithm="HS256")


def test_token_carries_company_from_our_directory(client):
    response = client.post("/auth/token", json={"employee_id": "ORB-1001"})
    claims = auth.verify_token(response.json()["access_token"])
    assert claims["company"] == "orbit"
    assert claims["role"] == "employee"


def test_login_cannot_choose_company(client):
    # A "company" field in the login body is ignored: the directory decides.
    response = client.post("/auth/token", json={"employee_id": "NIM-1001", "company": "vertex"})
    assert auth.verify_token(response.json()["access_token"])["company"] == "nimbus"


def test_unknown_employee_gets_plain_message(client):
    response = client.post("/auth/token", json={"employee_id": "NOPE-1"})
    assert response.status_code == 404
    assert "employee ID" in response.json()["answer"]


def test_missing_token_is_rejected(client):
    response = client.post("/ask", json={"question": "Who approves an airport trip?"})
    assert response.status_code == 401
    assert response.json()["status"] == "error"


def test_tampered_token_is_rejected(client):
    token = make_token()
    # Swap the payload for one claiming another company, keep the old signature.
    header, _, signature = token.split(".")
    _, forged_payload, _ = make_token(company="orbit").split(".")
    forged = f"{header}.{forged_payload}.{signature}"
    response = client.post("/ask", json={"question": "hi"}, headers={"Authorization": f"Bearer {forged}"})
    assert response.status_code == 401


def test_token_signed_with_other_secret_is_rejected(client):
    token = jwt.encode({"sub": "x", "company": "nimbus", "role": "employee", "exp": int(time.time()) + 600},
                       "some-other-secret-that-is-long-enough!!", algorithm="HS256")
    response = client.post("/ask", json={"question": "hi"}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_expired_token_is_rejected(client):
    token = make_token(exp=int(time.time()) - 10)
    response = client.post("/ask", json={"question": "hi"}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert "expired" in response.json()["answer"]


def test_alg_none_token_is_rejected(client):
    unsigned = jwt.encode({"sub": "x", "company": "orbit", "role": "admin", "exp": int(time.time()) + 600},
                          None, algorithm="none")
    response = client.post("/ask", json={"question": "hi"}, headers={"Authorization": f"Bearer {unsigned}"})
    assert response.status_code == 401


def test_token_for_unknown_company_is_rejected(client):
    token = make_token(company="evilcorp")
    response = client.post("/ask", json={"question": "hi"}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_company_in_question_body_is_ignored(client, monkeypatch):
    # The client tries to override the company. Retrieval must still use the token's company.
    use_llm(monkeypatch, fake_llm())
    response = client.post(
        "/ask",
        json={"question": "Who approves an airport trip?", "company": "orbit", "company_id": "vertex"},
        headers=headers_for("NIM-1001"),
    )
    body = response.json()
    assert body["status"] == "answered"
    assert all(c["document"] == "Nimbus Labs Employee Transport Policy" for c in body["citations"])
