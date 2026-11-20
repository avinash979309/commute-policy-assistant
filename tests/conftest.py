"""
Shared test setup.

* No test ever talks to a real LLM: API keys are blanked and tests hand the
  pipeline small fake "providers" (plain functions).
* Every test that needs data gets a brand-new storage folder with the three
  sample policies loaded, so tests can't affect each other.
"""

import json
import shutil

import pytest
from fastapi.testclient import TestClient

from app import auth, config, ingest, llm, store
from app.main import app


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEYS", [])
    monkeypatch.setattr(config, "GEMINI_API_KEYS", [])
    monkeypatch.setattr(config, "LLM_ORDER", ["groq", "gemini"])


@pytest.fixture(scope="session")
def seeded_template(tmp_path_factory):
    """Seed the sample policies ONCE; each test then gets its own copy (much faster)."""
    folder = tmp_path_factory.mktemp("seeded")
    real_dir = config.STORAGE_DIR
    config.STORAGE_DIR = folder
    store.reset()
    ingest.seed_default_policies()
    store.reset()
    config.STORAGE_DIR = real_dir
    return folder


@pytest.fixture
def fresh_store(seeded_template, tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    shutil.copytree(seeded_template, storage)
    monkeypatch.setattr(config, "STORAGE_DIR", storage)
    store.reset()
    yield
    store.reset()


@pytest.fixture
def client(fresh_store):
    with TestClient(app) as test_client:
        yield test_client


def fake_llm(covered=True, answer="Here is the answer.", sources=(1,)):
    """A fake provider that always returns this JSON reply."""
    def provider(system, user, timeout):
        return json.dumps({"covered": covered, "answer": answer, "sources": list(sources)})
    provider.name = "fake"
    return provider


def broken_llm(error=RuntimeError("provider down")):
    """A fake provider that always fails (timeout, 429, outage ...)."""
    def provider(system, user, timeout):
        raise error
    provider.name = "broken"
    return provider


def use_llm(monkeypatch, *providers):
    """Make the API use these fake providers for every question."""
    monkeypatch.setattr(llm, "pick_providers", lambda: list(providers))


def headers_for(employee_id):
    return {"Authorization": f"Bearer {auth.issue_token(employee_id)}"}


def read_policy(name):
    return (config.POLICIES_DIR / name).read_text()
