"""
Storage: where policy chunks and the "which version is live" pointer live.

Two pieces:
  1. Chroma. Every (company, version) gets its OWN collection, e.g.
     "policy_nimbus_v2". So one company's search physically cannot see
     another company's text, and dropping an old version is one call.
  2. SQLite, one tiny table `active_policy`: company -> live version.
     This pointer is the single source of truth. Retrieval only ever opens
     the collection the pointer names, so a half-written new version is
     invisible until the pointer is switched.
"""

import sqlite3

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from app import config

# Created on first use and then reused (opening them is slow-ish).
_client = None
_embedder = None


def reset():
    """Forget the Chroma client. Tests call this after pointing STORAGE_DIR elsewhere.
    The embedding model is kept: it doesn't depend on the folder and is slow to load."""
    global _client
    _client = None


def _chroma():
    global _client, _embedder
    if _embedder is None:
        # all-MiniLM-L6-v2 running locally through ONNX. No API key, no quota.
        _embedder = DefaultEmbeddingFunction()
    if _client is None:
        config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(config.STORAGE_DIR / "chroma"),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def warm_up():
    """Load the embedding model now, so the first employee question isn't slow."""
    _chroma()
    _embedder(["warm up"])


def collection_name(company, version):
    return f"policy_{company}_v{version}"


def create_collection(company, version):
    return _chroma().create_collection(
        collection_name(company, version),
        embedding_function=_embedder,
        configuration={"hnsw": {"space": "cosine"}},  # distance = 1 - cosine similarity
    )


def open_collection(company, version):
    return _chroma().get_collection(collection_name(company, version), embedding_function=_embedder)


def delete_collection(company, version):
    """Delete if it exists. Returns quietly if it doesn't."""
    try:
        _chroma().delete_collection(collection_name(company, version))
    except Exception:
        pass  # not found is fine: there was nothing to clean up


# ---------------- active-version pointer (SQLite) ----------------

def _db():
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.STORAGE_DIR / "registry.db")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS active_policy (
               company  TEXT PRIMARY KEY,
               version  INTEGER NOT NULL,
               document TEXT NOT NULL
           )"""
    )
    return conn


def get_active(company):
    """{'version': 2, 'document': 'Nimbus Labs ...'} or None if nothing is live."""
    conn = _db()
    try:
        row = conn.execute(
            "SELECT version, document FROM active_policy WHERE company = ?", (company,)
        ).fetchone()
    finally:
        conn.close()
    return {"version": row[0], "document": row[1]} if row else None


def set_active(company, version, document):
    """Switch the live version. One statement in one transaction: all or nothing."""
    conn = _db()
    try:
        with conn:  # commits on success, rolls back on error
            conn.execute(
                """INSERT INTO active_policy (company, version, document) VALUES (?, ?, ?)
                   ON CONFLICT(company) DO UPDATE SET version = excluded.version,
                                                      document = excluded.document""",
                (company, version, document),
            )
    finally:
        conn.close()
