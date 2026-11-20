"""
All settings live here, in one place.

Values come from environment variables (or the .env file), with safe defaults.
Other files always read them as `config.SOMETHING`, so tests can change a
setting with pytest's monkeypatch without touching the code.
"""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv():
    """Read KEY=VALUE lines from .env into os.environ (real env vars win).

    Docker Compose already loads .env for us; this is only for running
    locally with `uvicorn`, so we don't need an extra library for it.
    """
    env_file = PROJECT_DIR / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _key_list(name):
    """'a, b,,c' -> ['a', 'b', 'c']"""
    return [k.strip() for k in os.getenv(name, "").split(",") if k.strip()]


_load_dotenv()

# --- LLM providers (free tiers). Several keys each, used round-robin. ---
GROQ_API_KEYS = _key_list("GROQ_API_KEYS")
GEMINI_API_KEYS = _key_list("GEMINI_API_KEYS")
# Picked by measuring on this machine: gpt-oss-120b on Groq answers in ~0.5-1 s,
# gemini-3-flash-preview with minimal thinking in ~1.5 s (see EVALUATION.md).
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
# Which provider to try first. Providers without keys are skipped.
LLM_ORDER = [p.strip() for p in os.getenv("LLM_ORDER", "groq,gemini").split(",") if p.strip()]

# --- Auth ---
JWT_SECRET = os.getenv("JWT_SECRET", "dev-only-secret-please-change-me-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_MINUTES = 60

# --- Storage: Chroma folder + SQLite file live here ---
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", PROJECT_DIR / "storage"))
DATA_DIR = PROJECT_DIR / "data"          # companies.json, employees.json
POLICIES_DIR = PROJECT_DIR / "policies"  # policy files seeded at startup

# --- Retrieval ---
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")  # "vector" or "hybrid"
TOP_K = 3  # the brief allows at most 3 cited chunks per answer
# Cosine distance (0 = same meaning, 2 = opposite). If even the best chunk is
# further away than this, the policy does not talk about the question.
# Tuned on the eval set, see EVALUATION.md.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.72"))

# --- Limits from the brief ---
MAX_LLM_CALLS = 2               # at most 2 LLM calls per question
ANSWER_DEADLINE_SECONDS = float(os.getenv("ANSWER_DEADLINE_SECONDS", "2.8"))   # whole answer under 3 s
# Hard limit for one call. Groq usually answers in ~0.8 s; a slow one gets up to
# 2 s, because waiting a bit longer works better than switching late to Gemini
# (~1.5 s). Fast failures (quota, outage) still leave ~2.6 s for the fallback.
LLM_CALL_TIMEOUT_SECONDS = float(os.getenv("LLM_CALL_TIMEOUT_SECONDS", "2.0"))
RETRIEVAL_TIMEOUT_SECONDS = 1.0
MAX_QUESTION_CHARS = 500
