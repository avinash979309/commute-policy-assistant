# Commute Policy Assistant (Case Study B)

**Demo video (2 min):** https://youtu.be/Gnwsm5q0GTU

An assistant that answers an employee's transport questions **only from their
own company's current policy**, cites the document and section, says plainly
when the policy is silent, and never answers from another company's document.

Stack (as required): **Python, FastAPI, Chroma, a local embedding model
(all-MiniLM-L6-v2), free-tier LLMs (Groq `gpt-oss-120b`, Gemini
`gemini-3-flash-preview` as fallback), JWT.**

---

## 1. Run it (one command)

```bash
docker compose up --build
```

Then open **http://localhost:8000** (it redirects to the Swagger UI at `/docs`).
`--build` makes sure the image matches the code you have; on a fresh clone a
plain `docker compose up` does the same.

* The three sample policies load automatically on first start.
* **No API keys are committed** (the brief's rule). The real-LLM results are
  in `eval/results.jsonl` / `EVALUATION.md` and in the screen recording.
  To try real answers yourself, a free Groq key takes about a minute
  (https://console.groq.com, then API Keys) and is enough on its own; Gemini
  is only the fallback.
* LLM keys are optional. Copy `.env.example` to `.env` and add your free keys:
  ```
  GROQ_API_KEYS=gsk_aaa,gsk_bbb
  GEMINI_API_KEYS=AIza_ccc
  ```
  Without keys the service still runs in **offline mode**: it quotes the best
  matching policy section instead of writing a sentence. Everything else
  (auth, isolation, citations, versioning, "not covered") works the same.

### Run the tests (no keys, no network, ~30 s)

```bash
docker compose run --rm api pytest
```

### Run the evaluation set

```bash
docker compose run --rm api python -m eval.run_eval            # real LLM if keys are in .env
docker compose run --rm api python -m eval.run_eval --offline  # never calls an LLM
```

Writes `eval/results.jsonl` (fields `id, input, expected, actual, pass, notes`),
`eval/results_vector.jsonl`, `eval/results_hybrid.jsonl` and `eval/summary.json`.
The committed files are from a real-LLM run. Without keys (offline mode) the
same set runs in about 20 s and writes `eval/offline_*` files instead, so the
committed results are never overwritten. Offline it passes 32/38, because it
quotes the top section instead of reading all three (see `EVALUATION.md` 3a).

### Without Docker

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.lock   # Python 3.12
.venv/bin/uvicorn app.main:app --reload     # API on :8000
.venv/bin/pytest                            # tests
.venv/bin/python -m eval.run_eval           # eval
```

## 2. Try it

```bash
# 1. Log in (mock issuer). The company comes from our directory, not from you.
TOKEN=$(curl -s -X POST localhost:8000/auth/token -H 'Content-Type: application/json' \
        -d '{"employee_id":"NIM-1001"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')

# 2. Ask
curl -s -X POST localhost:8000/ask -H "Authorization: Bearer $TOKEN" \
     -H 'Content-Type: application/json' -d '{"question":"When do women get an escort on drops?"}'

# 3. An admin uploads a new version; from now on only v2 answers
ADMIN=$(curl -s -X POST localhost:8000/auth/token -H 'Content-Type: application/json' \
        -d '{"employee_id":"NIM-9001"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')
curl -s -X POST localhost:8000/admin/policies -H "Authorization: Bearer $ADMIN" \
     -H 'Content-Type: text/markdown' --data-binary @policies/updates/nimbus_v2.md
curl -s localhost:8000/policy -H "Authorization: Bearer $TOKEN"
```

Sample users (`data/employees.json`):

| employee_id | company | role |
|---|---|---|
| NIM-1001 / NIM-9001 | Nimbus Labs | employee / admin |
| ORB-1001 / ORB-9001 | Orbit Systems | employee / admin |
| VTX-1001 / VTX-9001 | Vertex Finance | employee / admin |

A reply always has the same shape:

```json
{
  "status": "answered",
  "answer": "On drops after 7:00 PM a woman employee is accompanied by a security escort until she reaches home.",
  "citations": [{"document": "Nimbus Labs Employee Transport Policy", "version": 2, "section": "3. Night Shift Safety"}],
  "llm_calls": 1,
  "latency_ms": 640
}
```

`status` is `answered`, `not_covered` (the policy is silent, or the question is
about another company; the message points to the transport desk) or `error`
(HTTP 503, "please try again").

## 3. How it works

```
POST /ask  (Bearer JWT, {"question"})
  auth.py       verify signature + expiry  ->  company = token["company"]   (only source)
  assistant.py  question names another company?          -> refuse, 0 LLM calls
                store: which version is live for company?  (SQLite pointer)
                retrieve: search ONLY that company's live collection, top 3   (1 s timeout)
                best match too far away?                    -> "not covered", 0 LLM calls
                llm: 1 call (Groq); 2nd call only if it fails (Gemini)        (≤ 2.8 s total)
                check reply JSON; citations built from chunk metadata by code
```

| File | Job |
|---|---|
| `app/config.py` | every setting and limit in one place |
| `app/auth.py` | mock JWT issuer, token verification |
| `app/store.py` | Chroma collection per (company, version) + SQLite "live version" pointer |
| `app/ingest.py` | policy markdown -> sections -> Chroma; safe version switch |
| `app/retrieve.py` | company-locked search; vector or hybrid (bonus) |
| `app/llm.py` | prompt, Groq/Gemini calls, fallback, reply checking, offline fake |
| `app/assistant.py` | the pipeline, top to bottom |
| `app/main.py` | routes and error handlers |

## 4. Design decisions

1. **Company comes only from the signed JWT.** `/ask` takes just
   `{"question"}`. A `company` field sent by the client is ignored (tested).
   The token's algorithm is pinned to HS256, so `alg: none` and forged
   payloads are rejected (tested).
2. **One Chroma collection per company *and* version** (`policy_nimbus_v2`).
   Search opens only the asking company's live collection, and also filters on
   `company` as a second lock. Another company's text never reaches the model,
   so no prompt, injected or not, can make it quote that text.
3. **A pointer decides which version is live.** Ingestion writes the whole new
   collection first, checks the count, then switches the SQLite pointer in one
   transaction. If anything fails, the new collection is deleted and the old
   version keeps answering. A crash halfway leaves only an unused collection,
   which is never searched and is cleaned up on the next try (tested by
   breaking the write halfway and by breaking the switch).
4. **One section = one chunk.** Policies are written in short, numbered
   sections; a section is one topic and is exactly what an employee wants
   cited. The heading becomes the citation.
5. **Citations are built by code, not written by the model.** The model only
   returns excerpt numbers (1–3); invalid numbers are dropped, and document,
   version and section come from chunk metadata. A citation can't point to
   something that wasn't retrieved.
6. **"Not covered" has two nets.** (a) In code: if even the best section is far
   from the question (cosine distance > 0.72), we answer "not covered" without
   calling the LLM. (b) The model must return `covered: false` when the
   excerpts don't answer the question.
7. **Instructions inside documents are treated as data.** Three layers:
   instruction-like sentences ("ignore your previous instructions…") are
   dropped at ingestion; excerpts go inside `<excerpt>` tags with `<`/`>`
   stripped so they can't fake or close a tag; the system prompt says excerpts
   are data. Isolation and citations never depend on the model, so the worst
   a successful injection could do is word an answer badly **inside the same
   company**.
8. **Two providers with fallback.** Groq (`openai/gpt-oss-120b`, low reasoning
   effort, ~0.5-1 s) first, then Gemini (`gemini-3-flash-preview`, minimal
   thinking, ~1.5 s) on a timeout, 429 quota error, 5xx error or broken JSON.
   Keys are rotated round-robin to spread free quota. Each call has a **hard
   wall-clock limit** (it runs in a worker thread), because HTTP client
   timeouts only limit each network read and a slowly trickling reply could
   otherwise run far past the deadline. Tests never use real providers: every
   test passes a fake one.

### LLM vs plain code

| Job | Who does it | Why |
|---|---|---|
| Understanding a paraphrased question, writing a short answer from the excerpts | **LLM** | language is what it is good at |
| Deciding whether the excerpts answer the question (`covered`) | **LLM** + code gate | judging meaning needs the LLM; the gate saves a call on obvious misses |
| Who the user is, which company, which version | **code** | must be impossible to talk around |
| Which text is searched | **code** (Chroma filter) | isolation can't depend on a prompt |
| Citations, ≤ 3 chunks, ≤ 2 calls, timeouts | **code** | must hold every time |
| Refusing questions about other companies | **code** (name check) | deterministic, free, testable |
| Error messages | **code** (fixed text) | always plain, never a stack trace |

## 5. Staying under the limits

| Limit | How |
|---|---|
| **≤ 2 LLM calls per question** | Normally exactly **1**. A 2nd call happens only if the 1st failed (fallback). `MAX_LLM_CALLS = 2` caps the loop. Questions that the gate or the company check answer use **0**. There is no "rewrite the question" or "judge the answer" call. Every reply reports `llm_calls`; tests check it never goes above 2. |
| **≤ 3 cited chunks** | Retrieval returns top 3 (`TOP_K = 3`), the model sees only those 3, and citations are capped at 3 in code (tested). |
| **3 s p95** | Local embeddings and local Chroma (~0.15 s together), a hard 1 s retrieval timeout, a fast first model, and a 2.8 s deadline for the whole answer: each LLM call gets at most 2 s (hard limit), and a fallback call only gets the time that is left (it isn't started if < 0.3 s remain). Measured p95 is in `EVALUATION.md`. |

## 6. Trade-offs

* **Refuse any question that names another company** (e.g. "Is my rule the same
  as Orbit's?"). Safe and deterministic, but it also refuses a few harmless
  questions. A fuzzy "the other company" still works, because only the
  asker's own text is searched anyway.
* **Fixed similarity threshold for "not covered".** Cheap and saves LLM calls,
  but a single number can be wrong at the edges; we set it on the eval set
  and keep the model's `covered=false` as the second net.
* **No second "judge" LLM call.** It would help catch a wrong answer, but it
  costs latency and a call. We check what can be checked in code (JSON shape,
  sources, other-company names) instead.
* **Hybrid search on 8-section policies gives only a small gain** (see
  `EVALUATION.md`). We ship it because it finds the right section in the top 3
  more often, and the model reads all 3; it costs about 5 ms and 40 lines.
* **Section-sized chunks** suit short policies. A real 30-page policy with long
  sections would need sections split into paragraphs, with the heading kept as
  the citation.

## 7. Limitations

* The mock login trusts the employee ID (a real system would use SSO). Swap
  `auth.issue_token` for the real issuer; nothing else changes.
* The injection filter is a keyword list. It catches obvious planted text; a
  reworded injection would reach the prompt layer, which is model-dependent.
* No conversation memory: each question stands alone (the brief asks for
  single questions).
* Old policy versions are deleted after a switch, so there is no rollback
  endpoint. Re-upload the old text with a higher version number instead.
* Offline mode (no keys) quotes the top section, so it can't say "not covered"
  for questions that pass the similarity gate (for example, pets).
