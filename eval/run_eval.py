"""
Run the fixed evaluation set and score it the same way every time.

    python -m eval.run_eval                 # both retrieval modes, real LLM if keys are set
    python -m eval.run_eval --offline       # never call an LLM (quotes the best section)
    python -m eval.run_eval --mode hybrid   # one mode only

Offline runs write offline_*.jsonl / offline_summary.json, so they never
overwrite the committed real-LLM results.

What it does:
  1. Fresh, throw-away storage. Seed the three v1 policies, then upload
     Nimbus v2 through the admin API, so every Nimbus answer must come from v2.
  2. Ask every question through the real HTTP API with that employee's JWT.
  3. Score each case (rules below) and write eval/results_<mode>.jsonl with
     exactly: id, input, expected, actual, pass, notes.
     eval/results.jsonl is the shipped mode (hybrid).
  4. Print and save a summary: pass rate, retrieval hit@1 / hit@3, p95 latency,
     LLM calls.

Scoring rules for one case:
  * status must match ("answered" or "not_covered")
  * answered cases: one citation must be an expected section; every
    must_include group needs one of its phrases in the answer; no
    must_not_include phrase may appear (catches cross-company and old-version
    leaks); every citation must be the asker's own document at the live version
  * all cases: at most 2 LLM calls and at most 3 citations
Retrieval is scored separately (did the expected section come back in the top 1 / top 3?).
"""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app import auth, config, llm, retrieve, store
from app.main import app

EVAL_DIR = Path(__file__).resolve().parent
EMPLOYEE = {"nimbus": "NIM-1001", "orbit": "ORB-1001", "vertex": "VTX-1001"}
DOCUMENT = {
    "nimbus": "Nimbus Labs Employee Transport Policy",
    "orbit": "Orbit Systems Commute Policy",
    "vertex": "Vertex Finance Transport Guidelines",
}


def load_cases():
    lines = (EVAL_DIR / "questions.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def score(case, actual):
    """Return (pass, list of reasons it failed)."""
    expected, problems = case["expected"], []
    company = case["input"]["company"]
    live_version = store.get_active(company)["version"]

    if actual["status"] != expected["status"]:
        problems.append(f"status {actual['status']!r}, expected {expected['status']!r}")

    if expected["status"] == "answered" and actual["status"] == "answered":
        answer = actual["answer"].lower()
        cited = [c["section"] for c in actual["citations"]]
        if expected["sections"] and not set(cited) & set(expected["sections"]):
            problems.append(f"cited {cited}, expected one of {expected['sections']}")
        for group in expected["must_include"]:
            if not any(phrase.lower() in answer for phrase in group):
                problems.append(f"answer lacks any of {group}")
        for phrase in expected["must_not_include"]:
            if phrase.lower() in answer:
                problems.append(f"answer contains forbidden {phrase!r}")
        for c in actual["citations"]:
            if c["document"] != DOCUMENT[company] or c["version"] != live_version:
                problems.append(f"citation from wrong document/version: {c}")

    if actual["llm_calls"] > config.MAX_LLM_CALLS:
        problems.append(f"{actual['llm_calls']} LLM calls")
    if len(actual["citations"]) > config.TOP_K:
        problems.append(f"{len(actual['citations'])} citations")
    return not problems, problems


def retrieval_hits(case):
    """(hit@1, hit@3) for answerable cases, None for 'not covered' ones."""
    expected = case["expected"]
    if expected["status"] != "answered" or not expected["sections"]:
        return None
    company = case["input"]["company"]
    chunks, _ = retrieve.search(company, store.get_active(company)["version"], case["input"]["question"])
    sections = [c["section"] for c in chunks]
    if not sections:
        return False, False
    return sections[0] in expected["sections"], bool(set(sections) & set(expected["sections"]))


def p95(values):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def run_mode(client, mode, cases, pause, prefix):
    config.RETRIEVAL_MODE = mode
    rows, latencies, calls, hits = [], [], [], []

    for case in cases:
        company, question = case["input"]["company"], case["input"]["question"]
        token = auth.issue_token(EMPLOYEE[company])
        started = time.monotonic()
        response = client.post("/ask", json={"question": question}, headers={"Authorization": f"Bearer {token}"})
        seconds = time.monotonic() - started
        body = response.json()

        passed, problems = score(case, body)
        hit = retrieval_hits(case)
        if hit is not None:
            hits.append(hit)
        latencies.append(seconds)
        calls.append(body["llm_calls"])

        actual = {
            "status": body["status"],
            "answer": body["answer"],
            "citations": body["citations"],
            "llm_calls": body["llm_calls"],
            "latency_ms": int(seconds * 1000),
        }
        notes = "; ".join(problems) if problems else "ok"
        if case.get("notes"):
            notes += f" | case: {case['notes']}"
        rows.append({"id": case["id"], "input": case["input"], "expected": case["expected"],
                     "actual": actual, "pass": passed, "notes": notes})
        print(f"  {'PASS' if passed else 'FAIL'} {case['id']:<10} {int(seconds * 1000):>5} ms  "
              f"calls={body['llm_calls']}  {body['status']:<11} {'' if passed else notes}")
        time.sleep(pause)

    with open(EVAL_DIR / f"{prefix}results_{mode}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return {
        "mode": mode,
        "cases": len(rows),
        "passed": sum(r["pass"] for r in rows),
        "pass_rate": round(sum(r["pass"] for r in rows) / len(rows), 3),
        "retrieval_hit_at_1": round(sum(h[0] for h in hits) / len(hits), 3),
        "retrieval_hit_at_3": round(sum(h[1] for h in hits) / len(hits), 3),
        "latency_p50_ms": int(statistics.median(latencies) * 1000),
        "latency_p95_ms": int(p95(latencies) * 1000),
        "llm_calls_max": max(calls),
        "llm_calls_mean": round(statistics.mean(calls), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Run the evaluation set.")
    parser.add_argument("--mode", choices=["vector", "hybrid", "both"], default="both")
    parser.add_argument("--offline", action="store_true", help="never call a real LLM")
    args = parser.parse_args()

    if args.offline:
        config.GROQ_API_KEYS, config.GEMINI_API_KEYS = [], []
    real_llm = bool(config.GROQ_API_KEYS or config.GEMINI_API_KEYS)
    # Free tiers allow roughly 30 requests a minute; pace real runs to stay under it.
    pause = 2.0 if real_llm else 0.0
    order = [name for name in config.LLM_ORDER if llm._keys(name)][: config.MAX_LLM_CALLS]
    llm_label = f"real LLM ({' then '.join(order)})" if real_llm else "offline (no LLM keys: quotes best section)"
    prefix = "" if real_llm else "offline_"
    cases = load_cases()
    modes = ["vector", "hybrid"] if args.mode == "both" else [args.mode]

    with tempfile.TemporaryDirectory() as storage:
        config.STORAGE_DIR = Path(storage)
        store.reset()
        with TestClient(app) as client:  # startup seeds the v1 policies
            admin = {"Authorization": f"Bearer {auth.issue_token('NIM-9001')}", "Content-Type": "text/markdown"}
            update = (config.POLICIES_DIR / "updates" / "nimbus_v2.md").read_bytes()
            assert client.post("/admin/policies", content=update, headers=admin).status_code == 200

            summaries = []
            for mode in modes:
                print(f"\n== {mode} retrieval, {llm_label}, {len(cases)} cases ==")
                summaries.append(run_mode(client, mode, cases, pause, prefix))
        store.reset()

    # results.jsonl = the mode the service ships with.
    shipped = config.RETRIEVAL_MODE if args.mode != "both" else "hybrid"
    (EVAL_DIR / f"{prefix}results.jsonl").write_text((EVAL_DIR / f"{prefix}results_{shipped}.jsonl").read_text())
    summary = {"llm": llm_label, "results": summaries}
    (EVAL_DIR / f"{prefix}summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n{'mode':<8} {'pass':>8} {'hit@1':>6} {'hit@3':>6} {'p50 ms':>7} {'p95 ms':>7} {'calls max':>9}")
    for s in summaries:
        print(f"{s['mode']:<8} {s['passed']:>3}/{s['cases']:<4} {s['retrieval_hit_at_1']:>6} "
              f"{s['retrieval_hit_at_3']:>6} {s['latency_p50_ms']:>7} {s['latency_p95_ms']:>7} {s['llm_calls_max']:>9}")
    print(f"\nWrote eval/{prefix}results.jsonl (shipped mode: {shipped}) and eval/{prefix}summary.json")


if __name__ == "__main__":
    main()
