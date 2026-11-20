"""
The answer pipeline. One function, read it top to bottom:

    question -> other-company check -> live version -> search -> relevance gate
             -> LLM (1 call, 2nd only as fallback) -> check reply -> citations

Every way out returns the same dict shape:
    {"status": "answered" | "not_covered" | "error",
     "answer": str, "citations": [...], "llm_calls": int, "latency_ms": int}

"error" means "please try again" (the API turns it into HTTP 503).
"""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as SearchTimeout

from app import config, llm, retrieve, store
from app.auth import COMPANIES

log = logging.getLogger(__name__)

# Messages the employee sees. Plain words, always a next step.
NOT_COVERED = (
    "The {name} transport policy does not cover this question, so I can't give you a reliable answer. "
    "Please contact {desk}."
)
OTHER_COMPANY = (
    "I can only answer questions about the {name} transport policy. "
    "For anything else, please contact {desk}."
)
NO_POLICY = "No transport policy is loaded for {name} yet. Please contact {desk}."
TRY_AGAIN = (
    "Sorry, {problem}. Please try again in a minute. "
    "If it keeps happening, contact {desk}."
)

# Searches run in a worker thread so we can stop waiting after a timeout.
_search_pool = ThreadPoolExecutor(max_workers=4)


def _mentions_other_company(company, question):
    """True if the question names a different company, e.g. 'What is Orbit's rule?'"""
    words = set(re.findall(r"[a-z]+", question.lower()))
    for other_id, other in COMPANIES.items():
        if other_id != company and words & set(other["names_in_text"]):
            return True
    return False


def answer_question(company, question, providers=None):
    started = time.monotonic()
    deadline = started + config.ANSWER_DEADLINE_SECONDS
    info = COMPANIES[company]  # company comes from the verified token, so it exists
    llm_calls = 0

    # Small helper so every exit builds the same shape. It reads llm_calls at
    # the moment it is called, so it always reports the latest count.
    def reply(status, answer, citations=()):
        result = {
            "status": status,
            "answer": answer,
            "citations": list(citations),
            "llm_calls": llm_calls,
            "latency_ms": int((time.monotonic() - started) * 1000),
        }
        log.info("company=%s status=%s llm_calls=%d latency_ms=%d",
                 company, status, llm_calls, result["latency_ms"])
        return result

    name, desk = info["name"], info["transport_desk"]

    # 1. Asking about another company? Refuse in code, no model involved.
    if _mentions_other_company(company, question):
        return reply("not_covered", OTHER_COMPANY.format(name=name, desk=desk))

    # 2. Which version is live for THIS company?
    active = store.get_active(company)
    if active is None:
        return reply("not_covered", NO_POLICY.format(name=name, desk=desk))

    # 3. Search only this company's live policy, but don't wait forever.
    try:
        future = _search_pool.submit(retrieve.search, company, active["version"], question)
        chunks, best_distance = future.result(timeout=config.RETRIEVAL_TIMEOUT_SECONDS)
    except SearchTimeout:
        return reply("error", TRY_AGAIN.format(problem="looking up your policy is taking too long", desk=desk))
    except Exception:
        log.exception("Search failed")
        return reply("error", TRY_AGAIN.format(problem="I couldn't look up your policy right now", desk=desk))

    # 4. Relevance gate: nothing close enough means the policy is silent.
    #    No point paying for an LLM call to say "not covered".
    if not chunks or best_distance > config.MAX_DISTANCE:
        return reply("not_covered", NOT_COVERED.format(name=name, desk=desk))

    # 5. Ask the model, using at most 3 excerpts.
    system, user = llm.build_prompt(name, question, chunks)
    try:
        answer, llm_calls = llm.ask(system, user, providers or llm.pick_providers(), deadline)
    except llm.LLMError as error:
        llm_calls = error.calls
        return reply("error", TRY_AGAIN.format(problem="the answer service is not responding", desk=desk))

    # 6. Check the reply. The model said the policy doesn't answer it -> trust that "no".
    if not answer["covered"]:
        return reply("not_covered", NOT_COVERED.format(name=name, desk=desk))
    # Belt and braces: an answer naming another company is never shown.
    if _mentions_other_company(company, answer["answer"]):
        log.warning("Model answer named another company; replaced with not-covered")
        return reply("not_covered", NOT_COVERED.format(name=name, desk=desk))

    # 7. Citations come from OUR chunk metadata. The model only picks excerpt
    #    numbers; numbers that don't exist are dropped. If it picked none, we
    #    cite every excerpt it was shown (still at most 3).
    picked = [n for n in dict.fromkeys(answer["sources"]) if 1 <= n <= len(chunks)]
    cited = [chunks[n - 1] for n in picked] or chunks
    citations = [
        {"document": c["document"], "version": c["version"], "section": c["section"]}
        for c in cited[: config.TOP_K]
    ]
    return reply("answered", answer["answer"], citations)
