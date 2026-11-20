"""
Retrieval: find the (at most) 3 policy sections that best match a question.

Isolation is decided HERE, in code, before any model sees anything:
we only open the collection of the asking employee's company, at its live
version, and we also filter on the company field as a second lock.

Two modes (config.RETRIEVAL_MODE):
  * "vector": meaning-based search with embeddings (Chroma).
  * "hybrid": vector search + BM25 keyword search, merged with Reciprocal
    Rank Fusion. This is the bonus "retrieval improvement".
"""

import re

from rank_bm25 import BM25Okapi

from app import config, store

RRF_K = 60  # standard constant for Reciprocal Rank Fusion

# Common words that match every section and only add noise to keyword search.
STOP_WORDS = set(
    "a an the is are am i me my we our you your it its of to in on at for and or but if do does "
    "did can could will would should what when who which how why there this that with from by be "
    "been as any all after before about".split()
)


def search(company, version, question, mode=None):
    """Return (chunks, best_distance).

    chunks: up to TOP_K dicts with text, document, section, version, company.
    best_distance: cosine distance of the closest chunk (smaller = closer),
                   used by the caller to decide "the policy doesn't cover this".
    """
    mode = mode or config.RETRIEVAL_MODE
    collection = store.open_collection(company, version)
    total = collection.count()
    if total == 0:
        return [], 2.0  # 2.0 = the largest cosine distance there is

    # Rank ALL sections of this one policy by meaning. Policies are short
    # (about 8 sections), so this costs nothing and hybrid needs the full list.
    result = collection.query(
        query_texts=[question],
        n_results=total,
        where={"company": company},  # second lock, on top of the per-company collection
        include=["documents", "metadatas", "distances"],
    )
    chunks = [
        {"text": text, "distance": distance, **meta}
        for text, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]
    if not chunks:
        return [], 2.0
    best_distance = chunks[0]["distance"]

    if mode == "hybrid":
        chunks = _rerank_with_keywords(question, chunks)

    return chunks[: config.TOP_K], best_distance


def _words(text):
    """Lower-case words for BM25, without stop words; 'trips' and 'trip' count as the same word."""
    words = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in STOP_WORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]  # naive plural -> singular
        words.append(word)
    return words


def _rerank_with_keywords(question, chunks):
    """Merge the vector ranking with a BM25 keyword ranking (RRF).

    Each chunk scores 1/(60 + its rank) in each list; scores are added.
    A chunk that is high in either list rises; one high in both wins.
    Helps with exact words like "no-show", "airport", "Saturday" that
    embeddings sometimes blur.
    """
    # The BM25 index is rebuilt per question. Fine for ~8 sections; cache it per
    # (company, version) if policies ever get large.
    bm25 = BM25Okapi([_words(c["text"]) for c in chunks])
    keyword_scores = bm25.get_scores(_words(question))
    keyword_order = sorted(range(len(chunks)), key=lambda i: keyword_scores[i], reverse=True)

    fused = [0.0] * len(chunks)
    for rank in range(len(chunks)):  # chunks arrive already in vector order
        fused[rank] += 1 / (RRF_K + rank + 1)
    for rank, i in enumerate(keyword_order):
        fused[i] += 1 / (RRF_K + rank + 1)

    order = sorted(range(len(chunks)), key=lambda i: fused[i], reverse=True)
    return [chunks[i] for i in order]
