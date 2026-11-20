"""
Ingestion: turn a policy file into searchable chunks, and switch versions safely.

A policy file looks like this:

    # Nimbus Labs Employee Transport Policy
    Company: nimbus
    Version: 2

    ## 1. Eligibility
    ...text...
    ## 2. Booking and Cut-off Timings
    ...text...

One "## " section becomes one chunk. A section is short, about one topic,
and has a heading we can cite, so it is the natural unit to search and quote.
"""

import logging
import re

from app import config, store
from app.auth import COMPANIES

log = logging.getLogger(__name__)


class IngestError(Exception):
    """The policy was NOT ingested. The message says why, in plain words."""


# A real policy talks to employees, never to "the AI". Sentences like
# "ignore your previous instructions" are planted text, so we drop them before
# they are stored. This is only the first net; the prompt rules in llm.py are
# the second, and isolation never depends on either.
# Known limit: a keyword list only catches obvious plants; reworded ones need a classifier.
SUSPICIOUS = re.compile(
    r"ignore (all |any |your |the )?(previous |prior |above )?instructions"
    r"|disregard .{0,40}instructions|\bAI assistant\b|system prompt",
    re.IGNORECASE,
)


def _drop_suspicious(body):
    """Return (body without instruction-like sentences, how many were dropped)."""
    sentences = re.split(r"(?<=[.!?\]])\s+", body)
    kept = [s for s in sentences if not SUSPICIOUS.search(s)]
    return " ".join(kept), len(sentences) - len(kept)


def parse_policy(text):
    """Read the header and the sections. Pure function: touches nothing."""
    title, company, version = None, None, None
    sections = []  # list of [heading, list of body lines]

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("## "):
            sections.append([line[3:].strip(), []])
        elif line.startswith("# ") and title is None:
            title = line[2:].strip()
        elif line.lower().startswith("company:") and not sections:
            company = line.split(":", 1)[1].strip().lower()
        elif line.lower().startswith("version:") and not sections:
            version = line.split(":", 1)[1].strip()
        elif line and sections:
            sections[-1][1].append(line)

    # Validate everything BEFORE we write anything.
    if not title:
        raise IngestError("The policy has no title line ('# Title').")
    if company not in COMPANIES:
        raise IngestError(f"Unknown or missing company in 'Company:' line: {company!r}.")
    if not version or not version.isdigit() or int(version) < 1:
        raise IngestError("The 'Version:' line must be a whole number like 1 or 2.")
    if not sections:
        raise IngestError("The policy has no sections ('## Heading').")

    chunks, dropped = [], 0
    for heading, body_lines in sections:
        body, dropped_here = _drop_suspicious(" ".join(body_lines))
        dropped += dropped_here
        if not body:
            raise IngestError(f"Section '{heading}' is empty.")
        chunks.append({"section": heading, "body": body})

    headings = [c["section"] for c in chunks]
    if len(set(headings)) != len(headings):
        raise IngestError("Two sections have the same heading; citations would be ambiguous.")

    return {"title": title, "company": company, "version": int(version), "chunks": chunks,
            "dropped_sentences": dropped}


def ingest_policy(text):
    """Store a new policy (or a new version) and make it the live one.

    Order matters, this is what keeps it all-or-nothing:
      1. parse + validate         (nothing written yet)
      2. write the new collection (old version is still the live one)
      3. check the write is complete
      4. flip the pointer         <- the commit point
      5. drop the old collection  (cleanup; harmless if it fails)
    If anything in 2-4 fails we delete what we wrote and the old version
    keeps answering as if nothing happened.
    """
    policy = parse_policy(text)
    company, version, title = policy["company"], policy["version"], policy["title"]

    active = store.get_active(company)
    if active and version <= active["version"]:
        raise IngestError(
            f"Version {version} is not newer than the live version {active['version']} for {company}."
        )

    try:
        store.delete_collection(company, version)  # leftovers from an earlier crash, if any
        collection = store.create_collection(company, version)
        collection.add(
            ids=[f"{company}-v{version}-{i}" for i in range(len(policy["chunks"]))],
            # The title + heading go into the text so the embedding knows the topic.
            documents=[f"{title} - {c['section']}\n{c['body']}" for c in policy["chunks"]],
            metadatas=[
                {"company": company, "version": version, "document": title, "section": c["section"]}
                for c in policy["chunks"]
            ],
        )
        if collection.count() != len(policy["chunks"]):
            raise IngestError("Not all sections were stored.")

        store.set_active(company, version, title)
    except Exception as error:
        store.delete_collection(company, version)  # roll back the half-written version
        log.exception("Ingestion failed for %s v%s", company, version)
        if isinstance(error, IngestError):
            raise
        raise IngestError("Saving the new policy failed, so the current version stays live. Please try again.")

    # The new version is live now. Removing the old one is just tidying up:
    # retrieval already ignores it because the pointer moved on.
    if active:
        store.delete_collection(company, active["version"])

    if policy["dropped_sentences"]:
        log.warning("%s v%s: dropped %d instruction-like sentence(s)", company, version, policy["dropped_sentences"])
    log.info("Policy live: %s v%s (%d sections)", company, version, len(policy["chunks"]))
    return {"company": company, "version": version, "document": title, "sections": len(policy["chunks"]),
            "dropped_sentences": policy["dropped_sentences"]}


def seed_default_policies():
    """At startup, load policies/*.md for any company that has no live policy yet."""
    for path in sorted(config.POLICIES_DIR.glob("*.md")):
        policy = parse_policy(path.read_text())
        if store.get_active(policy["company"]) is None:
            ingest_policy(path.read_text())
