"""Declarative-signal fact extraction (plan §2.2.3, Step 10 / TM-10).

Extraction is a set of deterministic rules, not a model: no LLM, no embedding,
no network. Five signals from §2.2.3 are recognised:

===============  =========================================================
``preference``   ``always`` / ``never`` / ``prefer`` / ``don't ever`` …
``decision``     ``we decided`` / ``locked`` / ``going with`` / ``settled on``
``architecture`` writes to ``docs/adr/*``, ``README*``, ``pyproject.toml``, or
                 a ``*.toml`` that mentions dependencies
``gotcha``       a tool error followed by a successful retry
``todo``         a ``TODO``/``FIXME`` introduced by a write
===============  =========================================================

The refusals are as much a part of the contract as the matches: binary content,
content above :data:`MAX_CONTENT_BYTES` (4 KB), and anything shaped like a
credential are never turned into a fact. A candidate that trips a refusal is
dropped; it does not abort the batch.

Caps are hard, and exceeding one drops the lowest-confidence candidate instead
of queueing: at most :data:`MAX_PER_CALL` (3) facts per tool call and
:data:`MAX_PER_SESSION` (20) per session.

Only the *inputs* of a tool call are scanned for declarative statements -- the
command the agent ran, the text it wrote, and the message it was thinking -- not
the tool's response, which is usually the contents of a file or the output of a
command. The response is read only to recognise an error.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import sqlite3


__all__ = [
    "MAX_CONTENT_BYTES",
    "MAX_PER_CALL",
    "MAX_PER_SESSION",
    "Candidate",
    "extract",
    "looks_like_secret",
    "store_candidates",
]

#: §2.2.3: a tool call may contribute at most this many facts.
MAX_PER_CALL = 3

#: §2.2.3: a session may accumulate at most this many extracted facts.
MAX_PER_SESSION = 20

#: §2.2.3: file contents longer than this are never mined for facts.
MAX_CONTENT_BYTES = 4096

#: Sentence-ish units: a line, or a run of text ending in ``. ! ?``.
_SENTENCE = re.compile(r"[^.!?\n]+")
_LEADING_MARKUP = re.compile(r"^[\s#>*\-]+")
_PREFERENCE = re.compile(r"\b(always|never|prefer|don't ever|do not ever)\b", re.IGNORECASE)
_DECISION = re.compile(r"\b(we decided|locked|going with|settled on)\b", re.IGNORECASE)
_TODO = re.compile(r"\b(TODO|FIXME)\b")
_FIRST_LINE = re.compile(r"\S[^\n]*")

#: Tool-input keys whose string value is text the agent authored.
_TEXT_KEYS = ("command", "file_text", "content", "new_str", "text")

#: Tool-input keys that name a file the agent touched.
_PATH_KEYS = ("path", "file_path", "filename", "new_path")

_ARCHITECTURE_PATH = re.compile(
    r"(^|/)(docs/adr/.+|readme[^/]*|pyproject\.toml)$|\.toml$", re.IGNORECASE
)
_DEPENDENCY_MARKER = re.compile(r"^\s*\[?dependencies", re.IGNORECASE | re.MULTILINE)

#: Confidence per signal: a decision is the most deliberate statement, a todo
#: the most incidental, so a cap trims in exactly that order.
_CONFIDENCE: Mapping[str, float] = {
    "decision": 0.9,
    "preference": 0.85,
    "architecture": 0.8,
    "gotcha": 0.7,
    "todo": 0.6,
}

#: Credential shapes this plugin refuses to store (R: never store a secret).
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b(?:password|passwd|api[_-]?key|token|secret|credential)\b\s*[:=]\s*\S{6,}"),
    re.compile(r"\b[A-Fa-f0-9]{32,}\b"),
    re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"),
)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One fact the rules produced, before the caps are applied."""

    text: str
    kind: str
    confidence: float
    signal: str


def looks_like_secret(text: str) -> bool:
    """Whether ``text`` matches a credential shape this plugin refuses to store."""
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


def extract(
    *,
    tool_name: str,
    tool_input: Mapping[str, Any],
    message: str = "",
    previous_error: str | None = None,
    response_ok: bool = True,
) -> list[Candidate]:
    """Return the candidates one tool call produced, best confidence first.

    Args:
        tool_name: the tool that ran, recorded as ``source_tool``.
        tool_input: the call's arguments; only its authored text and the file
            path are read.
        message: the agent's message accompanying the call, if any.
        previous_error: the error that preceded this call for the same tool,
            which makes a successful call a ``gotcha``.
        response_ok: whether this call succeeded. A failure is not a fact; the
            caller records it and waits for the retry.

    Returns:
        At most :data:`MAX_PER_CALL` candidates, sorted by confidence and then
        text, so the cap always discards the least confident ones.
    """
    if not response_ok:
        return []

    candidates: list[Candidate] = []
    architecture = _architecture(tool_name, tool_input)
    if architecture is not None:
        candidates.append(architecture)
    for blob in _blobs(tool_input, message):
        candidates.extend(_sentence_candidates(blob))
    if previous_error is not None:
        gotcha = _gotcha(tool_name, previous_error)
        if gotcha is not None:
            candidates.append(gotcha)

    unique: dict[tuple[str, str], Candidate] = {}
    for candidate in candidates:
        if (
            looks_like_secret(candidate.text)
            or len(candidate.text.encode("utf-8")) > MAX_CONTENT_BYTES
        ):
            continue
        unique.setdefault((candidate.kind, candidate.text), candidate)
    ordered = sorted(unique.values(), key=lambda item: (-item.confidence, item.text))
    return ordered[:MAX_PER_CALL]


def store_candidates(
    conn: sqlite3.Connection,
    candidates: Sequence[Candidate],
    *,
    session_id: str,
    tool_name: str,
) -> list[int]:
    """Insert ``candidates`` into the session store under the per-session cap.

    The cap counts every row this plugin extracted for the session (any row with
    a ``source_tool``), so re-stating an existing fact merges instead of
    consuming a slot. Candidates are taken best-confidence first and the rest
    are dropped, never queued.

    Returns:
        The ids of the rows written (merged rows report their existing id).
    """
    # Imported here so extraction -- the part of this module a hook runs on every
    # tool call -- does not drag ``facts`` and ``schema`` (and their ``logging``
    # and ``inspect`` imports) onto that path.
    from . import facts

    remaining = MAX_PER_SESSION - _extracted_count(conn)
    if remaining <= 0:
        return []
    ids: list[int] = []
    for candidate in list(candidates)[:remaining]:
        result = facts.insert(
            conn,
            candidate.text,
            kind=candidate.kind,
            confidence=candidate.confidence,
            source_tool=tool_name,
            source_session=session_id,
        )
        ids.append(result.id)
    return ids


def _extracted_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT count(*) FROM facts WHERE source_tool IS NOT NULL").fetchone()
    return 0 if row is None else int(row[0])


def _blobs(tool_input: Mapping[str, Any], message: str) -> list[str]:
    """The authored text of a call, with binary and oversized blobs removed."""
    blobs = [message] if message else []
    blobs.extend(
        value for key in _TEXT_KEYS if isinstance(value := tool_input.get(key), str) and value
    )
    return [
        blob
        for blob in blobs
        if "\x00" not in blob and len(blob.encode("utf-8")) <= MAX_CONTENT_BYTES
    ]


def _sentence_candidates(blob: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    for raw in _SENTENCE.findall(blob):
        sentence = _LEADING_MARKUP.sub("", raw).strip()
        if not sentence:
            continue
        if _TODO.search(sentence):
            candidates.append(Candidate(sentence, "todo", _CONFIDENCE["todo"], "todo"))
        elif _DECISION.search(sentence):
            candidates.append(Candidate(sentence, "decision", _CONFIDENCE["decision"], "decision"))
        elif _PREFERENCE.search(sentence):
            candidates.append(
                Candidate(sentence, "preference", _CONFIDENCE["preference"], "preference")
            )
    return candidates


def _architecture(tool_name: str, tool_input: Mapping[str, Any]) -> Candidate | None:
    """A write to an architecture-defining file is an architecture fact."""
    path = _path(tool_input)
    if path is None or _ARCHITECTURE_PATH.search(path) is None:
        return None
    if path.lower().endswith(".toml"):
        body = _joined_text(tool_input)
        if "pyproject.toml" not in path.lower() and _DEPENDENCY_MARKER.search(body) is None:
            return None
    text = f"{tool_name} edited {path}, an architecture-defining file"
    return Candidate(text, "architecture", _CONFIDENCE["architecture"], "architecture")


def _gotcha(tool_name: str, previous_error: str) -> Candidate | None:
    match = _FIRST_LINE.search(previous_error.strip())
    if match is None:
        return None
    excerpt = match.group(0).strip()[:200]
    text = f"{tool_name} needed a retry: {excerpt}"
    return Candidate(text, "gotcha", _CONFIDENCE["gotcha"], "retry")


def _path(tool_input: Mapping[str, Any]) -> str | None:
    for key in _PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _joined_text(tool_input: Mapping[str, Any]) -> str:
    return "\n".join(value for key in _TEXT_KEYS if isinstance(value := tool_input.get(key), str))
