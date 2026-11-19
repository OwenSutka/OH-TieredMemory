"""Fact-level writes: normalisation, dedupe, merge and supersede (plan §2.2.2, §2.2.6).

A fact is one declarative statement identified by ``content_hash`` (sha256 of its
normalised text), so two spellings that normalise alike are one row. Writing it
again merges: ``seen_count`` increments, ``distinct_sessions`` increments only
when the write arrives from a different ``source_session``, and ``confidence``
keeps the higher value. Nothing is ever deleted: contradiction is recorded by
setting ``superseded_by`` on the old row, which keeps its text and drops out of
:func:`live_facts` without leaving the store.

Every function takes a connection and never opens one. Callers get that
connection from :mod:`tiered_memory.store`, whose transaction (``BEGIN
IMMEDIATE``) is what makes the read-then-write in :func:`insert` atomic; the
store identity written on insert is read from that store's ``meta``, so there is
no module state and no second source of tier/scope truth.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from . import schema

__all__ = [
    "KINDS",
    "AlreadySuperseded",
    "Fact",
    "FactNotFound",
    "WriteResult",
    "content_hash",
    "insert",
    "live_facts",
    "normalise",
    "subject_key",
    "supersede",
]

#: §2.2.2: the only kinds the schema's CHECK constraint accepts.
KINDS = frozenset({"preference", "decision", "architecture", "fact", "todo", "gotcha"})

#: Default kind for a write that names none.
DEFAULT_KIND = "fact"

#: Plan §2.2.2: the normalised text is truncated to this many characters.
MAX_TEXT_CHARS = 500

#: Plan §2.2.2: ``subject_key`` keeps at most this many salient tokens.
MAX_SUBJECT_TOKENS = 6

_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

#: A fenced block: opening fence (info string allowed), body, closing fence on
#: its own line. An unterminated fence swallows the rest of the text.
_FENCE = re.compile(
    r"^[ \t]*(?:```|~~~)[^\n]*\n.*?(?:^[ \t]*(?:```|~~~)[ \t]*$|\Z)",
    re.DOTALL | re.MULTILINE,
)

_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"\w+")
_POSSESSIVE = re.compile(r"'s\b")

#: English function words carry no subject, so they never reach ``subject_key``.
_STOPWORD_CSV = (
    "a,about,above,after,again,against,all,also,am,an,and,any,are,as,at,be,because,been,before,"
    "being,below,between,both,but,by,can,could,did,do,does,doing,down,during,each,few,for,from,"
    "further,had,has,have,having,he,her,here,hers,herself,him,himself,his,how,i,if,in,into,is,it,"
    "its,itself,just,me,more,most,my,myself,no,nor,not,now,of,off,on,once,only,or,other,our,ours,"
    "ourselves,out,over,own,same,she,should,so,some,such,than,that,the,their,theirs,them,"
    "themselves,then,there,these,they,this,those,through,to,too,under,until,up,very,was,we,were,"
    "what,when,where,which,while,who,whom,why,will,with,would,you,your,yours,yourself,yourselves,"
)
_STOPWORDS = frozenset(_STOPWORD_CSV.split(","))


class FactNotFound(LookupError):
    """No fact has the requested id."""

    def __init__(self, fact_id: int) -> None:
        self.fact_id = fact_id
        super().__init__(f"fact {fact_id} does not exist")


class AlreadySuperseded(RuntimeError):
    """The fact already points at a successor, and history is never rewritten."""

    def __init__(self, fact_id: int, superseded_by: int) -> None:
        self.fact_id = fact_id
        self.superseded_by = superseded_by
        super().__init__(
            f"fact {fact_id} is already superseded by {superseded_by}; "
            "supersede the current fact instead"
        )


@dataclass(frozen=True, slots=True)
class WriteResult:
    """Outcome of :func:`insert`: the row id and whether it merged into an existing one."""

    id: int
    deduped: bool
    superseded_id: int | None = None


@dataclass(frozen=True, slots=True)
class Fact:
    """A live row, as returned by :func:`live_facts`."""

    id: int
    kind: str
    text: str
    subject_key: str
    seen_count: int
    distinct_sessions: int
    source_session: str | None


def normalise(text: str) -> str:
    """Return the canonical form of ``text`` used for hashing and storage.

    Fenced code blocks are removed with their contents, whitespace runs collapse
    to a single space, the result is trimmed, truncated to
    :data:`MAX_TEXT_CHARS` characters and lowercased (§2.2.2).

    Raises:
        ValueError: nothing but whitespace and code fences was left.
    """
    without_code = _FENCE.sub(" ", text)
    collapsed = _WHITESPACE.sub(" ", without_code).strip()
    normalised = collapsed[:MAX_TEXT_CHARS].lower()
    if not normalised:
        raise ValueError("fact text has no content once code fences and whitespace are removed")
    return normalised


def content_hash(text: str) -> str:
    """Return ``sha256`` of the normalised ``text`` (hex), the row's identity.

    Raises:
        ValueError: ``text`` normalises to nothing.
    """
    return _digest(normalise(text))


def subject_key(kind: str, text: str) -> str:
    """Return ``kind + ":" +`` up to six salient, distinct, non-stopword tokens.

    Tokens keep their order of first appearance in the text; possessives lose
    their ``'s`` suffix and repeated tokens are dropped. A text made only of
    stopwords yields ``"<kind>:"``.

    Raises:
        ValueError: ``kind`` is not one of :data:`KINDS`, or ``text`` normalises
            to nothing.
    """
    _check_kind(kind)
    stripped = _POSSESSIVE.sub(" ", normalise(text).replace("\u2019", "'"))
    tokens: list[str] = []
    for token in _TOKEN.findall(stripped):
        if token in _STOPWORDS or token in tokens:
            continue
        tokens.append(token)
        if len(tokens) == MAX_SUBJECT_TOKENS:
            break
    return f"{kind}:{' '.join(tokens)}"


def insert(
    conn: sqlite3.Connection,
    text: str,
    *,
    kind: str = DEFAULT_KIND,
    confidence: float = 1.0,
    source_tool: str | None = None,
    source_session: str | None = None,
    now: datetime | None = None,
) -> WriteResult:
    """Write one fact, merging into the existing row with the same ``content_hash``.

    On a duplicate, ``seen_count`` increments, ``last_seen_at`` and
    ``updated_at`` move to ``now``, ``confidence`` becomes the higher of the two
    and ``distinct_sessions`` increments only when ``source_session`` names a
    session other than the one the row was first seen from. The first
    ``source_session`` is kept: the schema records one session per row, and
    keeping it is what makes the count stable across interleaved sessions.

    The tier and scope recorded on a new row come from the store's own ``meta``,
    so the write can never claim an identity the store does not have.

    Args:
        conn: connection handed out by a :mod:`tiered_memory.store` transaction.
        text: the statement; stored normalised.
        kind: one of :data:`KINDS`.
        confidence: initial confidence, or the candidate value when merging.
        source_tool: tool that produced the fact, if any.
        source_session: session that produced the fact, if any.
        now: timestamp for this write; defaults to the current UTC time.

    Returns:
        The row id, whether it merged into an existing row, and (if any) the
        successor that already supersedes the merged row.

    Raises:
        ValueError: ``kind`` is unknown, or ``text`` normalises to nothing.
    """
    _check_kind(kind)
    normalised = normalise(text)
    digest = _digest(normalised)
    timestamp = _timestamp(now)
    meta = schema.read_meta(conn)

    existing = conn.execute(
        "SELECT id, source_session, superseded_by FROM facts WHERE content_hash = ?",
        (digest,),
    ).fetchone()
    if existing is not None:
        fact_id = int(existing[0])
        is_new_session = source_session is not None and source_session != existing[1]
        conn.execute(
            "UPDATE facts SET seen_count = seen_count + 1,"
            " distinct_sessions = distinct_sessions + ?,"
            " confidence = max(confidence, ?), updated_at = ?, last_seen_at = ?"
            " WHERE id = ?",
            (int(is_new_session), confidence, timestamp, timestamp, fact_id),
        )
        successor = int(existing[2]) if existing[2] is not None else None
        return WriteResult(id=fact_id, deduped=True, superseded_id=successor)

    row = conn.execute(
        "INSERT INTO facts (content_hash, subject_key, text, kind, tier, scope_key,"
        " source_tool, source_session, confidence, created_at, updated_at, last_seen_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
        (
            digest,
            subject_key(kind, normalised),
            normalised,
            kind,
            meta["tier"],
            meta["scope_key"],
            source_tool,
            source_session,
            confidence,
            timestamp,
            timestamp,
            timestamp,
        ),
    ).fetchone()
    if row is None:
        raise sqlite3.DatabaseError("insert returned no row id")
    return WriteResult(id=int(row[0]), deduped=False)


def supersede(
    conn: sqlite3.Connection, *, old_id: int, new_id: int, now: datetime | None = None
) -> None:
    """Record that ``new_id`` contradicts ``old_id``, keeping the old row.

    The old row keeps its text and gains ``superseded_by = new_id``; it then
    drops out of :func:`live_facts` but stays reachable as history. A row can be
    superseded once: history is never rewritten.

    Raises:
        ValueError: ``old_id`` and ``new_id`` are the same fact.
        FactNotFound: either id does not exist.
        AlreadySuperseded: ``old_id`` already points at a successor.
    """
    if old_id == new_id:
        raise ValueError("a fact cannot supersede itself")
    old = conn.execute("SELECT superseded_by FROM facts WHERE id = ?", (old_id,)).fetchone()
    if old is None:
        raise FactNotFound(old_id)
    if old[0] is not None:
        raise AlreadySuperseded(old_id, int(old[0]))
    if conn.execute("SELECT 1 FROM facts WHERE id = ?", (new_id,)).fetchone() is None:
        raise FactNotFound(new_id)
    conn.execute(
        "UPDATE facts SET superseded_by = ?, updated_at = ? WHERE id = ?",
        (new_id, _timestamp(now), old_id),
    )


def live_facts(conn: sqlite3.Connection, *, kind: str | None = None) -> list[Fact]:
    """Every fact that is neither superseded nor soft-deleted, ordered by id.

    Args:
        conn: connection handed out by a :mod:`tiered_memory.store` transaction.
        kind: restrict the result to one kind; ``None`` means every kind.

    Raises:
        ValueError: ``kind`` is not one of :data:`KINDS`.
    """
    if kind is not None:
        _check_kind(kind)
    statement = (
        "SELECT id, kind, text, subject_key, seen_count, distinct_sessions, source_session"
        " FROM facts WHERE superseded_by IS NULL AND deleted_at IS NULL"
    )
    params: tuple[object, ...] = ()
    if kind is not None:
        statement += " AND kind = ?"
        params = (kind,)
    statement += " ORDER BY id"
    return [
        Fact(
            id=int(row[0]),
            kind=str(row[1]),
            text=str(row[2]),
            subject_key=str(row[3]),
            seen_count=int(row[4]),
            distinct_sessions=int(row[5]),
            source_session=None if row[6] is None else str(row[6]),
        )
        for row in conn.execute(statement, params)
    ]


def _digest(normalised_text: str) -> str:
    return hashlib.sha256(normalised_text.encode("utf-8")).hexdigest()


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"unknown fact kind {kind!r}; expected one of {sorted(KINDS)}")


def _timestamp(moment: datetime | None) -> str:
    value = datetime.now(UTC) if moment is None else moment
    if value.tzinfo is not None:
        value = value.astimezone(UTC)
    return value.strftime(_TIMESTAMP_FORMAT)
