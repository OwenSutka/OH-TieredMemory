"""Keyword and semantic retrieval, fused by recency-weighted RRF (plan §2.2.5).

The two retrievers are independent: :func:`keyword_hits` is FTS5 + ``bm25`` and
:func:`semantic_hits` is ``sqlite-vec``. :func:`recall` runs both over every
store it is handed and fuses the per-store ranked lists with the plan's formula:

    rrf(d)    = sum_r 1 / (60 + rank_r(d))
    decay(d)  = 0.5 ** (age_days(d) / half_life(kind(d)))     durable -> 1.0
    use_boost = min(1.30, 1 + 0.05 * log1p(hits(d)))
    score(d)  = rrf(d) * decay(d) * use_boost(d)

Scope is applied per store *before* fusion: each store is queried on its own and
its hits enter the fusion at rank 1 upwards, so a crowded global store cannot
fill the slots a project store's facts are competing for.

Nothing here imports ``sqlite-vec`` or ``fastembed``. The extension is used only
through SQL (a store that lacks it records ``index_state = 'fts_only'`` and the
vector query is skipped), and the embedding model reaches this module as the
``embedder`` callable the caller passes in. Recall therefore never fails because
of index state; it degrades to keyword-only.

RRF ties are broken by recency (fresher first), then by insertion order, which is
store position and then fact id. Recency is the right tie-breaker for a
"recency-weighted" ranking, and it is the only way a two-day-old *decision* can
outrank a six-month-old one: both are durable, so their decay terms are equal.
"""

from __future__ import annotations

import math
import re
import sqlite3
import struct
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeVar

from . import facts, schema

H = TypeVar("H", bound=Hashable)

__all__ = [
    "DEFAULT_LIMIT",
    "HALF_LIVES",
    "KEYWORD_LIMIT",
    "MAX_HIT_CHARS",
    "SEMANTIC_LIMIT",
    "Candidate",
    "DocKey",
    "Hit",
    "K",
    "decay",
    "fuse",
    "half_life",
    "keyword_hits",
    "match_expression",
    "recall",
    "reciprocal_rank",
    "score",
    "semantic_hits",
    "use_boost",
]

#: The RRF constant from §2.2.5.
K = 60

#: ``memory_recall``'s default page size.
DEFAULT_LIMIT = 8

#: §2.2.5: the keyword retriever's candidate depth before fusion.
KEYWORD_LIMIT = 50

#: §2.2.5: the semantic retriever's ``k``.
SEMANTIC_LIMIT = 50

#: §2.2.2: every hit is truncated to this many characters.
MAX_HIT_CHARS = 240

_USE_BOOST_WEIGHT = 0.05
_MAX_USE_BOOST = 1.30

#: §2.2.5 per-kind half-life in days. ``inf`` means the kind never decays.
HALF_LIVES: Mapping[str, float] = {
    "preference": math.inf,
    "architecture": math.inf,
    "decision": math.inf,
    "fact": 180.0,
    "todo": 30.0,
    "gotcha": 30.0,
}

#: Fusion key: fact ids are per store, so a hit is identified by both.
DocKey = tuple[int, int]

_TOKEN = re.compile(r"\w+")
_TIMESTAMP = "%Y-%m-%dT%H:%M:%SZ"
_SECONDS_PER_DAY = 86_400.0

#: Index states whose vectors may be read (§5.1).
_READY = "ready"


@dataclass(frozen=True, slots=True)
class Hit:
    """One recalled fact, already truncated and scored for presentation."""

    tier: str
    id: int
    text: str
    kind: str
    score: float
    superseded: bool


@dataclass(frozen=True, slots=True)
class Candidate:
    """The ranking inputs of one fact, independent of which retriever found it."""

    kind: str
    age_days: float
    hits: int


@dataclass(frozen=True, slots=True)
class _Row:
    """A fact row as the retrievers select it, before scoring."""

    id: int
    tier: str
    text: str
    kind: str
    superseded: bool
    hits: int
    last_seen_at: str

    def hit(self, score: float) -> Hit:
        return Hit(
            tier=self.tier,
            id=self.id,
            text=self.text[:MAX_HIT_CHARS],
            kind=self.kind,
            score=score,
            superseded=self.superseded,
        )

    def candidate(self, *, now: datetime) -> Candidate:
        return Candidate(kind=self.kind, age_days=age_days(self.last_seen_at, now), hits=self.hits)


def match_expression(query: str) -> str:
    """Turn free text into an FTS5 ``MATCH`` expression of quoted tokens.

    Every token is quoted, so FTS5 operators and punctuation in the query are
    data rather than syntax, and the tokens are OR-ed: a fact needs one of them
    to match. Porter stemming is applied by the index, so the tokens themselves
    are passed through unchanged. An empty result means the query has no
    word characters at all.
    """
    tokens = _TOKEN.findall(query)
    return " OR ".join(f'"{token}"' for token in tokens)


def keyword_hits(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = KEYWORD_LIMIT,
    kinds: frozenset[str] | set[str] | None = None,
    include_history: bool = False,
) -> list[Hit]:
    """Return the keyword retriever's hits, best ``bm25`` first.

    ``score`` is ``-bm25`` so that higher is better, as it is for a fused hit.
    Soft-deleted rows never appear; superseded rows appear only with
    ``include_history``.

    Raises:
        ValueError: ``kinds`` names a kind outside :data:`facts.KINDS`.
    """
    _check_kinds(kinds)
    if limit <= 0:
        return []
    return [
        row.hit(-relevance)
        for row, relevance in _keyword_rows(
            conn, query, limit=limit, kinds=kinds, include_history=include_history
        )
    ]


def semantic_hits(
    conn: sqlite3.Connection,
    query_vector: Sequence[float],
    *,
    limit: int = SEMANTIC_LIMIT,
    kinds: frozenset[str] | set[str] | None = None,
    include_history: bool = False,
) -> list[Hit]:
    """Return the vector retriever's hits, nearest first.

    ``score`` is ``-distance`` so that higher is better. The store must already
    be able to answer: an absent ``facts_vec`` yields no hits instead of an
    error, because recall degrades rather than fails (§5.1).

    Raises:
        ValueError: ``kinds`` names a kind outside :data:`facts.KINDS`.
    """
    _check_kinds(kinds)
    if limit <= 0 or len(query_vector) != schema.EMBEDDING_DIMENSION:
        return []
    return [
        row.hit(-distance)
        for row, distance in _semantic_rows(
            conn, query_vector, limit=limit, kinds=kinds, include_history=include_history
        )
    ]


def recall(
    stores: Sequence[sqlite3.Connection],
    query: str,
    *,
    now: datetime | None = None,
    limit: int = DEFAULT_LIMIT,
    kinds: frozenset[str] | set[str] | None = None,
    include_history: bool = False,
    embedder: Callable[[str], Sequence[float] | None] | None = None,
) -> list[Hit]:
    """Fuse the keyword (and, when ready, semantic) retrievers over ``stores``.

    Each store is queried separately and its hits enter the fusion at rank 1
    upwards, so scope is applied per store before fusion. The semantic retriever
    contributes only for a store whose ``index_state`` is ``ready`` and whose
    ``facts_vec`` exists; a missing or warming index, or an ``embedder`` that
    returns ``None``, leaves the keyword path untouched.

    Args:
        stores: connections from :mod:`tiered_memory.store`, in precedence order.
        query: free text.
        now: the moment ages are measured against; defaults to the current UTC time.
        limit: maximum hits returned.
        kinds: restrict to these kinds before scoring.
        include_history: keep superseded rows.
        embedder: maps the query to a vector, or to ``None`` when the model is
            not usable. ``None`` disables the semantic retriever outright.

    Raises:
        ValueError: ``kinds`` names a kind outside :data:`facts.KINDS`.
    """
    _check_kinds(kinds)
    if limit <= 0:
        return []
    moment = datetime.now(UTC) if now is None else now
    keyword_ranks: dict[DocKey, int] = {}
    semantic_ranks: dict[DocKey, int] = {}
    rows: dict[DocKey, _Row] = {}

    for index, conn in enumerate(stores):
        keyword_rows = _keyword_rows(
            conn, query, limit=KEYWORD_LIMIT, kinds=kinds, include_history=include_history
        )
        for rank, (row, _relevance) in enumerate(keyword_rows, 1):
            key = (index, row.id)
            keyword_ranks.setdefault(key, rank)
            rows.setdefault(key, row)
        semantic_rows = _semantic_for(conn, query, embedder, kinds, include_history)
        for rank, (row, _distance) in enumerate(semantic_rows, 1):
            key = (index, row.id)
            semantic_ranks.setdefault(key, rank)
            rows.setdefault(key, row)

    candidates = {key: row.candidate(now=moment) for key, row in rows.items()}
    return [
        rows[key].hit(value)
        for key, value in _rank(keyword_ranks, semantic_ranks, candidates)[:limit]
    ]


def reciprocal_rank(ranks: Sequence[int]) -> float:
    """Return ``sum_r 1 / (K + rank_r)`` over the 1-based ``ranks``."""
    return sum(1.0 / (K + rank) for rank in ranks)


def half_life(kind: str) -> float:
    """Return the kind's half-life in days (``inf`` for durable kinds).

    Raises:
        ValueError: ``kind`` is not one of :data:`facts.KINDS`.
    """
    try:
        return HALF_LIVES[kind]
    except KeyError:
        raise ValueError(
            f"unknown fact kind {kind!r}; expected one of {sorted(facts.KINDS)}"
        ) from None


def decay(kind: str, *, age_days: float) -> float:
    """Return ``0.5 ** (age_days / half_life(kind))``, clamped to the past.

    A durable kind, or a timestamp in the future, yields ``1.0``: nothing is
    ever ranked *up* for being new, only down for being old.

    Raises:
        ValueError: ``kind`` is not one of :data:`facts.KINDS`.
    """
    life = half_life(kind)
    if math.isinf(life) or age_days <= 0:
        return 1.0
    return float(0.5 ** (age_days / life))


def use_boost(hits: int) -> float:
    """Return ``min(1.30, 1 + 0.05 * log1p(hits))``.

    The cap is what bounds how far repeated use can raise a fact: at most 30 %
    above its unboosted score.
    """
    return min(_MAX_USE_BOOST, 1.0 + _USE_BOOST_WEIGHT * math.log1p(max(hits, 0)))


def score(rrf: float, *, kind: str, age_days: float, hits: int) -> float:
    """Return ``rrf * decay(kind, age_days) * use_boost(hits)``.

    Raises:
        ValueError: ``kind`` is not one of :data:`facts.KINDS`.
    """
    return rrf * decay(kind, age_days=age_days) * use_boost(hits)


def fuse(
    keyword: Mapping[H, int],
    semantic: Mapping[H, int],
    rows: Mapping[H, Candidate],
) -> list[H]:
    """Fuse ranked lists into document keys, best first.

    ``keyword`` and ``semantic`` map a document to its 1-based rank in that
    retriever's list; ``rows`` supplies the per-document scoring inputs. A tie is
    broken by recency, then by the order in which ``rows`` was filled, so the
    result is deterministic for a deterministic caller.
    """
    return [key for key, _value in _rank(keyword, semantic, rows)]


def _rank(
    keyword: Mapping[H, int],
    semantic: Mapping[H, int],
    rows: Mapping[H, Candidate],
) -> list[tuple[H, float]]:
    scored = [
        (
            key,
            score(
                reciprocal_rank(
                    [rank for rank in (keyword.get(key), semantic.get(key)) if rank is not None]
                ),
                kind=candidate.kind,
                age_days=candidate.age_days,
                hits=candidate.hits,
            ),
        )
        for key, candidate in rows.items()
    ]
    scored.sort(key=lambda item: (-item[1], rows[item[0]].age_days))
    return scored


def _keyword_rows(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int,
    kinds: frozenset[str] | set[str] | None,
    include_history: bool,
) -> list[tuple[_Row, float]]:
    expression = match_expression(query)
    if not expression:
        return []
    filters, params = _filters(kinds, include_history=include_history)
    statement = (
        "SELECT f.id, f.tier, f.text, f.kind, f.superseded_by, f.hits, f.last_seen_at,"
        " bm25(facts_fts)"
        " FROM facts_fts JOIN facts AS f ON f.id = facts_fts.rowid"
        f" WHERE facts_fts MATCH ? AND {filters}"
        # An equal bm25 is a tie the index cannot break; recency does, so a
        # fresher fact of the same relevance takes the better rank.
        " ORDER BY bm25(facts_fts) ASC, f.last_seen_at DESC, f.id ASC LIMIT ?"
    )
    try:
        rows = conn.execute(statement, (expression, *params, limit)).fetchall()
    except sqlite3.OperationalError:
        # Tokenisation quotes everything, so a rejected expression means the
        # index itself is unusable here; recall answers with no hits.
        return []
    return [(_row(row), float(row[7])) for row in rows]


def _semantic_rows(
    conn: sqlite3.Connection,
    query_vector: Sequence[float],
    *,
    limit: int,
    kinds: frozenset[str] | set[str] | None,
    include_history: bool,
) -> list[tuple[_Row, float]]:
    if not _vec_available(conn):
        return []
    filters, params = _filters(kinds, include_history=include_history)
    statement = (
        "SELECT f.id, f.tier, f.text, f.kind, f.superseded_by, f.hits, f.last_seen_at,"
        " knn.distance"
        " FROM (SELECT rowid, distance FROM facts_vec WHERE embedding MATCH ?"
        "       ORDER BY distance LIMIT ?) AS knn"
        " JOIN facts AS f ON f.id = knn.rowid"
        f" WHERE {filters}"
        " ORDER BY knn.distance ASC, f.last_seen_at DESC, f.id ASC LIMIT ?"
    )
    try:
        rows = conn.execute(
            statement, (_vector_blob(query_vector), SEMANTIC_LIMIT, *params, limit)
        ).fetchall()
    except sqlite3.Error:
        # The extension answered "not loaded" or the table is unusable: the
        # keyword retriever is the fallback, never an exception (§5.1).
        return []
    return [(_row(row), float(row[7])) for row in rows]


def _semantic_for(
    conn: sqlite3.Connection,
    query: str,
    embedder: Callable[[str], Sequence[float] | None] | None,
    kinds: frozenset[str] | set[str] | None,
    include_history: bool,
) -> list[tuple[_Row, float]]:
    if embedder is None or not _index_ready(conn):
        return []
    vector = embedder(query)
    if vector is None:
        return []
    return _semantic_rows(
        conn, vector, limit=SEMANTIC_LIMIT, kinds=kinds, include_history=include_history
    )


def _row(row: tuple[Any, ...]) -> _Row:
    return _Row(
        id=int(row[0]),
        tier=str(row[1]),
        text=str(row[2]),
        kind=str(row[3]),
        superseded=row[4] is not None,
        hits=int(row[5]),
        last_seen_at=str(row[6]),
    )


def _filters(
    kinds: frozenset[str] | set[str] | None, *, include_history: bool
) -> tuple[str, tuple[object, ...]]:
    clauses = ["f.deleted_at IS NULL"]
    params: list[object] = []
    if not include_history:
        clauses.append("f.superseded_by IS NULL")
    if kinds:
        clauses.append(f"f.kind IN ({', '.join('?' for _ in kinds)})")
        params.extend(sorted(kinds))
    return " AND ".join(clauses), tuple(params)


def _index_ready(conn: sqlite3.Connection) -> bool:
    try:
        return schema.read_meta(conn).get("index_state") == _READY
    except sqlite3.Error:
        return False


def _vec_available(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'facts_vec'"
    ).fetchone()
    return row is not None


def _vector_blob(vector: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def age_days(timestamp: str, now: datetime) -> float:
    """Return how many days before ``now`` ``timestamp`` was.

    An unparseable timestamp yields ``0.0``: a row whose clock is unreadable is
    ranked as if it were fresh rather than dropped or crashed on.
    """
    try:
        seen = datetime.strptime(timestamp, _TIMESTAMP).replace(tzinfo=UTC)
    except ValueError:
        return 0.0
    return (now - seen).total_seconds() / _SECONDS_PER_DAY


def _check_kinds(kinds: frozenset[str] | set[str] | None) -> None:
    if kinds is None:
        return
    unknown = set(kinds) - facts.KINDS
    if unknown:
        raise ValueError(
            f"unknown fact kind(s) {sorted(unknown)}; expected one of {sorted(facts.KINDS)}"
        )
