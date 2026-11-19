"""``SessionEnd`` consolidation: promotion, dedupe and the prune pass (plan §3.4, TM-11).

Three things happen at session end, in this order:

1. **session -> project.** Every live, undeleted fact in the session store is
   inserted into the project store, where an identical ``content_hash`` merges
   (``seen_count++``, and ``distinct_sessions++`` when the arriving fact names a
   session the project row has not seen). The session row is *retained*: §5.2
   row 5 adds a project row, it does not move one.
2. **project -> global.** A project row whose ``kind`` is ``preference`` and
   whose ``distinct_sessions >= 2`` is inserted into the global store. Nothing
   else is ever promoted (R2), and re-running this is a no-op because the global
   insert merges on ``content_hash``.
3. **prune pass.** Superseded rows older than ``older_than_days`` are counted,
   and removed only when ``dry_run_prune`` is ``False``. The default is a dry
   run, so the automatic path never deletes (R4). Live facts are never eligible.

Running consolidation twice is safe by construction: all three steps are
idempotent, which is what makes the "``SessionEnd`` may not fire" fallback of
plan §10 (run it on the next ``SessionStart``) harmless.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

from . import config, facts, store

__all__ = ["DEFAULT_OLDER_THAN_DAYS", "ConsolidationResult", "consolidate", "prune_candidates"]

#: ``memory_prune``'s default age threshold, in days.
DEFAULT_OLDER_THAN_DAYS = 180

#: §5.2 row 6: the only kind that may leave the project tier, ...
_PROMOTABLE_KIND = "preference"

#: ... and only once this many distinct sessions have seen it.
_PROMOTION_THRESHOLD = 2

#: Why the global insert says it happened; ``memory_write`` names the tool.
_PROMOTION_TOOL = "promotion"

#: Session-store ``meta`` key recording the project this session was promoted
#: into. Without it a second pass from the same session would look like a new
#: session and inflate ``distinct_sessions``, which is what decides global
#: promotion (R2). ``meta`` accepts extra keys; the schema's set is the plan's
#: minimum, not a whitelist.
_MARKER_KEY = "consolidated_into"

_SECONDS_PER_DAY = 86_400.0
_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True, slots=True)
class ConsolidationResult:
    """What one consolidation pass did.

    ``promoted`` counts session rows newly added to the project store,
    ``merged`` the ones that were already there; ``global_promoted`` is the
    project rows newly added to global. The prune counts are the dry-run
    candidates and how many were actually removed.
    """

    promoted: int
    merged: int
    global_promoted: int
    prune_candidates: int
    pruned: int


def consolidate(
    working_dir: str,
    session_id: str,
    *,
    dry_run_prune: bool = True,
    older_than_days: int = DEFAULT_OLDER_THAN_DAYS,
    now: datetime | None = None,
) -> ConsolidationResult:
    """Promote ``session_id`` into its project store, promote preferences, prune.

    Args:
        working_dir: the project whose tier receives the session's facts.
        session_id: the session whose store is read; a missing store is a no-op.
        dry_run_prune: when ``True`` (the default) the prune pass only counts.
        older_than_days: a superseded row must be at least this old to be pruned.
        now: the clock used for written timestamps; defaults to current UTC.

    Returns:
        The counts described by :class:`ConsolidationResult`.

    Raises:
        ValueError: ``session_id`` is not a safe path component; raised by
            :func:`tiered_memory.config.session_db` before any store is opened.
    """
    project_key = config.project_key(working_dir)
    rows: list[facts.Fact] = []
    sources: dict[int, str | None] = {}
    if config.session_db(session_id).exists():
        with _session(session_id) as session_conn:
            if _consolidation_marker(session_conn) != project_key:
                rows = facts.live_facts(session_conn)
                sources = {row.id: _source_tool(session_conn, row.id) for row in rows}

    promoted = merged = 0
    if rows:
        with _project(working_dir) as project_conn:
            for row in rows:
                result = facts.insert(
                    project_conn,
                    row.text,
                    kind=row.kind,
                    source_tool=sources[row.id],
                    source_session=row.source_session or session_id,
                    now=now,
                )
                if result.deduped:
                    merged += 1
                else:
                    promoted += 1
        _mark_consolidated(session_id, project_key)

    global_promoted = _promote_preferences(working_dir, now=now)
    candidates, pruned = _prune(working_dir, dry_run=dry_run_prune, older_than_days=older_than_days)
    return ConsolidationResult(
        promoted=promoted,
        merged=merged,
        global_promoted=global_promoted,
        prune_candidates=candidates,
        pruned=pruned,
    )


def prune_candidates(conn: sqlite3.Connection, cutoff: str) -> list[int]:
    """Superseded, undeleted rows older than ``cutoff`` that are not successors.

    A row that is itself the successor of another is excluded, so pruning can
    never break a supersession chain. The SQL matches ``server.memory_prune`` so
    the automatic and manual passes agree on exactly what is eligible.
    """
    rows = conn.execute(
        "SELECT id FROM facts AS f"
        " WHERE f.superseded_by IS NOT NULL"
        "   AND f.deleted_at IS NULL"
        "   AND f.last_seen_at < ?"
        "   AND NOT EXISTS (SELECT 1 FROM facts AS peer WHERE peer.superseded_by = f.id)",
        (cutoff,),
    ).fetchall()
    return [int(row[0]) for row in rows]


def _promote_preferences(working_dir: str, *, now: datetime | None) -> int:
    """Copy promotable project rows into global; return how many were inserted.

    Promotion carries the evidence that justified it: the global row ends with
    at least the project row's ``seen_count`` and ``distinct_sessions``. A second
    pass that finds the global row already carries that evidence changes
    nothing, which is what makes repeated consolidation idempotent.
    """
    if not config.project_db(working_dir).exists():
        return 0
    with _project(working_dir) as project_conn:
        promotable = [
            row
            for row in facts.live_facts(project_conn)
            if row.kind == _PROMOTABLE_KIND and row.distinct_sessions >= _PROMOTION_THRESHOLD
        ]
        if not promotable:
            return 0
        inserted = 0
        with _global() as global_conn:
            for row in promotable:
                if _promote_row(global_conn, row, now=now):
                    inserted += 1
        return inserted


def _promote_row(conn: sqlite3.Connection, row: facts.Fact, *, now: datetime | None) -> bool:
    """Insert or refresh one global row; ``True`` when a new row was created."""
    existing = conn.execute(
        "SELECT id, seen_count, distinct_sessions FROM facts WHERE content_hash = ?",
        (facts.content_hash(row.text),),
    ).fetchone()
    if existing is not None:
        fact_id, seen, distinct = (int(existing[0]), int(existing[1]), int(existing[2]))
        if seen >= row.seen_count and distinct >= row.distinct_sessions:
            return False
        conn.execute(
            "UPDATE facts SET seen_count = max(seen_count, ?),"
            " distinct_sessions = max(distinct_sessions, ?) WHERE id = ?",
            (row.seen_count, row.distinct_sessions, fact_id),
        )
        return False

    result = facts.insert(
        conn,
        row.text,
        kind=row.kind,
        source_tool=_PROMOTION_TOOL,
        source_session=row.source_session,
        now=now,
    )
    conn.execute(
        "UPDATE facts SET seen_count = ?, distinct_sessions = ? WHERE id = ?",
        (row.seen_count, row.distinct_sessions, result.id),
    )
    return True


def _prune(working_dir: str, *, dry_run: bool, older_than_days: int) -> tuple[int, int]:
    """Count (and optionally remove) prunable rows in an existing project store.

    A project with no store has nothing to prune, and consolidation must not
    create one just to prune it.
    """
    if not config.project_db(working_dir).exists():
        return 0, 0
    cutoff = _cutoff(older_than_days)

    with _project(working_dir) as conn:
        ids = prune_candidates(conn, cutoff)
        if dry_run or not ids:
            return len(ids), 0
        conn.executemany("DELETE FROM facts WHERE id = ?", [(fact_id,) for fact_id in ids])
        return len(ids), len(ids)


def _cutoff(older_than_days: int) -> str:
    now = datetime.now(UTC).replace(microsecond=0)
    return datetime.fromtimestamp(
        now.timestamp() - older_than_days * _SECONDS_PER_DAY, tz=UTC
    ).strftime(_TIMESTAMP_FORMAT)


def _source_tool(conn: sqlite3.Connection, fact_id: int) -> str | None:
    row = conn.execute("SELECT source_tool FROM facts WHERE id = ?", (fact_id,)).fetchone()
    return None if row is None or row[0] is None else str(row[0])


def _consolidation_marker(conn: sqlite3.Connection) -> str | None:
    """The project key this session was last promoted into, if any."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (_MARKER_KEY,)).fetchone()
    return None if row is None else str(row[0])


def _mark_consolidated(session_id: str, project_key: str) -> None:
    """Record the promotion on the session store so a repeat pass is a no-op."""
    with _session(session_id) as conn:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (_MARKER_KEY, project_key),
        )


@contextmanager
def _session(session_id: str) -> Iterator[sqlite3.Connection]:
    with store.session(session_id) as conn:
        yield _connection(conn)


@contextmanager
def _project(working_dir: str) -> Iterator[sqlite3.Connection]:
    with store.project(working_dir) as conn:
        yield _connection(conn)


@contextmanager
def _global() -> Iterator[sqlite3.Connection]:
    with store.global_store() as conn:
        yield _connection(conn)


def _connection(conn: sqlite3.Connection | store.HookSkipped) -> sqlite3.Connection:
    """Refuse a skipped lock: consolidation runs synchronously, not fail-open."""
    if not isinstance(conn, sqlite3.Connection):
        raise RuntimeError("consolidation could not take the store lock")
    return conn
