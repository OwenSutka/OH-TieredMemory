"""MCP stdio entry point: the ten tools of plan §3.1.

The server is long-lived and owns the embedding model (R6): hooks never import
:mod:`tiered_memory.embeddings`, and only this process ever loads it.

Tool results are the plan's shapes and nothing more. Failures are loud where the
plan names them -- an unknown tier or scope is ``ValueError``, an unconfigured
data directory is :class:`tiered_memory.config.ConfigError`, an export/import
defect is :class:`tiered_memory.portable.ExportFormatError`. An empty result is
never used to stand in for a refusal, because that would make a broken store
look like an empty one.

Reads never create a store: a tier whose file does not exist simply contributes
no hits. Writes and maintenance tools open through :mod:`tiered_memory.store`,
which creates and gates the store and takes ``BEGIN IMMEDIATE``.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from tiered_memory import config, embeddings, facts, portable, schema, search, store

__all__ = [
    "memory_export",
    "memory_forget",
    "memory_import",
    "memory_prune",
    "memory_recall",
    "memory_reindex",
    "memory_search_fts",
    "memory_stats",
    "memory_warm",
    "memory_write",
    "server",
]

_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

#: Filesystem-safe timestamp for a default export name.
_FILE_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

_SECONDS_PER_DAY = 86_400.0

#: Aggregate-state ordering: a set of stores is only as ready as its least ready
#: member, so semantic recall is never claimed for a store that cannot serve it.
_STATE_SEVERITY: dict[str, int] = {"fts_only": 0, "warming": 1, "pending": 2, "ready": 3}

#: Quoted warm-up cost when the model is not loaded yet: it is a one-time download.
_COLD_START_ETA_SECONDS = 120.0

server = MCPServer(
    "tiered-memory",
    instructions=(
        "Three-tier local memory. Use memory_write to record durable facts, "
        "memory_recall to search them, and memory_stats to inspect the tiers."
    ),
)

#: The one embedding manager for the process.
_manager: embeddings.IndexManager | None = None

#: Whether the model has been loaded at least once, which is what ``eta`` costs.
_backend_loaded = False


def _index_manager() -> embeddings.IndexManager:
    """Return the process-wide :class:`IndexManager`, creating it on first use."""
    global _manager
    if _manager is None:
        _manager = embeddings.IndexManager()
    return _manager


# --------------------------------------------------------------------------- #
# scope and tier resolution
# --------------------------------------------------------------------------- #


def _project_dir() -> str:
    """``OPENHANDS_PROJECT_DIR``: the project whose tier a call targets.

    Raises:
        config.ConfigError: the variable is unset, so no project tier exists.
    """
    value = os.environ.get("OPENHANDS_PROJECT_DIR", "").strip()
    if not value:
        raise config.ConfigError(
            "OPENHANDS_PROJECT_DIR is not set, so the project tier cannot be resolved. "
            "Set it, or pass an explicit tier/scope."
        )
    return value


def _session_id_optional() -> str | None:
    """``OPENHANDS_SESSION_ID`` when set, else ``None``."""
    value = os.environ.get("OPENHANDS_SESSION_ID", "").strip()
    return value or None


def _session_id() -> str:
    """``OPENHANDS_SESSION_ID``, validated as one path component by ``config``.

    Raises:
        config.ConfigError: the variable is unset or is not a safe filename
            component (``InvalidSessionId`` is a ``ConfigError``).
    """
    value = _session_id_optional()
    if value is None:
        raise config.ConfigError(
            "OPENHANDS_SESSION_ID is not set, so the session tier cannot be resolved. "
            "Set it, or pass tier='project'."
        )
    return value


def _resolve_tiers(scope: str | None) -> list[str]:
    """Turn a ``scope`` argument into the tiers it names, most general first.

    ``"auto"`` -- and ``None``, which means the same -- resolves from the
    runtime environment: ``global`` always; ``project`` when
    ``OPENHANDS_PROJECT_DIR`` is set; the caller's own ``session`` tier when
    ``OPENHANDS_SESSION_ID`` is set. It never names another session's store.

    Raises:
        ValueError: ``scope`` is not ``"auto"``, ``None``, or a tier name.
    """
    if scope is None or scope == "auto":
        tiers = ["global"]
        if os.environ.get("OPENHANDS_PROJECT_DIR", "").strip():
            tiers.append("project")
        if _session_id_optional() is not None:
            tiers.append("session")
        return tiers
    if scope in schema.VALID_TIERS:
        return [scope]
    raise ValueError(f"unknown scope {scope!r}; expected 'auto', 'global', 'project' or 'session'")


def _check_tier(tier: str) -> str:
    """Return ``tier`` if it is one of the three tiers.

    Raises:
        ValueError: ``tier`` is not ``global``, ``project`` or ``session``.
    """
    if tier not in schema.VALID_TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(schema.VALID_TIERS)}")
    return tier


def _tier_path(tier: str) -> Path:
    """The store file for ``tier``, resolved through :mod:`tiered_memory.config`."""
    _check_tier(tier)
    if tier == "global":
        return config.global_db()
    if tier == "project":
        return config.project_db(_project_dir())
    return config.session_db(_session_id())


def _entry(tier: str) -> contextlib.AbstractContextManager[sqlite3.Connection | store.HookSkipped]:
    """A :mod:`tiered_memory.store` transaction over ``tier``."""
    _check_tier(tier)
    if tier == "global":
        return store.global_store()
    if tier == "project":
        return store.project(_project_dir())
    return store.session(_session_id())


@contextlib.contextmanager
def _transaction(tier: str) -> Iterator[sqlite3.Connection]:
    """Open ``tier`` for writing; the fail-open sentinel is never requested here."""
    with _entry(tier) as conn:
        if not isinstance(conn, sqlite3.Connection):
            raise RuntimeError(f"tier {tier!r} refused the write lock unexpectedly")
        yield conn


@contextlib.contextmanager
def _existing_stores(tiers: Sequence[str]) -> Iterator[dict[str, sqlite3.Connection]]:
    """Open only the ``tiers`` whose file exists, keyed by tier name.

    A read must not create a store: an absent file is simply an empty tier.
    """
    with contextlib.ExitStack() as stack:
        conns: dict[str, sqlite3.Connection] = {}
        for tier in tiers:
            if not _tier_path(tier).exists():
                continue
            conn = stack.enter_context(_entry(tier))
            if isinstance(conn, sqlite3.Connection):
                conns[tier] = conn
        yield conns


def _existing_tiers(scope: str | None = None) -> list[str]:
    """The tiers in ``scope`` that already have a store file."""
    return [tier for tier in _resolve_tiers(scope) if _tier_path(tier).exists()]


def _now() -> str:
    return datetime.now(UTC).strftime(_TIMESTAMP_FORMAT)


def _aggregate_state(states: Sequence[str]) -> str:
    """The least ready of ``states``; ``empty`` when there are none."""
    if not states:
        return "empty"
    return min(states, key=lambda state: _STATE_SEVERITY.get(state, -1))


# --------------------------------------------------------------------------- #
# recall / search
# --------------------------------------------------------------------------- #


def _hit_dict(hit: search.Hit) -> dict[str, Any]:
    return {
        "tier": hit.tier,
        "id": hit.id,
        "text": hit.text,
        "kind": hit.kind,
        "score": hit.score,
        "superseded": hit.superseded,
    }


def _record_hits(conns: dict[str, sqlite3.Connection], hits: Sequence[search.Hit]) -> None:
    """Advance ``hits``/``last_hit_at`` for the rows a read returned.

    ``use_boost`` is a function of ``hits``, so a fact that answers queries has
    to earn the credit; without this it could never outrank a fresher one.
    """
    timestamp = _now()
    for tier, conn in conns.items():
        ids = [hit.id for hit in hits if hit.tier == tier]
        if ids:
            conn.executemany(
                "UPDATE facts SET hits = hits + 1, last_hit_at = ? WHERE id = ?",
                [(timestamp, fact_id) for fact_id in ids],
            )


@server.tool()
def memory_recall(
    query: str,
    scope: str = "auto",
    limit: int = search.DEFAULT_LIMIT,
    kinds: list[str] | None = None,
    include_history: bool = False,
) -> list[dict[str, Any]]:
    """Rank facts from the scoped tiers and record the use.

    ``scope="auto"`` resolves from the runtime environment: the ``global`` store
    always, the ``project`` store named by ``OPENHANDS_PROJECT_DIR`` when it is
    set, and the caller's own ``session`` store named by
    ``OPENHANDS_SESSION_ID`` when it is set -- never another session's store.
    ``"global"``, ``"project"`` and ``"session"`` restrict the search to that one
    tier.

    Keyword and (when the model is warm) semantic hits are fused per store by
    §2.2.5, so a crowded global store cannot take a project fact's slot. Every
    returned row has ``hits`` and ``last_hit_at`` advanced, which is what the
    ``use_boost`` term reads.
    """
    tier_names = _resolve_tiers(scope)
    with _existing_stores(tier_names) as conns:
        if not conns:
            return []
        hits = search.recall(
            list(conns.values()),
            query,
            limit=limit,
            kinds=None if kinds is None else frozenset(kinds),
            include_history=include_history,
            embedder=_index_manager().embed_query,
        )
        _record_hits(conns, hits)
    return [_hit_dict(hit) for hit in hits]


@server.tool()
def memory_search_fts(
    query: str,
    scope: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Answer from the keyword index only: the debug and fallback path.

    ``scope`` follows :func:`memory_recall`, with ``None`` meaning ``"auto"``.
    The semantic retriever never contributes, so a store whose index is not
    ready still answers here.
    """
    tier_names = _resolve_tiers(scope)
    depth = search.KEYWORD_LIMIT if limit is None else limit
    with _existing_stores(tier_names) as conns:
        hits = [
            hit for conn in conns.values() for hit in search.keyword_hits(conn, query, limit=depth)
        ]
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return [_hit_dict(hit) for hit in hits[:depth]]


# --------------------------------------------------------------------------- #
# writes
# --------------------------------------------------------------------------- #


@server.tool()
def memory_write(
    text: str,
    tier: str = "project",
    kind: str = facts.DEFAULT_KIND,
    confidence: float = 1.0,
) -> dict[str, Any]:
    """Write one fact, merging into an identical row if one exists.

    Returns the row id, whether the write merged (``deduped``), and the id of a
    successor that already supersedes this row when it has one. The text is
    stored normalised, and ``kind`` must be one of the six plan kinds.
    """
    _check_tier(tier)
    with _transaction(tier) as conn:
        result = facts.insert(
            conn,
            text,
            kind=kind,
            confidence=confidence,
            source_tool="memory_write",
            source_session=_session_id_optional(),
        )
    return {
        "id": result.id,
        "deduped": result.deduped,
        "superseded_id": result.superseded_id,
    }


@server.tool()
def memory_forget(id: int, tier: str) -> dict[str, Any]:
    """Soft-delete one fact: every read stops returning it, and nothing is dropped.

    ``forgotten`` is ``False`` when no live row has that id, so repeating a
    forget is a no-op instead of an error.
    """
    _check_tier(tier)
    with _transaction(tier) as conn:
        cursor = conn.execute(
            "UPDATE facts SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL",
            (_now(), id),
        )
        forgotten = cursor.rowcount > 0
    return {"forgotten": forgotten}


# --------------------------------------------------------------------------- #
# inspection and maintenance
# --------------------------------------------------------------------------- #


@server.tool()
def memory_stats() -> dict[str, Any]:
    """Report per-tier live counts, index state, model identity and warnings.

    Works without ``TIERED_MEMORY_DATA``: the counts are zero and the warning
    says why, because an inspection tool must never be the thing that fails.
    """
    try:
        tiers = _existing_tiers()
    except config.ConfigError as exc:
        return {
            "global": 0,
            "project": 0,
            "session": 0,
            "index_state": "empty",
            "model": schema.DEFAULT_EMBEDDING_MODEL,
            "dim": schema.EMBEDDING_DIMENSION,
            "warnings": [str(exc)],
        }

    counts = {"global": 0, "project": 0, "session": 0}
    states: list[str] = []
    meta: dict[str, str] = {}
    with _existing_stores(tiers) as conns:
        for tier, conn in conns.items():
            counts[tier] = _live_count(conn)
            meta = meta or schema.read_meta(conn)
            states.append(meta.get("index_state", "warming"))

    warnings: list[str] = []
    total = sum(counts.values())
    if total > config.FACT_WARNING_THRESHOLD:
        warnings.append(
            f"{total} live facts exceed the {config.FACT_WARNING_THRESHOLD} threshold; "
            "consider memory_prune or an export."
        )
    return {
        **counts,
        "index_state": _aggregate_state(states),
        "model": meta.get("embedding_model", schema.DEFAULT_EMBEDDING_MODEL),
        "dim": int(meta.get("embedding_dim", str(schema.EMBEDDING_DIMENSION))),
        "warnings": warnings,
    }


@server.tool()
def memory_reindex(tier: str | None = None, force: bool = False) -> dict[str, Any]:
    """Backfill missing vectors, rebuilding every vector when ``force`` is set.

    ``tier=None`` covers every existing store in the auto scope. A tier with no
    store is reported as ``empty`` rather than created just to embed nothing.
    """
    manager = _index_manager()
    tiers = _existing_tiers() if tier is None else _existing_named(tier)
    embedded = 0
    states: list[str] = []
    for name in tiers:
        with _transaction(name) as conn:
            result = manager.reindex(conn, force=force)
        embedded += result.embedded
        states.append(result.state)
    return {"embedded": embedded, "state": _aggregate_state(states)}


def _existing_named(tier: str) -> list[str]:
    """The one named ``tier``, but only when its store already exists."""
    _check_tier(tier)
    return [tier] if _tier_path(tier).exists() else []


@server.tool()
def memory_prune(
    tier: str | None = None,
    older_than_days: int = 180,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Remove superseded rows older than ``older_than_days``; a dry run by default.

    Only rows that are already superseded and are not themselves a successor are
    eligible, so a live fact is never removed and pruning cannot break a
    supersession chain. ``candidates`` is what the pass found and ``pruned`` is
    what it removed, so ``dry_run=True`` reports the candidates and removes
    nothing.
    """
    tiers = _existing_tiers() if tier is None else _existing_named(tier)
    cutoff = _cutoff(older_than_days)
    candidates = pruned = 0
    for name in tiers:
        with _transaction(name) as conn:
            ids = _prune_candidates(conn, cutoff)
            candidates += len(ids)
            if dry_run or not ids:
                continue
            conn.executemany("DELETE FROM facts WHERE id = ?", [(fact_id,) for fact_id in ids])
            pruned += len(ids)
    return {"candidates": candidates, "pruned": pruned}


@server.tool()
def memory_export(tier: str | None = None, path: str | None = None) -> dict[str, Any]:
    """Write the scoped facts as JSONL: a header line, then one line per fact.

    No vectors are written -- they belong to this store's model and are
    recomputed on import. ``path=None`` writes into ``${PLUGIN_DATA}/exports/``.
    The file is replaced atomically, so an interrupted export never clobbers the
    previous one.
    """
    tiers = _existing_tiers(tier)
    target = Path(path) if path is not None else _default_export_path()
    with _existing_stores(tiers) as conns:
        result = portable.export(list(conns.values()), target, tier=tier)
    return {"path": str(result.path), "facts": result.facts}


@server.tool()
def memory_import(path: str, retier: bool = False) -> dict[str, Any]:
    """Merge a JSONL export into the stores its records belong to.

    The header must name this plugin's ``embedding_model``/``embedding_dim``; a
    mismatch is refused with both the expected and the actual values. Records
    keep their tier unless ``retier=True``, which lands the whole file in the
    current ``project`` tier. Conflicts merge, an unchanged record is skipped,
    and re-importing the same file therefore changes nothing. Vectors are
    recomputed locally, never read from the file.
    """
    header, records = portable.parse(path)
    portable.require_identity(
        header,
        expected_model=schema.DEFAULT_EMBEDDING_MODEL,
        expected_dim=schema.EMBEDDING_DIMENSION,
    )
    imported = merged = skipped = 0
    for tier, tier_records in _group_by_tier(records, retier=retier).items():
        with _transaction(tier) as conn:
            result = portable.apply(conn, tier_records, retier=retier)
        imported += result.imported
        merged += result.merged
        skipped += result.skipped
        _backfill(tier)
    return {"imported": imported, "merged": merged, "skipped": skipped}


def _group_by_tier(
    records: Sequence[portable.FactRecord], *, retier: bool
) -> dict[str, list[portable.FactRecord]]:
    """Split records by the store each belongs to."""
    grouped: dict[str, list[portable.FactRecord]] = {}
    for record in records:
        tier = "project" if retier else record.tier
        _check_tier(tier)
        grouped.setdefault(tier, []).append(record)
    return grouped


def _backfill(tier: str) -> None:
    """Embed the rows an import just added, so imported facts are recallable."""
    with _transaction(tier) as conn:
        _index_manager().reindex(conn)


@server.tool()
def memory_warm(wait: bool = False) -> dict[str, Any]:
    """Report warm-up state, loading the model first when ``wait`` is set.

    ``eta_seconds`` is ``0`` once the model is loaded and a cold-start estimate
    otherwise. ``state`` is the aggregate index state of the existing stores, and
    ``empty`` when there are none.
    """
    manager = _index_manager()
    if wait:
        _load_backend(manager)
    try:
        tiers = _existing_tiers()
    except config.ConfigError:
        tiers = []
    states: list[str] = []
    for name in tiers:
        if wait:
            with _transaction(name) as conn:
                try:
                    states.append(manager.warm(conn))
                except embeddings.ModelUnavailable:
                    states.append("fts_only")
        else:
            with _existing_stores([name]) as conns:
                states.append(schema.read_meta(conns[name]).get("index_state", "warming"))
    return {
        "state": _aggregate_state(states),
        "model": manager.model_name,
        "eta_seconds": 0.0 if _backend_loaded else _COLD_START_ETA_SECONDS,
    }


def _load_backend(manager: embeddings.IndexManager) -> None:
    """Load the model once; a failure leaves the store on the FTS path (§5.1)."""
    global _backend_loaded
    try:
        manager.backend.warm()
    except embeddings.ModelUnavailable:
        _backend_loaded = False
        return
    _backend_loaded = True


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _live_count(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT count(*) FROM facts WHERE superseded_by IS NULL AND deleted_at IS NULL"
    ).fetchone()
    return 0 if row is None else int(row[0])


def _cutoff(older_than_days: int) -> str:
    # Timestamps are fixed-width UTC, so comparing them as text compares them in time.
    now = datetime.now(UTC).replace(microsecond=0)
    return datetime.fromtimestamp(
        now.timestamp() - older_than_days * _SECONDS_PER_DAY, tz=UTC
    ).strftime(_TIMESTAMP_FORMAT)


def _prune_candidates(conn: sqlite3.Connection, cutoff: str) -> list[int]:
    rows = conn.execute(
        "SELECT id FROM facts AS f"
        " WHERE f.superseded_by IS NOT NULL"
        "   AND f.last_seen_at < ?"
        "   AND NOT EXISTS (SELECT 1 FROM facts AS peer WHERE peer.superseded_by = f.id)",
        (cutoff,),
    ).fetchall()
    return [int(row[0]) for row in rows]


def _default_export_path() -> Path:
    stamp = datetime.now(UTC).strftime(_FILE_TIMESTAMP_FORMAT)
    return config.data_dir() / "exports" / f"tiered-memory-{stamp}.jsonl"


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
