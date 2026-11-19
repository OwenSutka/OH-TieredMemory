"""SQLite schema, version gate and dimension guard for the three memory tiers.

The same DDL serves all three stores (plan §4); the tier is recorded in ``meta``
and nothing else differs. ``open_store`` is the only supported way to reach a
store: an existing file is inspected read-only first, so a refused open runs no
pragma and writes no byte, and a store that needs no state change is never
written to just because it was opened.

Creation is atomic: the whole bootstrap runs under ``BEGIN IMMEDIATE`` and a
second process that loses the race re-reads the winner's store instead of
failing. A failed bootstrap never unlinks the file, not even one this call
created: another process may already hold it open and be about to bootstrap a
store into it, and unlinking the path would strand that store in an orphaned
inode.

A pre-existing file is gated whatever its size. An empty leftover with no log
beside it is indistinguishable from a fresh store and is initialised; an empty
file whose log still holds pages belongs to a store that lost its identity and
is refused rather than overwritten.

The store path is contained before anything is created or opened: the leaf may
not be a symlink and the resolved path must stay inside its directory (and
inside the configured ``tiers/``, when the path is addressed through it).

``sqlite-vec`` is optional. This module never imports it at import time, and a
store whose extension cannot load is still opened with
``meta.index_state = 'fts_only'`` (plan §5.1) rather than raising.
"""

from __future__ import annotations

import contextlib
import errno
import importlib
import importlib.util
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from importlib.machinery import EXTENSION_SUFFIXES
from os import PathLike
from pathlib import Path
from typing import Any

from . import config

__all__ = [
    "BUSY_TIMEOUT_SECONDS",
    "DEFAULT_EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "INDEX_STATES",
    "META_KEYS",
    "SCHEMA_VERSION",
    "VALID_TIERS",
    "EmbeddingDimensionMismatch",
    "IncompleteSchemaError",
    "SchemaError",
    "SchemaVersionError",
    "StoreIdentityMismatch",
    "StorePathError",
    "is_lock_contention",
    "open_store",
    "read_meta",
]

_LOGGER = logging.getLogger(__name__)

#: How long lock waits may block before an open degrades (plan §3: 5 s).
BUSY_TIMEOUT_SECONDS = 5.0

#: Pause between retries while another process holds the SQLite lock.
_RETRY_SECONDS = 0.05

#: Busy timeout for the open-time pragma probe. A store locked by another
#: process must degrade quickly; a store merely mid-switch must still get a
#: few retries (``_switch_to_wal``).
_PROBE_MILLISECONDS = 200

#: Attempts at the bootstrap before a persistent lock is reported as a failure.
_CREATE_ATTEMPTS = 5

#: Attempts at snapshotting a store whose log is being checkpointed underneath
#: the gate. A peer deleting its ``-wal`` is ordinary WAL churn, not a defect.
_SNAPSHOT_ATTEMPTS = 5

#: How long the gate waits for a concurrent creator's first page before it
#: concludes a zero-byte file is genuinely empty. ``_ensure_file`` publishes
#: the name before SQLite writes page 1, so a peer can observe a 0-byte main
#: file beside a log that is not yet meaningful.
_FIRST_PAGE_WAIT_SECONDS = 0.05

_SNAPSHOT_PREFIX = "tiered-memory-gate-"

#: Bumped whenever the DDL in plan §4 changes; the gate compares against it.
SCHEMA_VERSION = 1

#: Width of ``all-MiniLM-L6-v2`` vectors and of ``facts_vec.embedding``.
EMBEDDING_DIMENSION = 384

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

#: One SQLite file per tier; the tier never changes the schema.
VALID_TIERS = frozenset({"global", "project", "session"})

#: Plan §5.1 index readiness states.
INDEX_STATES = frozenset({"warming", "pending", "ready", "fts_only"})

#: Meta keys named by plan §4. Not every key is seeded on a fresh store:
#: ``budget_tokens`` is written only when a project overrides the default.
META_KEYS = frozenset(
    {
        "schema_version",
        "tier",
        "scope_key",
        "embedding_model",
        "embedding_dim",
        "index_state",
        "budget_tokens",
        "created_at",
    }
)

_FTS_ONLY = "fts_only"

#: A new store has no vectors yet; the long-lived server backfills them.
_PENDING = "pending"

#: The only state that promises ``facts_vec`` exists (§5.1).
_READY = "ready"

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS facts (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        content_hash      TEXT    NOT NULL UNIQUE,
        subject_key       TEXT    NOT NULL,
        text              TEXT    NOT NULL,
        kind              TEXT    NOT NULL CHECK (kind IN
                            ('preference','decision','architecture','fact','todo','gotcha')),
        tier              TEXT    NOT NULL CHECK (tier IN ('global','project','session')),
        scope_key         TEXT    NOT NULL,
        source_tool       TEXT,
        source_session    TEXT,
        confidence        REAL    NOT NULL DEFAULT 1.0,
        seen_count        INTEGER NOT NULL DEFAULT 1,
        distinct_sessions INTEGER NOT NULL DEFAULT 1,
        created_at        TEXT    NOT NULL,
        updated_at        TEXT    NOT NULL,
        last_seen_at      TEXT    NOT NULL,
        superseded_by     INTEGER REFERENCES facts(id),
        deleted_at        TEXT,
        hits              INTEGER NOT NULL DEFAULT 0,
        last_hit_at       TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_facts_live
        ON facts(superseded_by) WHERE superseded_by IS NULL
    """,
    "CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(kind, subject_key)",
    "CREATE INDEX IF NOT EXISTS idx_facts_kind ON facts(kind)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
        text, content='facts', content_rowid='id', tokenize='porter unicode61'
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
      INSERT INTO facts_fts(rowid, text) VALUES (new.id, new.text);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
      INSERT INTO facts_fts(facts_fts, rowid, text) VALUES ('delete', old.id, old.text);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE OF text ON facts BEGIN
      INSERT INTO facts_fts(facts_fts, rowid, text) VALUES ('delete', old.id, old.text);
      INSERT INTO facts_fts(rowid, text) VALUES (new.id, new.text);
    END
    """,
)

#: Created separately: its failure degrades the store to FTS-only instead of
#: aborting the open.
_VEC_COLUMN = f"embedding float[{EMBEDDING_DIMENSION}]"
_FACTS_VEC_STATEMENT = f"CREATE VIRTUAL TABLE IF NOT EXISTS facts_vec USING vec0({_VEC_COLUMN})"


class SchemaError(RuntimeError):
    """Base class for every refusal caused by a store this module cannot serve."""


class SchemaVersionError(SchemaError):
    """The store's ``schema_version`` is not one this module can serve."""


class IncompleteSchemaError(SchemaError):
    """The store claims a servable version but lacks part of the §4 schema."""


class StoreIdentityMismatch(SchemaError):
    """The store's recorded tier/scope is not the one the caller asked for."""


class StorePathError(SchemaError, config.ConfigError):
    """The store path would leave the plugin's data root.

    A symlink at the store's leaf, or a ``tiers/`` directory that is (or
    resolves through) a symlink, would place the database -- and its
    ``-wal``/``-shm`` sidecars -- outside ``PLUGIN_DATA``. This is a refusal,
    never a degraded open, because writing there is the defect. It is both a
    ``SchemaError`` (the store cannot be served) and a ``ConfigError`` (the
    configured layout is unusable), so either layer's callers catch it.
    """


class EmbeddingDimensionMismatch(RuntimeError):
    """The store's vectors and the requested embedding width disagree.

    ``expected`` is the width this schema pins, ``actual`` is what the caller
    asked for or what the store records. The message always quotes the repair
    path: ``memory_reindex --force`` or deleting the store (plan §4, R9).
    """

    def __init__(self, expected: int, actual: int | str) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"embedding dimension mismatch: expected {expected}, got {actual}. "
            "Run memory_reindex --force to rebuild the vector index, or delete the "
            "store to start over."
        )


def read_meta(conn: sqlite3.Connection) -> dict[str, str]:
    """Every row of ``meta`` as ``{key: value}``, values coerced to ``str``."""
    rows = conn.execute("SELECT key, value FROM meta").fetchall()
    return {str(row[0]): str(row[1]) for row in rows}


def open_store(
    path: str | PathLike[str],
    *,
    tier: str,
    scope_key: str,
    embedding_dim: int = EMBEDDING_DIMENSION,
) -> sqlite3.Connection:
    """Open (creating if needed) the store at ``path`` and return its connection.

    The connection is in autocommit mode and carries no open transaction, so a
    second process can use the file as soon as this returns.

    An existing store is inspected through a read-only handle first: a refused
    open creates no ``-wal``/``-shm`` sidecar, runs no pragma and writes no
    byte. A store that needs no state change is not written to on open.

    ``scope_key`` must be a ``str``, like ``tier``: a non-string would be stored
    through SQLite's text affinity, so the identical call would later compare
    unequal to the coerced value it wrote and raise ``StoreIdentityMismatch``.
    Rejecting it up front keeps every successful call idempotent.

    If another process holds an existing store locked, the open degrades instead
    of raising: the pragmas that need the write lock are skipped and the
    connection is returned. Such a store was already gated through a read-only
    handle, so the refusal contract is unaffected; a later open retries the
    skipped state transitions. A store this call created is never another
    process's to lock, so a lock there is real contention and is retried.

    Raises:
        ValueError: ``tier`` is not one of ``VALID_TIERS``, or ``scope_key`` is
            not a ``str``; nothing is opened.
        StorePathError: ``path`` is a symlink, or resolves outside its directory
            (or outside ``tiers/`` when addressed through it). Nothing outside
            the store's directory is created or written.
        EmbeddingDimensionMismatch: ``embedding_dim`` or the value recorded in
            ``meta`` differs from ``EMBEDDING_DIMENSION``. A refused open does
            not create, rewrite or delete anything.
        SchemaVersionError: an existing store is unversioned, carries a version
            this module cannot serve, or lost its main file while its log still
            holds pages. Facts are never dropped.
        StoreIdentityMismatch: the store records a different tier or
            ``scope_key`` than the caller asked for.
        IncompleteSchemaError: the store is missing part of the §4 schema, or
            records an ``index_state`` that is not in ``INDEX_STATES``.
        sqlite3.DatabaseError: the file is not a SQLite database.
    """
    if tier not in VALID_TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(VALID_TIERS)}")
    if not isinstance(scope_key, str):
        # SQLite would store this through TEXT affinity, so the value written
        # would never equal the argument on the next open. Refuse like ``tier``.
        raise ValueError(
            f"scope_key must be a str, got {type(scope_key).__name__}; "
            "a coerced value would make the same call non-idempotent"
        )
    if embedding_dim != EMBEDDING_DIMENSION:
        raise EmbeddingDimensionMismatch(EMBEDDING_DIMENSION, embedding_dim)

    target = Path(path)
    _guard_store_path(target)
    created = _ensure_file(target)
    if not created:
        # A file that pre-existed is gated no matter its size: a store truncated
        # to zero bytes still owns the identity its meta recorded, and a crashed
        # journal may be the only remaining copy of it.
        _gate_read_only(target, tier=tier, scope_key=scope_key)

    conn: sqlite3.Connection | None = None
    try:
        conn = _connect(target)
        # Loaded before the first schema read: SQLite parses ``vec0`` tables once,
        # at read time, so a store adopted from a racing writer would otherwise
        # cache a schema without the extension's shadow tables.
        vec_ready = _load_sqlite_vec(conn)
        synced = _apply_pragmas(conn)
        if not synced and not created:
            # A store this call did not create is another process's to lock. It
            # was gated read-only above, so the refusal contract still holds;
            # return the connection and let a later open sync journal/index state.
            _LOGGER.warning("%s is locked; returning it without syncing state", target)
            return conn
        try:
            _adopt_or_create(conn, tier=tier, scope_key=scope_key, vec_ready=vec_ready)
            if not synced:
                # A racing creator held the file during the first attempt; with
                # the bootstrap done, the pragmas can be applied now.
                _apply_pragmas(conn)
        except sqlite3.OperationalError as exc:
            if created or not is_lock_contention(exc):
                raise
            _LOGGER.warning("%s is locked; returning it without syncing state: %s", target, exc)
    except BaseException:
        _release(conn)
        raise
    return conn


def _guard_store_path(target: Path) -> None:
    """Refuse a store path that could place the database outside its directory.

    The leaf is the only component the caller controls, so it must be a regular
    name (not a symlink, dangling or not) directly inside its directory. When
    the path is addressed through the configured ``tiers/``, that directory is
    resolved once and the resolved target must stay inside it, which also
    rejects a ``tiers/`` that is itself a symlink.
    """
    if target.is_symlink():
        raise StorePathError(
            f"{target} is a symlink; refusing to open it as a memory store because "
            "SQLite would write the database and its sidecars through the link"
        )
    parent = target.parent
    if parent.is_symlink():
        raise StorePathError(f"{parent} is a symlink; refusing to write memory stores through it")
    try:
        resolved_parent = parent.resolve(strict=True)
        resolved = target.resolve(strict=False)
    except OSError as exc:
        raise StorePathError(f"cannot resolve the store path {target}: {exc}") from exc
    if resolved.parent != resolved_parent or resolved.name != target.name:
        raise StorePathError(
            f"{target} resolves to {resolved}, outside its directory {resolved_parent}; "
            "refusing to open it as a memory store"
        )
    _guard_configured_tiers(target, resolved_parent)


def _guard_configured_tiers(target: Path, resolved_parent: Path) -> None:
    """Keep a store addressed through ``tiers/`` inside the resolved tiers dir."""
    if not os.environ.get(config.DATA_ENV, "").strip():
        # No plugin layout is configured (a direct library call): the leaf and
        # parent checks above already bound the path.
        return
    tiers = config.tiers_dir()
    if target.parent != tiers:
        return
    resolved_tiers = tiers.resolve()
    if resolved_tiers != resolved_parent or not target.resolve().is_relative_to(resolved_tiers):
        raise StorePathError(
            f"{target} resolves outside the tiers directory {resolved_tiers}; "
            "refusing to write the memory store there"
        )


def _ensure_file(target: Path) -> bool:
    """Create ``target`` if absent; report whether this call created it.

    ``O_EXCL`` never truncates a file another process made, and ``O_NOFOLLOW``
    on a directory descriptor makes the final component impossible to redirect
    through a symlink planted between the containment check and this call.
    Ownership is reported but deliberately unused for deletion: an unlink on
    failure is never safe (see ``_release``).
    """
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        dir_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as exc:
        raise StorePathError(f"cannot open the store directory {target.parent}: {exc}") from exc
    try:
        handle = os.open(target.name, flags, 0o600, dir_fd=dir_fd)
    except FileExistsError:
        return False
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            raise StorePathError(
                f"{target} is a symlink; refusing to create a memory store through it"
            ) from exc
        raise
    finally:
        os.close(dir_fd)
    os.close(handle)
    return True


def _connect(target: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(target), isolation_level=None, timeout=BUSY_TIMEOUT_SECONDS)
    conn.execute(f"PRAGMA busy_timeout = {int(BUSY_TIMEOUT_SECONDS * 1000)}")
    return conn


def _release(conn: sqlite3.Connection | None) -> None:
    """Close a connection whose open failed.

    The store file is deliberately never removed. Another process may already
    hold it open and be bootstrapping or committing to the very store this call
    could not finish; unlinking the path would leave that winner writing to an
    orphaned inode and its committed facts would be lost. A file left behind by
    a failed bootstrap therefore keeps whatever the winner wrote (or, if nothing
    was written, no schema at all, which the next open treats as fresh).
    """
    if conn is not None:
        with contextlib.suppress(sqlite3.Error):
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            conn.close()


def _gate_read_only(target: Path, *, tier: str, scope_key: str) -> None:
    """Refuse an unserviceable store without touching it.

    ``immutable=1`` lets SQLite read the main file without creating
    ``-wal``/``-shm`` sidecars or taking locks, so a refusal leaves the file
    byte-identical. That only holds when the main file is authoritative on its
    own, i.e. when no log beside it holds newer pages.

    A ``-wal`` that still has its ``-shm`` is a *live* store -- the normal state
    for a long-lived server -- and SQLite hands a read-only handle a consistent
    snapshot through that shared memory without creating a file, so it is gated
    in place. The main file and the WAL are never written; the only bytes a
    reader may touch are in the existing ``-shm``, which every reader of a live
    WAL updates and which is not durable store content. Only a ``-wal`` with no
    ``-shm`` (a crashed writer), an unusable one, or a hot ``-journal`` needs
    the log's pages copied aside, so the original directory stays untouched.

    A store that pre-existed and is now zero bytes cannot be read at all. When a
    ``-wal`` or ``-shm`` sits beside it, the store lost its main file while in
    use, so the open is refused instead of bootstrapping a new identity over
    it. Only an empty file with nothing but, at most, a bootstrap in flight is
    the fresh ``touch`` case.

    A peer may checkpoint or close between the check and the copy, deleting the
    log underneath the gate. That is WAL churn, not a defect: the snapshot is
    retried, and if the log keeps vanishing the main file is gated immutably,
    which is the only thing that cannot be raced away.
    """
    try:
        size = target.stat().st_size
    except OSError as exc:
        # ``_ensure_file`` reported that the file exists, so a failing stat is
        # an unusable path, never an empty store to bootstrap.
        raise SchemaError(f"cannot stat the store {target}: {exc}") from exc
    if size == 0:
        size = _await_first_page(target)
    if size == 0:
        if _lost_main_file(target):
            raise SchemaVersionError(
                f"store {target.name} is empty but its log or shared memory still holds "
                "state, so the store it belonged to lost its main file. This plugin "
                f"requires schema_version {SCHEMA_VERSION}; restore the file or delete "
                "the store."
            )
        return
    for attempt in range(1, _SNAPSHOT_ATTEMPTS + 1):
        if not _has_log(target):
            _inspect_read_only(target, "mode=ro&immutable=1", tier=tier, scope_key=scope_key)
            return
        if _wal_is_live(target):
            try:
                _inspect_read_only(target, "mode=ro", tier=tier, scope_key=scope_key)
                return
            except sqlite3.OperationalError:
                # A stale shared-memory file a crashed writer left behind cannot
                # be initialised through a read-only handle; the copy below
                # rebuilds the index from the WAL and reports the real verdict.
                pass
        try:
            _inspect_copy(target, tier=tier, scope_key=scope_key)
            return
        except FileNotFoundError:
            if attempt == _SNAPSHOT_ATTEMPTS:
                break
            time.sleep(_RETRY_SECONDS * attempt)
    _inspect_read_only(target, "mode=ro&immutable=1", tier=tier, scope_key=scope_key)


def _await_first_page(target: Path) -> int:
    """Wait briefly for a peer that created ``target`` to write page 1; return its size.

    ``_ensure_file`` creates the name with ``O_EXCL`` and returns before SQLite
    has initialised the file, so a second opener can legitimately see 0 bytes
    beside a ``-wal`` that the creator is about to fill. Waiting distinguishes
    that race from a store someone truncated to zero, which stays zero.
    """
    deadline = time.monotonic() + _FIRST_PAGE_WAIT_SECONDS
    while True:
        try:
            size = target.stat().st_size
        except OSError:
            return 0
        if size > 0 or time.monotonic() >= deadline:
            return size
        time.sleep(_RETRY_SECONDS / 5)


def _lost_main_file(target: Path) -> bool:
    """Whether sidecars beside an empty ``target`` still belong to a real store.

    Every store this module opens ends in WAL mode, so the newest committed
    pages of a store nobody checkpointed live in its ``-wal``, and any live
    writer keeps a ``-shm`` beside it. A main file of zero bytes next to either
    is therefore a store that lost its main file: its identity cannot be read,
    and bootstrapping a fresh one would record whatever ``scope_key`` the
    caller happened to pass. That is the "truncated while in use" case, and it
    is refused rather than silently rewritten.

    A transient rollback journal is *not* evidence of that. A writer creating a
    brand-new store publishes its ``O_EXCL`` name before SQLite writes page 1,
    and the short-lived journal beside it belongs to a bootstrap in flight.
    So an empty file with only a journal, or with nothing beside it at all, is
    left to the fresh-store path -- which is what a ``touch``ed file needs.
    """
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(target) + suffix)
        try:
            if sidecar.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


def _has_log(target: Path) -> bool:
    """Whether a WAL or hot journal beside ``target`` holds unmerged pages."""
    for suffix in ("-wal", "-journal"):
        log = Path(str(target) + suffix)
        try:
            if log.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


def _wal_is_live(target: Path) -> bool:
    """Whether the store's ``-wal`` is attached to the ``-shm`` of a live writer.

    SQLite keeps the shared-memory file beside the WAL for as long as any
    connection has the store open, so only a crashed writer leaves a ``-wal``
    without one. Telling the two apart is what lets the normal case avoid both
    the copy and a sidecar that a fresh read-only handle would create.
    """
    return Path(str(target) + "-wal").exists() and Path(str(target) + "-shm").exists()


def _inspect_copy(target: Path, *, tier: str, scope_key: str) -> None:
    """Gate a point-in-time copy of ``target`` and its log; the original is untouched.

    The copy is opened read-write on purpose: a hot journal (or a committed WAL
    a crashed writer left behind) is only replayed by a writer, and a
    read-only handle would instead fail with "attempt to write a readonly
    database" or, worse for an ``immutable`` handle, read the main file as if
    the log did not exist and report a missing ``schema_version``. The copy is
    in a private temporary directory, so replaying its log is free and the
    original directory is still never written.

    Raises ``FileNotFoundError`` when the log that ``_has_log`` saw disappears
    before it can be mirrored, so the caller retries and then gates the store in
    whatever state it settled into. Skipping the vanished log instead would gate
    a main file that is missing the newest committed pages.
    """
    with tempfile.TemporaryDirectory(prefix=_SNAPSHOT_PREFIX) as scratch:
        copy = Path(scratch) / target.name
        shutil.copyfile(target, copy)
        mirrored = False
        for suffix in ("-wal", "-journal"):
            log = Path(str(target) + suffix)
            if not log.exists():
                continue
            shutil.copyfile(log, Path(str(copy) + suffix))
            mirrored = True
        if not mirrored:
            raise FileNotFoundError(f"{target}: the log disappeared before it could be read")
        _inspect_read_only(copy, "mode=rw", tier=tier, scope_key=scope_key)


def _inspect_read_only(target: Path, params: str, *, tier: str, scope_key: str) -> None:
    uri = f"{target.resolve().as_uri()}?{params}"
    conn = sqlite3.connect(uri, uri=True, timeout=1.0)
    try:
        if not _table_exists(conn, "meta"):
            if _has_user_tables(conn):
                raise SchemaVersionError(
                    "store has tables but no meta/schema_version: the schema is unknown. "
                    f"This plugin requires schema_version {SCHEMA_VERSION}; delete or "
                    "recreate the store."
                )
            return
        _check_servable(conn, read_meta(conn), tier=tier, scope_key=scope_key)
    finally:
        conn.close()


def _check_servable(
    conn: sqlite3.Connection, meta: dict[str, str], *, tier: str, scope_key: str
) -> None:
    _check_schema_version(meta.get("schema_version"))
    _check_recorded_embedding_dim(meta.get("embedding_dim"))
    _check_identity(meta, tier=tier, scope_key=scope_key)
    _check_index_state(meta.get("index_state"))
    _check_complete(conn, meta)


def _adopt_existing(
    conn: sqlite3.Connection, *, tier: str, scope_key: str, vec_ready: bool
) -> None:
    """Gate an existing store and transition its state only when it is needed."""
    meta = read_meta(conn)
    _check_servable(conn, meta, tier=tier, scope_key=scope_key)
    _sync_index_state(conn, meta, vec_ready=vec_ready)


def _adopt_or_create(
    conn: sqlite3.Connection, *, tier: str, scope_key: str, vec_ready: bool
) -> None:
    """Gate an existing store, or bootstrap a new one under the write lock."""
    if _table_exists(conn, "meta"):
        _adopt_existing(conn, tier=tier, scope_key=scope_key, vec_ready=vec_ready)
    elif _has_user_tables(conn):
        raise SchemaVersionError(
            "store has tables but no meta/schema_version: the schema is unknown. "
            f"This plugin requires schema_version {SCHEMA_VERSION}; delete or "
            "recreate the store."
        )
    elif _create_store(conn, tier=tier, scope_key=scope_key, vec_ready=vec_ready):
        # Lost the creation race: the winner's store must be gated and synced
        # exactly like any other existing store.
        _adopt_existing(conn, tier=tier, scope_key=scope_key, vec_ready=vec_ready)


def _apply_pragmas(conn: sqlite3.Connection) -> bool:
    """Apply the open-time pragmas; ``False`` when another process holds the lock.

    ``foreign_keys`` is per-connection and never blocks, so it is always applied.
    ``journal_mode`` and ``synchronous`` need to read/write the store header, so
    an exclusively locked store (a rollback-journal one, in practice) refuses
    them; that is a degraded open, not a failure. WAL stores let readers in even
    under ``BEGIN EXCLUSIVE`` and take the normal path.

    The probe runs with a short busy timeout so an exclusively locked store is
    recognised immediately instead of after three full ``BUSY_TIMEOUT_SECONDS``
    waits. ``_switch_to_wal`` keeps its full retry budget, because two racing
    creators legitimately refuse each other's switch for a moment. The
    connection's normal timeout is restored before returning, so callers keep
    the documented wait behaviour.
    """
    conn.execute(f"PRAGMA busy_timeout = {_PROBE_MILLISECONDS}")
    try:
        synced = _sync_journal_pragmas(conn)
        conn.execute("PRAGMA foreign_keys = ON")
        return synced
    finally:
        conn.execute(f"PRAGMA busy_timeout = {int(BUSY_TIMEOUT_SECONDS * 1000)}")


def _sync_journal_pragmas(conn: sqlite3.Connection) -> bool:
    """Switch to WAL and apply ``synchronous``; ``False`` when the store is locked."""
    try:
        row = conn.execute("PRAGMA journal_mode").fetchone()
    except sqlite3.OperationalError as exc:
        if not is_lock_contention(exc):
            raise
        # Cannot even read the header: the store is exclusively locked, so the
        # switch below cannot succeed either. Degrade without trying.
        _LOGGER.warning("store is locked; leaving journal_mode unchanged: %s", exc)
        return False
    if row is None or str(row[0]).lower() != "wal":
        try:
            _switch_to_wal(conn)
        except sqlite3.OperationalError as exc:
            if not is_lock_contention(exc):
                raise
            _LOGGER.warning("store is locked; leaving journal_mode unchanged: %s", exc)
            return False
    try:
        conn.execute("PRAGMA synchronous = NORMAL")
    except sqlite3.OperationalError as exc:
        if not is_lock_contention(exc):
            raise
        _LOGGER.warning("store is locked; leaving synchronous unchanged: %s", exc)
        return False
    return True


def is_lock_contention(exc: sqlite3.Error) -> bool:
    """Whether ``exc`` means "another connection holds the store", not a defect.

    Classified by SQLite's own primary result code, never by the message:
    ``SQLITE_BUSY`` and ``SQLITE_LOCKED`` (including their extended codes) are
    contention, while the message text is not a contract and can quote a path
    that happens to contain "locked" or "busy".
    """
    code = getattr(exc, "sqlite_errorcode", None)
    if not isinstance(code, int):
        return False
    return code & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)


def _switch_to_wal(conn: sqlite3.Connection) -> None:
    """Switch to WAL, retrying while another process is mid-switch.

    SQLite refuses this switch immediately (not after ``busy_timeout``) when
    two processes reach it at once, so the retry is what makes a racing create
    succeed for both callers.
    """
    deadline = time.monotonic() + BUSY_TIMEOUT_SECONDS
    while True:
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            return
        except sqlite3.OperationalError as exc:
            if not is_lock_contention(exc) or time.monotonic() >= deadline:
                raise
            time.sleep(_RETRY_SECONDS)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def _has_user_tables(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchone()
    return row is not None and int(row[0]) > 0


def _check_schema_version(raw: str | None) -> None:
    if raw is None:
        raise SchemaVersionError(
            "store has no schema_version in meta: the schema is unknown. "
            f"This plugin requires schema_version {SCHEMA_VERSION}; delete or "
            "recreate the store."
        )
    # Exact match only: " 1", "+1" and "1_0" are not this schema's version.
    if raw != str(SCHEMA_VERSION):
        raise SchemaVersionError(
            f"store schema_version is {raw!r}, but this plugin requires exactly "
            f"{str(SCHEMA_VERSION)!r}. Delete the store to recreate it, or migrate "
            "it forward."
        )


def _check_recorded_embedding_dim(raw: str | None) -> None:
    if raw is None:
        raise EmbeddingDimensionMismatch(EMBEDDING_DIMENSION, "missing")
    try:
        recorded = int(raw)
    except ValueError:
        raise EmbeddingDimensionMismatch(EMBEDDING_DIMENSION, raw) from None
    if recorded != EMBEDDING_DIMENSION:
        raise EmbeddingDimensionMismatch(EMBEDDING_DIMENSION, recorded)


def _check_identity(meta: dict[str, str], *, tier: str, scope_key: str) -> None:
    recorded_tier = meta.get("tier")
    recorded_scope = meta.get("scope_key")
    if recorded_tier != tier or recorded_scope != scope_key:
        raise StoreIdentityMismatch(
            f"store belongs to tier {recorded_tier!r} scope {recorded_scope!r}, but it "
            f"was opened as tier {tier!r} scope {scope_key!r}; refusing to hand back "
            "the wrong store. Delete the store to recreate it."
        )


def _check_index_state(raw: str | None) -> None:
    if raw not in INDEX_STATES:
        raise IncompleteSchemaError(
            f"store meta.index_state is {raw!r}, not one of {sorted(INDEX_STATES)}; "
            "the index state is unknown. Delete the store to recreate it, or run "
            "memory_reindex --force."
        )


def _check_complete(conn: sqlite3.Connection, meta: dict[str, str]) -> None:
    missing = [name for name in ("facts", "facts_fts") if not _table_exists(conn, name)]
    if missing:
        raise IncompleteSchemaError(
            f"store schema_version {meta.get('schema_version')} is missing required "
            f"tables {sorted(missing)}: the schema is incomplete. Delete the store to "
            "recreate it, or run memory_reindex --force."
        )
    if meta.get("index_state") == _READY and not _table_exists(conn, "facts_vec"):
        raise IncompleteSchemaError(
            "store records index_state='ready' but facts_vec is absent: the vector "
            "index is incomplete. Run memory_reindex --force to rebuild it, or delete "
            "the store to start over."
        )


def _create_store(conn: sqlite3.Connection, *, tier: str, scope_key: str, vec_ready: bool) -> bool:
    """Bootstrap a new store atomically.

    Returns ``True`` when another process won the race, i.e. the store already
    existed by the time we held the write lock.
    """
    for attempt in range(1, _CREATE_ATTEMPTS + 1):
        try:
            with _transaction(conn):
                # Checked under the write lock, so "absent" cannot be stale.
                if _table_exists(conn, "meta"):
                    return True
                _bootstrap(conn, tier=tier, scope_key=scope_key, vec_ready=vec_ready)
            return False
        except sqlite3.OperationalError as exc:
            if not is_lock_contention(exc) or attempt == _CREATE_ATTEMPTS:
                raise
            time.sleep(_RETRY_SECONDS)
    raise AssertionError("unreachable: the final attempt either returns or raises")


def _bootstrap(conn: sqlite3.Connection, *, tier: str, scope_key: str, vec_ready: bool) -> None:
    index_state = _PENDING if vec_ready else _FTS_ONLY

    for statement in _SCHEMA_STATEMENTS:
        conn.execute(statement)

    if vec_ready and not _create_facts_vec(conn):
        index_state = _FTS_ONLY

    conn.executemany(
        "INSERT INTO meta (key, value) VALUES (?, ?)",
        _seed_meta(tier=tier, scope_key=scope_key, index_state=index_state),
    )


def _seed_meta(*, tier: str, scope_key: str, index_state: str) -> list[tuple[str, str]]:
    return [
        ("schema_version", str(SCHEMA_VERSION)),
        ("tier", tier),
        ("scope_key", scope_key),
        ("embedding_model", DEFAULT_EMBEDDING_MODEL),
        ("embedding_dim", str(EMBEDDING_DIMENSION)),
        ("index_state", index_state),
        ("created_at", datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")),
    ]


@contextlib.contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[None]:
    # The connection is in autocommit mode, so `with conn` would not open one.
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        with contextlib.suppress(sqlite3.Error):
            conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def _sync_index_state(conn: sqlite3.Connection, meta: dict[str, str], *, vec_ready: bool) -> None:
    """Bring ``index_state`` in line with what the extension can do right now.

    A needed write is best-effort: if the store is locked by another process,
    the open still succeeds and the next open retries the transition.
    """
    if not vec_ready:
        if meta.get("index_state") != _FTS_ONLY:
            _degrade_to_fts_only(conn)
        return
    if not _table_exists(conn, "facts_vec"):
        # The extension came back after a store was created FTS-only: restore the
        # §4 DDL and hand the rows to the backfill worker.
        _restore_vector_index(conn, was_fts_only=meta.get("index_state") == _FTS_ONLY)


def _degrade_to_fts_only(conn: sqlite3.Connection) -> None:
    try:
        _set_meta_value(conn, "index_state", _FTS_ONLY)
    except sqlite3.OperationalError as exc:
        _LOGGER.warning("store is locked; leaving index_state unchanged: %s", exc)


def _restore_vector_index(conn: sqlite3.Connection, *, was_fts_only: bool) -> None:
    try:
        with _transaction(conn):
            if _create_facts_vec(conn) and was_fts_only:
                _set_meta_value(conn, "index_state", _PENDING)
    except sqlite3.OperationalError as exc:
        _LOGGER.warning("could not rebuild facts_vec on open: %s", exc)


def _create_facts_vec(conn: sqlite3.Connection) -> bool:
    """Create ``facts_vec``; ``False`` when the extension cannot honour it."""
    try:
        conn.execute(_FACTS_VEC_STATEMENT)
    except sqlite3.Error:
        return False
    return True


def _set_meta_value(conn: sqlite3.Connection, key: str, value: str) -> None:
    statement = (
        "INSERT INTO meta (key, value) VALUES (?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value = excluded.value"
    )
    if conn.in_transaction:
        conn.execute(statement, (key, value))
        return
    with _transaction(conn):
        conn.execute(statement, (key, value))


def _load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Try to load the extension; ``False`` means the store is FTS-only.

    Loaded lazily and failure-tolerant on purpose: hooks open stores on the hot
    path and must work on a machine that has no ``sqlite-vec`` wheel, a broken
    one, or an extension that raises anything at all.

    The extension file is loaded directly rather than through
    ``sqlite_vec.load``: importing that package also imports ``numpy``, which
    costs ~55 ms and would dominate a PostToolUse hook's whole budget. An
    already-imported ``sqlite_vec`` module is honoured (tests and embedders
    substitute one), so the substitution contract is unchanged.
    """
    module = sys.modules.get("sqlite_vec")
    if module is not None:
        load: Any = getattr(module, "load", None)
        if not callable(load):
            return False
        return _try_load_extension(conn, lambda: load(conn))

    extension = _sqlite_vec_path()
    if extension is None:
        return False
    return _try_load_extension(conn, lambda: conn.load_extension(str(extension)))


def _try_load_extension(conn: sqlite3.Connection, load: Callable[[], None]) -> bool:
    try:
        conn.enable_load_extension(True)
        load()
    except Exception:  # noqa: BLE001 - any loader failure means FTS-only
        return False
    finally:
        with contextlib.suppress(Exception):
            conn.enable_load_extension(False)
    return True


def _sqlite_vec_path() -> Path | None:
    """The wheel's loadable extension file, or ``None`` when it is unavailable."""
    try:
        spec = importlib.util.find_spec("sqlite_vec")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    package = Path(next(iter(spec.submodule_search_locations)))
    candidates = [package / "vec0"]
    candidates.extend(package / f"vec0{suffix}" for suffix in EXTENSION_SUFFIXES)
    return next((candidate for candidate in candidates if candidate.is_file()), None)
