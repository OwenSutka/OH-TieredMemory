"""Tier entry points: resolve the file, take the write lock, hand out a connection.

One context manager per tier (``global_store``, ``project``, ``session``). Each
call resolves its path through :mod:`tiered_memory.config`, so project identity
(``sha1(realpath(working_dir))[:12]``) exists in exactly one place.

Every entry is a transaction: the schema's connection is opened (creating and
gating the store if needed) through :func:`tiered_memory.schema.open_store`, the
write lock is taken with ``BEGIN IMMEDIATE`` under a 5 s busy timeout, the block
runs, then the transaction commits on a clean exit, rolls back if the block
raises, and the connection is always closed. Holding the lock from ``__enter__``
rather than from the first write is what makes a losing writer wait instead of
discovering the conflict halfway through.

``fail_open=True`` is for hooks (plan R7). It turns *lock contention only* into a
warning plus :data:`HOOK_SKIPPED`, so the caller continues and the process exits
0. Everything else still raises: a missing ``TIERED_MEMORY_DATA``
(``ConfigError``), a corrupt file (``sqlite3.DatabaseError``), a store this
schema cannot serve, and a lost creation race are defects, and swallowing them
would turn a broken store into silent data loss.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from . import config, schema

__all__ = ["HOOK_SKIPPED", "HookSkipped", "global_store", "project", "session"]

_LOGGER = logging.getLogger(__name__)

#: Busy timeout for a fail-open caller's lock attempt. Long enough to absorb a
#: normal brief commit, short enough that sequential skips under sustained
#: contention stay cheap for a hook on the hot path.
_FAIL_FAST_MILLISECONDS = 50


class HookSkipped:
    """Yielded in place of a connection when ``fail_open`` skips lock contention."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "HOOK_SKIPPED"


#: The sentinel a fail-open entry yields instead of a connection. Compare with
#: ``is``: a hook checks ``conn is store.HOOK_SKIPPED`` and exits 0.
HOOK_SKIPPED = HookSkipped()


def global_store(
    *, fail_open: bool = False
) -> contextlib.AbstractContextManager[sqlite3.Connection | HookSkipped]:
    """Transaction over the global tier (``tiers/global.db``), read by every project."""
    return _open(config.global_db(), tier="global", scope_key="global", fail_open=fail_open)


def project(
    working_dir: str, *, fail_open: bool = False
) -> contextlib.AbstractContextManager[sqlite3.Connection | HookSkipped]:
    """Transaction over the project tier for ``working_dir``.

    The key is ``config.project_key``, which resolves symlinks, so two paths to
    one project share one store.
    """
    return _open(
        config.project_db(working_dir),
        tier="project",
        scope_key=config.project_key(working_dir),
        fail_open=fail_open,
    )


def session(
    session_id: str, *, fail_open: bool = False
) -> contextlib.AbstractContextManager[sqlite3.Connection | HookSkipped]:
    """Transaction over the session tier (``tiers/session_<id>.db``) for one session.

    ``session_id`` becomes part of a filename, so it must be a safe single path
    component (``config.session_db``): anything containing a separator, ``..``,
    or a character outside ``[A-Za-z0-9._-]`` raises ``ValueError`` before any
    directory is created.
    """
    return _open(
        config.session_db(session_id),
        tier="session",
        scope_key=session_id,
        fail_open=fail_open,
    )


@contextlib.contextmanager
def _open(
    path: Path, *, tier: str, scope_key: str, fail_open: bool
) -> Iterator[sqlite3.Connection | HookSkipped]:
    conn = _open_connection(path, tier=tier, scope_key=scope_key, fail_open=fail_open)
    if conn is None:
        yield HOOK_SKIPPED
        return
    try:
        if not _begin(conn, path=path, fail_open=fail_open):
            yield HOOK_SKIPPED
            return
        try:
            yield conn
        except BaseException:
            _rollback(conn)
            raise
        conn.execute("COMMIT")
    finally:
        if fail_open and _no_checkpoint_available():
            # A hook's connection is closed microseconds after the commit; the
            # full checkpoint that closing normally performs only duplicates the
            # fsync the commit already paid, and it is ~11 ms of the hot path.
            # The committed pages stay in the -wal, which the next open reads
            # and the next checkpoint reclaims.
            conn.setconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, True)
        conn.close()


def _open_connection(
    path: Path, *, tier: str, scope_key: str, fail_open: bool
) -> sqlite3.Connection | None:
    """Open the store, or ``None`` when ``fail_open`` skipped lock contention."""
    config.ensure_tiers_dir()
    try:
        conn = schema.open_store(path, tier=tier, scope_key=scope_key)
    except sqlite3.OperationalError as exc:
        if not (fail_open and schema.is_lock_contention(exc)):
            raise
        _LOGGER.warning("%s is locked; skipping this hook write: %s", path, exc)
        return None
    return conn


def _begin(conn: sqlite3.Connection, *, path: Path, fail_open: bool) -> bool:
    """Take the write lock now; ``False`` when ``fail_open`` skipped contention.

    A blocking caller waits out the documented 5 s busy timeout. A fail-open
    caller must not pay that per hook, so it probes with a short timeout and
    gives up instead of waiting: sustained contention then costs milliseconds
    per skip rather than seconds. The documented timeout is restored before a
    successful lock is handed to the block.
    """
    if not fail_open:
        conn.execute("BEGIN IMMEDIATE")
        return True
    conn.execute(f"PRAGMA busy_timeout = {_FAIL_FAST_MILLISECONDS}")
    try:
        conn.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError as exc:
        if not schema.is_lock_contention(exc):
            raise
        _LOGGER.warning("%s is locked; skipping this hook write: %s", path, exc)
        return False
    conn.execute(f"PRAGMA busy_timeout = {int(schema.BUSY_TIMEOUT_SECONDS * 1000)}")
    return True


def _no_checkpoint_available() -> bool:
    """Whether this interpreter can suppress the close-time checkpoint."""
    return hasattr(sqlite3.Connection, "setconfig")


def _rollback(conn: sqlite3.Connection) -> None:
    # Closing the connection would roll back anyway; this only ends the
    # transaction sooner. A failing rollback must not replace the body's error.
    with contextlib.suppress(sqlite3.Error):
        conn.execute("ROLLBACK")
