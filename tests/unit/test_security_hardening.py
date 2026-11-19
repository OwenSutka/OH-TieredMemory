"""Regression tests for the four storage defects and the lock classifier.

Each test reproduces the defect as it was reported and asserts the refusal (or
recovery) contract, using real files and real SQLite connections:

* a symlink at ``tiers/session_<id>.db`` (dangling or not) or at ``tiers/``
  itself must never be followed, and nothing may be written outside the root;
* a store truncated to zero bytes while its log is still live must be refused,
  not silently bootstrapped with a new identity;
* ``TIERED_MEMORY_DATA`` must be absolute and free of ``..``;
* a hot journal left by a killed writer must be replayed by the gate, so the
  next open succeeds and still sees the pre-crash data;
* lock contention is classified by ``sqlite_errorcode``, never by the message.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

import tiered_memory.config as config
import tiered_memory.schema as schema
import tiered_memory.store as store
from tiered_memory.config import ConfigError

_INSERT_FACT = (
    "INSERT INTO facts (content_hash, subject_key, text, kind, tier, scope_key,"
    " created_at, updated_at, last_seen_at)"
    " VALUES (?, ?, ?, 'fact', ?, ?, 't', 't', 't')"
)


@contextlib.contextmanager
def _open(
    path: Path, *, tier: str = "global", scope_key: str = "scope-global"
) -> Iterator[sqlite3.Connection]:
    conn = schema.open_store(path, tier=tier, scope_key=scope_key)
    try:
        yield conn
    finally:
        conn.close()


def _insert(conn: sqlite3.Connection, text: str, *, tier: str, scope_key: str) -> None:
    conn.execute(_INSERT_FACT, (f"hash:{text}", f"subject:{text}", text, tier, scope_key))


def _files(directory: Path) -> list[str]:
    return sorted(entry.name for entry in directory.iterdir())


# --------------------------------------------------------------------------- #
# CRITICAL: a symlink at the leaf or at tiers/ must not be followed
# --------------------------------------------------------------------------- #


def _layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A configured data root whose ``tiers/`` exists, plus an outside directory."""
    root = tmp_path / "plugin-data"
    root.mkdir()
    (root / config.TIERS_DIR_NAME).mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv(config.DATA_ENV, str(root))
    return root, outside


def test_a_dangling_symlink_leaf_is_refused_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dangling symlink would make SQLite create its target outside ``tiers/``.

    Before the fix, ``os.open(..., O_CREAT)`` followed the link and created both
    ``pwned.db`` in the outside directory and the store inside it.
    """
    _, outside = _layout(tmp_path, monkeypatch)
    target = config.session_db("s1")
    target.symlink_to(outside / "pwned.db")

    with pytest.raises(schema.StorePathError), store.session("s1"):
        pytest.fail("a dangling symlink leaf must not be opened")

    assert _files(outside) == [], "nothing may be written through the link"
    assert target.is_symlink(), "the planted link must be left untouched"


def test_a_symlink_leaf_to_an_existing_file_is_refused_and_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symlink to an existing empty file must not be rewritten with a store."""
    _, outside = _layout(tmp_path, monkeypatch)
    victim = outside / "victim.db"
    victim.write_bytes(b"")
    target = config.session_db("s1")
    target.symlink_to(victim)

    with pytest.raises(schema.StorePathError), store.session("s1"):
        pytest.fail("a symlink leaf must not be opened")

    assert victim.read_bytes() == b"", "the link target must not be written"
    assert _files(outside) == ["victim.db"], "no sidecar may appear beside the target"


def test_a_symlinked_tiers_directory_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symlinked ``tiers/`` would relocate every tier's store."""
    root = tmp_path / "plugin-data"
    root.mkdir()
    real = tmp_path / "outside-tiers"
    real.mkdir()
    (root / config.TIERS_DIR_NAME).symlink_to(real, target_is_directory=True)
    monkeypatch.setenv(config.DATA_ENV, str(root))

    with pytest.raises(ConfigError), store.global_store():
        pytest.fail("a symlinked tiers directory must not be written through")

    assert _files(real) == [], "nothing may be created in the link target"


def test_a_store_addressed_through_a_symlinked_directory_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resolved store must stay inside the resolved tiers directory."""
    root, outside = _layout(tmp_path, monkeypatch)
    real = outside / "elsewhere"
    real.mkdir()
    tiers = root / config.TIERS_DIR_NAME
    tiers.rmdir()
    tiers.symlink_to(real, target_is_directory=True)

    with pytest.raises(ConfigError), store.session("s1"):
        pytest.fail("the store must not resolve outside the data root")

    assert _files(real) == []


# --------------------------------------------------------------------------- #
# HIGH: a pre-existing store truncated to zero must be refused, not rebuilt
# --------------------------------------------------------------------------- #


def test_a_truncated_store_with_a_live_log_is_refused_not_rebootstrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``st_size == 0`` must not skip the gate and bootstrap a new identity.

    The store is truncated while a peer keeps its ``-wal`` alive, which is what
    a crash or an external ``> store.db`` does to a store a server holds open.
    Before the fix the gate was skipped entirely, ``open_store`` saw an empty
    database, and it wrote a fresh store recording the *wrong* scope over it.
    """
    monkeypatch.setenv(config.DATA_ENV, str(tmp_path / "plugin-data"))
    path = config.session_db("s1")
    config.ensure_tiers_dir()

    conn = schema.open_store(path, tier="session", scope_key="s1")
    try:
        _insert(conn, "kept", tier="session", scope_key="s1")
        os.truncate(path, 0)
        assert path.stat().st_size == 0
        wal = Path(str(path) + "-wal")
        assert wal.stat().st_size > 0, "the fixture needs a live WAL beside the truncated file"
        wal_before = wal.read_bytes()

        with pytest.raises(schema.SchemaVersionError):
            schema.open_store(path, tier="session", scope_key="s1")
        with pytest.raises(schema.SchemaVersionError):
            schema.open_store(path, tier="session", scope_key="other")

        assert path.stat().st_size == 0, "a refused open must not bootstrap a store"
        assert wal.read_bytes() == wal_before, "a refused open must not rewrite the log"
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# MEDIUM: TIERED_MEMORY_DATA must be absolute and must not contain '..'
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", ["relative-data", ".", "data/sub", ".."])
def test_a_relative_data_directory_is_refused(
    value: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A relative value would anchor the store to the caller's project directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(config.DATA_ENV, value)

    with pytest.raises(ConfigError), store.global_store():
        pytest.fail("a relative data directory must not be usable")


@pytest.mark.parametrize("value", ["/tmp/../etc/tiered", "/tmp/data/../escape"])
def test_a_data_directory_containing_dotdot_is_refused(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``..`` would let a configuration value climb out of the chosen root."""
    monkeypatch.setenv(config.DATA_ENV, value)

    with pytest.raises(ConfigError) as caught:
        config.data_dir()

    assert ".." in str(caught.value)
    with pytest.raises(ConfigError), store.global_store():
        pytest.fail("a data directory containing '..' must not be usable")


# --------------------------------------------------------------------------- #
# MEDIUM: a hot journal must be replayed by the gate, not mistaken for no schema
# --------------------------------------------------------------------------- #

#: A writer that dies with ``SIGKILL`` mid-transaction, leaving a hot journal.
_CRASHING_WRITER = """
import os
import signal
import sqlite3

conn = sqlite3.connect(os.environ["STORE_PATH"], isolation_level=None)
conn.execute("PRAGMA journal_mode = DELETE")
conn.execute("PRAGMA cache_size = 1")
conn.execute("BEGIN IMMEDIATE")
conn.execute(
    "INSERT INTO facts (content_hash, subject_key, text, kind, tier, scope_key,"
    " created_at, updated_at, last_seen_at)"
    " VALUES ('crash', 'crash', 'uncommitted', 'fact', 'global', 'global', 't', 't', 't')"
)
os.kill(os.getpid(), signal.SIGKILL)
"""


def test_a_hot_journal_is_replayed_before_the_gate_inspects_the_store(tmp_path: Path) -> None:
    """A killed writer's journal must not make the store permanently unopenable.

    The gate used to copy the files and inspect the copy with ``mode=ro``;
    SQLite then refused with "attempt to write a readonly database" (or, with
    ``immutable=1``, reported no ``schema_version``), which ``fail_open`` does
    not absorb. The copy is now opened read-write so recovery runs there, and
    the open succeeds with the pre-crash facts intact.
    """
    path = tmp_path / "hot.db"
    with _open(path) as conn:
        _insert(conn, "pre-crash", tier="global", scope_key="scope-global")

    # Leave a rollback journal behind: WAL is the module's default, so switch
    # back and checkpoint until no WAL sidecar remains.
    settle = sqlite3.connect(str(path), isolation_level=None)
    try:
        settle.execute("PRAGMA journal_mode = DELETE")
        settle.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        settle.close()
    assert not Path(str(path) + "-wal").exists(), "the fixture must have no WAL"

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_CRASHING_WRITER)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env={**os.environ, "STORE_PATH": str(path)},
    )
    assert result.returncode != 0, "the writer must have been killed"
    journal = Path(str(path) + "-journal")
    assert journal.stat().st_size > 0, "the fixture must leave a hot journal"

    conn = schema.open_store(path, tier="global", scope_key="scope-global")
    try:
        texts = sorted(str(row[0]) for row in conn.execute("SELECT text FROM facts"))
    finally:
        conn.close()

    assert texts == ["pre-crash"], "recovery must roll the uncommitted insert back"
    assert not journal.exists(), "the recovered copy's journal must not leak back"


# --------------------------------------------------------------------------- #
# contention is classified by sqlite_errorcode, not by the message
# --------------------------------------------------------------------------- #


def test_a_real_busy_error_is_classified_as_contention(tmp_path: Path) -> None:
    path = tmp_path / "busy.db"
    holder = sqlite3.connect(str(path), isolation_level=None, timeout=0.1)
    try:
        holder.execute("CREATE TABLE t(x)")
        holder.execute("BEGIN IMMEDIATE")

        rival = sqlite3.connect(str(path), isolation_level=None, timeout=0.1)
        try:
            with pytest.raises(sqlite3.OperationalError) as caught:
                rival.execute("BEGIN IMMEDIATE")
        finally:
            rival.close()
    finally:
        holder.close()

    assert schema.is_lock_contention(caught.value) is True
    assert caught.value.sqlite_errorname == "SQLITE_BUSY"


def test_a_message_that_mentions_locked_or_busy_is_not_contention() -> None:
    """A path containing "locked"/"busy" must not turn a real defect into a skip.

    The classifier used to substring-match the message, so an error quoting a
    store under ``/data/locked/busy/`` was silently swallowed by ``fail_open``.
    """
    misleading = sqlite3.OperationalError(
        "cannot open /data/locked/busy/session_s1.db: file is not a database"
    )

    assert schema.is_lock_contention(misleading) is False


def test_a_non_contention_error_without_a_code_is_not_contention() -> None:
    assert schema.is_lock_contention(sqlite3.OperationalError("database is locked")) is False
