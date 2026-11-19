"""Coverage-gap tests for ``tiered_memory.schema`` (plan §4, §5.1).

Each test here kills a mutation that the rest of the suite cannot distinguish
from correct behaviour:

* §4 declares ``id INTEGER PRIMARY KEY AUTOINCREMENT``. Dropping the keyword
  leaves an id column that still passes every ``PRAGMA table_info`` assertion,
  but reuses ids once the highest row is deleted.
* ``idx_facts_live`` is the *partial* index ``... WHERE superseded_by IS NULL``.
  Inverting the predicate keeps the index partial and keeps its column, so only
  the SQL stored in ``sqlite_master`` proves which rows it covers.
* ``open_store`` documents that an ``index_state`` outside ``INDEX_STATES`` is
  refused with ``IncompleteSchemaError``; no other test writes one.
* §5.1's "any -> fts_only" transition also applies to a store that was created
  *with* the extension. Creation-time fallback does not cover it, so the reopen
  runs in a subprocess where ``sqlite-vec`` really cannot be imported (an
  in-process monkeypatch cannot fake the lazy ``importlib.import_module``
  faithfully, and it would leave a loaded extension on the connection).
* Tier isolation is enforced by the store's recorded identity: reopening a
  global store as ``project`` (or under another ``scope_key``) must raise
  ``StoreIdentityMismatch``, and a refusal must not mutate the file.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import tiered_memory.schema as schema
from tiered_memory.schema import (
    IncompleteSchemaError,
    StoreIdentityMismatch,
    open_store,
    read_meta,
)

#: Directory holding the package, so a child interpreter imports the same code
#: under test even when the project is not installed into ``sys.path``.
_SRC_DIR = Path(schema.__file__).resolve().parents[1]

#: Ceiling for a single child. A healthy run finishes in well under a second.
_CHILD_TIMEOUT = 60.0


@contextlib.contextmanager
def _closed(path: Path, *, tier: str = "global", scope_key: str | None = None) -> Iterator[Any]:
    """Open a store with the suite's keyword contract and close it afterwards."""
    conn = open_store(path, tier=tier, scope_key=scope_key or f"scope-{tier}")
    try:
        yield conn
    finally:
        conn.close()


def _insert_fact(conn: sqlite3.Connection, text: str) -> None:
    conn.execute(
        "INSERT INTO facts (content_hash, subject_key, text, kind, tier, scope_key,"
        " created_at, updated_at, last_seen_at)"
        " VALUES (?, ?, ?, 'fact', 'global', 'scope-global', 't', 't', 't')",
        (f"hash:{text}", f"subject:{text}", text),
    )
    conn.commit()


def _normalised_sql(conn: sqlite3.Connection, name: str) -> str:
    """``sqlite_master.sql`` for ``name``, lower-cased with whitespace collapsed."""
    row = conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (name,)).fetchone()
    assert row is not None, f"{name} must exist in sqlite_master"
    return " ".join(str(row[0]).lower().split())


def _sidecars(path: Path) -> list[str]:
    return sorted(entry.name for entry in path.parent.iterdir())


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET"
        " value = excluded.value",
        (key, value),
    )
    conn.commit()


# --------------------------------------------------------------------------- #
# DDL details the column/index pragmas cannot see (§4)
# --------------------------------------------------------------------------- #


def test_facts_id_is_autoincrement(tmp_path: Path) -> None:
    """§4: ``id INTEGER PRIMARY KEY AUTOINCREMENT`` -- ids are never reused.

    ``PRAGMA table_info`` reports the same "INTEGER, primary key" shape with or
    without the keyword, so the stored DDL is the only witness.
    """
    with _closed(tmp_path / "autoincrement.db") as conn:
        sql = _normalised_sql(conn, "facts")
        assert "autoincrement" in sql, f"facts.id must be AUTOINCREMENT, got: {sql}"
        assert "id integer primary key autoincrement" in sql

        first = conn.execute(
            "INSERT INTO facts (content_hash, subject_key, text, kind, tier, scope_key,"
            " created_at, updated_at, last_seen_at)"
            " VALUES ('h1', 's1', 'first', 'fact', 'global', 'scope-global', 't', 't', 't')"
        ).lastrowid
        conn.execute("DELETE FROM facts WHERE id = ?", (first,))
        conn.commit()
        second = conn.execute(
            "INSERT INTO facts (content_hash, subject_key, text, kind, tier, scope_key,"
            " created_at, updated_at, last_seen_at)"
            " VALUES ('h2', 's2', 'second', 'fact', 'global', 'scope-global', 't', 't', 't')"
        ).lastrowid
        assert second != first, "AUTOINCREMENT must not reuse the deleted id"


def test_live_facts_index_covers_live_rows_only(tmp_path: Path) -> None:
    """§4: ``idx_facts_live`` is ``... WHERE superseded_by IS NULL``.

    ``PRAGMA index_list`` only says "partial"; the predicate that makes the
    index serve the live-facts query lives in the stored SQL alone.
    """
    with _closed(tmp_path / "live_index.db") as conn:
        sql = _normalised_sql(conn, "idx_facts_live")
        assert "where superseded_by is null" in sql, (
            f"idx_facts_live must be partial on live rows, got: {sql}"
        )
        assert "is not null" not in sql


# --------------------------------------------------------------------------- #
# an unknown index_state is refused, and the refusal is inert
# --------------------------------------------------------------------------- #


def test_store_with_unknown_index_state_is_refused_without_mutating(tmp_path: Path) -> None:
    """``open_store`` documents ``IncompleteSchemaError`` for a bogus ``index_state``.

    The store is otherwise perfectly valid, so nothing but this guard can refuse
    it. A refusal is read-only: the file stays byte-identical and no sidecar
    appears.
    """
    path = tmp_path / "bogus_index_state.db"
    with _closed(path) as conn:
        _set_meta(conn, "index_state", "bogus")

    before = path.read_bytes()
    assert _sidecars(path) == [path.name], "the fixture must be closed and checkpointed"

    with pytest.raises(IncompleteSchemaError) as caught:
        open_store(path, tier="global", scope_key="scope-global")

    message = str(caught.value)
    assert "index_state" in message
    assert "bogus" in message, "the refusal must name the value it found"
    assert path.read_bytes() == before, "a refused open must not write a byte"
    assert _sidecars(path) == [path.name], "a refused open must not create sidecars"

    # A refusal is not a corruption: fixing meta is all the repair needed.
    raw = sqlite3.connect(str(path))
    try:
        raw.execute("UPDATE meta SET value = 'pending' WHERE key = 'index_state'")
        raw.commit()
    finally:
        raw.close()
    with _closed(path) as conn:
        assert read_meta(conn)["index_state"] == "pending"


# --------------------------------------------------------------------------- #
# §5.1: a pending store degrades when the extension is unavailable
# --------------------------------------------------------------------------- #

#: A meta-path finder that makes every ``sqlite_vec`` import fail. Used in a
#: child interpreter: the absence has to be real for the lazy import to see it.
_BLOCK_SQLITE_VEC = """
import sys

class _Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sqlite_vec" or fullname.startswith("sqlite_vec."):
            raise ImportError("simulated: sqlite-vec is not installed")
        return None

sys.meta_path.insert(0, _Blocker())
sys.modules.pop("sqlite_vec", None)
"""

_REOPEN_WITHOUT_VEC = (
    _BLOCK_SQLITE_VEC
    + """
import json
import sys

import tiered_memory.schema as schema

conn = schema.open_store(sys.argv[1], tier="global", scope_key="scope-global")
try:
    meta = schema.read_meta(conn)
    virtual = sorted(
        row[1]
        for row in conn.execute("PRAGMA table_list")
        if row[0] == "main" and row[2] == "virtual"
    )
finally:
    conn.close()

print(json.dumps({"meta": meta, "virtual": virtual}))
"""
)


def test_pending_store_degrades_to_fts_only_when_the_extension_disappears(
    tmp_path: Path,
) -> None:
    """§5.1: "any -> fts_only" includes a store created while ``sqlite-vec`` worked.

    Creation-time fallback (covered elsewhere) records ``fts_only`` up front;
    this exercises the open-time ``pending -> fts_only`` transition, which only
    fires when a vector-capable store is reopened without the extension.
    """
    path = tmp_path / "degrade.db"
    with _closed(path) as conn:
        assert read_meta(conn)["index_state"] == "pending", (
            "fixture needs sqlite-vec to create a pending store"
        )
        virtual = {
            row[1]
            for row in conn.execute("PRAGMA table_list")
            if row[0] == "main" and row[2] == "virtual"
        }
        assert "facts_vec" in virtual, "fixture must be a vector-capable store"

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_REOPEN_WITHOUT_VEC), str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=_CHILD_TIMEOUT,
        env={**os.environ, "PYTHONPATH": str(_SRC_DIR)},
    )
    assert result.returncode == 0, f"open without sqlite-vec raised:\n{result.stderr}"
    payload = json.loads(result.stdout.strip().splitlines()[-1])

    assert payload["meta"]["index_state"] == "fts_only", (
        "opening a pending store without sqlite-vec must record fts_only, "
        f"got {payload['meta']['index_state']!r}"
    )
    assert "facts_fts" in payload["virtual"], "the FTS half must still exist"

    # Read the file with a raw connection: the degrade must be committed, not
    # merely visible on the child's connection.
    raw = sqlite3.connect(str(path))
    try:
        row = raw.execute("SELECT value FROM meta WHERE key = 'index_state'").fetchone()
    finally:
        raw.close()
    assert row == ("fts_only",), "the degrade must be committed"


# --------------------------------------------------------------------------- #
# identity: the recorded tier/scope must match the caller
# --------------------------------------------------------------------------- #


def test_reopening_a_global_store_as_project_is_refused_without_mutating(
    tmp_path: Path,
) -> None:
    """Tier isolation is a property of the file, guarded by its recorded identity."""
    path = tmp_path / "identity.db"
    with _closed(path) as conn:
        _insert_fact(conn, "a global fact that must survive")

    before = path.read_bytes()
    assert _sidecars(path) == [path.name], "the fixture must be closed and checkpointed"

    with pytest.raises(StoreIdentityMismatch) as caught:
        open_store(path, tier="project", scope_key="scope-global")

    message = str(caught.value)
    assert "global" in message, "the refusal must name the recorded tier"
    assert "project" in message, "the refusal must name the requested tier"
    assert path.read_bytes() == before, "a refused open must not write a byte"
    assert _sidecars(path) == [path.name], "a refused open must not create sidecars"

    # The store still opens under its own identity, with the fact intact.
    with _closed(path) as conn:
        rows = conn.execute("SELECT text FROM facts").fetchall()
        assert rows == [("a global fact that must survive",)]


def test_reopening_with_a_different_scope_key_is_refused_without_mutating(
    tmp_path: Path,
) -> None:
    """Same tier, different ``scope_key``: still a different store."""
    path = tmp_path / "scope_identity.db"
    with _closed(path, tier="project", scope_key="scope-one"):
        pass

    before = path.read_bytes()
    with pytest.raises(StoreIdentityMismatch) as caught:
        open_store(path, tier="project", scope_key="scope-two")

    message = str(caught.value)
    assert "scope-one" in message, "the refusal must name the recorded scope"
    assert "scope-two" in message, "the refusal must name the requested scope"
    assert path.read_bytes() == before, "a refused open must not write a byte"
    assert _sidecars(path) == [path.name], "a refused open must not create sidecars"


# --------------------------------------------------------------------------- #
# WAL churn: the gate must never surface a peer's checkpoint as a failure
# --------------------------------------------------------------------------- #


def _crash_state(path: Path, *, megabytes: int) -> None:
    """Leave a committed ``-wal`` beside a large main file, with no ``-shm``.

    That is exactly what a writer that was killed leaves behind: the WAL holds
    the newest committed pages, but nothing is holding the shared memory that
    a live writer would keep beside it. It is the one shape the gate must copy
    rather than read in place.
    """
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA wal_autocheckpoint = 0")
        conn.execute("CREATE TABLE ballast(x BLOB)")
        conn.execute("INSERT INTO ballast VALUES (zeroblob(?))", (megabytes * 1_000_000,))
        conn.execute("INSERT INTO ballast VALUES (zeroblob(100))")
        # Snapshot the WAL while it is still live, then close (which checkpoints
        # and unlinks it) and put the snapshot back: a durable crash state.
        shutil.copyfile(path, str(path) + ".save")
        shutil.copyfile(str(path) + "-wal", str(path) + "-wal.save")
    finally:
        conn.close()
    shutil.copyfile(str(path) + ".save", path)
    shutil.copyfile(str(path) + "-wal.save", str(path) + "-wal")
    for leftover in (str(path) + ".save", str(path) + "-wal.save"):
        os.unlink(leftover)


def test_a_vanishing_wal_during_the_gate_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A peer deleting its ``-wal`` mid-gate must never fail the open.

    The gate discovers a crash-style WAL (no ``-shm``), starts copying the files
    aside, and a peer checkpoints and unlinks the WAL underneath it. That race
    used to surface as ``FileNotFoundError`` from ``shutil.copyfile``: an
    ``OSError`` that neither ``store`` nor ``fail_open`` classifies. The gate
    must retry and open from whatever state the store is in now.

    The race is injected at the filesystem so it is deterministic: the real
    ``copyfile`` runs and raises the real ``FileNotFoundError`` after the WAL is
    gone, and the retry then has to succeed.
    """
    path = tmp_path / "churn.db"
    with _closed(path):
        pass
    _crash_state(path, megabytes=24)
    wal = Path(str(path) + "-wal")
    assert not Path(str(path) + "-shm").exists(), "fixture must look like a crashed writer"

    real_copyfile = shutil.copyfile
    vanished = [0]

    def racing_copyfile(source: object, destination: object) -> None:
        if str(source).endswith("-wal") and vanished[0] == 0:
            vanished[0] += 1
            os.unlink(str(source))
        real_copyfile(source, destination)

    monkeypatch.setattr(schema.shutil, "copyfile", racing_copyfile)

    conn = open_store(path, tier="global", scope_key="scope-global")
    try:
        assert read_meta(conn)["schema_version"] == str(schema.SCHEMA_VERSION)
    finally:
        conn.close()

    assert vanished[0] == 1, "the injected race must have fired"
    assert not wal.exists(), "the fixture's WAL should be gone"


def test_sustained_wal_churn_never_raises(tmp_path: Path) -> None:
    """14 s of one writer that opens and closes against six readers: zero errors.

    Closing the last connection is what makes SQLite checkpoint and unlink the
    WAL, so this writer reproduces the churn a long-lived server sees without
    any artificial seam. The claim under test is the module's own: an open
    during WAL churn never raises.
    """
    path = tmp_path / "sustained.db"
    with _closed(path) as conn:
        conn.execute("CREATE TABLE ballast(x BLOB)")
        conn.execute("INSERT INTO ballast VALUES (zeroblob(16000000))")

    stop = threading.Event()
    errors: list[str] = []
    opens = [0]

    def writer() -> None:
        while not stop.is_set():
            peer = sqlite3.connect(str(path), isolation_level=None, timeout=5.0)
            try:
                peer.execute("INSERT INTO ballast VALUES (zeroblob(2000))")
            finally:
                peer.close()

    def reader() -> None:
        while not stop.is_set():
            try:
                open_store(path, tier="global", scope_key="scope-global").close()
                opens[0] += 1
            except BaseException as exc:  # noqa: BLE001 - collected and asserted
                errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=writer)]
    threads += [threading.Thread(target=reader) for _ in range(6)]
    for thread in threads:
        thread.start()
    time.sleep(14)
    stop.set()
    for thread in threads:
        thread.join(timeout=60)
    assert not any(thread.is_alive() for thread in threads), "a worker hung"

    assert opens[0] > 0, "the readers never opened the store"
    assert errors == [], f"{len(errors)} opens raised during WAL churn: {errors[:5]}"


# --------------------------------------------------------------------------- #
# the gate must not copy the whole database on the normal path
# --------------------------------------------------------------------------- #


def test_open_with_a_live_wal_does_not_copy_the_database(tmp_path: Path) -> None:
    """Opening a 70 MB store with a live WAL must stay under 5 ms.

    ``copyfile`` of the database costs ~0.25 ms/MB (17 ms at this size) and used
    to run on every open while a server held a WAL; the gate must instead read
    the store in place through the existing ``-shm``, which creates no file.
    The threshold is deliberately far above the in-place cost (~0.4 ms) and far
    below the copy cost, so a machine without a warm page cache still passes.
    """
    path = tmp_path / "big.db"
    with _closed(path) as conn:
        conn.execute("CREATE TABLE ballast(x BLOB)")
        conn.execute("INSERT INTO ballast VALUES (zeroblob(70000000))")
    assert path.stat().st_size >= 70_000_000, "fixture must be a ~70 MB store"

    peer = sqlite3.connect(str(path), isolation_level=None)
    try:
        peer.execute("INSERT INTO ballast VALUES (zeroblob(100))")
        assert path.with_name(path.name + "-wal").stat().st_size > 0, "fixture needs a live WAL"

        # Warm the page cache once; the claim is about what every later open does.
        open_store(path, tier="global", scope_key="scope-global").close()

        started = time.perf_counter()
        for _ in range(5):
            open_store(path, tier="global", scope_key="scope-global").close()
        elapsed_ms = (time.perf_counter() - started) * 1000 / 5
    finally:
        peer.close()

    assert elapsed_ms < 5.0, (
        f"an open with a live WAL took {elapsed_ms:.2f} ms on a 70 MB store; "
        "the gate must not copy the database"
    )


def test_gate_on_a_live_wal_creates_no_sidecar_and_writes_nothing(tmp_path: Path) -> None:
    """The cheap path must keep the refusal contract for a live WAL too.

    Reading through the existing ``-shm`` is what makes the fast path legal: if
    it created a ``-shm`` (or rewrote the header) the "a refused open writes
    nothing" promise would be broken for every store a server holds open.
    """
    path = tmp_path / "live_refusal.db"
    with _closed(path) as conn:
        conn.execute("CREATE TABLE ballast(x BLOB)")
    peer = sqlite3.connect(str(path), isolation_level=None)
    try:
        peer.execute("INSERT INTO ballast VALUES (zeroblob(100))")
        peer.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
        # The writer's own commit leaves the WAL unmerged; no checkpoint yet.
        assert path.with_name(path.name + "-wal").stat().st_size > 0

        before = path.read_bytes()
        sidecars = _sidecars(path)
        with pytest.raises(schema.SchemaVersionError):
            open_store(path, tier="project", scope_key="scope-global")
        assert path.read_bytes() == before, "a refused open must not write a byte"
        assert _sidecars(path) == sidecars, "a refused open must not create a sidecar"
    finally:
        peer.close()
