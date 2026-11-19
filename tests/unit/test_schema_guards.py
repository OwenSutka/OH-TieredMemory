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
import sqlite3
import subprocess
import sys
import textwrap
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
