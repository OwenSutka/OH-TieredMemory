"""Acceptance tests for ``tiered_memory.schema`` (plan TM-02, plan §4 + §5.1).

Every assertion here is derived from ``plans/01-OH-TieredMemory.md``:

* §4 "Identical DDL in all three stores" -- the exact tables, columns, CHECK
  constraints, indexes, FTS5 external-content table + three triggers, and
  ``vec0(embedding float[384])``.
* §4 "Dimension guard" -- ``meta.embedding_dim`` compared on every open;
  mismatch raises ``EmbeddingDimensionMismatch`` carrying expected/actual and
  the instruction ``run memory_reindex --force`` or delete the store.
* Step 2 acceptance -- a fresh DB creates 2 tables + 2 virtual tables + 3
  triggers; a DB with ``schema_version = 0`` is refused with a message naming
  the fix; ``embedding_dim = 768`` raises ``EmbeddingDimensionMismatch``.
* §5.1 -- ``index_state`` is one of ``warming|pending|ready|fts_only`` and
  "``sqlite-vec`` fails to load" must record ``fts_only`` and never raise.

The public API asserted here is the one this suite fixes (the module does not
exist yet, so the names are chosen by the tests):

``SCHEMA_VERSION``, ``EMBEDDING_DIMENSION``, ``DEFAULT_EMBEDDING_MODEL``,
``VALID_TIERS``, ``INDEX_STATES``, ``META_KEYS``, ``SchemaVersionError``,
``EmbeddingDimensionMismatch``, ``open_store(path, *, tier, scope_key,
embedding_dim=...)`` returning a ``sqlite3.Connection``, and
``read_meta(conn) -> dict[str, str]``.

Real files in ``tmp_path`` throughout: the isolation this plugin promises is a
property of the SQLite file, so a fake connection would test nothing.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import struct
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

try:  # red phase: report the missing module per test instead of at collection
    from tiered_memory.schema import (
        DEFAULT_EMBEDDING_MODEL,
        EMBEDDING_DIMENSION,
        INDEX_STATES,
        META_KEYS,
        SCHEMA_VERSION,
        VALID_TIERS,
        EmbeddingDimensionMismatch,
        SchemaVersionError,
        open_store,
        read_meta,
    )
except Exception as exc:  # noqa: BLE001 - any import failure is the red signal
    _IMPORT_ERROR: BaseException | None = exc
else:
    _IMPORT_ERROR = None


@pytest.fixture(autouse=True)
def _schema_must_import() -> None:
    """Fail every test for the same honest reason while the module is absent."""
    if _IMPORT_ERROR is not None:
        pytest.fail(f"tiered_memory.schema is not importable: {_IMPORT_ERROR!r}")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

#: §4 column order, verbatim.
FACTS_COLUMNS = [
    "id",
    "content_hash",
    "subject_key",
    "text",
    "kind",
    "tier",
    "scope_key",
    "source_tool",
    "source_session",
    "confidence",
    "seen_count",
    "distinct_sessions",
    "created_at",
    "updated_at",
    "last_seen_at",
    "superseded_by",
    "deleted_at",
    "hits",
    "last_hit_at",
]

FACTS_TYPES = {
    "id": "INTEGER",
    "content_hash": "TEXT",
    "subject_key": "TEXT",
    "text": "TEXT",
    "kind": "TEXT",
    "tier": "TEXT",
    "scope_key": "TEXT",
    "source_tool": "TEXT",
    "source_session": "TEXT",
    "confidence": "REAL",
    "seen_count": "INTEGER",
    "distinct_sessions": "INTEGER",
    "created_at": "TEXT",
    "updated_at": "TEXT",
    "last_seen_at": "TEXT",
    "superseded_by": "INTEGER",
    "deleted_at": "TEXT",
    "hits": "INTEGER",
    "last_hit_at": "TEXT",
}

FACTS_NOT_NULL = {
    "content_hash",
    "subject_key",
    "text",
    "kind",
    "tier",
    "scope_key",
    "confidence",
    "seen_count",
    "distinct_sessions",
    "created_at",
    "updated_at",
    "last_seen_at",
    "hits",
}

#: The key vocabulary listed in the §4 ``meta`` comment.
PLAN_META_KEYS = {
    "schema_version",
    "tier",
    "scope_key",
    "embedding_model",
    "embedding_dim",
    "index_state",
    "budget_tokens",
    "created_at",
}

#: Keys a freshly created store must already carry for any later code to work.
FRESH_META_KEYS = {
    "schema_version",
    "tier",
    "scope_key",
    "embedding_model",
    "embedding_dim",
    "index_state",
    "created_at",
}

TRIGGER_NAMES = {"facts_ai", "facts_ad", "facts_au"}
INDEX_NAMES = {"idx_facts_live", "idx_facts_subject", "idx_facts_kind"}


def _open(path: Path, tier: str = "global", **kwargs: Any) -> sqlite3.Connection:
    """Open a store with the keyword contract this suite fixes."""
    conn: sqlite3.Connection = open_store(path, tier=tier, scope_key="scope-" + tier, **kwargs)
    return conn


@contextlib.contextmanager
def _closed(path: Path, tier: str = "global", **kwargs: Any) -> Any:
    """Open a store and close it at the end of the block."""
    conn = _open(path, tier=tier, **kwargs)
    try:
        yield conn
    finally:
        conn.close()


def _object_names(conn: sqlite3.Connection, kind: str) -> set[str]:
    """Names of ``type=kind`` objects in ``sqlite_master``."""
    rows = conn.execute("SELECT type, name FROM sqlite_master").fetchall()
    return {row[1] for row in rows if row[0] == kind}


def _table_list(conn: sqlite3.Connection, kind: str) -> set[str]:
    """Names from ``PRAGMA table_list`` with the given kind (main schema only).

    ``table_list`` is used rather than ``sqlite_master.type`` because it
    distinguishes ``table`` / ``virtual`` / ``shadow``; FTS5 and vec0 each
    create internal tables that must not be counted as user tables. The filter
    below drops those internals by name: sqlite-vec's ``facts_vec_*`` chunk
    tables are reported as ordinary ``table`` rows, not ``shadow``, so a kind
    filter alone cannot exclude them. An unexpected *unrelated* table still
    fails the assertion, which is the regression this must catch.
    """
    internal_prefixes = ("sqlite_", "facts_fts_", "facts_vec_")
    return {
        row[1]
        for row in conn.execute("PRAGMA table_list").fetchall()
        if row[0] == "main" and row[2] == kind and not row[1].startswith(internal_prefixes)
    }


def _columns(conn: sqlite3.Connection, table: str) -> list[tuple[Any, ...]]:
    return [tuple(row) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _schema_fingerprint(conn: sqlite3.Connection) -> tuple[Any, ...]:
    """Everything the "one schema for three tiers" rule must keep identical."""
    return (
        sorted(_table_list(conn, "table")),
        sorted(_table_list(conn, "virtual")),
        sorted(_object_names(conn, "trigger")),
        sorted(name for name in _object_names(conn, "index") if name.startswith("idx_")),
        [(row[1], row[2], row[3], row[4], row[5]) for row in _columns(conn, "facts")],
        [(row[1], row[2], row[3], row[4], row[5]) for row in _columns(conn, "meta")],
    )


def _insert_fact(
    conn: sqlite3.Connection,
    text: str = "a fact",
    *,
    kind: str = "fact",
    tier: str = "global",
    content_hash: str | None = None,
    superseded_by: int | None = None,
) -> int:
    """Insert a fact directly with SQL: TM-04 owns ``facts.py``, not this test."""
    cursor = conn.execute(
        "INSERT INTO facts (content_hash, subject_key, text, kind, tier, scope_key,"
        " superseded_by, created_at, updated_at, last_seen_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            content_hash if content_hash is not None else f"hash:{text}",
            f"subject:{text}",
            text,
            kind,
            tier,
            "scope",
            superseded_by,
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
        ),
    )
    conn.commit()
    assert cursor.lastrowid is not None
    return int(cursor.lastrowid)


def _insert_vector(conn: sqlite3.Connection, rowid: int, dims: int) -> None:
    payload = struct.pack(f"{dims}f", *([0.0] * dims))
    conn.execute(
        "INSERT INTO facts_vec(rowid, embedding) VALUES (?, ?)",
        (rowid, sqlite3.Binary(payload)),
    )
    conn.commit()


def _run_python(code: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a snippet in a real interpreter: import-time behaviour needs a fresh process."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


# --------------------------------------------------------------------------- #
# public constants
# --------------------------------------------------------------------------- #


def test_public_constants_match_the_plan() -> None:
    """The module's vocabulary is the plan's vocabulary."""
    assert SCHEMA_VERSION == 1, "plan §4 export header pins schema_version to 1"
    assert EMBEDDING_DIMENSION == 384, "all-MiniLM-L6-v2 is 384-dim"
    assert DEFAULT_EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    assert set(VALID_TIERS) == {"global", "project", "session"}
    assert set(INDEX_STATES) == {"warming", "pending", "ready", "fts_only"}
    assert set(META_KEYS) == PLAN_META_KEYS


# --------------------------------------------------------------------------- #
# the exact DDL (§4)
# --------------------------------------------------------------------------- #


def test_fresh_db_creates_two_tables_two_virtual_tables_three_triggers(tmp_path: Path) -> None:
    """Step 2 acceptance, verbatim: 2 tables + 2 virtual tables + 3 triggers."""
    with _closed(tmp_path / "fresh.db") as conn:
        assert _table_list(conn, "table") == {"meta", "facts"}
        assert _table_list(conn, "virtual") == {"facts_fts", "facts_vec"}
        assert _object_names(conn, "trigger") == TRIGGER_NAMES
        assert {name for name in _object_names(conn, "index")} >= INDEX_NAMES


def test_facts_table_columns_match_the_spec_ddl(tmp_path: Path) -> None:
    with _closed(tmp_path / "columns.db") as conn:
        info = _columns(conn, "facts")
        assert [row[1] for row in info] == FACTS_COLUMNS
        assert {row[1]: row[2].upper() for row in info} == FACTS_TYPES
        assert {row[1] for row in info if row[3]} == FACTS_NOT_NULL
        assert [row[1] for row in info if row[5]] == ["id"], "id is the only primary key"


def test_facts_column_defaults_match_the_spec_ddl(tmp_path: Path) -> None:
    with _closed(tmp_path / "defaults.db") as conn:
        defaults = {row[1]: row[4] for row in _columns(conn, "facts")}
        assert defaults["confidence"] is not None and float(defaults["confidence"]) == 1.0
        assert defaults["seen_count"] is not None and int(defaults["seen_count"]) == 1
        assert defaults["distinct_sessions"] is not None and int(defaults["distinct_sessions"]) == 1
        assert defaults["hits"] is not None and int(defaults["hits"]) == 0
        for nullable in ("source_tool", "source_session", "superseded_by", "deleted_at"):
            assert defaults[nullable] is None, f"{nullable} must have no DEFAULT"

        id_ = _insert_fact(conn, "defaults are applied")
        row = conn.execute(
            "SELECT confidence, seen_count, distinct_sessions, hits FROM facts WHERE id = ?",
            (id_,),
        ).fetchone()
        assert row is not None
        assert float(row[0]) == 1.0
        assert int(row[1]) == 1
        assert int(row[2]) == 1
        assert int(row[3]) == 0


def test_meta_table_columns_match_the_spec_ddl(tmp_path: Path) -> None:
    with _closed(tmp_path / "meta_columns.db") as conn:
        info = _columns(conn, "meta")
        assert [row[1] for row in info] == ["key", "value"]
        assert [row[1] for row in info if row[5]] == ["key"], "key is the primary key"
        assert [row[1] for row in info if row[3]] == ["value"], "value is NOT NULL"


def test_indexes_match_the_spec_ddl(tmp_path: Path) -> None:
    with _closed(tmp_path / "indexes.db") as conn:
        listing = {
            row[1]: {"unique": row[2], "partial": row[4]}
            for row in conn.execute("PRAGMA index_list(facts)").fetchall()
        }
        assert set(listing) >= INDEX_NAMES
        assert listing["idx_facts_live"]["partial"] == 1, "idx_facts_live is a partial index"
        columns = {
            name: [row[2] for row in conn.execute(f"PRAGMA index_info({name})").fetchall()]
            for name in INDEX_NAMES
        }
        assert columns["idx_facts_live"] == ["superseded_by"]
        assert columns["idx_facts_subject"] == ["kind", "subject_key"]
        assert columns["idx_facts_kind"] == ["kind"]
        assert listing["idx_facts_live"]["unique"] == 0


def test_pragmas_wal_normal_and_foreign_keys_are_applied(tmp_path: Path) -> None:
    with _closed(tmp_path / "pragmas.db") as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert int(conn.execute("PRAGMA synchronous").fetchone()[0]) == 1, "1 == NORMAL"
        assert int(conn.execute("PRAGMA foreign_keys").fetchone()[0]) == 1


def test_foreign_key_from_superseded_by_to_facts_is_declared_and_enforced(tmp_path: Path) -> None:
    """R5 keeps history: a supersede chain must point at a row that exists."""
    with _closed(tmp_path / "fk.db") as conn:
        fk_rows = [tuple(row) for row in conn.execute("PRAGMA foreign_key_list(facts)").fetchall()]
        assert any(
            row[2] == "facts" and row[3] == "superseded_by" and row[4] == "id" for row in fk_rows
        ), f"superseded_by must reference facts(id), got {fk_rows}"

        with pytest.raises(sqlite3.IntegrityError):
            _insert_fact(conn, "dangling supersede", superseded_by=999_999)

        original = _insert_fact(conn, "the original claim", content_hash="hash:original")
        replacement = _insert_fact(conn, "the replacement claim", superseded_by=original)
        row = conn.execute(
            "SELECT superseded_by FROM facts WHERE id = ?", (replacement,)
        ).fetchone()
        assert row is not None and int(row[0]) == original


def test_kind_check_constraint_rejects_unknown_kind(tmp_path: Path) -> None:
    """The six kinds in §4 are the only kinds: the CHECK lives in the schema."""
    with _closed(tmp_path / "kind_check.db") as conn:
        for kind in ("preference", "decision", "architecture", "fact", "todo", "gotcha"):
            _insert_fact(conn, f"valid {kind}", kind=kind)
        with pytest.raises(sqlite3.IntegrityError):
            _insert_fact(conn, "invalid kind", kind="gossip")


def test_tier_check_constraint_rejects_unknown_tier(tmp_path: Path) -> None:
    with _closed(tmp_path / "tier_check.db") as conn, pytest.raises(sqlite3.IntegrityError):
        _insert_fact(conn, "invalid tier", tier="galactic")


def test_content_hash_is_unique(tmp_path: Path) -> None:
    """Dedupe (TM-04) relies on the UNIQUE constraint existing in the DDL."""
    with _closed(tmp_path / "unique.db") as conn:
        _insert_fact(conn, "first copy", content_hash="hash:same")
        with pytest.raises(sqlite3.IntegrityError):
            _insert_fact(conn, "second copy", content_hash="hash:same")


# --------------------------------------------------------------------------- #
# FTS5 external-content table and its three triggers
# --------------------------------------------------------------------------- #


def test_facts_fts_uses_external_content_and_porter_tokenizer(tmp_path: Path) -> None:
    with _closed(tmp_path / "fts_ddl.db") as conn:
        sql = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'facts_fts'").fetchone()[0]
        normalised = " ".join(sql.lower().split())
        assert "content='facts'" in normalised
        assert "content_rowid='id'" in normalised
        assert "tokenize='porter unicode61'" in normalised


def test_insert_trigger_makes_a_fact_searchable_with_porter_stemming(tmp_path: Path) -> None:
    """``facts_ai`` fires on INSERT; porter stemming is what makes ``run`` match ``running``."""
    with _closed(tmp_path / "fts_insert.db") as conn:
        _insert_fact(conn, "the runner was running quickly")
        hits = conn.execute("SELECT rowid FROM facts_fts WHERE facts_fts MATCH 'run'").fetchall()
        assert len(hits) == 1


def test_update_trigger_resyncs_the_fts_row(tmp_path: Path) -> None:
    """``facts_au`` is ``AFTER UPDATE OF text``: stale text must not stay searchable."""
    with _closed(tmp_path / "fts_update.db") as conn:
        fact_id = _insert_fact(conn, "the runner was running")
        conn.execute("UPDATE facts SET text = ? WHERE id = ?", ("swimming in the sea", fact_id))
        conn.commit()

        old = conn.execute("SELECT rowid FROM facts_fts WHERE facts_fts MATCH 'run'").fetchall()
        new = conn.execute("SELECT rowid FROM facts_fts WHERE facts_fts MATCH 'swim'").fetchall()
        assert old == [], "the replaced text must no longer match"
        assert [row[0] for row in new] == [fact_id]


def test_delete_trigger_removes_the_fts_row(tmp_path: Path) -> None:
    with _closed(tmp_path / "fts_delete.db") as conn:
        fact_id = _insert_fact(conn, "the runner was running")
        conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        conn.commit()
        hits = conn.execute("SELECT rowid FROM facts_fts WHERE facts_fts MATCH 'run'").fetchall()
        assert hits == []


def test_fts_index_passes_integrity_check_after_crud(tmp_path: Path) -> None:
    """The external-content index must agree with ``facts`` after insert/update/delete.

    ``integrity-check`` with ``rank = 1`` is the fts5 mode that also verifies the
    *external content* rows agree with the index; the plain form does not, so a
    missing or wrong trigger would pass unnoticed. This is the assertion that
    fails if any of the three triggers is dropped from the DDL.
    """
    with _closed(tmp_path / "fts_integrity.db") as conn:
        first = _insert_fact(conn, "the runner was running")
        _insert_fact(conn, "a second fact about hindsight")
        conn.execute("UPDATE facts SET text = ? WHERE id = ?", ("swimming", first))
        _insert_fact(conn, "a third fact", content_hash="hash:third")
        conn.execute("DELETE FROM facts WHERE content_hash = ?", ("hash:third",))
        conn.commit()

        conn.execute("INSERT INTO facts_fts(facts_fts, rank) VALUES ('integrity-check', 1)")

        live = conn.execute("SELECT count(*) FROM facts").fetchone()[0]
        indexed = conn.execute("SELECT count(*) FROM facts_fts").fetchone()[0]
        assert int(indexed) == int(live) == 2


# --------------------------------------------------------------------------- #
# vec0 table
# --------------------------------------------------------------------------- #


def test_facts_vec_is_declared_at_the_pinned_dimension(tmp_path: Path) -> None:
    with _closed(tmp_path / "vec_ddl.db") as conn:
        row = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'facts_vec'").fetchone()
        assert row is not None, "facts_vec must exist when sqlite-vec loads"
        normalised = " ".join(str(row[0]).lower().split())
        assert "using vec0" in normalised
        assert f"float[{EMBEDDING_DIMENSION}]" in normalised.replace(" ", "")


def test_facts_vec_accepts_384_dimensions_and_rejects_768(tmp_path: Path) -> None:
    with _closed(tmp_path / "vec_insert.db") as conn:
        fact_id = _insert_fact(conn, "a fact to embed")
        _insert_vector(conn, fact_id, EMBEDDING_DIMENSION)
        stored = conn.execute("SELECT count(*) FROM facts_vec").fetchone()[0]
        assert int(stored) == 1

        with pytest.raises(sqlite3.OperationalError):
            _insert_vector(conn, fact_id + 1, 768)


# --------------------------------------------------------------------------- #
# meta contents (§4 key list, three tiers)
# --------------------------------------------------------------------------- #


def test_fresh_store_records_schema_tier_scope_and_embedding_facts(tmp_path: Path) -> None:
    with _closed(tmp_path / "meta.db", tier="project") as conn:
        meta = read_meta(conn)
        assert set(meta) >= FRESH_META_KEYS, f"missing meta keys: {FRESH_META_KEYS - set(meta)}"
        assert int(meta["schema_version"]) == SCHEMA_VERSION
        assert meta["tier"] == "project"
        assert meta["scope_key"] == "scope-project"
        assert meta["embedding_model"] == DEFAULT_EMBEDDING_MODEL
        assert int(meta["embedding_dim"]) == EMBEDDING_DIMENSION
        assert meta["index_state"] in INDEX_STATES
        assert meta["created_at"].strip(), "created_at must be recorded"


def test_read_meta_returns_every_row_as_a_string(tmp_path: Path) -> None:
    with _closed(tmp_path / "read_meta.db") as conn:
        conn.execute("INSERT INTO meta (key, value) VALUES ('extra_probe', 'probe')")
        conn.commit()
        meta = read_meta(conn)

        assert isinstance(meta, dict)
        assert all(isinstance(key, str) for key in meta)
        assert all(isinstance(value, str) for value in meta.values())
        rows = {row[0]: row[1] for row in conn.execute("SELECT key, value FROM meta").fetchall()}
        assert meta == rows
        assert meta["extra_probe"] == "probe"


def test_all_three_tiers_get_the_identical_schema(tmp_path: Path) -> None:
    """§4: "Identical DDL in all three stores"; the tier is only a meta value."""
    fingerprints: dict[str, tuple[Any, ...]] = {}
    for tier in ("global", "project", "session"):
        path = tmp_path / f"{tier}.db"
        with _closed(path, tier=tier) as conn:
            assert read_meta(conn)["tier"] == tier, f"{tier} store must record its tier"
            fingerprints[tier] = _schema_fingerprint(conn)

    assert fingerprints["global"] == fingerprints["project"] == fingerprints["session"]


def test_two_tiers_are_two_files_and_never_share_facts(tmp_path: Path) -> None:
    """Tier isolation is a property of the file (plan §1.2): no row leaks across."""
    global_path = tmp_path / "global-only.db"
    project_path = tmp_path / "project-only.db"

    with _closed(global_path, tier="global") as conn:
        _insert_fact(conn, "a global preference", kind="preference", tier="global")
        assert read_meta(conn)["scope_key"] == "scope-global"

    with _closed(project_path, tier="project") as conn:
        assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
        global_meta = read_meta(conn)
        assert global_meta["tier"] == "project"
        assert global_meta["scope_key"] == "scope-project"

    assert global_path.is_file() and project_path.is_file()


# --------------------------------------------------------------------------- #
# idempotency of a second open
# --------------------------------------------------------------------------- #


def test_second_open_is_idempotent_and_preserves_data(tmp_path: Path) -> None:
    path = tmp_path / "reopen.db"
    with _closed(path) as conn:
        before = read_meta(conn)
        fact_id = _insert_fact(conn, "a fact that must survive")
        before_objects = conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0]

    with _closed(path) as conn:
        after = read_meta(conn)
        assert after == before, "a reopen must not rewrite schema_version or created_at"
        assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 1
        assert conn.execute("SELECT text FROM facts WHERE id = ?", (fact_id,)).fetchone()[0] == (
            "a fact that must survive"
        )
        assert conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] == before_objects
        assert _table_list(conn, "table") == {"meta", "facts"}
        assert _table_list(conn, "virtual") == {"facts_fts", "facts_vec"}
        assert _object_names(conn, "trigger") == TRIGGER_NAMES
        assert {name for name in _object_names(conn, "index") if name.startswith("idx_")} == (
            INDEX_NAMES
        )


def test_second_open_keeps_recall_working_through_fts(tmp_path: Path) -> None:
    """CREATE ... IF NOT EXISTS must not rebuild the FTS index on reopen."""
    path = tmp_path / "reopen_fts.db"
    with _closed(path) as conn:
        _insert_fact(conn, "the runner was running")

    with _closed(path) as conn:
        hits = conn.execute("SELECT rowid FROM facts_fts WHERE facts_fts MATCH 'run'").fetchall()
        assert len(hits) == 1


# --------------------------------------------------------------------------- #
# schema_version gate
# --------------------------------------------------------------------------- #


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET"
        " value = excluded.value",
        (key, value),
    )
    conn.commit()


def test_schema_version_zero_is_refused_with_a_message_naming_the_fix(tmp_path: Path) -> None:
    """Step 2 acceptance: ``schema_version = 0`` is refused, and the fix is named."""
    path = tmp_path / "version_zero.db"
    with _closed(path) as conn:
        _set_meta(conn, "schema_version", "0")

    with pytest.raises(SchemaVersionError) as caught:
        _open(path)
    message = str(caught.value).lower()
    assert "0" in message, "the message must name the version found"
    assert str(SCHEMA_VERSION) in message, "the message must name the version required"
    assert any(word in message for word in ("delete", "recreate", "migrat")), (
        f"the message must name the fix, got: {message!r}"
    )


def test_schema_version_from_the_future_is_refused(tmp_path: Path) -> None:
    """A newer plugin's store must not be silently written by an older schema."""
    path = tmp_path / "version_future.db"
    with _closed(path) as conn:
        _set_meta(conn, "schema_version", "999")

    with pytest.raises(SchemaVersionError) as caught:
        _open(path)
    message = str(caught.value)
    assert "999" in message
    assert str(SCHEMA_VERSION) in message


def test_non_integer_schema_version_is_refused(tmp_path: Path) -> None:
    """Corrupt meta must fail loudly, never be interpreted as a version."""
    path = tmp_path / "version_garbage.db"
    with _closed(path) as conn:
        _set_meta(conn, "schema_version", "banana")

    with pytest.raises(SchemaVersionError) as caught:
        _open(path)
    assert str(caught.value).strip(), "the refusal must carry an explanation"


def test_missing_schema_version_on_a_populated_store_is_refused(tmp_path: Path) -> None:
    """A store with facts but no version is corrupt: stamping version 1 hides it.

    Interpretation recorded against the plan: §4 lists ``schema_version`` as a
    mandatory meta key, so its absence is an unknown-schema condition, not a
    fresh store. Silently re-initialising would be the "silent partial result"
    R9 forbids.
    """
    path = tmp_path / "version_missing.db"
    with _closed(path) as conn:
        _insert_fact(conn, "data written by an unknown schema")
        conn.execute("DELETE FROM meta WHERE key = 'schema_version'")
        conn.commit()

    with pytest.raises(SchemaVersionError) as caught:
        _open(path)
    message = str(caught.value).lower()
    assert "schema_version" in message or "version" in message


def test_an_empty_file_is_a_fresh_store_not_a_corrupt_one(tmp_path: Path) -> None:
    """A zero-byte file (touch'd, or left by a crash) must initialise normally."""
    path = tmp_path / "empty.db"
    path.touch()
    with _closed(path) as conn:
        assert int(read_meta(conn)["schema_version"]) == SCHEMA_VERSION
        assert _table_list(conn, "table") == {"meta", "facts"}


# --------------------------------------------------------------------------- #
# dimension guard (§4 / R9)
# --------------------------------------------------------------------------- #


def _assert_repair_contract(exc: EmbeddingDimensionMismatch) -> None:
    expected = int(exc.expected)
    actual = int(exc.actual)
    assert {expected, actual} == {EMBEDDING_DIMENSION, 768}
    message = str(exc)
    assert str(expected) in message
    assert str(actual) in message
    assert "memory_reindex --force" in message, "the repair path must be quoted verbatim"
    assert "delete" in message.lower(), "the alternative repair must be named"


def test_reopening_with_embedding_dim_768_raises_mismatch(tmp_path: Path) -> None:
    """Step 2 acceptance line, read as "a 768 demand on a 384 store is refused".

    Interpretation recorded against the plan: §4 says ``embedding_dim`` is
    "compared on every open", so a caller asking for 768 against a store that
    recorded 384 must be refused rather than silently handed a store that cannot
    hold its vectors -- exactly the silent partial result R9 forbids.
    """
    path = tmp_path / "dim_guard.db"
    with _closed(path):
        pass

    with pytest.raises(EmbeddingDimensionMismatch) as caught:
        _open(path, embedding_dim=768)
    _assert_repair_contract(caught.value)


def test_store_recording_embedding_dim_768_is_refused_on_open(tmp_path: Path) -> None:
    """The guard reads ``meta``, so a store written by another model is caught."""
    path = tmp_path / "dim_recorded.db"
    with _closed(path) as conn:
        _set_meta(conn, "embedding_dim", "768")

    with pytest.raises(EmbeddingDimensionMismatch) as caught:
        _open(path)
    _assert_repair_contract(caught.value)


def test_creating_a_store_at_768_is_refused(tmp_path: Path) -> None:
    """The exact DDL pins ``float[384]``; a 768 store cannot honour it.

    Interpretation recorded against the plan: creating at 768 would build a
    ``facts_vec`` whose width disagrees with ``meta.embedding_dim``, which is the
    silent partial result R9 forbids, so it is refused up front.
    """
    with pytest.raises(EmbeddingDimensionMismatch) as caught:
        _open(tmp_path / "dim_fresh.db", embedding_dim=768)
    assert {int(caught.value.expected), int(caught.value.actual)} == {EMBEDDING_DIMENSION, 768}


def test_dimension_mismatch_does_not_create_or_mutate_the_store(tmp_path: Path) -> None:
    """A refused open must leave the file alone: no half-built schema, no rewritten meta.

    This exercises the *recorded* dimension path -- the store disagrees with the
    schema's pinned width -- because that is the refusal that has to read the
    store first. (The ``embedding_dim=768`` argument path is rejected before the
    file is ever opened, so it cannot detect a mutation; see
    ``test_reopening_with_embedding_dim_768_raises_mismatch``.)

    The file header is compared byte-for-byte and the WAL sidecar set is checked,
    because ``PRAGMA journal_mode = WAL`` on a rollback-journal file rewrites
    bytes 18/19 of the header (from 1/1 to 2/2) and creates ``-wal``/``-shm``:
    that is the mutation a refusal must not perform. The fixture is deliberately
    left in rollback-journal mode so the mutation is observable.
    """
    path = tmp_path / "dim_untouched.db"
    with _closed(path) as conn:
        _set_meta(conn, "embedding_dim", "768")
        assert conn.execute("PRAGMA journal_mode = DELETE").fetchone()[0].lower() == "delete"

    before_header = path.read_bytes()[:100]
    assert path.read_bytes()[18:20] == b"\x01\x01", "fixture must start in rollback-journal mode"
    before_sidecars = sorted(p.name for p in tmp_path.glob("dim_untouched.db-*"))
    assert before_sidecars == [], "the store must be closed and checkpointed before the refusal"

    with pytest.raises(EmbeddingDimensionMismatch) as caught:
        _open(path)
    _assert_repair_contract(caught.value)

    after_sidecars = sorted(p.name for p in tmp_path.glob("dim_untouched.db-*"))
    assert after_sidecars == [], "a refused open must not create -wal/-shm sidecars"
    assert path.read_bytes()[:100] == before_header, (
        "a refused open must not rewrite the SQLite header (journal mode included)"
    )

    # A refusal is not a corruption: repairing the recorded dimension is enough.
    _repair_meta_out_of_band(path, "embedding_dim", str(EMBEDDING_DIMENSION))
    with _closed(path) as conn:
        assert read_meta(conn)["embedding_dim"] == str(EMBEDDING_DIMENSION)
        assert _table_list(conn, "virtual") == {"facts_fts", "facts_vec"}
        assert _object_names(conn, "trigger") == TRIGGER_NAMES


# --------------------------------------------------------------------------- #
# path handling
# --------------------------------------------------------------------------- #


def test_open_store_accepts_a_string_path(tmp_path: Path) -> None:
    path = tmp_path / "as_string.db"
    conn = open_store(str(path), tier="global", scope_key="scope-global")
    try:
        assert isinstance(conn, sqlite3.Connection)
        assert read_meta(conn)["tier"] == "global"
    finally:
        conn.close()
    assert path.is_file() and path.stat().st_size > 0


def test_open_store_accepts_a_path_object(tmp_path: Path) -> None:
    path = tmp_path / "as_path.db"
    conn = open_store(path, tier="session", scope_key="session-abc")
    try:
        assert read_meta(conn)["scope_key"] == "session-abc"
    finally:
        conn.close()
    assert path.is_file()


def test_invalid_tier_is_rejected_before_it_reaches_the_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError) as caught:
        open_store(tmp_path / "bad_tier.db", tier="galactic", scope_key="scope")
    assert "galactic" in str(caught.value)


# --------------------------------------------------------------------------- #
# import-time behaviour: no sqlite-vec, no model (§8 "FTS-only mode", R6)
# --------------------------------------------------------------------------- #

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


def test_schema_module_imports_with_sqlite_vec_uninstalled(tmp_path: Path) -> None:
    """Design constraint: a missing extension must not stop the module importing."""
    code = (
        _BLOCK_SQLITE_VEC
        + """
import json
import tiered_memory.schema as schema

print(json.dumps({
    "imported": True,
    "has_open_store": hasattr(schema, "open_store"),
    "has_read_meta": hasattr(schema, "read_meta"),
}))
"""
    )
    result = _run_python(code, tmp_path)
    assert result.returncode == 0, f"import failed with sqlite-vec blocked:\n{result.stderr}"
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["imported"] is True
    assert payload["has_open_store"] is True


def test_schema_module_does_not_import_the_embedding_model(tmp_path: Path) -> None:
    """R6: hooks reach the store; importing a 113 MB ONNX stack there is a defect."""
    code = """
import json
import sys

import tiered_memory.schema  # noqa: F401

loaded = sorted(
    name for name in sys.modules
    if name.split(".")[0] in {"fastembed", "onnxruntime", "onnx"}
)
print(json.dumps(loaded))
"""
    result = _run_python(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


_FTS_ONLY_HARNESS = """
import json
import pathlib
import sqlite3
import sys
import tempfile
import types

MODE = __MODE__

if MODE == "import_error":
    class _Blocker:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "sqlite_vec" or fullname.startswith("sqlite_vec."):
                raise ImportError("simulated: sqlite-vec is not installed")
            return None

    sys.meta_path.insert(0, _Blocker())
    sys.modules.pop("sqlite_vec", None)
else:
    def _boom(*args, **kwargs):
        raise sqlite3.OperationalError("simulated: cannot load the extension")

    stub = types.ModuleType("sqlite_vec")
    stub.load = _boom
    stub.loadable_path = _boom
    stub.__version__ = "0.1.9"
    sys.modules["sqlite_vec"] = stub

import tiered_memory.schema as schema

path = pathlib.Path(tempfile.mkdtemp()) / "store.db"
conn = schema.open_store(path, tier="global", scope_key="global")
meta = schema.read_meta(conn)
virtual = sorted(
    row[1] for row in conn.execute("PRAGMA table_list")
    if row[0] == "main" and row[2] == "virtual"
)
conn.execute(
    "INSERT INTO facts (content_hash, subject_key, text, kind, tier, scope_key,"
    " created_at, updated_at, last_seen_at) VALUES"
    " ('h1', 's1', 'the runner was running', 'fact', 'global', 'global', 't', 't', 't')"
)
conn.commit()
hits = [
    row[0] for row in
    conn.execute("SELECT rowid FROM facts_fts WHERE facts_fts MATCH 'run'").fetchall()
]
conn.close()

# A second open must not raise either.
again = schema.open_store(path, tier="global", scope_key="global")
second_state = schema.read_meta(again)["index_state"]
again.close()

print(json.dumps({
    "index_state": meta["index_state"],
    "virtual": virtual,
    "hits": hits,
    "second_state": second_state,
}))
"""


@pytest.mark.parametrize("mode", ["import_error", "load_error"])
def test_missing_extension_falls_back_to_fts_only(tmp_path: Path, mode: str) -> None:
    """§5.1: "any -> fts_only | sqlite-vec fails to load ... never raises".

    Run in a subprocess so the absence is real: a fake ``sqlite_vec`` module that
    cannot load is not something an in-process monkeypatch reproduces faithfully,
    and the module may bind the extension at import time.
    """
    code = _FTS_ONLY_HARNESS.replace("__MODE__", repr(mode))
    result = _run_python(code, tmp_path)
    assert result.returncode == 0, f"{mode} broke the store open:\n{result.stderr}"
    payload = json.loads(result.stdout.strip().splitlines()[-1])

    assert payload["index_state"] == "fts_only", (
        f"a store opened without sqlite-vec must record fts_only, got {payload['index_state']!r}"
    )
    assert payload["virtual"] == ["facts_fts"], "facts_vec cannot exist without the extension"
    assert payload["hits"] == [1], "the FTS path must keep working in fts_only mode"
    assert payload["second_state"] == "fts_only", "reopening must keep the fallback stable"


# --------------------------------------------------------------------------- #
# Committed, not pending: what a *separate* connection can see
# --------------------------------------------------------------------------- #


def test_seeded_meta_is_committed_not_left_in_an_open_transaction(tmp_path: Path) -> None:
    """The store must be usable by another process the moment ``open_store`` returns.

    A hook that opens a store and dies must not leave the schema uncommitted;
    and a caller should not inherit an open write transaction. An independent
    raw connection is the only honest witness here.
    """
    path = tmp_path / "committed.db"
    with _closed(path) as conn:
        assert not conn.in_transaction, "open_store handed back an open transaction"

    probe = sqlite3.connect(str(path))
    try:
        rows = dict(probe.execute("SELECT key, value FROM meta").fetchall())
        tables = {
            row[1]
            for row in probe.execute("PRAGMA table_list").fetchall()
            if row[0] == "main" and row[2] in ("table", "virtual")
        }
    finally:
        probe.close()

    assert rows.get("schema_version") == str(SCHEMA_VERSION), "meta seed was not committed"
    assert {"meta", "facts", "facts_fts"} <= tables


def test_open_store_does_not_leave_the_file_locked(tmp_path: Path) -> None:
    """A leaked write lock would stall the next hook for its whole busy timeout."""
    path = tmp_path / "unlocked.db"
    with _closed(path):
        pass

    probe = sqlite3.connect(str(path), timeout=0)
    try:
        probe.execute("BEGIN IMMEDIATE")
        probe.execute(
            "INSERT INTO facts (content_hash, subject_key, text, kind, tier,"
            " scope_key, created_at, updated_at, last_seen_at)"
            " VALUES ('h', 's', 't', 'fact', 'global', 'global', 't', 't', 't')"
        )
        probe.commit()
    finally:
        probe.close()


def test_second_open_succeeds_while_first_connection_is_still_open(tmp_path: Path) -> None:
    """Two sessions share the project store (§8 "concurrent sessions")."""
    path = tmp_path / "concurrent.db"
    first = open_store(path, tier="project", scope_key="scope-project")
    second: sqlite3.Connection | None = None
    try:
        second = open_store(path, tier="project", scope_key="scope-project")
        assert read_meta(second)["tier"] == "project"
        _insert_fact(first, "written by the first connection")
        rows = second.execute("SELECT count(*) FROM facts").fetchone()
        assert rows is not None and int(rows[0]) == 1
    finally:
        first.close()
        if second is not None:
            second.close()


# --------------------------------------------------------------------------- #
# Refused opens must not poison the file
# --------------------------------------------------------------------------- #


def _repair_meta_out_of_band(path: Path, key: str, value: str) -> None:
    """Fix ``meta`` the way an operator would: with a raw connection.

    ``open_store`` refuses a bad version by design, so the repair cannot go
    through it — that is the whole point of the gate being non-destructive.
    """
    raw = sqlite3.connect(str(path))
    try:
        raw.execute("UPDATE meta SET value = ? WHERE key = ?", (value, key))
        raw.commit()
    finally:
        raw.close()


def test_store_opens_normally_after_a_refused_version_gate(tmp_path: Path) -> None:
    """A refusal is a refusal, not a corruption: repairing meta must be enough."""
    path = tmp_path / "repair.db"
    with _closed(path) as conn:
        _set_meta(conn, "schema_version", "0")

    with pytest.raises(SchemaVersionError):
        _open(path)

    _repair_meta_out_of_band(path, "schema_version", str(SCHEMA_VERSION))

    with _closed(path) as conn:
        assert int(read_meta(conn)["schema_version"]) == SCHEMA_VERSION
        assert _table_list(conn, "table") == {"meta", "facts"}


def test_version_gate_does_not_drop_the_facts_it_refused_to_serve(tmp_path: Path) -> None:
    """R4: no automatic deletion, ever — including on a refused open."""
    path = tmp_path / "keep_facts.db"
    with _closed(path) as conn:
        _insert_fact(conn, "a fact that must never be deleted")
        _set_meta(conn, "schema_version", "0")

    with pytest.raises(SchemaVersionError):
        _open(path)

    _repair_meta_out_of_band(path, "schema_version", str(SCHEMA_VERSION))
    with _closed(path) as conn:
        rows = conn.execute("SELECT text FROM facts").fetchall()
    assert rows == [("a fact that must never be deleted",)]


# --------------------------------------------------------------------------- #
# Corrupt state
# --------------------------------------------------------------------------- #


def test_a_non_sqlite_file_is_refused_rather_than_overwritten(tmp_path: Path) -> None:
    """Corrupt bytes must raise, not be treated as an empty store.

    Interpretation recorded against the plan: the plan does not spell out a
    corrupt-file path, but R9 forbids a "silent partial result", and treating a
    garbage file as fresh would destroy whatever the user had. SQLite reports
    this as ``DatabaseError`` ("file is not a database"); the module may raise it
    directly or wrap it in a ``DatabaseError`` subclass.
    """
    path = tmp_path / "corrupt.db"
    original = b"this is definitely not a sqlite database\x00\x01\x02"
    path.write_bytes(original)

    with pytest.raises(sqlite3.DatabaseError):
        _open(path)

    assert path.read_bytes().startswith(b"this is definitely not"), (
        "a refused corrupt file must not be rewritten"
    )
