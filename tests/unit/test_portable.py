"""Behaviour of ``tiered_memory.portable`` (plan §2.2.7, Step 8 / TM-08).

The claims under test come straight from the plan:

* export is JSONL: one header line, then one line per fact, full metadata and
  **no vectors**; the file is replaced atomically;
* the header is validated against the local ``embedding_model`` /
  ``embedding_dim`` and a mismatch is refused with the expected values;
* the tier recorded on a fact is preserved, and a record for another tier is
  refused rather than silently re-tiered;
* conflicts merge (``seen_count + 1``, ``confidence = max``,
  ``last_seen = max``, ``distinct_sessions + 1`` only for a new session);
* importing the same file twice is a no-op the second time;
* a superseded row stays superseded, so import never resurrects it.

Every test uses real stores and real files: dedupe, supersede and the atomic
replace are properties of the file, so a fake connection would prove nothing.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import tiered_memory.config as config
import tiered_memory.facts as facts
import tiered_memory.portable as portable
import tiered_memory.schema as schema
import tiered_memory.store as store

_NOW = "2026-06-01T00:00:00Z"
_LATER = "2026-08-01T00:00:00Z"
_EVEN_LATER = "2026-09-01T00:00:00Z"

_COMPARED_COLUMNS = (
    "content_hash, subject_key, text, kind, tier, scope_key, source_tool,"
    " source_session, confidence, seen_count, distinct_sessions, created_at,"
    " updated_at, last_seen_at, superseded_by"
)


@pytest.fixture(autouse=True)
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate every test in its own plugin data root."""
    root = tmp_path / "plugin-data"
    monkeypatch.setenv(config.DATA_ENV, str(root))
    return root


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    path = tmp_path / "project"
    path.mkdir()
    return path


@pytest.fixture()
def export_path(tmp_path: Path) -> Path:
    return tmp_path / "export.jsonl"


def _rows(conn: sqlite3.Connection) -> list[tuple[object, ...]]:
    return [
        tuple(row) for row in conn.execute(f"SELECT {_COMPARED_COLUMNS} FROM facts ORDER BY id")
    ]


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in _lines(path)[1:] if line.strip()]


def _rewrite(path: Path, index: int, **changes: object) -> None:
    lines = _lines(path)
    record = json.loads(lines[index])
    record.update(changes)
    lines[index] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _superseding_pair(conn: sqlite3.Connection) -> tuple[int, int]:
    old_id = facts.insert(conn, "old gamma").id
    new_id = facts.insert(conn, "new gamma").id
    facts.supersede(conn, old_id=old_id, new_id=new_id)
    return old_id, new_id


def _restore(
    export_path: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[portable.FactRecord]:
    """Read the export against a fresh data root, as a restore onto another host."""
    monkeypatch.setenv(config.DATA_ENV, str(tmp_path / "restored-data"))
    _header_value, records = portable.read(export_path)
    return records


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


def test_export_writes_one_header_line_then_one_line_per_fact(
    project_dir: Path, export_path: Path
) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, "alpha fact", kind="fact")
        facts.insert(conn, "beta decision", kind="decision")
        result = portable.export([conn], export_path)

    lines = _lines(export_path)
    header = json.loads(lines[0])
    assert result.facts == 2
    assert result.path == export_path
    assert header["schema_version"] == portable.HEADER_SCHEMA_VERSION
    assert header["tier"] == "project"
    assert header["embedding_model"] == schema.DEFAULT_EMBEDDING_MODEL
    assert header["embedding_dim"] == schema.EMBEDDING_DIMENSION
    assert len(lines) == 3

    record = json.loads(lines[1])
    assert record["text"] == "alpha fact"
    assert record["tier"] == "project"
    assert "embedding" not in record
    assert record["superseded_by_hash"] is None


def test_export_omits_soft_deleted_facts(project_dir: Path, export_path: Path) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, "alpha kept")
        gone = facts.insert(conn, "alpha gone").id
        conn.execute("UPDATE facts SET deleted_at = 't' WHERE id = ?", (gone,))
        result = portable.export([conn], export_path)

    assert result.facts == 1
    assert [record["text"] for record in _records(export_path)] == ["alpha kept"]


def test_export_of_an_empty_store_writes_only_a_header(
    project_dir: Path, export_path: Path
) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        result = portable.export([conn], export_path)

    assert result.facts == 0
    assert len(_lines(export_path)) == 1


def test_export_replaces_the_target_and_leaves_no_temporary_file(
    project_dir: Path, export_path: Path
) -> None:
    export_path.write_text("stale contents\n", encoding="utf-8")
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, "alpha fact")
        portable.export([conn], export_path)

    assert "stale contents" not in export_path.read_text(encoding="utf-8")
    leftovers = [
        entry.name for entry in export_path.parent.iterdir() if entry.name.endswith(".tmp")
    ]
    assert leftovers == []


def test_export_of_several_stores_records_no_single_tier(
    project_dir: Path, export_path: Path
) -> None:
    with store.global_store() as global_conn, store.project(str(project_dir)) as project_conn:
        assert isinstance(global_conn, sqlite3.Connection)
        assert isinstance(project_conn, sqlite3.Connection)
        facts.insert(global_conn, "alpha global")
        facts.insert(project_conn, "alpha project")
        result = portable.export([project_conn, global_conn], export_path)

    assert result.facts == 2
    assert json.loads(_lines(export_path)[0])["tier"] is None
    assert {record["tier"] for record in _records(export_path)} == {"global", "project"}


# --------------------------------------------------------------------------- #
# header validation
# --------------------------------------------------------------------------- #


def test_a_dimension_mismatch_is_refused_with_the_expected_values(
    project_dir: Path, export_path: Path
) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, "alpha fact")
        portable.export([conn], export_path)
    _rewrite(export_path, 0, embedding_dim=768)

    with pytest.raises(portable.ImportMismatchError) as raised:
        portable.read(export_path)

    error = raised.value
    assert error.expected_dim == schema.EMBEDDING_DIMENSION
    assert error.actual_dim == 768
    assert "384" in str(error)
    assert "768" in str(error)


def test_a_model_mismatch_is_refused_with_the_expected_values(
    project_dir: Path, export_path: Path
) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        portable.export([conn], export_path)
    _rewrite(export_path, 0, embedding_model="some/other-model")

    with pytest.raises(portable.ImportMismatchError) as raised:
        portable.read(export_path)

    assert raised.value.expected_model == schema.DEFAULT_EMBEDDING_MODEL
    assert raised.value.actual_model == "some/other-model"


def test_an_unknown_header_schema_version_is_refused(project_dir: Path, export_path: Path) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        portable.export([conn], export_path)
    _rewrite(export_path, 0, schema_version=2)

    with pytest.raises(portable.ExportFormatError):
        portable.read(export_path)


def test_a_malformed_line_names_its_line_number(project_dir: Path, export_path: Path) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        portable.export([conn], export_path)
    with export_path.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")

    with pytest.raises(portable.ExportFormatError) as raised:
        portable.read(export_path)

    assert "line 2" in str(raised.value)


def test_a_record_whose_hash_disagrees_with_its_text_is_refused(
    project_dir: Path, export_path: Path
) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, "alpha fact")
        portable.export([conn], export_path)
    _rewrite(export_path, 1, text="tampered text")

    with pytest.raises(portable.ExportFormatError):
        portable.read(export_path)


def test_a_missing_export_file_is_a_format_error(tmp_path: Path) -> None:
    with pytest.raises(portable.ExportFormatError):
        portable.read(tmp_path / "absent.jsonl")


# --------------------------------------------------------------------------- #
# import
# --------------------------------------------------------------------------- #


def test_the_round_trip_restores_every_field(
    project_dir: Path, export_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, "alpha fact", kind="fact", confidence=0.9)
        facts.insert(conn, "beta decision", kind="decision")
        before = _rows(conn)
        portable.export([conn], export_path)

    records = _restore(export_path, monkeypatch, tmp_path)
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        result = portable.apply(conn, records)
        after = _rows(conn)

    assert (result.imported, result.merged, result.skipped) == (2, 0, 0)
    assert after == before


def test_importing_the_same_file_twice_changes_nothing(
    project_dir: Path, export_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, "alpha fact")
        facts.insert(conn, "beta decision", kind="decision")
        _superseding_pair(conn)
        portable.export([conn], export_path)

    records = _restore(export_path, monkeypatch, tmp_path)
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        first = portable.apply(conn, records)
        snapshot = _rows(conn)
        second = portable.apply(conn, records)

        assert first.imported == 4
        assert (second.imported, second.merged, second.skipped) == (0, 0, 4)
        assert _rows(conn) == snapshot


def test_a_conflict_merges_seen_count_confidence_and_last_seen(
    project_dir: Path, export_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, "alpha fact", confidence=0.5)
        facts.insert(conn, "alpha fact", confidence=0.6)
        conn.execute("UPDATE facts SET last_seen_at = ?, updated_at = ?", (_NOW, _NOW))
        portable.export([conn], export_path)

    records = _restore(export_path, monkeypatch, tmp_path)
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        portable.apply(conn, records)
        stored = conn.execute(
            "SELECT seen_count, confidence, last_seen_at, distinct_sessions FROM facts"
        ).fetchone()
        assert stored is not None
        _rewrite(export_path, 1, seen_count=5, confidence=0.99, last_seen_at=_LATER)
        _header_value, advanced = portable.read(export_path)
        result = portable.apply(conn, advanced)
        row = conn.execute(
            "SELECT seen_count, confidence, last_seen_at, distinct_sessions FROM facts"
        ).fetchone()

    assert stored[0] == 2
    assert result.merged == 1
    assert row is not None
    assert row[0] == stored[0] + 1
    assert row[1] == pytest.approx(0.99)
    assert row[2] == _LATER
    assert row[3] == stored[3]


def test_distinct_sessions_increments_only_for_a_different_source_session(
    project_dir: Path, export_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, "alpha fact", source_session="s1")
        portable.export([conn], export_path)

    records = _restore(export_path, monkeypatch, tmp_path)
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        portable.apply(conn, records)
        _rewrite(export_path, 1, last_seen_at=_LATER, source_session="s1")
        _header_value, same_session = portable.read(export_path)
        portable.apply(conn, same_session)
        same = conn.execute("SELECT distinct_sessions FROM facts").fetchone()

        _rewrite(export_path, 1, last_seen_at=_EVEN_LATER, source_session="s2")
        _header_value, other_session = portable.read(export_path)
        portable.apply(conn, other_session)
        other = conn.execute("SELECT distinct_sessions FROM facts").fetchone()

    assert same is not None and same[0] == 1
    assert other is not None and other[0] == 2


def test_a_superseded_fact_stays_superseded_after_import(
    project_dir: Path, export_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        _superseding_pair(conn)
        portable.export([conn], export_path)

    records = _restore(export_path, monkeypatch, tmp_path)
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        portable.apply(conn, records)
        live = [fact.text for fact in facts.live_facts(conn)]
        old_row = conn.execute(
            "SELECT superseded_by FROM facts WHERE content_hash = ?",
            (facts.content_hash("old gamma"),),
        ).fetchone()

    assert live == ["new gamma"]
    assert old_row is not None
    assert old_row[0] is not None


def test_a_record_for_another_tier_is_refused_and_nothing_is_written(
    project_dir: Path, export_path: Path
) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, "alpha fact")
        portable.export([conn], export_path)
    _header_value, records = portable.read(export_path)

    with store.global_store() as conn:
        assert isinstance(conn, sqlite3.Connection)
        with pytest.raises(portable.ExportFormatError):
            portable.apply(conn, records)
        assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0


def test_a_record_whose_successor_is_missing_is_refused_before_anything_is_written(
    project_dir: Path, export_path: Path
) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, "alpha fact")
        facts.insert(conn, "beta fact")
        portable.export([conn], export_path)
    _rewrite(export_path, 1, superseded_by_hash=facts.content_hash("never exported"))
    _header_value, records = portable.read(export_path)

    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        conn.execute("DELETE FROM facts")
        with pytest.raises(portable.ExportFormatError):
            portable.apply(conn, records)
        assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0


def test_import_keeps_the_target_stores_identity_not_the_files(
    project_dir: Path, export_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, "alpha fact")
        portable.export([conn], export_path)

    _rewrite(export_path, 1, scope_key="stale-scope")
    records = _restore(export_path, monkeypatch, tmp_path)
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        portable.apply(conn, records)
        row = conn.execute("SELECT tier, scope_key FROM facts").fetchone()

    assert row is not None
    assert tuple(row) == ("project", config.project_key(str(project_dir)))


def test_a_retiered_record_imports_into_the_target_tier(
    project_dir: Path, export_path: Path
) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, "alpha fact")
        portable.export([conn], export_path)
    _header_value, records = portable.read(export_path)

    with store.global_store() as conn:
        assert isinstance(conn, sqlite3.Connection)
        result = portable.apply(conn, records, retier=True)
        row = conn.execute("SELECT tier, scope_key FROM facts").fetchone()

    assert (result.imported, result.merged, result.skipped) == (1, 0, 0)
    assert row is not None
    assert tuple(row) == ("global", "global")


def test_read_accepts_an_explicit_expected_identity(project_dir: Path, export_path: Path) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        portable.export([conn], export_path)

    header, records = portable.read(
        export_path,
        expected_model=schema.DEFAULT_EMBEDDING_MODEL,
        expected_dim=schema.EMBEDDING_DIMENSION,
    )
    assert header.schema_version == portable.HEADER_SCHEMA_VERSION
    assert records == []
