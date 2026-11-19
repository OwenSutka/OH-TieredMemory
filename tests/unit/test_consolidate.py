"""Behaviour of ``tiered_memory.consolidate`` (plan §3.4, Step 11 / TM-11).

The claims under test:

* ``SessionEnd`` promotes session facts into the project store, once, however
  many times consolidation runs (dedupe, and the session row is retained);
* a preference seen in **one** session stays in the project store; the same
  preference seen in **two** distinct sessions reaches global (R2);
* a non-preference is never promoted to global, however many sessions saw it;
* the prune pass only ever targets superseded rows older than the threshold, and
  its default is a candidate count that deletes nothing (R4: no automatic
  deletion);
* consolidation of a missing session store is a no-op rather than an error.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

import tiered_memory.config as config
import tiered_memory.consolidate as consolidate
import tiered_memory.facts as facts
import tiered_memory.store as store


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


def _session_fact(session_id: str, text: str, kind: str) -> None:
    with store.session(session_id) as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, text, kind=kind, source_session=session_id, source_tool="terminal")


def _texts(path: Path) -> list[str]:
    if not path.exists():
        return []
    conn = sqlite3.connect(str(path), timeout=10.0)
    try:
        return sorted(str(row[0]) for row in conn.execute("SELECT text FROM facts"))
    finally:
        conn.close()


def _row(path: Path, text: str) -> tuple[int, int] | None:
    conn = sqlite3.connect(str(path), timeout=10.0)
    try:
        found = conn.execute(
            "SELECT seen_count, distinct_sessions FROM facts WHERE text = ?", (text,)
        ).fetchone()
        if found is None:
            return None
        return (int(found[0]), int(found[1]))
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# session -> project
# --------------------------------------------------------------------------- #


def test_a_session_fact_reaches_the_project_store(project_dir: Path) -> None:
    _session_fact("s1", "locked sqlite over hindsight", "decision")

    result = consolidate.consolidate(str(project_dir), "s1")

    assert result.promoted == 1
    assert _texts(config.project_db(str(project_dir))) == ["locked sqlite over hindsight"]


def test_the_session_row_is_retained_after_promotion(project_dir: Path) -> None:
    """§5.2 row 5: ``session -> project`` adds a project row; nothing is removed."""
    _session_fact("s1", "locked sqlite over hindsight", "decision")

    consolidate.consolidate(str(project_dir), "s1")

    assert _texts(config.session_db("s1")) == ["locked sqlite over hindsight"]


def test_consolidating_twice_does_not_duplicate_the_project_row(project_dir: Path) -> None:
    _session_fact("s1", "locked sqlite over hindsight", "decision")

    consolidate.consolidate(str(project_dir), "s1")
    second = consolidate.consolidate(str(project_dir), "s1")

    # A session is promoted once: a retry adds nothing, which is what stops a
    # repeat from inflating the project row's ``distinct_sessions``.
    assert second.promoted == 0
    assert second.merged == 0
    assert _texts(config.project_db(str(project_dir))) == ["locked sqlite over hindsight"]


def test_a_missing_session_store_is_a_no_op(project_dir: Path) -> None:
    result = consolidate.consolidate(str(project_dir), "never-existed")

    assert result == consolidate.ConsolidationResult(0, 0, 0, 0, 0)


def test_two_sessions_with_the_same_fact_count_as_two_distinct_sessions(
    project_dir: Path,
) -> None:
    _session_fact("s1", "prefer short commit messages", "preference")
    _session_fact("s2", "prefer short commit messages", "preference")

    consolidate.consolidate(str(project_dir), "s1")
    consolidate.consolidate(str(project_dir), "s2")

    row = _row(config.project_db(str(project_dir)), "prefer short commit messages")
    assert row is not None
    assert row[1] == 2


# --------------------------------------------------------------------------- #
# project -> global
# --------------------------------------------------------------------------- #


def test_a_one_session_preference_stays_in_the_project_store(project_dir: Path) -> None:
    _session_fact("s1", "prefer short commit messages", "preference")

    consolidate.consolidate(str(project_dir), "s1")

    assert _texts(config.global_db()) == []
    assert _texts(config.project_db(str(project_dir))) == ["prefer short commit messages"]


def test_a_preference_seen_in_two_sessions_reaches_global(project_dir: Path) -> None:
    """R2: ``kind = preference`` **and** >= 2 distinct sessions."""
    _session_fact("s1", "prefer short commit messages", "preference")
    _session_fact("s2", "prefer short commit messages", "preference")

    consolidate.consolidate(str(project_dir), "s1")
    result = consolidate.consolidate(str(project_dir), "s2")

    assert result.global_promoted == 1
    assert _texts(config.global_db()) == ["prefer short commit messages"]


def test_promoting_to_global_is_idempotent(project_dir: Path) -> None:
    _session_fact("s1", "prefer short commit messages", "preference")
    _session_fact("s2", "prefer short commit messages", "preference")
    consolidate.consolidate(str(project_dir), "s1")
    consolidate.consolidate(str(project_dir), "s2")

    third = consolidate.consolidate(str(project_dir), "s2")

    assert third.global_promoted == 0
    assert _texts(config.global_db()) == ["prefer short commit messages"]
    row = _row(config.global_db(), "prefer short commit messages")
    assert row is not None
    assert row[1] == 2


def test_the_global_row_records_the_distinct_session_count(project_dir: Path) -> None:
    _session_fact("s1", "prefer short commit messages", "preference")
    _session_fact("s2", "prefer short commit messages", "preference")

    consolidate.consolidate(str(project_dir), "s1")
    consolidate.consolidate(str(project_dir), "s2")

    row = _row(config.global_db(), "prefer short commit messages")
    assert row == (2, 2)


def test_a_non_preference_seen_in_two_sessions_never_reaches_global(project_dir: Path) -> None:
    _session_fact("s1", "locked sqlite over hindsight", "decision")
    _session_fact("s2", "locked sqlite over hindsight", "decision")

    consolidate.consolidate(str(project_dir), "s1")
    consolidate.consolidate(str(project_dir), "s2")

    assert _texts(config.global_db()) == []


# --------------------------------------------------------------------------- #
# prune pass
# --------------------------------------------------------------------------- #


def _superseded_old_row(project_dir: Path) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        old = facts.insert(
            conn, "old superseded fact", kind="fact", now=datetime(2020, 1, 1, tzinfo=UTC)
        )
        new = facts.insert(
            conn, "new fact that wins", kind="fact", now=datetime(2020, 1, 1, tzinfo=UTC)
        )
        facts.supersede(conn, old_id=old.id, new_id=new.id, now=datetime(2020, 1, 2, tzinfo=UTC))


def test_the_prune_pass_defaults_to_reporting_candidates_only(project_dir: Path) -> None:
    """R4: no automatic deletion, so the default counts and removes nothing."""
    _superseded_old_row(project_dir)

    result = consolidate.consolidate(str(project_dir), "s1")

    assert result.prune_candidates == 1
    assert result.pruned == 0
    assert _texts(config.project_db(str(project_dir))) == [
        "new fact that wins",
        "old superseded fact",
    ]


def test_the_prune_pass_removes_only_superseded_rows_when_not_a_dry_run(
    project_dir: Path,
) -> None:
    _superseded_old_row(project_dir)

    result = consolidate.consolidate(str(project_dir), "s1", dry_run_prune=False)

    assert result.pruned == 1
    assert _texts(config.project_db(str(project_dir))) == ["new fact that wins"]


def test_the_prune_pass_leaves_a_recent_superseded_row_alone(project_dir: Path) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        old = facts.insert(conn, "recently superseded fact", kind="fact")
        new = facts.insert(conn, "its successor", kind="fact")
        facts.supersede(conn, old_id=old.id, new_id=new.id)

    result = consolidate.consolidate(str(project_dir), "s1", dry_run_prune=False)

    assert result.pruned == 0
    assert "recently superseded fact" in _texts(config.project_db(str(project_dir)))


def test_a_live_fact_is_never_pruned(project_dir: Path) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, "a live fact", kind="fact")

    result = consolidate.consolidate(str(project_dir), "s1", dry_run_prune=False)

    assert result.pruned == 0
    assert _texts(config.project_db(str(project_dir))) == ["a live fact"]
