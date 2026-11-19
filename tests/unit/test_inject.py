"""Behaviour of ``tiered_memory.inject`` (plan §2.2.4, Step 9 / TM-09).

The claims under test come straight from the plan:

* the inherited block is built from **global + project only** -- a session fact
  must never appear, whatever session is active;
* an empty pair of stores emits **nothing at all**: no header, no blank block;
* an empty read never creates a store, because this runs before every turn;
* the block costs at most 1500 tokens, and a project store's
  ``meta.budget_tokens`` overrides that budget;
* ordering is project ``decision``/``architecture`` first, then global
  ``preference``, then the remainder by score.

Every test uses real stores in a real plugin-data directory: the tier isolation
this module exists to provide is a property of which file is opened, so a fake
connection would prove nothing.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import pytest

import tiered_memory.config as config
import tiered_memory.facts as facts
import tiered_memory.inject as inject
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


def _write_project(project_dir: Path, text: str, kind: str) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, text, kind=kind, source_session="seed")


def _write_global(text: str, kind: str) -> None:
    with store.global_store() as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, text, kind=kind, source_session="seed")


def _write_session(text: str, kind: str, *, session_id: str) -> None:
    with store.session(session_id) as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, text, kind=kind, source_session=session_id)


def _set_budget(project_dir: Path, value: int) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('budget_tokens', ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(value),),
        )


# --------------------------------------------------------------------------- #
# silence
# --------------------------------------------------------------------------- #


def test_empty_stores_emit_nothing_at_all(project_dir: Path) -> None:
    """Silent means the empty string -- not a header, not a blank line."""
    assert inject.inherited_block(str(project_dir)) == ""


def test_a_reading_hook_does_not_create_the_stores_it_reads(
    project_dir: Path, data_root: Path
) -> None:
    """An empty read must not bootstrap a store, or every prompt would write."""
    inject.inherited_block(str(project_dir))

    created = sorted(path.name for path in config.tiers_dir().glob("*.db"))
    assert created == []


# --------------------------------------------------------------------------- #
# tier scope
# --------------------------------------------------------------------------- #


def test_the_session_tier_is_never_read(project_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """§2.2.4: within-session continuity is the host's conversation state."""
    _write_global("always use uv", "preference")
    _write_session("never touch the session store", "preference", session_id="s1")
    monkeypatch.setenv("OPENHANDS_SESSION_ID", "s1")

    block = inject.inherited_block(str(project_dir))

    assert "always use uv" in block
    assert "session store" not in block


def test_a_project_fact_is_inherited(project_dir: Path) -> None:
    _write_project(project_dir, "locked sqlite over hindsight", "decision")

    assert "locked sqlite over hindsight" in inject.inherited_block(str(project_dir))


# --------------------------------------------------------------------------- #
# budget
# --------------------------------------------------------------------------- #


def test_the_default_budget_is_respected(project_dir: Path) -> None:
    """§2.2.4: 1500 tokens, with every fact truncated to 240 characters."""
    for index in range(60):
        _write_project(project_dir, f"project fact number {index} " + "x" * 300, "fact")

    block = inject.inherited_block(str(project_dir))

    assert block != ""
    assert inject.estimate_tokens(block) <= config.INJECTION_TOKEN_BUDGET
    assert all(len(line) <= 240 for line in block.splitlines())


def test_the_default_budget_is_the_configured_constant() -> None:
    assert inject.DEFAULT_BUDGET_TOKENS == config.INJECTION_TOKEN_BUDGET == 1500


def test_a_project_budget_override_is_honoured(project_dir: Path) -> None:
    """A project store's ``meta.budget_tokens`` wins over the default."""
    _set_budget(project_dir, 40)
    for index in range(10):
        _write_project(project_dir, f"short project fact {index}", "fact")

    block = inject.inherited_block(str(project_dir))

    assert block != ""
    assert inject.estimate_tokens(block) <= 40
    assert len(block.splitlines()) < 10


def test_an_explicit_budget_override_beats_the_project_meta(project_dir: Path) -> None:
    _set_budget(project_dir, 5)
    for index in range(6):
        _write_project(project_dir, f"short project fact {index}", "fact")

    permissive = inject.inherited_block(str(project_dir), budget_tokens=100)
    restrictive = inject.inherited_block(str(project_dir))

    assert restrictive == ""
    assert len(permissive.splitlines()) == 6


# --------------------------------------------------------------------------- #
# ordering
# --------------------------------------------------------------------------- #


def test_order_is_project_decision_then_global_preference_then_the_rest(
    project_dir: Path,
) -> None:
    _write_project(project_dir, "we decided to keep the flat schema", "decision")
    _write_global("prefer short commit messages", "preference")
    _write_project(project_dir, "the test suite runs in 27 seconds", "fact")

    lines = inject.inherited_block(str(project_dir)).splitlines()

    assert lines[0] == "we decided to keep the flat schema"
    assert lines[1] == "prefer short commit messages"
    assert lines[2] == "the test suite runs in 27 seconds"


def test_project_architecture_sorts_with_decisions_before_global_preferences(
    project_dir: Path,
) -> None:
    _write_global("always use uv", "preference")
    _write_project(project_dir, "architecture uses sqlite vec", "architecture")

    lines = inject.inherited_block(str(project_dir)).splitlines()

    assert lines == ["architecture uses sqlite vec", "always use uv"]


def test_the_same_fact_in_two_tiers_appears_once(project_dir: Path) -> None:
    _write_project(project_dir, "always use uv", "preference")
    _write_global("always use uv", "preference")

    lines = inject.inherited_block(str(project_dir)).splitlines()

    assert lines == ["always use uv"]


def test_estimate_tokens_is_one_quarter_of_the_length() -> None:
    assert inject.estimate_tokens("") == 0
    assert inject.estimate_tokens("abcd") == 1
    assert inject.estimate_tokens("abcde") == math.ceil(5 / 4)


def test_the_block_has_no_header(project_dir: Path) -> None:
    _write_project(project_dir, "locked sqlite over hindsight", "decision")

    block = inject.inherited_block(str(project_dir))

    assert block == "locked sqlite over hindsight"
