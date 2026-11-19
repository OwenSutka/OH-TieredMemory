"""The five isolation tests of plan §8, each with a mutation that breaks it (TM-12).

Isolation is the whole point of the tier split, so these tests are end-to-end:
real stores, real files, and the real hook process where a hook is what crosses a
tier boundary.

A green test proves nothing on its own, so each case is paired with a mutation
that removes the guard it depends on. The mutated run must then *fail* -- if it
passed, the test would be asserting something the mutation cannot reach. The
guards and their mutations:

=====================================  =====================================
guard                                  mutation
=====================================  =====================================
``config.project_key`` separates        both project paths resolve to project
projects                               A's store
``config.session_db`` is one file       every session id resolves to one file
per session
``store.session`` is the only write     session writes are redirected to the
path for automatic writes              project store
``consolidate._PROMOTION_THRESHOLD``    lower it to 1, so one session promotes
is 2
concurrency safety                      shared session file, so both sessions'
                                       facts collide
=====================================  =====================================
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import tiered_memory.config as config
import tiered_memory.consolidate as consolidate
import tiered_memory.facts as facts
import tiered_memory.inject as inject
import tiered_memory.store as store

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOOKS = _REPO_ROOT / "dev.openhands" / "hooks"
_SRC = _REPO_ROOT / "src"


@pytest.fixture(autouse=True)
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate every test in its own plugin data root."""
    root = tmp_path / "plugin-data"
    monkeypatch.setenv(config.DATA_ENV, str(root))
    return root


@pytest.fixture()
def project_a(tmp_path: Path) -> Path:
    path = tmp_path / "project-a"
    path.mkdir()
    return path


@pytest.fixture()
def project_b(tmp_path: Path) -> Path:
    path = tmp_path / "project-b"
    path.mkdir()
    return path


def _write(tier: str, target: str, text: str, kind: str = "fact") -> None:
    """Write one fact into ``global``, a ``project`` path, or a session id."""
    if tier == "global":
        entry = store.global_store()
    elif tier == "project":
        entry = store.project(target)
    else:
        entry = store.session(target)
    with entry as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, text, kind=kind, source_session=target or "global")


def _texts(path: Path) -> list[str]:
    if not path.exists():
        return []
    conn = sqlite3.connect(str(path), timeout=10.0)
    try:
        return sorted(str(row[0]) for row in conn.execute("SELECT text FROM facts"))
    finally:
        conn.close()


def _run_session_end(project_dir: Path, session_id: str) -> subprocess.CompletedProcess[str]:
    envelope = {
        "event_type": "SessionEnd",
        "session_id": session_id,
        "working_dir": str(project_dir),
    }
    return subprocess.run(
        [sys.executable, str(_HOOKS / "on_session_end.py")],
        input=json.dumps(envelope),
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "PYTHONPATH": str(_SRC),
            config.DATA_ENV: str(config.data_dir()),
        },
        timeout=120,
    )


# --------------------------------------------------------------------------- #
# 1. global -> other project
# --------------------------------------------------------------------------- #


def _assert_global_reaches_b_but_not_a_facts(project_a: Path, project_b: Path) -> None:
    _write("project", str(project_a), "project a keeps this private")
    _write("global", "", "always use uv", "preference")

    block = inject.inherited_block(str(project_b))

    assert "always use uv" in block
    assert "project a keeps this private" not in block


def _merge_projects(monkeypatch: pytest.MonkeyPatch, project_a: Path) -> None:
    """Mutation: both project paths resolve to project A's store."""
    original_db = config.project_db
    key = config.project_key(str(project_a))
    monkeypatch.setattr(config, "project_db", lambda _wd: original_db(str(project_a)))
    monkeypatch.setattr(config, "project_key", lambda _wd: key)


def test_global_reaches_another_project_but_project_a_facts_do_not(
    project_a: Path, project_b: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assert_global_reaches_b_but_not_a_facts(project_a, project_b)

    _merge_projects(monkeypatch, project_a)
    with pytest.raises(AssertionError):
        _assert_global_reaches_b_but_not_a_facts(project_a, project_b)


# --------------------------------------------------------------------------- #
# 2. session -> new session
# --------------------------------------------------------------------------- #


def _assert_sessions_are_disjoint(session_a: str, session_b: str) -> None:
    _write("session", session_a, "a secret plan for session a", "preference")
    _write("session", session_b, "a separate plan for session b", "preference")

    assert _texts(config.session_db(session_a)) == ["a secret plan for session a"]
    assert _texts(config.session_db(session_b)) == ["a separate plan for session b"]


def _share_one_session_store(monkeypatch: pytest.MonkeyPatch, session_a: str) -> None:
    """Mutation: every session id opens session A's store.

    ``config.session_db`` is left alone on purpose: collapsing the filenames
    would be caught by the store's identity check, and the property under test
    is what a caller observes, not which layer refuses. Writing through A's
    store directly is the leak.
    """
    original = store.session
    monkeypatch.setattr(
        store,
        "session",
        lambda _sid, *, fail_open=False: original(session_a, fail_open=fail_open),
    )


def test_session_a_facts_are_invisible_in_session_b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_sessions_are_disjoint("session-a", "session-b")

    _share_one_session_store(monkeypatch, "session-a")
    with pytest.raises(AssertionError):
        _assert_sessions_are_disjoint("session-c", "session-d")


# --------------------------------------------------------------------------- #
# 3. session -> project timing
# --------------------------------------------------------------------------- #


def _assert_project_untouched_until_session_end(project_a: Path, session_id: str) -> None:
    _write("session", session_id, "locked sqlite over hindsight", "decision")

    assert _texts(config.project_db(str(project_a))) == []


def _redirect_session_writes_to_project(monkeypatch: pytest.MonkeyPatch, project_a: Path) -> None:
    """Mutation: the automatic write path lands in the project store (R1 broken)."""
    original_project = store.project
    monkeypatch.setattr(
        store,
        "session",
        lambda session_id, *, fail_open=False: original_project(
            str(project_a), fail_open=fail_open
        ),
    )


def test_session_facts_reach_the_project_only_after_session_end(
    project_a: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assert_project_untouched_until_session_end(project_a, "s1")

    _redirect_session_writes_to_project(monkeypatch, project_a)
    with pytest.raises(AssertionError):
        _assert_project_untouched_until_session_end(project_a, "s2")


def test_the_session_end_hook_is_what_promotes(project_a: Path) -> None:
    _write("session", "s1", "locked sqlite over hindsight", "decision")
    assert _texts(config.project_db(str(project_a))) == []

    completed = _run_session_end(project_a, "s1")

    assert completed.returncode == 0, completed.stderr
    assert _texts(config.project_db(str(project_a))) == ["locked sqlite over hindsight"]


# --------------------------------------------------------------------------- #
# 4. project -> global
# --------------------------------------------------------------------------- #


def _assert_one_session_preference_stays_project(project_a: Path, session_id: str) -> None:
    _write("session", session_id, f"prefer short commit messages {session_id}", "preference")

    consolidate.consolidate(str(project_a), session_id)

    assert _texts(config.global_db()) == []
    assert _texts(config.project_db(str(project_a))) == [
        f"prefer short commit messages {session_id}"
    ]


def test_a_one_session_preference_stays_in_the_project_store(
    project_a: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assert_one_session_preference_stays_project(project_a, "s1")

    # Mutation: drop the >= 2 sessions requirement.
    monkeypatch.setattr(consolidate, "_PROMOTION_THRESHOLD", 1)
    with pytest.raises(AssertionError):
        _assert_one_session_preference_stays_project(project_a, "s2")


def test_a_preference_seen_in_two_sessions_reaches_global(project_a: Path) -> None:
    """R2: ``kind = preference`` **and** >= 2 distinct sessions."""
    _write("session", "s1", "prefer short commit messages", "preference")
    _write("session", "s2", "prefer short commit messages", "preference")

    consolidate.consolidate(str(project_a), "s1")
    consolidate.consolidate(str(project_a), "s2")

    assert _texts(config.global_db()) == ["prefer short commit messages"]


def test_a_non_preference_seen_in_two_sessions_never_reaches_global(project_a: Path) -> None:
    _write("session", "s1", "locked sqlite over hindsight", "decision")
    _write("session", "s2", "locked sqlite over hindsight", "decision")

    consolidate.consolidate(str(project_a), "s1")
    consolidate.consolidate(str(project_a), "s2")

    assert _texts(config.global_db()) == []


# --------------------------------------------------------------------------- #
# 5. concurrent sessions
# --------------------------------------------------------------------------- #


def _assert_concurrent_sessions_disjoint(project_a: Path, tag: str = "") -> None:
    session_one = f"s1{tag}"
    session_two = f"s2{tag}"

    def write_session(session_id: str) -> None:
        _write("session", session_id, f"a fact from {session_id}", "fact")
        for index in range(5):
            _write("project", str(project_a), f"shared fact {index} from {session_id}", "fact")
        consolidate.consolidate(str(project_a), session_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(write_session, name) for name in (session_one, session_two)]
        for future in futures:
            future.result(timeout=120)

    assert f"a fact from {session_one}" in _texts(config.session_db(session_one))
    assert f"a fact from {session_two}" not in _texts(config.session_db(session_one))


def test_two_sessions_share_the_project_store_without_lock_errors(
    project_a: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two sessions writing the shared project store must not collide."""
    _assert_concurrent_sessions_disjoint(project_a)

    _share_one_session_store(monkeypatch, "s1")
    with pytest.raises(AssertionError):
        _assert_concurrent_sessions_disjoint(project_a, tag="-mutated")


def test_concurrent_project_writes_succeed_for_both_sessions(project_a: Path) -> None:
    """The shared project store is the contended resource; both writers win."""

    def write(session_id: str) -> None:
        for index in range(10):
            _write("project", str(project_a), f"{session_id} row {index}", "fact")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(write, name) for name in ("s1", "s2")]
        for future in futures:
            future.result(timeout=120)

    stored = _texts(config.project_db(str(project_a)))
    assert len(stored) == 20, stored
