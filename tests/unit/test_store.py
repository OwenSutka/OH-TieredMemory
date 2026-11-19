"""Behaviour of ``tiered_memory.store`` (plan §2.2.1, Step 3 / TM-03).

This module's contract is about paths and locks, so every test uses real files
in a real plugin-data directory and real SQLite connections: nothing here can be
proved by a mock.

The claims under test:

* tier resolution reuses ``config`` (so two symlinked paths to one project are
  one store, and the hash is never duplicated);
* every tier entry is a ``BEGIN IMMEDIATE`` transaction under the schema's 5 s
  busy timeout, committing on a clean exit, rolling back on exception, and
  closing the connection either way;
* three tiers can be open at once and two sessions can share a project store
  without ``database is locked``;
* a hook that cannot take the lock does not raise: it logs a warning and yields
  ``HOOK_SKIPPED`` so the caller continues and the process exits 0 (plan R7).
"""

from __future__ import annotations

import functools
import logging
import os
import re
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import NamedTuple

import pytest

import tiered_memory.config as config
import tiered_memory.schema as schema
import tiered_memory.store as store
from tiered_memory.config import ConfigError

#: Directory holding the package, so a child interpreter imports the same code
#: under test even when the project is not installed into ``sys.path``.
_SRC_DIR = Path(store.__file__).resolve().parents[1]

_INSERT_FACT = """
INSERT INTO facts (content_hash, subject_key, text, kind, tier, scope_key,
                   created_at, updated_at, last_seen_at)
VALUES (?, ?, ?, 'fact', ?, ?, 't', 't', 't')
"""


class TierEntry(NamedTuple):
    """One tier's entry point plus the identity the store must record."""

    name: str
    tier: str
    scope_key: str
    path: Path
    open: Callable[..., AbstractContextManager[sqlite3.Connection | None]]


def _make_entry(name: str, project_dir: Path, session_id: str = "s1") -> TierEntry:
    if name == "global":
        return TierEntry(name, "global", "global", config.global_db(), store.global_store)
    if name == "project":
        key = config.project_key(str(project_dir))
        return TierEntry(
            name,
            "project",
            key,
            config.project_db(str(project_dir)),
            functools.partial(store.project, str(project_dir)),
        )
    return TierEntry(
        name,
        "session",
        session_id,
        config.session_db(session_id),
        functools.partial(store.session, session_id),
    )


def _insert(conn: sqlite3.Connection, text: str, *, tier: str, scope_key: str) -> None:
    conn.execute(
        _INSERT_FACT,
        (f"hash:{text}", f"subject:{text}", text, tier, scope_key),
    )


def _texts_on(conn: sqlite3.Connection) -> list[str]:
    return sorted(str(row[0]) for row in conn.execute("SELECT text FROM facts"))


def _texts(path: Path) -> list[str]:
    conn = sqlite3.connect(str(path), timeout=10.0)
    try:
        return _texts_on(conn)
    finally:
        conn.close()


@pytest.fixture()
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A plugin data root that the store must create the tiers directory in."""
    root = tmp_path / "plugin-data"
    monkeypatch.setenv(config.DATA_ENV, str(root))
    return root


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    path = tmp_path / "project"
    path.mkdir()
    return path


@pytest.fixture(params=["global", "project", "session"])
def entry(request: pytest.FixtureRequest, data_root: Path, project_dir: Path) -> TierEntry:
    return _make_entry(str(request.param), project_dir)


# --------------------------------------------------------------------------- #
# tier resolution and identity
# --------------------------------------------------------------------------- #


def test_tier_identity_is_recorded_and_the_file_lands_at_the_config_path(
    entry: TierEntry,
) -> None:
    """Every tier opens the file ``config`` names and records its identity.

    The path is taken from ``config`` rather than rebuilt here: duplicating the
    hash is the drift this test exists to prevent.
    """
    with entry.open() as conn:
        assert isinstance(conn, sqlite3.Connection)
        meta = schema.read_meta(conn)

    assert entry.path.is_file(), f"{entry.name} store was not created at {entry.path}"
    assert meta["tier"] == entry.tier
    assert meta["scope_key"] == entry.scope_key


def test_project_store_filename_is_the_config_project_key(
    data_root: Path, project_dir: Path
) -> None:
    with store.project(str(project_dir)):
        pass

    db = config.project_db(str(project_dir))
    assert db.is_file()
    assert re.fullmatch(r"project_[0-9a-f]{12}\.db", db.name), db.name


def test_session_store_filename_is_the_session_id(data_root: Path) -> None:
    with store.session("abc-123"):
        pass

    assert config.session_db("abc-123").is_file()


def test_two_symlinked_paths_to_one_project_resolve_to_one_store(
    data_root: Path, tmp_path: Path
) -> None:
    """plan §2.2.1: symlinks must not fork a project store.

    Writing through the real path and reading through the symlink is the
    observable form of "one store": a store keyed on the unresolved path would
    create a second file and see nothing.
    """
    real = tmp_path / "real-project"
    real.mkdir()
    link = tmp_path / "linked-project"
    link.symlink_to(real, target_is_directory=True)

    key = config.project_key(str(real))
    assert config.project_key(str(link)) == key

    with store.project(str(real)) as conn:
        _insert(conn, "reached through both paths", tier="project", scope_key=key)

    with store.project(str(link)) as conn:
        assert _texts_on(conn) == ["reached through both paths"]

    project_files = sorted(path.name for path in config.tiers_dir().glob("project_*.db"))
    assert project_files == [config.project_db(str(real)).name]


def test_distinct_projects_resolve_to_distinct_stores(data_root: Path, tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    with store.project(str(first)) as conn:
        _insert(conn, "first only", tier="project", scope_key=config.project_key(str(first)))

    with store.project(str(second)) as conn:
        assert _texts_on(conn) == []
        _insert(conn, "second only", tier="project", scope_key=config.project_key(str(second)))

    assert config.project_db(str(first)) != config.project_db(str(second))


def test_tiers_are_independent_stores(data_root: Path, project_dir: Path) -> None:
    """A session fact must not leak into the project or global store."""
    with store.session("s1") as conn:
        _insert(conn, "session only", tier="session", scope_key="s1")

    with store.project(str(project_dir)) as conn:
        assert _texts_on(conn) == []
    with store.global_store() as conn:
        assert _texts_on(conn) == []


def test_missing_tiers_directory_is_created(data_root: Path) -> None:
    assert not data_root.exists()

    with store.global_store() as conn:
        _insert(conn, "first", tier="global", scope_key="global")

    assert config.tiers_dir().is_dir()
    assert config.global_db().is_file()


# --------------------------------------------------------------------------- #
# transaction behaviour shared by every tier
# --------------------------------------------------------------------------- #


def test_clean_exit_commits(entry: TierEntry) -> None:
    with entry.open() as conn:
        _insert(conn, "kept", tier=entry.tier, scope_key=entry.scope_key)

    assert _texts(entry.path) == ["kept"]


def test_exception_rolls_back_and_propagates(entry: TierEntry) -> None:
    with pytest.raises(RuntimeError, match="boom"), entry.open() as conn:
        _insert(conn, "doomed", tier=entry.tier, scope_key=entry.scope_key)
        raise RuntimeError("boom")

    assert _texts(entry.path) == []


def test_connection_is_closed_after_the_block(entry: TierEntry) -> None:
    with entry.open() as conn:
        assert conn is not None

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_transaction_is_immediate_with_a_five_second_busy_timeout(entry: TierEntry) -> None:
    """BEGIN IMMEDIATE takes the write lock at ``__enter__``, not at first write.

    A rival with a 50 ms timeout must fail immediately: that is only true when
    the lock is already held, i.e. the block began with IMMEDIATE.
    """
    with entry.open() as conn:
        assert conn is not None
        assert conn.isolation_level is None, "the transaction must be driven explicitly"
        assert conn.in_transaction
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000

        rival = sqlite3.connect(str(entry.path), isolation_level=None, timeout=0.05)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                rival.execute("BEGIN IMMEDIATE")
        finally:
            rival.close()


def test_successive_entries_use_fresh_connections(data_root: Path) -> None:
    """No module-level connection or state may survive between entries."""
    with store.session("s1") as first:
        pass
    with store.session("s1") as second:
        _insert(second, "written by the second entry", tier="session", scope_key="s1")

    assert first is not second
    assert _texts(config.session_db("s1")) == ["written by the second entry"]


# --------------------------------------------------------------------------- #
# concurrency
# --------------------------------------------------------------------------- #


def _run_threads(actions: list[Callable[[], None]]) -> list[BaseException]:
    """Run ``actions`` concurrently and collect failures instead of swallowing."""
    failures: list[BaseException] = []

    def worker(action: Callable[[], None]) -> None:
        try:
            action()
        except BaseException as exc:  # noqa: BLE001 - collected and asserted below
            failures.append(exc)

    threads = [threading.Thread(target=worker, args=(action,)) for action in actions]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not any(thread.is_alive() for thread in threads), "a worker hung"
    return failures


def test_three_tiers_open_concurrently_without_lock_errors(
    data_root: Path, project_dir: Path
) -> None:
    """All three tiers held open at once must not deadlock or raise."""
    barrier = threading.Barrier(3)
    key = config.project_key(str(project_dir))
    plan: list[tuple[str, str, str, Callable[[], AbstractContextManager[object]]]] = [
        ("g", "global", "global", store.global_store),
        ("p", "project", key, functools.partial(store.project, str(project_dir))),
        ("s", "session", "s1", functools.partial(store.session, "s1")),
    ]

    def action(
        text: str,
        tier: str,
        scope: str,
        opener: Callable[[], AbstractContextManager[object]],
    ) -> Callable[[], None]:
        def run() -> None:
            with opener() as conn:
                assert conn is not None
                barrier.wait(timeout=30)
                _insert(conn, text, tier=tier, scope_key=scope)

        return run

    failures = _run_threads(
        [action(text, tier, scope, opener) for text, tier, scope, opener in plan]
    )

    assert failures == [], failures
    assert _texts(config.global_db()) == ["g"]
    assert _texts(config.project_db(str(project_dir))) == ["p"]
    assert _texts(config.session_db("s1")) == ["s"]


def test_two_sessions_share_the_project_store_without_lock_errors(
    data_root: Path, project_dir: Path
) -> None:
    """plan §8: two sessions write the shared project store, no lock errors.

    The second writer must wait out the first (inside the 5 s budget) rather
    than surface ``database is locked``, and both facts must survive.
    """
    barrier = threading.Barrier(2)
    key = config.project_key(str(project_dir))

    def writer(session_id: str) -> Callable[[], None]:
        def run() -> None:
            barrier.wait(timeout=30)
            with store.project(str(project_dir)) as conn:
                assert conn is not None
                _insert(conn, session_id, tier="project", scope_key=key)
                time.sleep(0.2)

        return run

    failures = _run_threads([writer(session_id) for session_id in ("a", "b")])

    assert failures == [], failures
    assert _texts(config.project_db(str(project_dir))) == ["a", "b"]


# --------------------------------------------------------------------------- #
# fail-open for hooks (plan R7)
# --------------------------------------------------------------------------- #

#: One tier, opened in its own process behind a file barrier. Hooks run as
#: separate processes, so cross-process contention is the real shape.
_TIER_CHILD = """
import os
import pathlib
import sys
import time

from tiered_memory import store

tier, barrier, tag, workers, project_dir = sys.argv[1:6]
pathlib.Path(barrier, tag).write_text("ready")
deadline = time.monotonic() + 30
while len(os.listdir(barrier)) < int(workers):
    if time.monotonic() > deadline:
        raise SystemExit("barrier timed out")
    time.sleep(0.001)

opener = {
    "global": store.global_store,
    "project": lambda: store.project(project_dir),
    "session": lambda: store.session(tag + "-session"),
}[tier]
with opener() as conn:
    value = "global" if tier == "global" else tag
    conn.execute(
        "INSERT INTO facts (content_hash, subject_key, text, kind, tier, scope_key,"
        " created_at, updated_at, last_seen_at)"
        " VALUES (?, ?, ?, 'fact', ?, ?, 't', 't', 't')",
        ("hash-" + tag, "subject-" + tag, "fact-" + tag, tier, value),
    )
print("ok")
"""


def test_three_tiers_open_concurrently_in_separate_processes(
    data_root: Path, project_dir: Path
) -> None:
    """The acceptance bullet, as hooks really run it: three processes, three tiers.

    Every child must exit 0 and land its fact; a ``database is locked`` here is
    the failure the 5 s busy timeout exists to prevent.
    """
    barrier = project_dir.parent / "tier-barrier"
    barrier.mkdir()
    tiers = ("global", "project", "session")
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                textwrap.dedent(_TIER_CHILD),
                tier,
                str(barrier),
                tier,
                str(len(tiers)),
                str(project_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONPATH": str(_SRC_DIR)},
        )
        for tier in tiers
    ]
    results = [(process, process.communicate(timeout=120)) for process in processes]

    failures = [
        (process.returncode, err.strip()[-400:])
        for process, (_out, err) in results
        if process.returncode != 0
    ]
    assert failures == [], failures

    assert _texts(config.global_db()) == ["fact-global"]
    assert _texts(config.project_db(str(project_dir))) == ["fact-project"]
    assert _texts(config.session_db("session-session")) == ["fact-session"]


#: A child process standing in for a hook: it uses the fail-open entry, reports
#: what it got, and must exit 0 either way.
_HOOK_CHILD = """
import logging
import sys

from tiered_memory import store

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)


def handler() -> int:
    with store.session(sys.argv[1], fail_open=True) as conn:
        if conn is store.HOOK_SKIPPED:
            print("skipped")
            return 0
        conn.execute(
            "INSERT INTO facts (content_hash, subject_key, text, kind, tier, scope_key,"
            " created_at, updated_at, last_seen_at)"
            " VALUES ('hook', 'hook', 'hook', 'fact', 'session', ?, 't', 't', 't')",
            (sys.argv[1],),
        )
    print("stored")
    return 0


sys.exit(handler())
"""


def test_fail_open_yields_the_sentinel_and_warns_when_the_lock_is_held(
    data_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A contended hook entry must not raise: warning + documented sentinel."""
    with store.session("s1") as held:
        _insert(held, "held", tier="session", scope_key="s1")
        with (
            caplog.at_level(logging.WARNING, logger="tiered_memory.store"),
            store.session("s1", fail_open=True) as conn,
        ):
            assert conn is store.HOOK_SKIPPED

    assert [record for record in caplog.records if record.levelno == logging.WARNING], caplog.text


def test_hook_process_exits_zero_and_logs_when_the_lock_is_held(data_root: Path) -> None:
    """Acceptance: "a hook that cannot take the lock logs and returns 0"."""
    with store.session("hook-session") as held:
        _insert(held, "held", tier="session", scope_key="hook-session")
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(_HOOK_CHILD), "hook-session"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            env={**os.environ, "PYTHONPATH": str(_SRC_DIR)},
        )

    assert result.returncode == 0, result.stderr
    assert "skipped" in result.stdout
    assert "lock" in result.stderr.lower(), result.stderr


def test_fail_open_does_not_hide_configuration_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail-open is about lock contention, not about a missing data directory."""
    monkeypatch.delenv(config.DATA_ENV, raising=False)

    with pytest.raises(ConfigError), store.global_store(fail_open=True):
        pytest.fail("the block must not run without a configured data directory")


def test_fail_open_propagates_errors_that_are_not_lock_contention(
    data_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A corrupt store is a defect, not contention: it must still raise.

    Catching every ``sqlite3.Error`` here would turn a broken store into silent
    data loss, which is exactly what fail-open must not mean.
    """
    path = config.session_db("corrupt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is not a sqlite database" * 64)

    with (
        caplog.at_level(logging.WARNING, logger="tiered_memory.store"),
        pytest.raises(sqlite3.DatabaseError),
        store.session("corrupt", fail_open=True),
    ):
        pytest.fail("the block must not run against a corrupt store")


# --------------------------------------------------------------------------- #
# caller-supplied path components are validated (traversal)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "session_id",
    [
        "x/../../../../tmp/outside/pwned",
        "../outside",
        "..",
        ".",
        "a/b",
        "a\\b",
        "",
        "sess\x00id",
        "s" * 129,
    ],
)
def test_traversal_session_ids_are_refused_and_create_nothing(
    data_root: Path, session_id: str
) -> None:
    """A session id becomes a filename, so it must be one safe component.

    ``config.session_db`` interpolates the id straight into
    ``tiers/session_<id>.db``; an id containing ``/`` or ``..`` would write
    outside ``PLUGIN_DATA``. It must raise ``ValueError`` before any directory or
    file exists, and nothing may be created outside ``tiers/``.
    """
    with pytest.raises(ValueError), store.session(session_id):
        pytest.fail("an unsafe session id must not open a store")

    assert not data_root.exists(), "nothing may be created for a refused session id"


def test_a_traversal_session_id_writes_no_file_outside_the_tiers_directory(
    data_root: Path, tmp_path: Path
) -> None:
    """Even with the data root present, a traversal id must not escape ``tiers/``."""
    data_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError), store.session("../outside/pwned"):
        pytest.fail("a traversal session id must not open a store")

    assert list(outside.iterdir()) == [], "a refusal must not write outside tiers/"
    assert not (data_root / "outside").exists()
    assert not config.tiers_dir().exists(), "no tiers directory may be created"


def test_plain_session_ids_still_open(data_root: Path) -> None:
    """The charset allows the ids that are actually used: hex, dashes, dots."""
    assert config.session_db("abc-123").name == "session_abc-123.db"
    assert config.session_db("dead.beef_00").name == "session_dead.beef_00.db"
    assert config.session_db("a" * 128).name == f"session_{'a' * 128}.db"


# --------------------------------------------------------------------------- #
# fail-open must not pay the full busy timeout under sustained contention
# --------------------------------------------------------------------------- #


def test_ten_fail_open_skips_under_contention_are_fast(data_root: Path) -> None:
    """Sustained contention must not cost 5 s per hook.

    A fail-open caller cannot write anyway, so waiting out the full busy
    timeout ten times (50 s) buys nothing. It must probe with a short timeout
    and give up, keeping ten sequential skips well under 10 s.
    """
    with store.session("s1") as held:
        _insert(held, "held", tier="session", scope_key="s1")

        started = time.monotonic()
        for _ in range(10):
            with store.session("s1", fail_open=True) as conn:
                assert conn is store.HOOK_SKIPPED
        elapsed = time.monotonic() - started

    assert elapsed < 10.0, f"10 fail-open skips took {elapsed:.2f}s under contention"


def test_a_blocking_caller_still_waits_for_the_lock(data_root: Path) -> None:
    """The fail-fast probe must not shorten the wait for a blocking caller.

    A writer released shortly after the blocking open starts must still be
    admitted: a fail-open-style 50 ms probe would instead surface "locked".
    """
    entered = threading.Event()

    def hold_briefly() -> None:
        with store.session("s1") as held:
            _insert(held, "held", tier="session", scope_key="s1")
            entered.set()
            time.sleep(0.3)

    holder = threading.Thread(target=hold_briefly)
    holder.start()
    try:
        assert entered.wait(timeout=10)
        # The holder keeps the lock for ~0.3 s: far beyond the fail-fast probe,
        # far inside the documented 5 s budget.
        with store.session("s1") as conn:
            assert conn is not store.HOOK_SKIPPED
            _insert(conn, "waited", tier="session", scope_key="s1")
    finally:
        holder.join(timeout=30)

    assert _texts(config.session_db("s1")) == ["held", "waited"]


# --------------------------------------------------------------------------- #
# an unusable data directory is a classified configuration error
# --------------------------------------------------------------------------- #


def test_tiers_path_that_is_a_regular_file_raises_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``mkdir`` on a path that is a regular file must surface as ``ConfigError``.

    ``Path.mkdir`` raises a bare ``FileExistsError``; a fail-open hook classifies
    only ``ConfigError``, so an unusable data directory has to be reported as
    one rather than escaping as an ``OSError`` no caller handles.
    """
    root = tmp_path / "plugin-data"
    root.mkdir()
    (root / "tiers").write_text("not a directory")
    monkeypatch.setenv(config.DATA_ENV, str(root))

    with pytest.raises(ConfigError), store.session("s1"):
        pytest.fail("the block must not run when tiers/ is a regular file")
