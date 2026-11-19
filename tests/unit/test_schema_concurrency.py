"""Cross-process guarantees of ``tiered_memory.schema`` (plan §4, §8).

The module's central claim is that store creation is atomic, and that a racer
which loses or fails never damages the store another process is building. No
in-process assertion can prove that: the failure modes are about who holds a
file descriptor and who unlinks a path. Every test here therefore runs real
interpreters against real files.

Three shapes of test:

* A **real race**: processes start behind a file barrier and all open the same
  brand-new path, then each commits one fact. All must succeed and every fact
  must be present at the end. This is the only test of the atomicity claim the
  module docstring makes.
* A **forced interleaving**: cleanup only runs when an open fails, and no
  natural error lands at the interesting moment. These tests inject an
  exception (or a pause) at a named seam to drive the production cleanup path
  deterministically. The assertion is always about the production behaviour
  afterwards, never about the seam.
* A **held lock**: another process holds ``BEGIN EXCLUSIVE`` while this one
  opens, which is what a second session on the same store looks like.

Rounds are few, barriers small and timeouts generous, so the file stays fast on
CI while never hanging on a stuck child.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import tiered_memory.schema as schema

#: Directory holding the package, so a child interpreter imports the same code
#: under test even when the project is not installed into ``sys.path``.
_SRC_DIR = Path(schema.__file__).resolve().parents[1]

#: Ceiling for a single child. A healthy run finishes in well under a second.
_CHILD_TIMEOUT = 60.0

_INSERT_FACT = """
INSERT INTO facts (content_hash, subject_key, text, kind, tier, scope_key,
                   created_at, updated_at, last_seen_at)
VALUES (?, ?, ?, 'fact', 'global', 'global', 't', 't', 't')
"""


def _child_env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(_SRC_DIR)}


def _run_child(
    code: str, *args: str, cwd: Path, timeout: float = _CHILD_TIMEOUT
) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter that can import this package."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=_child_env(),
    )


def _spawn(code: str, *args: str, cwd: Path) -> subprocess.Popen[str]:
    """Start ``code`` in a fresh interpreter without waiting for it."""
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_env(),
    )


def _finish(process: subprocess.Popen[str]) -> tuple[int, str, str]:
    out, err = process.communicate(timeout=_CHILD_TIMEOUT)
    return process.returncode, out, err


def _wait_for(path: Path, *, what: str, timeout: float = 30.0) -> None:
    """Poll until ``path`` appears; the children coordinate through flag files."""
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {what} ({path})")
        time.sleep(0.005)


def _failures(results: list[tuple[int, str, str]]) -> list[tuple[int, str]]:
    return [(code, err.strip()[-400:]) for code, _out, err in results if code != 0]


# --------------------------------------------------------------------------- #
# the creation race is safe for every racer
# --------------------------------------------------------------------------- #

#: One racer: wait until every worker has reached the barrier, then open the
#: shared path, commit a fact private to this worker, and close.
_RACER = """
import os
import sys
import time

import tiered_memory.schema as schema

path, barrier, tag, racers = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
open(os.path.join(barrier, tag), "w").close()
deadline = time.monotonic() + 30
while len(os.listdir(barrier)) < racers:
    if time.monotonic() > deadline:
        raise SystemExit("barrier timed out")
    time.sleep(0.001)

conn = schema.open_store(path, tier="global", scope_key="global")
conn.execute(
    "INSERT INTO facts (content_hash, subject_key, text, kind, tier, scope_key,"
    " created_at, updated_at, last_seen_at) VALUES (?, ?, ?, 'fact', 'global',"
    " 'global', 't', 't', 't')",
    (f"hash-{tag}", f"subject-{tag}", f"fact-{tag}"),
)
conn.commit()
conn.close()
print("ok")
"""


@pytest.mark.parametrize("round_index", [0, 1, 2])
def test_creation_race_lets_every_racer_commit_facts(tmp_path: Path, round_index: int) -> None:
    """Every racer on a new path must succeed, and no committed fact may be lost.

    ``BEGIN IMMEDIATE`` serialises the bootstrap and a loser adopts the winner's
    store. The facts are asserted, not just the exit codes, because losing a
    write is precisely the regression this guards.
    """
    racers = 8
    path = tmp_path / f"race-{round_index}.db"
    barrier = tmp_path / f"barrier-{round_index}"
    barrier.mkdir()

    processes = [
        _spawn(_RACER, str(path), str(barrier), str(index), str(racers), cwd=tmp_path)
        for index in range(racers)
    ]
    results = [_finish(process) for process in processes]

    assert _failures(results) == [], f"a racer failed: {_failures(results)}"

    probe = sqlite3.connect(str(path), timeout=10.0)
    try:
        texts = sorted(row[0] for row in probe.execute("SELECT text FROM facts").fetchall())
        metadata = dict(probe.execute("SELECT key, value FROM meta").fetchall())
    finally:
        probe.close()

    assert texts == sorted(f"fact-{index}" for index in range(racers))
    assert metadata["schema_version"] == str(schema.SCHEMA_VERSION)
    assert metadata["tier"] == "global"


# --------------------------------------------------------------------------- #
# a failed loser must not damage the winner's store
# --------------------------------------------------------------------------- #

#: The loser creates the file, pauses just before writing the schema, then fails
#: once the winner is committed to using that same file.
_LOSER = """
import pathlib
import sys
import time

import tiered_memory.schema as schema

target, flags = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
flags.mkdir(exist_ok=True)


def failing_bootstrap(*args, **kwargs):
    (flags / "loser_ready").write_text("ready")
    deadline = time.monotonic() + 30
    while not (flags / "winner_ready").exists():
        if time.monotonic() > deadline:
            raise SystemExit("winner never arrived")
        time.sleep(0.005)
    raise RuntimeError("injected loser failure")


schema._bootstrap = failing_bootstrap
try:
    schema.open_store(target, tier="global", scope_key="global")
finally:
    # Signal only after open_store's own cleanup path has run.
    (flags / "loser_done").write_text("done")
"""

#: The winner opens while the loser is paused, waits for the loser to finish and
#: clean up, and only then bootstraps and commits. If the loser unlinked the
#: file, this commit cannot land in the store the path names.
_WINNER = """
import pathlib
import sys
import time

import tiered_memory.schema as schema

target, flags = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
deadline = time.monotonic() + 30
while not (flags / "loser_ready").exists():
    if time.monotonic() > deadline:
        raise SystemExit("loser never started")
    time.sleep(0.005)

original = schema._load_sqlite_vec


def pausing_load(conn):
    (flags / "winner_ready").write_text("ready")
    deadline = time.monotonic() + 30
    while not (flags / "loser_done").exists():
        if time.monotonic() > deadline:
            raise SystemExit("loser never finished")
        time.sleep(0.005)
    return original(conn)


schema._load_sqlite_vec = pausing_load
conn = schema.open_store(target, tier="global", scope_key="global")
conn.execute(
    "INSERT INTO facts (content_hash, subject_key, text, kind, tier, scope_key,"
    " created_at, updated_at, last_seen_at) VALUES"
    " ('h', 's', 'the winner fact', 'fact', 'global', 'global', 't', 't', 't')"
)
conn.commit()
conn.close()
print("winner committed")
"""


def test_failed_loser_never_deletes_the_winners_store(tmp_path: Path) -> None:
    """A loser's cleanup must leave the file the winner is using alone.

    Only the loser created the path, so an ownership test says "mine to delete";
    that is exactly why ownership is not enough. The file is the rendezvous the
    winner already opened, and unlinking it strands the winner's commit in an
    orphaned inode (visible as ``disk I/O error`` here, or as a vanished store).
    """
    target = tmp_path / "store.db"
    flags = tmp_path / "flags"

    loser = _spawn(_LOSER, str(target), str(flags), cwd=tmp_path)
    winner = _spawn(_WINNER, str(target), str(flags), cwd=tmp_path)
    loser_result = _finish(loser)
    winner_result = _finish(winner)

    assert loser_result[0] != 0, "the loser was supposed to fail"
    assert "injected loser failure" in loser_result[2]
    assert winner_result[0] == 0, f"the winner failed: {winner_result[2].strip()[-600:]}"

    assert target.exists(), "the loser's cleanup deleted the file the winner used"
    probe = sqlite3.connect(str(target), timeout=10.0)
    try:
        texts = [row[0] for row in probe.execute("SELECT text FROM facts").fetchall()]
    finally:
        probe.close()
    assert texts == ["the winner fact"]


def test_failed_open_on_a_preexisting_store_never_deletes_it(tmp_path: Path) -> None:
    """A failed open of an existing store must leave every byte and fact in place."""
    path = tmp_path / "existing.db"
    seed = schema.open_store(path, tier="global", scope_key="global")
    try:
        seed.execute(_INSERT_FACT, ("h", "s", "a fact that must survive"))
        seed.commit()
    finally:
        seed.close()

    result = _run_child(
        """
        import sys

        import tiered_memory.schema as schema

        def failing_adopt(*args, **kwargs):
            raise RuntimeError("injected failure during adopt")

        schema._adopt_existing = failing_adopt
        schema.open_store(sys.argv[1], tier="global", scope_key="global")
        """,
        str(path),
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "injected failure during adopt" in result.stderr
    assert path.is_file(), "a failed open deleted a pre-existing store"

    probe = sqlite3.connect(str(path))
    try:
        texts = [row[0] for row in probe.execute("SELECT text FROM facts").fetchall()]
        version = probe.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    finally:
        probe.close()
    assert texts == ["a fact that must survive"]
    assert version == (str(schema.SCHEMA_VERSION),)


# --------------------------------------------------------------------------- #
# a repeated identical call is idempotent
# --------------------------------------------------------------------------- #


def test_non_string_scope_key_is_rejected_before_it_reaches_the_store(
    tmp_path: Path,
) -> None:
    """``scope_key`` is validated like ``tier``, so the same call is idempotent.

    A non-str would be stored through SQLite's TEXT affinity; the next identical
    call would then compare its ``int`` argument with the ``'5'`` it wrote and
    raise ``StoreIdentityMismatch``. Rejecting it up front means every call that
    succeeds can be repeated, and nothing is created for a call that failed.
    """
    path = tmp_path / "typed.db"

    with pytest.raises(ValueError) as caught:
        schema.open_store(path, tier="global", scope_key=5)  # type: ignore[arg-type]
    assert "scope_key" in str(caught.value)
    assert not path.exists(), "a refused open must not create the store"

    first = schema.open_store(path, tier="global", scope_key="5")
    try:
        assert schema.read_meta(first)["scope_key"] == "5"
    finally:
        first.close()

    second = schema.open_store(path, tier="global", scope_key="5")
    try:
        assert schema.read_meta(second)["scope_key"] == "5"
    finally:
        second.close()


# --------------------------------------------------------------------------- #
# a held lock degrades the open instead of raising
# --------------------------------------------------------------------------- #

#: Hold ``BEGIN EXCLUSIVE`` on a rollback-journal store, the way a second
#: session or a maintenance tool would, and stay alive until killed.
_LOCK_HOLDER = """
import pathlib
import sqlite3
import sys
import time

conn = sqlite3.connect(sys.argv[1], isolation_level=None)
conn.execute("PRAGMA journal_mode = DELETE")
conn.execute("BEGIN EXCLUSIVE")
pathlib.Path(sys.argv[2]).write_text("locked")
time.sleep(120)
"""


def test_locked_rollback_journal_store_degrades_instead_of_raising(tmp_path: Path) -> None:
    """WAL stores already survive an exclusive holder; DELETE-journal ones must too.

    The open may return the connection or degrade, but it must not raise, and it
    must do so in bounded time rather than after one busy timeout per pragma.
    """
    path = tmp_path / "rollback.db"
    bootstrap = schema.open_store(path, tier="global", scope_key="global")
    bootstrap.close()  # sqlite3.Connection's context manager commits, it does not close
    raw = sqlite3.connect(str(path))
    try:
        raw.execute(_INSERT_FACT, ("h", "s", "a fact behind the lock"))
        raw.commit()
        assert raw.execute("PRAGMA journal_mode = DELETE").fetchone()[0].lower() == "delete"
    finally:
        raw.close()
    assert path.read_bytes()[18:20] == b"\x01\x01", "fixture must be rollback-journal"

    flag = tmp_path / "locked"
    holder = _spawn(_LOCK_HOLDER, str(path), str(flag), cwd=tmp_path)
    try:
        _wait_for(flag, what="the lock holder")
        started = time.monotonic()
        conn = schema.open_store(path, tier="global", scope_key="global")
        elapsed = time.monotonic() - started
        try:
            assert elapsed < schema.BUSY_TIMEOUT_SECONDS, (
                f"a locked open took {elapsed:.2f}s; it must degrade promptly"
            )
            assert int(conn.execute("PRAGMA foreign_keys").fetchone()[0]) == 1
        finally:
            conn.close()
    finally:
        holder.kill()
        holder.communicate(timeout=_CHILD_TIMEOUT)

    # The lock is gone: the same path opens normally and still holds the fact.
    conn = schema.open_store(path, tier="global", scope_key="global")
    try:
        assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()
