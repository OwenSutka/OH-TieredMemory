"""The hook entrypoints under ``dev.openhands/hooks`` (plan §3.4, R6, R7).

These are process-level tests: every hook is run as a child process with a JSON
envelope on stdin, because the contract they must satisfy is about the process
(exit code, stdout) and not about a function's return value.

The claims under test:

* the inject hook emits **nothing** -- no JSON, no blank block -- when both
  stores are empty, and emits project facts but never session facts otherwise;
* a hook whose handler raises still exits 0 (R7, fail-open);
* no hook imports the embedding module (R6), checked by inspecting the child's
  ``sys.modules`` after the run;
* ``hooks.json`` registers the three events and every command references
  ``${PLUGIN_ROOT}``, which is what the manifest gate enforces.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import tiered_memory.config as config
import tiered_memory.facts as facts
import tiered_memory.store as store

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOOKS = _REPO_ROOT / "dev.openhands" / "hooks"
_SRC = _REPO_ROOT / "src"


@pytest.fixture()
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "plugin-data"
    monkeypatch.setenv(config.DATA_ENV, str(root))
    return root


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    path = tmp_path / "project"
    path.mkdir()
    return path


def _run_hook(
    script: str,
    envelope: dict[str, object],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one hook as a child process with ``envelope`` on stdin.

    The child gets a minimal environment plus whatever the test's fixtures set:
    inheriting the whole of ``os.environ`` would let a stray variable on the
    developer's machine change what the hook does.
    """
    inherited = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            config.DATA_ENV,
            "OPENHANDS_PROJECT_DIR",
            "OPENHANDS_SESSION_ID",
            "OPENHANDS_EVENT_TYPE",
            "OPENHANDS_TOOL_NAME",
            "PLUGIN_DATA",
        }
    }
    child_env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "PYTHONPATH": str(_SRC),
        **inherited,
        **(env or {}),
    }
    return subprocess.run(
        [sys.executable, str(_HOOKS / script)],
        input=json.dumps(envelope),
        capture_output=True,
        text=True,
        env=child_env,
        timeout=120,
    )


def _envelope(project_dir: Path, **extra: object) -> dict[str, object]:
    return {
        "event_type": "UserPromptSubmit",
        "session_id": "s1",
        "working_dir": str(project_dir),
        **extra,
    }


def _project_fact(project_dir: Path, text: str, kind: str) -> None:
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, text, kind=kind, source_session="seed")


def _global_fact(text: str, kind: str) -> None:
    with store.global_store() as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, text, kind=kind, source_session="seed")


def _session_fact(session_id: str, text: str, kind: str) -> None:
    with store.session(session_id) as conn:
        assert isinstance(conn, sqlite3.Connection)
        facts.insert(conn, text, kind=kind, source_session=session_id)


# --------------------------------------------------------------------------- #
# (a) silent when empty
# --------------------------------------------------------------------------- #


def test_the_inject_hook_emits_nothing_when_both_stores_are_empty(
    project_dir: Path, data_root: Path
) -> None:
    completed = _run_hook("on_user_prompt.py", _envelope(project_dir))

    assert completed.returncode == 0
    assert completed.stdout == ""


def test_the_inject_hook_emits_nothing_without_a_working_directory() -> None:
    completed = _run_hook("on_user_prompt.py", {"event_type": "UserPromptSubmit"})

    assert completed.returncode == 0
    assert completed.stdout == ""


# --------------------------------------------------------------------------- #
# (b) project facts in, session facts out
# --------------------------------------------------------------------------- #


def test_the_inject_hook_emits_project_and_global_facts(project_dir: Path, data_root: Path) -> None:
    _project_fact(project_dir, "locked sqlite over hindsight", "decision")
    _global_fact("always use uv", "preference")

    completed = _run_hook("on_user_prompt.py", _envelope(project_dir))

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert set(payload) == {"additionalContext"}
    assert "locked sqlite over hindsight" in payload["additionalContext"]
    assert "always use uv" in payload["additionalContext"]


def test_the_inject_hook_never_emits_session_facts(project_dir: Path, data_root: Path) -> None:
    """Both delivery channels for the session id are set, as in a real run.

    ``hook_runtime`` prefers ``OPENHANDS_SESSION_ID`` over the envelope, so the
    child gets both: a mutation that reads the session tier through either
    channel must be caught here.
    """
    _project_fact(project_dir, "locked sqlite over hindsight", "decision")
    _session_fact("s1", "a session-only secret plan", "preference")

    completed = _run_hook(
        "on_user_prompt.py",
        {"event_type": "UserPromptSubmit"},
        env={"OPENHANDS_SESSION_ID": "s1", "OPENHANDS_PROJECT_DIR": str(project_dir)},
    )

    assert completed.returncode == 0
    assert "session-only secret plan" not in completed.stdout


# --------------------------------------------------------------------------- #
# (c) fail-open
# --------------------------------------------------------------------------- #


def test_a_hook_whose_handler_raises_still_exits_zero(
    project_dir: Path, data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R7: a broken hook must never block a session.

    ``TIERED_MEMORY_DATA`` is pointed at a regular file, which is not a usable
    data directory. The inject handler therefore raises ``ConfigError`` while
    trying to resolve the tier path, and the runtime must still exit 0.
    """
    broken = tmp_path / "not-a-directory"
    broken.write_text("", encoding="utf-8")

    completed = _run_hook(
        "on_user_prompt.py",
        _envelope(project_dir),
        env={"TIERED_MEMORY_DATA": str(broken)},
    )

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert "Traceback" not in completed.stderr


def test_a_hook_with_malformed_stdin_still_exits_zero(project_dir: Path, data_root: Path) -> None:
    child_env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "PYTHONPATH": str(_SRC)}
    completed = subprocess.run(
        [sys.executable, str(_HOOKS / "on_user_prompt.py")],
        input="not json at all",
        capture_output=True,
        text=True,
        env=child_env,
        timeout=120,
    )

    assert completed.returncode == 0


def test_a_hook_fails_open_even_though_the_event_could_block(
    project_dir: Path, tmp_path: Path
) -> None:
    """``UserPromptSubmit`` *may* block; this plugin never asks it to."""
    broken = tmp_path / "still-not-a-directory"
    broken.write_text("", encoding="utf-8")

    completed = _run_hook(
        "on_user_prompt.py",
        _envelope(project_dir),
        env={"TIERED_MEMORY_DATA": str(broken), "OPENHANDS_EVENT_TYPE": "UserPromptSubmit"},
    )

    assert completed.returncode == 0
    assert "deny" not in completed.stdout


# --------------------------------------------------------------------------- #
# (d) no embedding in the hook path
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("script", ["on_user_prompt.py", "on_post_tool.py", "on_session_end.py"])
def test_no_hook_imports_the_embedding_module(
    script: str, project_dir: Path, data_root: Path
) -> None:
    """R6: the 113 MB model must never be reachable from a hook.

    The probe imports the hook module in-process and asserts the embedding
    module stayed out of ``sys.modules``. It also asserts ``fastembed`` is
    absent, which is the dependency that would actually cost a download.
    """
    module = script.removesuffix(".py")
    probe = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('{module}', r'{_HOOKS / script}')\n"
        "assert spec is not None and spec.loader is not None\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "assert 'tiered_memory.embeddings' not in sys.modules, 'embeddings imported'\n"
        "assert 'fastembed' not in sys.modules, 'fastembed imported'\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "PYTHONPATH": str(_SRC),
            "TIERED_MEMORY_DATA": str(data_root),
        },
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr


# --------------------------------------------------------------------------- #
# post-tool extraction end to end
# --------------------------------------------------------------------------- #


def test_the_post_tool_hook_stores_into_the_session_tier_only(
    project_dir: Path, data_root: Path
) -> None:
    envelope = {
        "event_type": "PostToolUse",
        "session_id": "s1",
        "working_dir": str(project_dir),
        "tool_name": "terminal",
        "tool_input": {"command": "echo hi"},
        "message": "always use uv",
        "tool_response": "hi",
    }

    completed = _run_hook("on_post_tool.py", envelope)

    assert completed.returncode == 0
    with store.session("s1") as conn:
        assert isinstance(conn, sqlite3.Connection)
        texts = [str(row[0]) for row in conn.execute("SELECT text FROM facts")]
    assert "always use uv" in texts
    assert not config.project_db(str(project_dir)).exists()
    assert not config.global_db().exists()


def test_the_post_tool_hook_records_a_gotcha_after_a_failure_then_success(
    project_dir: Path, data_root: Path
) -> None:
    base = {
        "event_type": "PostToolUse",
        "session_id": "s1",
        "working_dir": str(project_dir),
        "tool_name": "terminal",
        "tool_input": {"command": "run"},
    }
    _run_hook("on_post_tool.py", {**base, "tool_response": "Error: no such module: vec0"})
    _run_hook("on_post_tool.py", {**base, "tool_response": "ok"})

    with store.session("s1") as conn:
        assert isinstance(conn, sqlite3.Connection)
        kinds = [str(row[0]) for row in conn.execute("SELECT kind FROM facts")]
    assert "gotcha" in kinds


# --------------------------------------------------------------------------- #
# session end
# --------------------------------------------------------------------------- #


def test_the_session_end_hook_promotes_into_the_project_store(
    project_dir: Path, data_root: Path
) -> None:
    _session_fact("s1", "locked sqlite over hindsight", "decision")

    completed = _run_hook(
        "on_session_end.py",
        {
            "event_type": "SessionEnd",
            "session_id": "s1",
            "working_dir": str(project_dir),
        },
    )

    assert completed.returncode == 0
    with store.project(str(project_dir)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        texts = [str(row[0]) for row in conn.execute("SELECT text FROM facts")]
    assert texts == ["locked sqlite over hindsight"]


def test_the_session_end_hook_is_fail_open_with_a_broken_data_root(
    project_dir: Path, tmp_path: Path
) -> None:
    broken = tmp_path / "file-not-dir"
    broken.write_text("", encoding="utf-8")

    completed = _run_hook(
        "on_session_end.py",
        {"event_type": "SessionEnd", "session_id": "s1", "working_dir": str(project_dir)},
        env={"TIERED_MEMORY_DATA": str(broken)},
    )

    assert completed.returncode == 0


# --------------------------------------------------------------------------- #
# hooks.json
# --------------------------------------------------------------------------- #


def test_hooks_json_registers_the_three_events() -> None:
    manifest = json.loads((_HOOKS / "hooks.json").read_text(encoding="utf-8"))

    assert set(manifest["hooks"]) == {"UserPromptSubmit", "PostToolUse", "SessionEnd"}


def test_every_hook_command_uses_plugin_root_and_names_a_real_script() -> None:
    manifest = json.loads((_HOOKS / "hooks.json").read_text(encoding="utf-8"))

    for event, groups in manifest["hooks"].items():
        for group in groups:
            for hook in group["hooks"]:
                command = hook["command"]
                assert "${PLUGIN_ROOT}" in command, event
                script = command.rsplit(" ", 1)[-1]
                assert (_HOOKS / script).is_file(), f"{event} names a missing script {script}"


def test_the_post_tool_hook_is_async() -> None:
    """§7: extraction must never add latency to a tool call."""
    manifest = json.loads((_HOOKS / "hooks.json").read_text(encoding="utf-8"))

    for group in manifest["hooks"]["PostToolUse"]:
        for hook in group["hooks"]:
            assert hook["async"] is True


def test_the_hook_launcher_is_executable() -> None:
    launcher = _REPO_ROOT / "bin" / "tiered-memory-hook"

    assert launcher.exists()
    assert launcher.stat().st_mode & 0o111


def test_the_launcher_runs_the_hook_scripts_the_manifest_names(
    project_dir: Path, data_root: Path
) -> None:
    """``hooks.json`` calls the launcher, not Python, so prove the launcher works."""
    launcher = _REPO_ROOT / "bin" / "tiered-memory-hook"
    completed = subprocess.run(
        [str(launcher), "on_user_prompt.py"],
        input=json.dumps(_envelope(project_dir)),
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "PLUGIN_ROOT": str(_REPO_ROOT),
            "TIERED_MEMORY_DATA": str(data_root),
            "TIERED_MEMORY_PYTHON": sys.executable,
        },
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""


def test_the_launcher_defaults_its_root_to_its_own_location(
    project_dir: Path, data_root: Path
) -> None:
    """With ``PLUGIN_ROOT`` unset, the launcher must still find the scripts."""
    launcher = _REPO_ROOT / "bin" / "tiered-memory-hook"
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PLUGIN_ROOT", "TIERED_MEMORY_PYTHON"}
    }
    env.update({"TIERED_MEMORY_DATA": str(data_root), "TIERED_MEMORY_PYTHON": sys.executable})
    completed = subprocess.run(
        [str(launcher), "on_user_prompt.py"],
        input=json.dumps(_envelope(project_dir)),
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr


def test_the_launcher_maps_plugin_data_onto_the_store_root(
    project_dir: Path, tmp_path: Path
) -> None:
    """The runner exports ``PLUGIN_DATA``; the store reads ``TIERED_MEMORY_DATA``.

    ``mcp.json`` performs this mapping for the server, so the launcher must
    perform it for the hooks -- otherwise they write to a different store than
    the one ``memory_*`` reads, and memory silently never round-trips.
    """
    launcher = _REPO_ROOT / "bin" / "tiered-memory-hook"
    plugin_data = tmp_path / "plugin-data"
    completed = subprocess.run(
        [str(launcher), "on_post_tool.py"],
        input=json.dumps(
            {
                "event_type": "PostToolUse",
                "session_id": "s1",
                "working_dir": str(project_dir),
                "tool_name": "terminal",
                "tool_input": {"command": "echo hi"},
                "message": "always use uv",
                "tool_response": "hi",
            }
        ),
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "PLUGIN_ROOT": str(_REPO_ROOT),
            "PLUGIN_DATA": str(plugin_data),
            "TIERED_MEMORY_PYTHON": sys.executable,
        },
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert (plugin_data / "tiers" / "session_s1.db").is_file()


def test_the_launcher_prefers_an_explicit_store_root_over_plugin_data(
    project_dir: Path, tmp_path: Path
) -> None:
    """An inherited ``TIERED_MEMORY_DATA`` wins, so tests and operators can pin it."""
    launcher = _REPO_ROOT / "bin" / "tiered-memory-hook"
    explicit = tmp_path / "explicit"
    ignored = tmp_path / "ignored"
    completed = subprocess.run(
        [str(launcher), "on_post_tool.py"],
        input=json.dumps(
            {
                "event_type": "PostToolUse",
                "session_id": "s1",
                "working_dir": str(project_dir),
                "tool_name": "terminal",
                "tool_input": {"command": "echo hi"},
                "message": "always use uv",
                "tool_response": "hi",
            }
        ),
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "PLUGIN_ROOT": str(_REPO_ROOT),
            "PLUGIN_DATA": str(ignored),
            "TIERED_MEMORY_DATA": str(explicit),
            "TIERED_MEMORY_PYTHON": sys.executable,
        },
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert (explicit / "tiers" / "session_s1.db").is_file()
    # The runner's log home is still PLUGIN_DATA; only the store root moves.
    assert not (ignored / "tiers").exists()


def test_a_locked_store_is_logged_and_the_hook_still_exits_zero(tmp_path: Path) -> None:
    """Plan Step 3: a hook that cannot take the lock "logs and returns 0".

    Before this test the hook exited 0 but left no trace: store.py's logging
    warning goes to stderr, which the runner discards, and hook_runtime only
    writes its line after the handler returns, so a skip looked identical to a
    call that simply had nothing to extract.
    """
    import sqlite3

    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "PYTHONPATH": str(_SRC),
        config.DATA_ENV: str(tmp_path / "data"),
        "PLUGIN_DATA": str(tmp_path / "data"),
        "OPENHANDS_SESSION_ID": "locked-session",
        "OPENHANDS_EVENT_TYPE": "PostToolUse",
    }

    # Populate the session store, then hold its write lock from another process.
    seed = subprocess.run(
        [sys.executable, str(_HOOKS / "on_post_tool.py")],
        input=json.dumps(
            {
                "tool_name": "terminal",
                "tool_input": {"command": "echo we decided to use sqlite"},
                "tool_response": "ok",
            }
        ),
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert seed.returncode == 0, seed.stderr

    database = Path(env["TIERED_MEMORY_DATA"]) / "tiers" / "session_locked-session.db"
    holder = sqlite3.connect(database, timeout=1)
    holder.execute("BEGIN IMMEDIATE")  # hold the write lock
    try:
        blocked = subprocess.run(
            [sys.executable, str(_HOOKS / "on_post_tool.py")],
            input=json.dumps(
                {
                    "tool_name": "terminal",
                    "tool_input": {"command": "echo always use uv, never pip"},
                    "tool_response": "ok",
                }
            ),
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
            check=False,
        )
    finally:
        holder.rollback()
        holder.close()

    assert blocked.returncode == 0, "a locked store must never fail the hook"

    trail = (Path(env["PLUGIN_DATA"]) / "logs" / "hooks.jsonl").read_text(encoding="utf-8")
    skips = [json.loads(line) for line in trail.splitlines() if "skipped" in line]
    assert skips, f"the skip left no trace in the hook log:\n{trail}"
    assert skips[-1]["skipped"] == "lock_contention"
    assert skips[-1]["session"] == "locked-session"
