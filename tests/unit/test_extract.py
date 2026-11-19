"""Behaviour of ``tiered_memory.extract`` (plan §2.2.3, Step 10 / TM-10).

Each §2.2.3 signal row has a positive and a negative fixture, because a rule
that matches everything extracts noise and a rule that matches nothing misses the
only fact that mattered:

* preference, decision, architecture, gotcha and todo each fire on their example;
* a sentence with no signal produces no candidate;
* binary content, content above 4 KB and credential-shaped text are refused;
* at most three facts per tool call and twenty per session, dropping the least
  confident rather than queueing;
* extraction never touches the embedding model, so it is safe on a hot path.

Every test drives the real rules over real strings; nothing is mocked.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import tiered_memory.config as config
import tiered_memory.extract as extract
import tiered_memory.store as store


@pytest.fixture(autouse=True)
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate every test in its own plugin data root."""
    root = tmp_path / "plugin-data"
    monkeypatch.setenv(config.DATA_ENV, str(root))
    return root


def _kinds(candidates: list[extract.Candidate]) -> list[str]:
    return [candidate.kind for candidate in candidates]


def _texts(candidates: list[extract.Candidate]) -> list[str]:
    return [candidate.text for candidate in candidates]


# --------------------------------------------------------------------------- #
# preference
# --------------------------------------------------------------------------- #


def test_preference_fires_on_a_declarative_statement() -> None:
    candidates = extract.extract(
        tool_name="terminal", tool_input={}, message="always use uv, never pip"
    )

    assert "preference" in _kinds(candidates)
    assert any("always use uv" in text for text in _texts(candidates))


def test_preference_does_not_fire_on_a_plain_description() -> None:
    candidates = extract.extract(
        tool_name="terminal", tool_input={}, message="the test suite runs in 27 seconds"
    )

    assert candidates == []


# --------------------------------------------------------------------------- #
# decision
# --------------------------------------------------------------------------- #


def test_decision_fires_on_the_locked_phrasing() -> None:
    candidates = extract.extract(
        tool_name="terminal", tool_input={}, message="locked SQLite over Hindsight"
    )

    assert _kinds(candidates) == ["decision"]


def test_decision_does_not_fire_on_a_question() -> None:
    candidates = extract.extract(
        tool_name="terminal", tool_input={}, message="should we lock the schema or not"
    )

    assert candidates == []


# --------------------------------------------------------------------------- #
# architecture
# --------------------------------------------------------------------------- #


def test_architecture_fires_on_a_write_to_an_adr() -> None:
    candidates = extract.extract(
        tool_name="str_replace_editor",
        tool_input={"path": "docs/adr/0001-sqlite.md", "file_text": "flat schema"},
    )

    assert _kinds(candidates) == ["architecture"]
    assert "docs/adr/0001-sqlite.md" in _texts(candidates)[0]


def test_architecture_fires_on_pyproject_toml() -> None:
    candidates = extract.extract(
        tool_name="str_replace_editor",
        tool_input={"path": "pyproject.toml", "file_text": "[dependencies]\n"},
    )

    assert _kinds(candidates) == ["architecture"]


def test_architecture_does_not_fire_on_an_unrelated_source_file() -> None:
    candidates = extract.extract(
        tool_name="str_replace_editor",
        tool_input={"path": "src/tiered_memory/store.py", "file_text": "def f() -> None: ..."},
    )

    assert candidates == []


def test_architecture_does_not_fire_on_a_toml_without_dependencies() -> None:
    candidates = extract.extract(
        tool_name="str_replace_editor",
        tool_input={"path": "config/settings.toml", "file_text": "key = 'value'\n"},
    )

    assert candidates == []


# --------------------------------------------------------------------------- #
# gotcha
# --------------------------------------------------------------------------- #


def test_gotcha_fires_after_error_then_success() -> None:
    candidates = extract.extract(
        tool_name="terminal",
        tool_input={},
        previous_error="sqlite3.OperationalError: no such module: vec0",
        response_ok=True,
    )

    assert _kinds(candidates) == ["gotcha"]
    assert "vec0" in _texts(candidates)[0]


def test_gotcha_does_not_fire_on_a_first_attempt_failure() -> None:
    candidates = extract.extract(
        tool_name="terminal",
        tool_input={},
        response_ok=False,
    )

    assert candidates == []


def test_gotcha_does_not_fire_without_a_previous_error() -> None:
    candidates = extract.extract(tool_name="terminal", tool_input={}, message="done")

    assert candidates == []


# --------------------------------------------------------------------------- #
# todo
# --------------------------------------------------------------------------- #


def test_todo_fires_on_a_todo_introduced_by_a_write() -> None:
    candidates = extract.extract(
        tool_name="str_replace_editor",
        tool_input={"path": "src/x.py", "file_text": "# TODO: handle the empty store\n"},
    )

    assert _kinds(candidates) == ["todo"]
    assert "handle the empty store" in _texts(candidates)[0]


def test_todo_fires_on_fixme() -> None:
    candidates = extract.extract(
        tool_name="str_replace_editor",
        tool_input={"path": "src/x.py", "file_text": "// FIXME: this leaks a lock\n"},
    )

    assert _kinds(candidates) == ["todo"]


def test_todo_does_not_fire_on_the_word_todolist() -> None:
    candidates = extract.extract(
        tool_name="str_replace_editor",
        tool_input={"path": "src/x.py", "file_text": "the todolist widget"},
    )

    assert candidates == []


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #


def test_binary_content_is_never_extracted() -> None:
    candidates = extract.extract(
        tool_name="str_replace_editor",
        tool_input={"path": "image.png", "content": "always use uv\x00\x01binary"},
    )

    assert candidates == []


def test_content_above_four_kilobytes_is_never_extracted() -> None:
    huge = "always use uv " + "x" * extract.MAX_CONTENT_BYTES

    candidates = extract.extract(
        tool_name="str_replace_editor", tool_input={"path": "notes.md", "content": huge}
    )

    assert candidates == []


def test_a_credential_shaped_blob_is_never_extracted() -> None:
    candidates = extract.extract(
        tool_name="terminal",
        tool_input={"command": "export API_KEY=sk-abcdefghijklmnopqrstuvwx && always use uv"},
    )

    assert candidates == []


def test_a_credential_shaped_candidate_is_refused() -> None:
    assert extract.looks_like_secret("password = hunter2hunter2") is True
    assert extract.looks_like_secret("always use uv") is False


# --------------------------------------------------------------------------- #
# caps
# --------------------------------------------------------------------------- #


def test_a_call_emitting_many_candidates_keeps_three() -> None:
    message = (
        "we decided to use sqlite. always use uv. never use pip. prefer ruff. "
        "locked the schema. going with fts5."
    )

    candidates = extract.extract(tool_name="terminal", tool_input={}, message=message)

    assert len(candidates) == extract.MAX_PER_CALL == 3
    # Five decisions at 0.9 and two preferences at 0.85: ties are broken by text,
    # so the three lowest-confidence candidates (both preferences + one decision)
    # are what the cap discards.
    assert _kinds(candidates) == ["decision", "decision", "decision"]
    assert _texts(candidates) == sorted(_texts(candidates))


def test_the_lowest_confidence_candidate_is_the_one_dropped() -> None:
    message = "# TODO: polish. always use uv. we decided on sqlite. we locked the schema."

    candidates = extract.extract(tool_name="terminal", tool_input={}, message=message)

    assert len(candidates) == extract.MAX_PER_CALL
    assert "todo" not in _kinds(candidates)


def test_a_session_never_exceeds_twenty_stored_facts() -> None:
    with store.session("s1") as conn:
        assert isinstance(conn, sqlite3.Connection)
        for index in range(10):
            candidates = [
                extract.Candidate(
                    f"prefer option {index} variant {variant}", "preference", 0.9, "p"
                )
                for variant in range(extract.MAX_PER_CALL)
            ]
            extract.store_candidates(conn, candidates, session_id="s1", tool_name="terminal")

    with store.session("s1") as conn:
        assert isinstance(conn, sqlite3.Connection)
        row = conn.execute("SELECT count(*) FROM facts").fetchone()
        assert row is not None and int(row[0]) == extract.MAX_PER_SESSION == 20


def test_the_session_cap_is_applied_across_calls() -> None:
    """Once twenty facts exist, the next call stores nothing instead of queueing."""
    with store.session("s2") as conn:
        assert isinstance(conn, sqlite3.Connection)
        for index in range(7):
            full = [
                extract.Candidate(f"prefer option {index} variant {v}", "preference", 0.9, "p")
                for v in range(extract.MAX_PER_CALL)
            ]
            extract.store_candidates(conn, full, session_id="s2", tool_name="terminal")
        over = [extract.Candidate("prefer one more", "preference", 0.9, "p")]
        stored = extract.store_candidates(conn, over, session_id="s2", tool_name="terminal")

    assert stored == []


# --------------------------------------------------------------------------- #
# isolation and the hot path
# --------------------------------------------------------------------------- #


def test_candidates_are_written_to_the_session_tier_only() -> None:
    with store.session("s3") as conn:
        assert isinstance(conn, sqlite3.Connection)
        extract.store_candidates(
            conn,
            [extract.Candidate("always use uv", "preference", 0.9, "p")],
            session_id="s3",
            tool_name="terminal",
        )

    assert config.session_db("s3").is_file()
    assert not config.global_db().exists()
    assert not config.project_db("/tmp/some-project").exists()


def test_extraction_does_not_import_the_embedding_module() -> None:
    """R6: a hook must never drag the 113 MB model onto the hot path."""
    import subprocess
    import sys

    source_root = Path(extract.__file__).resolve().parents[1]
    script = (
        "import sys\n"
        "import tiered_memory.extract as e\n"
        "e.extract(tool_name='terminal', tool_input={'command': 'always use uv'}, message='')\n"
        "assert 'tiered_memory.embeddings' not in sys.modules, 'embeddings imported'\n"
        "assert 'fastembed' not in sys.modules, 'fastembed imported'\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(source_root), "PATH": "/usr/bin:/bin:/usr/local/bin"},
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
