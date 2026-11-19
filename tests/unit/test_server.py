"""Behaviour of the ten MCP tools in ``tiered_memory.server`` (plan §3.1, TM-08).

The claims under test are the tool contract itself: the exact names and inputs,
the documented output shapes, and the behaviours they promise.

* every tool is registered under its §3.1 name with the documented parameters;
* ``memory_write`` dedupes and reports supersession, ``memory_recall`` ranks and
  increments ``hits``/``last_hit_at`` (``use_boost`` depends on it);
* ``scope="auto"`` resolves from ``OPENHANDS_PROJECT_DIR`` and
  ``OPENHANDS_SESSION_ID``, and never reads another session's store;
* ``memory_forget`` is a soft delete that every read honours;
* ``memory_prune`` defaults to a dry run and only ever targets old superseded rows;
* ``memory_export``/``memory_import`` round-trip through JSONL, and a second
  import is a no-op;
* the server answers an ``initialize`` handshake with ``TIERED_MEMORY_DATA``
  unset, because the handshake must never depend on the data directory.

Every test drives real stores. The embedding backend is the only fake, because
the real one is a one-time 90 MB download and the code under test is the tool
layer, not the ONNX graph.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

import tiered_memory.config as config
import tiered_memory.embeddings as embeddings
import tiered_memory.portable as portable
import tiered_memory.schema as schema
import tiered_memory.server as srv
import tiered_memory.store as store

_REPO_ROOT = Path(srv.__file__).resolve().parents[2]
_LAUNCHER = _REPO_ROOT / "bin" / "tiered-memory-mcp"

_ALPHA = [1.0] + [0.0] * 383
_BETA = [0.0, 1.0] + [0.0] * 382

_TOOL_INPUTS: dict[str, set[str]] = {
    "memory_recall": {"query"},
    "memory_write": {"text"},
    "memory_search_fts": {"query"},
    "memory_forget": {"id", "tier"},
    "memory_stats": set(),
    "memory_reindex": set(),
    "memory_prune": set(),
    "memory_export": set(),
    "memory_import": {"path"},
    "memory_warm": set(),
}


class FakeBackend:
    """Deterministic stand-in for ``fastembed``: vectors keyed by a token."""

    model_name = schema.DEFAULT_EMBEDDING_MODEL
    dimension = schema.EMBEDDING_DIMENSION

    def __init__(self) -> None:
        self.loaded = False
        self.calls = 0

    def warm(self) -> None:
        self.loaded = True

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        if not self.loaded:
            self.warm()
        return [_ALPHA if "alpha" in text else _BETA for text in texts]


@pytest.fixture(autouse=True)
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A plugin data root, an OpenHands project and session, and a fake model."""
    monkeypatch.setenv(config.DATA_ENV, str(tmp_path / "plugin-data"))
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("OPENHANDS_PROJECT_DIR", str(project))
    monkeypatch.setenv("OPENHANDS_SESSION_ID", "s1")
    monkeypatch.setattr(srv, "_manager", embeddings.IndexManager(FakeBackend()))
    monkeypatch.setattr(srv, "_backend_loaded", False)
    yield


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """The project directory the auto scope resolves to."""
    return tmp_path / "project"


def _db(tier: str, project: Path, session: str = "s1") -> Path:
    if tier == "global":
        return config.global_db()
    if tier == "project":
        return config.project_db(str(project))
    return config.session_db(session)


def _rows(project: Path, tier: str = "project") -> list[sqlite3.Row]:
    path = _db(tier, project)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute("SELECT * FROM facts ORDER BY id"))
    finally:
        conn.close()


def _write(text: str, tier: str = "project", **kwargs: Any) -> dict[str, Any]:
    return srv.memory_write(text, tier=tier, **kwargs)


def _project() -> Path:
    return Path(os.environ["OPENHANDS_PROJECT_DIR"])


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #


def test_all_ten_tools_are_registered_under_their_plan_names() -> None:
    tools = asyncio.run(srv.server.list_tools())
    assert {tool.name for tool in tools} == set(_TOOL_INPUTS)


def test_each_tool_declares_its_required_inputs() -> None:
    tools = {tool.name: tool for tool in asyncio.run(srv.server.list_tools())}
    for name, required in _TOOL_INPUTS.items():
        schema_properties = set(tools[name].input_schema.get("properties", {}))
        assert required <= schema_properties, name
        assert set(tools[name].input_schema.get("required", [])) == required, name


def test_the_handshake_succeeds_and_lists_the_tools_without_a_data_directory() -> None:
    """The launcher is the real entry point; the handshake must not need storage.

    Driven incrementally rather than by piping every request and closing stdin:
    the stdio transport treats EOF as shutdown and can exit before answering a
    request that was already buffered, which made this test fail ~half the time.
    Reading each reply before sending the next removes the race without weakening
    what is asserted.
    """
    environment = {key: value for key, value in os.environ.items() if key != config.DATA_ENV}
    initialize = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "probe", "version": "0"},
            },
        }
    )
    initialized = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
    tools_list = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})

    process = subprocess.Popen(
        [str(_LAUNCHER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        assert process.stdin is not None
        assert process.stdout is not None

        def exchange(payload: str) -> dict[str, Any]:
            process.stdin.write(payload + "\n")
            process.stdin.flush()
            for _ in range(200):
                line = process.stdout.readline()
                if not line:
                    break
                parsed = json.loads(line)
                if "id" in parsed:
                    return parsed
            raise AssertionError("no JSON-RPC reply arrived")

        assert exchange(initialize)["result"]["serverInfo"]["name"] == "tiered-memory"
        process.stdin.write(initialized + "\n")
        process.stdin.flush()
        names = {tool["name"] for tool in exchange(tools_list)["result"]["tools"]}
        assert names == set(_TOOL_INPUTS)
    finally:
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            process.kill()
            process.wait()


# --------------------------------------------------------------------------- #
# memory_write / memory_recall
# --------------------------------------------------------------------------- #


def test_write_reports_an_id_and_marks_a_repeat_as_deduped() -> None:
    first = _write("alpha fact")
    second = _write("alpha fact")

    assert set(first) == {"id", "deduped", "superseded_id"}
    assert first["deduped"] is False
    assert second["deduped"] is True
    assert second["id"] == first["id"]
    assert second["superseded_id"] is None
    assert len(_rows(_project())) == 1


def test_write_stores_the_requested_kind_and_confidence() -> None:
    _write("always use uv", kind="preference", confidence=0.75)

    rows = _rows(_project())
    assert rows[0]["kind"] == "preference"
    assert rows[0]["confidence"] == pytest.approx(0.75)


def test_recall_returns_the_plan_hit_shape_and_ranks_the_match_first() -> None:
    _write("alpha fact about sqlite")
    _write("beta decision about postgres")

    hits = srv.memory_recall("sqlite")

    assert hits
    assert set(hits[0]) == {"tier", "id", "text", "kind", "score", "superseded"}
    assert hits[0]["text"] == "alpha fact about sqlite"
    assert hits[0]["tier"] == "project"
    assert hits[0]["superseded"] is False
    assert hits[0]["score"] > 0


def test_recall_increments_hits_and_last_hit_at_for_returned_rows() -> None:
    _write("alpha fact about sqlite")

    srv.memory_recall("sqlite")
    first = _rows(_project())[0]
    assert first["hits"] == 1
    assert first["last_hit_at"] is not None

    srv.memory_recall("sqlite")
    second = _rows(_project())[0]
    assert second["hits"] == 2
    assert second["last_hit_at"] >= first["last_hit_at"]


def test_recall_honours_the_kinds_filter() -> None:
    _write("alpha fact about sqlite")
    _write("alpha decision about sqlite", kind="decision")

    hits = srv.memory_recall("sqlite", kinds=["decision"])

    assert [hit["kind"] for hit in hits] == ["decision"]


def test_recall_honours_the_limit() -> None:
    for index in range(5):
        _write(f"alpha sqlite fact number {index}")

    assert len(srv.memory_recall("sqlite", limit=2)) == 2


def test_recall_excludes_superseded_rows_unless_history_is_requested() -> None:
    older = _write("alpha sqlite is the store")["id"]
    newer = _write("alpha sqlite is not the store anymore")["id"]
    with store.project(str(_project())) as conn:
        assert isinstance(conn, sqlite3.Connection)
        conn.execute("UPDATE facts SET superseded_by = ? WHERE id = ?", (newer, older))

    current = srv.memory_recall("sqlite")
    history = srv.memory_recall("sqlite", include_history=True)

    assert {hit["id"] for hit in current} == {newer}
    assert older in {hit["id"] for hit in history}


def test_recall_auto_reads_the_session_project_and_global_tiers(project: Path) -> None:
    _write("alpha global note", tier="global")
    _write("alpha project note", tier="project")
    _write("alpha session note", tier="session")

    hits = srv.memory_recall("alpha")

    assert {hit["tier"] for hit in hits} == {"global", "project", "session"}
    assert _db("session", project).exists()
    assert _db("global", project).exists()


def test_recall_auto_does_not_read_the_session_store_without_a_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write("alpha session note", tier="session")
    monkeypatch.delenv("OPENHANDS_SESSION_ID")

    hits = srv.memory_recall("alpha")

    assert hits == []


def test_recall_scope_session_never_reads_another_sessions_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write("alpha session note", tier="session")
    monkeypatch.setenv("OPENHANDS_SESSION_ID", "s2")

    assert srv.memory_recall("alpha", scope="session") == []


def test_recall_scope_project_does_not_read_global_facts(project: Path) -> None:
    _write("alpha global note", tier="global")

    assert srv.memory_recall("alpha", scope="project") == []


def test_recall_rejects_an_unknown_scope() -> None:
    with pytest.raises(ValueError, match="scope"):
        srv.memory_recall("alpha", scope="everywhere")


def test_recall_reads_vectors_only_after_the_model_is_warm() -> None:
    _write("alpha vector fact")
    assert srv._manager is not None

    assert srv.memory_recall("completely unrelated query") == []

    assert srv.memory_warm(wait=True)["state"] in {"pending", "ready"}
    assert srv.memory_reindex(force=True)["state"] == "ready"

    hits = srv.memory_recall("completely unrelated query")
    assert [hit["text"] for hit in hits] == ["alpha vector fact"]


# --------------------------------------------------------------------------- #
# memory_search_fts
# --------------------------------------------------------------------------- #


def test_search_fts_returns_keyword_hits_only() -> None:
    _write("alpha fact about sqlite")
    _write("beta decision about postgres")

    hits = srv.memory_search_fts("sqlite")

    assert [hit["text"] for hit in hits] == ["alpha fact about sqlite"]
    assert set(hits[0]) == {"tier", "id", "text", "kind", "score", "superseded"}


def test_search_fts_honours_scope_and_limit() -> None:
    _write("alpha global sqlite note", tier="global")
    _write("alpha project sqlite note", tier="project")

    hits = srv.memory_search_fts("sqlite", scope="project", limit=1)

    assert len(hits) == 1
    assert hits[0]["tier"] == "project"


def test_search_fts_rejects_an_unknown_scope() -> None:
    with pytest.raises(ValueError, match="scope"):
        srv.memory_search_fts("alpha", scope="elsewhere")


# --------------------------------------------------------------------------- #
# memory_forget
# --------------------------------------------------------------------------- #


def test_forget_soft_deletes_a_fact_and_every_read_hides_it() -> None:
    fact_id = _write("alpha fact to forget")["id"]

    assert srv.memory_forget(fact_id, tier="project") == {"forgotten": True}
    assert srv.memory_recall("alpha", include_history=True) == []
    row = _rows(_project())[0]
    assert row["deleted_at"] is not None
    assert row["superseded_by"] is None


def test_forget_of_an_unknown_id_reports_false() -> None:
    _write("alpha fact")

    assert srv.memory_forget(9999, tier="project") == {"forgotten": False}


def test_forget_rejects_an_unknown_tier() -> None:
    with pytest.raises(ValueError, match="tier"):
        srv.memory_forget(1, tier="archive")


# --------------------------------------------------------------------------- #
# memory_stats
# --------------------------------------------------------------------------- #


def test_stats_reports_counts_index_state_model_and_dimension() -> None:
    _write("alpha project fact")
    _write("alpha global fact", tier="global")

    stats = srv.memory_stats()

    assert stats["project"] == 1
    assert stats["global"] == 1
    assert stats["session"] == 0
    assert stats["model"] == schema.DEFAULT_EMBEDDING_MODEL
    assert stats["dim"] == schema.EMBEDDING_DIMENSION
    assert stats["index_state"] in schema.INDEX_STATES | {"empty"}
    assert isinstance(stats["warnings"], list)


def test_stats_reports_a_warning_above_the_live_fact_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "FACT_WARNING_THRESHOLD", 1)
    _write("alpha one")
    _write("alpha two")

    stats = srv.memory_stats()

    assert any("threshold" in warning for warning in stats["warnings"])


def test_stats_answers_without_a_data_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(config.DATA_ENV)

    stats = srv.memory_stats()

    assert (stats["global"], stats["project"], stats["session"]) == (0, 0, 0)
    assert stats["index_state"] == "empty"
    assert stats["warnings"]


def test_a_write_without_a_data_directory_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(config.DATA_ENV)

    with pytest.raises(config.ConfigError):
        _write("alpha fact")


# --------------------------------------------------------------------------- #
# memory_reindex / memory_warm
# --------------------------------------------------------------------------- #


def test_reindex_embeds_pending_rows_and_reports_ready() -> None:
    _write("alpha fact")
    _write("beta fact")

    result = srv.memory_reindex()

    assert result["embedded"] == 2
    assert result["state"] == "ready"


def test_reindex_of_a_tier_that_does_not_exist_embeds_nothing() -> None:
    result = srv.memory_reindex(tier="global")

    assert result == {"embedded": 0, "state": "empty"}


def test_warm_waits_for_the_model_and_reports_its_identity() -> None:
    _write("alpha fact")

    result = srv.memory_warm(wait=True)

    assert result["model"] == schema.DEFAULT_EMBEDDING_MODEL
    assert result["state"] in {"pending", "ready"}
    assert result["eta_seconds"] == 0


def test_warm_without_waiting_reports_the_current_state() -> None:
    result = srv.memory_warm()

    assert result["model"] == schema.DEFAULT_EMBEDDING_MODEL
    assert result["state"] in schema.INDEX_STATES | {"empty"}
    assert result["eta_seconds"] >= 0


# --------------------------------------------------------------------------- #
# memory_prune
# --------------------------------------------------------------------------- #


def _insert_superseded(project: Path, *, seen_at: str) -> None:
    """Insert a live fact and an older superseded one, without going through the API."""
    with store.project(str(project)) as conn:
        assert isinstance(conn, sqlite3.Connection)
        successor = conn.execute(
            "INSERT INTO facts (content_hash, subject_key, text, kind, tier, scope_key,"
            " created_at, updated_at, last_seen_at)"
            " VALUES ('new', 'subject', 'new text', 'fact', 'project', 'k', ?, ?, ?)"
            " RETURNING id",
            (seen_at, seen_at, seen_at),
        ).fetchone()
        assert successor is not None
        conn.execute(
            "INSERT INTO facts (content_hash, subject_key, text, kind, tier, scope_key,"
            " created_at, updated_at, last_seen_at, superseded_by)"
            " VALUES ('old', 'subject', 'old text', 'fact', 'project', 'k', ?, ?, ?, ?)",
            (seen_at, seen_at, seen_at, int(successor[0])),
        )


def test_prune_defaults_to_a_dry_run(project: Path) -> None:
    _insert_superseded(project, seen_at="2000-01-01T00:00:00Z")

    result = srv.memory_prune()

    assert result == {"candidates": 1, "pruned": 0}
    assert len(_rows(project)) == 2


def test_prune_deletes_only_superseded_rows_older_than_the_threshold(
    project: Path,
) -> None:
    _insert_superseded(project, seen_at="2000-01-01T00:00:00Z")

    result = srv.memory_prune(dry_run=False, older_than_days=180)

    assert result == {"candidates": 1, "pruned": 1}
    assert [row["text"] for row in _rows(project)] == ["new text"]


def test_prune_leaves_recent_superseded_rows_alone(project: Path) -> None:
    _insert_superseded(project, seen_at="2999-01-01T00:00:00Z")

    assert srv.memory_prune(dry_run=False) == {"candidates": 0, "pruned": 0}
    assert len(_rows(project)) == 2


# --------------------------------------------------------------------------- #
# memory_export / memory_import
# --------------------------------------------------------------------------- #


def test_export_writes_to_a_default_path_inside_the_data_root(tmp_path: Path) -> None:
    _write("alpha fact")

    result = srv.memory_export()

    path = Path(result["path"])
    assert result["facts"] == 1
    assert path.is_absolute()
    assert path.is_relative_to(config.data_dir())
    assert path.exists()


def test_export_import_round_trip_and_a_second_import_is_a_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write("alpha fact")
    _write("beta decision", kind="decision")
    target = tmp_path / "out.jsonl"

    exported = srv.memory_export(path=str(target))
    with store.project(str(_project())) as conn:
        assert isinstance(conn, sqlite3.Connection)
        before = [tuple(row) for row in conn.execute("SELECT * FROM facts ORDER BY id")]

    monkeypatch.setenv(config.DATA_ENV, str(tmp_path / "restored"))
    first = srv.memory_import(path=str(target))
    second = srv.memory_import(path=str(target))

    assert exported["facts"] == 2
    assert first == {"imported": 2, "merged": 0, "skipped": 0}
    assert second == {"imported": 0, "merged": 0, "skipped": 2}
    with store.project(str(_project())) as conn:
        assert isinstance(conn, sqlite3.Connection)
        after = [tuple(row) for row in conn.execute("SELECT * FROM facts ORDER BY id")]
    assert after == before


def test_import_refuses_a_mismatched_embedding_dimension(tmp_path: Path) -> None:
    _write("alpha fact")
    target = tmp_path / "out.jsonl"
    srv.memory_export(path=str(target))
    lines = target.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["embedding_dim"] = 768
    target.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n", encoding="utf-8")

    with pytest.raises(portable.ImportMismatchError) as raised:
        srv.memory_import(path=str(target))

    assert raised.value.expected_dim == schema.EMBEDDING_DIMENSION
    assert raised.value.actual_dim == 768


def test_import_requires_the_session_env_for_session_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write("alpha session fact", tier="session")
    target = tmp_path / "out.jsonl"
    srv.memory_export(path=str(target))
    monkeypatch.delenv("OPENHANDS_SESSION_ID")

    with pytest.raises(config.ConfigError, match="OPENHANDS_SESSION_ID"):
        srv.memory_import(path=str(target))


def test_import_recomputes_vectors_locally(tmp_path: Path) -> None:
    _write("alpha fact")
    target = tmp_path / "out.jsonl"
    srv.memory_export(path=str(target))
    with store.project(str(_project())) as conn:
        assert isinstance(conn, sqlite3.Connection)
        conn.execute("DELETE FROM facts")
        conn.execute("DELETE FROM facts_vec")

    srv.memory_import(path=str(target))

    with store.project(str(_project())) as conn:
        assert isinstance(conn, sqlite3.Connection)
        vectors = conn.execute("SELECT count(*) FROM facts_vec").fetchone()[0]
        state = schema.read_meta(conn)["index_state"]
    assert vectors == 1
    assert state == "ready"
