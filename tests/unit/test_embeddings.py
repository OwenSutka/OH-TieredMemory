"""Behaviour of ``tiered_memory.embeddings`` (plan §5.1, Step 6 / TM-06).

The claims under test:

* the embedding backend is lazy: nothing is imported or downloaded until
  ``warm`` is called, and both ``fastembed`` and ``sqlite-vec`` are optional at
  import time;
* the model cache lives under ``${PLUGIN_DATA}/models/``;
* ``index_state`` follows §5.1: ``warming -> pending`` on model load,
  ``pending -> ready`` when every row has a vector, ``ready -> warming`` when the
  model changes, and ``* -> fts_only`` when ``sqlite-vec`` is unusable;
* the FTS path answers while the index warms, and semantic hits appear only once
  the state is ``ready``;
* changing the model disables the vector retriever instead of raising;
* ``memory_reindex`` backfills missing vectors, and ``force`` rebuilds them.

Every test uses a real store. The backend is a deterministic fake where the
model would otherwise be a 90 MB download: the state machine and the vec0 I/O
are the code under test, not the ONNX graph.
"""

from __future__ import annotations

import importlib.abc
import json
import os
import sqlite3
import struct
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

import tiered_memory.config as config
import tiered_memory.embeddings as embeddings
import tiered_memory.facts as facts
import tiered_memory.schema as schema
import tiered_memory.search as search
import tiered_memory.store as store

_SRC_DIR = Path(embeddings.__file__).resolve().parents[1]

_NOW = datetime(2026, 6, 1, tzinfo=UTC)

#: 384-dim one-hot vectors, so a nearest neighbour is a fact of the fixture.
_ALPHA = [1.0] + [0.0] * 383
_BETA = [0.0, 1.0] + [0.0] * 382


class FakeBackend:
    """Deterministic stand-in for ``fastembed``: vectors keyed by a token."""

    def __init__(self, *, model_name: str = "fake-model", dimension: int = 384) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self.loaded = False
        self.calls = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def warm(self) -> None:
        self.loaded = True

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        if not self.loaded:
            self.warm()
        return [_ALPHA if "alpha" in text else _BETA for text in texts]


@pytest.fixture(autouse=True)
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate every test in its own plugin data root."""
    root = tmp_path / "plugin-data"
    monkeypatch.setenv(config.DATA_ENV, str(root))
    return root


@pytest.fixture()
def conn(data_root: Path) -> Iterator[sqlite3.Connection]:
    with store.global_store() as opened:
        assert isinstance(opened, sqlite3.Connection)
        yield opened


def _write(conn: sqlite3.Connection, text: str, *, kind: str = "fact") -> int:
    return facts.insert(conn, text, kind=kind, now=_NOW).id


def _state(conn: sqlite3.Connection) -> str:
    return schema.read_meta(conn)["index_state"]


def _vectors(conn: sqlite3.Connection) -> dict[int, bytes]:
    return {
        int(row[0]): bytes(row[1]) for row in conn.execute("SELECT rowid, embedding FROM facts_vec")
    }


def _blob(vector: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


# --------------------------------------------------------------------------- #
# optional dependencies and the model cache
# --------------------------------------------------------------------------- #


def test_importing_the_module_does_not_import_fastembed() -> None:
    code = """
import sys
import tiered_memory.embeddings
print("fastembed" in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(_SRC_DIR)},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


class _BlockFastEmbed(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if fullname == "fastembed" or fullname.startswith("fastembed."):
            raise ImportError("simulated: fastembed is not installed")


_BLOCK_FASTEMBED = """
import importlib.abc
import sys


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "fastembed" or fullname.startswith("fastembed."):
            raise ImportError("simulated: fastembed is not installed")
        return None


sys.meta_path.insert(0, _Blocker())
sys.modules.pop("fastembed", None)
"""


def test_a_missing_fastembed_degrades_to_fts_only_without_raising(tmp_path: Path) -> None:
    code = (
        _BLOCK_FASTEMBED
        + """
import json
import tiered_memory.embeddings as embeddings
import tiered_memory.facts as facts
import tiered_memory.schema as schema
import tiered_memory.search as search
import tiered_memory.store as store

backend = embeddings.FastEmbedBackend()
manager = embeddings.IndexManager(backend)

with store.global_store() as conn:
    facts.insert(conn, "the lockfile is uv", kind="decision")
    try:
        state = manager.warm(conn)
        error = None
    except embeddings.ModelUnavailable as exc:
        state = schema.read_meta(conn)["index_state"]
        error = str(exc)

with store.global_store() as conn:
    hits = search.recall([conn], "lockfile", embedder=manager.embed_query)

print(json.dumps({"state": state, "error": error, "texts": [hit.text for hit in hits]}))
"""
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(_SRC_DIR), config.DATA_ENV: str(tmp_path)},
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["state"] == "fts_only"
    assert payload["error"] and "fastembed" in payload["error"]
    assert payload["texts"] == ["the lockfile is uv"]


def test_the_model_cache_lives_under_the_plugin_data_root(data_root: Path) -> None:
    assert embeddings.models_dir() == data_root / "models"


def test_the_backend_points_fastembed_at_the_plugin_cache() -> None:
    backend = embeddings.FastEmbedBackend()

    assert backend.cache_dir == embeddings.models_dir()
    assert backend.model_name == "all-MiniLM-L6-v2"
    assert backend.model_name == schema.DEFAULT_EMBEDDING_MODEL
    assert backend.fastembed_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert backend.dimension == schema.EMBEDDING_DIMENSION


def test_both_model_spellings_resolve_to_the_store_identity() -> None:
    """Stores and exports record the plan's short name, not fastembed's."""
    assert embeddings.resolve_model_name("all-MiniLM-L6-v2") == "all-MiniLM-L6-v2"
    assert (
        embeddings.resolve_model_name("sentence-transformers/all-MiniLM-L6-v2")
        == "all-MiniLM-L6-v2"
    )
    assert embeddings.resolve_model_name("custom/model") == "custom/model"
    assert embeddings.fastembed_model_name("all-MiniLM-L6-v2") == (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert embeddings.fastembed_model_name("custom/model") == "custom/model"


# --------------------------------------------------------------------------- #
# §5.1 state machine
# --------------------------------------------------------------------------- #


def test_a_fresh_store_starts_before_ready(conn: sqlite3.Connection) -> None:
    assert _state(conn) in {"pending", "warming", "fts_only"}
    assert _state(conn) != "ready"


def test_warming_a_cold_index_moves_it_to_pending(conn: sqlite3.Connection) -> None:
    backend = FakeBackend()
    manager = embeddings.IndexManager(backend)
    conn.execute("UPDATE meta SET value = 'warming' WHERE key = 'index_state'")

    state = manager.warm(conn)

    assert backend.loaded is True
    assert state == "pending"
    assert _state(conn) == "pending"


def test_backfill_moves_pending_to_ready(conn: sqlite3.Connection) -> None:
    _write(conn, "alpha fact")
    _write(conn, "beta fact")
    manager = embeddings.IndexManager(FakeBackend())
    manager.warm(conn)

    embedded = manager.backfill(conn)

    assert embedded == 2
    assert _state(conn) == "ready"
    assert len(_vectors(conn)) == 2


def test_an_empty_store_becomes_ready_without_embedding(conn: sqlite3.Connection) -> None:
    manager = embeddings.IndexManager(FakeBackend())
    manager.warm(conn)

    assert manager.backfill(conn) == 0
    assert _state(conn) == "ready"


def test_backfill_only_embeds_rows_without_a_vector(conn: sqlite3.Connection) -> None:
    first = _write(conn, "alpha fact")
    manager = embeddings.IndexManager(FakeBackend())
    manager.warm(conn)
    manager.backfill(conn)
    second = _write(conn, "beta fact")

    assert set(_vectors(conn)) == {first}
    assert manager.backfill(conn) == 1
    assert set(_vectors(conn)) == {first, second}


def test_backfill_leaves_ready_when_nothing_is_missing(conn: sqlite3.Connection) -> None:
    _write(conn, "alpha fact")
    manager = embeddings.IndexManager(FakeBackend())
    manager.warm(conn)
    manager.backfill(conn)

    assert manager.backfill(conn) == 0
    assert _state(conn) == "ready"


def test_changing_the_model_flips_ready_back_to_warming(conn: sqlite3.Connection) -> None:
    manager = embeddings.IndexManager(FakeBackend(model_name="model-a"))
    manager.warm(conn)
    manager.backfill(conn)
    assert _state(conn) == "ready"

    changed = embeddings.IndexManager(FakeBackend(model_name="model-b"))
    state = changed.sync_model(conn)

    assert state == "warming"
    assert _state(conn) == "warming"
    assert schema.read_meta(conn)["embedding_model"] == "model-b"


def test_a_default_backend_does_not_disturb_a_fresh_store(conn: sqlite3.Connection) -> None:
    """The store records the plan's model name, so a default backend matches it."""
    manager = embeddings.IndexManager()
    before = schema.read_meta(conn)

    state = manager.sync_model(conn)

    assert state == before["index_state"] == "pending"
    assert schema.read_meta(conn) == before


def test_warming_after_a_model_change_keeps_the_retriever_disabled(
    conn: sqlite3.Connection,
) -> None:
    """Backfill only fills gaps, so stale vectors must not be declared ready."""
    _write(conn, "alpha fact")
    first = embeddings.IndexManager(FakeBackend(model_name="model-a"))
    first.warm(conn)
    first.backfill(conn)
    assert _state(conn) == "ready"

    changed = embeddings.IndexManager(FakeBackend(model_name="model-b"))
    assert changed.warm(conn) == "warming"
    assert _state(conn) == "warming"
    assert changed.embed_query("alpha") is None

    # A backfill pass must not launder the stale vectors into "ready".
    assert changed.backfill(conn) == 0
    assert _state(conn) == "warming"

    assert changed.reindex(conn, force=True).state == "ready"


def test_syncing_an_unchanged_model_leaves_ready_alone(conn: sqlite3.Connection) -> None:
    manager = embeddings.IndexManager(FakeBackend(model_name="model-a"))
    manager.warm(conn)
    manager.backfill(conn)

    assert manager.sync_model(conn) == "ready"
    assert _state(conn) == "ready"


# --------------------------------------------------------------------------- #
# the two retrievers and readiness
# --------------------------------------------------------------------------- #


def test_semantic_hits_are_empty_until_the_index_is_ready(conn: sqlite3.Connection) -> None:
    _write(conn, "the build artifact")
    backend = FakeBackend()
    manager = embeddings.IndexManager(backend)
    assert manager.embed_query("alpha") is None

    assert search.recall([conn], "alpha", embedder=manager.embed_query) == []


def test_cold_start_answers_from_fts_while_warming(conn: sqlite3.Connection) -> None:
    keyword_id = _write(conn, "alpha lockfile")
    conn.execute("UPDATE meta SET value = 'warming' WHERE key = 'index_state'")
    backend = FakeBackend()
    manager = embeddings.IndexManager(backend)

    hits = search.recall([conn], "lockfile", embedder=manager.embed_query)

    assert [hit.id for hit in hits] == [keyword_id]


def test_after_warm_up_the_semantic_retriever_returns_results(conn: sqlite3.Connection) -> None:
    semantic_only = _write(conn, "the build artifact")
    manager = embeddings.IndexManager(FakeBackend())
    manager.warm(conn)
    manager.backfill(conn)

    hits = search.recall([conn], "alpha", embedder=manager.embed_query)

    assert [hit.id for hit in hits] == [semantic_only]


def test_a_model_change_disables_the_vector_retriever_not_the_query(
    conn: sqlite3.Connection,
) -> None:
    keyword_id = _write(conn, "alpha lockfile")
    ready = embeddings.IndexManager(FakeBackend(model_name="model-a"))
    ready.warm(conn)
    ready.backfill(conn)

    changed = embeddings.IndexManager(FakeBackend(model_name="model-b"))
    changed.sync_model(conn)

    hits = search.recall([conn], "lockfile", embedder=changed.embed_query)

    assert [hit.id for hit in hits] == [keyword_id]


# --------------------------------------------------------------------------- #
# memory_reindex
# --------------------------------------------------------------------------- #


def test_reindex_backfills_missing_vectors(conn: sqlite3.Connection) -> None:
    _write(conn, "alpha fact")
    _write(conn, "beta fact")
    manager = embeddings.IndexManager(FakeBackend())

    result = manager.reindex(conn)

    assert result.embedded == 2
    assert result.state == "ready"
    assert len(_vectors(conn)) == 2


def test_reindex_without_force_keeps_existing_vectors(conn: sqlite3.Connection) -> None:
    fact_id = _write(conn, "alpha fact")
    manager = embeddings.IndexManager(FakeBackend())
    manager.warm(conn)
    manager.backfill(conn)
    before = _vectors(conn)[fact_id]

    result = manager.reindex(conn)

    assert result.embedded == 0
    assert _vectors(conn)[fact_id] == before


def test_reindex_force_rebuilds_every_vector(conn: sqlite3.Connection) -> None:
    fact_id = _write(conn, "alpha fact")
    manager = embeddings.IndexManager(FakeBackend())
    manager.warm(conn)
    manager.backfill(conn)
    conn.execute("UPDATE facts_vec SET embedding = ? WHERE rowid = ?", (_blob(_BETA), fact_id))

    result = manager.reindex(conn, force=True)

    assert result.embedded == 1
    assert result.state == "ready"
    assert _vectors(conn)[fact_id] == _blob(_ALPHA)


def test_reindex_after_a_model_change_rewrites_every_row(conn: sqlite3.Connection) -> None:
    _write(conn, "alpha one")
    _write(conn, "beta two")
    first = embeddings.IndexManager(FakeBackend(model_name="model-a"))
    first.warm(conn)
    first.backfill(conn)

    second = embeddings.IndexManager(FakeBackend(model_name="model-b"))
    second.sync_model(conn)
    result = second.reindex(conn, force=True)

    assert result.embedded == 2
    assert result.state == "ready"
    assert schema.read_meta(conn)["embedding_model"] == "model-b"


def test_reindex_refuses_a_backend_of_the_wrong_width(conn: sqlite3.Connection) -> None:
    _write(conn, "alpha fact")
    manager = embeddings.IndexManager(FakeBackend(dimension=768))

    with pytest.raises(schema.EmbeddingDimensionMismatch):
        manager.reindex(conn, force=True)


def test_a_wrong_width_backend_disables_the_vector_retriever(conn: sqlite3.Connection) -> None:
    _write(conn, "alpha fact")
    manager = embeddings.IndexManager(FakeBackend(dimension=768))

    state = manager.sync_model(conn)

    assert state == "warming"
    assert manager.embed_query("alpha") is None


# --------------------------------------------------------------------------- #
# deleted rows are not re-embedded
# --------------------------------------------------------------------------- #


def test_soft_deleted_rows_are_not_backfilled(conn: sqlite3.Connection) -> None:
    live = _write(conn, "alpha live")
    gone = _write(conn, "beta gone")
    conn.execute("UPDATE facts SET deleted_at = '2026-01-01T00:00:00Z' WHERE id = ?", (gone,))
    manager = embeddings.IndexManager(FakeBackend())
    manager.warm(conn)

    assert manager.backfill(conn) == 1
    assert set(_vectors(conn)) == {live}


# --------------------------------------------------------------------------- #
# fts_only is a terminal, non-raising degradation
# --------------------------------------------------------------------------- #

_BLOCK_SQLITE_VEC = """
import importlib.abc
import sys


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sqlite_vec" or fullname.startswith("sqlite_vec."):
            raise ImportError("simulated: sqlite-vec is not installed")
        return None


sys.meta_path.insert(0, _Blocker())
sys.modules.pop("sqlite_vec", None)
"""


def test_without_sqlite_vec_the_index_is_fts_only_and_never_raises(tmp_path: Path) -> None:
    code = (
        _BLOCK_SQLITE_VEC
        + """
import json
import tiered_memory.embeddings as embeddings
import tiered_memory.facts as facts
import tiered_memory.schema as schema
import tiered_memory.search as search
import tiered_memory.store as store


class _Backend:
    model_name = "fake-model"
    dimension = 384

    def warm(self):
        return None

    def embed(self, texts):
        return [[1.0] + [0.0] * 383 for _ in texts]


manager = embeddings.IndexManager(_Backend())

with store.global_store() as conn:
    facts.insert(conn, "the lockfile is uv", kind="decision")
    state = manager.warm(conn)
    embedded = manager.backfill(conn)
    reindexed = manager.reindex(conn, force=True)

with store.global_store() as conn:
    hits = search.recall([conn], "lockfile", embedder=manager.embed_query)

print(json.dumps({
    "state": state,
    "embedded": embedded,
    "reindexed": reindexed.state,
    "texts": [hit.text for hit in hits],
}))
"""
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(_SRC_DIR), config.DATA_ENV: str(tmp_path)},
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["state"] == "fts_only"
    assert payload["embedded"] == 0
    assert payload["reindexed"] == "fts_only"
    assert payload["texts"] == ["the lockfile is uv"]


# --------------------------------------------------------------------------- #
# the backfill worker the server starts at boot
# --------------------------------------------------------------------------- #


def _worker(manager: embeddings.IndexManager) -> embeddings.BackfillWorker:
    return embeddings.BackfillWorker(manager, store.global_store)


def _write_committed(text: str, *, kind: str = "fact") -> int:
    """Write and release the write lock, so the worker can take it next."""
    with store.global_store() as opened:
        assert isinstance(opened, sqlite3.Connection)
        return _write(opened, text, kind=kind)


def _committed_state() -> str:
    with store.global_store() as opened:
        assert isinstance(opened, sqlite3.Connection)
        return schema.read_meta(opened)["index_state"]


def test_backfill_worker_embeds_pending_rows_in_one_pass() -> None:
    _write_committed("alpha one")
    _write_committed("beta two")

    result = _worker(embeddings.IndexManager(FakeBackend())).run_once()

    assert result.embedded == 2
    assert result.state == "ready"
    assert _committed_state() == "ready"


def test_backfill_worker_leaves_fts_only_stores_alone() -> None:
    _write_committed("alpha one")
    with store.global_store() as opened:
        assert isinstance(opened, sqlite3.Connection)
        opened.execute("UPDATE meta SET value = 'fts_only' WHERE key = 'index_state'")

    result = _worker(embeddings.IndexManager(FakeBackend())).run_once()

    assert result.embedded == 0
    assert result.state == "fts_only"
    assert _committed_state() == "fts_only"


def test_backfill_worker_reports_a_missing_model_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_committed("alpha one")

    def boom(self: embeddings.FastEmbedBackend) -> None:
        raise embeddings.ModelUnavailable("simulated: model download failed")

    monkeypatch.setattr(embeddings.FastEmbedBackend, "warm", boom)

    result = _worker(embeddings.IndexManager()).run_once()

    assert result.state == "fts_only"


def test_backfill_worker_thread_stops_cleanly() -> None:
    _write_committed("alpha one")
    worker = embeddings.BackfillWorker(
        embeddings.IndexManager(FakeBackend()), store.global_store, interval_seconds=0.01
    )

    worker.start()
    try:
        deadline = time.monotonic() + 30.0
        while _committed_state() != "ready":
            assert time.monotonic() < deadline, "the worker never finished a pass"
            time.sleep(0.02)
    finally:
        worker.stop(timeout=10.0)

    assert worker.is_running is False


# --------------------------------------------------------------------------- #
# dependency direction
# --------------------------------------------------------------------------- #


def test_hooks_do_not_import_the_embeddings_module() -> None:
    """Plan R6: ``No hook may import this module.``

    Hook modules are named ``on_*.py`` (TM-09..TM-11) plus the vendored
    ``hook_runtime``; none of them may reach the model.
    """
    package = Path(embeddings.__file__).resolve().parent
    hook_modules = [*sorted(package.glob("on_*.py")), package / "hook_runtime.py"]

    offenders = {
        module.name: "embeddings"
        for module in hook_modules
        if "tiered_memory.embeddings" in module.read_text(encoding="utf-8")
        or "from .embeddings" in module.read_text(encoding="utf-8")
        or "from . import embeddings" in module.read_text(encoding="utf-8")
    }

    assert offenders == {}
