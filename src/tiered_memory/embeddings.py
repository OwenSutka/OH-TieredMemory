"""Embedding model, vector backfill and §5.1 index readiness.

``fastembed`` is the only embedding backend and it is optional: this module
imports without it, :class:`FastEmbedBackend` loads it inside :meth:`warm`, and a
missing or broken install raises :class:`ModelUnavailable` -- never an import
error. Hooks must never reach this module (plan R6); they insert rows with
``embedding = NULL`` and the long-lived server runs the backfill worker.

The state machine is §5.1 and lives in :meth:`IndexManager.warm`,
:meth:`IndexManager.backfill`, :meth:`IndexManager.sync_model` and
:meth:`IndexManager.reindex`:

* ``warming -> pending``: the model finished loading (or the model changed, or
  the backend's width no longer matches the store's).
* ``pending -> ready``: every live fact has a vector.
* ``ready -> warming``: the store records a different ``embedding_model`` than
  the backend serves, or the width disagrees. The vector retriever is disabled
  by :meth:`embed_query` returning ``None``, and recall keeps answering from
  FTS.
* ``any -> fts_only``: ``sqlite-vec`` is unusable. Recorded, never raised.

Vectors are written through a :mod:`tiered_memory.store` transaction; nothing
here takes a lock of its own, and there is no module-level model.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import struct
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from . import config, schema

__all__ = [
    "BATCH_SIZE",
    "DEFAULT_MODEL_NAME",
    "MODEL_ALIASES",
    "Backend",
    "BackfillResult",
    "BackfillWorker",
    "FastEmbedBackend",
    "IndexManager",
    "ModelUnavailable",
    "ReindexResult",
    "fastembed_model_name",
    "models_dir",
    "resolve_model_name",
]

_LOGGER = logging.getLogger(__name__)

#: Plan §2.2.2: the model is ``all-MiniLM-L6-v2``, pinned at 384 dimensions.
#: This is the *store-facing* identity, matching ``schema.DEFAULT_EMBEDDING_MODEL``
#: and the export header in plan §4, not fastembed's ``org/name`` spelling.
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

#: Store-facing identity -> fastembed's own model name.
MODEL_ALIASES: dict[str, str] = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}

#: How many rows one backfill batch embeds.
BATCH_SIZE = 32

_READY = "ready"
_PENDING = "pending"
_WARMING = "warming"
_FTS_ONLY = "fts_only"


class ModelUnavailable(RuntimeError):
    """The embedding model cannot be used on this machine.

    Raised by :meth:`FastEmbedBackend.warm` when ``fastembed`` is absent or the
    model cannot be loaded. Callers treat it as "stay in ``fts_only``".
    """


class Backend(Protocol):
    """What :class:`IndexManager` needs from an embedding backend."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def warm(self) -> None: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class ReindexResult:
    """Outcome of :meth:`IndexManager.reindex`."""

    embedded: int
    state: str


def models_dir() -> Path:
    """The ``${PLUGIN_DATA}/models/`` cache fastembed downloads into."""
    return config.data_dir() / "models"


def resolve_model_name(name: str) -> str:
    """Return the store-facing identity for ``name``, whichever spelling it uses.

    Both the plan's short name and fastembed's ``org/name`` spelling resolve to
    the short name, because that is what a store records and an export quotes.
    An unknown model keeps its own name, so a custom model round-trips.
    """
    for identity, fastembed_name in MODEL_ALIASES.items():
        if name in (identity, fastembed_name):
            return identity
    return name


def fastembed_model_name(identity: str) -> str:
    """Return the name fastembed's registry expects for a store identity."""
    return MODEL_ALIASES.get(identity, identity)


class FastEmbedBackend:
    """Lazy ``fastembed`` wrapper: the model loads on :meth:`warm`, never at import.

    The ONNX model is cached under the plugin's own ``models/`` directory, so the
    one-time download lands inside ``PLUGIN_DATA`` and every subsequent start is
    offline.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        dimension: int = schema.EMBEDDING_DIMENSION,
    ) -> None:
        self._model_name = resolve_model_name(model_name)
        self._dimension = dimension
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def cache_dir(self) -> Path:
        return models_dir()

    @property
    def fastembed_name(self) -> str:
        """The ``org/name`` spelling fastembed's registry uses."""
        return fastembed_model_name(self._model_name)

    def warm(self) -> None:
        """Load the model, downloading it once when the cache is cold.

        Raises:
            ModelUnavailable: ``fastembed`` is absent, or the model cannot be
                downloaded or loaded. The message names the reason.
        """
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding
        except Exception as exc:
            raise ModelUnavailable(f"fastembed is not available: {exc}") from exc
        try:
            self._model = TextEmbedding(
                model_name=self.fastembed_name, cache_dir=str(self.cache_dir)
            )
        except Exception as exc:
            raise ModelUnavailable(
                f"cannot load embedding model {self.fastembed_name!r}: {exc}"
            ) from exc

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts``; loads the model first when it is cold.

        Raises:
            ModelUnavailable: the model cannot be loaded.
        """
        self.warm()
        model = self._model
        if model is None:  # pragma: no cover - warm either loads or raises
            raise ModelUnavailable("embedding model is not loaded")
        return [[float(value) for value in vector] for vector in model.embed(list(texts))]


class IndexManager:
    """Owns one backend and drives the §5.1 transitions over a store."""

    def __init__(self, backend: Backend | None = None) -> None:
        self._backend: Backend = FastEmbedBackend() if backend is None else backend
        self._model_ready = False

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def model_name(self) -> str:
        return self._backend.model_name

    def sync_model(self, conn: sqlite3.Connection) -> str:
        """Align the store's recorded model with the backend; return ``index_state``.

        A store that records another model, or another width, cannot reuse its
        vectors: it flips to ``warming`` and must be reindexed. An unchanged
        model is left exactly as it was, so this is safe to call on every open.
        """
        meta = _meta(conn)
        recorded_model = meta.get("embedding_model")
        recorded_dim = _as_int(meta.get("embedding_dim"))
        compatible = (
            recorded_model == self._backend.model_name
            and recorded_dim == self._backend.dimension
            and self._backend.dimension == schema.EMBEDDING_DIMENSION
        )
        if not compatible:
            self._model_ready = False
            _set_meta(conn, "embedding_model", self._backend.model_name)
            _set_meta(conn, "index_state", _WARMING)
            return _WARMING
        return meta.get("index_state", _WARMING)

    def warm(self, conn: sqlite3.Connection) -> str:
        """Load the model and move the store to ``pending``; return the new state.

        A store that cannot use ``sqlite-vec`` is recorded as ``fts_only`` and
        keeps answering from FTS: it has nowhere to put a vector, so ``pending``
        would be a lie.

        Raises:
            ModelUnavailable: the backend cannot load its model. The store is
                recorded as ``fts_only`` first, so the failure degrades this
                store rather than escaping as an unexplained error.
        """
        if _fts_only(conn):
            _set_meta(conn, "index_state", _FTS_ONLY)
            return _FTS_ONLY
        state = self.sync_model(conn)
        if self._backend.dimension != schema.EMBEDDING_DIMENSION:
            # Vectors of this width cannot live in this store; the model changed
            # shape, so the index must be rebuilt before it can be trusted.
            _set_meta(conn, "index_state", _WARMING)
            return _WARMING
        try:
            self._backend.warm()
        except ModelUnavailable:
            _set_meta(conn, "index_state", _FTS_ONLY)
            raise
        if state == _WARMING and _vector_count(conn):
            # ``warming`` after a model change leaves vectors of the old model
            # in place. The backfill worker only fills gaps, so promoting this
            # to ``pending`` would declare a mixed index ready: the retriever
            # stays disabled until an explicit ``memory_reindex --force``.
            return _WARMING
        self._model_ready = True
        if state == _READY:
            # Vectors are already there and match the model just loaded.
            return _READY
        _set_meta(conn, "index_state", _PENDING)
        return _PENDING

    def embed_query(self, text: str) -> list[float] | None:
        """Embed one query, or ``None`` when vectors are not usable right now.

        Returning ``None`` is what disables the semantic retriever: recall then
        fuses a one-element list, i.e. keyword-only, without an error.
        """
        if not self._model_ready:
            return None
        try:
            (vector,) = self._backend.embed([text])
        except ModelUnavailable:
            self._model_ready = False
            return None
        if len(vector) != schema.EMBEDDING_DIMENSION:
            self._model_ready = False
            return None
        return vector

    def backfill(self, conn: sqlite3.Connection) -> int:
        """Embed every live, undeleted row that has no vector; return the count.

        On completion the state is ``ready`` when nothing is missing -- except
        from ``warming``, which means a model change left vectors of the old
        model behind. Filling the gaps there would mix two vector spaces and
        declare the result ready, so ``warming`` is only cleared by
        :meth:`reindex` with ``force``.

        A store without ``facts_vec`` is ``fts_only`` and nothing is done.
        """
        if _fts_only(conn) or not _has_vec_table(conn):
            _set_meta(conn, "index_state", _FTS_ONLY)
            return 0
        warming = _state(conn) == _WARMING and _vector_count(conn) > 0
        embedded = self._embed_rows(conn, _pending_rows(conn))
        if not warming and not _pending_rows(conn):
            _set_meta(conn, "index_state", _READY)
        return embedded

    def reindex(self, conn: sqlite3.Connection, *, force: bool = False) -> ReindexResult:
        """Backfill vectors, rebuilding every vector when ``force``.

        ``force`` drops the existing vectors first, which is what makes a model
        change recoverable; without it only rows with no vector are embedded.

        Raises:
            schema.EmbeddingDimensionMismatch: the backend's width is not the
                store's. The store is left ``warming`` and the vector retriever
                stays disabled; nothing is written.
        """
        if self._backend.dimension != schema.EMBEDDING_DIMENSION:
            _set_meta(conn, "index_state", _WARMING)
            raise schema.EmbeddingDimensionMismatch(
                schema.EMBEDDING_DIMENSION, self._backend.dimension
            )
        if _fts_only(conn) or not _has_vec_table(conn):
            _set_meta(conn, "index_state", _FTS_ONLY)
            return ReindexResult(embedded=0, state=_FTS_ONLY)
        if force:
            conn.execute("DELETE FROM facts_vec")
        self._backend.warm()
        self._model_ready = True
        embedded = self._embed_rows(conn, _pending_rows(conn))
        state = _READY if not _pending_rows(conn) else _WARMING
        _set_meta(conn, "index_state", state)
        return ReindexResult(embedded=embedded, state=state)

    def _embed_rows(self, conn: sqlite3.Connection, rows: Sequence[tuple[int, str]]) -> int:
        embedded = 0
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start : start + BATCH_SIZE]
            vectors = self._backend.embed([text for _id, text in batch])
            for (fact_id, _text), vector in zip(batch, vectors, strict=True):
                _put_vector(conn, fact_id, vector)
                embedded += 1
        return embedded


class BackfillWorker:
    """Embeds rows with no vector, one store at a time, off the request path.

    The server owns one instance and runs :meth:`start` at boot (plan §3.4). Each
    pass opens a store through the ``open_tier`` context manager the caller
    supplies -- a :mod:`tiered_memory.store` entry point -- so the worker takes
    the same ``BEGIN IMMEDIATE`` lock every other writer does and never holds a
    connection between passes.

    A pass never raises: a missing model, a store it cannot open, or a lock it
    cannot take leaves the state as it was and the next pass retries. That is
    what keeps "recall never fails because of index state" true at the process
    level as well.
    """

    def __init__(
        self,
        manager: IndexManager,
        open_tier: Callable[[], contextlib.AbstractContextManager[Any]],
        *,
        interval_seconds: float = 0.05,
    ) -> None:
        self._manager = manager
        self._open_tier = open_tier
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run_once(self) -> BackfillResult:
        """Run a single pass; return what it embedded and the resulting state."""
        try:
            with self._open_tier() as conn:
                if not isinstance(conn, sqlite3.Connection):
                    return BackfillResult(embedded=0, state=_WARMING)
                self._manager.warm(conn)
                embedded = self._manager.backfill(conn)
                return BackfillResult(embedded=embedded, state=_state(conn))
        except (ModelUnavailable, sqlite3.Error, config.ConfigError):
            _LOGGER.warning("backfill pass skipped; the next pass retries", exc_info=True)
            return BackfillResult(embedded=0, state=_FTS_ONLY)
        except OSError:
            _LOGGER.warning("backfill pass could not open a store; retrying later")
            return BackfillResult(embedded=0, state=_WARMING)

    def start(self) -> None:
        """Start the worker thread unless it is already running."""
        if self.is_running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="memory-backfill", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        """Signal the worker to finish and join it within ``timeout`` seconds."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        self._thread = None

    def _loop(self) -> None:
        # Polling rather than a queue: a write is what makes rows pending, and
        # the cost of an idle pass is one meta read.
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self._interval)


@dataclass(frozen=True, slots=True)
class BackfillResult:
    """Outcome of one :class:`BackfillWorker` pass."""

    embedded: int
    state: str


def _meta(conn: sqlite3.Connection) -> dict[str, str]:
    return schema.read_meta(conn)


def _state(conn: sqlite3.Connection) -> str:
    return _meta(conn).get("index_state", _WARMING)


def _fts_only(conn: sqlite3.Connection) -> bool:
    return _state(conn) == _FTS_ONLY or not _vec_loaded(conn)


def _vec_loaded(conn: sqlite3.Connection) -> bool:
    """Whether this connection can use ``vec0`` at all.

    ``sqlite-vec`` is loaded per connection by :func:`schema.open_store`, so
    probing it is a property of the connection, not of the store file.
    """
    try:
        conn.execute("SELECT vec_version()").fetchone()
    except sqlite3.Error:
        return False
    return True


def _has_vec_table(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'facts_vec'"
    ).fetchone()
    return row is not None


def _vector_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT count(*) FROM facts_vec").fetchone()
    return 0 if row is None else int(row[0])


def _pending_rows(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    rows = conn.execute(
        "SELECT f.id, f.text FROM facts AS f"
        " WHERE f.deleted_at IS NULL"
        "   AND NOT EXISTS (SELECT 1 FROM facts_vec AS v WHERE v.rowid = f.id)"
        " ORDER BY f.id"
    ).fetchall()
    return [(int(row[0]), str(row[1])) for row in rows]


def _put_vector(conn: sqlite3.Connection, fact_id: int, vector: Sequence[float]) -> None:
    payload = struct.pack(f"{len(vector)}f", *vector)
    conn.execute("DELETE FROM facts_vec WHERE rowid = ?", (fact_id,))
    conn.execute(
        "INSERT INTO facts_vec(rowid, embedding) VALUES (?, ?)",
        (fact_id, sqlite3.Binary(payload)),
    )


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    statement = (
        "INSERT INTO meta (key, value) VALUES (?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value = excluded.value"
    )
    conn.execute(statement, (key, value))


def _as_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None
