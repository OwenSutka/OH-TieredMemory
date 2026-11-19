"""JSONL export and import of facts (plan §2.2.7).

An export is a header line followed by one line per fact, with the full row
metadata and **no vectors**: vectors are a local artifact of one store's model
and are recomputed on import. The file is written to a temporary sibling and
moved into place with :func:`os.replace`, so a reader never sees a half-written
export and a failed write leaves the previous file intact.

Import validates the header before touching the store. ``embedding_model`` and
``embedding_dim`` must be the ones the target store expects; a mismatch raises
:class:`ImportMismatchError` carrying both the expected and the actual values,
because a store must never silently accept vectors of another space (R9).

Tier is preserved: a record is only applied to a store of its own tier, and
``retier=True`` is the explicit way to do otherwise. A conflict merges the way a
duplicate write does -- ``seen_count + 1``, ``confidence = max``,
``last_seen = max``, and ``distinct_sessions + 1`` only when the record arrives
from a different source session -- but only when the record carries something
the stored row does not. Re-importing the same file therefore changes nothing:
every record already matches on ``seen_count``, ``confidence``, ``last_seen_at``
and ``source_session``, so it is counted as ``skipped``.

``superseded_by`` is a store-local row id, so the export records
``superseded_by_hash`` instead: the ``content_hash`` of the successor. Import
inserts every row first and resolves those links afterwards, which is what keeps
history intact across a round trip without trusting foreign ids.

Nothing here opens a connection or takes a lock: callers pass connections from
:mod:`tiered_memory.store`, whose ``BEGIN IMMEDIATE`` transaction is what makes
the read-then-write in :func:`apply` atomic.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import facts, schema

__all__ = [
    "HEADER_SCHEMA_VERSION",
    "ExportFormatError",
    "ExportHeader",
    "ExportResult",
    "FactRecord",
    "ImportMismatchError",
    "ImportResult",
    "apply",
    "export",
    "parse",
    "read",
    "require_identity",
]

#: Bumped when the export layout changes; ``read`` accepts only this value.
HEADER_SCHEMA_VERSION = 1

_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

#: Columns copied verbatim; ``subject_key`` is recomputed from ``kind``/``text``
#: so a file can never carry a subject the local tokeniser would not produce.
_SELECT_RECORDS = (
    "SELECT id, content_hash, text, kind, tier, confidence, seen_count,"
    " distinct_sessions, created_at, updated_at, last_seen_at, source_tool,"
    " source_session, superseded_by FROM facts WHERE deleted_at IS NULL ORDER BY id"
)

_REQUIRED_STRINGS = (
    "content_hash",
    "text",
    "kind",
    "tier",
    "created_at",
    "updated_at",
    "last_seen_at",
)


class ExportFormatError(ValueError):
    """The file is not an export this module can read, or not one this store may take."""


class ImportMismatchError(ExportFormatError):
    """The export's embedding identity is not the target store's.

    Mirrors the plan's dimension guard (R9): the message quotes both the values
    this store expects and the values the file carries, so the repair is obvious.
    """

    def __init__(
        self,
        *,
        expected_model: str | None,
        actual_model: str | None,
        expected_dim: int | None,
        actual_dim: int | None,
    ) -> None:
        self.expected_model = expected_model
        self.actual_model = actual_model
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim
        super().__init__(
            f"export was written for embedding {actual_model!r} ({actual_dim} dimensions), "
            f"but this store expects {expected_model!r} ({expected_dim} dimensions). "
            "Re-export with the local model, or reindex the store to the export's model."
        )


@dataclass(frozen=True, slots=True)
class ExportHeader:
    """The first line of an export: what wrote it, and in which vector space."""

    schema_version: int
    exported_at: str
    tier: str | None
    scope_key: str | None
    embedding_model: str
    embedding_dim: int


@dataclass(frozen=True, slots=True)
class FactRecord:
    """One exported fact. ``superseded_by_hash`` names the successor, not its id."""

    content_hash: str
    subject_key: str
    text: str
    kind: str
    tier: str
    confidence: float
    seen_count: int
    distinct_sessions: int
    created_at: str
    updated_at: str
    last_seen_at: str
    source_tool: str | None = None
    source_session: str | None = None
    superseded_by_hash: str | None = None


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Outcome of :func:`export`."""

    path: Path
    facts: int


@dataclass(frozen=True, slots=True)
class ImportResult:
    """Outcome of :func:`apply`: new rows, merged conflicts and no-op records."""

    imported: int
    merged: int
    skipped: int


def export(
    stores: Sequence[sqlite3.Connection],
    path: str | Path,
    *,
    tier: str | None = None,
) -> ExportResult:
    """Write ``stores`` as JSONL to ``path``, atomically; return the fact count.

    Soft-deleted rows are omitted: they are forgotten, and an export must not
    resurrect them. Superseded rows are kept, with their successor named by
    content hash, so history survives the round trip.

    Args:
        stores: connections in export order; each must be a store transaction.
        path: destination file, replaced atomically.
        tier: header tier; when omitted it is derived from the stores, and is
            ``None`` when they disagree (a multi-store export).
    """
    target = Path(path)
    header = _header_for(stores, tier=tier)
    lines = [json.dumps(asdict(header), sort_keys=True)]
    count = 0
    for conn in stores:
        for record in _records_of(conn):
            lines.append(json.dumps(asdict(record), sort_keys=True))
            count += 1
    _atomic_write(target, lines)
    return ExportResult(path=target, facts=count)


def parse(path: str | Path) -> tuple[ExportHeader, list[FactRecord]]:
    """Parse an export, validating its shape but not its embedding identity.

    Use this when the target store is not yet known -- an import may touch
    several stores, each with its own recorded model. Pair it with
    :func:`require_identity` once the destination is open.

    Raises:
        ExportFormatError: the file is unreadable, the header is not this
            layout, or a line is not a valid record.
    """
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExportFormatError(f"cannot read export file {target}: {exc}") from exc
    lines = text.splitlines()
    if not lines or not lines[0].strip():
        raise ExportFormatError(f"{target} is empty: an export starts with a header line")
    header = _parse_header(lines[0], target)
    records = [
        _parse_record(line, number, target)
        for number, line in enumerate(lines[1:], start=2)
        if line.strip()
    ]
    return header, records


def read(
    path: str | Path,
    *,
    expected_model: str | None = None,
    expected_dim: int | None = None,
) -> tuple[ExportHeader, list[FactRecord]]:
    """Parse an export and check it against the local embedding identity.

    Args:
        path: the JSONL file.
        expected_model: the target store's ``embedding_model``; defaults to the
            schema's model.
        expected_dim: the target store's ``embedding_dim``; defaults to the
            schema's width.

    Raises:
        ImportMismatchError: the header names another model or width.
        ExportFormatError: the file cannot be parsed.
    """
    header, records = parse(path)
    require_identity(
        header,
        expected_model=schema.DEFAULT_EMBEDDING_MODEL if expected_model is None else expected_model,
        expected_dim=schema.EMBEDDING_DIMENSION if expected_dim is None else expected_dim,
    )
    return header, records


def require_identity(header: ExportHeader, *, expected_model: str, expected_dim: int) -> None:
    """Refuse ``header`` unless it names exactly ``expected_model``/``expected_dim``.

    Raises:
        ImportMismatchError: the header names another model or width. The error
            carries both the expected and the actual values.
    """
    if header.embedding_model != expected_model or header.embedding_dim != expected_dim:
        raise ImportMismatchError(
            expected_model=expected_model,
            actual_model=header.embedding_model,
            expected_dim=expected_dim,
            actual_dim=header.embedding_dim,
        )


def apply(
    conn: sqlite3.Connection,
    records: Sequence[FactRecord],
    *,
    retier: bool = False,
) -> ImportResult:
    """Merge ``records`` into the open store; return imported/merged/skipped counts.

    Must run inside a :mod:`tiered_memory.store` transaction. Records are
    applied in file order, so an export whose successor follows its predecessor
    resolves its history links in the second pass. Tier and scope recorded on a
    new row come from the store's own ``meta``, never from the file, unless
    ``retier`` is set -- in which case the file's tier is ignored for routing and
    every record lands in this store.

    Args:
        conn: connection handed out by a :mod:`tiered_memory.store` transaction.
        records: validated records from :func:`read`.
        retier: accept records of another tier and store them here.

    Raises:
        ExportFormatError: a record belongs to another tier, names a kind the
            schema does not accept, or points at a successor absent from ``records``.
    """
    meta = schema.read_meta(conn)
    store_tier = meta["tier"]
    store_scope = meta["scope_key"]
    _check_records(conn, records, store_tier=store_tier, retier=retier)
    imported = merged = skipped = 0
    links: list[tuple[str, str]] = []

    for record in records:
        existing = _existing(conn, record.content_hash)
        if existing is None:
            _insert(conn, record, tier=store_tier, scope_key=store_scope)
            imported += 1
            if record.superseded_by_hash is not None:
                links.append((record.content_hash, record.superseded_by_hash))
        elif _already_present(existing, record):
            skipped += 1
        else:
            _merge(conn, existing, record)
            merged += 1

    _resolve_links(conn, links)
    return ImportResult(imported=imported, merged=merged, skipped=skipped)


def _check_records(
    conn: sqlite3.Connection,
    records: Sequence[FactRecord],
    *,
    store_tier: str,
    retier: bool,
) -> None:
    """Refuse the whole import before writing anything.

    A file that names another tier, or points at a successor it does not carry,
    is a defect in the file, not a partial import: discovering it halfway
    through would leave the store with rows the operator never asked for.
    """
    present = {record.content_hash for record in records}
    stored = {str(row[0]) for row in conn.execute("SELECT content_hash FROM facts")}
    for record in records:
        if not retier and record.tier != store_tier:
            raise ExportFormatError(
                f"record {record.content_hash[:12]!r} belongs to tier {record.tier!r}, but "
                f"this store is {store_tier!r}; pass retier=true to import it here"
            )
        successor = record.superseded_by_hash
        if successor is not None and successor not in present and successor not in stored:
            raise ExportFormatError(
                f"record {record.content_hash[:12]!r} is superseded by {successor[:12]!r}, "
                "which is neither in this import nor already stored"
            )


def _header_for(stores: Sequence[sqlite3.Connection], *, tier: str | None) -> ExportHeader:
    metas = [schema.read_meta(conn) for conn in stores]
    tiers = {meta.get("tier") for meta in metas}
    scopes = {meta.get("scope_key") for meta in metas}
    model = metas[0].get("embedding_model") if metas else None
    dim = _as_int(metas[0].get("embedding_dim")) if metas else None
    return ExportHeader(
        schema_version=HEADER_SCHEMA_VERSION,
        exported_at=datetime.now(UTC).strftime(_TIMESTAMP_FORMAT),
        tier=tier if tier is not None else (tiers.pop() if len(tiers) == 1 else None),
        scope_key=scopes.pop() if len(scopes) == 1 else None,
        embedding_model=model or schema.DEFAULT_EMBEDDING_MODEL,
        embedding_dim=dim if dim is not None else schema.EMBEDDING_DIMENSION,
    )


def _records_of(conn: sqlite3.Connection) -> list[FactRecord]:
    """Every fact of ``conn``, with the successor's hash in place of its id."""
    hashes = {
        int(row[0]): str(row[1]) for row in conn.execute("SELECT id, content_hash FROM facts")
    }
    records: list[FactRecord] = []
    for row in conn.execute(_SELECT_RECORDS):
        successor = row[13]
        records.append(
            FactRecord(
                content_hash=str(row[1]),
                subject_key=facts.subject_key(str(row[3]), str(row[2])),
                text=str(row[2]),
                kind=str(row[3]),
                tier=str(row[4]),
                confidence=float(row[5]),
                seen_count=int(row[6]),
                distinct_sessions=int(row[7]),
                created_at=str(row[8]),
                updated_at=str(row[9]),
                last_seen_at=str(row[10]),
                source_tool=None if row[11] is None else str(row[11]),
                source_session=None if row[12] is None else str(row[12]),
                superseded_by_hash=hashes.get(int(successor)) if successor is not None else None,
            )
        )
    return records


def _parse_header(line: str, path: Path) -> ExportHeader:
    data = _object(line, 1, path)
    version = data.get("schema_version")
    if version != HEADER_SCHEMA_VERSION:
        raise ExportFormatError(
            f"{path} line 1: schema_version is {version!r}, but this plugin writes and "
            f"accepts exactly {HEADER_SCHEMA_VERSION!r}"
        )
    tier = data.get("tier")
    if tier is not None and tier not in schema.VALID_TIERS:
        raise ExportFormatError(
            f"{path} line 1: tier is {tier!r}, not one of {sorted(schema.VALID_TIERS)}"
        )
    model = data.get("embedding_model")
    dim = data.get("embedding_dim")
    if not isinstance(model, str) or not isinstance(dim, int):
        raise ExportFormatError(
            f"{path} line 1: embedding_model and embedding_dim must be a string and an integer"
        )
    return ExportHeader(
        schema_version=HEADER_SCHEMA_VERSION,
        exported_at=str(data.get("exported_at", "")),
        tier=tier if isinstance(tier, str) else None,
        scope_key=data.get("scope_key") if isinstance(data.get("scope_key"), str) else None,
        embedding_model=model,
        embedding_dim=dim,
    )


def _parse_record(line: str, number: int, path: Path) -> FactRecord:
    data = _object(line, number, path)
    for key in _REQUIRED_STRINGS:
        value = data.get(key)
        if not isinstance(value, str) or not value:
            raise ExportFormatError(f"{path} line {number}: {key} must be a non-empty string")
    text = str(data["text"])
    kind = str(data["kind"])
    tier = str(data["tier"])
    if kind not in facts.KINDS:
        raise ExportFormatError(
            f"{path} line {number}: kind {kind!r} is not one of {sorted(facts.KINDS)}"
        )
    if tier not in schema.VALID_TIERS:
        raise ExportFormatError(
            f"{path} line {number}: tier {tier!r} is not one of {sorted(schema.VALID_TIERS)}"
        )
    if data["content_hash"] != facts.content_hash(text):
        raise ExportFormatError(
            f"{path} line {number}: content_hash does not match the record's text"
        )
    for key in ("created_at", "updated_at", "last_seen_at"):
        _check_timestamp(str(data[key]), key=key, number=number, path=path)
    return FactRecord(
        content_hash=str(data["content_hash"]),
        subject_key=facts.subject_key(kind, text),
        text=text,
        kind=kind,
        tier=tier,
        confidence=_number(data, "confidence", number=number, path=path),
        seen_count=_integer(data, "seen_count", number=number, path=path),
        distinct_sessions=_integer(data, "distinct_sessions", number=number, path=path),
        created_at=str(data["created_at"]),
        updated_at=str(data["updated_at"]),
        last_seen_at=str(data["last_seen_at"]),
        source_tool=_optional_string(data, "source_tool"),
        source_session=_optional_string(data, "source_session"),
        superseded_by_hash=_optional_string(data, "superseded_by_hash"),
    )


def _object(line: str, number: int, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ExportFormatError(f"{path} line {number}: not valid JSON ({exc.msg})") from exc
    if not isinstance(value, dict):
        raise ExportFormatError(f"{path} line {number}: expected a JSON object")
    return value


def _check_timestamp(value: str, *, key: str, number: int, path: Path) -> None:
    try:
        datetime.strptime(value, _TIMESTAMP_FORMAT)
    except ValueError:
        raise ExportFormatError(
            f"{path} line {number}: {key} must be {_TIMESTAMP_FORMAT!r}, got {value!r}"
        ) from None


def _number(data: dict[str, Any], key: str, *, number: int, path: Path) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ExportFormatError(f"{path} line {number}: {key} must be a number")
    return float(value)


def _integer(data: dict[str, Any], key: str, *, number: int, path: Path) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExportFormatError(f"{path} line {number}: {key} must be a non-negative integer")
    return value


def _optional_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) and value else None


def _existing(conn: sqlite3.Connection, digest: str) -> tuple[Any, ...] | None:
    row = conn.execute(
        "SELECT id, seen_count, confidence, last_seen_at, source_session"
        " FROM facts WHERE content_hash = ?",
        (digest,),
    ).fetchone()
    return None if row is None else tuple(row)


def _already_present(existing: tuple[Any, ...], record: FactRecord) -> bool:
    """Whether the stored row already carries exactly this record's evidence."""
    return (
        int(existing[1]) == record.seen_count
        and float(existing[2]) == record.confidence
        and str(existing[3]) == record.last_seen_at
        and existing[4] == record.source_session
    )


def _insert(conn: sqlite3.Connection, record: FactRecord, *, tier: str, scope_key: str) -> None:
    conn.execute(
        "INSERT INTO facts (content_hash, subject_key, text, kind, tier, scope_key,"
        " source_tool, source_session, confidence, seen_count, distinct_sessions,"
        " created_at, updated_at, last_seen_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record.content_hash,
            record.subject_key,
            record.text,
            record.kind,
            tier,
            scope_key,
            record.source_tool,
            record.source_session,
            record.confidence,
            record.seen_count,
            record.distinct_sessions,
            record.created_at,
            record.updated_at,
            record.last_seen_at,
        ),
    )


def _merge(conn: sqlite3.Connection, existing: tuple[Any, ...], record: FactRecord) -> None:
    new_session = record.source_session is not None and record.source_session != existing[4]
    conn.execute(
        "UPDATE facts SET seen_count = seen_count + 1,"
        " distinct_sessions = distinct_sessions + ?,"
        " confidence = max(confidence, ?),"
        " last_seen_at = max(last_seen_at, ?),"
        " updated_at = max(updated_at, ?),"
        " source_tool = coalesce(source_tool, ?)"
        " WHERE id = ?",
        (
            int(new_session),
            record.confidence,
            record.last_seen_at,
            record.updated_at,
            record.source_tool,
            int(existing[0]),
        ),
    )


def _resolve_links(conn: sqlite3.Connection, links: Sequence[tuple[str, str]]) -> None:
    for own_hash, successor_hash in links:
        successor = conn.execute(
            "SELECT id FROM facts WHERE content_hash = ?", (successor_hash,)
        ).fetchone()
        if successor is None:  # pragma: no cover - _check_records proved it exists
            raise ExportFormatError(
                f"record {own_hash[:12]!r} is superseded by {successor_hash[:12]!r}, "
                "which is not in this import"
            )
        conn.execute(
            "UPDATE facts SET superseded_by = ? WHERE content_hash = ?",
            (int(successor[0]), own_hash),
        )


def _atomic_write(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            for line in lines:
                stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


def _as_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None
