"""The inherited context block injected on ``UserPromptSubmit`` (plan §2.2.4, TM-09).

Built from **global + project only**. The session tier is never opened here:
within-session continuity is the host's own conversation state, and a fact
captured this turn must not be echoed back next turn as if it were durable.

The block is deliberately shaped for a prompt that pays for it every turn:

* **silent** -- an empty pair of stores returns ``""``, not a header and not a
  blank line, so an empty memory costs exactly zero tokens;
* **budgeted** -- ``DEFAULT_BUDGET_TOKENS`` (1500) by default, overridable per
  project through the project store's ``meta.budget_tokens``, with every fact
  truncated to ``MAX_FACT_CHARS`` (240);
* **ordered** -- project ``decision``/``architecture`` first, then global
  ``preference``, then everything else by recency-weighted score, so the facts
  that change behaviour are the ones that survive truncation.

Reads are read-only: a prompt hook must never create a store or write a pragma,
because it runs before every turn. A store this module cannot serve is treated
as empty rather than raising; :mod:`tiered_memory.hook_runtime` is what finally
guarantees fail-open, and this module does not lean on it.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from . import config, schema, search

__all__ = [
    "DEFAULT_BUDGET_TOKENS",
    "MAX_FACT_CHARS",
    "build_inherited_block",
    "estimate_tokens",
    "inherited_block",
]

#: §2.2.4: the inherited block's default cost, per turn, in estimated tokens.
DEFAULT_BUDGET_TOKENS = config.INJECTION_TOKEN_BUDGET

#: §2.2.5: every presented fact is truncated to this many characters.
MAX_FACT_CHARS = search.MAX_HIT_CHARS

#: Characters per token in the ``len/4`` estimate (§10, deferred item 3).
_CHARS_PER_TOKEN = 4

#: First group: these project kinds describe how the project is built.
_PROJECT_FIRST_KINDS = frozenset({"decision", "architecture"})

#: Second group: a global preference is the operator's cross-project habit.
_GLOBAL_SECOND_KIND = "preference"


@dataclass(frozen=True, slots=True)
class _Record:
    text: str
    kind: str
    content_hash: str
    tier: str
    hits: int
    last_seen_at: str

    def score(self) -> float:
        """§2.2.5's score without a retrieval rank: decay and use-boost only."""
        return search.score(
            1.0,
            kind=self.kind,
            age_days=search.age_days(self.last_seen_at, datetime.now(UTC)),
            hits=self.hits,
        )


def estimate_tokens(text: str) -> int:
    """Estimate ``text``'s token count as ``ceil(len(text) / 4)``.

    The plan's deferred item 3 pins this estimate: it must agree with the real
    tokenizer within 10 %, and it is the only tokenizer available to a hook that
    must not load the embedding model.
    """
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


def inherited_block(working_dir: str, *, budget_tokens: int | None = None) -> str:
    """Return the block for ``working_dir``, or ``""`` when there is nothing to say.

    Args:
        working_dir: the project whose tier is inherited, plus the global tier.
        budget_tokens: override the project's ``meta.budget_tokens`` and the
            default; ``None`` means "project override, else default".

    Returns:
        Newline-separated facts with no header, or ``""``.
    """
    global_conn = _open_read_only(config.global_db())
    project_conn = _open_read_only(config.project_db(working_dir))
    try:
        chosen = budget_tokens if budget_tokens is not None else _budget(project_conn)
        return build_inherited_block(project_conn, global_conn, budget_tokens=chosen)
    finally:
        for conn in (global_conn, project_conn):
            if conn is not None:
                conn.close()


def build_inherited_block(
    project_conn: sqlite3.Connection | None,
    global_conn: sqlite3.Connection | None,
    *,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
) -> str:
    """Build the block from already-open connections; ``None`` means an empty tier.

    Ordering is §2.2.4's: project ``decision``/``architecture``, then global
    ``preference``, then the remainder by score. A fact present in both tiers
    appears once (the project copy wins), and a fact that would exceed the
    budget is skipped so a shorter one behind it can still be included.
    """
    records = _dedupe(_records(project_conn, tier="project") + _records(global_conn, tier="global"))
    project_first = [
        record
        for record in records
        if record.tier == "project" and record.kind in _PROJECT_FIRST_KINDS
    ]
    global_second = [
        record
        for record in records
        if record.tier == "global" and record.kind == _GLOBAL_SECOND_KIND
    ]
    chosen = {id(record) for record in project_first} | {id(record) for record in global_second}
    remainder = [record for record in records if id(record) not in chosen]

    lines: list[str] = []
    spent = 0
    for record in _ordered(project_first) + _ordered(global_second) + _ordered(remainder):
        text = record.text[:MAX_FACT_CHARS]
        cost = estimate_tokens(text + "\n")
        if spent + cost > budget_tokens:
            continue
        lines.append(text)
        spent += cost
    return "\n".join(lines)


def _ordered(records: list[_Record]) -> list[_Record]:
    """Best score first, ties by text so the block is deterministic."""
    return sorted(records, key=lambda record: (-record.score(), record.text))


def _dedupe(records: list[_Record]) -> list[_Record]:
    seen: set[str] = set()
    unique: list[_Record] = []
    for record in records:
        if record.content_hash in seen:
            continue
        seen.add(record.content_hash)
        unique.append(record)
    return unique


def _records(conn: sqlite3.Connection | None, *, tier: str) -> list[_Record]:
    """Live, undeleted facts from ``conn`` with the inputs the score needs."""
    if conn is None or not _servable(conn):
        return []
    try:
        rows = conn.execute(
            "SELECT text, kind, content_hash, hits, last_seen_at FROM facts"
            " WHERE superseded_by IS NULL AND deleted_at IS NULL ORDER BY id"
        ).fetchall()
    except sqlite3.Error:
        return []
    return [
        _Record(
            text=str(row[0]),
            kind=str(row[1]),
            content_hash=str(row[2]),
            tier=tier,
            hits=int(row[3]),
            last_seen_at=str(row[4]),
        )
        for row in rows
    ]


def _budget(project_conn: sqlite3.Connection | None) -> int:
    """The project's ``meta.budget_tokens`` when it is a positive integer."""
    if project_conn is None:
        return DEFAULT_BUDGET_TOKENS
    try:
        raw = schema.read_meta(project_conn).get("budget_tokens")
    except sqlite3.Error:
        return DEFAULT_BUDGET_TOKENS
    if raw is None:
        return DEFAULT_BUDGET_TOKENS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_BUDGET_TOKENS
    return value if value > 0 else DEFAULT_BUDGET_TOKENS


def _servable(conn: sqlite3.Connection) -> bool:
    """Whether the store's ``schema_version`` is the one this module understands."""
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    except sqlite3.Error:
        return False
    return row is not None and str(row[0]) == str(schema.SCHEMA_VERSION)


def _open_read_only(path: Path) -> sqlite3.Connection | None:
    """Open an existing store read-only, or ``None`` when it is absent or unusable.

    A symlink is refused (as everywhere else in this plugin), and a missing file
    is simply an empty tier -- a read path must never create a store.
    """
    try:
        if path.is_symlink() or not path.exists():
            return None
        return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=1.0)
    except (OSError, sqlite3.Error):
        return None
