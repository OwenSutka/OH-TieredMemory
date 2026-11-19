"""Behaviour of the keyword retriever in ``tiered_memory.search`` (plan §2.2.5, Step 5 / TM-05).

The claims under test:

* ``facts_fts MATCH ?`` returns hits ranked by ``bm25``;
* the keyword path is independent of ``sqlite-vec``: a store opened without the
  extension (``index_state = 'fts_only'``) still answers;
* synonym-free partial tokens match through the FTS5 ``porter`` tokeniser, so a
  stem or a shorter inflected form reaches the same row;
* a query full of FTS5 operators or punctuation is a query, never a syntax
  error, because every token is quoted before it reaches ``MATCH``;
* superseded rows are excluded unless ``include_history``, and soft-deleted
  rows are always excluded.

Real stores and real SQLite throughout: ranking and stemming are properties of
the index, so a fake connection would prove nothing.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

import tiered_memory.config as config
import tiered_memory.facts as facts
import tiered_memory.search as search
import tiered_memory.store as store

#: Directory holding the package, so a child interpreter imports the same code
#: under test even when the project is not installed into ``sys.path``.
_SRC_DIR = Path(search.__file__).resolve().parents[1]


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
    return facts.insert(conn, text, kind=kind).id


def _hits(conn: sqlite3.Connection, query: str, **kwargs: object) -> list[search.Hit]:
    """Keyword hits from the fixture's own connection (which holds the write lock)."""
    return search.keyword_hits(conn, query, **kwargs)


# --------------------------------------------------------------------------- #
# match expression
# --------------------------------------------------------------------------- #


def test_match_expression_quotes_every_token_and_ors_them() -> None:
    assert search.match_expression("sqlite-vec MATCH") == '"sqlite" OR "vec" OR "MATCH"'


def test_match_expression_has_no_tokens_for_punctuation_only_input() -> None:
    assert search.match_expression('  - ( ) * " ') == ""


def test_match_expression_keeps_unicode_tokens() -> None:
    assert search.match_expression("Café über") == '"Café" OR "über"'


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #


def test_match_returns_hits_ranked_by_bm25(conn: sqlite3.Connection) -> None:
    sparse = _write(conn, "uv is the installer")
    dense = _write(conn, "uv uv uv is the installer used everywhere")
    unrelated = _write(conn, "rust compiles slowly")

    hits = _hits(conn, "uv")

    assert [hit.id for hit in hits] == [dense, sparse]
    assert unrelated not in [hit.id for hit in hits]
    assert hits[0].score >= hits[1].score


def test_hits_carry_tier_kind_text_and_score(conn: sqlite3.Connection) -> None:
    fact_id = _write(conn, "the lockfile is uv", kind="decision")

    (hit,) = _hits(conn, "lockfile")

    assert hit == search.Hit(
        tier="global",
        id=fact_id,
        text="the lockfile is uv",
        kind="decision",
        score=hit.score,
        superseded=False,
    )
    assert hit.score > 0


def test_limit_caps_the_number_of_hits(conn: sqlite3.Connection) -> None:
    for index in range(5):
        _write(conn, f"lockfile number {index}")

    assert len(_hits(conn, "lockfile")) == 5
    assert len(_hits(conn, "lockfile", limit=2)) == 2


def test_kinds_filter_is_applied(conn: sqlite3.Connection) -> None:
    decision = _write(conn, "lockfile is uv", kind="decision")
    todo = _write(conn, "lockfile cleanup", kind="todo")

    assert [hit.id for hit in _hits(conn, "lockfile", kinds={"todo"})] == [todo]
    both = _hits(conn, "lockfile", kinds={"decision", "todo"})
    assert {hit.id for hit in both} == {decision, todo}


def test_unknown_kind_filter_is_refused(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError):
        _hits(conn, "lockfile", kinds={"rumour"})


# --------------------------------------------------------------------------- #
# history and deletion
# --------------------------------------------------------------------------- #


def test_superseded_rows_are_excluded_unless_history_is_requested(
    conn: sqlite3.Connection,
) -> None:
    old = _write(conn, "we use pip")
    new = _write(conn, "we use uv")
    facts.supersede(conn, old_id=old, new_id=new)

    assert [hit.id for hit in _hits(conn, "use")] == [new]
    history = _hits(conn, "use", include_history=True)
    assert [hit.id for hit in history] == [old, new]
    assert history[0].superseded is True


def test_soft_deleted_rows_are_never_returned(conn: sqlite3.Connection) -> None:
    gone = _write(conn, "the lockfile is uv")
    conn.execute("UPDATE facts SET deleted_at = '2026-01-01T00:00:00Z' WHERE id = ?", (gone,))

    assert _hits(conn, "lockfile", include_history=True) == []


# --------------------------------------------------------------------------- #
# porter stemming
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("query", ["migrating", "migration", "migrat"])
def test_inflected_and_truncated_roots_reach_the_stemmed_row(
    conn: sqlite3.Connection, query: str
) -> None:
    """No synonyms, no prefix trick: ``porter`` folds all three to one stem."""
    target = _write(conn, "we are migrating the lockfile")

    assert [hit.id for hit in _hits(conn, query)] == [target]


def test_stemming_matches_a_shorter_verb_form(conn: sqlite3.Connection) -> None:
    target = _write(conn, "deploying needs the token")

    assert [hit.id for hit in _hits(conn, "deploy")] == [target]


# --------------------------------------------------------------------------- #
# hostile queries
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("query", ['"sqlite-vec" AND (', "*", "^lockfile", "NEAR(a b)", "-"])
def test_fts_operator_characters_are_quoted_not_interpreted(
    conn: sqlite3.Connection, query: str
) -> None:
    _write(conn, "sqlite vec extension")

    assert isinstance(_hits(conn, query), list)


def test_a_query_with_no_tokens_returns_nothing(conn: sqlite3.Connection) -> None:
    _write(conn, "sqlite vec extension")

    assert _hits(conn, "  ()  ") == []


# --------------------------------------------------------------------------- #
# the keyword path does not need sqlite-vec
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


def test_keyword_path_works_with_sqlite_vec_uninstalled(tmp_path: Path) -> None:
    """Test matrix: "FTS-only mode: recall works with sqlite-vec uninstalled"."""
    code = (
        _BLOCK_SQLITE_VEC
        + """
import json
import tiered_memory.facts as facts
import tiered_memory.schema as schema
import tiered_memory.search as search
import tiered_memory.store as store

with store.global_store() as conn:
    facts.insert(conn, "the lockfile is uv", kind="decision")

with store.global_store() as conn:
    state = schema.read_meta(conn)["index_state"]
    hits = search.keyword_hits(conn, "lockfile")

print(json.dumps({"state": state, "texts": [hit.text for hit in hits]}))
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
    assert '"state": "fts_only"' in result.stdout
    assert '"the lockfile is uv"' in result.stdout
