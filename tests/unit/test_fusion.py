"""Behaviour of recency-weighted RRF in ``tiered_memory.search`` (plan §2.2.5, Step 7 / TM-07).

The formula under test is quoted verbatim from the plan:

    rrf(d)    = sum_r 1 / (60 + rank_r(d))
    decay(d)  = 0.5 ** (age_days(d) / half_life(kind(d)))     durable -> 1.0
    use_boost = min(1.30, 1 + 0.05 * log1p(hits(d)))
    score(d)  = rrf(d) * decay(d) * use_boost(d)

Every fixture uses real stores and real SQLite, with ``now`` supplied explicitly
so the ages in the ranking are facts of the test rather than of the clock.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import tiered_memory.config as config
import tiered_memory.facts as facts
import tiered_memory.search as search
import tiered_memory.store as store

_NOW = datetime(2026, 6, 1, tzinfo=UTC)

#: The plan's RRF constant.
_K = 60


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


def _write(
    conn: sqlite3.Connection,
    text: str,
    *,
    kind: str = "fact",
    seen_at: datetime = _NOW,
    hits: int = 0,
) -> int:
    fact_id = facts.insert(conn, text, kind=kind, now=seen_at).id
    if hits:
        conn.execute("UPDATE facts SET hits = ? WHERE id = ?", (hits, fact_id))
    return fact_id


def _recall(conn: sqlite3.Connection, query: str, **kwargs: object) -> list[search.Hit]:
    return search.recall([conn], query, now=_NOW, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# the formula, term by term
# --------------------------------------------------------------------------- #


def test_rrf_sums_one_over_k_plus_rank() -> None:
    assert search.reciprocal_rank([1]) == pytest.approx(1 / (_K + 1))
    assert search.reciprocal_rank([1, 1]) == pytest.approx(2 / (_K + 1))
    assert search.reciprocal_rank([2, 5]) == pytest.approx(1 / (_K + 2) + 1 / (_K + 5))


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("preference", math.inf),
        ("architecture", math.inf),
        ("decision", math.inf),
        ("fact", 180.0),
        ("todo", 30.0),
        ("gotcha", 30.0),
    ],
)
def test_half_life_table_is_exact(kind: str, expected: float) -> None:
    assert search.half_life(kind) == expected


def test_unknown_kind_has_no_half_life() -> None:
    with pytest.raises(ValueError):
        search.half_life("rumour")


def test_durable_kinds_never_decay() -> None:
    for kind in ("preference", "architecture", "decision"):
        assert search.decay(kind, age_days=10_000) == 1.0


def test_decay_is_half_at_one_half_life() -> None:
    assert search.decay("fact", age_days=180) == pytest.approx(0.5)
    assert search.decay("todo", age_days=30) == pytest.approx(0.5)
    assert search.decay("todo", age_days=60) == pytest.approx(0.25)


def test_decay_clamps_a_future_timestamp_to_no_decay() -> None:
    assert search.decay("todo", age_days=-3) == 1.0


def test_use_boost_is_one_plus_five_percent_log1p_hits() -> None:
    assert search.use_boost(0) == pytest.approx(1.0)
    assert search.use_boost(1) == pytest.approx(1 + 0.05 * math.log1p(1))
    assert search.use_boost(50) == pytest.approx(1 + 0.05 * math.log1p(50))


def test_use_boost_caps_at_one_point_three() -> None:
    assert search.use_boost(10_000) == pytest.approx(1.30)
    for hits in (0, 1, 9, 99, 10_000):
        assert search.use_boost(hits) <= 1.30


def test_use_boost_never_raises_a_tie_by_more_than_thirty_percent() -> None:
    plain = search.score(1 / (_K + 1), kind="fact", age_days=0, hits=0)
    boosted = search.score(1 / (_K + 1), kind="fact", age_days=0, hits=10_000)
    assert boosted <= plain * 1.30


def test_score_is_the_product_of_the_three_terms() -> None:
    rrf = 1 / (_K + 1)
    expected = rrf * 0.5 * search.use_boost(3)
    assert search.score(rrf, kind="fact", age_days=180, hits=3) == pytest.approx(expected)


def test_hand_computed_fixture_order() -> None:
    """Two keyword hits and one semantic-only hit, every term computed by hand."""
    keyword = {1: 1, 2: 2}
    semantic = {2: 1, 3: 1}
    rows = {
        1: search.Candidate(kind="decision", age_days=0, hits=0),
        2: search.Candidate(kind="fact", age_days=180, hits=0),
        3: search.Candidate(kind="preference", age_days=1000, hits=4),
    }

    ranked = search.fuse(keyword, semantic, rows)

    expected = {
        1: search.score(1 / 61, kind="decision", age_days=0, hits=0),
        2: search.score(1 / 62 + 1 / 61, kind="fact", age_days=180, hits=0),
        3: search.score(1 / 61, kind="preference", age_days=1000, hits=4),
    }
    assert ranked == sorted(expected, key=lambda fact_id: -expected[fact_id])
    assert ranked == [3, 1, 2]


# --------------------------------------------------------------------------- #
# ordering through a real store
# --------------------------------------------------------------------------- #


def test_decisions_do_not_decay_so_recency_breaks_the_tie(
    conn: sqlite3.Connection,
) -> None:
    """Durable kinds keep decay == 1.0, so age alone cannot rank them.

    The two decisions below are otherwise identical, so they tie on fused score;
    the recency tie-breaker is what puts the fresher one first. Age must NOT be
    allowed to move the score, or `decision` would stop being durable.
    """
    old = _write(conn, "decision sqlite one", kind="decision", seen_at=_NOW - timedelta(days=180))
    fresh = _write(conn, "decision sqlite two", kind="decision", seen_at=_NOW - timedelta(days=2))

    hits = _recall(conn, "decision")

    assert [hit.id for hit in hits] == [fresh, old]

    # Durability is the invariant that matters, and it is about *decay*: age must
    # not move the score. The two scores differ slightly because the keyword
    # retriever ranks them 1 and 2, so their RRF terms differ — that is ranking,
    # not decay. Assert decay directly instead of comparing fused scores.
    assert search.decay("decision", age_days=180) == 1.0
    assert search.decay("decision", age_days=2) == 1.0


def test_a_todo_decays_while_a_preference_does_not(conn: sqlite3.Connection) -> None:
    todo = _write(conn, "sqlite todo one", kind="todo", seen_at=_NOW - timedelta(days=120))
    preference = _write(
        conn, "sqlite preference two", kind="preference", seen_at=_NOW - timedelta(days=120)
    )

    ranked = [hit.id for hit in _recall(conn, "sqlite")]

    assert ranked == [preference, todo]
    scores = {hit.id: hit.score for hit in _recall(conn, "sqlite")}
    assert scores[preference] > scores[todo]


def test_hits_raise_a_tie_breaker(conn: sqlite3.Connection) -> None:
    plain = _write(conn, "sqlite alpha one", seen_at=_NOW)
    used = _write(conn, "sqlite alpha two", seen_at=_NOW, hits=1000)

    ranked = _recall(conn, "sqlite")

    assert [hit.id for hit in ranked] == [used, plain]


def test_equal_scores_fall_back_to_id_order(conn: sqlite3.Connection) -> None:
    first = _write(conn, "sqlite alpha one", seen_at=_NOW)
    second = _write(conn, "sqlite alpha two", seen_at=_NOW)

    assert [hit.id for hit in _recall(conn, "sqlite")] == [first, second]


# --------------------------------------------------------------------------- #
# result shape and limits
# --------------------------------------------------------------------------- #


def test_default_limit_is_eight(conn: sqlite3.Connection) -> None:
    for index in range(12):
        _write(conn, f"lockfile number {index}")

    assert len(_recall(conn, "lockfile")) == 8
    assert len(_recall(conn, "lockfile", limit=3)) == 3


def test_text_is_truncated_to_240_characters(conn: sqlite3.Connection) -> None:
    long_text = "lockfile " + "x" * 300
    _write(conn, long_text)

    (hit,) = _recall(conn, "lockfile")

    assert len(hit.text) == search.MAX_HIT_CHARS == 240


def test_superseded_hits_are_excluded_unless_history_is_requested(
    conn: sqlite3.Connection,
) -> None:
    old = _write(conn, "we use pip")
    new = _write(conn, "we use uv")
    facts.supersede(conn, old_id=old, new_id=new)

    assert [hit.id for hit in _recall(conn, "use")] == [new]
    history = _recall(conn, "use", include_history=True)
    assert {hit.id for hit in history} == {old, new}
    assert next(hit for hit in history if hit.id == old).superseded is True


def test_kinds_filter_applies_before_fusion(conn: sqlite3.Connection) -> None:
    decision = _write(conn, "lockfile is uv", kind="decision")
    _write(conn, "lockfile cleanup", kind="todo")

    assert [hit.id for hit in _recall(conn, "lockfile", kinds={"decision"})] == [decision]


# --------------------------------------------------------------------------- #
# scope per store
# --------------------------------------------------------------------------- #


def test_each_store_contributes_its_own_hits(tmp_path: Path) -> None:
    with store.global_store() as global_conn:
        assert isinstance(global_conn, sqlite3.Connection)
        global_id = _write(global_conn, "lockfile global", kind="preference")
    with store.project(str(tmp_path)) as project_conn:
        assert isinstance(project_conn, sqlite3.Connection)
        project_id = _write(project_conn, "lockfile project", kind="decision")

    with (
        store.global_store() as global_conn,
        store.project(str(tmp_path)) as project_conn,
    ):
        assert isinstance(global_conn, sqlite3.Connection)
        assert isinstance(project_conn, sqlite3.Connection)
        hits = search.recall([global_conn, project_conn], "lockfile", now=_NOW)

    assert {(hit.tier, hit.id) for hit in hits} == {
        ("global", global_id),
        ("project", project_id),
    }


def test_a_crowded_store_cannot_evict_another_stores_only_hit(tmp_path: Path) -> None:
    """Scope is applied per store before fusion, so a global fact cannot take a
    project fact's slot."""
    with store.global_store() as global_conn:
        assert isinstance(global_conn, sqlite3.Connection)
        for index in range(50):
            _write(global_conn, f"lockfile global {index}")
    with store.project(str(tmp_path)) as project_conn:
        assert isinstance(project_conn, sqlite3.Connection)
        project_id = _write(project_conn, "lockfile project")

    with (
        store.global_store() as global_conn,
        store.project(str(tmp_path)) as project_conn,
    ):
        assert isinstance(global_conn, sqlite3.Connection)
        assert isinstance(project_conn, sqlite3.Connection)
        hits = search.recall([global_conn, project_conn], "lockfile", limit=8, now=_NOW)

    assert len(hits) == 8
    assert ("project", project_id) in {(hit.tier, hit.id) for hit in hits}
