"""Behaviour of ``tiered_memory.facts`` (plan §2.2.2, §2.2.6, Step 4 / TM-04).

The claims under test come straight from the plan:

* normalisation strips fenced code, collapses whitespace, trims, truncates to
  500 characters and lowercases (§2.2.2);
* ``content_hash`` is ``sha256`` of the normalised text, so two spellings that
  normalise alike are one fact;
* ``subject_key`` is ``kind + ":" + up to 6 salient non-stopword tokens``;
* writing the same text twice merges into one row (``seen_count++``), and
  ``distinct_sessions`` counts sessions, not writes;
* superseding keeps the old row and its text, sets ``superseded_by``, and the
  old row is then absent from the live query (§2.2.6).

Every test uses a real store opened through :mod:`tiered_memory.store` and real
SQLite: the dedupe and supersede guarantees are properties of the file, so a
fake connection would prove nothing.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import tiered_memory.config as config
import tiered_memory.facts as facts
import tiered_memory.store as store


@pytest.fixture(autouse=True)
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate every test in its own plugin data root, helpers included."""
    root = tmp_path / "plugin-data"
    monkeypatch.setenv(config.DATA_ENV, str(root))
    return root


@pytest.fixture()
def conn(data_root: Path) -> Iterator[sqlite3.Connection]:
    with store.global_store() as opened:
        assert isinstance(opened, sqlite3.Connection)
        yield opened


def _query(sql: str, params: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    """Read through a fresh store transaction, as a later process would."""
    with store.global_store() as opened:
        assert isinstance(opened, sqlite3.Connection)
        return [tuple(row) for row in opened.execute(sql, params)]


def _write(text: str, **kwargs: object) -> facts.WriteResult:
    with store.global_store() as opened:
        assert isinstance(opened, sqlite3.Connection)
        return facts.insert(opened, text, **kwargs)


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #


def test_normalise_strips_fenced_code_and_its_contents() -> None:
    text = "Always use uv.\n```python\npip install thing\n```\nNever pip."
    assert facts.normalise(text) == "always use uv. never pip."


def test_normalise_strips_an_unterminated_fence_to_the_end() -> None:
    assert facts.normalise("Use uv.\n```\nuv sync") == "use uv."


def test_normalise_collapses_whitespace_and_trims() -> None:
    assert facts.normalise("  a\t\n b   c \r\n ") == "a b c"


def test_normalise_lowercases_unicode_without_mangling_it() -> None:
    assert facts.normalise("  ÜBER   Café  ") == "über café"


def test_normalise_truncates_to_500_characters() -> None:
    assert len(facts.normalise("x" * 600)) == 500


def test_normalise_truncates_before_lowercasing() -> None:
    assert facts.normalise("A" * 600) == "a" * 500


def test_normalise_leaves_inline_backticks_alone() -> None:
    assert facts.normalise("`content_hash` is sha256") == "`content_hash` is sha256"


@pytest.mark.parametrize("text", ["", "   \n\t ", "```\nonly code\n```"])
def test_normalise_refuses_text_with_nothing_left(text: str) -> None:
    with pytest.raises(ValueError):
        facts.normalise(text)


# --------------------------------------------------------------------------- #
# content_hash
# --------------------------------------------------------------------------- #


def test_content_hash_is_sha256_of_the_normalised_text() -> None:
    assert facts.content_hash("Always use uv.") == hashlib.sha256(b"always use uv.").hexdigest()


def test_content_hash_is_stable_across_whitespace_and_case() -> None:
    assert facts.content_hash("  Always   USE uv ") == facts.content_hash("always use uv")


def test_content_hash_ignores_fenced_code() -> None:
    with_code = "Always use uv.\n```\nuv pip install x\n```"
    assert facts.content_hash(with_code) == facts.content_hash("always use uv.")


def test_content_hash_separates_different_texts() -> None:
    assert facts.content_hash("use uv") != facts.content_hash("use pip")


# --------------------------------------------------------------------------- #
# subject_key
# --------------------------------------------------------------------------- #


def test_subject_key_prefixes_the_kind_and_drops_stopwords() -> None:
    assert facts.subject_key("decision", "Locked SQLite over Hindsight") == (
        "decision:locked sqlite hindsight"
    )


def test_subject_key_strips_punctuation() -> None:
    assert facts.subject_key("fact", "Store, in the plugin's data dir.") == (
        "fact:store plugin data dir"
    )


def test_subject_key_caps_at_six_tokens() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta"
    assert facts.subject_key("fact", text) == "fact:alpha beta gamma delta epsilon zeta"


def test_subject_key_deduplicates_tokens_case_insensitively() -> None:
    assert facts.subject_key("fact", "sqlite SQLite sqlite works") == "fact:sqlite works"


def test_subject_key_keeps_unicode_tokens() -> None:
    assert facts.subject_key("fact", "Café über naïve") == "fact:café über naïve"


def test_subject_key_has_no_tokens_when_everything_is_a_stopword() -> None:
    assert facts.subject_key("fact", "the and of") == "fact:"


def test_subject_key_refuses_an_unknown_kind() -> None:
    with pytest.raises(ValueError):
        facts.subject_key("rumour", "anything")


# --------------------------------------------------------------------------- #
# insert and merge
# --------------------------------------------------------------------------- #


def test_insert_writes_the_normalised_fact(conn: sqlite3.Connection) -> None:
    result = facts.insert(conn, "  Always USE uv. ", kind="preference")

    assert result.deduped is False
    rows = [
        tuple(row)
        for row in conn.execute(
            "SELECT content_hash, subject_key, text, kind, tier, scope_key, confidence,"
            " seen_count, distinct_sessions FROM facts WHERE id = ?",
            (result.id,),
        )
    ]
    assert rows == [
        (
            facts.content_hash("Always USE uv."),
            "preference:always use uv",
            "always use uv.",
            "preference",
            "global",
            "global",
            1.0,
            1,
            1,
        )
    ]


def test_insert_stamps_every_timestamp(conn: sqlite3.Connection) -> None:
    moment = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    result = facts.insert(conn, "use uv", now=moment)

    row = conn.execute(
        "SELECT created_at, updated_at, last_seen_at FROM facts WHERE id = ?", (result.id,)
    ).fetchone()
    assert row == ("2026-01-02T03:04:05Z", "2026-01-02T03:04:05Z", "2026-01-02T03:04:05Z")


def test_same_text_twice_is_one_row_seen_twice() -> None:
    first = _write("Always use uv.", source_session="s1")
    second = _write("  always   USE uv. ", source_session="s1")

    assert first.deduped is False
    assert second.deduped is True
    assert second.id == first.id
    assert _query("SELECT count(*) FROM facts") == [(1,)]
    assert _query("SELECT seen_count, distinct_sessions FROM facts") == [(2, 1)]


def test_two_sessions_make_two_distinct_sessions() -> None:
    _write("Always use uv", source_session="s1")
    _write("  always   USE uv ", source_session="s2")

    assert _query("SELECT seen_count, distinct_sessions FROM facts") == [(2, 2)]


def test_repeating_in_the_same_session_does_not_add_a_distinct_session() -> None:
    _write("Always use uv", source_session="s1")
    _write("always USE uv", source_session="s1")
    _write(" always use uv ", source_session="s1")

    assert _query("SELECT seen_count, distinct_sessions FROM facts") == [(3, 1)]


def test_merge_refreshes_last_seen_but_keeps_the_first_sighting() -> None:
    moment = datetime(2026, 1, 1, tzinfo=UTC)
    _write("use uv", now=moment)
    _write("use uv", now=moment + timedelta(days=2))

    assert _query("SELECT created_at, updated_at, last_seen_at FROM facts") == [
        ("2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z", "2026-01-03T00:00:00Z")
    ]


def test_merge_keeps_the_higher_confidence() -> None:
    _write("use uv", confidence=0.4)
    _write("use uv", confidence=0.9)

    assert _query("SELECT confidence FROM facts") == [(0.9,)]


def test_different_text_is_a_second_row(conn: sqlite3.Connection) -> None:
    first = facts.insert(conn, "use uv")
    second = facts.insert(conn, "use pip")

    assert first.id != second.id
    assert second.deduped is False
    assert len(facts.live_facts(conn)) == 2


def test_insert_refuses_an_unknown_kind(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError):
        facts.insert(conn, "use uv", kind="rumour")
    assert list(conn.execute("SELECT count(*) FROM facts")) == [(0,)]


def test_insert_refuses_empty_text(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError):
        facts.insert(conn, "   ")
    assert list(conn.execute("SELECT count(*) FROM facts")) == [(0,)]


# --------------------------------------------------------------------------- #
# supersede
# --------------------------------------------------------------------------- #


def test_supersede_keeps_the_old_row_and_its_text(conn: sqlite3.Connection) -> None:
    old = facts.insert(conn, "we use pip", kind="decision")
    new = facts.insert(conn, "we use uv", kind="decision")

    facts.supersede(conn, old_id=old.id, new_id=new.id)

    assert [
        tuple(row)
        for row in conn.execute("SELECT text, superseded_by FROM facts WHERE id = ?", (old.id,))
    ] == [("we use pip", new.id)]
    assert facts.live_facts(conn) == [
        facts.Fact(
            id=new.id,
            kind="decision",
            text="we use uv",
            subject_key="decision:use uv",
            seen_count=1,
            distinct_sessions=1,
            source_session=None,
        )
    ]


def test_supersede_hides_the_old_row_from_the_live_query(conn: sqlite3.Connection) -> None:
    old = facts.insert(conn, "we use pip")
    new = facts.insert(conn, "we use uv")
    facts.supersede(conn, old_id=old.id, new_id=new.id)

    assert [fact.id for fact in facts.live_facts(conn)] == [new.id]


def test_supersede_refuses_a_missing_fact(conn: sqlite3.Connection) -> None:
    live = facts.insert(conn, "use uv")

    with pytest.raises(facts.FactNotFound):
        facts.supersede(conn, old_id=999, new_id=live.id)
    with pytest.raises(facts.FactNotFound):
        facts.supersede(conn, old_id=live.id, new_id=999)


def test_supersede_refuses_to_rewrite_history(conn: sqlite3.Connection) -> None:
    old = facts.insert(conn, "we use pip")
    first = facts.insert(conn, "we use uv")
    second = facts.insert(conn, "we use poetry")
    facts.supersede(conn, old_id=old.id, new_id=first.id)

    with pytest.raises(facts.AlreadySuperseded):
        facts.supersede(conn, old_id=old.id, new_id=second.id)
    assert [
        tuple(row)
        for row in conn.execute("SELECT superseded_by FROM facts WHERE id = ?", (old.id,))
    ] == [(first.id,)]


def test_supersede_refuses_self_reference(conn: sqlite3.Connection) -> None:
    live = facts.insert(conn, "use uv")

    with pytest.raises(ValueError):
        facts.supersede(conn, old_id=live.id, new_id=live.id)


# --------------------------------------------------------------------------- #
# live query
# --------------------------------------------------------------------------- #


def test_live_facts_filters_by_kind_and_orders_by_id(conn: sqlite3.Connection) -> None:
    decision = facts.insert(conn, "we use uv", kind="decision")
    fact = facts.insert(conn, "uv is fast", kind="fact")
    todo = facts.insert(conn, "migrate the lockfile", kind="todo")

    assert [entry.id for entry in facts.live_facts(conn)] == [decision.id, fact.id, todo.id]
    assert [entry.id for entry in facts.live_facts(conn, kind="fact")] == [fact.id]
    assert facts.live_facts(conn, kind="gotcha") == []


def test_live_facts_refuses_an_unknown_kind(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError):
        facts.live_facts(conn, kind="rumour")
