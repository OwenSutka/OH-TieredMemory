"""The skill and command docs (plan Step 13 / TM-13).

Documentation that names a tool the server does not ship is worse than no
documentation: it teaches a call that fails. The claims under test:

* every ``memory_*`` name in the skill and the two commands exists in
  ``server.py``;
* the skill states R1-R4 from plan §1.4, which is what the plan's acceptance
  criterion asks for;
* the skill says plainly that within-session recall is conversation state, so an
  agent does not treat the session tier as a scratchpad.
"""

from __future__ import annotations

import re
from pathlib import Path

import tiered_memory.server as srv

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SKILL = _REPO_ROOT / "skills" / "remember" / "SKILL.md"
_REMEMBER = _REPO_ROOT / "dev.openhands" / "commands" / "remember.md"
_MEMORY = _REPO_ROOT / "dev.openhands" / "commands" / "memory.md"

_TOOL_NAME = re.compile(r"\bmemory_[a-z_]+\b")


def _shipped_tools() -> set[str]:
    return {name for name in dir(srv) if name.startswith("memory_")}


def test_the_docs_exist() -> None:
    assert _SKILL.is_file()
    assert _REMEMBER.is_file()
    assert _MEMORY.is_file()


def test_every_tool_named_in_the_docs_exists_in_the_server() -> None:
    shipped = _shipped_tools()
    for path in (_SKILL, _REMEMBER, _MEMORY):
        named = set(_TOOL_NAME.findall(path.read_text(encoding="utf-8")))
        assert named <= shipped, f"{path.name} names unknown tools: {sorted(named - shipped)}"


def test_the_docs_name_all_ten_tools() -> None:
    """A doc set that silently omits half the surface is not a reference."""
    text = "\n".join(path.read_text(encoding="utf-8") for path in (_SKILL, _REMEMBER, _MEMORY))
    named = set(_TOOL_NAME.findall(text))

    assert named == _shipped_tools()


def test_the_skill_states_r1_through_r4() -> None:
    # Markdown wraps at 80 columns, so compare on collapsed whitespace.
    text = " ".join(_SKILL.read_text(encoding="utf-8").split())

    for rule in ("R1", "R2", "R3", "R4"):
        assert rule in text, f"the skill does not state {rule}"
    assert "session store only" in text
    assert "≥ 2 distinct sessions" in text
    assert "Never another session's" in text
    assert "No automatic deletion, ever" in text


def test_the_skill_says_within_session_recall_is_conversation_state() -> None:
    text = _SKILL.read_text(encoding="utf-8").lower()

    assert "conversation state" in text


def test_the_remember_command_names_the_explicit_global_path() -> None:
    text = _REMEMBER.read_text(encoding="utf-8")

    assert "--tier global" in text
    assert "preference" in text


def test_the_memory_command_defaults_prune_to_a_dry_run() -> None:
    text = _MEMORY.read_text(encoding="utf-8")

    assert "dry_run=True" in text or "--apply" in text
