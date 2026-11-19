#!/usr/bin/env python3
"""``PostToolUse`` hook: extract declarative facts into the session store (TM-10).

Async and fail-open by design (plan §3.4, R7): extraction adds nothing to a
session's latency and a broken extractor must never block a tool call. Three
guarantees shape this file:

* **No embedding.** Only :mod:`tiered_memory.extract` and
  :mod:`tiered_memory.store` are imported here (the latter lazily, once a
  candidate exists), so the model cannot be pulled onto the hot path (R6). Rows
  land with ``embedding = NULL`` and the long-lived MCP server backfills them.
* **Session tier only.** Every row is written through ``store.session(...)``,
  which is what R1 means in practice.
* **Bounded.** ``extract.extract`` caps a call at three candidates and
  ``extract.store_candidates`` caps a session at twenty; the overflow is dropped,
  never queued.

The one piece of cross-call state is the previous failure per tool, which is what
turns "error, then success" into a ``gotcha``. It lives in
``${PLUGIN_DATA}/state/post_tool.json`` and is trimmed on every write.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tiered_memory import config, extract, hook_runtime  # noqa: E402

__all__ = ["main"]

#: How many ``(session, tool)`` failures are remembered at once.
_MAX_TRACKED_FAILURES = 64

_STATE_FILE = "post_tool.json"


def _handler(ctx: hook_runtime.HookContext) -> hook_runtime.HookResult | None:
    """Extract and store facts for one tool call; never returns a decision."""
    session_id = ctx.session_id
    if not session_id or not ctx.tool_name:
        return None

    state = _load_state()
    key = f"{session_id}\x00{ctx.tool_name}"
    previous_error = state.get(key)
    if ctx.tool_response and _is_error(ctx.tool_response):
        _remember_failure(state, key, ctx.tool_response)
        return None
    state.pop(key, None)

    candidates = extract.extract(
        tool_name=ctx.tool_name,
        tool_input=ctx.tool_input,
        message=ctx.message,
        previous_error=previous_error,
    )
    if candidates:
        # Imported here, not at module scope: opening a store is the expensive
        # half of this hook (sqlite3 + the schema module), and most tool calls
        # yield no candidate at all, where that cost would be pure waste.
        from tiered_memory import store

        with store.session(session_id, fail_open=True) as conn:
            if isinstance(conn, store.HookSkipped):
                # Plan Step 3 requires that a hook which cannot take the lock
                # "logs and returns 0". store.py emits a logging warning, but a
                # hook's stderr is discarded by the runner, so record the skip in
                # the JSONL trail explicitly or it leaves no trace at all.
                hook_runtime.log_record(
                    name="tiered-memory",
                    event="PostToolUse",
                    tool=ctx.tool_name,
                    session=session_id,
                    extra={"skipped": "lock_contention"},
                )
                return None
            extract.store_candidates(
                conn, candidates, session_id=session_id, tool_name=ctx.tool_name
            )
    _save_state(state)
    return None


def main() -> int:
    """Run the hook; always exits 0 because the event fails open."""
    return hook_runtime.run(_handler, name="tiered-memory", timeout=20)


def _is_error(response: str) -> bool:
    """Whether a tool response reports failure.

    The runner hands an error back as text, so this is a shape test rather than
    an exception: an ``Error``/``Traceback`` heading, or an exit status that is
    not zero.
    """
    head = response.lstrip()[:200].lower()
    return (
        head.startswith("error")
        or "traceback (most recent call last)" in head
        or "exit code: 1" in head
        or "exit status 1" in head
    )


def _remember_failure(state: dict[str, str], key: str, response: str) -> None:
    state[key] = response[:2000]
    while len(state) > _MAX_TRACKED_FAILURES:
        state.pop(next(iter(state)))
    _save_state(state)


def _state_path() -> Path | None:
    """``${PLUGIN_DATA}/state/post_tool.json``, or ``None`` when unconfigured."""
    try:
        return config.ensure_dir(config.data_dir() / "state") / _STATE_FILE
    except config.ConfigError:
        return None


def _load_state() -> dict[str, str]:
    path = _state_path()
    if path is None or not path.exists():
        return {}
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def _save_state(state: dict[str, str]) -> None:
    """Write the state atomically; a failure to persist is not a hook failure."""
    path = _state_path()
    if path is None:
        return
    try:
        handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix="post_tool-")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(state, stream, separators=(",", ":"))
        os.replace(temp_name, path)
    except OSError:
        return


if __name__ == "__main__":
    raise SystemExit(main())
