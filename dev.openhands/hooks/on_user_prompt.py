#!/usr/bin/env python3
"""``UserPromptSubmit`` hook: inject the inherited block (plan §2.2.4, TM-09).

Runs on every turn, so it does exactly one thing: build the block from the global
and project stores and hand it to the runner as ``additionalContext``. It never
reads the session tier, never creates a store, and never imports the embedding
model (R6).

Fail-open is not optional here (R7): a broken memory hook must never block a
prompt. ``hook_runtime.run`` swallows any error, logs it and exits 0, and this
module does not pass ``fail_closed=True`` -- so even though the runtime *could*
block on this event, this plugin never asks it to.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tiered_memory import hook_runtime, inject  # noqa: E402

__all__ = ["main"]


def _handler(ctx: hook_runtime.HookContext) -> hook_runtime.HookResult | None:
    """Return the inherited block, or ``None`` when there is nothing to inject."""
    working_dir = ctx.working_dir or os.environ.get("OPENHANDS_PROJECT_DIR", "")
    if not working_dir:
        return None
    block = inject.inherited_block(working_dir)
    if not block:
        return None
    return hook_runtime.HookResult.context(block)


def main() -> int:
    """Run the hook; always exits 0 because the event fails open."""
    return hook_runtime.run(_handler, name="tiered-memory", timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
