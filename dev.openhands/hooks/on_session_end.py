#!/usr/bin/env python3
"""``SessionEnd`` hook: consolidate the session into its project store (TM-11).

Promotes session facts to the project tier and eligible preferences to global,
then runs the prune pass in its default dry-run mode, so the automatic path
reports candidates and deletes nothing (R4).

Synchronous with the plan's 60 s budget, and fail-open (R7): a consolidation
failure is logged and the session ends normally. ``RunError`` from a store this
hook cannot open is not something to crash the session over -- the next session's
consolidation, or the next ``SessionStart``, picks the work up because every
step is idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tiered_memory import consolidate, hook_runtime  # noqa: E402

__all__ = ["main"]


def _handler(ctx: hook_runtime.HookContext) -> hook_runtime.HookResult | None:
    """Consolidate this session; a missing id or working directory is a no-op."""
    session_id = ctx.session_id
    working_dir = ctx.working_dir
    if not session_id or not working_dir:
        return None
    consolidate.consolidate(working_dir, session_id)
    return None


def main() -> int:
    """Run the hook; always exits 0 because the event fails open."""
    return hook_runtime.run(_handler, name="tiered-memory", timeout=60)


if __name__ == "__main__":
    raise SystemExit(main())
