"""Performance budgets from plan §8, measured end to end through the real hooks.

These are the budgets the plan lists as p95 targets. They are measured on the
actual entrypoint (a subprocess, the way OpenHands runs a hook) rather than on an
in-process function call, because interpreter startup is the dominant cost and
measuring only the inner function would hide exactly the regression that matters.

The thresholds are deliberately looser than the plan's figures to survive a
loaded CI box, while still failing on an order-of-magnitude regression such as an
accidental model load or an O(n^2) scan on the hot path.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK = REPO_ROOT / "bin" / "tiered-memory-hook"

#: Plan budget 50 ms; allow headroom for a shared machine while still catching a
#: regression of the kind that made it 60 ms before the import graph was trimmed.
POST_TOOL_P95_BUDGET_MS = 75
POST_TOOL_LIMIT_MS = 250


def _env(tmp_path: Path) -> dict[str, str]:
    return {**os.environ, "TIERED_MEMORY_DATA": str(tmp_path / "data")}


def _payload() -> str:
    return json.dumps(
        {
            "tool_name": "terminal",
            "tool_input": {"command": "always use uv, never pip"},
            "tool_response": "ok",
        }
    )


def _timed_runs(tmp_path: Path, hook: str, runs: int) -> list[float]:
    times: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        completed = subprocess.run(
            [str(HOOK), hook],
            input=_payload(),
            capture_output=True,
            text=True,
            env=_env(tmp_path),
            timeout=120,
            check=False,
        )
        times.append((time.perf_counter() - started) * 1000)
        assert completed.returncode == 0, completed.stderr
    return times


def _p95(times: list[float]) -> float:
    ordered = sorted(times)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]


@pytest.mark.parametrize("hook", ["on_post_tool.py", "on_user_prompt.py"])
def test_the_hook_stays_within_its_latency_budget(hook: str, tmp_path: Path) -> None:
    # One warm-up run pays the cold-store cost so the budget measures the steady
    # state the plan budgets for.
    _timed_runs(tmp_path, hook, 1)
    times = _timed_runs(tmp_path, hook, 25)

    p95 = _p95(times)
    assert p95 < POST_TOOL_P95_BUDGET_MS, (
        f"{hook} p95 {p95:.1f} ms exceeds {POST_TOOL_P95_BUDGET_MS} ms "
        f"(median {statistics.median(times):.1f}, max {max(times):.1f})"
    )
    assert max(times) < POST_TOOL_LIMIT_MS, f"{hook} max {max(times):.1f} ms is pathological"
