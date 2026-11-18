"""Vendored hook runtime for OpenHands plugin hooks.

CANONICAL COPY: OpenHandsPlugins/tooling/templates/hook_runtime.py

Copy this file verbatim into ``src/<pkg>/hook_runtime.py`` in each plugin repo.
Do not edit the copy: edit the canonical file, re-copy, and update
``CANONICAL_SHA256`` in ``tests/unit/test_hook_runtime_hash.py`` in every repo
that vendors it. That test is what stops the four copies drifting apart.

Responsibilities (master plan §5.1):

* read the JSON envelope from stdin, never assuming a key exists
* resolve the event from ``OPENHANDS_EVENT_TYPE`` or the payload ``event_type``
* enforce a soft deadline of ``timeout - 2s``
* fail **open** by default so a broken hook cannot block the user's session,
  and fail **closed** only where a plugin explicitly asks for it
* emit ``{"decision", "reason", "additionalContext"}`` or nothing at all
* append one JSONL line per invocation to ``${PLUGIN_DATA}/logs/hooks.jsonl``,
  rotating at 5 MB and keeping 3 files, so logs never land in the user's project

Stdlib only. Hooks run on every tool call; an import of a third-party package
here would put that package on the hot path of every session.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DECISION_ALLOW",
    "DECISION_DENY",
    "DEFAULT_TIMEOUT",
    "FAIL_CLOSED_EVENTS",
    "FAIL_OPEN_EVENTS",
    "Exit",
    "HookContext",
    "HookResult",
    "log_dir",
    "run",
]

DECISION_ALLOW = "allow"
DECISION_DENY = "deny"

#: A broken hook must not be able to wedge a session, so these events swallow
#: every error. Blocking is only meaningful before an operation, not after it.
FAIL_OPEN_EVENTS = frozenset({"PostToolUse", "SessionStart", "SessionEnd", "Stop"})

#: PreToolUse and UserPromptSubmit can block, so they are the only events where
#: fail-closed is a coherent policy — and even there only when a plugin opts in.
FAIL_CLOSED_EVENTS = frozenset({"PreToolUse", "UserPromptSubmit"})

DEFAULT_TIMEOUT = 60
DEADLINE_MARGIN = 2

LOG_ROTATE_BYTES = 5 * 1024 * 1024
LOG_KEEP = 3
LOG_FILE = "hooks.jsonl"

EXIT_OK = 0
EXIT_BLOCK = 2


class Exit:
    """Exit codes understood by the OpenHands hook runner."""

    OK = EXIT_OK
    BLOCK = EXIT_BLOCK


@dataclass(frozen=True)
class HookResult:
    """What a handler returns. Anything falsy means "no opinion, proceed"."""

    decision: str | None = None
    reason: str | None = None
    additional_context: str | None = None

    @classmethod
    def allow(cls) -> HookResult:
        return cls(decision=DECISION_ALLOW)

    @classmethod
    def deny(cls, reason: str) -> HookResult:
        return cls(decision=DECISION_DENY, reason=reason)

    @classmethod
    def context(cls, text: str) -> HookResult:
        return cls(additional_context=text)

    def blocked(self) -> bool:
        return self.decision == DECISION_DENY


@dataclass
class HookContext:
    """The parsed envelope plus the conveniences every handler wants."""

    envelope: Mapping[str, Any] = field(default_factory=dict)
    event: str = ""

    @property
    def tool_name(self) -> str:
        return os.environ.get("OPENHANDS_TOOL_NAME") or str(self.envelope.get("tool_name") or "")

    @property
    def tool_input(self) -> Mapping[str, Any]:
        value = self.envelope.get("tool_input")
        return value if isinstance(value, Mapping) else {}

    @property
    def tool_response(self) -> str:
        value = self.envelope.get("tool_response")
        return value if isinstance(value, str) else ""

    @property
    def message(self) -> str:
        return str(self.envelope.get("message") or "")

    @property
    def session_id(self) -> str:
        return os.environ.get("OPENHANDS_SESSION_ID") or str(self.envelope.get("session_id") or "")

    @property
    def working_dir(self) -> str:
        return os.environ.get("OPENHANDS_PROJECT_DIR") or str(
            self.envelope.get("working_dir") or ""
        )

    @property
    def command(self) -> str:
        """The shell command, when the tool input carries one."""
        value = self.tool_input.get("command")
        return value if isinstance(value, str) else ""


Handler = Callable[[HookContext], "HookResult | None"]


def _read_envelope() -> dict[str, Any]:
    """Parse stdin. Any malformed or absent payload yields an empty envelope."""
    try:
        if sys.stdin is None or sys.stdin.isatty():
            return {}
        raw = sys.stdin.read()
    except (OSError, ValueError):
        return {}
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _resolve_event(envelope: Mapping[str, Any]) -> str:
    """Env wins over the payload: the runner sets it explicitly per event."""
    from_env = os.environ.get("OPENHANDS_EVENT_TYPE", "").strip()
    if from_env:
        return from_env
    return str(envelope.get("event_type") or "").strip()


def _timeout_budget() -> float:
    """Seconds the runner allows us, minus a margin to exit cleanly."""
    raw = os.environ.get("OPENHANDS_HOOK_TIMEOUT", "").strip()
    try:
        limit = float(raw) if raw else float(DEFAULT_TIMEOUT)
    except ValueError:
        limit = float(DEFAULT_TIMEOUT)
    if limit <= 0:
        limit = float(DEFAULT_TIMEOUT)
    return max(1.0, limit - DEADLINE_MARGIN)


def log_dir() -> Path | None:
    """``${PLUGIN_DATA}/logs``, or None when PLUGIN_DATA is not set.

    Returning None rather than defaulting into the working directory is
    deliberate: hook logs must never appear in the user's project.
    """
    data = os.environ.get("PLUGIN_DATA", "").strip()
    if not data:
        return None
    return Path(data) / "logs"


def _rotate(path: Path) -> None:
    """Size-checked rotation: hooks.jsonl -> .1 -> .2, oldest dropped."""
    try:
        if path.stat().st_size < LOG_ROTATE_BYTES:
            return
    except OSError:
        return
    for index in range(LOG_KEEP - 1, 0, -1):
        older = path.with_suffix(path.suffix + f".{index}")
        newer = path if index == 1 else path.with_suffix(path.suffix + f".{index - 1}")
        try:
            if newer.exists():
                older.unlink(missing_ok=True)
                newer.rename(older)
        except OSError:
            return


def _log(record: Mapping[str, Any]) -> None:
    """Append one JSONL line. Logging never raises into the hook."""
    directory = log_dir()
    if directory is None:
        return
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / LOG_FILE
        _rotate(path)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), separators=(",", ":")) + "\n")
    except OSError:
        return


def _emit(result: HookResult | None) -> None:
    """Print the contract JSON, or nothing when the handler had no opinion."""
    if result is None:
        return
    if not (result.decision or result.reason or result.additional_context):
        return
    payload: dict[str, str] = {}
    if result.decision:
        payload["decision"] = result.decision
    if result.reason:
        payload["reason"] = result.reason
    if result.additional_context:
        payload["additionalContext"] = result.additional_context
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def run(
    handler: Handler,
    *,
    name: str,
    fail_closed: bool = False,
    timeout: float | None = None,
) -> int:
    """Run one hook invocation and return the process exit code.

    ``fail_closed`` only takes effect on events that can block; the caller is
    still responsible for passing it, so the decision stays visible in the
    plugin's own source rather than hidden here.
    """
    started = time.monotonic()
    envelope = _read_envelope()
    event = _resolve_event(envelope)
    ctx = HookContext(envelope=envelope, event=event)

    blocks_on_error = bool(fail_closed) and event in FAIL_CLOSED_EVENTS
    budget = _timeout_budget() if timeout is None else max(1.0, timeout)

    result: HookResult | None = None
    error: str | None = None
    try:
        result = handler(ctx)
    except Exception:  # noqa: BLE001 - a hook must never propagate
        error = traceback.format_exc(limit=3)

    elapsed_ms = int((time.monotonic() - started) * 1000)
    over_budget = elapsed_ms > budget * 1000

    _log(
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "plugin": name,
            "event": event,
            "tool": ctx.tool_name,
            "session": ctx.session_id,
            "ms": elapsed_ms,
            "over_budget": over_budget,
            "decision": (result.decision if result else None)
            or ("deny" if error and blocks_on_error else "allow"),
            "error": (error.strip().splitlines()[-1] if error else None),
        }
    )

    if error is not None:
        if blocks_on_error:
            _emit(HookResult.deny(f"{name}: hook failed and this event fails closed"))
            return EXIT_BLOCK
        return EXIT_OK

    if over_budget and blocks_on_error:
        # Past the runner's patience: do not also block on a stale decision.
        return EXIT_OK

    _emit(result)
    if result is not None and result.blocked():
        return EXIT_BLOCK
    return EXIT_OK
