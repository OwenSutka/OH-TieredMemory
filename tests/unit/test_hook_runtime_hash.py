"""Hash-lock test for the vendored hook_runtime.py.

CANONICAL COPY: OpenHandsPlugins/tooling/templates/test_hook_runtime_hash.py

Copy verbatim into ``tests/unit/test_hook_runtime_hash.py`` in each repo that
vendors ``hook_runtime.py``. When the canonical runtime changes on purpose, edit
``CANONICAL_SHA256`` here *and* in every copy — a deliberate, reviewable diff in
four places is the point. Silent drift between repos is what this prevents.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: sha256 of OpenHandsPlugins/tooling/templates/hook_runtime.py
CANONICAL_SHA256 = "c12a266d89b58e3debae449a9674fd8cf48ea39f5d35472eabbc3bf8a4114d59"

RUNTIME_FILENAME = "hook_runtime.py"


def _runtime_path() -> Path:
    package = Path(__file__).resolve().parents[2] / "src"
    candidates = sorted(package.glob(f"*/{RUNTIME_FILENAME}"))
    assert candidates, f"no src/*/{RUNTIME_FILENAME} found under {package}"
    assert len(candidates) == 1, f"expected one runtime, found {candidates}"
    return candidates[0]


def test_hook_runtime_matches_canonical_copy() -> None:
    actual = hashlib.sha256(_runtime_path().read_bytes()).hexdigest()
    assert actual == CANONICAL_SHA256, (
        "src/*/hook_runtime.py has drifted from the canonical control-plane copy.\n"
        f"  expected: {CANONICAL_SHA256}\n"
        f"  actual:   {actual}\n"
        "Re-copy OpenHandsPlugins/tooling/templates/hook_runtime.py and update "
        "CANONICAL_SHA256 in every repo that vendors it, or revert the local edit."
    )
