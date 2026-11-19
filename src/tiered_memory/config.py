"""Paths, budgets and caps for the memory tiers.

Every path derives from ``TIERED_MEMORY_DATA`` (the plugin's ``PLUGIN_DATA``), so
nothing is ever written into the user's project.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

DATA_ENV = "TIERED_MEMORY_DATA"

#: Tokens the injected inherited block may occupy in a prompt.
INJECTION_TOKEN_BUDGET = 1500

#: Files above this many live facts raise a warning in ``memory_stats``.
FACT_WARNING_THRESHOLD = 50_000

GLOBAL_DB_NAME = "global.db"
TIERS_DIR_NAME = "tiers"

#: A caller-supplied path component (e.g. a session id) is interpolated into a
#: filename, so it may only contain characters that cannot change the path.
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9._-]{1,128}")


class ConfigError(RuntimeError):
    """Raised when the plugin data directory is not configured or unusable."""


class InvalidSessionId(ValueError, ConfigError):
    """A session id is not a safe single path component.

    Subclassing ``ValueError`` lets a caller that validates its own input catch
    it as an argument error, while still being a ``ConfigError`` for the paths
    layer that raised it. ``store.session`` therefore surfaces it either way.
    """


def safe_component(name: str, *, what: str) -> str:
    """Return ``name`` if it is a safe single path component, else refuse.

    A component is safe when it matches ``[A-Za-z0-9._-]{1,128}`` as a whole and
    is neither ``.`` nor ``..``. Anything else could escape the tiers directory
    once interpolated into a filename, so it raises ``InvalidSessionId`` naming
    ``what``.
    """
    if _SAFE_COMPONENT.fullmatch(name) is None or name in {".", ".."}:
        raise InvalidSessionId(
            f"invalid {what} {name!r}: it must be 1-128 characters of "
            "[A-Za-z0-9._-], and must not be '.' or '..'"
        )
    return name


def data_dir() -> Path:
    """The plugin data root, from ``TIERED_MEMORY_DATA``.

    The value is used verbatim only after two checks: it must be absolute, and
    it must not contain a ``..`` component. A relative value would anchor to
    whatever the caller's working directory happens to be -- typically the
    user's project -- and ``..`` would let a configuration value climb out of
    the root the plugin runtime chose. Both raise ``ConfigError``.
    """
    raw = os.environ.get(DATA_ENV, "").strip()
    if not raw:
        raise ConfigError(
            f"{DATA_ENV} is not set. It is supplied by the plugin runtime as "
            "${PLUGIN_DATA}; set it explicitly when running the server by hand."
        )
    path = Path(raw)
    if not path.is_absolute():
        raise ConfigError(
            f"{DATA_ENV} must be an absolute path, got {raw!r}; a relative value "
            "would put the memory store under the caller's working directory."
        )
    if ".." in path.parts:
        raise ConfigError(
            f"{DATA_ENV} must not contain '..' components, got {raw!r}; the store "
            "must stay inside the configured data root."
        )
    return path


def tiers_dir() -> Path:
    """Directory holding one SQLite file per tier."""
    return data_dir() / TIERS_DIR_NAME


def ensure_tiers_dir() -> Path:
    """Create the data root and ``tiers/``, refusing an unsafe tiers path.

    A ``tiers/`` that is a symlink -- dangling or not -- or that resolves
    outside the data root would silently relocate every store, so it is refused
    before anything is created. The data root itself is trusted (the runtime
    supplies it); only the directory this plugin owns below it is checked.
    """
    root = ensure_dir(data_dir())
    tiers = root / TIERS_DIR_NAME
    resolved_root = root.resolve()
    if tiers.is_symlink():
        raise ConfigError(f"{tiers} is a symlink; refusing to write memory stores through it")
    resolved = tiers.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ConfigError(
            f"{tiers} resolves to {resolved}, outside the data root {resolved_root}; "
            "refusing to write memory stores there"
        )
    ensure_dir(tiers)
    if tiers.is_symlink():
        raise ConfigError(f"{tiers} is a symlink; refusing to write memory stores through it")
    return tiers


def logs_dir() -> Path:
    """Hook log directory (see ``hook_runtime.log_dir``)."""
    return data_dir() / "logs"


def global_db() -> Path:
    return tiers_dir() / GLOBAL_DB_NAME


def ensure_dir(path: Path) -> Path:
    """Create ``path`` and its parents, or refuse with ``ConfigError``.

    ``Path.mkdir`` raises a bare ``OSError`` (``FileExistsError`` when the path
    is a regular file); callers such as the fail-open hook path classify only
    ``ConfigError``, so an unusable data directory must surface as one.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigError(f"cannot use {path} as a directory: {exc}") from exc
    return path


def project_key(working_dir: str) -> str:
    """Stable project identity: sha1 of the resolved path, first 12 hex chars.

    Resolving first is what makes two symlinked paths to one project share a
    store instead of forking one.
    """
    resolved = Path(working_dir).resolve()
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()
    return digest[:12]


def project_db(working_dir: str) -> Path:
    return tiers_dir() / f"project_{project_key(working_dir)}.db"


def session_db(session_id: str) -> Path:
    """Store file for one session; ``session_id`` is validated as one component."""
    safe_component(session_id, what="session_id")
    return tiers_dir() / f"session_{session_id}.db"
