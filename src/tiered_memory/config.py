"""Paths, budgets and caps for the memory tiers.

Every path derives from ``TIERED_MEMORY_DATA`` (the plugin's ``PLUGIN_DATA``), so
nothing is ever written into the user's project.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

DATA_ENV = "TIERED_MEMORY_DATA"

#: Tokens the injected inherited block may occupy in a prompt.
INJECTION_TOKEN_BUDGET = 1500

#: Files above this many live facts raise a warning in ``memory_stats``.
FACT_WARNING_THRESHOLD = 50_000

GLOBAL_DB_NAME = "global.db"
TIERS_DIR_NAME = "tiers"


class ConfigError(RuntimeError):
    """Raised when the plugin data directory is not configured."""


def data_dir() -> Path:
    """The plugin data root, from ``TIERED_MEMORY_DATA``."""
    raw = os.environ.get(DATA_ENV, "").strip()
    if not raw:
        raise ConfigError(
            f"{DATA_ENV} is not set. It is supplied by the plugin runtime as "
            "${PLUGIN_DATA}; set it explicitly when running the server by hand."
        )
    return Path(raw)


def tiers_dir() -> Path:
    """Directory holding one SQLite file per tier."""
    return data_dir() / TIERS_DIR_NAME


def logs_dir() -> Path:
    """Hook log directory (see ``hook_runtime.log_dir``)."""
    return data_dir() / "logs"


def global_db() -> Path:
    return tiers_dir() / GLOBAL_DB_NAME


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
    return tiers_dir() / f"session_{session_id}.db"
