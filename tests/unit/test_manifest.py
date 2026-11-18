"""Manifest conformance for the tiered-memory plugin package."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
MCP_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"


def _load(name: str) -> dict[str, object]:
    return json.loads((REPO_ROOT / name).read_text(encoding="utf-8"))


def test_plugin_manifest() -> None:
    manifest = _load("plugin.json")
    assert manifest["$schema"] == PLUGIN_SCHEMA
    assert manifest["name"] == "tiered-memory"
    assert manifest["version"] == "0.1.0"
    assert manifest["license"] == "MIT"


def test_mcp_manifest() -> None:
    servers = _load("mcp.json")
    assert servers["$schema"] == MCP_SCHEMA
    entry = servers["mcpServers"]["tiered-memory"]  # type: ignore[index]
    assert entry["type"] == "stdio"
    assert entry["command"] == "./bin/tiered-memory-mcp"


def test_launcher_is_executable() -> None:
    launcher = REPO_ROOT / "bin" / "tiered-memory-mcp"
    assert launcher.exists()
    assert launcher.stat().st_mode & 0o111
