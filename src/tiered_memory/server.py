"""MCP stdio entry point for tiered-memory.

The scaffold exposes ``memory_stats`` only: it proves the handshake and the
launcher wiring. The remaining nine tools land in TM-08 on top of the storage
engine.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

server = MCPServer(
    "tiered-memory",
    instructions=(
        "Three-tier local memory. Use memory_write to record durable facts, "
        "memory_recall to search them, and memory_stats to inspect the tiers."
    ),
)


@server.tool()
def memory_stats() -> dict[str, Any]:
    """Report per-tier fact counts and semantic index state."""
    return {
        "global": 0,
        "project": 0,
        "session": 0,
        "index_state": "empty",
    }


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
