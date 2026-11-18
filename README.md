# tiered-memory

Self-contained three-tier SQLite memory for OpenHands — **global → project →
session** — replacing the token-heavy native two-tier memory with keyword and
semantic recall fused by recency-weighted Reciprocal Rank Fusion.

The session tier starts empty and inherits from global and project, so a new
session is informed from turn one without carrying the previous session's
detritus. All automatic writes land in the session tier; promotion to a shared
tier is a deliberate event. Isolation is a property of the *file*, not a column.

Repo `OH-TieredMemory` maps to plugin name **`tiered-memory`**.

## Status

**Not installable yet.** This repository is scaffolded (Wave 0, TM-01): the
manifest, launcher and MCP handshake work, and `memory_stats` answers. The
storage engine (schema, tiers, FTS5, embeddings, fusion) and the remaining nine
tools land in Wave 1.

## Layout

```
plugin.json                       manifest (Agent Plugins 1.0.0)
mcp.json                          stdio server declaration
bin/tiered-memory-mcp             launcher, exec'd by mcp.json
src/tiered_memory/
  hook_runtime.py                 vendored hook contract, hash-locked
  config.py                       paths, budgets, caps
  server.py                       MCP stdio dispatch
tests/unit/                       manifest + hash-lock tests
```

## Development

```bash
uv sync
uv run pytest -q
uv run ruff check . && uv run ruff format --check . && uv run mypy src
```

`src/tiered_memory/hook_runtime.py` is a verbatim copy of the control-plane
canonical runtime. Do not edit it here: edit the canonical copy, re-copy, and
update `CANONICAL_SHA256` in `tests/unit/test_hook_runtime_hash.py`.

## Licence

MIT.
