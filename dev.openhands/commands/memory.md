---
description: Inspect the tiered memory — stats, list, search, forget, prune.
---

# /memory

Read the three-tier memory without guessing.

## Usage

```
/memory stats
/memory search <query>
/memory list [--kind <kind>]
/memory forget <id> --tier <tier>
/memory prune [--older-than-days N] [--apply]
```

## The sub-commands

| Sub-command | Tool | What it does |
|---|---|---|
| `stats` | `memory_stats` | Per-tier live counts, index state, model, dimension, warnings |
| `search <query>` | `memory_recall` | Ranked hits with tier, kind, score |
| `list` | `memory_search_fts` | Keyword listing; filter with `--kind` |
| `forget <id>` | `memory_forget` | Soft delete one fact; the row stays as history |
| `prune` | `memory_prune` | Reports prunable rows; deletes only with `--apply` |

## How to carry each one out

**stats** — call `memory_stats()`. Read `index_state`: `ready` means semantic
recall is live, `warming` or `pending` means the keyword path is what answers,
`fts_only` means `sqlite-vec` is unavailable and nothing is broken. Report the
counts per tier and any `warnings`.

**search** — call `memory_recall(query, scope="auto", limit=8)`. Show `tier`,
`kind`, the text, and the score. Scope is applied per store, so a global hit and
a project hit are ranked against their own tier first — say which tier a hit
came from rather than implying one pool.

**list** — call `memory_search_fts(query, limit=20)`. This is the debug path; it
shows exactly what the keyword index matched, with no semantic contribution.

**forget** — confirm the id and tier first, then call `memory_forget(id, tier)`.
`forgotten: false` means no live row has that id; it is not an error, and
repeating a forget is safe. Never edit the database by hand.

**prune** — call `memory_prune(older_than_days=N, dry_run=True)` first and show
the candidate count. Only call it again with `dry_run=False` when the user asked
for deletion explicitly. Pruning removes **only** superseded rows older than the
threshold; a live fact is never eligible (#R4).

## What not to do

- Do not write with this command. `/remember` is the write path.
- Do not report an empty result as a failure. An empty tier is a valid answer.
- Do not infer that a fact was lost because a search missed it: superseded rows
  are excluded unless history is requested.

## Backup and maintenance

Backup and restore are the same two calls: `memory_export(tier, path)` writes
JSONL with no vectors, and `memory_import(path, retier=False)` merges it back,
recomputing vectors locally. Importing the same file twice is a no-op. Use
`memory_reindex(tier, force)` to backfill or rebuild vectors, and
`memory_warm(wait)` to report — or await — model warm-up.
