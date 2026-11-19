---
name: remember
description: Decide whether something belongs in the three-tier memory, which tier it belongs in, and which kind to record it as. Use when the user states a preference, a decision, an architecture constraint, a gotcha, or a task to remember, or when recalling earlier context.
---

# Remembering things

Three physical tiers, three different lifetimes. Choosing the tier correctly is
most of the work; the rest is picking a `kind`.

## The rules (non-negotiable)

- **R1** — Automatic writes go to the session store only.
- **R2** — Global promotion requires `kind = preference` **and** observation in
  ≥ 2 distinct sessions, or an explicit `/remember --tier global`.
- **R3** — A session reads only its own + project + global stores. Never
  another session's.
- **R4** — No automatic deletion, ever. Consolidation merges and supersedes; it
  does not evict.

Hooks never write to `project` or `global`; if a fact should outlive the
session, say so with `/remember --tier project`. Architecture decisions are
project-scoped by design and do not belong in a global store however strongly
you feel about them. There is no way to address another session's tier, and
nothing in this plugin tries to.

Within-session recall is **conversation state**, not memory. The session tier is
not injected into the prompt and must never be treated as a scratchpad for
things the conversation already knows.

## Choosing a tier

| Tier | Lifetime | Write it when |
|---|---|---|
| `session` | This session only; promoted at `SessionEnd` | Something you learned now that may be worth keeping but is not yet a rule |
| `project` | Every session in this project | An architecture or decision fact that shapes how this repo is built |
| `global` | Every session, every project | A cross-project preference about how the operator wants work done |

Default `memory_write` to `project`. Use `session` when you are unsure: the
promotion step is where unsure facts earn their place, and a wrong session fact
costs nothing.

## Choosing a kind

| Kind | Use for | Example |
|---|---|---|
| `preference` | A standing instruction about how to work | "always use uv, never pip" |
| `decision` | A choice that was made and would be expensive to revisit | "locked SQLite over Hindsight" |
| `architecture` | How the system is put together | "the store is one SQLite file per tier" |
| `fact` | A durable, checkable statement about the project | "the test suite runs in 27 seconds" |
| `todo` | Work that is still outstanding | "TODO: handle the empty-store case" |
| `gotcha` | A trap that cost time and would cost it again | "sqlite-vec needs load_extension enabled" |

## Using the tools

- `memory_write(text, tier, kind, confidence)` — one declarative sentence. Write
  the statement, not the conversation: "locked SQLite over Hindsight", not
  "the user and I talked about databases".
- `memory_recall(query, scope, limit, kinds)` — ranked hits across the scoped
  tiers. `scope="auto"` means global + project + this session.
- `memory_search_fts(query, scope, limit)` — keyword only; use it when the
  embedding index is still warming, or to check exactly what matched.
- `memory_stats()` — per-tier counts, index state, model and dimension.
- `memory_forget(id, tier)` — soft delete; the row stays as history.
- `memory_prune(tier, older_than_days, dry_run)` — dry run by default; only ever
  removes superseded rows.
- `memory_export(tier, path)` / `memory_import(path, retier)` — JSONL backup and
  restore. The header pins the embedding model and dimension; an import never
  reads vectors from the file.
- `memory_reindex(tier, force)` — backfill missing vectors, or rebuild them all.
- `memory_warm(wait)` — report warm-up state, loading the model first when
  `wait=true`.

## What not to write

- Secrets. Nothing shaped like a credential is stored, by hook or by tool.
- File contents. A fact is one statement, not a copy of a file.
- Anything already in the conversation. That is what the conversation is for.
