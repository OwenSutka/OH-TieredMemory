---
description: Record one fact in the tiered memory, choosing the tier and kind deliberately.
---

# /remember

Record a fact the user wants kept. This is the explicit write path — the one
place a fact can enter `project` or `global` directly, with no promotion step.

## Usage

```
/remember <text>
/remember --tier project <text>
/remember --tier global <text>
/remember --kind decision --tier project <text>
```

`--tier` defaults to `project`; `--kind` defaults to `fact`.

## How to carry it out

1. **Read the text as one declarative statement.** If it spans several claims,
   ask which one to keep, or record them as separate facts. A fact is one
   statement, not a paragraph.
2. **Pick the kind.** A standing instruction is `preference`; a choice that was
   made is `decision`; how the system is built is `architecture`; a checkable
   statement is `fact`; outstanding work is `todo`; a time-costing trap is
   `gotcha`.
3. **Respect the tier rules.** `--tier global` is refused for anything but a
   `preference`: global memory is shared by every project, and #R2 exists so it
   does not fill with other projects' architecture.
4. **Call the tool.**

   ```
   memory_write(text="always use uv, never pip", tier="project", kind="preference")
   ```

   The result is `{id, deduped, superseded_id}`. A `deduped: true` means the fact
   was already known and its `seen_count` went up; that is a success, not a
   conflict. A non-null `superseded_id` means an older contradicting fact was
   retired — the old row is retained as history.
5. **Report what happened in one line.** "Remembered as a project preference
   (id 12)." Do not re-read the whole store to confirm.

## If the tool refuses

- An unknown `kind` — retry with one of `preference`, `decision`,
  `architecture`, `fact`, `todo`, `gotcha`.
- A credential-shaped value — refuse the write and tell the user why. Nothing
  secret is ever stored.
- A missing `TIERED_MEMORY_DATA` — the MCP server is not configured; say so
  rather than retrying blindly.
