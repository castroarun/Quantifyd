---
description: Place impact SFX on a clip's audio_impact timestamps (no cutting), output an ffmpeg mix
---

Place impact sound effects onto an existing clip using `events.json`. This does
**no cutting** — it only adds SFX. Use it standalone (footage you don't want to
re-cut) or after `/auto-cut` on the already-rendered highlight.

## Input

- `events.json` for the timeline (`audio_impacts[]`).
- If running against a rendered highlight whose timestamps differ from the
  source, the user must say so — ask which timeline the SFX times refer to
  (source vs cut) if it's ambiguous.

## Rules

1. For each `audio_impact` above intensity **0.3** (or the threshold in
   $ARGUMENTS), place one SFX exactly on its `t`.
2. **Pick by type** from `sfx/`:
   - `plate_clank` / `bar_rack` → `hit.wav`
   - `thud` → `bass_drop.wav`
   - generic / fast-motion-no-impact → `whoosh.wav`
3. **Scale SFX gain by intensity** — louder impact = hotter SFX (e.g. map
   intensity 0.3→-6dB, 1.0→0dB) so it feels natural, not machine-gunned.
4. Avoid stacking: if two impacts are < 0.15s apart, keep the louder one.

## Output

1. A table: `t | type | intensity | chosen SFX | gain`.
2. The **ffmpeg** command using `adelay` to place each SFX at its timestamp and
   `amix` to blend them with the original audio. Write to `output/with_sfx.mp4`.

Show the table + command, then STOP for approval before rendering.
