# music/

Background music for the cut. Three ways to drive it (see
`.claude/commands/music.md`):

1. **You supply a track** → put it in `music/selected/`. The `/music` command
   uses it directly (most reliable).
2. **Claude auto-picks** → from `library.json` by mood/bpm tags inferred from
   `events.json`.
3. **Claude asks** → if the temperament is ambiguous.

## Adding library tracks

1. Put the `.mp3` here in `music/`.
2. Add a row to `library.json` with an accurate `mood` and `bpm`:

   ```json
   { "file": "your_track.mp3", "mood": "high_energy", "bpm": 132 }
   ```

   Valid moods used by the command: `high_energy`, `cinematic`, `chill`.
   The `bpm` is what enables beat-aligned cuts — fill it in accurately.

> Claude cannot judge an untagged file's mood by listening, and cannot generate
> original music. Use royalty-free tracks and mind licensing if you publish.
> Audio files here are gitignored by default; `library.json` is committed.
