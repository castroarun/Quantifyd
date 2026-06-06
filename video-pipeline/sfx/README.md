# sfx/

Drop your impact sound effects here. The slash commands reference these names:

| File | Used for |
|------|----------|
| `whoosh.wav` | fast motion / transitions with no hard impact |
| `hit.wav` | plate clanks, bar racking (metal-on-metal) |
| `bass_drop.wav` | heavy thuds (dumbbell/barbell hitting the floor) |

Use short (`< 1s`), royalty-free WAVs. Good free sources: freesound.org,
Pixabay, the YouTube Audio Library. Keep names exactly as above (the
`/auto-cut` and `/impact-sfx` type→file mapping keys off them), or update the
mapping in `.claude/commands/auto-cut.md` and `impact-sfx.md` if you rename.

> These audio files are gitignored by default (see `video-pipeline/.gitignore`).
