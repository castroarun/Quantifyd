# Silent-Video Editing Pipeline

AI-orchestrated editor for **silent footage** (gym/workout, ~4 angles, no
voice). The trick: turn *what HAPPENED* in the video into a structured text
timeline (`events.json`), let Claude reason over that text to make editorial
decisions, and let **ffmpeg** execute them.

```
extract  →  Claude decides  →  ffmpeg renders
(detectors)   (slash command)    (cut + SFX + music)
```

This is **Flavour 2** from the brief (silent video). Flavour 1 (talking-head,
transcript-driven) keys off Whisper instead and isn't built here.

## Layout

```
video-pipeline/
├── .claude/commands/
│   ├── auto-cut.md       # silent-video highlight editor (EDL → render)
│   ├── impact-sfx.md     # SFX placement only
│   ├── music.md          # background music (user / auto / ask)
│   └── reframe.md        # 16:9 → 9:16
├── extractors/
│   └── extract_events.py # runs detectors → events.json
├── sfx/                  # whoosh.wav, hit.wav, bass_drop.wav  (you add these)
├── music/
│   ├── library.json      # mood/bpm tags for auto-selection
│   ├── selected/         # user-supplied track goes here
│   └── *.mp3             # library tracks (you add these)
├── input/                # raw footage (gitignored)
├── output/               # rendered clips (gitignored)
├── sample.events.json    # reference timeline (matches brief schema)
└── events.json           # generated per video (gitignored)
```

## Install

```bash
# system: ffmpeg (required for audio extraction + render)
brew install ffmpeg            # macOS
sudo apt install ffmpeg        # Debian/Ubuntu

# python detectors (install what you need; missing ones degrade to [] cleanly)
cd video-pipeline
pip install -r requirements.txt
```

The extractor is **graceful**: each detector runs only if its library is
present. With nothing installed you still get a valid (empty) `events.json`;
install detectors incrementally and re-run.

## End-to-end on one test clip

1. **Drop footage** into `input/`, e.g. `input/clip.mp4`.

2. **Extract the timeline:**

   ```bash
   # fast first pass — shots + audio + motion (skip the slow pose pass)
   python extractors/extract_events.py input/clip.mp4 -o events.json --no-pose

   # full pass with rep detection once the fast pass looks right
   python extractors/extract_events.py input/clip.mp4 -o events.json --exercise squat
   ```

   Inspect `events.json`. It should read like a clean, time-sorted list of
   shots, audio impacts, motion peaks, and reps. (See `sample.events.json` for
   the target shape.)

3. **Add assets:** put `whoosh.wav` / `hit.wav` / `bass_drop.wav` in `sfx/`,
   and (optionally) a music track in `music/selected/` or library tracks in
   `music/` tagged in `library.json`.

4. **Let Claude cut it.** From this directory, run the slash command:

   ```
   /auto-cut          # optionally: /auto-cut 15s
   ```

   Claude reads `events.json`, ranks "crucial moments" (motion peak + audio
   impact + rep bottom converging within 0.4s), and prints a **ranked moment
   table + an EDL** — then STOPS for your approval. Edit the EDL, approve, and
   it writes the ffmpeg render to `output/highlight.mp4`.

5. **Music + SFX:**

   ```
   /music             # lays the bed, beat-aligns, ducks -8dB around SFX
   /impact-sfx        # (if you want SFX without re-cutting)
   ```

6. **Reframe for Reels (optional):**

   ```
   /reframe           # 16:9 → 9:16 (center crop / split / tracked)
   ```

> The slash commands live in `video-pipeline/.claude/commands/`. To use them as
> Claude Code slash commands, run Claude from inside `video-pipeline/`, or copy
> them into the repo-root `.claude/commands/` to expose them everywhere.

## Tuning thresholds for gym footage

Thresholds live in the `Config` dataclass in `extract_events.py` and are
overridable from the CLI. Re-running the extractor is cheap — tune, inspect
`events.json`, repeat. **Tune one signal at a time.**

| Symptom | Flag | Direction |
|---|---|---|
| Too many tiny "impacts" (breathing, clothing) | `--impact-min-intensity` | raise (e.g. 0.45) |
| Missing real plate clanks | `--onset-delta` | lower (e.g. 0.30 → more sensitive) |
| Motion peaks too noisy / every twitch | `--motion-percentile` | raise (e.g. 93–95) |
| Missing explosive reps | `--motion-percentile` | lower (e.g. 85) |
| Pose pass too slow | `--no-pose`, or `pose_sample_fps` ↓ | lower fps |
| Rep bottoms over-counted (bouncing) | `--rep-prominence` | raise (e.g. 0.10) |
| Reps missed (shallow ROM) | `--rep-prominence` | lower (e.g. 0.04) |
| Over-/under-segmenting camera angles | `--scene-threshold` | raise = fewer cuts |

Examples:

```bash
# stricter impacts, hotter motion sensitivity, deeper-rep requirement
python extractors/extract_events.py input/clip.mp4 \
    --impact-min-intensity 0.45 --motion-percentile 92 --rep-prominence 0.10

# four-angle clip that's being over-cut into too many "shots"
python extractors/extract_events.py input/clip.mp4 --scene-threshold 35
```

Once dialed in for "gym clips," the config is reusable across similar footage
(per the brief — thresholds need per-footage-type tuning, then they stick).

## Honest caveats

- Claude works from **extracted data + sampled frames**, not real-time viewing.
- **Subject-tracked reframe** is the trickiest piece; fixed crop / 50-50 split
  are easy, tracking needs detection + tuning (see `reframe.md` Strategy C).
- **Rendering takes real wall-clock time** — not instant.
- **Always human-review the EDL** before the final render (especially rep
  detection and which moments got kept).
- The `audio_impacts[].type` and pose `exercise` labels start **crude**; Claude
  can sample a few frames to confirm before finalizing.
