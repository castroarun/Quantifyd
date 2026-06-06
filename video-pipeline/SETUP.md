# Run the Silent-Video Editing Pipeline in VS Code — Full Setup Guide

This is the **single, self-contained** instruction file. Follow it top to bottom
on your own machine (with VS Code + Claude Code installed). It takes you from a
fresh clone to a finished, auto-edited workout highlight.

> **The idea in one line:** detectors turn *what HAPPENED* in your silent footage
> into a text timeline (`events.json`); Claude reads that text and decides the
> cut; ffmpeg renders it.
>
> ```
> extract  →  Claude decides  →  ffmpeg renders
> ```

---

## 0. What needs to exist on YOUR machine

Claude Code is the **brain**, but the **muscle** (ffmpeg + python libraries) must
be installed locally — Claude Code does **not** bundle them. You need:

| Thing | Why | How |
|---|---|---|
| **git** | get the code | already have it |
| **ffmpeg** | extract audio + render video | `brew/apt/choco install ffmpeg` |
| **Python 3.11 or 3.12** | run the detectors | python.org or your package manager |
| **python detector libs** | shots/audio/motion/reps | `pip install -r requirements.txt` |
| **VS Code + Claude Code** | the editorial brain + slash commands | you have this |
| **a video clip** | something to edit | your footage in `input/` |

> ⚠️ **Use Python 3.11 or 3.12, not 3.13+.** `mediapipe` (rep detection) lags on
> the newest Python. If you don't care about rep detection, any Python 3.10+ is
> fine — just run with `--no-pose`.

---

## 1. Get the branch

```bash
git fetch origin claude/video-editing-OYDCk
git checkout claude/video-editing-OYDCk
cd video-pipeline
```

Everything below is run from inside the `video-pipeline/` folder.

---

## 2. Install ffmpeg (system dependency)

```bash
# macOS
brew install ffmpeg

# Debian / Ubuntu / WSL
sudo apt update && sudo apt install -y ffmpeg

# Windows (PowerShell, with Chocolatey)
choco install ffmpeg
```

Verify it's on your PATH:

```bash
ffmpeg -version      # should print version info, not "command not found"
```

---

## 3. Install the python detectors

Recommended: use a virtual environment so you don't pollute system Python.

```bash
# create + activate a venv (Python 3.11/3.12)
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\Activate.ps1     # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

**The pipeline degrades gracefully** — if any library fails to install, that one
signal just becomes empty (`[]`) with a warning instead of crashing. Minimum
useful install if you hit trouble:

```bash
pip install numpy scenedetect librosa opencv-python   # everything except reps
# add this only on Python 3.11/3.12 for rep detection:
pip install mediapipe
```

Sanity check what's available:

```bash
python -c "import importlib.util as u; [print(m, 'OK' if u.find_spec(m) else 'MISSING') for m in ['numpy','scenedetect','librosa','cv2','mediapipe']]"
```

---

## 4. Add your assets

```
video-pipeline/
├── input/          ← drop your raw clip here, e.g. input/clip.mp4
├── sfx/            ← whoosh.wav, hit.wav, bass_drop.wav  (short royalty-free WAVs)
└── music/
    ├── selected/   ← OPTIONAL: a music track you want to use, e.g. selected/mytrack.mp3
    ├── *.mp3       ← OPTIONAL: library tracks
    └── library.json← tag each library track with mood + bpm
```

- **SFX:** grab short (`< 1s`) royalty-free WAVs from freesound.org / Pixabay /
  YouTube Audio Library. Name them exactly `whoosh.wav`, `hit.wav`,
  `bass_drop.wav` (or rename in the slash commands).
- **Music:** easiest path is to put ONE track in `music/selected/`. Or add
  library tracks to `music/` and tag them in `music/library.json` like:
  ```json
  { "file": "hype_trap_140.mp3", "mood": "high_energy", "bpm": 140 }
  ```
  Valid moods: `high_energy`, `cinematic`, `chill`. Accurate `bpm` is what
  enables beat-aligned cuts.

> You don't strictly need SFX/music to test the cut — `/auto-cut` works on
> `events.json` alone. Add them when you want the full mix.

---

## 5. Extract the timeline (`events.json`)

This runs the detectors and writes the text timeline Claude will read.

```bash
# fast first pass — shots + audio + motion (skips the slow pose/rep pass)
python extractors/extract_events.py input/clip.mp4 -o events.json --no-pose

# full pass WITH rep detection, once the fast pass looks right
python extractors/extract_events.py input/clip.mp4 -o events.json --exercise squat
```

Open `events.json` and eyeball it. It should be a clean, time-sorted list of
`shots`, `audio_impacts`, `motion_peaks`, and `reps`. Compare against
`sample.events.json` for the expected shape.

### Tuning thresholds for YOUR footage

Re-running is cheap. Tune **one signal at a time**, re-run, re-inspect.

| Symptom | Flag | Direction |
|---|---|---|
| Too many tiny "impacts" (breathing, cloth) | `--impact-min-intensity` | raise → `0.45` |
| Missing real plate clanks | `--onset-delta` | lower → `0.30` (more sensitive) |
| Motion peaks too noisy | `--motion-percentile` | raise → `92`–`95` |
| Missing explosive reps | `--motion-percentile` | lower → `85` |
| Pose pass too slow | `--no-pose` | skip reps |
| Rep bottoms over-counted (bouncing) | `--rep-prominence` | raise → `0.10` |
| Reps missed (shallow ROM) | `--rep-prominence` | lower → `0.04` |
| Camera angles over-/under-split | `--scene-threshold` | raise = fewer cuts |

Example:

```bash
python extractors/extract_events.py input/clip.mp4 \
    --impact-min-intensity 0.45 --motion-percentile 92 --rep-prominence 0.10
```

Once dialed in for "gym clips," the same flags are reusable across similar
footage.

---

## 6. Let Claude cut it (the slash commands)

1. In VS Code, **open the `video-pipeline/` folder as your workspace** (File →
   Open Folder → select `video-pipeline`). This matters: Claude Code finds the
   slash commands in `.claude/commands/` relative to the open folder.

   > Alternative: copy the four `.md` files from
   > `video-pipeline/.claude/commands/` into your repo-root `.claude/commands/`
   > to make the commands available anywhere.

2. Open the Claude Code panel and type:

   ```
   /auto-cut
   ```
   (optionally `/auto-cut 15s` to target a 15-second highlight)

   Claude reads `events.json`, ranks "crucial moments" (a motion peak + audio
   impact + rep bottom converging within 0.4s), and prints a **ranked moment
   table + an editable EDL** — then **STOPS and waits for your approval**.

3. Review/edit the EDL, tell Claude **"OK render"**, and it writes the ffmpeg
   render to `output/highlight.mp4`.

4. Add audio:
   ```
   /music         # lays the music bed, beat-aligns, ducks -8dB around SFX
   /impact-sfx    # places SFX on impacts (use if you skipped music or want SFX only)
   ```

5. Vertical for Reels/Shorts (optional):
   ```
   /reframe       # 16:9 → 9:16  (center crop / 50-50 split / subject-tracked)
   ```

Each command shows its plan/command and **stops for your approval before
rendering** — always human-review before the final render.

---

## 7. Optional: smoke-test with NO real footage

If you want to verify the whole loop before bringing in real video, you can make
a fake "workout" clip with ffmpeg (color bars + tone bursts that look like
impacts to the detectors):

```bash
# 10s test clip with periodic beeps (stand-ins for plate clanks)
ffmpeg -y -f lavfi -i testsrc=size=1280x720:rate=30:duration=10 \
       -f lavfi -i "sine=frequency=800:duration=10" \
       -shortest input/test.mp4

python extractors/extract_events.py input/test.mp4 -o events.json --no-pose
```

Then run `/auto-cut` against it. (Reps will be empty — `testsrc` has no human —
but shots/audio/motion exercise the full plumbing.)

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `ffmpeg: command not found` | ffmpeg isn't on PATH — reinstall (step 2), restart terminal |
| `pip install mediapipe` fails | You're on Python 3.13+. Use 3.11/3.12, or run `--no-pose` |
| `events.json` all empty `[]` | A lib is missing — check the warnings the extractor prints; re-run the sanity check in step 3 |
| Extractor warns "no audio track" | Your clip has no audio → `audio_impacts` will be empty; that's fine, cut still works on motion |
| `/auto-cut` not recognized | Open the `video-pipeline/` folder as the workspace, or copy `.claude/commands/*.md` to repo-root `.claude/commands/` |
| Rep counts look wrong | Tune `--rep-prominence` and `--exercise`; reps are the crudest signal — review them |
| Render is slow | Normal — rendering is real wall-clock work, not instant |

---

## 9. Honest caveats (so nothing over-promises)

- Claude edits from **extracted data + a few sampled frames**, not by watching
  the video in real time.
- **Subject-tracked reframe** is the trickiest piece; fixed crop / 50-50 split
  are easy, tracking needs tuning.
- Audio-impact `type` and pose `exercise` labels start **crude** — Claude can
  sample frames to confirm, but **always human-review the EDL** before the final
  render.
- Thresholds need **per-footage-type tuning** once; then the config is reusable.

---

## Quick reference card

```bash
# one-time setup
git checkout claude/video-editing-OYDCk && cd video-pipeline
brew install ffmpeg                          # or apt/choco
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# per clip
python extractors/extract_events.py input/clip.mp4 -o events.json --no-pose
#   → inspect events.json, tune flags, re-run
# in Claude Code (workspace = video-pipeline/):
#   /auto-cut    → review EDL → "OK render"  → output/highlight.mp4
#   /music  /impact-sfx  /reframe
```
