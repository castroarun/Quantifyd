---
description: Add a background music bed to the cut (user track / ask / auto-pick from mood), with fades + ducking
---

Add background music to the highlight cut. You do NOT listen to audio — you
select by metadata/tags and reason from `events.json`. Three modes with a
fallback chain so it always works:

## Mode selection (in order)

1. **User-supplied track (most reliable).** If the user named a track in
   $ARGUMENTS, or a file exists in `music/selected/`, use it. No intelligence
   needed — just mix.
2. **Auto-pick from the video's temperament.** Infer mood from `events.json`
   (NOT by listening):
   - high `motion_peaks` density **and** frequent hard `audio_impacts`
     (many with intensity > 0.7) → **"high_energy"**
   - sparse motion + few impacts → **"cinematic"** or **"chill"**
   Match that mood against `music/library.json` and propose **2–3 options**
   with their bpm.
3. **Ask the user.** If `events.json` is ambiguous, or `music/library.json` is
   thin/missing the inferred mood, ASK: energy level? genre? BPM? Then pick.

> Honest constraint: you can only auto-pick music you can *identify* by
> filename/tag. You cannot judge an untagged file's mood by listening, and you
> cannot generate original music. Use the royalty-free library; mind licensing
> if publishing.

## Mixing

Once a track is chosen:
- Lay the bed under the cut.
- **Beat alignment (the pro touch):** the chosen track's `bpm` (from
  `library.json`) gives beat spacing = `60/bpm` seconds. Where feasible, nudge
  the first cut to land on the first strong beat, and report which cut points
  align to the beat grid.
- **Fades:** fade in **0.5s**, fade out **1.0s**.
- **Ducking:** duck the bed **-8dB for 0.3s** around each SFX timestamp from the
  `/auto-cut` EDL (volume automation or sidechain compression).

## Output

Show your **pick + reasoning** (which mode fired, why this track, which cuts hit
the beat) and the **ffmpeg audio command** (bed + fades + ducks). STOP and wait
for the user's OK before rendering the final mix.
