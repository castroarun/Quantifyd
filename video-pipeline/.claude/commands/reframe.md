---
description: Reframe 16:9 footage to 9:16 vertical for Reels/Shorts (fixed crop, split, or tracked)
---

Reframe a 16:9 clip to **9:16 vertical** (1080×1920) for Reels/Shorts/TikTok.
Three strategies, easiest → hardest. Pick based on $ARGUMENTS or ask which the
user wants.

## Strategy A — fixed center crop (easiest, default)

Crop the central 9:16 region. Good when the subject stays roughly centered.

```
ffmpeg -i input.mp4 -vf "crop=ih*9/16:ih,scale=1080:1920" -c:a copy output/vertical.mp4
```

Offer a horizontal offset (`crop=ih*9/16:ih:x=<px>:0`) if the lifter is off to
one side — read shot framing from a sampled frame to choose the offset.

## Strategy B — 50/50 stacked split (easy, good for 2 angles)

Stack two source regions (or two camera angles) into top/bottom halves of the
vertical frame. Useful for showing a wide + close angle at once.

```
ffmpeg -i input.mp4 -filter_complex \
 "[0:v]crop=iw:ih/2:0:0,scale=1080:960[top]; \
  [0:v]crop=iw:ih/2:0:ih/2,scale=1080:960[bot]; \
  [top][bot]vstack" -c:a copy output/vertical_split.mp4
```

## Strategy C — subject-tracked crop (hardest — needs detection + tuning)

Track the lifter and pan the crop window to keep them centered. This is the
trickiest piece and needs per-clip tuning:

1. Run a per-frame person/pose detector (MediaPipe pose or YOLO) to get the
   subject's center-x over time (you can extend `extract_events.py` to dump a
   `track.json` of `{t, cx}`).
2. Smooth the cx path (moving average) to avoid jitter.
3. Generate a time-varying crop. Either:
   - bake keyframed `crop=...:x='<expr>'` with `sendcmd`/`zoompan`, or
   - emit per-segment crops and concat.

State honestly that tracking needs tuning and show the subject-center path you
derived before rendering.

## Output

State which strategy you used and why, show the ffmpeg command, then STOP for
approval before rendering. Always preserve audio (`-c:a copy`) unless the user
is re-mixing.
