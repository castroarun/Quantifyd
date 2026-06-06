#!/usr/bin/env python3
"""
extract_events.py — Silent-video "intelligence layer" extractor.

Turns raw footage into a structured, human-readable `events.json` timeline that
Claude (the editorial brain) reads to make cut decisions. The detectors here are
the "dumb muscle"; the JSON they emit is the only thing Claude needs to read.

Four signals (each independent + individually optional):
  - shots         : camera-angle / scene cuts        (PySceneDetect)
  - audio_impacts : plate clanks, thuds, bar racks    (librosa onset detection)
  - motion_peaks  : explosive movement bursts          (OpenCV frame differencing)
  - reps          : squat bottoms / deadlift lockouts  (MediaPipe pose, 2nd pass)

Design rules:
  * GRACEFUL DEGRADATION. Every detector is wrapped so a missing library or a
    detector failure degrades that signal to [] with a warning, instead of
    crashing the whole run. You can produce a partial events.json with whatever
    libraries you have installed.
  * TUNABLE. All thresholds live in the CONFIG dataclass and are overridable
    from the CLI (gym footage needs different sensitivity than product shots).
  * DETERMINISTIC SCHEMA. Output matches the brief's schema exactly, sorted by
    time, rounded for human readability.

Usage:
    python extract_events.py input/clip.mp4
    python extract_events.py input/clip.mp4 -o events.json
    python extract_events.py input/clip.mp4 --no-pose          # skip slow 2nd pass
    python extract_events.py input/clip.mp4 --motion-percentile 92 --onset-delta 0.5
    python extract_events.py input/clip.mp4 --exercise squat   # hint for rep labelling

Dependencies (all optional — install what you need):
    ffmpeg (system)   required for audio extraction + duration
    pip install scenedetect      -> shots
    pip install librosa numpy    -> audio_impacts
    pip install opencv-python    -> motion_peaks
    pip install mediapipe        -> reps   (or adapt to ultralytics YOLO-pose)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Any


# --------------------------------------------------------------------------- #
# Config — tune these for your footage type. Defaults aim at gym/workout clips. #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # --- shot detection ---
    scene_threshold: float = 27.0        # PySceneDetect ContentDetector threshold
    scene_min_len_sec: float = 0.6       # ignore sub-flash "scenes"

    # --- audio onsets / impacts ---
    onset_delta: float = 0.45            # librosa onset peak-pick delta (higher = fewer)
    onset_min_gap_sec: float = 0.20      # merge onsets closer than this
    impact_min_intensity: float = 0.30   # drop weak onsets below this (0..1 normalised)

    # --- motion energy ---
    motion_sample_fps: float = 12.0      # downsample analysis fps (speed vs resolution)
    motion_resize_width: int = 320       # downscale frames for optical-flow/diff speed
    motion_percentile: float = 90.0      # a "peak" = motion above this percentile
    motion_min_gap_sec: float = 0.50     # merge peaks closer than this

    # --- pose / reps ---
    pose_sample_fps: float = 10.0        # pose runs every Nth frame
    rep_min_gap_sec: float = 0.80        # ignore rep bottoms closer than this
    rep_prominence: float = 0.06         # how deep an angle-minimum must be (0..1)
    default_exercise: str = "unknown"    # label used when not hinted

    # --- general ---
    round_dp: int = 2                    # decimals in output timestamps


# --------------------------------------------------------------------------- #
# Small utilities                                                             #
# --------------------------------------------------------------------------- #
def _warn(msg: str) -> None:
    print(f"  [warn] {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    print(f"  {msg}", file=sys.stderr)


def _have(module: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module) is not None


def _ffprobe_duration(video_path: str) -> float:
    """Return duration in seconds via ffprobe; 0.0 if unavailable."""
    if not shutil.which("ffprobe"):
        _warn("ffprobe not found — duration will be inferred from detectors")
        return 0.0
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path,
            ],
            stderr=subprocess.STDOUT,
        )
        return float(out.decode().strip())
    except Exception as e:  # noqa: BLE001
        _warn(f"ffprobe duration failed: {e}")
        return 0.0


def _extract_audio_wav(video_path: str, dst_wav: str, sr: int = 22050) -> bool:
    """Extract mono wav with ffmpeg. Returns True on success."""
    if not shutil.which("ffmpeg"):
        _warn("ffmpeg not found — cannot extract audio for onset detection")
        return False
    try:
        subprocess.check_call(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-ac", "1", "-ar", str(sr), dst_wav,
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return os.path.exists(dst_wav) and os.path.getsize(dst_wav) > 0
    except subprocess.CalledProcessError:
        _warn("ffmpeg audio extraction failed (clip may have no audio track)")
        return False


def _merge_close(times: list[float], min_gap: float) -> list[int]:
    """Given sorted times, return indices to KEEP after merging close ones
    (keeps the first of each cluster)."""
    keep: list[int] = []
    last = -1e9
    for i, t in enumerate(times):
        if t - last >= min_gap:
            keep.append(i)
            last = t
    return keep


# --------------------------------------------------------------------------- #
# Detector 1 — shots (PySceneDetect)                                          #
# --------------------------------------------------------------------------- #
def detect_shots(video_path: str, cfg: Config) -> list[dict[str, Any]]:
    if not _have("scenedetect"):
        _warn("scenedetect not installed -> shots = []  (pip install scenedetect)")
        return []
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        video = open_video(video_path)
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=cfg.scene_threshold))
        sm.detect_scenes(video)
        scene_list = sm.get_scene_list()

        shots: list[dict[str, Any]] = []
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        idx = 0
        for start, end in scene_list:
            s, e = start.get_seconds(), end.get_seconds()
            if e - s < cfg.scene_min_len_sec:
                continue
            shots.append({
                "id": letters[idx % 26],
                "start": round(s, cfg.round_dp),
                "end": round(e, cfg.round_dp),
                # angle is a placeholder label; Claude can sample frames to confirm
                "angle": "unknown",
            })
            idx += 1
        if not shots:
            _warn("no scene cuts found — treating whole clip as one shot")
        return shots
    except Exception as e:  # noqa: BLE001
        _warn(f"shot detection failed: {e} -> shots = []")
        return []


# --------------------------------------------------------------------------- #
# Detector 2 — audio_impacts (librosa onset detection)                        #
# --------------------------------------------------------------------------- #
def detect_audio_impacts(wav_path: str | None, cfg: Config) -> list[dict[str, Any]]:
    if wav_path is None:
        return []
    if not _have("librosa") or not _have("numpy"):
        _warn("librosa/numpy not installed -> audio_impacts = []  (pip install librosa numpy)")
        return []
    try:
        import numpy as np
        import librosa

        y, sr = librosa.load(wav_path, sr=None, mono=True)
        if y.size == 0:
            return []

        # Onset strength envelope -> peak picking.
        env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=env, sr=sr, units="frames",
            delta=cfg.onset_delta, backtrack=False,
        )
        if len(onset_frames) == 0:
            return []
        times = librosa.frames_to_time(onset_frames, sr=sr)

        # Intensity = onset-envelope strength at each onset, normalised 0..1.
        env_max = float(env.max()) or 1.0
        strengths = env[onset_frames] / env_max

        # Spectral centroid -> crude type label (low = thud/bass, high = clank/metal).
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        cent_times = librosa.times_like(cent, sr=sr)

        def label_for(t: float) -> str:
            j = int(np.argmin(np.abs(cent_times - t)))
            hz = float(cent[j])
            if hz < 1200:
                return "thud"
            if hz < 2800:
                return "bar_rack"
            return "plate_clank"

        order = list(range(len(times)))
        keep = _merge_close([times[i] for i in order], cfg.onset_min_gap_sec)

        impacts: list[dict[str, Any]] = []
        for i in keep:
            inten = round(float(strengths[i]), cfg.round_dp)
            if inten < cfg.impact_min_intensity:
                continue
            t = float(times[i])
            impacts.append({
                "t": round(t, cfg.round_dp),
                "intensity": inten,
                "type": label_for(t),
            })
        impacts.sort(key=lambda d: d["t"])
        return impacts
    except Exception as e:  # noqa: BLE001
        _warn(f"audio onset detection failed: {e} -> audio_impacts = []")
        return []


# --------------------------------------------------------------------------- #
# Detector 3 — motion_peaks (OpenCV frame differencing)                       #
# --------------------------------------------------------------------------- #
def detect_motion_peaks(video_path: str, cfg: Config) -> tuple[list[dict[str, Any]], float]:
    """Returns (peaks, duration_seconds_from_video)."""
    if not _have("cv2") or not _have("numpy"):
        _warn("opencv-python/numpy not installed -> motion_peaks = []  (pip install opencv-python numpy)")
        return [], 0.0
    try:
        import numpy as np
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _warn(f"opencv could not open {video_path} -> motion_peaks = []")
            return [], 0.0

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        duration = (frame_count / src_fps) if frame_count else 0.0
        step = max(1, int(round(src_fps / cfg.motion_sample_fps)))

        prev_gray = None
        times: list[float] = []
        energies: list[float] = []
        fidx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fidx % step != 0:
                fidx += 1
                continue
            t = fidx / src_fps
            # downscale for speed
            h, w = frame.shape[:2]
            if w > cfg.motion_resize_width:
                scale = cfg.motion_resize_width / w
                frame = cv2.resize(frame, (cfg.motion_resize_width, int(h * scale)))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                energies.append(float(diff.mean()))
                times.append(t)
            prev_gray = gray
            fidx += 1
        cap.release()

        if not energies:
            return [], duration
        if duration == 0.0 and times:
            duration = times[-1]

        e = np.asarray(energies, dtype=float)
        e_norm = e / (e.max() or 1.0)
        thresh = float(np.percentile(e, cfg.motion_percentile))

        # local maxima above the percentile threshold
        raw: list[tuple[float, float]] = []
        for i in range(1, len(e) - 1):
            if e[i] >= thresh and e[i] >= e[i - 1] and e[i] >= e[i + 1]:
                raw.append((times[i], float(e_norm[i])))
        raw.sort(key=lambda x: x[0])
        keep = _merge_close([t for t, _ in raw], cfg.motion_min_gap_sec)

        peaks = [
            {"t": round(raw[i][0], cfg.round_dp), "energy": round(raw[i][1], cfg.round_dp)}
            for i in keep
        ]
        return peaks, round(duration, cfg.round_dp)
    except Exception as e:  # noqa: BLE001
        _warn(f"motion detection failed: {e} -> motion_peaks = []")
        return [], 0.0


# --------------------------------------------------------------------------- #
# Detector 4 — reps (MediaPipe pose, 2nd pass)                                #
# --------------------------------------------------------------------------- #
def detect_reps(video_path: str, cfg: Config, exercise: str,
                shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect rep bottoms/lockouts by tracking a joint angle (knee for squat,
    hip for deadlift/hinge) and finding its local minima (bottom) + the
    following local maxima (lockout). This is intentionally simple and meant to
    be tuned per footage; Claude can confirm exercise labels from sampled frames.
    """
    if not _have("cv2") or not _have("numpy"):
        _warn("opencv/numpy missing -> reps = []")
        return []
    if not _have("mediapipe"):
        _warn("mediapipe not installed -> reps = []  (pip install mediapipe)")
        return []
    try:
        import numpy as np
        import cv2
        import mediapipe as mp

        mp_pose = mp.solutions.pose
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(src_fps / cfg.pose_sample_fps)))

        # landmark indices
        L = mp_pose.PoseLandmark
        # knee angle = hip-knee-ankle ; hip angle = shoulder-hip-knee
        if exercise in ("deadlift", "hinge", "rdl", "clean"):
            triple = (L.LEFT_SHOULDER, L.LEFT_HIP, L.LEFT_KNEE)
        else:  # squat / default -> knee angle
            triple = (L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE)

        def angle(a, b, c) -> float:
            ba = np.array([a.x - b.x, a.y - b.y])
            bc = np.array([c.x - b.x, c.y - b.y])
            denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) or 1e-9
            cosang = float(np.dot(ba, bc) / denom)
            cosang = max(-1.0, min(1.0, cosang))
            return math.degrees(math.acos(cosang))

        times: list[float] = []
        angles: list[float] = []
        fidx = 0
        with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if fidx % step != 0:
                    fidx += 1
                    continue
                t = fidx / src_fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    ang = angle(lm[triple[0]], lm[triple[1]], lm[triple[2]])
                    times.append(t)
                    angles.append(ang)
                fidx += 1
        cap.release()

        if len(angles) < 5:
            _warn("not enough pose frames to detect reps -> reps = []")
            return []

        a = np.asarray(angles, dtype=float)
        # normalise 0..1 then find local minima (rep bottoms) with prominence
        a_min, a_max = a.min(), a.max()
        rng = (a_max - a_min) or 1.0
        a_n = (a - a_min) / rng

        def shot_at(t: float) -> str:
            for sh in shots:
                if sh["start"] <= t <= sh["end"]:
                    return sh["id"]
            return shots[0]["id"] if shots else "A"

        reps: list[dict[str, Any]] = []
        rep_no = 0
        last_bottom_t = -1e9
        for i in range(1, len(a_n) - 1):
            is_min = a_n[i] <= a_n[i - 1] and a_n[i] <= a_n[i + 1]
            if not is_min:
                continue
            # prominence: must be at least rep_prominence below the local max ahead
            look = a_n[i:i + int(cfg.pose_sample_fps * 2)]
            if look.size == 0:
                continue
            depth = float(look.max() - a_n[i])
            if depth < cfg.rep_prominence:
                continue
            bottom_t = times[i]
            if bottom_t - last_bottom_t < cfg.rep_min_gap_sec:
                continue
            # lockout = next local max after the bottom
            j = i + int(np.argmax(look))
            lockout_t = times[min(j, len(times) - 1)]
            rep_no += 1
            last_bottom_t = bottom_t
            reps.append({
                "exercise": exercise if exercise != "unknown" else cfg.default_exercise,
                "rep": rep_no,
                "bottom_t": round(float(bottom_t), cfg.round_dp),
                "lockout_t": round(float(lockout_t), cfg.round_dp),
                "shot": shot_at(bottom_t),
            })
        return reps
    except Exception as e:  # noqa: BLE001
        _warn(f"rep detection failed: {e} -> reps = []")
        return []


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def build_events(video_path: str, cfg: Config, *, do_pose: bool,
                 exercise: str) -> dict[str, Any]:
    _info(f"Extracting events from: {video_path}")

    duration = _ffprobe_duration(video_path)

    _info("[1/4] shot detection (PySceneDetect)...")
    shots = detect_shots(video_path, cfg)

    _info("[2/4] motion energy (OpenCV)...")
    motion_peaks, vid_duration = detect_motion_peaks(video_path, cfg)
    if duration == 0.0:
        duration = vid_duration

    _info("[3/4] audio onsets (librosa)...")
    audio_impacts: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "audio.wav")
        wav_ok = _extract_audio_wav(video_path, wav)
        audio_impacts = detect_audio_impacts(wav if wav_ok else None, cfg)

    reps: list[dict[str, Any]] = []
    if do_pose:
        _info("[4/4] pose / rep detection (MediaPipe)...")
        reps = detect_reps(video_path, cfg, exercise, shots)
    else:
        _info("[4/4] pose / rep detection SKIPPED (--no-pose)")

    # If no shots detected at all, synthesise a single full-length shot so
    # downstream tooling always has at least one angle to reference.
    if not shots and duration > 0:
        shots = [{"id": "A", "start": 0.0, "end": round(duration, cfg.round_dp),
                  "angle": "unknown"}]

    return {
        "duration": round(duration, cfg.round_dp),
        "shots": shots,
        "audio_impacts": audio_impacts,
        "motion_peaks": motion_peaks,
        "reps": reps,
        "_meta": {
            "source": os.path.basename(video_path),
            "config": asdict(cfg),
            "detectors": {
                "shots": bool(shots),
                "audio_impacts": bool(audio_impacts),
                "motion_peaks": bool(motion_peaks),
                "reps": bool(reps),
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Extract events.json from a video.")
    p.add_argument("video", help="path to input video")
    p.add_argument("-o", "--output", default=None,
                   help="output JSON path (default: ./events.json)")
    p.add_argument("--no-pose", action="store_true",
                   help="skip the slow MediaPipe rep-detection second pass")
    p.add_argument("--exercise", default="unknown",
                   help="exercise hint for rep labelling (squat/deadlift/...)")
    # threshold overrides
    p.add_argument("--scene-threshold", type=float)
    p.add_argument("--onset-delta", type=float)
    p.add_argument("--impact-min-intensity", type=float)
    p.add_argument("--motion-percentile", type=float)
    p.add_argument("--motion-sample-fps", type=float)
    p.add_argument("--rep-prominence", type=float)
    args = p.parse_args(argv)

    if not os.path.exists(args.video):
        print(f"error: input video not found: {args.video}", file=sys.stderr)
        return 2

    cfg = Config()
    for attr, val in [
        ("scene_threshold", args.scene_threshold),
        ("onset_delta", args.onset_delta),
        ("impact_min_intensity", args.impact_min_intensity),
        ("motion_percentile", args.motion_percentile),
        ("motion_sample_fps", args.motion_sample_fps),
        ("rep_prominence", args.rep_prominence),
    ]:
        if val is not None:
            setattr(cfg, attr, val)

    events = build_events(args.video, cfg, do_pose=not args.no_pose,
                          exercise=args.exercise)

    out = args.output or os.path.join(os.getcwd(), "events.json")
    with open(out, "w") as f:
        json.dump(events, f, indent=2)

    m = events["_meta"]["detectors"]
    print(f"\nWrote {out}")
    print(f"  duration      : {events['duration']}s")
    print(f"  shots         : {len(events['shots'])}    "
          f"{'OK' if m['shots'] else '(none)'}")
    print(f"  audio_impacts : {len(events['audio_impacts'])}    "
          f"{'OK' if m['audio_impacts'] else '(none)'}")
    print(f"  motion_peaks  : {len(events['motion_peaks'])}    "
          f"{'OK' if m['motion_peaks'] else '(none)'}")
    print(f"  reps          : {len(events['reps'])}    "
          f"{'OK' if m['reps'] else '(none/skipped)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
