"""H-Pattern (pole -> flag -> continuation) detector + intraday trade simulator.

The "H": impulse leg (left post) -> retracement that stalls into a sideways
band (the crossbar) -> continuation in the impulse direction (right post).
Mechanically a pole+flag. We detect the BEARISH form (impulse down) on the raw
series, and the BULLISH form by running the very same detector on a negated
series (high'=-low, low'=-high, close'=-close) -- a short in transformed space
is a long in the original, and R-multiple outcomes are sign-invariant, so one
code path serves both directions.

All logic is intraday on 5-min bars: pole + flag + trade live inside one session,
hard square-off at the last bar (15:25). Outcomes are reported as R-multiples
(R = entry-to-stop distance) so results are comparable across instruments.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# ---- Detection params (FIXED for the baseline; Phase-2 sweeps these) --------
POLE_MAX = 10        # max bars from pole-high to pole-low
POLE_ATR = 2.5       # pole height must be >= this * ATR
EFF = 0.50           # downward efficiency: pole / sum(|dclose|) >= EFF
FLAG_MIN = 3         # min flag bars
FLAG_MAX = 12        # max flag bars
RETR_MAX = 0.70      # retracement of the pole the flag may make
WIDTH_ATR = 1.5      # flag channel width <= this * ATR
LOW_TOL = 0.10       # flag may dip this * ATR below the pole low and still "hold"

# ---- Trade params -----------------------------------------------------------
STOP_BUF = 0.25      # stop = crossbar_top + STOP_BUF * ATR
BREAK_WAIT = 10      # bars to wait for a breakout fill (styles A, C)
RETEST_BARS = 10     # bars to wait for a retest fill (style C)
MAXHOLD = 30         # time stop in bars
TRAIL_ATR = 1.0      # ATR ratchet for the TRAIL exit
COST_BPS = 6.0       # round-trip cost (~3 bps/side) in basis points of entry

ENTRY_STYLES = ("breakout", "fade", "retest")
TARGETS = ("1R", "2R", "3R", "MM", "TRAIL")
VARIANTS = [f"{e}|{t}" for e in ENTRY_STYLES for t in TARGETS]


@dataclass
class Setup:
    """An armed H, in the coordinate space it was detected in (short space)."""
    arm_idx: int          # last flag bar (armed here)
    crossbar_top: float   # flag_high (stop reference / fade entry)
    breakout_level: float # flag_low (breakout trigger)
    pole_height: float    # pole_high - pole_low
    atr: float


def find_setups(h, l, c, atr, n):
    """Scan one day's bars (short/bearish space) for armed H setups.

    Non-overlapping: after a setup arms, scanning resumes past its flag.
    """
    setups = []
    lo = POLE_MAX
    while lo < n - FLAG_MIN:
        # --- pole: highest high in the lookback, must precede this low pivot ---
        w0 = lo - POLE_MAX
        hi_idx = w0 + int(np.argmax(h[w0:lo + 1]))
        pole_high = h[hi_idx]
        pole_low = l[lo]
        # lo must be the lowest low since the pole high, and high must precede it
        if hi_idx >= lo or pole_low > np.min(l[hi_idx:lo + 1]) + 1e-9:
            lo += 1
            continue
        pole = pole_high - pole_low
        a = atr[lo]
        if a <= 0 or pole < POLE_ATR * a:
            lo += 1
            continue
        path = np.sum(np.abs(np.diff(c[hi_idx:lo + 1])))
        if path <= 0 or pole / path < EFF:
            lo += 1
            continue
        # --- flag: first f in [lo+FLAG_MIN, lo+FLAG_MAX] that satisfies the band ---
        armed = None
        fmax = min(lo + FLAG_MAX, n - 1)
        for f in range(lo + FLAG_MIN, fmax + 1):
            fh = float(np.max(h[lo + 1:f + 1]))
            fl = float(np.min(l[lo + 1:f + 1]))
            if fh >= pole_high:                       # not a lower high -> reject
                break
            if (fh - pole_low) / pole > RETR_MAX:      # retraced too far
                break
            if fl < pole_low - LOW_TOL * a:            # already broke out / failed
                break
            if (fh - fl) > WIDTH_ATR * a:              # band too wide
                continue
            armed = Setup(arm_idx=f, crossbar_top=fh, breakout_level=fl,
                          pole_height=pole, atr=a)
            break
        if armed is not None:
            setups.append(armed)
            lo = armed.arm_idx + 1
        else:
            lo += 1
    return setups


def _manage(entry, stop, target_kind, pole_height, h, l, c, n, entry_idx):
    """Manage an open SHORT from entry_idx+1 to exit. Returns GROSS result in R
    (R = stop-entry). Stop is checked before target on same-bar ties (worst case).
    Exits on stop, target, MAXHOLD time stop, or end-of-day, whichever first."""
    R = stop - entry
    if R <= 0:
        return None
    if target_kind == "MM":
        target = entry - pole_height          # measured move projected from entry
    elif target_kind in ("1R", "2R", "3R"):
        target = entry - int(target_kind[0]) * R
    else:
        target = None                          # TRAIL handled below

    end = min(entry_idx + MAXHOLD, n - 1)
    trail = stop                               # ratchets DOWN for a short
    run_low = entry
    for t in range(entry_idx + 1, end + 1):
        if target_kind == "TRAIL":
            run_low = min(run_low, l[t])
            trail = min(trail, run_low + TRAIL_ATR * R)
            if h[t] >= trail:                  # trailing stop hit
                return (entry - trail) / R
        else:
            if h[t] >= stop:                   # protective stop (checked first)
                return (entry - stop) / R
            if target is not None and l[t] <= target:
                return (entry - target) / R
        if t == end:                           # time stop / square-off at close
            return (entry - c[t]) / R
    return (entry - c[end]) / R


def simulate(setup: Setup, style: str, target_kind: str, h, l, c, n):
    """Return (result_R_gross, entry_price, stop_price) or None for no-fill."""
    f = setup.arm_idx
    fh, fl, a = setup.crossbar_top, setup.breakout_level, setup.atr
    stop = fh + STOP_BUF * a

    if style == "fade":
        # "as it forms": once the H is recognised at the arm bar, enter at market
        # (close of the arm bar) betting on continuation. Causal entry price; stop
        # sits above the established crossbar. Entering at flag_high would be
        # look-ahead (flag_high is the whole-flag max -> stop never at risk).
        entry = c[f]
        if entry >= stop:                      # already above the stop -> no trade
            return None
        r = _manage(entry, stop, target_kind, setup.pole_height, h, l, c, n, f)
        return None if r is None else (r, entry, stop)

    # styles needing a breakout below fl
    btb = None
    for t in range(f + 1, min(f + BREAK_WAIT, n - 1) + 1):
        if c[t] > fh:                          # invalidated
            return None
        if l[t] < fl:
            btb = t
            break
    if btb is None:
        return None

    if style == "breakout":
        entry = fl
        r = _manage(entry, stop, target_kind, setup.pole_height, h, l, c, n, btb)
        return None if r is None else (r, entry, stop)

    # retest: after breakout, wait for a pullback that retags fl
    for t in range(btb + 1, min(btb + RETEST_BARS, n - 1) + 1):
        if c[t] > fh:                          # invalidated before retest
            return None
        if h[t] >= fl:                         # retest tag
            entry = fl
            r = _manage(entry, stop, target_kind, setup.pole_height, h, l, c, n, t)
            return None if r is None else (r, entry, stop)
    return None


def process_day(h, l, c, atr, direction):
    """Yield (variant, r_gross, cost_R, direction, entry_orig) for one day.

    Gross and cost are kept SEPARATE so the aggregator can show the pattern's raw
    edge independent of the cost assumption (tight intraday stops make a flat-bps
    cost dominate, so gross-vs-net is the honest way to read this).

    direction = -1 for bearish-H (short, raw arrays); +1 for bullish-H, where the
    caller passes negated arrays so the short-space logic applies.
    """
    n = len(c)
    if n < POLE_MAX + FLAG_MIN + 2:
        return
    setups = find_setups(h, l, c, atr, n)
    for s in setups:
        for style in ENTRY_STYLES:
            for tk in TARGETS:
                out = simulate(s, style, tk, h, l, c, n)
                if out is None:
                    continue
                r_gross, entry, stop = out
                R = stop - entry
                cost_R = (COST_BPS / 1e4 * abs(entry)) / R if R > 0 else 0.0
                entry_orig = entry if direction < 0 else -entry
                yield (f"{style}|{tk}", r_gross, cost_R, direction, entry_orig)
