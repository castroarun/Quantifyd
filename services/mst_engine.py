"""MST Index Strategy Engine
=========================

Master SuperTrend (MST) + Child Stochastic (CST) with Pyramid trigger on
NIFTY 30-min. Generates signals, manages position state machine, and routes
orders through the executor.

Spec: docs/Design/MST-INDEX-STRATEGY-DESIGN.md
Research: research/35_*, research/36_*

State machine:
  NO_POSITION → ARMED → DEBIT_OPEN_L1 → CONDOR_OPEN_L1
                                              ↓ pyramid (D AND B)
                                         DEBIT_OPEN_L2
                                              ↓ next CST (credit OK)
                                         CONDOR_OPEN_L2 (capped)
  Forks at any state: MST flip, T-1 EOD, kill switch.

This module owns the SIGNAL logic + state transitions. Order placement is
delegated to mst_executor.py (which knows about paper/live modes).
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, Any
from collections import deque

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---- SuperTrend + Stochastic (vectorized; same impl as research/35) ----

def _true_range(high, low, close):
    prev = np.roll(close, 1)
    prev[0] = close[0]
    return np.maximum.reduce([high - low, np.abs(high - prev), np.abs(low - prev)])


def _atr_wilder(high, low, close, period):
    tr = _true_range(high, low, close)
    n = len(tr)
    atr = np.full(n, np.nan)
    if n < period:
        return atr
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _supertrend(high, low, close, period, multiplier):
    """Returns (direction, st_line, atr, upper_final, lower_final)."""
    n = len(close)
    atr = _atr_wilder(high, low, close, period)
    hl2 = (high + low) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr
    upper_final = np.full(n, np.nan)
    lower_final = np.full(n, np.nan)
    direction = np.zeros(n, dtype=np.int8)
    st_line = np.full(n, np.nan)
    seed = period
    if seed >= n:
        return direction.astype(float), st_line, atr, upper_final, lower_final

    upper_final[seed] = upper_basic[seed]
    lower_final[seed] = lower_basic[seed]
    direction[seed] = 1 if close[seed] > upper_basic[seed] else -1
    st_line[seed] = lower_final[seed] if direction[seed] == 1 else upper_final[seed]

    for i in range(seed + 1, n):
        if upper_basic[i] < upper_final[i - 1] or close[i - 1] > upper_final[i - 1]:
            upper_final[i] = upper_basic[i]
        else:
            upper_final[i] = upper_final[i - 1]
        if lower_basic[i] > lower_final[i - 1] or close[i - 1] < lower_final[i - 1]:
            lower_final[i] = lower_basic[i]
        else:
            lower_final[i] = lower_final[i - 1]
        prev_dir = direction[i - 1]
        if prev_dir == 1:
            direction[i] = -1 if close[i] < lower_final[i] else 1
        else:
            direction[i] = 1 if close[i] > upper_final[i] else -1
        st_line[i] = lower_final[i] if direction[i] == 1 else upper_final[i]

    direction_f = direction.astype(float)
    direction_f[:seed] = np.nan
    return direction_f, st_line, atr, upper_final, lower_final


def _stochastic(high, low, close, k_period=14, d_period=3, smooth=3):
    n = len(close)
    k_raw = np.full(n, np.nan)
    for i in range(k_period - 1, n):
        hh = high[i - k_period + 1:i + 1].max()
        ll = low[i - k_period + 1:i + 1].min()
        if hh - ll > 0:
            k_raw[i] = 100 * (close[i] - ll) / (hh - ll)
    k = pd.Series(k_raw).rolling(smooth, min_periods=smooth).mean().values
    d = pd.Series(k).rolling(d_period, min_periods=d_period).mean().values
    return k, d


# ---- Engine state ----

@dataclass
class MSTState:
    """In-memory engine state. Persisted to mst_events and mst_positions."""
    state: str = "NO_POSITION"      # NO_POSITION | ARMED | DEBIT_OPEN_L1 | CONDOR_OPEN_L1 | DEBIT_OPEN_L2 | CONDOR_OPEN_L2
    mst_direction: int = 0           # +1 long, -1 short, 0 idle
    armed_high: Optional[float] = None  # flip-bar high (for break-of-extreme on long)
    armed_low: Optional[float] = None
    activated_at_bar: Optional[str] = None
    activated_at_atm: Optional[int] = None  # entry-time ATM (rounded to 50)
    pyramid_atm: Optional[int] = None       # spot at pyramid trigger time (rounded to 50)
    last_cst_bar: Optional[str] = None       # bar_dt of the most recent CST trigger
    last_cst_high: Optional[float] = None    # high of CST bar (for pyramid trigger D)
    last_cst_low: Optional[float] = None     # low of CST bar (for pyramid trigger D - short)
    pyramid_level: int = 1                   # 1 or 2
    # Pyramid trigger B tracking: did %K leave the OB/OS zone since last CST?
    stoch_left_zone_since_cst: bool = False
    # Current week's expiry and T-1 (str ISO date)
    current_expiry_dt: Optional[str] = None
    current_t_minus_1: Optional[str] = None


# ---- Engine ----

class MSTEngine:
    """
    Drive 30-min bar evaluation, state transitions, and event emission.

    External callers:
      - on_30min_bar(bar)        : feed a new bar (called by NasTicker callback)
      - on_t_minus_1_check(today): scheduled cron at 15:25 — closes any
                                   positions whose t_minus_1_date == today
      - kill_switch()            : emergency close + halt
    """
    # Constants from research/35 + research/36
    ATR_PERIOD = 21
    MULTIPLIER = 5.0
    STOCH_K = 14
    STOCH_D = 3
    STOCH_SMOOTH = 3
    STOCH_OB = 80
    STOCH_OS = 20
    PYRAMID_OB_EXIT_THRESHOLD = 70  # %K must drop below 70 (long bias) before re-entering OB
    PYRAMID_OS_EXIT_THRESHOLD = 30
    PYRAMID_D_LOOKBACK = 6          # cumulative count window (bars after CST)
    PYRAMID_D_THRESHOLD = 3         # net (above-below) within lookback >= threshold
    PYRAMID_SAFETY_WING_PCT = 0.5    # safety trigger fires at K3 + 0.5 * (K4-K3)
    SPREAD_WIDTH = 200               # standard structure
    RESET_WIDTH = 100                # reset (Reading D) structure
    OTM_OFFSET = 50                  # debit spread anchor: 1st OTM (50 pts on NIFTY 50-pt strikes)
    STRIKE_INTERVAL = 50             # NIFTY weekly options strike spacing
    LOTS = 1
    MIN_DTE = 6
    PYRAMID_MAX_LEVEL = 2
    MIN_CREDIT_PER_LOT = 1000        # rupees

    BUFFER_SIZE = 250                # rolling bar buffer (covers ATR=50 + Stoch warmup)

    def _compute_anchor(self, spot: float, direction: int) -> int:
        """Compute the debit spread anchor strike (long leg of the bull-call /
        bear-put). With OTM_OFFSET=50, this is 1 strike OTM relative to spot.
        OTM_OFFSET=0 would restore ATM-anchored behavior.

        Long MST: anchor = ATM + offset (1st OTM CE side)
        Short MST: anchor = ATM - offset (1st OTM PE side)
        """
        base_atm = round(spot / self.STRIKE_INTERVAL) * self.STRIKE_INTERVAL
        if direction == 1:
            return int(base_atm + self.OTM_OFFSET)
        if direction == -1:
            return int(base_atm - self.OTM_OFFSET)
        return int(base_atm)

    def __init__(self, executor=None, calendar=None, paper_mode: bool = True,
                 enabled: bool = True):
        from services import mst_db
        from services.trading_calendar import get_default_calendar
        mst_db.init_db()
        self.db = mst_db
        self.calendar = calendar or get_default_calendar()
        self.executor = executor      # MSTExecutor; None = signal-only mode
        self.paper_mode = paper_mode
        self.enabled = enabled
        self.state = MSTState()
        self.bars: deque[dict] = deque(maxlen=self.BUFFER_SIZE)
        self._restored_from_db = False

    # ---- Bar buffer + indicator computation ----

    def add_historical_bar(self, bar: dict) -> None:
        """Append a historical bar to the buffer (used during seeding)."""
        self.bars.append(bar)

    def restore_state_from_open_positions(self) -> None:
        """If there are OPEN positions in mst_positions DB at engine startup,
        reconstruct the state machine so the engine continues managing them.

        Idempotent — safe to call on every bootstrap. Examines:
        - distinct directions / pyramid levels in open legs → infers state
        - week_label, expiry_date, t_minus_1_date → restores cycle context
        - leg_roles → infers DEBIT_OPEN_* vs CONDOR_OPEN_* state
        """
        try:
            open_legs = self.db.get_open_positions()
        except Exception:
            return
        if not open_legs:
            return

        # Group by week_label + direction to find the active cycle
        directions = sorted({int(p['direction']) for p in open_legs})
        if len(directions) > 1:
            logger.warning(f"[MST] Restore: multiple directions in open legs {directions} — leaving state untouched")
            return
        direction = directions[0]

        # Identify leg roles to figure out which state we're in
        roles = {p['leg_role'] for p in open_legs}
        levels = sorted({int(p.get('pyramid_level', 1)) for p in open_legs})
        max_level = max(levels)

        debit_l1_roles = {'bull_long', 'bull_short'} if direction == 1 else {'put_long', 'put_short'}
        credit_l1_roles = {'bear_short', 'bear_long'} if direction == 1 else {'putw_short', 'putw_long'}

        has_l1_debit = debit_l1_roles.issubset({p['leg_role'] for p in open_legs if int(p.get('pyramid_level', 1)) == 1})
        has_l1_credit = credit_l1_roles.issubset({p['leg_role'] for p in open_legs if int(p.get('pyramid_level', 1)) == 1})
        has_l2_debit = max_level >= 2 and debit_l1_roles.issubset({p['leg_role'] for p in open_legs if int(p.get('pyramid_level', 1)) == 2})
        has_l2_credit = max_level >= 2 and credit_l1_roles.issubset({p['leg_role'] for p in open_legs if int(p.get('pyramid_level', 1)) == 2})

        if has_l2_credit:
            new_state = "CONDOR_OPEN_L2"
            self.state.pyramid_level = 2
        elif has_l2_debit:
            new_state = "DEBIT_OPEN_L2"
            self.state.pyramid_level = 2
        elif has_l1_credit:
            new_state = "CONDOR_OPEN_L1"
            self.state.pyramid_level = 1
        elif has_l1_debit:
            new_state = "DEBIT_OPEN_L1"
            self.state.pyramid_level = 1
        else:
            logger.warning(f"[MST] Restore: open legs but cannot map roles {roles} to a known state — leaving NO_POSITION")
            return

        # Find the L1 anchor (long leg of the L1 debit spread)
        l1_long_role = 'bull_long' if direction == 1 else 'put_long'
        l1_long_leg = next((p for p in open_legs if p['leg_role'] == l1_long_role and int(p.get('pyramid_level', 1)) == 1), None)
        l2_long_leg = next((p for p in open_legs if p['leg_role'] == l1_long_role and int(p.get('pyramid_level', 1)) == 2), None)

        self.state.state = new_state
        self.state.mst_direction = direction
        if l1_long_leg:
            self.state.activated_at_atm = int(l1_long_leg['strike'])
            self.state.activated_at_bar = l1_long_leg.get('entry_time')
            self.state.current_expiry_dt = l1_long_leg['expiry_date']
            self.state.current_t_minus_1 = l1_long_leg['t_minus_1_date']
        if l2_long_leg:
            self.state.pyramid_atm = int(l2_long_leg['strike'])

        logger.warning(
            f"[MST] State RESTORED from {len(open_legs)} open legs → state={new_state} "
            f"direction={direction} L1_anchor={self.state.activated_at_atm} "
            f"expiry={self.state.current_expiry_dt}"
        )

    def _arrays(self):
        """Return numpy arrays for current buffer."""
        if not self.bars:
            return None
        df = pd.DataFrame(list(self.bars))
        return {
            "high": df["high"].to_numpy(),
            "low": df["low"].to_numpy(),
            "close": df["close"].to_numpy(),
            "bar_dt": [b["bar_dt"] for b in self.bars],
        }

    def _compute_indicators(self):
        """Compute ST, ATR, Stoch on the current buffer. Returns dict of latest values + arrays."""
        arr = self._arrays()
        if not arr or len(arr["close"]) < max(self.ATR_PERIOD, self.STOCH_K) + 5:
            return None
        direction, st_line, atr, upper, lower = _supertrend(
            arr["high"], arr["low"], arr["close"], self.ATR_PERIOD, self.MULTIPLIER
        )
        k, d = _stochastic(arr["high"], arr["low"], arr["close"],
                           self.STOCH_K, self.STOCH_D, self.STOCH_SMOOTH)
        return {
            "direction": direction, "st_line": st_line, "atr": atr,
            "upper": upper, "lower": lower, "k": k, "d": d,
            "high": arr["high"], "low": arr["low"], "close": arr["close"],
            "bar_dt": arr["bar_dt"],
        }

    # ---- Main bar handler ----

    def on_30min_bar(self, bar: dict) -> list[dict]:
        """Process a completed 30-min bar.

        Returns a list of events that fired on this bar (for caller to dispatch
        as alerts).
        """
        if not self.enabled:
            return []

        self.bars.append(bar)
        ind = self._compute_indicators()
        if ind is None:
            return []

        i = len(ind["close"]) - 1
        events: list[dict] = []

        # Persist bar with indicators
        bar_record = dict(bar)
        bar_record["atr21"] = float(ind["atr"][i]) if not np.isnan(ind["atr"][i]) else None
        bar_record["st_value"] = float(ind["st_line"][i]) if not np.isnan(ind["st_line"][i]) else None
        bar_record["st_upper"] = float(ind["upper"][i]) if not np.isnan(ind["upper"][i]) else None
        bar_record["st_lower"] = float(ind["lower"][i]) if not np.isnan(ind["lower"][i]) else None
        bar_record["direction"] = int(ind["direction"][i]) if not np.isnan(ind["direction"][i]) else None
        bar_record["stoch_k"] = float(ind["k"][i]) if not np.isnan(ind["k"][i]) else None
        bar_record["stoch_d"] = float(ind["d"][i]) if not np.isnan(ind["d"][i]) else None
        self.db.save_bar(bar_record)

        # State machine evaluation
        events.extend(self._evaluate_mst_flip(ind, i, bar))
        events.extend(self._evaluate_break_of_extreme(ind, i, bar))
        events.extend(self._evaluate_cst_and_pyramid(ind, i, bar))

        return events

    # ---- MST flip detection ----

    def _evaluate_mst_flip(self, ind, i, bar):
        events = []
        cur_dir = ind["direction"][i]
        prev_dir = ind["direction"][i - 1] if i > 0 else None
        if cur_dir is None or np.isnan(cur_dir) or prev_dir is None or np.isnan(prev_dir):
            return events
        cur_dir = int(cur_dir)
        prev_dir = int(prev_dir)
        if cur_dir == prev_dir:
            return events

        # MST flipped on this bar's close
        flip_high = float(ind["high"][i])
        flip_low = float(ind["low"][i])
        flip_price = float(ind["close"][i])

        # If we have an open position, close it (MST flip = close all)
        if self.state.state in ("DEBIT_OPEN_L1", "CONDOR_OPEN_L1",
                                "DEBIT_OPEN_L2", "CONDOR_OPEN_L2"):
            self._close_all_legs(reason="mst_flip", bar=bar)
            self.db.log_event("mst_flip_close", direction=prev_dir,
                              bar_dt=bar["bar_dt"], price=flip_price,
                              notes=f"MST flipped {prev_dir}→{cur_dir}; closed all legs")
            events.append({"type": "mst_flip_close", "from_direction": prev_dir,
                           "to_direction": cur_dir, "price": flip_price})

        # If we were in ARMED state with the OPPOSITE direction, the prior arm is discarded
        if self.state.state == "ARMED" and self.state.mst_direction != cur_dir:
            self.db.log_event("flip_discarded", direction=self.state.mst_direction,
                              bar_dt=bar["bar_dt"], price=flip_price,
                              notes=f"Prior arm discarded — MST flipped to {cur_dir}")
            events.append({"type": "flip_discarded",
                           "prior_direction": self.state.mst_direction,
                           "new_direction": cur_dir})

        # Re-arm in the new direction
        self.state.state = "ARMED"
        self.state.mst_direction = cur_dir
        self.state.armed_high = flip_high
        self.state.armed_low = flip_low
        self.state.activated_at_bar = None
        self.state.activated_at_atm = None
        self.state.pyramid_atm = None
        self.state.last_cst_bar = None
        self.state.last_cst_high = None
        self.state.last_cst_low = None
        self.state.pyramid_level = 1
        self.state.stoch_left_zone_since_cst = False

        self.db.log_event("flip_armed", direction=cur_dir, bar_dt=bar["bar_dt"],
                          price=flip_price, flip_high=flip_high, flip_low=flip_low,
                          notes=f"MST flipped to {'LONG' if cur_dir == 1 else 'SHORT'}; armed for break-of-extreme")
        events.append({"type": "flip_armed", "direction": cur_dir, "price": flip_price,
                       "flip_high": flip_high, "flip_low": flip_low})
        return events

    # ---- Break-of-extreme entry confirmation ----

    def _evaluate_break_of_extreme(self, ind, i, bar):
        events = []
        if self.state.state != "ARMED":
            return events
        if self.state.armed_high is None or self.state.armed_low is None:
            return events
        bar_high = float(ind["high"][i])
        bar_low = float(ind["low"][i])

        if self.state.mst_direction == 1 and bar_high > self.state.armed_high:
            entry_price = self.state.armed_high  # break level = where stop would fill
            self._activate_position(entry_price, bar, ind, i)
            events.append({"type": "flip_activated", "direction": 1,
                           "entry_price": entry_price, "spot": float(ind["close"][i])})
        elif self.state.mst_direction == -1 and bar_low < self.state.armed_low:
            entry_price = self.state.armed_low
            self._activate_position(entry_price, bar, ind, i)
            events.append({"type": "flip_activated", "direction": -1,
                           "entry_price": entry_price, "spot": float(ind["close"][i])})
        return events

    def _activate_position(self, entry_price, bar, ind, i):
        """Open the level-1 debit spread."""
        spot = float(ind["close"][i])
        # Anchor the debit spread's long leg at 1st OTM (ATM ± OTM_OFFSET).
        # The whole condor shifts with this — see _compute_anchor.
        entry_atm = self._compute_anchor(spot, self.state.mst_direction)

        # Compute weekly expiry & T-1
        bar_date = self._bar_date(bar)
        expiry_dt = self.calendar.next_weekly_expiry(bar_date, min_dte=self.MIN_DTE)
        t_minus_1 = self.calendar.t_minus_1(expiry_dt)

        self.state.state = "DEBIT_OPEN_L1"
        self.state.activated_at_bar = bar["bar_dt"]
        self.state.activated_at_atm = entry_atm
        self.state.current_expiry_dt = expiry_dt.isoformat()
        self.state.current_t_minus_1 = t_minus_1.isoformat()

        self.db.log_event("flip_activated",
                          direction=self.state.mst_direction,
                          bar_dt=bar["bar_dt"], price=entry_price,
                          flip_high=self.state.armed_high,
                          flip_low=self.state.armed_low,
                          pyramid_level=1,
                          notes=f"Break confirmed at {entry_price:.2f}, spot={spot:.2f}, "
                                f"ATM={entry_atm}, expiry={expiry_dt}, T-1={t_minus_1}")

        # Place debit spread via executor
        if self.executor:
            self.executor.place_debit_spread(
                direction=self.state.mst_direction,
                atm=entry_atm,
                width=self.SPREAD_WIDTH,
                expiry=expiry_dt,
                t_minus_1=t_minus_1,
                pyramid_level=1,
                bar_dt=bar["bar_dt"],
            )

    # ---- CST + Pyramid trigger ----

    def _evaluate_cst_and_pyramid(self, ind, i, bar):
        events = []
        if self.state.state not in ("DEBIT_OPEN_L1", "CONDOR_OPEN_L1",
                                    "DEBIT_OPEN_L2", "CONDOR_OPEN_L2"):
            return events

        k_now = ind["k"][i]
        k_prev = ind["k"][i - 1] if i > 0 else None
        d_now = ind["d"][i]
        d_prev = ind["d"][i - 1] if i > 0 else None
        if any(x is None or np.isnan(x) for x in (k_now, k_prev, d_now, d_prev)):
            return events

        # Pyramid trigger B tracking: has %K left the extreme zone since last CST?
        if self.state.last_cst_bar is not None:
            if self.state.mst_direction == 1 and k_now < self.PYRAMID_OB_EXIT_THRESHOLD:
                self.state.stoch_left_zone_since_cst = True
            elif self.state.mst_direction == -1 and k_now > self.PYRAMID_OS_EXIT_THRESHOLD:
                self.state.stoch_left_zone_since_cst = True

        # ---- Pyramid trigger: (D_cumulative AND B) OR safety_wing_breach ----
        # Only fires while in CONDOR_OPEN_L1 (level 1 condor already built).
        # Safety has 0% FP rate (validated on 6.3 yrs); D_cumulative+B has 13.2% FP.
        # The two triggers are OR'd: whichever fires first pyramids.
        if self.state.state == "CONDOR_OPEN_L1":
            safety_fired = self._check_safety_trigger(ind, i)
            db_fired = False
            if self.state.last_cst_bar is not None:
                d_fired = self._check_trigger_d(ind, i)
                b_fired = self._check_trigger_b(k_now, k_prev)
                db_fired = d_fired and b_fired
            if safety_fired or db_fired:
                trigger_kind = "safety_wing_breach" if safety_fired and not db_fired else (
                    "d_cumulative_and_b" if db_fired and not safety_fired else "both")
                events.extend(self._fire_pyramid(ind, i, bar, trigger_kind=trigger_kind))
                return events

        # ---- CST trigger ----
        is_cst = self._check_cst(k_now, k_prev, d_now, d_prev)
        if not is_cst:
            return events

        # CST always logs an event
        cst_kind = "first" if self.state.last_cst_bar is None else "subsequent"
        self.state.last_cst_bar = bar["bar_dt"]
        self.state.last_cst_high = float(ind["high"][i])
        self.state.last_cst_low = float(ind["low"][i])
        self.state.stoch_left_zone_since_cst = False

        if self.state.state == "DEBIT_OPEN_L1":
            # First CST in week → check credit, build condor or roll
            events.extend(self._handle_first_cst_l1(ind, i, bar))
        elif self.state.state == "DEBIT_OPEN_L2":
            # First CST after pyramid → second hedge
            events.extend(self._handle_first_cst_l2(ind, i, bar))
        else:
            # CONDOR_OPEN_L1 / CONDOR_OPEN_L2 → log only
            self.db.log_event("cst_subsequent",
                              direction=self.state.mst_direction,
                              bar_dt=bar["bar_dt"],
                              price=float(ind["close"][i]),
                              pyramid_level=self.state.pyramid_level,
                              notes=f"Subsequent CST in {self.state.state} — informational, no action")
            events.append({"type": "cst_subsequent",
                           "spot": float(ind["close"][i]),
                           "state": self.state.state})
        return events

    def _check_cst(self, k_now, k_prev, d_now, d_prev) -> bool:
        if self.state.mst_direction == 1:
            return k_prev >= d_prev and k_now < d_now and k_prev >= self.STOCH_OB
        if self.state.mst_direction == -1:
            return k_prev <= d_prev and k_now > d_now and k_prev <= self.STOCH_OS
        return False

    def _check_trigger_d(self, ind, i) -> bool:
        """D_cumulative = within last LOOKBACK bars after CST, net (closes
        beyond CST level − closes against) >= THRESHOLD AND current bar is
        beyond. Replaced D_strict (consecutive) per research/36 D-variants test:
        same coverage, FP rate 18.7% → 13.2% (-30%), catches staircase patterns
        that strict missed.
        """
        if self.state.last_cst_high is None or self.state.last_cst_low is None:
            return False
        if self.state.last_cst_bar is None:
            return False
        # Find the CST bar's position in the buffer
        bar_dts = ind["bar_dt"]
        cst_pos = None
        for idx in range(len(bar_dts) - 1, -1, -1):
            if bar_dts[idx] == self.state.last_cst_bar:
                cst_pos = idx
                break
        if cst_pos is None or i <= cst_pos:
            return False
        # Lookback window
        start = max(cst_pos + 1, i - self.PYRAMID_D_LOOKBACK + 1)
        window = ind["close"][start:i + 1]
        c_now = ind["close"][i]
        if self.state.mst_direction == 1:
            level = self.state.last_cst_high
            cur_beyond = c_now > level
            if not cur_beyond:
                return False
            above = sum(1 for c in window if c > level)
            below = sum(1 for c in window if c < level)
            return (above - below) >= self.PYRAMID_D_THRESHOLD
        if self.state.mst_direction == -1:
            level = self.state.last_cst_low
            cur_beyond = c_now < level
            if not cur_beyond:
                return False
            below = sum(1 for c in window if c < level)
            above = sum(1 for c in window if c > level)
            return (below - above) >= self.PYRAMID_D_THRESHOLD
        return False

    def _check_safety_trigger(self, ind, i) -> bool:
        """Safety trigger — fires when spot has breached past the credit spread's
        short strike and entered halfway into the upper/lower wing.

        Long MST condor strikes (entry-anchored): K1 = entry_atm, K2 = entry_atm + W,
        K3 = entry_atm + 2W, K4 = entry_atm + 3W (W = SPREAD_WIDTH = 200).
        Upper wing = K3 to K4. Safety threshold = K3 + WING_PCT * (K4 - K3) = entry_atm + 2.5W.
        Mirror for short MST.

        Empirically validated (research/36 safety_trigger_value test):
          - 0% FP rate (never fires when CST was correct)
          - +2.1% incremental coverage on TREND_CONTINUED
          - Fires before D_cumulative+B in 8.6% of cases (earlier pyramid = more upside)
        """
        if self.state.activated_at_atm is None:
            return False
        if self.state.state != "CONDOR_OPEN_L1":
            return False
        atm = self.state.activated_at_atm
        offset = (2 + self.PYRAMID_SAFETY_WING_PCT) * self.SPREAD_WIDTH
        # Use bar high for long (most aggressive), bar low for short
        bar_high = ind["high"][i]
        bar_low = ind["low"][i]
        if self.state.mst_direction == 1:
            return bar_high >= atm + offset
        if self.state.mst_direction == -1:
            return bar_low <= atm - offset
        return False

    def _check_trigger_b(self, k_now, k_prev) -> bool:
        """B = Stoch K returned to OB/OS after leaving the zone since last CST."""
        if not self.state.stoch_left_zone_since_cst:
            return False
        if self.state.mst_direction == 1:
            return k_now >= self.STOCH_OB
        if self.state.mst_direction == -1:
            return k_now <= self.STOCH_OS
        return False

    def _fire_pyramid(self, ind, i, bar, trigger_kind: str = "d_cumulative_and_b"):
        events = []
        if self.state.pyramid_level >= self.PYRAMID_MAX_LEVEL:
            self.db.log_event("pyramid_capped", direction=self.state.mst_direction,
                              bar_dt=bar["bar_dt"], price=float(ind["close"][i]),
                              pyramid_level=self.state.pyramid_level,
                              notes=f"Pyramid trigger fired but level already at max (L{self.state.pyramid_level}); kind={trigger_kind}")
            return events

        spot = float(ind["close"][i])
        new_atm = self._compute_anchor(spot, self.state.mst_direction)
        self.state.pyramid_atm = new_atm
        self.state.pyramid_level = 2
        self.state.state = "DEBIT_OPEN_L2"
        # Reset CST tracking — next CST will trigger the L2 hedge
        self.state.last_cst_bar = None
        self.state.stoch_left_zone_since_cst = False

        self.db.log_event("pyramid_fired", direction=self.state.mst_direction,
                          bar_dt=bar["bar_dt"], price=spot,
                          pyramid_level=2,
                          notes=f"Pyramid trigger={trigger_kind}; new debit spread at ATM={new_atm}")
        events.append({"type": "pyramid_fired", "direction": self.state.mst_direction,
                       "spot": spot, "new_atm": new_atm,
                       "trigger_kind": trigger_kind})

        # Place the new debit spread (anchored to pyramid-time ATM)
        if self.executor and self.state.current_expiry_dt and self.state.current_t_minus_1:
            self.executor.place_debit_spread(
                direction=self.state.mst_direction,
                atm=new_atm,
                width=self.SPREAD_WIDTH,
                expiry=date.fromisoformat(self.state.current_expiry_dt),
                t_minus_1=date.fromisoformat(self.state.current_t_minus_1),
                pyramid_level=2,
                bar_dt=bar["bar_dt"],
            )
        return events

    def _handle_first_cst_l1(self, ind, i, bar):
        """First CST in DEBIT_OPEN_L1 — check credit, build condor or roll."""
        events = []
        spot = float(ind["close"][i])
        # Anchor credit spread at entry-time ATM + 400 (long) / -400 (short)
        atm = self.state.activated_at_atm
        if self.state.mst_direction == 1:
            short_strike = atm + 2 * self.SPREAD_WIDTH    # ATM + 400
            long_strike = atm + 3 * self.SPREAD_WIDTH     # ATM + 600
        else:
            short_strike = atm - 2 * self.SPREAD_WIDTH    # ATM - 400
            long_strike = atm - 3 * self.SPREAD_WIDTH     # ATM - 600

        # Credit check (delegate to executor; for signal-only mode, assume OK)
        credit_ok = True
        credit_per_lot = None
        if self.executor:
            credit_per_lot = self.executor.estimate_credit(
                direction=self.state.mst_direction,
                short_strike=short_strike,
                long_strike=long_strike,
                expiry=date.fromisoformat(self.state.current_expiry_dt),
            )
            credit_ok = credit_per_lot is not None and credit_per_lot >= self.MIN_CREDIT_PER_LOT

        if credit_ok:
            self.state.state = "CONDOR_OPEN_L1"
            self.db.log_event("condor_built", direction=self.state.mst_direction,
                              bar_dt=bar["bar_dt"], price=spot, pyramid_level=1,
                              notes=f"L1 condor built at strikes [{atm}, {atm + (1 if self.state.mst_direction == 1 else -1) * self.SPREAD_WIDTH}, {short_strike}, {long_strike}], "
                                    f"credit≈₹{credit_per_lot or 0:.0f}/lot")
            events.append({"type": "condor_built", "level": 1, "spot": spot,
                           "credit_per_lot": credit_per_lot})
            if self.executor:
                self.executor.place_credit_spread(
                    direction=self.state.mst_direction,
                    short_strike=short_strike,
                    long_strike=long_strike,
                    expiry=date.fromisoformat(self.state.current_expiry_dt),
                    t_minus_1=date.fromisoformat(self.state.current_t_minus_1),
                    pyramid_level=1,
                    bar_dt=bar["bar_dt"],
                )
        else:
            # Roll-and-reset: close L1 debit, open fresh next-week reset condor
            self._roll_to_next_week(spot, bar, reason="credit_too_low",
                                    credit_value=credit_per_lot or 0)
            events.append({"type": "rolled", "reason": "credit_too_low",
                           "credit": credit_per_lot})
        return events

    def _handle_first_cst_l2(self, ind, i, bar):
        """First CST after pyramid — second hedge at pyramid-anchored strikes."""
        events = []
        spot = float(ind["close"][i])
        atm = self.state.pyramid_atm
        if self.state.mst_direction == 1:
            short_strike = atm + 2 * self.SPREAD_WIDTH
            long_strike = atm + 3 * self.SPREAD_WIDTH
        else:
            short_strike = atm - 2 * self.SPREAD_WIDTH
            long_strike = atm - 3 * self.SPREAD_WIDTH

        credit_ok = True
        credit_per_lot = None
        if self.executor:
            credit_per_lot = self.executor.estimate_credit(
                direction=self.state.mst_direction,
                short_strike=short_strike,
                long_strike=long_strike,
                expiry=date.fromisoformat(self.state.current_expiry_dt),
            )
            credit_ok = credit_per_lot is not None and credit_per_lot >= self.MIN_CREDIT_PER_LOT

        if credit_ok:
            self.state.state = "CONDOR_OPEN_L2"
            self.db.log_event("condor_built", direction=self.state.mst_direction,
                              bar_dt=bar["bar_dt"], price=spot, pyramid_level=2,
                              notes=f"L2 condor (max level) built at strikes [{atm}, {atm + (1 if self.state.mst_direction == 1 else -1) * self.SPREAD_WIDTH}, {short_strike}, {long_strike}], "
                                    f"credit≈₹{credit_per_lot or 0:.0f}/lot")
            events.append({"type": "condor_built", "level": 2, "spot": spot,
                           "credit_per_lot": credit_per_lot})
            if self.executor:
                self.executor.place_credit_spread(
                    direction=self.state.mst_direction,
                    short_strike=short_strike, long_strike=long_strike,
                    expiry=date.fromisoformat(self.state.current_expiry_dt),
                    t_minus_1=date.fromisoformat(self.state.current_t_minus_1),
                    pyramid_level=2, bar_dt=bar["bar_dt"],
                )
        else:
            # L2 credit also too low — roll everything to next week
            self._roll_to_next_week(spot, bar, reason="l2_credit_too_low",
                                    credit_value=credit_per_lot or 0)
            events.append({"type": "rolled", "reason": "l2_credit_too_low",
                           "credit": credit_per_lot})
        return events

    # ---- Rollover (T-1 close + reopen) ----

    def on_t_minus_1_check(self, today: date) -> list[dict]:
        """Called by scheduler at 15:25 IST every weekday.

        Closes any open positions whose t_minus_1_date == today, then if MST
        is still active, immediately opens a new debit spread at current spot
        ATM for next weekly expiry (≥6 DTE rule).
        """
        if not self.enabled:
            return []
        events = []
        positions_to_close = self.db.get_positions_for_t_minus_1(today.isoformat())
        if not positions_to_close:
            return events

        # Close all matching positions
        if self.executor:
            for pos in positions_to_close:
                self.executor.close_position(pos, reason="t_minus_1_eod")

        self.db.log_event("t_minus_1_close", direction=self.state.mst_direction,
                          bar_dt=datetime.now().isoformat(),
                          price=None, pyramid_level=self.state.pyramid_level,
                          notes=f"T-1 EOD close: {len(positions_to_close)} legs squared")
        events.append({"type": "t_minus_1_close", "leg_count": len(positions_to_close)})

        # Rollover if MST still active
        if self.state.state in ("DEBIT_OPEN_L1", "CONDOR_OPEN_L1",
                                "DEBIT_OPEN_L2", "CONDOR_OPEN_L2"):
            # Get current spot from last bar
            if self.bars:
                spot = self.bars[-1]["close"]
                bar_date = self._bar_date(self.bars[-1])
                new_expiry = self.calendar.next_weekly_expiry(bar_date, min_dte=self.MIN_DTE)
                new_t_minus_1 = self.calendar.t_minus_1(new_expiry)
                new_atm = self._compute_anchor(spot, self.state.mst_direction)

                # Reset state to L1
                self.state.state = "DEBIT_OPEN_L1"
                self.state.activated_at_bar = self.bars[-1]["bar_dt"]
                self.state.activated_at_atm = new_atm
                self.state.pyramid_atm = None
                self.state.last_cst_bar = None
                self.state.last_cst_high = None
                self.state.last_cst_low = None
                self.state.pyramid_level = 1
                self.state.stoch_left_zone_since_cst = False
                self.state.current_expiry_dt = new_expiry.isoformat()
                self.state.current_t_minus_1 = new_t_minus_1.isoformat()

                self.db.log_event("rolled",
                                  direction=self.state.mst_direction,
                                  bar_dt=self.bars[-1]["bar_dt"], price=spot,
                                  pyramid_level=1,
                                  notes=f"Rollover at T-1: new ATM={new_atm}, expiry={new_expiry}, "
                                        f"T-1={new_t_minus_1}, level reset to L1")
                events.append({"type": "rolled", "reason": "t_minus_1_rollover",
                               "new_atm": new_atm, "new_expiry": new_expiry.isoformat()})

                if self.executor:
                    self.executor.place_debit_spread(
                        direction=self.state.mst_direction,
                        atm=new_atm, width=self.SPREAD_WIDTH,
                        expiry=new_expiry, t_minus_1=new_t_minus_1,
                        pyramid_level=1, bar_dt=self.bars[-1]["bar_dt"],
                    )
        else:
            # No active MST → just go to NO_POSITION
            self.state.state = "NO_POSITION"

        return events

    def _roll_to_next_week(self, spot, bar, reason, credit_value=0):
        """Close current week's positions and open fresh on next week with reset
        structure (Reading D — 100/100/100 spot-centered)."""
        # Close existing positions
        if self.executor:
            for pos in self.db.get_open_positions():
                self.executor.close_position(pos, reason=reason)

        bar_date = self._bar_date(bar)
        new_expiry = self.calendar.next_weekly_expiry(bar_date, min_dte=self.MIN_DTE)
        new_t_minus_1 = self.calendar.t_minus_1(new_expiry)
        new_atm = self._compute_anchor(spot, self.state.mst_direction)

        self.state.state = "DEBIT_OPEN_L1"
        self.state.activated_at_bar = bar["bar_dt"]
        self.state.activated_at_atm = new_atm
        self.state.pyramid_atm = None
        self.state.last_cst_bar = None
        self.state.pyramid_level = 1
        self.state.stoch_left_zone_since_cst = False
        self.state.current_expiry_dt = new_expiry.isoformat()
        self.state.current_t_minus_1 = new_t_minus_1.isoformat()

        self.db.log_event("rolled", direction=self.state.mst_direction,
                          bar_dt=bar["bar_dt"], price=spot, pyramid_level=1,
                          notes=f"{reason}: credit={credit_value:.0f}/lot < threshold; "
                                f"new ATM={new_atm} (Reading D reset structure), "
                                f"expiry={new_expiry}")

        # Open Reading D reset condor on new week
        if self.executor:
            self.executor.place_reset_condor(
                direction=self.state.mst_direction,
                atm=new_atm, width=self.RESET_WIDTH,
                expiry=new_expiry, t_minus_1=new_t_minus_1,
                bar_dt=bar["bar_dt"],
            )

    def _close_all_legs(self, reason: str, bar: dict) -> None:
        """Close every open leg via executor, reset state to NO_POSITION."""
        if self.executor:
            for pos in self.db.get_open_positions():
                self.executor.close_position(pos, reason=reason)
        self.state.state = "NO_POSITION"
        self.state.activated_at_bar = None
        self.state.activated_at_atm = None
        self.state.pyramid_atm = None
        self.state.last_cst_bar = None
        self.state.last_cst_high = None
        self.state.last_cst_low = None
        self.state.pyramid_level = 1
        self.state.stoch_left_zone_since_cst = False

    # ---- Kill switch ----

    def kill_switch(self) -> dict:
        """Emergency: close all legs, halt entries."""
        if self.executor:
            for pos in self.db.get_open_positions():
                self.executor.close_position(pos, reason="kill_switch")
        self.enabled = False
        self.state.state = "NO_POSITION"
        self.db.log_event("kill_switch", direction=None,
                          bar_dt=datetime.now().isoformat(), price=None,
                          notes="Kill switch activated — all legs closed, entries halted")
        return {"closed": True, "halted": True}

    # ---- Helpers ----

    def _bar_date(self, bar) -> date:
        s = bar["bar_dt"]
        if isinstance(s, str):
            return datetime.fromisoformat(s).date()
        return s.date() if hasattr(s, "date") else s

    def get_state_snapshot(self) -> dict:
        """Return a JSON-serializable view of engine state for the API."""
        return {
            "state_machine": self.state.state,
            "mst_direction": self.state.mst_direction,
            "armed_high": self.state.armed_high,
            "armed_low": self.state.armed_low,
            "activated_at_bar": self.state.activated_at_bar,
            "activated_at_atm": self.state.activated_at_atm,
            "pyramid_atm": self.state.pyramid_atm,
            "pyramid_level": self.state.pyramid_level,
            "current_expiry_dt": self.state.current_expiry_dt,
            "current_t_minus_1": self.state.current_t_minus_1,
            "last_cst_bar": self.state.last_cst_bar,
            "last_cst_high": self.state.last_cst_high,
            "last_cst_low": self.state.last_cst_low,
            "stoch_left_zone_since_cst": self.state.stoch_left_zone_since_cst,
            "enabled": self.enabled,
            "paper_mode": self.paper_mode,
            "buffer_size": len(self.bars),
        }
