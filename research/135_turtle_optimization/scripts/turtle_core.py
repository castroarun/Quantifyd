"""research/135 - unit-level Turtle simulator (faithful Rule 1-5 incl.
pyramiding) + a daily mark-to-market book NAV.

r/83's simulator was position-level and could not express Rule 4 (adding to
winners). This one tracks UNITS, so a position may hold 1..max_units, all
sharing one stop that ratchets up to `stop_mult`*N below the LAST unit's
entry - the faithful Turtle behaviour.

Conventions kept identical to r/83 so results are comparable:
  - entry/exit fills are next-day OPEN after a close-based signal
  - the stop is evaluated intrabar; a gap through it fills at the open
  - N = ATR20 (simple mean of true range) at the signal bar, FIXED for the
    life of the position
  - long only (shorts closed by r/81 + r/82 + r/83)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

R81 = Path(__file__).resolve().parents[2] / "81_swing_edge_discovery"
for p in (str(R81), str(R81.parents[1])):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine.costs import CostConfig                     # noqa: E402

COST = CostConfig(product="FUTURES_PROXY")
MAX_POS_NOTIONAL = 0.25
BOOK_NOTIONAL_CAP = 1.5


# --------------------------------------------------------------- signal layer

def atr20(df: pd.DataFrame) -> np.ndarray:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(20).mean().to_numpy(float)


def turtle_positions(df: pd.DataFrame, sym: str, n_in: int, n_out: int,
                     stop_mult=2.0, max_units: int = 1,
                     add_step: float = 0.5) -> list:
    """Long-only Turtle on one symbol. Returns positions, each with its units.

    A position's units all exit together (trail break or stop). `add_step` is
    in units of N; `stop_mult` None means no hard stop (trail only).
    """
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    idx = df.index
    hi_in = df["high"].rolling(n_in).max().shift(1).to_numpy(float)
    lo_out = df["low"].rolling(n_out).min().shift(1).to_numpy(float)
    atr = atr20(df)

    positions = []
    n = len(df)
    i = max(n_in, 20) + 1
    in_pos = False
    units = []
    stop = N = next_add = None

    while i < n - 1:
        if not in_pos:
            if c[i] > hi_in[i] and not np.isnan(atr[i]) and atr[i] > 0 \
                    and not np.isnan(hi_in[i]):
                ent = i + 1
                px = o[ent]
                if np.isnan(px) or px <= 0:
                    i += 1
                    continue
                N = atr[i]
                units = [{"day": idx[ent], "raw": float(px)}]
                stop = (px - stop_mult * N) if stop_mult else -np.inf
                next_add = px + add_step * N
                in_pos = True
                i = ent + 1
                continue
            i += 1
            continue

        # ---- in position: stop first (conservative), then adds, then re-check
        exited = False
        if o[i] <= stop:
            positions.append({"symbol": sym, "units": units, "exit_day": idx[i],
                              "exit_raw": float(o[i]), "reason": "STOP", "N": float(N)})
            exited = True
        elif lo[i] <= stop:
            positions.append({"symbol": sym, "units": units, "exit_day": idx[i],
                              "exit_raw": float(stop), "reason": "STOP", "N": float(N)})
            exited = True
        if exited:
            in_pos = False
            units = []
            i += 1
            continue

        added = False
        while len(units) < max_units and h[i] >= next_add:
            fillpx = o[i] if o[i] >= next_add else next_add
            units.append({"day": idx[i], "raw": float(fillpx)})
            if stop_mult:
                stop = fillpx - stop_mult * N
            next_add = fillpx + add_step * N
            added = True
        if added and stop_mult and lo[i] <= stop:
            positions.append({"symbol": sym, "units": units, "exit_day": idx[i],
                              "exit_raw": float(stop), "reason": "STOP", "N": float(N)})
            in_pos = False
            units = []
            i += 1
            continue

        if c[i] < lo_out[i] and i + 1 < n:
            positions.append({"symbol": sym, "units": units, "exit_day": idx[i + 1],
                              "exit_raw": float(o[i + 1]), "reason": "TRAIL",
                              "N": float(N)})
            in_pos = False
            units = []
            i += 2
            continue
        i += 1

    return positions


# ----------------------------------------------------------------- book layer

def _size(eq, px, sizing, unit_frac, risk_pct, stop_mult, N):
    if sizing == "EQ":
        return eq * unit_frac / px
    if not stop_mult or N <= 0:
        return eq * unit_frac / px
    return eq * risk_pct / (stop_mult * N)


def book_nav(positions: list, closes: dict, gate: dict, cal, cap: int = 12,
             sizing: str = "EQ", unit_frac: float = 0.10, risk_pct: float = 0.01,
             stop_mult=2.0, costs_on: bool = True, put_cost_bps: float = 0.0,
             put_payoff=None):
    """Daily MTM NAV. Units are sized independently; the position cap counts
    SYMBOLS (a pyramided position is one slot, not four).

    `put_payoff` (optional Series of daily NAV-fraction P&L from a protective
    put overlay) is added to the book each day, and `put_cost_bps` is charged
    daily as a fraction of NAV - used by Stage E.
    """
    by_day = {}
    for p in positions:
        by_day.setdefault(p["units"][0]["day"], []).append(p)

    def fillp(raw, is_buy):
        return COST.fill_price(raw, is_buy) if costs_on else raw

    def charge(px, qty, is_buy):
        return COST.side_cost(px, qty, is_buy) if costs_on else 0.0

    eq = 1.0
    live = []
    equity = pd.Series(np.nan, index=cal)

    for d in cal:
        pnl = 0.0

        # 1. exits at the open
        still = []
        for p in live:
            if d >= p["exit_day"]:
                xp = fillp(p["exit_raw"], is_buy=False)
                for u in p["open_units"]:
                    pnl += u["qty"] * (xp - u["mark"])
                    pnl -= charge(xp, u["qty"], False)
            else:
                still.append(p)
        live = still

        book_notional = sum(u["qty"] * u["mark"] for p in live for u in p["open_units"])

        # 2. pyramid adds on live positions
        for p in live:
            while p["next_unit"] < len(p["units"]) and \
                    p["units"][p["next_unit"]]["day"] <= d:
                u = p["units"][p["next_unit"]]
                p["next_unit"] += 1
                if u["day"] != d:
                    continue
                px = fillp(u["raw"], is_buy=True)
                qty = _size(eq, px, sizing, unit_frac, risk_pct, stop_mult, p["N"])
                held = sum(x["qty"] * x["mark"] for x in p["open_units"])
                head = min(eq * BOOK_NOTIONAL_CAP - book_notional,
                           eq * MAX_POS_NOTIONAL - held)
                qty = min(qty, max(head, 0.0) / px)
                if qty * px < eq * 0.005:
                    continue
                pnl -= charge(px, qty, True)
                p["open_units"].append({"qty": qty, "mark": px})
                book_notional += qty * px

        # 3. new positions at the open
        for p in by_day.get(d, []):
            if not bool(gate.get(d, False)) or len(live) >= cap:
                continue
            u0 = p["units"][0]
            px = fillp(u0["raw"], is_buy=True)
            qty = _size(eq, px, sizing, unit_frac, risk_pct, stop_mult, p["N"])
            head = min(eq * BOOK_NOTIONAL_CAP - book_notional, eq * MAX_POS_NOTIONAL)
            qty = min(qty, max(head, 0.0) / px)
            if qty * px < eq * 0.005:
                continue
            pnl -= charge(px, qty, True)
            live.append({"symbol": p["symbol"], "units": p["units"], "next_unit": 1,
                         "exit_day": p["exit_day"], "exit_raw": p["exit_raw"],
                         "N": p["N"], "open_units": [{"qty": qty, "mark": px}]})
            book_notional += qty * px

        # 4. mark to close
        for p in live:
            cl = closes[p["symbol"]].get(d, np.nan)
            if not np.isnan(cl):
                for u in p["open_units"]:
                    pnl += u["qty"] * (cl - u["mark"])
                    u["mark"] = cl

        # 5. optional protective-put overlay (Stage E)
        if put_payoff is not None:
            pnl += eq * float(put_payoff.get(d, 0.0))
        if put_cost_bps:
            pnl -= eq * put_cost_bps / 1e4

        eq += pnl
        if eq <= 0:
            eq = 1e-9
        equity[d] = eq

    return equity.ffill()
