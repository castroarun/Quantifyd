#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/132 shared helpers: chain loading, forward reading, Black-76 delta, costs.

READ-ONLY on every DB. Nothing here writes outside research/132/results.
"""
import os
import sqlite3
from datetime import date, timedelta
from math import log, sqrt, exp, erf, pi

Q = "/home/arun/quantifyd/"
CHAIN = Q + "backtest_data/options_data.db"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
LOG = os.path.join(RES, "stage.log")

VENUE = {
    "NIFTY":  dict(lot=65, step=50),
    "SENSEX": dict(lot=20, step=100),
}

# --- MEASURED cost model, lifted verbatim from research/122 stage_a_alldays.py -------
SLIP_ENTRY = 0.0
SLIP_TIME = 0.178
SLIP_STOP = 6.548
NLOTS_REF = 10


def cost_per_lot(credit, exitp, lot, reason):
    sell = credit * lot
    buy = exitp * lot
    tot = sell + buy
    brok = 80.0 / NLOTS_REF
    stt = 0.001 * sell
    txn = 0.0003503 * tot
    ipft = 0.0000050 * tot
    sebi = 0.0000010 * tot
    stamp = 0.00003 * buy
    gst = 0.18 * (brok + txn + ipft + sebi)
    slip = 2 * SLIP_ENTRY + 2 * (SLIP_STOP if reason == "SL" else SLIP_TIME)
    return brok + stt + txn + ipft + sebi + stamp + gst + slip * lot


SESS_END_M = 15 * 60 + 20
WD = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def log_line(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
    print(m, flush=True)


def hm2m(h):
    return int(h[:2]) * 60 + int(h[3:5])


def m2hm(m):
    return "%02d:%02d" % (m // 60, m % 60)


def trading_dte(day, exp):
    d0, d1 = date.fromisoformat(day), date.fromisoformat(exp)
    n, d = 0, d0
    while d < d1:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def ro(path):
    return sqlite3.connect("file:%s?mode=ro" % path, uri=True)


def all_days(c, sym):
    return [r[0] for r in c.execute(
        "SELECT DISTINCT substr(snapshot_time,1,10) d FROM download_log WHERE symbol=? "
        "ORDER BY d", (sym,)) if date.fromisoformat(r[0]).weekday() < 5]


def load_day(c, sym, day, all_expiries=False):
    """Return (front_expiry, {minute: spot}, {minute: {K: (ce, pe)}}) or None.

    Guards: frozen-chain holiday (<50 distinct spot prints), partial session
    (last snapshot before 15:15).
    """
    rows = c.execute(
        "SELECT snapshot_time, expiry_date, strike, instrument_type, ltp, underlying_spot "
        "FROM option_chain WHERE symbol=? AND snapshot_time>=? AND snapshot_time<? "
        "AND ltp IS NOT NULL", (sym, day, day + "z")).fetchall()
    if not rows:
        return None
    last_snap = max(r[0] for r in rows)
    if last_snap[11:16] < "15:15":
        return None
    exps = sorted({e for (_, e, _, _, _, _) in rows if e and e >= day})
    if not exps:
        return None
    fexp = exps[0]
    spot, chain = {}, {}
    for st, e, k, it, ltp, sp in rows:
        mi = hm2m(st[11:16])
        if sp and mi not in spot:
            spot[mi] = sp
        if e != fexp:
            continue
        d = chain.setdefault(mi, {}).setdefault(k, {})
        d[it] = ltp
    if len(set(spot.values())) < 50:
        return None
    ch2 = {}
    for mi, ks in chain.items():
        ch2[mi] = {k: (v["CE"], v["PE"]) for k, v in ks.items() if "CE" in v and "PE" in v}
    return fexp, spot, ch2


def read_forward(ks, spot, step):
    """Synthetic forward from put-call parity, read at the spot-nearest strike, with a
    +/-1-step cross-check.

    Returns (F, K_ref, spread) where spread is the max-min across the up-to-three PCP
    readings (the measurement noise floor). None if unreadable.
    """
    if not ks or spot is None:
        return None
    kref = round(spot / step) * step
    reads = []
    for kk in (kref - step, kref, kref + step):
        v = ks.get(kk) or ks.get(float(kk))
        if v is None:
            continue
        ce, pe = v
        if ce is None or pe is None or ce <= 0 or pe <= 0:
            continue
        reads.append((kk, kk + (ce - pe)))
    if not reads:
        return None
    mid = [f for (kk, f) in reads if abs(kk - kref) < 1e-6]
    F = mid[0] if mid else reads[len(reads) // 2][1]
    fs = [f for (_, f) in reads]
    spread = max(fs) - min(fs) if len(fs) > 1 else 0.0
    return F, kref, spread


# --- Black-76 ---------------------------------------------------------------------
def _N(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _b76_straddle(F, K, T, s):
    if s <= 0 or T <= 0:
        return max(F - K, 0.0) + max(K - F, 0.0)
    d1 = (log(F / K) + 0.5 * s * s * T) / (s * sqrt(T))
    d2 = d1 - s * sqrt(T)
    call = F * _N(d1) - K * _N(d2)
    put = K * _N(-d2) - F * _N(-d1)
    return call + put


def implied_vol_straddle(F, K, T, price):
    """Bisection on [1%, 400%] for the straddle combined premium. None on failure."""
    if price is None or price <= 0 or T <= 0 or F <= 0 or K <= 0:
        return None
    lo, hi = 0.01, 4.0
    if _b76_straddle(F, K, T, lo) > price or _b76_straddle(F, K, T, hi) < price:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _b76_straddle(F, K, T, mid) < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


T_FLOOR = 1.0 / (252.0 * 6.25)      # one trading hour, in years


def tenor_years(dte_trd, minute=None):
    """Trading-day tenor. DTE0 gets the fraction of the session still to run."""
    if dte_trd is None:
        return T_FLOOR
    if dte_trd <= 0:
        if minute is None:
            return T_FLOOR
        rem = max(SESS_END_M - minute, 5) / 375.0     # 375-min session
        return max(rem / 252.0, T_FLOOR)
    frac = 1.0
    if minute is not None:
        frac = max(SESS_END_M - minute, 5) / 375.0
    return max((dte_trd + frac) / 252.0, T_FLOOR)


def net_delta_short_straddle(F, K, T, sigma):
    """Net delta of a SHORT straddle at K when the forward is F. Positive = net long."""
    if sigma is None or sigma <= 0 or T <= 0:
        return None
    d1 = (log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt(T))
    return 1.0 - 2.0 * _N(d1)
