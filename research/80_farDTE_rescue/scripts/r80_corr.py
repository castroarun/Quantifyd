#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/80 G6 — the question that decides everything: is this a NEW bet, or the SAME bet again?

Every options book Arun runs is short volatility: the 916 ATM straddles, the squeeze strangles,
NAS-OPT. The overnight condor is ALSO short volatility. If it loses on the same days, it does not
diversify the book -- it concentrates it, and on the day the steamroller arrives everything goes at
once. That is the "correlation / single-factor" deadly sin, and it matters more than the Calmar.

So: replay the EXISTING intraday book on the same 11 years with the same engine --
    NAS-916 proxy : SELL ATM straddle 09:16 -> exit 15:15, per-leg 30% stop
    NAS-OPT proxy : SELL ~100pt-OTM strangle 09:20 -> 14:45, +/-0.4% move-stop, 0/1-DTE only
-- and measure, against the overnight condor:
    * correlation of P&L
    * what the intraday book did on the condor's 10 worst days   <- the real question
    * the COMBINED equity curve and drawdown vs the condor alone
Reads only.
"""
import json
import math
import sqlite3
from collections import defaultdict
from datetime import date, timedelta

import numpy as np

ROOT = "/home/arun/quantifyd"
CAL = json.load(open(f"{ROOT}/research/80_farDTE_rescue/results/engine_calib.json"))["iv_mult_by_dte"]
QTY, BROK, SLIP = 130, 20, 0.01


def ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs(S, K, T, iv, kind):
    if T <= 0 or iv <= 0:
        return max(0.0, (S - K) if kind == "CE" else (K - S))
    d1 = (math.log(S / K) + 0.5 * iv * iv * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    if kind == "CE":
        return S * ncdf(d1) - K * ncdf(d2)
    return K * ncdf(-d2) - S * ncdf(-d1)


def ivm(d):
    if str(d) in CAL:
        return CAL[str(d)]
    ks = sorted(int(k) for k in CAL)
    return CAL[str(min(ks, key=lambda k: abs(k - d)))]


def next_exp(d):
    t = 1 if d >= date(2025, 9, 1) else 3
    a = (t - d.weekday()) % 7
    return d if a == 0 else d + timedelta(days=a)


c = sqlite3.connect(f"file:{ROOT}/backtest_data/market_data.db?mode=ro", uri=True)
day_c = {r[0][:10]: r[1] for r in c.execute(
    "SELECT date, close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day'")}
vix = {r[0][:10]: r[1] for r in c.execute(
    "SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day'")}
bars = c.execute("SELECT date, open, high, low, close FROM market_data_unified "
                 "WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date").fetchall()
c.close()

intraday = defaultdict(list)
for d, o, h, l, cl in bars:
    intraday[str(d)[:10]].append((str(d)[11:16], o, h, l, cl))

def mn(x):
    return int(x[:2]) * 60 + int(x[3:5])

sess = sorted(d for d in day_c if d in vix)
idx = {d: i for i, d in enumerate(sess)}

# ---------------- the OVERNIGHT CONDOR (the candidate) -----------------------
cond = {}
for day in sess:
    y, m, dd = map(int, day.split("-"))
    d0 = date(y, m, dd)
    exp = next_exp(d0)
    if (exp - d0).days != 6:
        continue
    es = exp.isoformat()
    if es not in idx:
        continue
    S0, v0 = day_c[day], vix[day]
    iv0, T0 = (v0 / 100.0) * ivm(6), 6 / 365.0
    Kc, Kp = round((S0 + 150) / 50) * 50, round((S0 - 150) / 50) * 50
    Kcw, Kpw = Kc + 250, Kp - 250
    cred = (bs(S0, Kc, T0, iv0, "CE") + bs(S0, Kp, T0, iv0, "PE")
            - bs(S0, Kcw, T0, iv0, "CE") - bs(S0, Kpw, T0, iv0, "PE"))
    if cred < 2:
        continue
    val = cred
    for s in sess[idx[day] + 1: idx[es] + 1]:
        Sx, vx = day_c[s], vix.get(s, v0)
        sy, sm, sd = map(int, s.split("-"))
        left = (exp - date(sy, sm, sd)).days
        if left <= 0:
            val = (max(0.0, Sx - Kc) + max(0.0, Kp - Sx)
                   - max(0.0, Sx - Kcw) - max(0.0, Kpw - Sx))
            break
        ivx, Tx = (vx / 100.0) * ivm(left), left / 365.0
        val = (bs(Sx, Kc, Tx, ivx, "CE") + bs(Sx, Kp, Tx, ivx, "PE")
               - bs(Sx, Kcw, Tx, ivx, "CE") - bs(Sx, Kpw, Tx, ivx, "PE"))
        if val >= cred * 2.0:
            break
    # book the P&L on the day it is CLOSED (that is when the loss lands)
    cond[day] = (cred * (1 - SLIP) - val * (1 + SLIP)) * QTY - 4 * BROK

# ---------------- the EXISTING INTRADAY BOOK (proxy) --------------------------
straddle, nasopt = {}, {}
for day in sess:
    b = sorted(intraday.get(day, []))
    if len(b) < 40:
        continue
    v0 = vix.get(day)
    if not v0:
        continue
    y, m, dd = map(int, day.split("-"))
    d0 = date(y, m, dd)
    dte = (next_exp(d0) - d0).days
    iv = (v0 / 100.0) * ivm(dte)

    e916 = next((x for x in b if mn(x[0]) >= 9 * 60 + 16), None)
    e_opt = next((x for x in b if mn(x[0]) >= 9 * 60 + 20), None)
    if not e916 or not e_opt:
        continue

    # --- 916 ATM straddle: sell ATM, per-leg 30% stop, exit 15:15
    S0 = e916[4]
    K = round(S0 / 50) * 50
    T0 = max((dte + 6.0 / 24) / 365.0, 1e-6)
    ce0, pe0 = bs(S0, K, T0, iv, "CE"), bs(S0, K, T0, iv, "PE")
    ce_x = pe_x = None
    win = [x for x in b if 9 * 60 + 16 <= mn(x[0]) <= 15 * 60 + 15]
    for x in win:
        left_h = max((15 * 60 + 30 - mn(x[0])) / 60.0, 0.05)
        Tx = max((dte + left_h / 24) / 365.0, 1e-6)
        ce = bs(x[4], K, Tx, iv, "CE")
        pe = bs(x[4], K, Tx, iv, "PE")
        if ce_x is None and ce >= ce0 * 1.3:
            ce_x = ce
        if pe_x is None and pe >= pe0 * 1.3:
            pe_x = pe
    last = win[-1][4]
    Tl = max((dte + 0.25 / 24) / 365.0, 1e-6)
    ce_x = ce_x if ce_x is not None else bs(last, K, Tl, iv, "CE")
    pe_x = pe_x if pe_x is not None else bs(last, K, Tl, iv, "PE")
    straddle[day] = ((ce0 + pe0) * (1 - SLIP) - (ce_x + pe_x) * (1 + SLIP)) * QTY - 2 * BROK

    # --- NAS-OPT: 100pt OTM strangle, 0.4% move stop, 0/1-DTE only
    if dte <= 1:
        S0 = e_opt[4]
        Kc, Kp = round((S0 + 100) / 50) * 50, round((S0 - 100) / 50) * 50
        T0 = max((dte + 6.0 / 24) / 365.0, 1e-6)
        c0, p0 = bs(S0, Kc, T0, iv, "CE"), bs(S0, Kp, T0, iv, "PE")
        ex = None
        for x in [z for z in b if 9 * 60 + 20 <= mn(z[0]) <= 14 * 60 + 45]:
            left_h = max((15 * 60 + 30 - mn(x[0])) / 60.0, 0.05)
            Tx = max((dte + left_h / 24) / 365.0, 1e-6)
            if abs(x[4] - S0) / S0 >= 0.004:
                ex = bs(x[4], Kc, Tx, iv, "CE") + bs(x[4], Kp, Tx, iv, "PE")
                break
            last_t, last_s = Tx, x[4]
        if ex is None:
            ex = bs(last_s, Kc, last_t, iv, "CE") + bs(last_s, Kp, last_t, iv, "PE")
        nasopt[day] = ((c0 + p0) * (1 - SLIP) - ex * (1 + SLIP)) * QTY - 2 * BROK

# ---------------- correlate -------------------------------------------------
common = sorted(set(cond) & set(straddle))
A = np.array([cond[d] for d in common])
B = np.array([straddle[d] for d in common])
common2 = sorted(set(cond) & set(nasopt))
A2 = np.array([cond[d] for d in common2])
C2 = np.array([nasopt[d] for d in common2])

print("=== IS THE OVERNIGHT CONDOR A NEW BET, OR THE SAME BET AGAIN? ===\n")
print(f"  condor vs 916-straddle proxy : corr {np.corrcoef(A, B)[0,1]:+.3f}   (n={len(common)} shared days)")
print(f"  condor vs NAS-OPT proxy      : corr {np.corrcoef(A2, C2)[0,1]:+.3f}   (n={len(common2)})")

print("\n\n--- THE REAL TEST: what did the intraday book do on the condor's 10 WORST days? ---\n")
print(f"{'day':>12s} {'CONDOR':>10s} {'916 straddle':>14s} {'both lose?':>12s}")
worst = sorted(cond, key=lambda d: cond[d])[:10]
both = 0
for d in worst:
    s = straddle.get(d)
    flag = ""
    if s is not None and s < 0:
        both += 1
        flag = "YES <-"
    print(f"{d:>12s} {cond[d]:>+10,.0f} {('' if s is None else format(s, '+,.0f')):>14s} {flag:>12s}")
print(f"\n  the intraday book ALSO lost on {both} of the condor's 10 worst days.")

print("\n\n--- combined book: does adding the condor make the drawdown better or worse? ---\n")
alld = sorted(set(cond) | set(straddle))
def dd(series):
    eq = np.cumsum(series)
    return (eq - np.maximum.accumulate(eq)).min()
s_only = np.array([straddle.get(d, 0.0) for d in alld])
c_only = np.array([cond.get(d, 0.0) for d in alld])
print(f"  916 straddle alone : total {s_only.sum():+,.0f}   maxDD {dd(s_only):+,.0f}")
print(f"  condor alone       : total {c_only.sum():+,.0f}   maxDD {dd(c_only):+,.0f}")
print(f"  BOTH together      : total {(s_only+c_only).sum():+,.0f}   maxDD {dd(s_only+c_only):+,.0f}")
add = dd(s_only + c_only) - dd(s_only)
print(f"\n  adding the condor changes the book's max drawdown by {add:+,.0f}")
print("  (if the condor diversified, the combined DD would be LESS than the sum of the parts)")
print(f"  sum of the two separate DDs = {dd(s_only)+dd(c_only):+,.0f}")
print("\nDONE")
