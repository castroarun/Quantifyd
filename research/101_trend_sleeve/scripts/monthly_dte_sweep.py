# -*- coding: utf-8 -*-
"""MONTHLY NIFTY short straddle — full DTE sweep (2011-2026, 15y). Enter at DTE-N before the monthly
expiry (open-based ATM, no look-ahead), hold to DTE-1. Longer DTE = more exposure = bigger tail.
Then blend the BNF long-vol cushion with the higher-exposure straddles to see where it earns its keep."""
import sqlite3, datetime as dt
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
COST, SLIP, QTY = 160, 0.003, 50
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=30000")
NF = {d: (o, cl) for d, o, cl in c.execute("SELECT date,open,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day' AND close>0")}
BN = {d: (o, cl) for d, o, cl in c.execute("SELECT date,open,close FROM market_data_unified WHERE symbol='BANKNIFTY' AND timeframe='day' AND close>0")}
def monthly_exp(sym):
    allexp = sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol=? AND expiry_date>='2011-01-01'", (sym,))})
    bymon = defaultdict(list)
    for e in allexp: bymon[e[:7]].append(e)
    return sorted(max(v) for v in bymon.values())
tdays = sorted({r[0] for r in c.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol='NIFTY'")})
tset = set(tdays)
def dte(E, d): return (dt.date.fromisoformat(E) - dt.date.fromisoformat(d)).days
def leg(sym, d, E, K, ot):
    r = c.execute("SELECT open,close,open_interest FROM nse_options_bhav WHERE symbol=? AND trade_date=? AND expiry_date=? AND strike=? AND option_type=?", (sym, d, E, K, ot)).fetchone()
    if not r: return None
    return (r[0] if r[0] else r[1], r[1], r[2] or 0)
NEXP = monthly_exp("NIFTY")
def straddle(N):
    out = {}
    for E in NEXP:
        tgt = dt.date.fromisoformat(E) - dt.timedelta(days=N); d = None
        for k in range(6):
            cd = (tgt - dt.timedelta(days=k)).isoformat()
            if cd in tset and cd in NF: d = cd; break
        if not d or dte(E, d) < 1: continue
        K = round(NF[d][0] / 50) * 50
        ce = leg("NIFTY", d, E, K, "CE"); pe = leg("NIFTY", d, E, K, "PE")
        if not ce or not pe or ce[2] < 50 or pe[2] < 50: continue
        hold = [x for x in tdays if d <= x <= E and dte(E, x) >= 1]; xd = hold[-1] if hold else d
        cf = leg("NIFTY", xd, E, K, "CE"); pf = leg("NIFTY", xd, E, K, "PE")
        if not cf or not pf: continue
        out[E[:7]] = round((ce[0] + pe[0] - (cf[1] + pf[1])) * QTY - COST - SLIP * (ce[0] + pe[0] + cf[1] + pf[1]) * QTY)
    return out
def bnf_cushion():
    out = defaultdict(float)
    for E in monthly_exp("BANKNIFTY"):
        if E not in BN: continue
        d = None
        for k in range(8):
            cd = (dt.date.fromisoformat(E) - dt.timedelta(days=28 + k)).isoformat()
            if cd in tset and cd in BN: d = cd; break
        if not d: continue
        atm = round(BN[d][0] / 100) * 100; Kc = round(atm * 1.01 / 100) * 100; Kp = round(atm * 0.99 / 100) * 100
        ce = leg("BANKNIFTY", d, E, Kc, "CE"); pe = leg("BANKNIFTY", d, E, Kp, "PE")
        if not ce or not pe or ce[2] < 30 or pe[2] < 30: continue
        out[E[:7]] = (max(BN[E][1] - Kc, 0) + max(Kp - BN[E][1], 0) - (ce[0] + pe[0]) - SLIP * (ce[0] + pe[0])) * 15
    return out
def stats(series):
    mm = sorted(series); pl = [series[m] for m in mm]; n = len(pl); tot = sum(pl)
    cum = pk = mdd = 0
    for x in pl: cum += x; pk = max(pk, cum); mdd = min(mdd, cum - pk)
    win = 100 * sum(1 for x in pl if x > 0) / n
    negy = sum(1 for y in {m[:4] for m in mm} if sum(series[m] for m in mm if m[:4] == y) < 0)
    cal = (tot / (n / 12)) / abs(mdd) if mdd else 0
    return n, tot, cal, mdd, win, min(pl), negy
print("MONTHLY NIFTY short straddle — DTE-entry sweep (2011-2026, QTY50, hold to DTE-1):\n")
print(f"{'entry':>6}  {'n':>3} {'total':>10} {'Calmar':>7} {'maxDD':>10} {'win':>5} {'worstMo':>9} {'negYrs':>6}")
S = {}
for N in [1, 3, 5, 7, 10, 15, 20, 28]:
    s = straddle(N); S[N] = s
    n, tot, cal, mdd, win, worst, negy = stats(s)
    print(f"DTE-{N:<2}  {n:>3} {tot:>+10,} {cal:>7.2f} {mdd:>+10,} {win:>4.0f}% {worst:>+9,} {negy:>6}")
bnf = bnf_cushion()
print("\nBNF cushion blend on the higher-exposure straddles (combined Calmar):")
for N in [15, 20, 28]:
    mm = sorted(S[N]); best = (stats(S[N])[2], 0)
    line = []
    for w in [0, 1, 2, 3]:
        comb = {m: S[N][m] + w * bnf.get(m, 0) for m in mm}
        _, _, cal, mdd, _, _, _ = stats(comb); line.append(f"w{w}:{cal:.2f}")
        if cal > best[0]: best = (cal, w)
    print(f"  DTE-{N}: " + " ".join(line) + f"  -> best w={best[1]} Calmar {best[0]:.2f}")
c.close()
