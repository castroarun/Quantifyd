# -*- coding: utf-8 -*-
"""PHASE 2 · C2 full — dual-ST FUTURES + option HEDGE (Arun's spec). On a counter-trend child flip,
sell 1 ATM option (short CE if long fut / short PE if short fut) = covered-call/put cushion; buy it
back and pyramid +1 future on the next with-trend child flip; close all on master flip. NIFTY daily,
EOD-close actions, real option premiums. Compare hedge on/off standalone + straddle blend."""
import sqlite3, sys, math, json, datetime as dt
import pandas as pd
from collections import defaultdict
sys.path.insert(0, "/home/arun/quantifyd")
from services.maruthi_strategy import compute_dual_supertrend
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
LOT, PT_COST, MAXLOTS, SLIP = 75, 3.0, 3, 0.003
c = sqlite3.connect(DB)
rows = c.execute("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day' AND close>0 AND date>='2018-01-01' ORDER BY date").fetchall()
base = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close'])
# NIFTY monthly expiries + ATM premium lookup
allexp = sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date>='2018-01-01'")})
bymon = defaultdict(list)
for e in allexp: bymon[e[:7]].append(e)
monthly = sorted(max(v) for v in bymon.values())
def dte(E, d): return (dt.date.fromisoformat(E) - dt.date.fromisoformat(d)).days
def near_monthly(d):
    for E in monthly:
        if dte(E, d) >= 10: return E
    return None
def opt_close(d, E, K, ot):
    r = c.execute("SELECT close FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date=? AND expiry_date=? AND strike=? AND option_type=?", (d, E, K, ot)).fetchone()
    return float(r[0]) if r and r[0] else None

def backtest(mp, mm, cp, cm, hedge_on):
    df = compute_dual_supertrend(base.copy(), master_period=mp, master_mult=mm, child_period=cp, child_mult=cm)
    md = df['master_dir'].values; cfb = df['child_flip_bull'].values; cfe = df['child_flip_bear'].values
    cl = df['close'].values; dates = [str(x) for x in df['date'].values]
    n = len(df); warm = max(mp, cp) + 2
    pos = 0; prev = 0; hedge = None; daily = []
    for t in range(warm, n - 1):
        d = int(md[t]); day = dates[t]; ev = 0.0
        def close_hedge(dd):
            nonlocal hedge
            if not hedge: return 0.0
            K, ot, ps, E = hedge; hedge = None
            pc = opt_close(dd, E, K, ot)
            if pc is None: pc = max((cl[t] - K), 0) if ot == "CE" else max((K - cl[t]), 0)  # intrinsic fallback
            return (ps - pc) * LOT - SLIP * (ps + pc) * LOT
        if d != prev:                                    # master flip → close all + reverse
            ev += close_hedge(day); ev -= (abs(pos) + 1) * PT_COST; pos = d
        else:
            if abs(pos) < MAXLOTS and ((d == 1 and cfb[t]) or (d == -1 and cfe[t])):   # with-trend child → resume
                ev += close_hedge(day); pos += d; ev -= PT_COST
            elif hedge_on and hedge is None and ((d == 1 and cfe[t]) or (d == -1 and cfb[t])):  # counter child → hedge
                E = near_monthly(day); K = round(cl[t] / 50) * 50; ot = "CE" if d == 1 else "PE"
                ps = opt_close(day, E, K, ot) if E else None
                if ps: hedge = (K, ot, ps, E)
        daily.append((dates[t + 1], pos * (cl[t + 1] - cl[t]) + ev))
        prev = d
    return daily

def stats(daily):
    p = [x[1] for x in daily]; n = len(p); tot = sum(p); mu = tot / n
    sd = (sum((x - mu) ** 2 for x in p) / n) ** 0.5
    cum = pk = mdd = 0
    for x in p: cum += x; pk = max(pk, cum); mdd = min(mdd, cum - pk)
    yrs = n / 252
    return tot, (mu / sd * math.sqrt(252) if sd else 0), mdd, ((tot / yrs) / abs(mdd) if mdd else 0)

# straddle monthly for blend
strad = defaultdict(float)
for t in json.load(open("/tmp/realistic_trades.json"))["dte3"]: strad[t["exit"][:7]] += t["naked"]
def blend_calmar(daily):
    tmon = defaultdict(float)
    for dd, p in daily: tmon[dd[:7]] += p * 1   # already ₹ (LOT applied in futures pnl? no — futures pnl is points*pos; multiply by LOT)
    months = sorted(set(tmon) & set(strad))
    best = (-9, 0)
    for w in [0, 1, 2, 3, 5]:
        pl = [strad[m] + w * tmon[m] * LOT for m in months]
        cum = pk = mdd = 0
        for x in pl: cum += x; pk = max(pk, cum); mdd = min(mdd, cum - pk)
        cal = (sum(pl) / (len(pl) / 12)) / abs(mdd) if mdd else 0
        if cal > best[0]: best = (cal, w)
    return best

print("Dual-ST + hedge — NIFTY daily 2018-2026 (points; ×₹75/lot). Best params ST(14,3)/ST(7,2.5):\n")
for mp, mm, cp, cm in [(14, 3.0, 7, 2.5), (14, 3.0, 10, 2.5), (10, 4.0, 7, 2.0)]:
    for hedge_on in [False, True]:
        d = backtest(mp, mm, cp, cm, hedge_on)
        tot, sh, mdd, cal = stats(d)
        bc = blend_calmar(d)
        tag = "HEDGED" if hedge_on else "naked "
        print(f"ST({mp},{mm:.0f})/ST({cp},{cm:.1f}) {tag}: +{tot:>6.0f}pts Sharpe {sh:+.2f} maxDD {mdd:>+7.0f} Calmar {cal:+.2f} | best blend w={bc[1]} Calmar {bc[0]:.2f} (straddle-alone 2.66)")
c.close()
