# -*- coding: utf-8 -*-
"""PHASE 2 · Test A (long-vol tail hedge blended with naked straddle) + Test B (weekly multi-asset trend).
A: does an optimally-sized long-OTM-options overlay lift the naked straddle's combined Calmar net of carry?
B: does trend have a standalone edge on WEEKLY bars / with GOLD, where daily NIFTY had none?"""
import sqlite3, json, math, datetime as dt
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=20000")
strad = defaultdict(float)
for t in json.load(open("/tmp/realistic_trades.json"))["dte3"]: strad[t["exit"][:7]] += t["naked"]
def calmar_of(pl):
    cum = pk = mdd = 0
    for x in pl: cum += x; pk = max(pk, cum); mdd = min(mdd, cum - pk)
    return (sum(pl) / (len(pl) / 12)) / abs(mdd) if mdd else 0, mdd
st_months = sorted(strad)
st_cal, st_mdd = calmar_of([strad[m] for m in st_months])

def bars(sym, frm="2018-01-01"):
    return c.execute("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol=? AND timeframe='day' AND close>0 AND date>=? ORDER BY date", (sym, frm)).fetchall()

# ---------- Test A: long-vol overlay ----------
def long_strangle_monthly(sym, step, lot, pct, frm="2016-06-01"):
    spot = {d: (o, cl) for d, o, h, l, cl in bars(sym, frm)}
    allexp = sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol=? AND expiry_date>=?", (sym, frm))})
    bymon = defaultdict(list)
    for e in allexp: bymon[e[:7]].append(e)
    monthly = sorted(max(v) for v in bymon.values())
    tset = set(spot)
    def leg(d, E, K, ot):
        r = c.execute("SELECT open,close,open_interest FROM nse_options_bhav WHERE symbol=? AND trade_date=? AND expiry_date=? AND strike=? AND option_type=?", (sym, d, E, K, ot)).fetchone()
        return (r[0] if r and r[0] else (r[1] if r else None), r[2] if r else 0) if r else None
    out = defaultdict(float)
    for E in monthly:
        if E not in spot: continue
        d = None
        for k in range(8):
            cd = (dt.date.fromisoformat(E) - dt.timedelta(days=28 + k)).isoformat()
            if cd in tset: d = cd; break
        if not d: continue
        atm = round(spot[d][0] / step) * step
        Kc = round(atm * (1 + pct / 100) / step) * step; Kp = round(atm * (1 - pct / 100) / step) * step
        ce = leg(d, E, Kc, "CE"); pe = leg(d, E, Kp, "PE")
        if not ce or not pe or ce[1] < 50 or pe[1] < 50: continue
        cost = ce[0] + pe[0]; sexp = spot[E][1]
        payoff = max(sexp - Kc, 0) + max(Kp - sexp, 0)
        out[E[:7]] += (payoff - cost - 0.003 * cost) * lot
    return out

print(f"STRADDLE alone (monthly): Calmar {st_cal:.2f}, maxDD ₹{st_mdd:+,.0f}\n== TEST A: long-vol overlay blended with the naked straddle ==")
for name, ov in [("NIFTY long 3% strangle (lot75)", long_strangle_monthly("NIFTY50", 50, 75, 3)),
                 ("BNF long 1% strangle (lot15)", long_strangle_monthly("BANKNIFTY", 100, 15, 1))]:
    best = (st_cal, 0)
    line = []
    for w in [0, 1, 2, 3, 5, 8]:
        pl = [strad[m] + w * ov.get(m, 0) for m in st_months]
        cal, mdd = calmar_of(pl)
        line.append(f"w{w}:{cal:.2f}")
        if cal > best[0]: best = (cal, w)
    print(f"  {name}: " + " ".join(line) + f"  -> best w={best[1]} Calmar {best[0]:.2f}")

# ---------- Test B: weekly multi-asset trend ----------
def weekly(sym, frm="2016-01-01"):
    b = bars(sym, frm); wk = {}
    for d, o, h, l, cl in b:
        y, w, _ = dt.date.fromisoformat(d).isocalendar()
        key = (y, w)
        if key not in wk: wk[key] = [d, o, h, l, cl]
        else:
            wk[key][2] = max(wk[key][2], h); wk[key][3] = min(wk[key][3], l); wk[key][4] = cl; wk[key][0] = d
    return [wk[k] for k in sorted(wk)]
def donch_weekly(sym, N=10):
    b = weekly(sym); out = []; pos = 0
    for i in range(N, len(b) - 1):
        hh = max(x[2] for x in b[i-N:i]); ll = min(x[3] for x in b[i-N:i]); cl = b[i][4]
        if cl > hh: pos = 1
        elif cl < ll: pos = -1
        out.append((b[i+1][0], pos * (b[i+1][4] / b[i][4] - 1)))
    return out
print("\n== TEST B: weekly Donchian-10 (does higher timeframe / gold have standalone edge?) ==")
for sym in ["NIFTY50", "BANKNIFTY", "GOLDBEES"]:
    s = donch_weekly(sym); r = [x[1] for x in s]; n = len(r); mu = sum(r) / n
    sd = (sum((x - mu) ** 2 for x in r) / n) ** 0.5; sh = mu / sd * math.sqrt(52) if sd else 0
    cum = pk = mdd = 0
    for x in r: cum += x; pk = max(pk, cum); mdd = min(mdd, cum - pk)
    tot = sum(r)
    print(f"  {sym:>10}: weekly Sharpe {sh:+.2f} | total {tot*100:+.0f}% | maxDD {mdd*100:+.0f}% | ann {tot/(n/52)*100:+.1f}%")
# multi-asset weekly trend basket monthly, correlation with straddle
basket = defaultdict(list)
for sym in ["NIFTY50", "BANKNIFTY", "GOLDBEES"]:
    for d, ret in donch_weekly(sym): basket[d[:7]].append(ret)
tmon = {m: sum(v) / len(v) for m, v in basket.items()}
common = sorted(set(tmon) & set(strad))
xs = [tmon[m] for m in common]; ys = [strad[m] for m in common]; nn = len(xs)
mx = sum(xs)/nn; my = sum(ys)/nn
cov = sum((xs[i]-mx)*(ys[i]-my) for i in range(nn))/nn
sx = (sum((x-mx)**2 for x in xs)/nn)**0.5; sy = (sum((y-my)**2 for y in ys)/nn)**0.5
print(f"  multi-asset weekly-trend basket vs straddle: corr {cov/(sx*sy) if sx and sy else 0:+.2f} (n={nn} months)")
c.close()
