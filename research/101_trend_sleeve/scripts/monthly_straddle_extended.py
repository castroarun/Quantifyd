# -*- coding: utf-8 -*-
"""2011-2026 MONTHLY NIFTY straddle (DTE-3, open-based ATM, net) + BNF long-vol cushion, tested across
MULTIPLE crashes (2011 EU, 2013 taper, 2015-16 China, 2018, 2020) instead of just COVID. Does the
cushion help repeatedly, or was 2020 a one-off? 15 years of real option premiums."""
import sqlite3, datetime as dt
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
COST, SLIP = 160, 0.003
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=30000")
def idx(sym): return {d: (o, cl) for d, o, cl in c.execute("SELECT date,open,close FROM market_data_unified WHERE symbol=? AND timeframe='day' AND close>0", (sym,))}
NF, BN = idx("NIFTY50"), idx("BANKNIFTY")
def monthly_exp(sym, frm):
    allexp = sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol=? AND expiry_date>=?", (sym, frm))})
    bymon = defaultdict(list)
    for e in allexp: bymon[e[:7]].append(e)
    return sorted(max(v) for v in bymon.values())
tset = {r[0] for r in c.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol='NIFTY'")}
def leg(sym, d, E, K, ot):
    r = c.execute("SELECT open,close,open_interest FROM nse_options_bhav WHERE symbol=? AND trade_date=? AND expiry_date=? AND strike=? AND option_type=?", (sym, d, E, K, ot)).fetchone()
    if not r: return None
    return (r[0] if r[0] else r[1], r[1], r[2] or 0)
def dte(E, d): return (dt.date.fromisoformat(E) - dt.date.fromisoformat(d)).days
# --- NIFTY monthly straddle (DTE-3, QTY 50) ---
strad = {}
for E in monthly_exp("NIFTY", "2011-01-01"):
    tgt = dt.date.fromisoformat(E) - dt.timedelta(days=3); d = None
    for k in range(5):
        cd = (tgt - dt.timedelta(days=k)).isoformat()
        if cd in tset and cd in NF: d = cd; break
    if not d or dte(E, d) < 1: continue
    K = round(NF[d][0] / 50) * 50
    ce = leg("NIFTY", d, E, K, "CE"); pe = leg("NIFTY", d, E, K, "PE")
    if not ce or not pe or ce[2] < 50 or pe[2] < 50: continue
    hold = [x for x in sorted(tset) if d <= x <= E and dte(E, x) >= 1]; xd = hold[-1] if hold else d
    cf = leg("NIFTY", xd, E, K, "CE"); pf = leg("NIFTY", xd, E, K, "PE")
    if not cf or not pf: continue
    strad[E[:7]] = round((ce[0] + pe[0] - (cf[1] + pf[1])) * 50 - COST - SLIP * (ce[0] + pe[0] + cf[1] + pf[1]) * 50)
# --- BNF 1% long strangle cushion (QTY 15, ~30 DTE monthly) ---
bnf = defaultdict(float)
for E in monthly_exp("BANKNIFTY", "2011-01-01"):
    if E not in BN: continue
    d = None
    for k in range(8):
        cd = (dt.date.fromisoformat(E) - dt.timedelta(days=28 + k)).isoformat()
        if cd in tset and cd in BN: d = cd; break
    if not d: continue
    atm = round(BN[d][0] / 100) * 100; Kc = round(atm * 1.01 / 100) * 100; Kp = round(atm * 0.99 / 100) * 100
    ce = leg("BANKNIFTY", d, E, Kc, "CE"); pe = leg("BANKNIFTY", d, E, Kp, "PE")
    if not ce or not pe or ce[2] < 30 or pe[2] < 30: continue
    sexp = BN[E][1]
    bnf[E[:7]] = (max(sexp - Kc, 0) + max(Kp - sexp, 0) - (ce[0] + pe[0]) - SLIP * (ce[0] + pe[0])) * 15
def calmar(mm, series):
    pl = [series[m] for m in mm]; cum = pk = mdd = 0
    for x in pl: cum += x; pk = max(pk, cum); mdd = min(mdd, cum - pk)
    return sum(pl), (sum(pl) / (len(mm) / 12)) / abs(mdd) if mdd else 0, mdd
mm = sorted(strad)
print(f"NIFTY monthly straddle (DTE-3, QTY50): {mm[0]}..{mm[-1]} ({len(mm)} months / {len(mm)/12:.1f}y)")
tot, cal, mdd = calmar(mm, strad)
print(f"  total ₹{tot:+,} | Calmar {cal:.2f} | maxDD ₹{mdd:+,}")
print("  per-year:", " ".join(f"{y}:{sum(strad[m] for m in mm if m[:4]==y)//1000:+}k" for y in sorted({m[:4] for m in mm})))
print("\nCRASH MONTHS — straddle vs BNF cushion (3 lots):")
for cm in ["2011-08", "2011-09", "2013-06", "2013-08", "2015-08", "2016-02", "2018-02", "2018-10", "2020-03"]:
    if cm in strad:
        print(f"  {cm}: straddle ₹{strad[cm]:>+9,} | BNF×3 ₹{3*bnf.get(cm,0):>+9,.0f}  {'CUSHION ✓' if bnf.get(cm,0)>0 else '—'}")
print("\nBNF cushion blend (combined Calmar):")
c0 = calmar(mm, strad)[1]
for w in [0, 1, 2, 3, 5]:
    comb = {m: strad[m] + w * bnf.get(m, 0) for m in mm}
    _, cal, mdd = calmar(mm, comb)
    print(f"  +{w} BNF lots: Calmar {cal:.2f} maxDD ₹{mdd:+,.0f}")
c.close()
