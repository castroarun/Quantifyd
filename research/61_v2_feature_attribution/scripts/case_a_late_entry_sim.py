"""CASE A — synthetic-options late-entry simulation.
For each V2 cycle the combo-filter SKIPS (narrow prior-day CPR <0.10% OR inside-week),
compare three choices on a SYNTHETIC iron fly (BS priced off the daily India-VIX path):
  (a) SCHEDULED entry (what we'd skip) -> full-hold P&L
  (b) LATE entry next trading day IF the filter then clears (CPR ok AND not inside-week),
      same expiry, 1 fewer DTE -> full-hold P&L
  (c) SKIP = 0
Both held to a common synthetic expiry (entry+5 TD) with 2% underlying move-stop + PT40.
Engine is CALIBRATED against the real AlgoTest C P&L on the kept cycles first.
Approximate (BS not live chain, daily not intraday) -> read the RELATIVE deltas, not absolutes.
Read-only. No orders.
"""
import io, contextlib, json, math
import numpy as np, pandas as pd
from kiteconnect import KiteConnect
import config

R = 0.065; QTY = 650; WING = 500; STOP = 0.02; PT = 0.40
SLIP = 0.0025; FEES = 160.0

def Nc(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
def bs(S, K, T, s, t):
    if T <= 1e-9 or s <= 1e-9: return max(0.0, (S-K) if t == 'C' else (K-S))
    srt = s*math.sqrt(T); d1 = (math.log(S/K)+(R+0.5*s*s)*T)/srt; d2 = d1-srt
    return (S*Nc(d1)-K*math.exp(-R*T)*Nc(d2)) if t == 'C' else (K*math.exp(-R*T)*Nc(-d2)-S*Nc(-d1))
def fly_cost(S, K, T, s):   # net debit to CLOSE the short fly (>=0)
    return (bs(S,K,T,s,'C')+bs(S,K,T,s,'P')) - (bs(S,K+WING,T,s,'C')+bs(S,K-WING,T,s,'P'))

# ---- data ----
_b = io.StringIO()
with contextlib.redirect_stdout(_b): exec(open("/tmp/cd_data.py").read())
TR = [(d, v, p) for d, v, p in C if d not in ("2020-03-13", "2020-03-20")]
tok = json.load(open("backtest_data/access_token.json"))
k = KiteConnect(api_key=config.KITE_API_KEY); k.set_access_token(tok["access_token"])
def pull(t):
    out = {}
    for a, b in [("2017-01-01","2021-06-30"),("2021-07-01","2026-06-09")]:
        for c in k.historical_data(t, a, b, "day"):
            out[c["date"].strftime("%Y-%m-%d")] = (c["open"], c["high"], c["low"], c["close"])
    return out
bars = pull(256265); vixb = pull(264969)
dates = sorted(bars); idx = {d: i for i, d in enumerate(dates)}
vix = {d: vixb[d][3]/100.0 for d in vixb}
def ivat(d):
    j = idx[d]
    for kk in range(j, max(j-6, -1), -1):
        if dates[kk] in vix: return vix[dates[kk]]
    return 0.13

def cprw(d):
    i = idx[d]; ph, pl, pc = bars[dates[i-1]][1], bars[dates[i-1]][2], bars[dates[i-1]][3]
    piv = (ph+pl+pc)/3; bc = (ph+pl)/2; tc = 2*piv-bc; return abs(tc-bc)/bars[d][0]*100
# weekly inside flag (causal at entry d): last completed week inside the week before
wdf = pd.DataFrame([(d, *bars[d]) for d in dates], columns=["date","o","h","l","c"])
wdf["date"] = pd.to_datetime(wdf["date"]); wdf = wdf.set_index("date")
wk = wdf.resample("W-FRI").agg({"o":"first","h":"max","l":"min","c":"last"}).dropna()
wk["inside"] = (wk.h < wk.h.shift(1)) & (wk.l > wk.l.shift(1))
def inside_at(d):
    dt = pd.Timestamp(d); monday = dt - pd.Timedelta(days=dt.weekday())
    comp = wk[wk.index < monday]
    return bool(comp["inside"].iloc[-1]) if len(comp) else False
def clears(d):  # full combo filter PASSES at entry day d?
    return (cprw(d) >= 0.10) and (not inside_at(d))

Ts = pd.Timestamp
def sim_fly(d0, Eidx=None, exit_i=None):
    """Short-fly entered d0; expiry at index Eidx (default d0+4 TD = AlgoTest 4-DTE entry),
    rolled/exited at exit_i (default expiry-1 = 1 DTE), with 2% move-stop + PT40."""
    i0 = idx[d0]
    if Eidx is None: Eidx = i0+4
    if exit_i is None: exit_i = Eidx-1
    if Eidx >= len(dates): return None
    S0 = bars[d0][0]; K = round(S0/50)*50; Edate = dates[Eidx]
    s0 = ivat(d0); T0 = max((Ts(Edate)-Ts(d0)).days, 0)/365.0
    credit = fly_cost(S0, K, T0, s0)
    for j in range(i0+1, exit_i+1):
        dd = dates[j]; o, h, l, c = bars[dd]
        T = max((Ts(Edate)-Ts(dd)).days, 0)/365.0; s = ivat(dd)
        if max(abs(h-S0), abs(l-S0))/S0 >= STOP:
            Sstop = S0*(1+STOP) if abs(h-S0) > abs(l-S0) else S0*(1-STOP)
            return _net(credit - fly_cost(Sstop, K, T, s), credit)
        pnl = credit - fly_cost(c, K, T, s)
        if pnl >= PT*credit: return _net(pnl, credit)
    dd = dates[exit_i]; T = max((Ts(Edate)-Ts(dd)).days, 0)/365.0   # roll at 1 DTE
    return _net(credit - fly_cost(bars[dd][3], K, T, ivat(dd)), credit)
def _net(pnl_unit, credit):
    # fees + round-trip slippage ~ 0.25% each way on the credit premium traded
    return pnl_unit*QTY - FEES - SLIP*2*abs(credit)*QTY

# ---- calibration vs real C on KEPT cycles ----
kept = [(d, v, p) for d, v, p in TR if v >= 13 and clears(d)]
syn = []; act = []
for d, v, p in kept:
    s = sim_fly(d)
    if s is None: continue
    syn.append(s); act.append(p)
syn = np.array(syn); act = np.array(act)
print(f"CALIBRATION on kept cycles n={len(syn)}: corr(synthetic, AlgoTest)={np.corrcoef(syn,act)[0,1]:+.2f}")
print(f"  sign agreement = {(np.sign(syn)==np.sign(act)).mean()*100:.0f}%  "
      f"synthetic mean={syn.mean():,.0f}  AlgoTest mean={act.mean():,.0f}")

# ---- the skipped cycles: scheduled vs late entry ----
skipped = [(d, v, p) for d, v, p in TR if v >= 13 and not clears(d)]
print(f"\nSkipped cycles (VIX>=13, filter blocks): n={len(skipped)}, "
      f"actual AlgoTest sum = {sum(p for _,_,p in skipped):,.0f}")
sched = []; late = []; cleared = 0; never = 0
for d, v, p in skipped:
    i = idx[d]
    sched.append(sim_fly(d, Eidx=i+4, exit_i=i+3))
    # try next trading day; per user 'Monday' = +1 TD (then +2 fallback). SAME expiry (i+4).
    placed = None
    for step in (1, 2):
        if i+step <= i+3 and i+4 < len(dates) and clears(dates[i+step]):
            placed = sim_fly(dates[i+step], Eidx=i+4, exit_i=i+3); break
    if placed is None: never += 1; late.append(0.0)
    else: cleared += 1; late.append(placed)
sched = np.array(sched, dtype=float); late = np.array(late)
print(f"\n--- SKIPPED-CYCLE OUTCOMES (synthetic) ---")
print(f"  (a) SCHEDULED entry  : sum {sched.sum():>10,.0f}  mean {sched.mean():>8,.0f}  win {(sched>0).mean()*100:.0f}%")
print(f"  (b) LATE entry       : sum {late.sum():>10,.0f}  mean {late.mean():>8,.0f}  "
      f"({cleared} cleared & entered, {never} never cleared -> stayed skipped)")
print(f"  (c) SKIP entirely    : sum          0")
print(f"\n  late-vs-scheduled delta = {late.sum()-sched.sum():,.0f}  "
      f"(positive => delaying beats taking the scheduled trade)")
print(f"  late-vs-skip delta      = {late.sum():,.0f}  "
      f"(positive => delaying beats sitting out)")
print("\nDONE.")
