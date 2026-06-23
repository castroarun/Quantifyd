"""
Intraday PANIC / SPIKE quantification for NIFTY.

Goal: a desk-side number (no news needed) that says "this move is beyond normal -> act".
Sources:
  - options_data.db underlying_spot (per-minute NIFTY, since 2026-04-20) -> recent intraday spike events + today
  - market_data.db market_data_unified NIFTY50 5minute (2015-2026)       -> long-history tail for thresholds

Metrics per day:
  V15  = worst (most negative) rolling 15-min return  (velocity)
  V30  = worst rolling 30-min return
  DDhi = max drop from the running intraday high       (depth)
  RNG  = day high-low range %
  SIG15= std of 15-min returns that day
  Z15  = |V15| / SIG15                                  (abnormality, sigmas)
and the CLOCK TIME of the worst 15-min window (to spot the ~3PM events).
"""
import sqlite3, datetime as dt
from collections import defaultdict

def pct(sorted_vals, p):
    if not sorted_vals: return float('nan')
    i = min(len(sorted_vals)-1, max(0, int(round(p/100*(len(sorted_vals)-1)))))
    return sorted_vals[i]

def stdev(xs):
    n=len(xs)
    if n<2: return 0.0
    m=sum(xs)/n
    return (sum((x-m)**2 for x in xs)/(n-1))**0.5

# ---------- helpers on a per-day minute/bar series ----------
def day_metrics(times_min, prices):
    """times_min: minutes-from-midnight (ints, ascending); prices aligned."""
    n=len(prices)
    if n<6: return None
    # build minute->price map for time-based windows
    pm={t:p for t,p in zip(times_min,prices)}
    def price_at_or_before(t):
        # nearest price at minute <= t (within 3 min)
        for k in range(t, t-4, -1):
            if k in pm: return pm[k]
        return None
    def worst_window(win):
        worst=0.0; worst_t=None
        for t,p in zip(times_min,prices):
            p0=price_at_or_before(t-win)
            if p0 and p0>0:
                r=(p-p0)/p0*100
                if r<worst: worst=r; worst_t=t
        return worst, worst_t
    v15,t15=worst_window(15)
    v30,t30=worst_window(30)
    # 15-min returns sample for sigma
    rets=[]
    for t,p in zip(times_min,prices):
        p0=price_at_or_before(t-15)
        if p0 and p0>0: rets.append((p-p0)/p0*100)
    sig15=stdev(rets)
    # drawdown from running high
    hi=prices[0]; ddhi=0.0
    for p in prices:
        if p>hi: hi=p
        d=(p-hi)/hi*100
        if d<ddhi: ddhi=d
    rng=(max(prices)-min(prices))/min(prices)*100
    z15=abs(v15)/sig15 if sig15>0 else 0.0
    return dict(v15=v15,t15=t15,v30=v30,t30=t30,ddhi=ddhi,rng=rng,sig15=sig15,z15=z15,
                day_move=(prices[-1]-prices[0])/prices[0]*100)

def hm_to_min(hm):
    return int(hm[:2])*60+int(hm[3:5])
def min_to_hm(m):
    if m is None: return "--:--"
    return f"{m//60:02d}:{m%60:02d}"

# ================= RECENT: per-minute from recorder =================
oc=sqlite3.connect('backtest_data/options_data.db')
rows=oc.execute("SELECT snapshot_time,spot_price FROM underlying_spot WHERE symbol='NIFTY' AND spot_price>0 ORDER BY snapshot_time").fetchall()
byday=defaultdict(dict)
for t,p in rows:
    d=t[:10]; m=hm_to_min(t[11:16])
    byday[d][m]=p   # last price wins per minute
recent=[]
for d in sorted(byday):
    items=sorted(byday[d].items())
    tm=[x[0] for x in items]; px=[x[1] for x in items]
    mt=day_metrics(tm,px)
    if mt: recent.append((d,mt))

print("="*78)
print("RECENT INTRADAY MOVES — per-minute recorder (NIFTY, since",recent[0][0] if recent else "n/a",")")
print("="*78)
print(f"{'date':11} {'V15%':>7} {'@time':>6} {'V30%':>7} {'DDhi%':>7} {'rng%':>6} {'sig15':>6} {'Z15':>5} {'dayMv%':>7}")
for d,m in sorted(recent,key=lambda x:x[1]['v15'])[:12]:
    print(f"{d:11} {m['v15']:7.2f} {min_to_hm(m['t15']):>6} {m['v30']:7.2f} {m['ddhi']:7.2f} {m['rng']:6.2f} {m['sig15']:6.3f} {m['z15']:5.1f} {m['day_move']:7.2f}")

# today (last day in recorder)
if recent:
    td,tm_=recent[-1]
    print("\nTODAY in recorder:",td)
    print(f"  V15={tm_['v15']:.2f}% @ {min_to_hm(tm_['t15'])}  V30={tm_['v30']:.2f}%  DDfromHigh={tm_['ddhi']:.2f}%  range={tm_['rng']:.2f}%  sig15={tm_['sig15']:.3f}  Z15={tm_['z15']:.1f}  dayMove={tm_['day_move']:.2f}%")

# ~3PM events scan (worst 15-min window between 14:30-15:30)
print("\n3PM-WINDOW falls (worst 15-min move that occurred 14:30-15:30):")
pm_events=[]
for d,m in recent:
    if m['t15'] is not None and 14*60+30 <= m['t15'] <= 15*60+30 and m['v15']<=-0.25:
        pm_events.append((d,m))
for d,m in sorted(pm_events,key=lambda x:x[1]['v15'])[:8]:
    print(f"  {d}  V15={m['v15']:.2f}% @ {min_to_hm(m['t15'])}  DDhi={m['ddhi']:.2f}%  dayMove={m['day_move']:.2f}%")
if not pm_events: print("  (none >=0.25% in recorder window)")

# ================= LONG HISTORY: 5-min bars =================
mc=sqlite3.connect('backtest_data/market_data.db')
bar=mc.execute("SELECT date,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' AND close>0 ORDER BY date").fetchall()
byday5=defaultdict(list)
for date_s,c in bar:
    d=date_s[:10]; hm=date_s[11:16]
    byday5[d].append((hm_to_min(hm),c))
long_=[]
for d in sorted(byday5):
    items=sorted(byday5[d]); tm=[x[0] for x in items]; px=[x[1] for x in items]
    mt=day_metrics(tm,px)
    if mt: long_.append((d,mt))

v15s=sorted(m['v15'] for _,m in long_)         # most negative first
z15s=sorted((m['z15'] for _,m in long_),reverse=True)
dd=sorted(m['ddhi'] for _,m in long_)
print("\n"+"="*78)
print(f"LONG HISTORY 5-min NIFTY50: {len(long_)} days, {long_[0][0]}..{long_[-1][0]}")
print("="*78)
print("Worst-15min-drop distribution (V15, % ):")
for p in (50,75,90,95,99,99.5,99.9):
    # percentile of |drop|: take from the negative tail
    idx=int(round((1-p/100)*(len(v15s)-1)))
    print(f"  p{p:<5}: {v15s[idx]:6.2f}%   (i.e. {100-p:.1f}% of days had a 15-min drop worse than this)")
print("Intraday Z15 (sigmas) distribution:")
for p in (50,90,95,99,99.9):
    print(f"  p{p:<5}: {pct(sorted(z15s),p):5.1f} sigma")
print("Max drop-from-high (DDhi, %) distribution:")
for p in (50,90,95,99,99.9):
    idx=int(round((1-p/100)*(len(dd)-1)))
    print(f"  p{p:<5}: {dd[idx]:6.2f}%")

print("\nTop 10 worst 15-min intraday drops in 5-min history (with clock time):")
for d,m in sorted(long_,key=lambda x:x[1]['v15'])[:10]:
    print(f"  {d}  V15={m['v15']:6.2f}% @ {min_to_hm(m['t15'])}  DDhi={m['ddhi']:6.2f}%  rng={m['rng']:5.2f}%  Z15={m['z15']:4.1f}  dayMove={m['day_move']:6.2f}%")
