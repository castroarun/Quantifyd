"""
research/100 G1c — CONSOLIDATION breakout to a new ATH on volume ("resistance wall
broken / tight base explodes"). The strongest form of the hypothesis. Controls:
 - random-day + same volume spike (does the SETUP beat a random big-volume day?)
 - plain ATH + volume (does the CONSOLIDATION add anything over any ATH breakout?)

Setup: over the prior W days the base is TIGHT ((maxclose-minclose)/minclose <= T) AND
today's close breaks above that base high (the wall) AND is a new all-time high AND
volume-ratio >= K. Exit close-Donchian-20, net 0.4% RT. PIT MidSmallcap-400.

Run: venv/bin/python research/100_ath_breakout/scripts/g1c_consolidation.py
"""
import importlib.util, csv
from pathlib import Path
import numpy as np, pandas as pd

RES = Path("/home/arun/quantifyd/research/100_ath_breakout/results")
ENGINE = Path("/home/arun/quantifyd/research/41_midsmall400_mq_concentrated/scripts/02_rs_sweep.py")
_s = importlib.util.spec_from_file_location("rs2", str(ENGINE)); rs2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(rs2)

RT = 0.004; START = pd.Timestamp("2014-01-01"); MIN_HIST = 252; MAX_HOLD = 504; N = 20
BAND = "combo"; VOL_LB = 20; K = 2.0; T = 0.15         # base tightness 15%
WS = [20, 40, 60]                                       # consolidation windows
rng = np.random.default_rng(42)

print("Loading ...", flush=True)
close, tv = rs2.load(); BENCH = rs2.NIFTY_SYM
mes = [d for d in rs2.month_ends(close.index) if d >= START - pd.Timedelta(days=400)]
memb = {d: set(rs2.pit_universe(tv, close, d, BAND)) for d in mes}
me_idx = pd.DatetimeIndex(sorted(memb))
print(f"  {close.shape[1]} syms, {len(memb)} snapshots", flush=True)


def eligible(sym, d):
    p = me_idx.searchsorted(d, side="right") - 1
    return p >= 0 and sym in memb[me_idx[p]]


def don_ret(c, i):
    entry = c[i]
    if entry <= 0 or not np.isfinite(entry):
        return None
    for j in range(i + 1, min(len(c), i + 1 + MAX_HOLD)):
        if c[j] < np.nanmin(c[max(0, j - N):j]):
            return c[j] / entry - 1 - RT
    return c[-1] / entry - 1 - RT


cons = {w: [] for w in WS}                 # consolidation-breakout+vol
rndvol = []                                # random + vol spike
plain_ath_vol = []                         # any fresh ATH + vol (no consolidation)
syms = [s for s in close.columns if s != BENCH]
for si, sym in enumerate(syms):
    c = close[sym].dropna()
    if len(c) < MIN_HIST + 70:
        continue
    cv = c.values; dates = c.index
    t = tv[sym].reindex(c.index).values
    med = pd.Series(t).rolling(VOL_LB, min_periods=VOL_LB).median().shift(1).values
    volratio = np.divide(t, med, out=np.full_like(t, np.nan, float), where=(med > 0))
    cummax = np.maximum.accumulate(cv)
    is_ath = cv >= cummax * (1 - 1e-9)
    fresh = is_ath & ~np.concatenate([[True], is_ath[:-1]])
    base_ok = np.array([dates[i] >= START and eligible(sym, dates[i]) for i in range(len(cv))])
    elig_i = [i for i in range(MIN_HIST, len(cv) - 5) if base_ok[i]]
    if not elig_i:
        continue
    volok = np.isfinite(volratio) & (volratio >= K)
    # plain ATH + vol
    pav = [i for i in range(MIN_HIST, len(cv) - 5) if fresh[i] and base_ok[i] and volok[i]]
    for i in pav:
        r = don_ret(cv, i)
        if r is not None:
            plain_ath_vol.append(r)
    # random + vol (matched to plain-ath-vol count)
    rv = [i for i in elig_i if volok[i]]
    if rv and pav:
        for i in rng.choice(rv, size=min(len(pav), len(rv)), replace=False):
            r = don_ret(cv, int(i))
            if r is not None:
                rndvol.append(r)
    # consolidation breakout + vol, per W
    for w in WS:
        for i in range(MIN_HIST, len(cv) - 5):
            if not (fresh[i] and base_ok[i] and volok[i]):
                continue
            base = cv[i - w:i]
            if len(base) < w:
                continue
            lo, hi = np.nanmin(base), np.nanmax(base)
            if lo <= 0:
                continue
            if (hi - lo) / lo <= T and cv[i] > hi:      # tight base + clears the wall
                r = don_ret(cv, i)
                if r is not None:
                    cons[w].append(r)
    if si % 300 == 0:
        print(f"  {si}/{len(syms)} ...", flush=True)


def st(a):
    a = np.array(a, float); a = a[np.isfinite(a)]
    if len(a) < 2:
        return dict(n=len(a), mean=0, win=0, t=0), a
    se = a.std(ddof=1)/np.sqrt(len(a))
    return dict(n=len(a), mean=a.mean()*100, win=(a > 0).mean()*100, t=a.mean()/se if se > 0 else 0), a


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0
    return (a.mean()-b.mean())/np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))


rS, rA = st(rndvol); pS, pA = st(plain_ath_vol)
print(f"\nDonchian-20 net 0.4%. Random+vol(K{K}): n={rS['n']} mean={rS['mean']:.2f}% win={rS['win']:.1f}% t={rS['t']:.1f}")
print(f"Plain-ATH+vol:          n={pS['n']} mean={pS['mean']:.2f}% win={pS['win']:.1f}% t={pS['t']:.1f}\n")
print(f"{'W':>3} | {'cons+vol n':>10} {'mean%':>6} {'win%':>5} {'t':>5} | vs Random+vol t | vs plain-ATH+vol t")
rows = []
for w in WS:
    cS, cA = st(cons[w])
    tr = welch(cA, rA); tp = welch(cA, pA)
    print(f"{w:>3} | {cS['n']:>10} {cS['mean']:>6.2f} {cS['win']:>5.1f} {cS['t']:>5.1f} | {tr:>+15.2f} | {tp:>+17.2f}")
    rows.append(dict(W=w, n=cS['n'], mean=round(cS['mean'],2), win=round(cS['win'],1), t=round(cS['t'],1),
                     vs_rndvol_t=round(tr,2), vs_plainathvol_t=round(tp,2)))
with open(RES/"g1c_consolidation.csv", "w", newline="") as f:
    ww = csv.DictWriter(f, fieldnames=list(rows[0].keys())); ww.writeheader(); [ww.writerow(x) for x in rows]
print("\nGATE: consolidation+vol must beat Random+vol (t>~2) to be a real setup edge.")
print("wrote results/g1c_consolidation.csv")
