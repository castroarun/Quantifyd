"""
research/100 G1b — ATH breakout WITH VOLUME confirmation on the ATH candle.
Volume gauge = traded-value ratio (tv[i] / trailing-20d median tv). Tests whether a
volume-confirmed ATH beats (a) plain random entry, and crucially (b) a RANDOM day that
ALSO had the same volume spike — isolating the all-time-high from just "a big-volume day".
Exit = close-based Donchian-20. Net 0.4% RT. Same PIT MidSmallcap-400 universe.

Run: venv/bin/python research/100_ath_breakout/scripts/g1b_volume.py
"""
import importlib.util, csv
from pathlib import Path
import numpy as np, pandas as pd

RES = Path("/home/arun/quantifyd/research/100_ath_breakout/results")
ENGINE = Path("/home/arun/quantifyd/research/41_midsmall400_mq_concentrated/scripts/02_rs_sweep.py")
_s = importlib.util.spec_from_file_location("rs2", str(ENGINE)); rs2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(rs2)

RT = 0.004; START = pd.Timestamp("2014-01-01"); MIN_HIST = 252; MAX_HOLD = 504; N = 20
BAND = "combo"; VOL_LB = 20
KS = [1.5, 2.0, 3.0]                     # volume-spike multiples to test
rng = np.random.default_rng(42)

print("Loading ...", flush=True)
close, tv = rs2.load(); BENCH = rs2.NIFTY_SYM
mes = [d for d in rs2.month_ends(close.index) if d >= START - pd.Timedelta(days=400)]
memb = {d: set(rs2.pit_universe(tv, close, d, BAND)) for d in mes}
me_idx = pd.DatetimeIndex(sorted(memb))
print(f"  {close.shape[1]} syms, {len(memb)} universe snapshots", flush=True)


def eligible(sym, d):
    pos = me_idx.searchsorted(d, side="right") - 1
    return pos >= 0 and sym in memb[me_idx[pos]]


def don_ret(c, i):
    entry = c[i]
    if entry <= 0 or not np.isfinite(entry):
        return None
    for j in range(i + 1, min(len(c), i + 1 + MAX_HOLD)):
        if c[j] < np.nanmin(c[max(0, j - N):j]):
            return c[j] / entry - 1 - RT
    return c[-1] / entry - 1 - RT


coh = {("ath", k): [] for k in KS}
coh.update({("rndvol", k): [] for k in KS})
coh[("rnd", 0)] = []
syms = [s for s in close.columns if s != BENCH]
for si, sym in enumerate(syms):
    c = close[sym].dropna()
    if len(c) < MIN_HIST + 30:
        continue
    cv = c.values; dates = c.index
    t = tv[sym].reindex(c.index).values
    volratio = t / pd.Series(t).rolling(VOL_LB, min_periods=VOL_LB).median().shift(1).values
    cummax = np.maximum.accumulate(cv)
    is_ath = cv >= cummax * (1 - 1e-9)
    fresh = is_ath & ~np.concatenate([[True], is_ath[:-1]])
    base_ok = np.array([dates[i] >= START and eligible(sym, dates[i]) for i in range(len(cv))])
    ath_i = [i for i in range(MIN_HIST, len(cv) - 5) if fresh[i] and base_ok[i]]
    elig_i = [i for i in range(MIN_HIST, len(cv) - 5) if base_ok[i]]
    if not elig_i:
        continue
    # plain random baseline (matched to # of fresh ATH)
    if ath_i:
        picks = rng.choice(elig_i, size=min(len(ath_i), len(elig_i)), replace=False)
        for i in picks:
            r = don_ret(cv, int(i))
            if r is not None:
                coh[("rnd", 0)].append(r)
    for k in KS:
        av = [i for i in ath_i if np.isfinite(volratio[i]) and volratio[i] >= k]
        for i in av:
            r = don_ret(cv, i)
            if r is not None:
                coh[("ath", k)].append(r)
        # random days that ALSO had a >=k volume spike (matched count)
        rv = [i for i in elig_i if np.isfinite(volratio[i]) and volratio[i] >= k]
        if rv and av:
            picks = rng.choice(rv, size=min(len(av), len(rv)), replace=False)
            for i in picks:
                r = don_ret(cv, int(i))
                if r is not None:
                    coh[("rndvol", k)].append(r)
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


rndS, rndA = st(coh[("rnd", 0)])
print(f"\nDonchian-20, net 0.4%RT. Plain RANDOM: n={rndS['n']} mean={rndS['mean']:.2f}% win={rndS['win']:.1f}% t={rndS['t']:.1f}\n")
print(f"{'K':>4} | {'ATH+vol n':>9} {'mean%':>6} {'win%':>5} {'t':>5} | {'RndVol mean%':>12} {'win%':>5} | ATHvol-vs-RndVol t | ATHvol-vs-Rnd t")
rows = []
for k in KS:
    aS, aA = st(coh[("ath", k)]); rS, rA = st(coh[("rndvol", k)])
    t_vs_rndvol = welch(aA, rA); t_vs_rnd = welch(aA, rndA)
    print(f"{k:>4} | {aS['n']:>9} {aS['mean']:>6.2f} {aS['win']:>5.1f} {aS['t']:>5.1f} | {rS['mean']:>12.2f} {rS['win']:>5.1f} | {t_vs_rndvol:>+18.2f} | {t_vs_rnd:>+13.2f}")
    rows.append(dict(K=k, ath_n=aS['n'], ath_mean=round(aS['mean'],2), ath_win=round(aS['win'],1), ath_t=round(aS['t'],1),
                     rndvol_mean=round(rS['mean'],2), rndvol_win=round(rS['win'],1),
                     t_vs_rndvol=round(t_vs_rndvol,2), t_vs_rnd=round(t_vs_rnd,2)))
with open(RES/"g1b_volume.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(x) for x in rows]
print("\nGATE: ATH+vol must beat RANDOM+vol (t_vs_rndvol >~2) — else the edge is the volume spike, not the ATH.")
print("wrote results/g1b_volume.csv")
