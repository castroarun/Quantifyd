"""
research/100 G1 probe — All-Time-High breakout, entry at ATH-day close, close-based
Donchian-N trailing exit, on the MidSmallcap-400 (PIT combo band). MANDATORY control:
random-entry (same stocks, same exit) to isolate whether ATH TIMING adds value over just
holding a strong name (drift/survivorship). Kill-cheap: close-only; SuperTrend/OHLC in G2.

Run on VPS: venv/bin/python research/100_ath_breakout/scripts/g1_probe.py
"""
import importlib.util, csv
from pathlib import Path
import numpy as np, pandas as pd

RES = Path("/home/arun/quantifyd/research/100_ath_breakout/results"); RES.mkdir(parents=True, exist_ok=True)
ENGINE = Path("/home/arun/quantifyd/research/41_midsmall400_mq_concentrated/scripts/02_rs_sweep.py")
_s = importlib.util.spec_from_file_location("rs2", str(ENGINE)); rs2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(rs2)

RT = 0.004
START = pd.Timestamp("2014-01-01")
MIN_HIST = 252            # need >=1y history before an "all-time high" counts
MAX_HOLD = 504           # cap a trade at ~2y
BAND = "combo"           # MidSmallcap-400 = ranks 100..500 by trailing traded value
rng = np.random.default_rng(42)

print("Loading price+tv ...", flush=True)
close, tv = rs2.load()
BENCH = rs2.NIFTY_SYM
print(f"  {close.shape[1]} symbols {close.index.min().date()}..{close.index.max().date()}", flush=True)

# monthly PIT universe membership (combo = midsmall-400)
mes = [d for d in rs2.month_ends(close.index) if d >= START - pd.Timedelta(days=400)]
memb = {}
for d in mes:
    try:
        memb[d] = set(rs2.pit_universe(tv, close, d, BAND))
    except Exception:
        memb[d] = set()
me_idx = pd.DatetimeIndex(sorted(memb))
print(f"  built {len(memb)} monthly universe snapshots (avg {np.mean([len(v) for v in memb.values()]):.0f} names)", flush=True)


def eligible(sym, d):
    pos = me_idx.searchsorted(d, side="right") - 1
    return pos >= 0 and sym in memb[me_idx[pos]]


def donchian_exit_return(c, entry_i, N):
    """Enter at c[entry_i]; exit at first close < min(prior N closes); return (net_ret, hold)."""
    n = len(c)
    entry = c[entry_i]
    if entry <= 0 or not np.isfinite(entry):
        return None
    for j in range(entry_i + 1, min(n, entry_i + 1 + MAX_HOLD)):
        lo = np.nanmin(c[max(0, j - N):j])          # lowest close of prior N days (excl. today)
        if c[j] < lo:
            return (c[j] / entry - 1 - RT, j - entry_i)
    # exit at last bar
    return (c[-1] / entry - 1 - RT, n - 1 - entry_i)


NS = [10, 20, 30]
ath_trades = {N: [] for N in NS}
rnd_trades = {N: [] for N in NS}
tradelog = []

syms = [s for s in close.columns if s != BENCH]
for si, sym in enumerate(syms):
    c = close[sym].dropna()
    if len(c) < MIN_HIST + 30:
        continue
    cv = c.values; dates = c.index
    cummax = np.maximum.accumulate(cv)
    is_ath = cv >= cummax * (1 - 1e-9)
    fresh = is_ath & ~np.concatenate([[True], is_ath[:-1]])   # new-high, prior day not high
    # candidate entry positions: fresh ATH, enough history, in-universe, >=START
    cand = [i for i in range(MIN_HIST, len(cv))
            if fresh[i] and dates[i] >= START and eligible(sym, dates[i])]
    if not cand:
        continue
    # de-overlap ATH entries: skip a new entry while an earlier one would still be open
    for N in NS:
        open_until = -1
        for i in cand:
            if i <= open_until:
                continue
            r = donchian_exit_return(cv, i, N)
            if r is None:
                continue
            ath_trades[N].append(r[0]); open_until = i + r[1]
            if N == 20:
                tradelog.append((sym, str(dates[i].date()), round(r[0]*100, 2), r[1]))
    # RANDOM control: same # of entries as ATH(N=20), random eligible positions
    elig_pos = [i for i in range(MIN_HIST, len(cv) - 5) if dates[i] >= START and eligible(sym, dates[i])]
    if elig_pos:
        for N in NS:
            k = len([1 for i in cand if True])   # match candidate count
            k = min(k, len(elig_pos))
            if k == 0:
                continue
            picks = rng.choice(elig_pos, size=k, replace=False)
            for i in sorted(picks):
                r = donchian_exit_return(cv, int(i), N)
                if r is not None:
                    rnd_trades[N].append(r[0])
    if si % 200 == 0:
        print(f"  {si}/{len(syms)} stocks ... ATH(N20)={len(ath_trades[20])} trades", flush=True)


def stats(arr):
    a = np.array(arr, dtype=float); a = a[np.isfinite(a)]
    if len(a) == 0:
        return dict(n=0, mean=0, win=0, med=0, t=0)
    se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.nan
    return dict(n=len(a), mean=a.mean()*100, win=(a > 0).mean()*100, med=np.median(a)*100,
                t=(a.mean()/se) if se and se > 0 else 0)


print("\n=== PER-TRADE (net of 0.4% RT), ATH-entry vs RANDOM-entry, same Donchian exit ===")
print(f"{'N':>3} | {'cohort':7} | {'trades':>6} {'mean%':>7} {'win%':>6} {'med%':>6} {'t':>6} | ATH-vs-RND Welch-t")
rows = []
for N in NS:
    A = stats(ath_trades[N]); R = stats(rnd_trades[N])
    a = np.array(ath_trades[N]); r = np.array(rnd_trades[N])
    # Welch t of difference
    if len(a) > 1 and len(r) > 1:
        wt = (a.mean() - r.mean()) / np.sqrt(a.var(ddof=1)/len(a) + r.var(ddof=1)/len(r))
    else:
        wt = 0
    print(f"{N:>3} | {'ATH':7} | {A['n']:>6} {A['mean']:>7.2f} {A['win']:>6.1f} {A['med']:>6.2f} {A['t']:>6.1f} | diff-t={wt:+.2f}")
    print(f"{N:>3} | {'RANDOM':7} | {R['n']:>6} {R['mean']:>7.2f} {R['win']:>6.1f} {R['med']:>6.2f} {R['t']:>6.1f} |")
    rows.append(dict(N=N, ath_n=A['n'], ath_mean=round(A['mean'],2), ath_win=round(A['win'],1), ath_t=round(A['t'],1),
                     rnd_n=R['n'], rnd_mean=round(R['mean'],2), rnd_win=round(R['win'],1), rnd_t=round(R['t'],1),
                     diff_welch_t=round(wt,2)))

with open(RES/"g1_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(x) for x in rows]
pd.DataFrame(tradelog, columns=["sym","entry","net_ret_pct","hold_days"]).to_csv(RES/"g1_trades_N20.csv", index=False)
print("\nwrote results/g1_summary.csv + g1_trades_N20.csv")
print("VERDICT GATE: ATH must beat RANDOM (diff-t >~2) on >=2 of 3 N to proceed to G2.")
