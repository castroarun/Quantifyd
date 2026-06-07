#!/usr/bin/env python3
"""
G1 probe — NIFTY 30-min double-Supertrend regime machine, SIGNAL ONLY.

Question: does the dual-Supertrend regime (MST slow + CST fast) capture NIFTY
trend net of switching cost? Posture per 30-min bar:
    posture = mst_dir if mst_dir == cst_dir else 0   (+1 long, -1 short, 0 flat)
Held over the NEXT bar's return (causal, no look-ahead). Equity compounded.

This is NOT an options P&L. It is the kill-test: if the regime classifier isn't
on the right side of NIFTY net of cost, the options wrapper is irrelevant.

Pure stdlib. Reads market_data.db read-only. Designed to run on the VPS.
Outputs results/g1_ranking.csv + a console summary.
"""
import sqlite3, math, csv, os
from datetime import datetime
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RESDIR, exist_ok=True)

# VPS canonical path; fall back to relative for portability
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
if not os.path.exists(DB):
    DB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                      "backtest_data", "market_data.db")

INSTRUMENTS = ["NIFTY50", "BANKNIFTY"]
START = "2024-03-01"
PERIODS = [7, 10, 14]
MST_MULTS = [3.0, 4.0, 5.0]
CST_MULTS = [1.5, 2.0, 2.5]
COSTS_BPS = [0, 5, 10, 20]      # per posture-change, reported as sensitivity
BARS_PER_YEAR = 11 * 250        # ~11 30-min bars/session * 250 sessions

# ---------------------------------------------------------------------------
def load_5min(con, sym):
    rows = con.execute(
        "SELECT date,open,high,low,close FROM market_data_unified "
        "WHERE symbol=? AND timeframe='5minute' AND date>=? ORDER BY date",
        (sym, START)).fetchall()
    out = []
    for d, o, h, l, c in rows:
        dt = datetime.strptime(d, "%Y-%m-%d %H:%M:%S")
        out.append((dt, float(o), float(h), float(l), float(c)))
    return out

def resample_30min(bars5):
    """Group 5-min bars into 30-min session buckets (09:15 anchored)."""
    buckets = defaultdict(list)
    for dt, o, h, l, c in bars5:
        mins = dt.hour * 60 + dt.minute
        if mins < 9 * 60 + 15 or mins > 15 * 60 + 30:
            continue
        bidx = (mins - (9 * 60 + 15)) // 30
        buckets[(dt.date(), bidx)].append((dt, o, h, l, c))
    out = []
    for key in sorted(buckets):
        seg = buckets[key]
        o = seg[0][1]
        h = max(x[2] for x in seg)
        l = min(x[3] for x in seg)
        c = seg[-1][4]
        out.append((seg[-1][0], o, h, l, c))   # timestamp = last 5-min ts
    return out

def supertrend_dir(bars, period, mult):
    """Canonical Wilder-ATR Supertrend. Returns list of +1/-1 directions."""
    n = len(bars)
    if n < period + 1:
        return [1] * n
    tr = [0.0] * n
    for i in range(n):
        _, o, h, l, c = bars[i]
        tr[i] = (h - l) if i == 0 else max(h - l, abs(h - bars[i-1][4]),
                                           abs(l - bars[i-1][4]))
    atr = [0.0] * n
    atr[period-1] = sum(tr[:period]) / period
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    fub = [0.0] * n      # final upper band
    flb = [0.0] * n      # final lower band
    st  = [0.0] * n
    direction = [1] * n
    for i in range(n):
        _, o, h, l, c = bars[i]
        if i < period:
            continue
        mid = (h + l) / 2.0
        bub = mid + mult * atr[i]
        blb = mid - mult * atr[i]
        pf_ub = fub[i-1] if fub[i-1] else bub
        pf_lb = flb[i-1] if flb[i-1] else blb
        fub[i] = bub if (bub < pf_ub or bars[i-1][4] > pf_ub) else pf_ub
        flb[i] = blb if (blb > pf_lb or bars[i-1][4] < pf_lb) else pf_lb
        prev_st = st[i-1] if st[i-1] else fub[i]
        if prev_st == (fub[i-1] if fub[i-1] else fub[i]):
            st[i] = fub[i] if c <= fub[i] else flb[i]
        else:
            st[i] = flb[i] if c >= flb[i] else fub[i]
        direction[i] = 1 if c >= st[i] else -1
    return direction

# ---------------------------------------------------------------------------
def evaluate(bars, period, mst_mult, cst_mult, cost_bps):
    mst = supertrend_dir(bars, period, mst_mult)
    cst = supertrend_dir(bars, period, cst_mult)
    n = len(bars)
    warm = period + 1
    posture = [0] * n
    for i in range(n):
        if i < warm:
            posture[i] = 0
        else:
            posture[i] = mst[i] if mst[i] == cst[i] else 0

    cost = cost_bps / 10000.0
    eq = 1.0
    curve = [1.0]
    peak = 1.0; maxdd = 0.0
    rets = []
    switches = 0
    state_bars = {"bull": 0, "bear": 0, "neutral": 0}
    neutral_fwd = []   # next-bar drift while flat (to check 'flat is justified')
    prev_p = 0
    for i in range(warm, n - 1):
        p = posture[i]
        if p == 1: state_bars["bull"] += 1
        elif p == -1: state_bars["bear"] += 1
        else: state_bars["neutral"] += 1
        fwd = (bars[i+1][4] - bars[i][4]) / bars[i][4]
        if p == 0:
            neutral_fwd.append(fwd)
        r = p * fwd
        if p != prev_p:
            switches += 1
            r -= cost * abs(p - prev_p) / 1.0   # cost scales with size of change
        prev_p = p
        eq *= (1 + r)
        curve.append(eq)
        rets.append(r)
        peak = max(peak, eq)
        dd = (eq - peak) / peak
        maxdd = min(maxdd, dd)

    # buy & hold over same window
    bh = (bars[n-1][4] - bars[warm][4]) / bars[warm][4]
    bh_eq = 1.0; bh_peak = 1.0; bh_maxdd = 0.0
    for i in range(warm, n - 1):
        bh_eq *= (1 + (bars[i+1][4] - bars[i][4]) / bars[i][4])
        bh_peak = max(bh_peak, bh_eq)
        bh_maxdd = min(bh_maxdd, (bh_eq - bh_peak) / bh_peak)

    nbars = len(rets)
    years = nbars / BARS_PER_YEAR if nbars else 1
    cagr = eq ** (1/years) - 1 if eq > 0 and years > 0 else -1
    bh_cagr = (1 + bh) ** (1/years) - 1 if years > 0 else 0
    mean = sum(rets)/len(rets) if rets else 0
    var = sum((x-mean)**2 for x in rets)/len(rets) if rets else 0
    sd = math.sqrt(var)
    sharpe = (mean/sd*math.sqrt(BARS_PER_YEAR)) if sd > 0 else 0
    calmar = (cagr/abs(maxdd)) if maxdd < 0 else 0
    bh_calmar = (bh_cagr/abs(bh_maxdd)) if bh_maxdd < 0 else 0
    neu_drift = sum(neutral_fwd)/len(neutral_fwd) if neutral_fwd else 0

    # per-year net return
    by_year = defaultdict(float)
    j = 0
    for i in range(warm, n - 1):
        by_year[bars[i][0].year] += rets[j]; j += 1

    return dict(
        cagr=cagr*100, sharpe=sharpe, maxdd=maxdd*100, calmar=calmar,
        final_mult=eq, switches=switches, switches_yr=switches/years,
        bull_pct=100*state_bars["bull"]/max(1,sum(state_bars.values())),
        bear_pct=100*state_bars["bear"]/max(1,sum(state_bars.values())),
        neutral_pct=100*state_bars["neutral"]/max(1,sum(state_bars.values())),
        neutral_drift_bps=neu_drift*10000,
        bh_cagr=bh_cagr*100, bh_maxdd=bh_maxdd*100, bh_calmar=bh_calmar,
        by_year={y: round(v*100, 1) for y, v in sorted(by_year.items())},
    )

# ---------------------------------------------------------------------------
def main():
    print(f"DB: {DB}")
    con = sqlite3.connect(DB)
    FIELDS = ["instrument","period","mst_mult","cst_mult","cost_bps",
              "cagr","sharpe","maxdd","calmar","switches_yr",
              "bull_pct","bear_pct","neutral_pct","neutral_drift_bps",
              "bh_cagr","bh_maxdd","bh_calmar","by_year"]
    out_csv = os.path.join(RESDIR, "g1_ranking.csv")
    with open(out_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    best = {}
    for inst in INSTRUMENTS:
        bars5 = load_5min(con, inst)
        if not bars5:
            print(f"  {inst}: NO 5-min data, skipping"); continue
        bars = resample_30min(bars5)
        print(f"\n=== {inst} ===  {len(bars5)} 5-min -> {len(bars)} 30-min bars  "
              f"[{bars[0][0].date()} .. {bars[-1][0].date()}]")
        rows = []
        for period in PERIODS:
            for mm in MST_MULTS:
                for cm in CST_MULTS:
                    if cm >= mm:
                        continue
                    base = evaluate(bars, period, mm, cm, 5)   # headline at 5bps
                    row = dict(instrument=inst, period=period, mst_mult=mm,
                               cst_mult=cm, cost_bps=5, **base)
                    rows.append(row)
        rows.sort(key=lambda r: r["calmar"], reverse=True)
        # write all
        with open(out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            for r in rows:
                w.writerow({k: r.get(k) for k in FIELDS})
        bh = rows[0]
        print(f"  Buy&Hold {inst}: CAGR {bh['bh_cagr']:.1f}%  MaxDD {bh['bh_maxdd']:.1f}%  "
              f"Calmar {bh['bh_calmar']:.2f}")
        print(f"  {'rank':>4} {'per/mm/cm':>12} {'CAGR%':>7} {'MaxDD%':>7} "
              f"{'Calmar':>7} {'Shrp':>5} {'sw/yr':>6} {'bull%':>6} {'bear%':>6} "
              f"{'neu%':>5} {'neuDrift':>8}")
        for k, r in enumerate(rows[:8]):
            print(f"  {k+1:>4} {r['period']:>2}/{r['mst_mult']:.0f}/{r['cst_mult']:.1f}   "
                  f"{r['cagr']:>7.1f} {r['maxdd']:>7.1f} {r['calmar']:>7.2f} "
                  f"{r['sharpe']:>5.2f} {r['switches_yr']:>6.0f} {r['bull_pct']:>6.1f} "
                  f"{r['bear_pct']:>6.1f} {r['neutral_pct']:>5.1f} "
                  f"{r['neutral_drift_bps']:>8.2f}")
        # cost sensitivity on the top config
        top = rows[0]
        print(f"  -- cost sensitivity on top ({top['period']}/{top['mst_mult']:.0f}/"
              f"{top['cst_mult']:.1f}):")
        for cb in COSTS_BPS:
            e = evaluate(bars, top['period'], top['mst_mult'], top['cst_mult'], cb)
            print(f"       {cb:>3} bps/switch -> CAGR {e['cagr']:6.1f}%  "
                  f"Calmar {e['calmar']:5.2f}  MaxDD {e['maxdd']:6.1f}%")
        print(f"  -- top per-year net (5bps): {top['by_year']}")
        best[inst] = rows[0]
    con.close()
    print(f"\nWrote {out_csv}")
    print("\nVERDICT INPUTS: compare each top Calmar vs its Buy&Hold Calmar; "
          "check monotonicity across the table, neutral_drift (should be ~0 or "
          "negative to justify going flat), and per-year stability.")

if __name__ == "__main__":
    main()
