"""Research 62 — PHASE 2c: runner-capture analysis (did we CATCH the runners?).

Within the tradeable Nifty-500 universe, find the biggest 'runners' (stocks that
ran >=200% in a 12-month window while liquid/tradeable), then measure — for a given
strategy band — whether the strategy actually HELD each runner and what fraction of
its run it banked. Answers: are we missing catchable runners (selection problem) or
are they simply outside our universe (coverage problem)?

held-recording: at each day's end we log the held set; a runner's 'captured' return =
compound of daily returns on days it was held within its run window; capture ratio =
captured / run_return. Run for band=top200 (our actual universe) and top500 (would a
wider net catch more, ignoring capacity?).

Reuses 62d (band_universe, eligible_band, BANDS, EXCLUDE).
"""
from __future__ import annotations
import importlib.util, time
from pathlib import Path
import numpy as np
import pandas as pd
import logging
logging.disable(logging.WARNING)

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
_d = importlib.util.spec_from_file_location(
    "m62d", str(HERE / "scripts" / "62d_universe_bands.py"))
b = importlib.util.module_from_spec(_d); _d.loader.exec_module(b)
rs2 = b.rs2
BENCH = rs2.NIFTY_SYM
DAY_CASH = b.DAY_CASH
START = b.START


def run_capture(close, tv, band, N=8, buffer=22, gate=100, donch=15, score="rsblend"):
    """Run the strategy on `band`; return held_on = {stock: set(dates held at close)}."""
    band_def = b.BANDS[band]
    idx = close.index[close.index >= START]
    me = set(rs2.month_ends(close.index))
    iso = idx.isocalendar()
    wk_last = set(pd.Series(idx, index=idx).groupby(
        [iso.year.values, iso.week.values]).last().values)
    cash, held = 1.0, {}
    roll_low = close.rolling(donch, min_periods=donch).min().shift(1)
    held_on = {}
    prev = None
    for d in idx:
        px = close.loc[d]
        if held and prev is not None:
            for s, stt in held.items():
                p1 = px.get(s, np.nan); p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    stt[0] *= p1 / p0; stt[3] = max(stt[3], p1)
        if cash > 0:
            cash *= DAY_CASH
        h = close.loc[:d]
        if held:
            lows = roll_low.loc[d]
            for s in list(held):
                p1 = px.get(s, np.nan); ln = lows.get(s, np.nan)
                if pd.notna(p1) and pd.notna(ln) and p1 < ln:
                    cash += held.pop(s)[0]
        derisk = False
        if gate and d in wk_last:
            bb = h[BENCH].dropna()
            if len(bb) >= gate and bb.iloc[-1] < bb.tail(gate).mean():
                for s in list(held):
                    cash += held.pop(s)[0]
                derisk = True
        if d in me and not derisk:
            etf, adv = b.eligible_band(close, tv, d, band_def, score, 30)
            if etf:
                top = etf[:N]; buf = set(etf[:buffer])
                for s in list(held):
                    if s not in buf:
                        cash += held.pop(s)[0]
                keep = [s for s in held if s in buf]
                names = (keep + [s for s in top if s not in keep])[:N]
                names = [s for s in names if pd.notna(px.get(s, np.nan))]
                if names:
                    tot = sum(v[0] for v in held.values()) + cash
                    w = tot / len(names)
                    nh = {}
                    for s in names:
                        nh[s] = held[s] if s in held else [w, w, d, px[s]]
                        nh[s][0] = w
                    held = nh; cash = tot - w * len(names)
        for s in held:                              # record holdings at close of d
            held_on.setdefault(s, set()).add(d)
        prev = d
    return held_on


def monthly_sets(close, tv):
    """Precompute top200 & top500 membership per month-end (for sustained-liquidity test)."""
    s200, s500 = {}, {}
    for d in rs2.month_ends(close.index):
        if d < START:
            continue
        s500[d] = set(b.band_universe(tv, close, d, 0, 500, 0).index)
        s200[d] = set(b.band_universe(tv, close, d, 0, 200, 0).index)
    return s200, s500


def months_in(sets, s, t0, t1):
    """How many month-ends in [t0,t1] had stock s in the given membership sets."""
    return sum(1 for d, mem in sets.items() if t0 <= d <= t1 and s in mem)


def find_runners(close, tv, s200, s500, min_run=2.0, topk=30, min_months=6):
    """Best forward-252d run per stock, kept only if SUSTAINED-liquid: in top500 for
    >= min_months of the ~12 month-ends in its run window (filters transient pump spikes)."""
    cols = [s for s in close.columns if s not in b.EXCLUDE]
    fwd = close[cols].shift(-252) / close[cols] - 1.0
    rows = []
    for s in cols:
        col = fwd[s].dropna(); col = col[col.index >= START]
        if len(col) == 0:
            continue
        R = col.max()
        if R < min_run:
            continue
        t0 = col.idxmax(); t1 = t0 + pd.Timedelta(days=370)
        m500 = months_in(s500, s, t0, t1)
        if m500 < min_months:                  # not sustainedly liquid → skip (pump)
            continue
        m200 = months_in(s200, s, t0, t1)
        rows.append((s, R, t0, t1, m200, m500))
    rows.sort(key=lambda x: -x[1])
    return rows[:topk]


def capture_for(close, runners, held_on):
    """For each runner, compute captured fraction of its run window."""
    out = []
    for s, R, t0, t1, m200, m500 in runners:
        win = close[s].loc[t0:t1].dropna()
        if len(win) < 30:
            continue
        rets = win.pct_change().fillna(0.0)
        held_days = held_on.get(s, set())
        cap = 1.0; ndh = 0; prevd = None
        for d, r in rets.items():
            if prevd is not None and prevd in held_days:
                cap *= (1 + r); ndh += 1
            prevd = d
        captured = cap - 1.0
        avail = win.iloc[-1] / win.iloc[0] - 1.0
        ratio = captured / avail if avail > 0 else 0.0
        out.append(dict(stock=s, run_pct=round(R * 100),
                        in_top200=(m200 >= 6), mo_in200=m200, mo_in500=m500,
                        held=ndh > 0, captured_pct=round(captured * 100),
                        capture_ratio=round(ratio, 2), run_start=str(t0.date())))
    return out


def main():
    t0 = time.time()
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    print(f"  {close.shape[1]} syms", flush=True)

    print("Precomputing monthly top200/top500 membership ...", flush=True)
    s200, s500 = monthly_sets(close, tv)
    runners = find_runners(close, tv, s200, s500, min_run=2.0, topk=30, min_months=6)
    print(f"  SUSTAINED-liquid runners (>=200% 1y, in top500 >=6mo of run): {len(runners)}",
          flush=True)

    for band in ["top200", "top500"]:
        print(f"\n=== CAPTURE for band={band} ===", flush=True)
        held_on = run_capture(close, tv, band)
        df = pd.DataFrame(capture_for(close, runners, held_on))
        df.to_csv(RES / f"p2c_capture_{band}.csv", index=False)
        tot = len(df)
        held_n = int(df['held'].sum())
        avg_cap = df.loc[df['held'], 'capture_ratio'].mean() if held_n else 0.0
        # split by whether the runner was sustainedly in OUR universe (top200)
        in200 = df[df['in_top200']]; out200 = df[~df['in_top200']]
        h200 = int(in200['held'].sum()); h_out = int(out200['held'].sum())
        cap200 = in200.loc[in200['held'], 'capture_ratio'].mean() if h200 else 0.0
        print(df[['stock', 'run_pct', 'in_top200', 'mo_in200', 'held',
                  'captured_pct', 'capture_ratio', 'run_start']].to_string(index=False),
              flush=True)
        print(f"\n  SUMMARY [{band}]: held {held_n}/{tot} sustained-liquid runners; "
              f"avg capture (when held) = {avg_cap:.2f}", flush=True)
        print(f"    of {len(in200)} runners IN top200 (our universe): held {h200}, "
              f"avg capture {cap200:.2f}", flush=True)
        print(f"    of {len(out200)} runners only in 200-500 (outside our universe): "
              f"held {h_out}", flush=True)
    print(f"\nDONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
