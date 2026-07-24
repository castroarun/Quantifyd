"""Research 62 — PHASE 2d: (1) FAIR-GATE test + (2) DIVERSIFIED midcap sleeve.

(1) Fair gate: the mid/small momentum books were gated by the large-cap NIFTYBEES
    100-DMA (mis-matched). Re-gate them with a BAND-MATCHED trend signal = a synthetic
    equal-weight index of the band's own members vs its 100-DMA. Does the deep drawdown
    shrink? (Is mid/small's risk real, or an artifact of my mismatched gate?)
(2) Diversified sleeve: hold N=8/15/20/30 names in the mid band (not just 8), fair-gated,
    with the capacity model — does diversifying trade concentration-alpha for TRADEABILITY
    (lower participation) while keeping return? ≈ what Nifty Midcap150 Momentum 50 is.

Reuses 62d (band_universe, eligible_band, BANDS, EXCLUDE, impact constants).
"""
from __future__ import annotations
import importlib.util, time, csv
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


def synth_index(close, tv, lo, hi):
    """Causal equal-weight band index: each day = mean daily return across the band's
    members (membership from the prior month-end). Cumulated to a level series."""
    mes = [d for d in rs2.month_ends(close.index) if d >= pd.Timestamp("2013-06-01")]
    memb = {d: set(b.band_universe(tv, close, d, lo, hi, 0).index) for d in mes}
    idx = close.index[close.index >= pd.Timestamp("2013-07-01")]
    lvl = 1.0; out = {}
    cur = set(); mi = 0
    prev = None
    mes_sorted = sorted(memb)
    for d in idx:
        while mi < len(mes_sorted) and mes_sorted[mi] <= d:
            cur = memb[mes_sorted[mi]]; mi += 1
        if prev is not None and cur:
            r = []
            for s in cur:
                p1 = close.at[d, s] if s in close.columns else np.nan
                p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    r.append(p1 / p0 - 1)
            if r:
                lvl *= (1 + np.mean(r))
        out[d] = lvl
        prev = d
    return pd.Series(out)


def run(close, tv, band, gate_level, N=8, buffer=22, donch=15, aum=None,
        score="rsblend", stcg=0.0, base_rt=0.004, etf_size=70):
    band_def = b.BANDS[band]
    idx = close.index[close.index >= START]
    me = set(rs2.month_ends(close.index))
    iso = idx.isocalendar()
    wk_last = set(pd.Series(idx, index=idx).groupby(
        [iso.year.values, iso.week.values]).last().values)
    cash, held = 1.0, {}
    derisked = False
    roll_low = close.rolling(donch, min_periods=donch).min().shift(1)
    st = dict(slip=0.0, part_sum=0.0, part_n=0, maxpart=0.0, donch=0, gate=0)

    def E():
        return sum(v[0] for v in held.values())

    def vol_of(s, d):
        r = close[s].loc[:d].tail(120).pct_change().dropna()
        return float(r.std()) if len(r) > 20 else 0.03

    def adv_of(s, d, adv_ser):
        if adv_ser is not None and s in adv_ser.index:
            return float(adv_ser[s])
        m = tv[s].loc[:d].tail(b.LIQ_TD).median()
        return float(m) if pd.notna(m) and m > 0 else np.nan

    def impact(s, traded_nav, port_nav, d, adv_ser=None):
        if aum is None or traded_nav <= 0 or port_nav <= 0:
            return 0.0
        adv = adv_of(s, d, adv_ser)
        if not (adv and adv > 0):
            return traded_nav * 0.02
        part = (traded_nav / port_nav) * aum / adv
        ow = min(b.FIXED_OW + b.COEFF * vol_of(s, d) * np.sqrt(part), 0.50)
        st['part_sum'] += part; st['part_n'] += 1; st['maxpart'] = max(st['maxpart'], part)
        return traded_nav * ow

    def realize(stt, d):
        g = stt[0] - stt[1]
        if stcg and g > 0 and (d - stt[2]).days < 365:
            return g * stcg
        return 0.0

    def sell(s, d, adv_ser=None):
        nonlocal cash
        sl = impact(s, held[s][0], E() + cash, d, adv_ser); st['slip'] += sl
        cash += held.pop(s)[0] - realize(held.get(s, [0, 0, d, 0]), d) - sl

    def do_fill(d, px):
        nonlocal cash, held, derisked
        etf, adv = b.eligible_band(close, tv, d, band_def, score, etf_size)
        if not etf:
            return
        top = etf[:N]; buf = set(etf[:buffer])
        for s in list(held):
            if s not in buf:
                sl = impact(s, held[s][0], E() + cash, d, adv); st['slip'] += sl
                cash += held.pop(s)[0] - sl
        keep = [s for s in held if s in buf]
        names = (keep + [s for s in top if s not in keep])[:N]
        names = [s for s in names if pd.notna(px.get(s, np.nan))]
        if not names:
            return
        tot = E() + cash; w = tot / len(names); fill_slip = 0.0
        nh = {}
        for s in names:
            if s in held:
                fill_slip += impact(s, abs(w - held[s][0]), tot, d, adv)
                stt = held[s]; stt[0] = w; nh[s] = stt
            else:
                fill_slip += impact(s, w, tot, d, adv); nh[s] = [w, w, d, px[s]]
        traded = sum(abs(w - held.get(s, [0])[0]) for s in names)
        held = nh; st['slip'] += fill_slip
        cash = tot - w * len(names) - traded * base_rt - fill_slip
        derisked = False

    nav, prev = [], None
    for d in idx:
        px = close.loc[d]
        if held and prev is not None:
            for s, stt in held.items():
                p1 = px.get(s, np.nan); p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    stt[0] *= p1 / p0; stt[3] = max(stt[3], p1)
        if cash > 0:
            cash *= DAY_CASH
        if held:
            lows = roll_low.loc[d]
            for s in list(held):
                p1 = px.get(s, np.nan); ln = lows.get(s, np.nan)
                if pd.notna(p1) and pd.notna(ln) and p1 < ln:
                    sell(s, d); st['donch'] += 1
        if gate_level is not None and d in wk_last:
            lv = gate_level.loc[:d].dropna()
            roff = len(lv) >= 100 and lv.iloc[-1] < lv.tail(100).mean()
            if roff and held:
                for s in list(held):
                    sell(s, d)
                derisked = True; st['gate'] += 1
            elif not roff:
                derisked = False
        if d in me and not (gate_level is not None and derisked):
            do_fill(d, px)
        nav.append((d, E() + cash))
        prev = d

    n = pd.DataFrame(nav, columns=["date", "nav"]).set_index("date")["nav"]
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cagr = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1 if n.iloc[-1] > 0 else -1
    dr = n.pct_change().dropna()
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
    dd = ((n - n.cummax()) / n.cummax()).min()
    calmar = cagr / abs(dd) if dd < 0 else np.nan
    ap = st['part_sum'] / st['part_n'] if st['part_n'] else 0.0
    return dict(cagr=cagr * 100, dd=dd * 100, sharpe=sharpe, calmar=calmar,
                avg_part=ap, maxpart=st['maxpart'], donch=st['donch'], gate=st['gate'],
                nav=n)


def main():
    t0 = time.time()
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    print(f"  {close.shape[1]} syms", flush=True)
    print("Building synthetic band trend indices (mid, small) ...", flush=True)
    synth_mid = synth_index(close, tv, 100, 250)
    synth_small = synth_index(close, tv, 250, 500)
    nifty = close[BENCH]
    print(f"  synth_mid {synth_mid.index.min().date()}..{synth_mid.index.max().date()} "
          f"final={synth_mid.iloc[-1]:.1f}x", flush=True)

    out1 = RES / "p2d_fairgate.csv"
    with open(out1, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["band", "gate", "aum", "cagr", "maxdd",
                                      "sharpe", "calmar", "maxpart"]).writeheader()
    print("\n=== TEST 1: FAIR GATE (band-matched) vs NIFTYBEES gate, N=8 ===", flush=True)
    for band, synth in [("mid", synth_mid), ("small", synth_small)]:
        for gname, glvl in [("nifty", nifty), ("fair", synth)]:
            for alab, aum in [("gross", None), ("1cr", 1e7), ("10cr", 1e8)]:
                r = run(close, tv, band, glvl, N=8, buffer=22, aum=aum)
                row = dict(band=band, gate=gname, aum=alab, cagr=round(r['cagr'], 1),
                           maxdd=round(r['dd'], 1), sharpe=round(r['sharpe'], 2),
                           calmar=round(r['calmar'], 2) if pd.notna(r['calmar']) else "",
                           maxpart=round(r['maxpart'], 1))
                with open(out1, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=row.keys()).writerow(row)
                print(f"  {band:6} gate={gname:5} {alab:5} CAGR={row['cagr']:6.1f}% "
                      f"DD={row['maxdd']:6.1f}% Cal={row['calmar']} maxPart={row['maxpart']}",
                      flush=True)

    out2 = RES / "p2d_diversified.csv"
    with open(out2, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["band", "N", "aum", "cagr", "maxdd",
                                      "sharpe", "calmar", "avg_part", "maxpart"]).writeheader()
    print("\n=== TEST 2: DIVERSIFIED mid sleeve (fair-gated), N sweep ===", flush=True)
    for N in [8, 15, 20, 30]:
        for alab, aum in [("gross", None), ("1cr", 1e7), ("10cr", 1e8)]:
            r = run(close, tv, "mid", synth_mid, N=N, buffer=int(N * 1.5), aum=aum)
            row = dict(band="mid", N=N, aum=alab, cagr=round(r['cagr'], 1),
                       maxdd=round(r['dd'], 1), sharpe=round(r['sharpe'], 2),
                       calmar=round(r['calmar'], 2) if pd.notna(r['calmar']) else "",
                       avg_part=round(r['avg_part'], 3), maxpart=round(r['maxpart'], 1))
            with open(out2, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=row.keys()).writerow(row)
            print(f"  N={N:2} {alab:5} CAGR={row['cagr']:6.1f}% DD={row['maxdd']:6.1f}% "
                  f"Cal={row['calmar']} avgPart={row['avg_part']:.3f} maxPart={row['maxpart']}",
                  flush=True)
    print(f"\nDONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
