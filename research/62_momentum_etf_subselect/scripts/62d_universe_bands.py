"""Research 62 — PHASE 2a: universe-band capacity study.

Hold the G2 WINNER mechanics fixed (rsblend score, N8, buffer-22, gate-100,
Donchian-15, monthly, daily-marked) and vary ONLY the universe band. Layer a
realistic square-root market-impact model on EVERY buy AND sell, at a chosen AUM,
so we see whether lower-cap momentum survives the cost of actually trading it.

Bands (by PIT trailing-6mo traded-value rank): top200(0-200, baseline), top500(0-500),
mid(100-250), small(250-500), micro(500-1000 with ADV>=Rs5cr/day floor).
AUM sweep: gross(no impact) / Rs1cr / Rs10cr / Rs50cr.

Impact:  one_way = FIXED + COEFF * daily_vol_stock * sqrt(participation),
         participation = traded_Rs / median_daily_traded_value(stock).
Reuses research/41 02_rs_sweep (load, rs_scores, month_ends) + research/62 mom30_score.
"""
from __future__ import annotations
import importlib.util, time, csv, logging
from pathlib import Path
import numpy as np
import pandas as pd
logging.disable(logging.WARNING)

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
ENG = HERE / "scripts" / "62_mom30_subselect.py"
_s = importlib.util.spec_from_file_location("m62", str(ENG))
m62 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m62)
rs2 = m62.rs2

BENCH = rs2.NIFTY_SYM
DAY_CASH = (1 + rs2.CASH_ANNUAL) ** (1 / 252)
START = pd.Timestamp("2014-01-01")
EXCLUDE = {BENCH, "NIFTY50", "BANKNIFTY", "INDIAVIX", "NIFTYJR", "NIFTYBEES",
           "NIFTYMID", "NIFTYIT", "FINNIFTY", "MIDCPNIFTY"}
LIQ_TD = int(126 * 1.6)          # ~200-day window for ADV/eligibility

# bands: (rank_lo, rank_hi, min_adv_cr)
BANDS = {
    "top200": (0, 200, 0.0),
    "top500": (0, 500, 0.0),
    "mid":    (100, 250, 0.0),
    "small":  (250, 500, 0.0),
    "micro":  (500, 1000, 5.0),   # rank 500-1000 but only ADV >= Rs5cr/day
}
# impact model
FIXED_OW = 0.0008     # 8 bps one-way base (brokerage+STT+stamp+min spread)
COEFF = 1.0           # square-root impact coefficient (~Almgren)


def band_universe(tv, close, d, lo, hi, min_adv_cr):
    """Return Series {symbol -> median daily traded value (Rs)} for the band."""
    w = tv.loc[:d].tail(LIQ_TD)
    cnt = w.notna().sum()
    medtv = w.median()
    elig = medtv[(cnt >= 75) & (medtv > 0)].sort_values(ascending=False)
    elig = elig[[s for s in elig.index if s not in EXCLUDE]]
    ranked = elig.iloc[lo:hi]
    if min_adv_cr > 0:
        ranked = ranked[ranked >= min_adv_cr * 1e7]
    return ranked


def eligible_band(close, tv, d, band_def, score_fn, etf_size):
    adv = band_universe(tv, close, d, *band_def)
    if adv is None or len(adv) < 8:
        return None, None
    uni = set(adv.index)
    if score_fn == "mom30":
        sc = m62.mom30_score(close, d, uni)
    else:  # rsblend
        sc = rs2.rs_scores(close, d, None)
        if sc is not None:
            sc = sc[[s for s in sc.index if s in uni]]
    if sc is None or len(sc) == 0:
        return None, None
    sc = sc[[s for s in sc.index if s not in EXCLUDE]].sort_values(ascending=False)
    return list(sc.index[:etf_size]), adv


def run(close, tv, band="top200", score_fn="rsblend", N=8, buffer=22, gate=100,
        donchian=15, etf_size=30, stcg=0.0, aum=None, base_rt=0.004):
    """aum=None → GROSS (only base_rt round-trip, no impact). aum=Rs → net of
    square-root impact at that book size, applied to every buy & sell."""
    band_def = BANDS[band]
    idx = close.index[close.index >= START]
    me = set(rs2.month_ends(close.index))
    iso = idx.isocalendar()
    wk_last = set(pd.Series(idx, index=idx).groupby(
        [iso.year.values, iso.week.values]).last().values)
    cash, held = 1.0, {}
    derisked = False
    nav, prev = [], None
    st = dict(tax=0.0, cost=0.0, slip=0.0, fills=0, donch_exits=0, gate_derisk=0,
              turn_sum=0.0, part_sum=0.0, part_n=0, maxpart=0.0)
    roll_low = (close.rolling(donchian, min_periods=donchian).min().shift(1)
                if donchian else None)

    def E():
        return sum(v[0] for v in held.values())

    def vol_of(s, d):
        cs = close[s].loc[:d].tail(120)
        r = cs.pct_change().dropna()
        return float(r.std()) if len(r) > 20 else 0.03

    def adv_of(s, d, adv_ser):
        if adv_ser is not None and s in adv_ser.index:
            return float(adv_ser[s])
        v = tv[s].loc[:d].tail(LIQ_TD)
        m = v.median()
        return float(m) if pd.notna(m) and m > 0 else np.nan

    def impact(s, traded_nav, port_nav, d, adv_ser):
        """Return slippage as a NAV-fraction cost for trading `traded_nav` of s.
        Capacity abstraction = FIXED deployment size `aum`: participation uses the
        FRACTION of the book traded × aum (constant ₹ scale), so the result is
        'impact at a ₹X book', independent of how the backtest compounds."""
        if aum is None or traded_nav <= 0 or port_nav <= 0:
            return 0.0
        adv = adv_of(s, d, adv_ser)
        if not (adv and adv > 0):
            return traded_nav * 0.02            # unknown ADV → 2% penalty
        traded_rs = (traded_nav / port_nav) * aum   # fraction-of-book × fixed AUM
        part = traded_rs / adv
        ow = FIXED_OW + COEFF * vol_of(s, d) * np.sqrt(part)
        ow = min(ow, 0.50)                            # cap one-way slippage at 50%
        st['part_sum'] += part; st['part_n'] += 1
        st['maxpart'] = max(st['maxpart'], part)
        return traded_nav * ow

    def realize(stt, d):
        g = stt[0] - stt[1]
        if stcg and g > 0 and (d - stt[2]).days < 365:
            t = g * stcg; st['tax'] += t; return t
        return 0.0

    def sell(s, d, adv_ser=None):
        nonlocal cash
        val = held[s][0]
        t = realize(held[s], d)
        sl = impact(s, val, E() + cash, d, adv_ser)
        st['slip'] += sl
        cash += held.pop(s)[0] - t - sl

    def do_fill(d, px):
        nonlocal cash, held, derisked
        etf, adv_ser = eligible_band(close, tv, d, band_def, score_fn, etf_size)
        if not etf:
            return
        top = etf[:N]
        buf = set(etf[:buffer])
        for s in list(held):
            if s not in buf:
                sell(s, d, adv_ser)
        keep = [s for s in held if s in buf]
        add = [s for s in top if s not in keep]
        target = (keep + add)[:N]
        target = [s for s in target if pd.notna(px.get(s, np.nan))]
        if not target:
            return
        tot = E() + cash
        w = tot / len(target)
        port = tot
        nh = {}
        sl_total = 0.0
        for s in target:
            if s in held:
                old = held[s][0]
                traded = abs(w - old)
                sl_total += impact(s, traded, port, d, adv_ser)
                stt = held[s]; stt[0] = w; nh[s] = stt
            else:
                sl_total += impact(s, w, port, d, adv_ser)
                nh[s] = [w, w, d, px[s]]
        turn = len(set(held).symmetric_difference(set(target))) / max(1, 2 * N)
        held = nh
        c = tot * base_rt * turn * 2
        cash = tot - w * len(target) - c - sl_total
        st['cost'] += c; st['slip'] += sl_total
        st['fills'] += 1; st['turn_sum'] += turn
        derisked = False

    for d in idx:
        px = close.loc[d]
        if held and prev is not None:
            for s, stt in held.items():
                p1 = px.get(s, np.nan)
                p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    stt[0] *= p1 / p0
                    stt[3] = max(stt[3], p1)
        if cash > 0:
            cash *= DAY_CASH
        h = close.loc[:d]
        if donchian and held:
            lows = roll_low.loc[d]
            for s in list(held):
                p1 = px.get(s, np.nan); low_n = lows.get(s, np.nan)
                if pd.notna(p1) and pd.notna(low_n) and p1 < low_n:
                    sell(s, d); st['donch_exits'] += 1
        if gate and d in wk_last:
            b = h[BENCH].dropna()
            roff = len(b) >= gate and b.iloc[-1] < b.tail(gate).mean()
            if roff and held:
                for s in list(held):
                    sell(s, d)
                derisked = True; st['gate_derisk'] += 1
            elif not roff:
                derisked = False
        if d in me and not (gate and derisked):
            do_fill(d, px)
        nav.append((d, E() + cash))
        prev = d

    n = pd.DataFrame(nav, columns=["date", "nav"]).set_index("date")["nav"]
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cagr = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1
    dr = n.pct_change().dropna()
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
    dd = ((n - n.cummax()) / n.cummax()).min()
    calmar = cagr / abs(dd) if dd < 0 else np.nan
    avg_part = (st['part_sum'] / st['part_n']) if st['part_n'] else 0.0
    return dict(cagr=cagr * 100, dd=dd * 100, sharpe=sharpe, calmar=calmar,
                slip_pct=st['slip'] * 100, avg_part=avg_part, maxpart=st['maxpart'],
                donch=st['donch_exits'], fills=st['fills'], st=st, nav=n)


FIELDS = ["band", "aum", "cagr", "maxdd", "sharpe", "calmar",
          "slip_drag_pct", "avg_participation", "max_participation",
          "donch_exits", "fills"]


def main():
    t0 = time.time()
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    print(f"  {close.shape[1]} syms {close.index.min().date()}..{close.index.max().date()}",
          flush=True)
    out = RES / "p2_universe_bands.csv"
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(r['band'], r['aum']) for r in csv.DictReader(f)}
    else:
        with open(out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    aums = [("gross", None), ("1cr", 1e7), ("10cr", 1e8), ("50cr", 5e8)]
    rows = []
    for band in ["top200", "top500", "mid", "small", "micro"]:
        for alab, aum in aums:
            if (band, alab) in done:
                continue
            ts = time.time()
            r = run(close, tv, band=band, aum=aum)
            row = dict(band=band, aum=alab, cagr=round(r['cagr'], 1),
                       maxdd=round(r['dd'], 1), sharpe=round(r['sharpe'], 2),
                       calmar=round(r['calmar'], 2) if pd.notna(r['calmar']) else "",
                       slip_drag_pct=round(r['slip_pct'], 1),
                       avg_participation=round(r['avg_part'], 3),
                       max_participation=round(r['maxpart'], 2),
                       donch_exits=r['donch'], fills=r['fills'])
            rows.append(row)
            with open(out, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
            print(f"  {band:7} {alab:6} CAGR={row['cagr']:6.1f}% DD={row['maxdd']:6.1f}% "
                  f"Cal={row['calmar']} slipDrag={row['slip_drag_pct']:5.1f}% "
                  f"avgPart={row['avg_participation']:.3f} maxPart={row['max_participation']:.1f} "
                  f"[{time.time()-ts:.0f}s]", flush=True)
    print(f"\nDONE {time.time()-t0:.0f}s", flush=True)
    if rows:
        df = pd.DataFrame(rows)
        print("\n=== NET Calmar by band × AUM (the verdict grid) ===", flush=True)
        piv = df.pivot(index='band', columns='aum', values='calmar')
        cols = [c for c in ['gross', '1cr', '10cr', '50cr'] if c in piv.columns]
        print(piv[cols].to_string(), flush=True)


if __name__ == "__main__":
    main()
