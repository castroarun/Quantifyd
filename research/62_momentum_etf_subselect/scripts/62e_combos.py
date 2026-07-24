"""Research 62 — PHASE 2b: multi-band combos & partial allocations (capacity-aware).

Hold several bands as independent sleeves (each = its own top-8 momentum sub-book
with buffer-22 + Donchian-15), capital split by allocation weight, one book-wide
gate-100. Same square-root impact model as Phase 2a, at a fixed AUM. Full-rebuild
monthly. Sanity: the 100%-top200 combo must reproduce the single-band top200 result.

Reuses 62d (band_universe, eligible_band, BANDS, EXCLUDE, impact constants).
"""
from __future__ import annotations
import importlib.util, time, csv, logging
from pathlib import Path
import numpy as np
import pandas as pd
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


def run(close, tv, portfolio, aum=None, N=8, buffer=22, gate=100, donch=15,
        score="rsblend", stcg=0.0, base_rt=0.004):
    bands = list(portfolio.items())               # [(band, weight)]
    idx = close.index[close.index >= START]
    me = set(rs2.month_ends(close.index))
    iso = idx.isocalendar()
    wk_last = set(pd.Series(idx, index=idx).groupby(
        [iso.year.values, iso.week.values]).last().values)
    cash = 1.0
    held = {bn: {} for bn, _ in bands}            # held[band][sym]=[val,cost,buy,peak]
    roll_low = close.rolling(donch, min_periods=donch).min().shift(1) if donch else None
    st = dict(slip=0.0, cost=0.0, tax=0.0, fills=0, donch=0, gate=0,
              part_sum=0.0, part_n=0, maxpart=0.0)

    def E():
        return sum(v[0] for bn in held for v in held[bn].values())

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
        st['part_sum'] += part; st['part_n'] += 1
        st['maxpart'] = max(st['maxpart'], part)
        return traded_nav * ow

    def realize(stt, d):
        g = stt[0] - stt[1]
        if stcg and g > 0 and (d - stt[2]).days < 365:
            t = g * stcg; st['tax'] += t; return t
        return 0.0

    def sell(bn, s, d, adv_ser=None):
        nonlocal cash
        val = held[bn][s][0]
        t = realize(held[bn][s], d)
        sl = impact(s, val, E() + cash, d, adv_ser)
        st['slip'] += sl
        cash += held[bn].pop(s)[0] - t - sl

    def do_fill(d, px):
        """Full-rebuild to target weights. Self-contained cash accounting:
        tot already holds ALL value (sold names' value returns to cash implicitly),
        so cash = tot − redeployed − base_cost − fill_slip − fill_tax. No sell() here."""
        nonlocal cash
        tot = E() + cash
        targets = {}                               # (band,sym)->val
        advs = {}
        for bn, wgt in bands:
            etf, adv = b.eligible_band(close, tv, d, b.BANDS[bn], score, 30)
            advs[bn] = adv
            if not etf:
                continue
            top = etf[:N]; buf = set(etf[:buffer]); h = held[bn]
            keep = [s for s in h if s in buf]
            names = (keep + [s for s in top if s not in keep])[:N]
            names = [s for s in names if pd.notna(px.get(s, np.nan))]
            if not names:
                continue
            w = wgt * tot / len(names)
            for s in names:
                targets[(bn, s)] = w
        fill_slip = 0.0; fill_tax = 0.0; traded = 0.0
        for bn in held:                            # current holds: sells & rebalances
            for s in list(held[bn]):
                cur = held[bn][s][0]; tgt = targets.get((bn, s), 0.0)
                if tgt == 0.0:                     # full sell
                    fill_tax += realize(held[bn][s], d)
                    fill_slip += impact(s, cur, tot, d, advs.get(bn)); traded += cur
                else:                              # rebalance toward tgt
                    fill_slip += impact(s, abs(tgt - cur), tot, d, advs.get(bn))
                    traded += abs(tgt - cur)
        for (bn, s), tgt in targets.items():       # new buys
            if s not in held[bn]:
                fill_slip += impact(s, tgt, tot, d, advs.get(bn)); traded += tgt
        c = traded * base_rt
        nh = {bn: {} for bn, _ in bands}
        for (bn, s), tgt in targets.items():
            if s in held[bn]:
                stt = held[bn][s]; stt[0] = tgt; nh[bn][s] = stt
            else:
                nh[bn][s] = [tgt, tgt, d, px[s]]
        held.clear(); held.update(nh)
        cash = tot - sum(targets.values()) - c - fill_slip - fill_tax
        st['cost'] += c; st['slip'] += fill_slip; st['fills'] += 1

    nav, prev = [], None
    for d in idx:
        px = close.loc[d]
        if prev is not None:
            for bn in held:
                for s, stt in held[bn].items():
                    p1 = px.get(s, np.nan); p0 = close.at[prev, s] if s in close.columns else np.nan
                    if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                        stt[0] *= p1 / p0; stt[3] = max(stt[3], p1)
        if cash > 0:
            cash *= DAY_CASH
        h = close.loc[:d]
        if donch:
            lows = roll_low.loc[d]
            for bn in list(held):
                for s in list(held[bn]):
                    p1 = px.get(s, np.nan); ln = lows.get(s, np.nan)
                    if pd.notna(p1) and pd.notna(ln) and p1 < ln:
                        sell(bn, s, d); st['donch'] += 1
        derisk = False
        if gate and d in wk_last:
            bb = h[BENCH].dropna()
            if len(bb) >= gate and bb.iloc[-1] < bb.tail(gate).mean():
                for bn in list(held):
                    for s in list(held[bn]):
                        sell(bn, s, d)
                derisk = True; st['gate'] += 1
        if d in me and not derisk:
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
    avg_part = st['part_sum'] / st['part_n'] if st['part_n'] else 0.0
    return dict(cagr=cagr * 100, dd=dd * 100, sharpe=sharpe, calmar=calmar,
                avg_part=avg_part, maxpart=st['maxpart'], donch=st['donch'])


# Microcap dropped (Phase 2a: un-bankable). Combos within Nifty-500 only.
PORTFOLIOS = [
    ("100_top200",             {"top200": 1.0}),
    ("100_top500",             {"top500": 1.0}),
    ("80top200_20mid",         {"top200": 0.8, "mid": 0.2}),
    ("60top200_40mid",         {"top200": 0.6, "mid": 0.4}),
    ("50top200_50mid",         {"top200": 0.5, "mid": 0.5}),
    ("80top200_20small",       {"top200": 0.8, "small": 0.2}),
    ("60top200_40small",       {"top200": 0.6, "small": 0.4}),
    ("70top200_15mid_15small", {"top200": 0.7, "mid": 0.15, "small": 0.15}),
    ("60top200_20mid_20small", {"top200": 0.6, "mid": 0.2, "small": 0.2}),
    ("33each_top200_mid_small", {"top200": 1 / 3, "mid": 1 / 3, "small": 1 / 3}),
    ("50mid_50small",          {"mid": 0.5, "small": 0.5}),
]
AUMS = [("gross", None), ("1cr", 1e7), ("10cr", 1e8)]
FIELDS = ["portfolio", "aum", "cagr", "maxdd", "sharpe", "calmar",
          "avg_participation", "max_participation", "donch_exits"]


def main():
    t0 = time.time()
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    print(f"  {close.shape[1]} syms", flush=True)
    out = RES / "p2b_combos.csv"
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(r['portfolio'], r['aum']) for r in csv.DictReader(f)}
    else:
        with open(out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    rows = []
    for name, pf in PORTFOLIOS:
        for alab, aum in AUMS:
            if (name, alab) in done:
                continue
            ts = time.time()
            r = run(close, tv, pf, aum=aum)
            row = dict(portfolio=name, aum=alab, cagr=round(r['cagr'], 1),
                       maxdd=round(r['dd'], 1), sharpe=round(r['sharpe'], 2),
                       calmar=round(r['calmar'], 2) if pd.notna(r['calmar']) else "",
                       avg_participation=round(r['avg_part'], 3),
                       max_participation=round(r['maxpart'], 1), donch_exits=r['donch'])
            rows.append(row)
            with open(out, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
            print(f"  {name:30} {alab:5} CAGR={row['cagr']:6.1f}% DD={row['maxdd']:6.1f}% "
                  f"Cal={row['calmar']} maxPart={row['max_participation']} [{time.time()-ts:.0f}s]",
                  flush=True)
    print(f"\nDONE {time.time()-t0:.0f}s", flush=True)
    if rows:
        df = pd.DataFrame(rows)
        df['calmar'] = pd.to_numeric(df['calmar'], errors='coerce')
        print("\n=== NET Calmar by portfolio × AUM ===", flush=True)
        print(df.pivot(index='portfolio', columns='aum', values='calmar')
              .reindex(columns=['gross', '1cr', '10cr']).to_string(), flush=True)


if __name__ == "__main__":
    main()
