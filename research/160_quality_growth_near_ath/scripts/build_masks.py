# -*- coding: utf-8 -*-
"""research/160 step 5 - eligibility masks from the point-in-time panel.

One .npz per mask, in the r/158 contract so existing loader code works unchanged:

    z['dates']  <U10 ISO 'YYYY-MM-01', monthly, ascending
    z['cols']   symbol strings - market_data.db spelling, series suffix INTACT
    z['mask']   bool (len(dates), len(cols))

COLUMNS ARE THE WHOLE UNIVERSE, not just the names Screener carries. A name with no
fundamental record is False in every mask and False in has_data, so the engine can tell
"failed the screen" from "could not be screened" - which is the entire point of shipping
has_data alongside. Run every arm twice:

    eligible_strict   = mask                 # missing -> ineligible
    eligible_generous = mask | ~has_data     # missing -> eligible

and report both. The gap between them IS the coverage bias (playbook 5A, binding).

WHY SO MANY MASKS. A single pass rate cannot say which criterion is doing the work. r/158
found ROCE completely inert on its 638 names - dropping it moved the pass rate not at all,
because ROE had already rejected everything ROCE would have. Leave-one-out variants are the
only way to see that, and threshold neighbours are the only way to tell a plateau from a
peak. Arun's numbers are treated as the centre of a neighbourhood, never as revealed truth.

LENDERS pass the ROCE test on ROE alone. Screener publishes no ROCE for banks and NBFCs
because capital employed is not a meaningful denominator for them; failing them would quietly
remove the whole financial sector, which is a sector bet wearing a quality filter's clothes.

THE QUARTERLY OPM MASK IS A RECENT-WINDOW MASK. Screener's quarterly table holds ~13 quarters,
so opm_rising_q can only be evaluated from about mid-2023. Its own row in INDEX.csv says so,
and its pass rate before then is zero by absence, not by rejection.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
RES = ROOT / 'research/160_quality_growth_near_ath/results'
OUT = RES / 'masks'
OUT.mkdir(parents=True, exist_ok=True)

START, END = '2015-01-01', '2026-09-01'


def main():
    uni = pd.read_csv(RES / 'universe.csv')
    cols = np.array(sorted(uni.symbol.unique()))
    ci = {s: i for i, s in enumerate(cols)}
    dates = pd.date_range(START, END, freq='MS')
    di = {str(d.date()): i for i, d in enumerate(dates)}
    dstr = np.array([str(d.date()) for d in dates])
    shape = (len(dates), len(cols))

    p = pd.read_csv(RES / 'features_pit_monthly.csv.gz')
    p = p[p.symbol.isin(ci) & p.date.isin(di)].copy()
    r = p.date.map(di).values
    c = p.symbol.map(ci).values
    print('panel %d rows -> grid %d months x %d symbols' % (len(p), *shape), flush=True)

    def grid(series, fill=np.nan):
        g = np.full(shape, fill, dtype=float)
        g[r, c] = pd.to_numeric(series, errors='coerce').values
        return g

    def bgrid(series):
        g = np.zeros(shape, dtype=bool)
        g[r, c] = series.astype(bool).values
        return g

    n_fy = grid(p.n_fy_usable, 0.0)
    sg = grid(p.sales_g3)
    pg = grid(p.profit_g3)
    roe = grid(p.roe_avg3)
    roce = grid(p.roce_latest)
    de = grid(p.de_latest)
    mcap = grid(p.mcap_pit)
    opm_sl = grid(p.opm_slope3)
    opm_rg = grid(p.opm_range3)
    opm_mn = grid(p.opm_min3)
    opm_qs = grid(p.opm_q_slope8)
    lender = bgrid(p.is_lender)
    neg = bgrid(p.neg3)

    has_data = n_fy >= 4
    ok = lambda a: np.nan_to_num(a, nan=-np.inf)          # NaN never passes a > test  # noqa: E731

    def growth(t):
        return (ok(sg) > t) & (ok(pg) > t)

    def quality(roe_t, roce_t):
        # a lender is judged on ROE alone; ROCE is not applicable, not failed
        return (ok(roe) > roe_t) & (((ok(roce) > roce_t) & ~lender) | lender)

    no_neg = ~neg
    masks, defs = {}, {}

    def add(name, m, d):
        masks[name] = (m & has_data)
        defs[name] = d

    add('has_data', has_data, 'n_fy_usable >= 4 (four filed years = one 3-year growth rate)')
    STRICT = growth(20) & quality(15, 15) & (de <= 0.2) & (ok(mcap) > 1000) & no_neg
    add('arun_strict', STRICT,
        'sales_g3>20 & profit_g3>20 & roe_avg3>15 & roce>15 (lenders: ROE only) '
        '& de<=0.2 & mcap_pit>1000cr & no negatives')

    # ---- leave-one-out: which criterion is actually doing the work? ------------------
    add('no_growth', quality(15, 15) & (de <= 0.2) & (ok(mcap) > 1000) & no_neg,
        'arun_strict without the growth test')
    add('no_roe', growth(20) & (((ok(roce) > 15) & ~lender) | lender) & (de <= 0.2)
        & (ok(mcap) > 1000) & no_neg, 'arun_strict without the ROE test')
    add('no_roce', growth(20) & (ok(roe) > 15) & (de <= 0.2) & (ok(mcap) > 1000) & no_neg,
        'arun_strict without the ROCE test')
    add('no_de', growth(20) & quality(15, 15) & (ok(mcap) > 1000) & no_neg,
        'arun_strict without the debt/equity test')
    add('no_mcap', growth(20) & quality(15, 15) & (de <= 0.2) & no_neg,
        'arun_strict without the market-cap floor')

    # ---- threshold neighbourhoods: plateau or peak? ---------------------------------
    for g in (15, 20, 25, 30):
        for mc in (500, 1000, 2500):
            add('g%d_mc%d' % (g, mc),
                growth(g) & quality(15, 15) & (de <= 0.2) & (ok(mcap) > mc) & no_neg,
                'strict with growth>%d%% and mcap>%dcr' % (g, mc))
    for d_ in (0.2, 0.5, 1.0):
        add('de%s' % str(d_).replace('.', 'p'),
            growth(20) & quality(15, 15) & (de <= d_) & (ok(mcap) > 1000) & no_neg,
            'strict with debt/equity <= %.1f' % d_)
    for q in (12, 15, 20):
        add('q%d' % q, growth(20) & quality(q, q) & (de <= 0.2) & (ok(mcap) > 1000) & no_neg,
            'strict with ROE and ROCE > %d%%' % q)

    # ---- Arun's manual margin step, four readings of "steady or rising" -------------
    add('opm_slope_pos', STRICT & (ok(opm_sl) >= 0),
        'strict AND 3-year annual OPM slope >= 0 (margins not falling)')
    add('opm_steady', STRICT & (np.nan_to_num(opm_rg, nan=np.inf) <= 5),
        'strict AND 3-year annual OPM range <= 5pp (margins stable)')
    add('opm_rising_q', STRICT & (ok(opm_qs) > 0),
        'strict AND 8-quarter OPM slope > 0 - RECENT WINDOW ONLY (~mid-2023 on): '
        'Screener carries ~13 quarters, so this is NaN and therefore False before that')
    add('opm_min', STRICT & (ok(opm_mn) >= 10),
        'strict AND minimum annual OPM over 3 years >= 10pp')

    # ---- the two halves on their own ------------------------------------------------
    add('growth_only', growth(20) & no_neg, 'sales_g3>20 & profit_g3>20 & no negatives')
    add('quality_only', quality(15, 15) & (de <= 0.2),
        'roe_avg3>15 & roce>15 (lenders: ROE only) & de<=0.2')

    # ---- provenance, so a mask built mid-fetch can never be mistaken for the product ----
    import json as _json
    n_cached = len(list((RES / 'screener_cache').glob('*.json')))
    n_tickers = uni.screener_ticker.nunique()
    complete = n_cached >= n_tickers
    _json.dump(dict(built_at=pd.Timestamp.now().isoformat(timespec='seconds'),
                    tickers_expected=int(n_tickers), tickers_cached=int(n_cached),
                    fetch_complete=bool(complete),
                    panel_rows=int(len(p)), panel_symbols=int(p.symbol.nunique()),
                    universe_symbols=int(len(cols))),
               open(OUT / 'PROVENANCE.json', 'w'), indent=1)
    flag = OUT / 'PRELIMINARY_DO_NOT_USE.txt'
    if complete and flag.exists():
        flag.unlink()
    elif not complete:
        flag.write_text(
            'PARTIAL BUILD: %d of %d tickers cached. Every un-fetched symbol is False in\n'
            'has_data and in every mask, so pass RATES against the column count read far too\n'
            'low. Rebuild when PROVENANCE.json says fetch_complete: true.\n'
            % (n_cached, n_tickers))

    yrs = np.array([int(s[:4]) for s in dstr])
    hd_n = has_data.sum(1)
    idx = []
    for name, m in masks.items():
        np.savez_compressed(OUT / ('%s.npz' % name), dates=dstr, cols=cols, mask=m)
        n = m.sum(1)
        row = dict(name=name, definition=defs[name],
                   mean_n_pass=round(float(n.mean()), 1),
                   n_pass_2026=int(n[-1]),
                   pct_of_has_data=round(100.0 * n.sum() / max(hd_n.sum(), 1), 2))
        for y in range(2015, 2027):
            sel = yrs == y
            row['y%d' % y] = round(float(n[sel].mean()), 1) if sel.any() else np.nan
        idx.append(row)
    ix = pd.DataFrame(idx)
    ix.to_csv(OUT / 'INDEX.csv', index=False)

    print()
    print('%-16s %6s %6s  %s' % ('mask', 'mean', 'latest', 'mean names passing, by year'))
    for _, rw in ix.iterrows():
        print('%-16s %6.0f %6d  %s' % (
            rw['name'], rw.mean_n_pass, rw.n_pass_2026,
            ' '.join('%d:%.0f' % (y, rw['y%d' % y]) for y in range(2015, 2027))))
    print()
    print('has_data names per month: min %d, median %d, max %d'
          % (hd_n.min(), int(np.median(hd_n)), hd_n.max()))
    st = masks['arun_strict'].sum(1)
    print('arun_strict names per month: min %d, median %d, max %d  (a four-digit count '
          'would mean a bug, not an edge)' % (st.min(), int(np.median(st)), st.max()))

    # ---- the starvation diagnostic: WHICH leg is doing the rejecting, month by month ----
    # A low pass count is only interesting once you know whether it is the screen or the
    # coverage. Printed for the last 24 months, against has_data rather than the column count.
    show = ['has_data', 'arun_strict', 'no_mcap', 'no_growth', 'no_de', 'no_roe', 'no_roce']
    print()
    print('last 24 months, names passing (denominator = has_data, not the 2,158 columns):')
    print('%-10s %s' % ('month', ' '.join('%11s' % s for s in show)))
    for i in range(max(0, len(dstr) - 24), len(dstr)):
        print('%-10s %s' % (dstr[i], ' '.join('%11d' % masks[s][i].sum() for s in show)))
    hd = has_data[-1].sum()
    print()
    print('at %s, of %d screenable names: strict %d (%.1f%%), without mcap %d, without '
          'growth %d, without D/E %d'
          % (dstr[-1], hd, masks['arun_strict'][-1].sum(),
             100.0 * masks['arun_strict'][-1].sum() / max(hd, 1),
             masks['no_mcap'][-1].sum(), masks['no_growth'][-1].sum(),
             masks['no_de'][-1].sum()))
    prov = _json.load(open(OUT / 'PROVENANCE.json'))
    print('PROVENANCE: %d of %d tickers cached, fetch_complete=%s'
          % (prov['tickers_cached'], prov['tickers_expected'], prov['fetch_complete']))
    print('wrote %d masks + INDEX.csv to %s' % (len(masks), OUT))


if __name__ == '__main__':
    sys.exit(main())
