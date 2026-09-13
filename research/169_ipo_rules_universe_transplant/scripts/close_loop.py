# -*- coding: utf-8 -*-
"""research/169 close-the-loop.

Builds every table from the result files (no hand-copied numbers), writes results/RESULTS.md
from RESULTS_TEMPLATE.md, and inserts the study into the registers. Every register file is read
fresh at run time and every insertion is idempotent (skipped if research/169 is already there).
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path('/home/arun/quantifyd')
H = ROOT / 'research/169_ipo_rules_universe_transplant'
R = H / 'results'
M = '−'
SLUG = 'ipo-rules-universe-transplant-research169'
URL = '/app/backtest/' + SLUG


def n(x, nd=2, sign=False, pct=False):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return '—'
    s = (f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}').replace('-', M)
    return s + ('%' if pct else '')


NAMES = {
    'all__le6': 'IPO Spec A — all stocks, listed ≤ 6 months',
    'all__le6__mb60_livebook': 'IPO live book as coded — ≤ 6 months, ≥ 60 bars',
    'top50__none': 'Nifty-50-like, no age limit', 'top50__gt6': 'Nifty-50-like, seasoned > 6m',
    'top100__none': 'Nifty-100-like, no age limit', 'top100__gt6': 'Nifty-100-like, seasoned > 6m',
    'top200__none': 'Nifty-200-like, no age limit', 'top200__gt6': 'Nifty-200-like, seasoned > 6m',
    'top500__none': 'Nifty-500-like, no age limit', 'top500__gt6': 'Nifty-500-like, seasoned > 6m',
    'mid101_250__none': 'Midcap-like (rank 101-250), no age limit',
    'mid101_250__gt6': 'Midcap-like, seasoned > 6m',
    'small251_500__none': 'Smallcap-like (rank 251-500), no age limit',
    'small251_500__gt6': 'Smallcap-like, seasoned > 6m',
    'rank501plus__none': 'Beyond 500 (rank 501+), no age limit',
    'rank501plus__gt6': 'Beyond 500, seasoned > 6m',
    'all__none': 'All stocks, no age limit', 'all__gt6': 'All stocks, seasoned > 6m',
    'all__le12': 'All stocks, listed ≤ 12 months', 'all__le24': 'All stocks, listed ≤ 24 months',
    'all__gt24': 'All stocks, seasoned > 24 months', 'all__vet_any': 'All vetted listings, any age',
}


def nm(label):
    return NAMES.get(label, label)


def passes(r):
    return bool(r['w2_edge'] >= 1.0 and r['w2_wins'] >= 25 and r['wa_wins'] >= 25
                and r['wb_wins'] >= 25)


def md(cols, rows):
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    return '\n'.join(out + ['| ' + ' | '.join(r) + ' |' for r in rows])


s2 = pd.read_csv(R / 's2_transplant.csv').drop_duplicates('label', keep='last')
s3 = pd.read_csv(R / 's3_age.csv').drop_duplicates('label', keep='last')
allc = pd.concat([s2, s3]).set_index('label')
s2b = pd.read_csv(R / 's2b_mcap_universes_2018.csv').drop_duplicates('label', keep='last')
s5 = pd.read_csv(R / 's5_costs.csv').drop_duplicates('label', keep='last').set_index('label')
s1c = pd.read_csv(R / 's1c_null_attribution.csv').drop_duplicates('label', keep='last')
s1b = json.load(open(R / 's1b_minbars_r167engine.json'))
s6 = json.load(open(R / 's6_blend.json'))
py = json.load(open(R / 's6_peryear.json'))
s7 = json.load(open(R / 's7_proxy.json'))
carried = json.load(open(R / 'carried.json'))

# ── T1 transplant
T1c = ['Universe (Spec A rules otherwise identical)', 'CAGR', 'worst seed', 'MaxDD (worst seed)',
       'Calmar', 'null CAGR', 'W2 edge [range]', 'wins W2 / WA / WB', 'WA real / null',
       'WB real / null', 'edge test']
T1r = []
for lab in s2.label:
    r = allc.loc[lab]
    T1r.append([nm(lab), n(r.w2_cagr, pct=True), n(r.w2_cagr_lo), f'{n(r.w2_dd)}% ({n(r.w2_dd_worst)}%)',
                n(r.w2_calmar, 3), n(r.w2_null_cagr, pct=True),
                f'{n(r.w2_edge, sign=True)}pp [{n(r.w2_edge_lo, sign=True)}..{n(r.w2_edge_hi, sign=True)}]',
                f'{int(r.w2_wins)} / {int(r.wa_wins)} / {int(r.wb_wins)}',
                f'{n(r.wa_cagr)} / {n(r.wa_null_cagr)}', f'{n(r.wb_cagr)} / {n(r.wb_null_cagr)}',
                'PASS' if passes(r) else 'FAIL'])

# ── T2 tradeability + capacity
T2c = ['Universe', 'trades / yr', 'median hold (days)', 'invested', 'win rate', 'avg win / avg loss',
       'longest losing streak', 'net expectancy / trade (after 50 bps)', 'ten best trades’ share',
       'median position, % of 20d traded value @ ₹10 L', 'p90 position @ ₹10 L',
       'book size where p90 = 5%']
T2r = []
for lab in s2.label:
    r = allc.loc[lab]
    T2r.append([nm(lab), n(r.w2_tpy, 1), n(r.hold_median, 0), n(r.w2_inv, 1, pct=True),
                n(r.w2_win, 1, pct=True), f'{n(r.w2_avg_win, 1, sign=True)}% / {n(r.w2_avg_loss, 1)}%',
                str(int(r.w2_streak)), n(r.w2_netexp, 2, sign=True, pct=True),
                n(r.top10_share_pct, 1, pct=True), n(r.cap_med_pct_tv_10L, 2, pct=True),
                n(r.cap_p90_pct_tv_10L, 2, pct=True), f'₹{r.book_L_at_p90_5pct:.0f} L'])

# ── T3 market-cap universes 2018+
def mcap_name(lab):
    if lab.startswith('specA'):
        return 'IPO Spec A (same window)'
    kind, rest = lab.split('_', 1)
    u = rest.split('__')[0]
    base = {'top50': 'Nifty-50-like', 'top100': 'Nifty-100-like', 'top200': 'Nifty-200-like',
            'top500': 'Nifty-500-like', 'mid101_250': 'Midcap-like', 'small251_500': 'Smallcap-like'}[u]
    return base + (' — by PIT market cap' if kind == 'mcap' else ' — by traded value')


T3c = ['Universe, no age limit (2018-09 → 2026-09)', 'signals', 'CAGR', 'MaxDD', 'Calmar',
       'null CAGR', 'edge (seeds real wins / 30)']
T3r = [[mcap_name(r.label), f'{int(r.n_signals_w2):,}', n(r.w2_cagr, pct=True), n(r.w2_dd, pct=True),
        n(r.w2_calmar, 3), n(r.w2_null_cagr, pct=True), f'{n(r.w2_edge, sign=True)}pp ({int(r.w2_wins)})']
       for r in s2b.itertuples(index=False)]

# ── T4 cost ladder
T4c = ['Spec', '25 bps a side', '40 bps', '60 bps', 'CAGR lost 25 → 60']
T4r = []
for s in carried:
    lab = s['label']
    if lab not in allc.index:
        continue
    b = allc.loc[lab]
    c40, c60 = s5.loc[lab + '__c40'], s5.loc[lab + '__c60']
    T4r.append([nm(lab), f'{n(b.w2_cagr, pct=True)} / {n(b.w2_dd, pct=True)}',
                f'{n(c40.w2_cagr, pct=True)} / {n(c40.w2_dd, pct=True)}',
                f'{n(c60.w2_cagr, pct=True)} / {n(c60.w2_dd, pct=True)}',
                f'{n(c60.w2_cagr - b.w2_cagr, sign=True)}pp'])

# ── T5 age axis
T5c = ['Age band (all-stock universe)', 'signals', 'CAGR', 'MaxDD', 'Calmar', 'null CAGR',
       'W2 edge (wins)', 'WA edge (wins)', 'WB edge (wins)', 'trades on stocks listed ≤ 6m']
T5r = []
for lab in ('all__le6', 'all__le12', 'all__le24', 'all__vet_any', 'all__none', 'all__gt6',
            'all__gt24'):
    r = allc.loc[lab]
    T5r.append([nm(lab), f'{int(r.n_signals_w2):,}', n(r.w2_cagr, pct=True), n(r.w2_dd, pct=True),
                n(r.w2_calmar, 3), n(r.w2_null_cagr, pct=True),
                f'{n(r.w2_edge, sign=True)}pp ({int(r.w2_wins)})',
                f'{n(r.wa_edge, sign=True)}pp ({int(r.wa_wins)})',
                f'{n(r.wb_edge, sign=True)}pp ({int(r.wb_wins)})', n(r.young6m_share_pct, 1, pct=True)])

# ── T6 blend
mc = s6['correlations']['monthly']
T6c = ['Candidate', 'monthly corr TN / OA·BaseAge / IPO-A',
       'replacing IPO-A at 25%: CAGR / MaxDD / Calmar', 'ΔCAGR (paths better)',
       'ΔCalmar (paths better)', 'vs risk-matched cash (paths better)',
       '4th sleeve at 10%: ΔCAGR / ΔCalmar / vs cash', '4th sleeve at 25%: ΔCAGR / ΔCalmar / vs cash']
T6r = []
for lab, t in s6['tests'].items():
    rp, f10, f25 = t['replace_ipo_at_25'], t['fourth']['10'], t['fourth']['25']
    T6r.append([nm(lab), f"{mc[lab]['TN']:.2f} / {mc[lab]['OA_BaseAge']:.2f} / {mc[lab]['IPO_A']:.2f}",
                f"{n(rp['cagr'], pct=True)} / {n(rp['dd'], pct=True)} / {n(rp['calmar'], 3)}",
                f"{n(rp['d_cagr'], sign=True)}pp ({rp['cagr_wins']}/30)",
                f"{n(rp['d_calmar'], 3, sign=True)} ({rp['calmar_wins']}/30)",
                f"{n(rp['d_cagr_vs_cash'], sign=True)}pp ({rp['wins_vs_cash']}/30)",
                f"{n(f10['d_cagr'], sign=True)}pp / {n(f10['d_calmar'], 3, sign=True)} / {n(f10['d_cagr_vs_cash'], sign=True)}pp ({f10['wins_vs_cash']}/30)",
                f"{n(f25['d_cagr'], sign=True)}pp / {n(f25['d_calmar'], 3, sign=True)} / {n(f25['d_cagr_vs_cash'], sign=True)}pp ({f25['wins_vs_cash']}/30)"])

# ── T7 null attribution
PAN = {'N1_r167like': 'research/167’s panel (union-date windows)',
       'N2_plus_robust': '+ per-symbol NaN-robust windows',
       'N3_plus_phantom_drop': '+ phantom holiday rows dropped',
       'N4_plus_split_adjust': '+ split back-adjustment (this study’s panel)'}
T7c = ['Panel', 'signals', 'W2 real / null', 'W2 edge (wins)', 'WA real / null', 'WA edge (wins)',
       'WB real / null', 'WB edge (wins)']
T7r = [[PAN[r.label], f'{int(r.n_signals_w2):,}', f'{n(r.w2_cagr)} / {n(r.w2_null_cagr)}',
        f'{n(r.w2_edge, sign=True)}pp ({int(r.w2_wins)})', f'{n(r.wa_cagr)} / {n(r.wa_null_cagr)}',
        f'{n(r.wa_edge, sign=True)}pp ({int(r.wa_wins)})', f'{n(r.wb_cagr)} / {n(r.wb_null_cagr)}',
        f'{n(r.wb_edge, sign=True)}pp ({int(r.wb_wins)})'] for r in s1c.itertuples(index=False)]

# ── T8 min_bars
T8c = ['Minimum bars since listing', 'signals', 'CAGR', 'worst seed', 'MaxDD', 'Calmar', 'WA / WB',
       'null CAGR', 'edge (wins / 30)']
T8r = []
for k, lbl in (('25', '25 — the validated Spec A'), ('40', '40'), ('60', '60 — what services/ipo_paper.py runs')):
    v = s1b[k]
    T8r.append([lbl, f"{v['signals']:,}", n(v['w2_cagr'], pct=True), n(v['w2_worst']),
                n(v['w2_dd'], pct=True), n(v['w2_calmar'], 3), f"{n(v['wa'])} / {n(v['wb'])}",
                n(v['null'], pct=True), f"{n(v['edge'], sign=True)}pp ({v['wins']})"])

# ── T9 capital desk
B3, live, clean = s6['B3'], s6['tests']['all__le6__mb60_livebook'], s6['tests']['all__le6']
lr = live['replace_ipo_at_25']
cr = clean['replace_ipo_at_25']
lad = live['cost_ladder']
T9c = ['Third sleeve at 25% (TN 37.5 / OA·BaseAge 37.5)', 'blend CAGR', 'MaxDD', 'Calmar',
       'vs IPO-A at 25%', 'vs cash at equal drawdown', '40 bps / 60 bps CAGR']
T9r = [['IPO-A as validated (min_bars 25) — research/168', n(B3['cagr'], pct=True), n(B3['dd'], pct=True),
        n(B3['calmar'], 3), '—', '+2.52pp, 30/30 (research/168)',
        f"{n(lad['40']['B3'][0], pct=True)} / {n(lad['60']['B3'][0], pct=True)}"],
       ['IPO live book as coded (min_bars 60)', n(lr['cagr'], pct=True), n(lr['dd'], pct=True),
        n(lr['calmar'], 3),
        f"{n(lr['d_cagr'], sign=True)}pp CAGR ({lr['cagr_wins']}/30 better); {n(lr['d_calmar'], 3, sign=True)} Calmar ({lr['calmar_wins']}/30)",
        f"{n(lr['d_cagr_vs_cash'], sign=True)}pp ({lr['wins_vs_cash']}/30)",
        f"{n(lad['40']['replace'][0], pct=True)} / {n(lad['60']['replace'][0], pct=True)}"],
       ['Spec A on this study’s clean panel (min_bars 25), check', n(cr['cagr'], pct=True),
        n(cr['dd'], pct=True), n(cr['calmar'], 3),
        f"{n(cr['d_cagr'], sign=True)}pp ({cr['cagr_wins']}/30); {n(cr['d_calmar'], 3, sign=True)} Calmar ({cr['calmar_wins']}/30)",
        f"{n(cr['d_cagr_vs_cash'], sign=True)}pp ({cr['wins_vs_cash']}/30)", '—'],
       ['two-sleeve TN + OA·BaseAge 50:50 (no IPO)', n(B3['base_cagr'], pct=True),
        n(B3['base_dd'], pct=True), n(B3['base_calmar'], 3), '—', '—', '—']]

# ── T10 proxy
med, lat = s7['median'], s7['latest_vs_official']
T10c = ['Proxy band (traded-value rank)', 'overlap with the PIT market-cap band (median month, Aug-2018+)',
        'overlap with today’s official index', 'official index']
T10r = [['top 50', n(100 * med['top50'], 0, pct=True), n(100 * lat['top50_vs_nifty50'], 0, pct=True), 'Nifty 50'],
        ['top 100', n(100 * med['top100'], 0, pct=True), n(100 * lat['top100_vs_nifty100'], 0, pct=True), 'Nifty 50 + Next 50'],
        ['top 200', n(100 * med['top200'], 0, pct=True), n(100 * lat['top200_vs_nifty200'], 0, pct=True), 'Nifty 200'],
        ['top 500', n(100 * med['top500'], 0, pct=True), n(100 * lat['top500_vs_nifty500'], 0, pct=True), 'Nifty 500 (proxy list)'],
        ['ranks 101-250', n(100 * med['mid101_250'], 0, pct=True), n(100 * lat['mid_vs_midcap150'], 0, pct=True), 'Nifty Midcap 150'],
        ['ranks 251-500', n(100 * med['small251_500'], 0, pct=True), n(100 * lat['small_vs_smallcap250'], 0, pct=True), 'Nifty Smallcap 250']]
proxy_note = (f"Spearman rank correlation, traded value vs PIT market cap, within the top 500 by market cap: "
              f"{med['spearman_top500mcap']:.2f} (median of {s7['months']} months).")

# ── T11 YoY
YC = [('TN', 'TN'), ('OA BaseAge', 'OA · Base Age'), ('IPO-A', 'IPO-A (validated)'),
      ('all__le6__mb60_livebook', 'IPO live (60 bars)'), ('mid101_250__none', 'Midcap-like transplant'),
      ('TN/OA/IPO-A 37.5/37.5/25', 'TN / OA / IPO-A 37.5/37.5/25'),
      ('TN/OA/all__le6__mb60_livebook 37.5/37.5/25', 'TN / OA / IPO live 37.5/37.5/25'),
      ('TN/OA/mid101_250__none 37.5/37.5/25', 'TN / OA / Midcap transplant 37.5/37.5/25'),
      ('NIFTYBEES', 'NIFTYBEES')]


def yoy(html):
    cols = ['Year'] + [b for _, b in YC] + ['BEST CAGR', 'LEAST DD', 'BEST OVERALL']
    rows = []
    pick = [a for a, _ in YC if a != 'NIFTYBEES']
    lab = dict(YC)
    for y in sorted(py['TN']['years'], key=int):
        cells = [y]
        for a, _ in YC:
            v = py[a]['years'].get(y)
            if not v:
                cells.append('')
            elif html:
                cells.append(f'{n(v[0], 1, sign=True)}<br><sub>({n(v[1], 1)})</sub>')
            else:
                cells.append(f'{n(v[0], 1, sign=True)} ({n(v[1], 1)})')
        vals = {a: py[a]['years'][y] for a in pick if y in py[a]['years']}
        cells += [lab[max(vals, key=lambda a: vals[a][0])], lab[max(vals, key=lambda a: vals[a][1])],
                  lab[max(vals, key=lambda a: vals[a][0] + vals[a][1])]]
        rows.append(cells)
    if html:
        rows.append(['**full**'] + [f"**{n(py[a]['cagr'])}**<br><sub>{n(py[a]['dd'], 1)} / {n(py[a]['calmar'], 2)}</sub>" for a, _ in YC] + ['', '', ''])
    else:
        rows.append(['full: CAGR (MaxDD / Calmar)'] + [f"{n(py[a]['cagr'])} ({n(py[a]['dd'], 1)} / {n(py[a]['calmar'], 2)})" for a, _ in YC] + ['', '', ''])
    return cols, rows


# ── RESULTS.md
tpl = (H / 'scripts/RESULTS_TEMPLATE.md').read_text()
yc, yr = yoy(True)
fills = {'{{TRANSPLANT_TABLE}}': md(T1c, T1r), '{{TRADE_TABLE}}': md(T2c, T2r),
         '{{MCAP_TABLE}}': md(T3c, T3r), '{{COST_TABLE}}': md(T4c, T4r), '{{AGE_TABLE}}': md(T5c, T5r),
         '{{BLEND_TABLE}}': md(T6c, T6r), '{{NULL_ATTR_TABLE}}': md(T7c, T7r),
         '{{MINBARS_TABLE}}': md(T8c, T8r) + '\n\nresearch/167 engine and panel, idle cash 5.0%, 30 seeds.',
         '{{DESK_TABLE}}': md(T9c, T9r), '{{PROXY_TABLE}}': md(T10c, T10r) + '\n\n' + proxy_note,
         '{{YOY_TABLE}}': md(yc, yr)}
for k, v in fills.items():
    assert k in tpl, k
    tpl = tpl.replace(k, v)
(R / 'RESULTS.md').write_text(tpl)
print('RESULTS.md written', len(tpl), 'chars')


# ── app entry
def tbl(title, caption, cols, rows, hl=None):
    d = dict(title=title, caption=caption, columns=cols, rows=rows)
    if hl:
        d['highlightRows'] = hl
    return d


yc2, yr2 = yoy(False)
entry = dict(
    slug=SLUG,
    title='Why is IPO Base IPO-specific? Its rules transplanted to Nifty 50, 100, 200, Midcap, 500, Smallcap and all stocks',
    verdict=('NO EDGE OUTSIDE YOUNG LISTINGS — AND THE LIVE IPO BOOK IS NOT RUNNING THE VALIDATED SPEC. '
             'IPO Base’s Spec A rules, with the age band removed or on seasoned names only, fail a date-matched '
             'random-entry control on all 16 size universes (2–9% a year after tax at −34% to −54% drawdown), '
             'and every one makes the TN / OA·Base Age / IPO book worse on 30 of 30 paths. The return lives in a '
             'stock’s first months after listing: widen the band from 6 to 12 to 24 months and the book falls '
             '22.4% → 16.3% → 14.7%. Two corrections to IPO Base itself: research/167’s +4.8pp edge over random '
             'is +2.3pp on a NaN-robust panel and zero since 2016; and the live book’s MIN_BARS = 60 (the spec was '
             'validated at 25) backtests at 12.4% and lowers the blend’s CAGR by 2.5pp on every path. Nothing live '
             'was changed.'),
    status='COMPLETE',
    date='2026-09-13',
    cardBlurb=('Arun asked why the IPO system only trades IPOs. Because that is where the money is: the same rules on '
               'Nifty 50 through all stocks do not beat random picks on any universe. And the live IPO book turns out '
               'to be running a 12% version of a 22% spec.'),
    cardStats=[{'label': 'transplants beating their random control', 'value': '0 / 16'},
               {'label': 'IPO spec as validated / as coded live', 'value': '22.4% / 12.4%'},
               {'label': 'live-book cost to the 3-sleeve blend', 'value': f"{n(lr['d_cagr'], sign=True)}pp, 30/30"}],
    systemRules=dict(
        intro=('Everything is research/167 Spec A, the rules the live IPO book was re-fitted to on 12-Sep-2026. '
               'Only the universe and the age band move. Size universes are a causal traded-value proxy, because no '
               'point-in-time index membership history exists in the project.'),
        sharedCoreTitle='Held identical across every universe',
        sharedCore=[
            {'k': 'Base', 'v': 'last 25 bars; pivot = highest close, shifted one bar; depth pivot → base low ≤ 30%; prior close below the pivot'},
            {'k': 'Trigger and fill', 'v': 'close above the pivot; next day a buy-stop AT the pivot, filled max(pivot, open), only if the day’s high reached it'},
            {'k': 'Exits', 'v': 'stop close ≤ 0.90×fill → target close ≥ 1.25×fill → close below SMA-50'},
            {'k': 'Gate', 'v': 'no new entries while NIFTYBEES closes below its 150-day average'},
            {'k': 'Liquidity', 'v': '20-day median traded value ≥ ₹5 cr at the prior close; funds excluded by instrument long name'},
            {'k': 'Book', 'v': '8 slots at 18.75%, ₹10 L, 25 bps a side, after tax with FY loss netting, idle cash 5.2% post-tax'},
            {'k': 'Minimum bars', 'v': '25 — what research/167 validated. The live book’s 60 is carried as its own labelled arm'},
        ],
        riskLayer=tbl('What changes: the universe and the age band',
                      'Ranks are recomputed on the first trading day of each month from the trailing 126-bar median traded value known at the prior close (the research/41 method). Checked against point-in-time market cap from Aug-2018 (Q7 table).',
                      ['Universe label', 'Definition'],
                      [['Nifty-50 / 100 / 200 / 500-like', 'traded-value rank 1-50 / 1-100 / 1-200 / 1-500'],
                       ['Midcap-like', 'rank 101-250'], ['Smallcap-like', 'rank 251-500'],
                       ['Beyond 500', 'rank 501+ (still ≥ ₹5 cr)'], ['All stocks', 'every name clearing the ₹5 cr floor'],
                       ['age: no age limit', 'the IPO condition removed'],
                       ['age: seasoned > 6m / > 24m', 'provably listed longer ago (vetted listing date, or data back beyond Jun-2005, or first database bar further back than the band)'],
                       ['age: listed ≤ 6 / 12 / 24 months', 'vetted listing date within the band (≤ 6 months = Spec A)']]),
    ),
    system=dict(
        intro=('For a stock listed under six months, “the highest close of the last 25 bars” is close to its all-time '
               'high since listing and the base is its first base. For a seasoned stock the same words are a 25-day '
               'Donchian close breakout — a different signal. So the transplant answers two questions at once, and the '
               'study separates them: is the edge in the young-stock condition or in the breakout-plus-trail mechanics, '
               'and is a transplanted version just an existing breakout book in disguise?'),
        rows=[{'k': 'Reproduction gate', 'v': 'research/167’s engine reproduces Spec A exactly (21.80% / −26.63%, all 30 seed paths identical to research/168’s curves, null edge +4.78pp 30/30). The new full-universe panel, set to research/167’s conventions, reproduces 21.80% with the same 1,545 signals'},
              {'k': 'The decisive control', 'v': 'date-matched random entry on every universe: the same days and count, names drawn from THAT universe’s eligible set whose high reached their own pivot, the same fill, exits and gate, 30 paired seeds, in all three windows'},
              {'k': 'Pre-registered edge test', 'v': 'W2 paired edge ≥ +1.0pp with ≥ 25/30 wins, AND ≥ 25/30 in both WA 2006-2015 and WB 2016-2026'},
              {'k': 'Pre-registered sleeve bar', 'v': 'research/168’s: +0.10 Calmar or +2pp CAGR at no worse drawdown on ≥ 20/30 paths, beating risk-matched cash on ≥ 25/30, correlation to OA·Base Age below 0.60'},
              {'k': 'Data defenses', 'v': 'per-symbol NaN-robust windows, phantom holiday rows of 24-Apr and 15-Oct-2014 dropped, split back-adjustment (273 events, 38 on ≥ ₹5 cr names); each switched on one at a time and its effect disclosed'},
              {'k': 'Scale', 'v': 'about 100 backtest cells, 30 seeds each, most in three windows; the survivor re-fit was pre-registered and skipped because nothing survived'}]),
    conditions=dict(
        intro='After tax (20% STCG / 12.5% LTCG, Indian FY loss netting), net of 25 bps a side, idle cash 5.2% post-tax, medians of 30 paired seeds.',
        rows=[{'k': 'Period', 'v': 'W2 2006-01-01 → 2026-09-04; WA 2006-2015; WB 2016-2026; market-cap re-run 2018-09 → 2026-09'},
              {'k': 'Book', 'v': '₹10,00,000, 8 slots at 18.75%'},
              {'k': 'Blend', 'v': 'research/168’s 30 paired paths, TN 37.5 / OA·Base Age 37.5 / IPO 25, monthly, 2006-04-03 → 2026-09-03'},
              {'k': 'Cost ladder', 'v': '25 / 40 / 60 bps a side'}]),
    comparisons=[
        tbl('THE TABLE THAT DECIDES IT — Spec A’s rules on every universe, each against its own random-entry control',
            'After tax, 30 paired seeds. The first two rows are IPO Base itself: as validated, and as the live book is coded. No transplant passes; the best, Midcap-like with no age limit, wins 20 of 30 and only 15 of 30 after 2016.',
            T1c, T1r, [0]),
        tbl('Widen the age band and the book decays — and past six months the random control BEATS the breakout',
            'All-stock universe. Spec A itself clears 2006-2015 on every seed and ties chance after 2016 on this clean panel.', T5c, T5r, [0]),
        tbl('Second check on TRUE point-in-time market cap (2018-09 → 2026-09)',
            'fundamentals.db mcap_pit, previous month’s row. Market-cap universes are no better than the traded-value proxies, and Spec A ties its control in this window.', T3c, T3r),
        tbl('Why research/167’s +4.8pp edge over random is really +2.3pp — and zero since 2016',
            'Spec A real vs control, idle cash 5.0% so the first row matches research/167 exactly. One missing row on a union date index made a name’s pivot and trail NaN for weeks; per-symbol windows fix it, the real book barely moves, the control rises.', T7c, T7r, [1]),
        tbl('The live book’s MIN_BARS = 60 is worth about 10 points of CAGR less than the validated 25',
            'research/167’s own engine and panel, unchanged, idle cash 5.0%. The 6-Sep decision read the harness’s “n >= 60” as bars at the signal date; it counts rows over the whole database today.', T8c, T8r, [2]),
    ],
    results=dict(
        metrics=[{'label': 'transplant cells passing the control', 'value': '0 / 16', 'tone': 'neg'},
                 {'label': 'IPO Spec A, validated', 'value': n(allc.loc['all__le6'].w2_cagr, pct=True), 'hint': 'clean panel, 5.2% cash'},
                 {'label': 'IPO live book as coded', 'value': n(allc.loc['all__le6__mb60_livebook'].w2_cagr, pct=True), 'tone': 'neg', 'hint': 'MIN_BARS 60'},
                 {'label': 'best transplant', 'value': n(allc.loc['mid101_250__none'].w2_cagr, pct=True), 'tone': 'neg', 'hint': 'Midcap-like, −50% DD'},
                 {'label': 'live book vs validated, inside the blend', 'value': f"{n(lr['d_cagr'], sign=True)}pp", 'tone': 'neg', 'hint': '30 of 30 paths'}],
        tables=[
            tbl('Portfolio fit — every transplant makes the TN / OA·Base Age / IPO book worse',
                'Paired on research/168’s 30 paths against TN 37.5 / OA·Base Age 37.5 / IPO-A 25 (21.18% / −24.01% / 0.885). “vs cash” = the same book with cash in place of the candidate, the cash weight solved to the same median drawdown.', T6c, T6r),
            tbl('The Capital Desk decision — what 25% to IPO buys today',
                'The validated spec earns its 25%. The version the live book runs lowers the blend’s CAGR on every path and adds barely more than cash at equal risk.', T9c, T9r, [1]),
            tbl('Tradeability and capacity',
                'Bigger universes do lift capacity — on books that earn 2-9% with no edge. Note Spec A’s p90 position is ~10% of a name’s 20-day traded value at ₹10 L: research/167 quoted its median (1.56%) as the p90.', T2c, T2r),
            tbl('Cost ladder', 'After tax. Transplants trade ~36 times a year and lose 4-6 points from 25 to 60 bps.', T4c, T4r),
            tbl('How good is the size proxy?', proxy_note, T10c, T10r),
            tbl('Per-year house table — return (intra-year drawdown from the full curve’s running peak)',
                'Median-CAGR path of each ensemble; summary row = 30-path medians. Best-of columns exclude NIFTYBEES. 2006-04-03 → 2026-09-03.', yc2, yr2),
        ]),
    winners=[dict(
        config='No new sleeve. IPO Base stays the third sleeve — at the VALIDATED min_bars 25',
        summary=('Nothing transplants. The decision this study changes is not which sleeve but which version of it: '
                 'the live book should run MIN_BARS = 25 before it is funded at 25%. That is a strategy change with its '
                 'own STATUS doc, a capacity check on the thinner early entries, and an after-15:40 deploy — not done here.'),
        metrics=[{'k': 'Validated Spec A (clean panel)', 'v': f"{n(allc.loc['all__le6'].w2_cagr, pct=True)} / {n(allc.loc['all__le6'].w2_dd, pct=True)} / Calmar {n(allc.loc['all__le6'].w2_calmar, 3)}"},
                 {'k': 'Live book as coded (60 bars)', 'v': f"{n(allc.loc['all__le6__mb60_livebook'].w2_cagr, pct=True)} / {n(allc.loc['all__le6__mb60_livebook'].w2_dd, pct=True)} / Calmar {n(allc.loc['all__le6__mb60_livebook'].w2_calmar, 3)}"},
                 {'k': 'Blend with validated IPO at 25%', 'v': f"{n(B3['cagr'], pct=True)} / {n(B3['dd'], pct=True)} / {n(B3['calmar'], 3)}"},
                 {'k': 'Blend with live-coded IPO at 25%', 'v': f"{n(lr['cagr'], pct=True)} / {n(lr['dd'], pct=True)} / {n(lr['calmar'], 3)}"}],
        rejected=['Every size-universe transplant, with or without the age band (0 of 16 beat their control)',
                  'Wider age bands of 12 and 24 months (the random control beats the breakout)',
                  'The live book’s MIN_BARS = 60 as a 25% sleeve (−2.46pp blend CAGR on 30/30 paths)'])],
    caveats=['Size universes are a causal traded-value proxy; the Midcap- and Smallcap-like bands overlap the real market-cap bands only ~45%. The 2018+ market-cap re-run agrees, but it is eight years, not twenty.',
             'Survivorship: names never onboarded to Kite cannot be measured; delisted names that are in the database are traded.',
             'The ₹5 cr floor is nominal, so every universe admits fewer names in 2006 than in 2026.',
             'Split back-adjustment treats demerger drops as price adjustments (total-return convention).',
             'The survivor re-fit (trail × base length) was pre-registered and not run because nothing survived; OA · Base Age and research/71 already cover seasoned-stock breakouts with other exits.',
             'The Capital Desk table mixes research/168’s curves (research/167 panel) with this study’s clean-panel live-book curve; the clean-panel Spec A ties IPO-A inside the blend (+0.05pp, correlation 0.925), so they are commensurable for this purpose.',
             'No equity-curve chart pack: the family died at the control gate, before a tearsheet is owed.'],
    reports=[{'label': 'research/167 — the spec transplanted here', 'href': '/app/backtest/ipo-base-honest-reopt-research167'},
             {'label': 'research/168 — the three-sleeve blend this is tested against', 'href': 'https://github.com/castroarun/Quantifyd/blob/main/research/168_three_sleeve_blend/results/RESULTS.md'}],
    githubLinks=[{'label': 'research/169 folder', 'href': 'https://github.com/castroarun/Quantifyd/tree/main/research/169_ipo_rules_universe_transplant'}],
    projectPaths=['research/169_ipo_rules_universe_transplant/results/RESULTS.md',
                  'research/169_ipo_rules_universe_transplant/IPO_RULES_UNIVERSE_TRANSPLANT_DAILY_SWEEP_STATUS.md',
                  'research/169_ipo_rules_universe_transplant/scripts/xpanel.py',
                  'research/169_ipo_rules_universe_transplant/scripts/run169.py'],
)


def insert_after(path, anchor_pred, text, guard, start_pred=None):
    lines = path.read_text().split('\n')
    if guard in '\n'.join(lines):
        print('skip (already present):', path.name, guard[:40])
        return False
    i0 = 0
    if start_pred:
        i0 = next(i for i, ln in enumerate(lines) if start_pred(ln))
    idx = next(i for i in range(i0, len(lines)) if anchor_pred(lines[i]))
    lines.insert(idx + 1, text)
    path.write_text('\n'.join(lines))
    print('inserted into', path.name, 'after line', idx + 1)
    return True


# backtests.ts - new entry at the top of the array (newest first, as research/167 is)
bt = ROOT / 'frontend/src/data/backtests.ts'
obj = json.dumps(entry, ensure_ascii=False, indent=2)
obj = '\n'.join('  ' + ln for ln in obj.split('\n')) + ','
insert_after(bt, lambda ln: ln.startswith('export const BACKTEST_STUDIES'), obj, "\"slug\": \"" + SLUG + "\"")
cav = ('      ' + json.dumps(
    'CORRECTED 13-Sep-2026 by research/169 (' + URL + '). (1) The +4.78pp / 30-of-30 edge over the random control '
    'was measured on a panel whose rolling windows ran on a union date index, so one missing row made a name’s '
    'pivot and trail NaN for weeks. On per-symbol windows the real book is unchanged (22.24%) but the control rises '
    'to 20.03%: edge +2.25pp, +4.62pp in 2006-2015 and −0.18pp (13 of 30) in 2016-2026, a window this study never '
    'ran the control on. (2) “The p90 position is 1.56% of 20-day traded value at ₹10 L” is the MEDIAN; the '
    'p90 is 9.05%. (3) This spec was validated at min_bars 25; at 60, which the live book runs, the same engine '
    'returns 11.57%.', ensure_ascii=False) + ',')
insert_after(bt, lambda ln: ln.strip().startswith('caveats: ['), cav, 'CORRECTED 13-Sep-2026 by research/169',
             start_pred=lambda ln: "slug: 'ipo-base-honest-reopt-research167'" in ln)

# strategies.ts - IPO row: study link + dated change-log note (no rule changed)
st = ROOT / 'frontend/src/data/strategies.ts'
link = ('      { slug: ' + json.dumps(SLUG) + ', title: ' + json.dumps(
    'Why is it IPO-specific? The rules on Nifty 50 → all stocks fail everywhere — and the live MIN_BARS 60 is not the validated spec',
    ensure_ascii=False) + ", verdict: 'NO EDGE' },")
insert_after(st, lambda ln: ln.strip().startswith('studies: ['), link, SLUG,
             start_pred=lambda ln: "id: 'ipo-base'" in ln)
note = ("      { date: '13 Sep 2026', text: " + json.dumps(
    'research/169 FINDING, NO RULE CHANGED: MIN_BARS 60 is not what was validated. The 6-Sep reading of the harness '
    'was wrong — its “n >= 60” counts rows over the whole database today, not bars at the signal date, so '
    'research/153 and research/167 did scan stocks from their 25th bar. research/167’s own engine: min_bars 25 '
    '→ 21.80% after tax, 60 → 11.57%. Inside TN 37.5 / OA 37.5 / IPO 25 the 60-bar book costs '
    + n(lr['d_cagr'], sign=True) + 'pp of CAGR on 30 of 30 paths against the validated spec. Also: the HARD CAP line’s '
    '“90th percentile” figure from research/167 is its MEDIAN (1.56%); the p90 position is 9.05% of a name’s '
    '20-day traded value at ₹10 L. Decision owed before the 26-Sep funding call; review 19-Sep-2026. ' + URL,
    ensure_ascii=False) + ' },')
insert_after(st, lambda ln: ln.strip().startswith('changeLog: ['), note, 'research/169 FINDING',
             start_pred=lambda ln: "id: 'ipo-base'" in ln)

# research/INDEX.md - append newest at the bottom
ix = ROOT / 'research/INDEX.md'
txt = ix.read_text()
if '| 169 |' not in txt:
    row = ('| 169 | [**Why is IPO Base IPO-specific? Its Spec A rules transplanted to Nifty-50 / 100 / 200 / Midcap / 500 / '
           'Smallcap / all-stock universes**](169_ipo_rules_universe_transplant/results/RESULTS.md) — Arun: *"Why is '
           'the IPO system IPO specific, what if we apply this to other universes like nifty 50, 100, 200, midcap, 500, '
           'small cap, all stocks?"* Asked before moving the Capital Desk to TN 37.5 / OA 37.5 / IPO 25. Spec A with the '
           'age band removed and with seasoned names only, on 8 causal traded-value size universes, each against its '
           'own date-matched random-entry control; the age band swept; a point-in-time market-cap re-run 2018+; '
           'correlation and paired blends against research/168’s book | daily 2006-01-01 → 2026-09-04; ~100 '
           'cells × 30 seeds × 3 windows; 8 slots @ 18.75%, ₹10 L, after tax, 25/40/60 bps, cash 5.2% | '
           '**0 of 16 transplants beat their control**: 2.3-9.0% after tax at −34 to −54% DD, W2 edges '
           '−3.24 to +1.52pp; market-cap universes no better; every transplant lowers the three-sleeve blend on '
           '30/30 paths (replacing IPO-A: −3.5 to −3.8pp CAGR). **Why IPO-specific: the return lives in the first '
           'months after listing** — age ≤ 6m 22.39% → ≤ 12m 16.28% → ≤ 24m 14.68% → none 6.97% '
           '→ seasoned 5.86%, and past 6 months the random control BEATS the breakout. **Three corrections to IPO '
           'Base**: (1) r/167’s +4.78pp edge over random is +2.25pp on a NaN-robust panel, +4.62pp 2006-15 and '
           '−0.18pp (13/30) 2016-26 — its union-date windows went NaN after any missing row; (2) its capacity line '
           'quoted the MEDIAN position (1.56% of 20d traded value) as the p90 (9.05%); (3) **the live book runs '
           'MIN_BARS 60 but Spec A was validated at 25** — r/167’s own engine: 21.80% at 25, 11.57% at 60; inside '
           'TN/OA/IPO 37.5/37.5/25 the 60-bar book costs −2.46pp CAGR on 30/30 paths | **NO EDGE (transplants) · '
           'IPO Base stays the third sleeve, but only the VALIDATED min_bars 25 earns the 25%; nothing live changed; '
           'review 2026-09-19** |')
    ix.write_text(txt.rstrip('\n') + '\n' + row + '\n')
    print('INDEX row appended')

# TODO.md - new section at the top
td = ROOT / 'TODO.md'
t = td.read_text()
if 'research/169' not in t:
    sec = f"""## \U0001F534 2026-09-13 — research/169: why IPO Base is IPO-specific — the rules do NOT transplant, and **the live IPO book runs MIN_BARS 60, not the validated 25**

Arun asked why the IPO system only trades IPOs, and what happens on Nifty 50 / 100 / 200 / Midcap / 500 /
Smallcap / all stocks, before moving the Capital Desk to TN 37.5 / OA 37.5 / IPO 25.
Published at `{URL}`. **Nothing live was changed.**

**IPO — transplants: NO EDGE.** 0 of 16 size-universe transplants (age band removed, or seasoned names
only) beat their own date-matched random-entry control: 2.3-9.0% after tax at −34% to −54% drawdown.
A point-in-time market-cap re-run 2018+ agrees. Every transplant lowers the three-sleeve blend on 30 of 30
paths. **Why it is IPO-specific:** the return lives in a stock's first months after listing — age ≤ 6m
22.39%, ≤ 12m 16.28%, ≤ 24m 14.68%, no limit 6.97% — and past six months random young names beat the
breakout.

**IPO — what is OWED (by Arun, before the 26-Sep funding call; review registered 19-Sep-2026):**
- **Decide MIN_BARS.** `services/ipo_paper.py` runs 60; Spec A was validated at 25. The 6-Sep comment
  misread the harness (`n >= 60` counts rows over the whole DB today, not bars at the signal date).
  research/167's own engine: **21.80% at 25, 11.57% at 60**. Inside TN/OA/IPO 37.5/37.5/25 the 60-bar
  book costs **{n(lr['d_cagr'], sign=True)}pp CAGR on 30/30 paths** and beats risk-matched cash by only
  {n(lr['d_cagr_vs_cash'], sign=True)}pp. Changing it = its own STATUS doc + capacity check on entries
  25-60 sessions after listing + after-15:40 deploy.
- **Read research/167's null claim as +2.25pp, not +4.78pp** — +4.62pp in 2006-15, −0.18pp (13/30)
  in 2016-26 on a NaN-robust panel. The sleeve is a young-listing cohort harvest with a good exit.
- **Capacity**: research/167's "p90 1.56% of traded value at ₹10 L" is the MEDIAN; the p90 is 9.05%.
  The ₹20-25 L hard cap in the Strategies register is optimistic on the tail.

- Full write-up: `research/169_ipo_rules_universe_transplant/results/RESULTS.md`
- Status doc: `research/169_ipo_rules_universe_transplant/IPO_RULES_UNIVERSE_TRANSPLANT_DAILY_SWEEP_STATUS.md`

"""
    i = t.index('\n## ') + 1
    td.write_text(t[:i] + sec + t[i:])
    print('TODO section inserted')

# ops_center.py REVIEWS
oc = ROOT / 'research/111_sensex_manual_mgmt/scripts/ops_center.py'
o = oc.read_text()
if 'research/169' not in o:
    title = 'research/169 - IPO Base: the live book runs MIN_BARS 60, not the validated 25. Decide BEFORE funding IPO at 25%'
    ev = ('research/169 (13-Sep-2026) transplanted IPO Spec A to Nifty-50/100/200/500/Midcap/Smallcap/all-stock '
          'universes: 0 of 16 beat their random-entry control, so IPO Base stays the only third sleeve. But '
          'services/ipo_paper.py runs MIN_BARS = 60 while research/167 validated 25 - the 6-Sep comment read the '
          'harness n >= 60 as bars at the signal date; it counts rows over the whole DB today. research/167 own '
          'engine: 21.80% after tax at 25, 11.57% at 60 (-38.97% DD). Inside TN 37.5 / OA 37.5 / IPO 25 the 60-bar '
          'book costs ' + n(lr['d_cagr'], sign=True).replace(M, '-') + 'pp CAGR on 30/30 paths and beats '
          'risk-matched cash by only ' + n(lr['d_cagr_vs_cash'], sign=True).replace(M, '-') + 'pp. OWED BY ARUN: '
          'decide MIN_BARS; if 25, open a strategy-change STATUS doc, check capacity for entries 25-60 sessions '
          'after listing (Spec A p90 position is 9.05% of 20-day traded value at Rs 10L - research/167 quoted the '
          'median 1.56% as the p90), deploy after 15:40, and update the Strategies register. PASS = a recorded '
          'MIN_BARS decision with the evidence line, before the 26-Sep funding call. Also read research/167 null '
          'edge as +2.25pp (2016-26: -0.18pp, 13/30), not +4.78pp. Evidence: '
          'research/169_ipo_rules_universe_transplant/results/RESULTS.md Q5-Q6.')
    tup = '    (' + repr(title) + ', ' + repr('2026-09-19') + ', ' + repr('PENDING') + ', ' + repr(ev) + '),'
    lines = o.split('\n')
    k = next(i for i, ln in enumerate(lines) if ln.startswith('REVIEWS = ['))
    lines.insert(k + 1, tup)
    oc.write_text('\n'.join(lines))
    print('ops REVIEWS entry inserted')

# docs/LABS_AND_JOBS_REFERENCE.md mirror
lb = ROOT / 'docs/LABS_AND_JOBS_REFERENCE.md'
l = lb.read_text()
if 'research/169' not in l:
    l = l.rstrip('\n') + ('\n\n## Review 2026-09-19 - IPO Base MIN_BARS 60 vs the validated 25 (research/169, added 13-Sep-2026)\n'
                          'The live IPO book (`services/ipo_paper.py`) runs `MIN_BARS = 60`; research/167 validated Spec A at 25. '
                          'research/167’s own engine: 21.80% after tax at 25, 11.57% at 60; inside TN/OA/IPO 37.5/37.5/25 the '
                          '60-bar book lowers blend CAGR by ' + n(lr['d_cagr'], sign=True) + 'pp on 30 of 30 paths. Owed by Arun before the '
                          '26-Sep funding call: a recorded MIN_BARS decision (a change is its own STATUS doc, capacity check and '
                          'after-15:40 deploy). Registered in `ops_center.py` REVIEWS. Evidence: '
                          '`research/169_ipo_rules_universe_transplant/results/RESULTS.md` Q5-Q6.\n')
    lb.write_text(l)
    print('LABS doc mirrored')
print('close_loop done')
