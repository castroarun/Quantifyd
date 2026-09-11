"""
research/159 v3 — summary: per-year counts, expected-check table, blind short-list,
and live candidates from the last 10 trading days.

SELECTION IS BLIND TO OUTCOME: ranked by `pattern_quality` only (computed from bars
<= the base-qualify day). Info_* columns are printed but never read by selection.
"""
import csv
import sqlite3
from collections import Counter
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / 'results'
DB = Path(__file__).resolve().parents[3] / 'backtest_data' / 'market_data.db'
MODE_PREF = {'VAR': 0, 'L250': 1, 'L180': 2, 'L120': 3}
DROP_SYMBOLS = {'SILLYMONKS'}          # duplicate of CRESTO (identical series)

EXPECT = [
    ('KMEW',        '2025-09-12', 'FIRE'),
    ('CENTURYPLY',  '2017-03-30', 'FIRE'),
    ('JAYSREETEA',  '2009-08-12', 'FIRE'),
    ('MONARCH',     '2023-10-06', 'FIRE'),
    ('NAM-INDIA',   '2025-06-06', 'FIRE'),
    ('SAPPHIRE',    '2022-10-06', 'FIRE'),
    ('COROMANDEL',  '2009-09-29', 'FIRE'),
    ('CHOLAHLDNG',  '2025-04',    'NOFIRE'),
    ('SKFINDIA',    '2025-05',    'NOFIRE'),
]


def f(x, d=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def main():
    rows = [r for r in csv.DictReader(open(RES / 'rounding_base_events_v3.csv', encoding='utf-8'))
            if r['symbol'] not in DROP_SYMBOLS]
    out = []

    def p(s=''):
        print(s)
        out.append(s)

    p('=' * 100)
    p('research/159 v3 — SAUCER -> SHELF BREAKOUT NEAR THE ALL-TIME HIGH')
    p('Trigger: close > shelf high (prior 15 bars, range <=12%) AND close >= 0.90 x causal ATH')
    p('         AND volume >= 3x prior-20-bar median AND up-candle, after the saucer qualified.')
    p('Fill: next-day open.   Duplicate series dropped: SILLYMONKS (= CRESTO).')
    p('=' * 100)
    p('raw events (symbol x window-mode): %d' % len(rows))

    best = {}
    for r in rows:
        k = (r['symbol'], r['trigger_date'])
        cur = best.get(k)
        if cur is None or MODE_PREF[r['mode']] < MODE_PREF[cur['mode']]:
            best[k] = r
    ded = sorted(best.values(), key=lambda r: r['trigger_date'])
    p('de-duplicated events (symbol x trigger date): %d' % len(ded))
    p('distinct symbols: %d' % len({r['symbol'] for r in ded}))
    p('pass OBV accumulation filter: %d (%.0f%%)'
      % (sum(1 for r in ded if r['obv_filter_pass'] == '1'),
         100.0 * sum(1 for r in ded if r['obv_filter_pass'] == '1') / len(ded)))
    p('')
    p('funnel: %d bases qualified -> %d triggered, %d expired without a shelf breakout'
      % (7874, len(rows), 5842))

    p('')
    p('--- EVENTS PER YEAR (de-duplicated) ---')
    cy = Counter(r['trigger_date'][:4] for r in ded)
    for y in sorted(cy):
        p('%-6s %6d   %s' % (y, cy[y], '#' * min(cy[y], 70)))
    p('%-6s %6d' % ('TOTAL', sum(cy.values())))
    p('mean %.1f events/year across %d years' % (sum(cy.values()) / len(cy), len(cy)))

    p('')
    p('--- DISTANCE TO ATH AT THE TRIGGER ---')
    da = sorted(x for x in (f(r['dist_to_ath_pct']) for r in ded) if x is not None)
    p('  median %+.1f%%  p10 %+.1f%%  p90 %+.1f%%' % (da[len(da) // 2], da[int(.1 * len(da))], da[int(.9 * len(da))]))
    p('  at a NEW all-time high: %d (%.0f%%) | within 5%% of ATH: %d (%.0f%%)'
      % (sum(1 for r in ded if r['ath_new_high'] == '1'),
         100.0 * sum(1 for r in ded if r['ath_new_high'] == '1') / len(ded),
         sum(1 for r in ded if r['ath_ge_095'] == '1'),
         100.0 * sum(1 for r in ded if r['ath_ge_095'] == '1') / len(ded)))
    sr = sorted(x for x in (f(r['shelf_range_pct']) for r in ded) if x is not None)
    p('  shelf range: median %.1f%%  p10 %.1f%%  p90 %.1f%%' % (sr[len(sr) // 2], sr[int(.1 * len(sr))], sr[int(.9 * len(sr))]))
    vx = sorted(x for x in (f(r['vol_multiple']) for r in ded) if x is not None)
    p('  volume multiple: median %.1fx  p90 %.1fx  (>=5x: %d, >=9x: %d)'
      % (vx[len(vx) // 2], vx[int(.9 * len(vx))],
         sum(1 for x in vx if x >= 5), sum(1 for x in vx if x >= 9)))

    p('')
    p('--- EXPECTED-CHECK TABLE ---')
    p('  %-12s %-14s %-8s %s' % ('symbol', 'expected', 'result', 'detail'))
    allpass = True
    for sym, when, kind in EXPECT:
        hits = sorted({r['trigger_date'] for r in rows if r['symbol'] == sym})
        if kind == 'FIRE':
            ok = when in hits
            ev = next((r for r in rows if r['symbol'] == sym and r['trigger_date'] == when), None)
            detail = ('close Rs%s shelf Rs%s dATH %s%% vol %sx entry %s @ Rs%s'
                      % (ev['trigger_close'], ev['shelf_high'], ev['dist_to_ath_pct'],
                         ev['vol_multiple'], ev['entry_date'], ev['entry_open'])) if ev else \
                     ('NOT FOUND; events: %s' % (', '.join(hits) or 'none'))
        else:
            bad = [d for d in hits if d.startswith(when)]
            ok = not bad
            detail = ('correctly absent; other events: %s' % (', '.join(hits) or 'none')) if ok \
                else ('UNEXPECTEDLY FIRED on %s' % ', '.join(bad))
        allpass &= ok
        p('  %-12s %-14s %-8s %s' % (sym, when, 'PASS' if ok else 'FAIL', detail))
    p('  ==> %s' % ('ALL EXPECTED CHECKS PASS' if allpass else '*** SOME CHECKS FAILED ***'))

    # ---------------- short-list: 6-8 blind picks ----------------
    def bucket(r):
        tv = f(r['tv20_cr_at_trigger'], 0.0) or 0.0
        return 'small' if tv < 10 else ('mid' if tv < 50 else 'large')

    maxd = max(r['trigger_date'] for r in ded)
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    last10 = [r[0] for r in con.execute(
        "SELECT DISTINCT date FROM market_data_unified WHERE timeframe='day' "
        "ORDER BY date DESC LIMIT 10")]
    con.close()
    cutoff = min(last10)

    pool = [r for r in ded if f(r['entry_open']) is not None and r['trigger_date'] < cutoff]
    bysym = {}
    for r in pool:
        k = r['symbol']
        if k not in bysym or f(r['pattern_quality'], 0) > f(bysym[k]['pattern_quality'], 0):
            bysym[k] = r
    ranked = sorted(bysym.values(), key=lambda r: -f(r['pattern_quality'], 0))

    p('')
    p('--- SHORT-LIST: 8 SAMPLE TRADES (ranked by pattern_quality ONLY) ---')
    p('    diversified: at most one per year and per liquidity bucket until exhausted')
    picks, uy, ub = [], Counter(), Counter()
    for r in ranked:
        y, b = r['trigger_date'][:4], bucket(r)
        if uy[y] >= 1 or ub[b] >= 3:
            continue
        picks.append(r); uy[y] += 1; ub[b] += 1
        if len(picks) == 8:
            break
    p('')
    p('  %-12s %-12s %-10s %-9s %-8s %-7s %-12s %-10s %s'
      % ('symbol', 'breakout', 'close', 'shelf_hi', 'shlfRng', 'dATH', 'entry', 'entryPx', 'score'))
    for r in picks:
        p('  %-12s %-12s %-10s %-9s %-8s %-7s %-12s %-10s %s'
          % (r['symbol'], r['trigger_date'], r['trigger_close'], r['shelf_high'],
             r['shelf_range_pct'] + '%', r['dist_to_ath_pct'] + '%', r['entry_date'],
             r['entry_open'], r['pattern_quality']))
    p('')
    p('  detail (info_* columns are NOT used in selection):')
    for i, r in enumerate(picks, 1):
        p('   [%d] %s  trough %s @ Rs%s (depth %s%%)  ATH Rs%s  vol %sx  hist %s bars  %s cap'
          % (i, r['symbol'], r['trough_date'], r['trough_close'], r['depth_pct'],
             r['ath_before_breakout'], r['vol_multiple'], r['hist_bars'], bucket(r)))
        p('       [INFO] fwd250 %s%%  ST(7,3) %s%% over %s bars  MFE %s%%  MAE %s%%'
          % (r['info_fwd250_pct'], r['info_st73_ret_pct'], r['info_st73_bars_held'],
             r['info_mfe_250_pct'], r['info_mae_250_pct']))

    p('')
    p('--- LIVE CANDIDATES: triggered in the last 10 trading days (>= %s) ---' % cutoff)
    live = [r for r in ded if r['trigger_date'] >= cutoff]
    if not live:
        p('  none')
    for r in sorted(live, key=lambda r: -f(r['pattern_quality'], 0)):
        p('  %-12s trigger %s close Rs%-9s shelf Rs%-9s rng %-6s dATH %-7s vol %-6sx entry %s @ Rs%s  score %s'
          % (r['symbol'], r['trigger_date'], r['trigger_close'], r['shelf_high'],
             r['shelf_range_pct'] + '%', r['dist_to_ath_pct'] + '%', r['vol_multiple'],
             r['entry_date'] or '(next open)', r['entry_open'] or '-', r['pattern_quality']))

    import math
    vals = sorted(x for x in (f(r['info_fwd250_pct']) for r in ded)
                  if x is not None and math.isfinite(x))
    p('')
    p('--- INFORMATION ONLY: +250-bar return from the fill, all %d events ---' % len(vals))
    p('  NOT a backtest: no costs, taxes, sizing, slot limit or market gate; survivorship-biased.')
    p('  median %+.1f%%  mean %+.1f%%  p10 %+.1f%%  p90 %+.1f%%  frac>0 %.0f%%'
      % (vals[len(vals) // 2], sum(vals) / len(vals), vals[int(.1 * len(vals))],
         vals[int(.9 * len(vals))], 100.0 * sum(1 for x in vals if x > 0) / len(vals)))

    (RES / 'summary_v3.txt').write_text('\n'.join(out), encoding='utf-8')
    print('\nwrote %s' % (RES / 'summary_v3.txt'))


if __name__ == '__main__':
    main()
