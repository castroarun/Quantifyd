"""
research/159 v2 — summarise the base-ceiling-breakout event table.

SELECTION IS BLIND TO OUTCOME. The only ranking input is `pattern_quality`, computed
from bars <= the base-qualify day. Every `info_*` column (forward returns, SuperTrend
outcome, MFE/MAE) is printed for context but is NEVER read by the selection code.
"""
import csv
from collections import Counter
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / 'results'
CSV_IN = RES / 'rounding_base_events_v2.csv'
OUT = RES / 'summary_v2.txt'
MODE_PREF = {'VAR': 0, 'L250': 1, 'L180': 2, 'L120': 3}
CHECK = ['SKFINDIA', 'KMEW', 'CENTURYPLY', 'DIXON', 'SRF']


def f(x, d=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def main():
    rows = list(csv.DictReader(open(CSV_IN, encoding='utf-8')))
    out = []

    def p(s=''):
        print(s)
        out.append(s)

    p('=' * 100)
    p('research/159 v2 — ROUNDING BASE -> BASE-CEILING BREAKOUT ON VOLUME')
    p('Entry = first close above the prior 60-bar high, on >=3x the prior 20-bar median')
    p('volume, as an up-candle, after the saucer base is recognised. Fill = next-day open.')
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
    obv = [r for r in ded if r['obv_filter_pass'] == '1']
    p('pass the OBV/volume accumulation filter: %d (%.0f%%)'
      % (len(obv), 100.0 * len(obv) / max(len(ded), 1)))

    p('')
    p('--- EVENTS PER YEAR (de-duplicated) ---')
    cy = Counter(r['trigger_date'][:4] for r in ded)
    cyo = Counter(r['trigger_date'][:4] for r in obv)
    p('%-6s %8s %8s' % ('year', 'all', 'OBV_ON'))
    for y in sorted(cy):
        p('%-6s %8d %8d   %s' % (y, cy[y], cyo[y], '#' * min(cy[y] // 4, 60)))
    p('%-6s %8d %8d' % ('TOTAL', sum(cy.values()), sum(cyo.values())))
    p('mean %.1f events/year across %d years' % (sum(cy.values()) / len(cy), len(cy)))

    p('')
    p('--- VOLUME MULTIPLE AT THE TRIGGER (the K axis, for later filtering) ---')
    vx = sorted(x for x in (f(r['vol_multiple']) for r in ded) if x is not None)
    if vx:
        p('  median %.1fx   p10 %.1fx   p90 %.1fx' % (vx[len(vx) // 2], vx[int(.1 * len(vx))], vx[int(.9 * len(vx))]))
        for k in (3, 5, 9):
            p('  events with volume >= %dx : %d (%.0f%%)'
              % (k, sum(1 for x in vx if x >= k), 100.0 * sum(1 for x in vx if x >= k) / len(vx)))

    p('')
    p('--- DISTANCE FROM ENTRY TO THE OLD v1 LEFT RIM (overhead supply ahead) ---')
    dd = sorted(x for x in (f(r['dist_to_left_rim_pct']) for r in ded) if x is not None)
    if dd:
        p('  median %+.1f%%   p10 %+.1f%%   p90 %+.1f%%   (positive = entry BELOW the left rim)'
          % (dd[len(dd) // 2], dd[int(.1 * len(dd))], dd[int(.9 * len(dd))]))
        p('  entries below the left rim: %d of %d (%.0f%%) — v2 buys the base, not the recovery'
          % (sum(1 for x in dd if x > 0), len(dd), 100.0 * sum(1 for x in dd if x > 0) / len(dd)))

    p('')
    p('--- CONFIRMATION ON THE NAMED EXAMPLES ---')
    for sym in CHECK:
        hits = sorted([r for r in rows if r['symbol'] == sym], key=lambda r: r['trigger_date'])
        recent = [r for r in hits if r['trigger_date'] >= '2005-01-01']
        p('')
        p('  %s — %d event(s) total' % (sym, len({r['trigger_date'] for r in hits})))
        seen = set()
        for r in recent:
            if r['trigger_date'] in seen:
                continue
            seen.add(r['trigger_date'])
            p('    %s  close Rs%-9s ceiling Rs%-9s vol %sx  | base q=%s rim Rs%s (%s) trough Rs%s (%s) depth %s%%'
              % (r['trigger_date'], r['trigger_close'], r['trigger_level'], r['vol_multiple'],
                 r['base_qualify_date'], r['left_rim_level'], r['left_rim_date'],
                 r['trough_close'], r['trough_date'], r['depth_pct']))
            p('        entry %s @ Rs%s | distToLeftRim %s%% | score %s | [info] fwd250=%s%% ST(7,3) %s%%'
              % (r['entry_date'], r['entry_open'], r['dist_to_left_rim_pct'],
                 r['pattern_quality'], r['info_fwd250_pct'], r['info_st73_ret_pct']))

    # ---------- SHORT-LIST: pattern_quality ONLY ----------
    p('')
    p('--- SHORT-LIST (ranked by pattern_quality ONLY; info_* never consulted) ---')

    def bucket(r):
        tv = f(r['tv20_cr_at_trigger'], 0.0) or 0.0
        return 'small(<Rs10cr)' if tv < 10 else ('mid(Rs10-50cr)' if tv < 50 else 'large(>=Rs50cr)')

    pool = [r for r in ded if r['obv_filter_pass'] == '1'
            and f(r['entry_open']) is not None
            and r['trigger_date'] < '2025-09-01']
    bysym = {}
    for r in pool:
        k = r['symbol']
        if k not in bysym or f(r['pattern_quality'], 0) > f(bysym[k]['pattern_quality'], 0):
            bysym[k] = r
    ranked = sorted(bysym.values(), key=lambda r: -f(r['pattern_quality'], 0))
    p('eligible pool: %d events / %d distinct symbols' % (len(pool), len(bysym)))

    picks, uy, ub = [], set(), set()
    for r in ranked:
        y, b = r['trigger_date'][:4], bucket(r)
        if y in uy or b in ub:
            continue
        picks.append(r); uy.add(y); ub.add(b)
        if len(picks) == 3:
            break
    for i, r in enumerate(picks, 1):
        p('')
        p('  [%d] %s  mode=%s  score=%s  liquidity=%s' % (i, r['symbol'], r['mode'],
                                                          r['pattern_quality'], bucket(r)))
        p('      left rim (supply): %s @ Rs%s   trough: %s @ Rs%s (depth %s%%, %s bars)'
          % (r['left_rim_date'], r['left_rim_level'], r['trough_date'],
             r['trough_close'], r['depth_pct'], r['base_len_bars']))
        p('      base recognised  : %s' % r['base_qualify_date'])
        p('      BASE CEILING     : Rs%s   BREAKOUT %s close Rs%s on %sx volume (%s bars after q)'
          % (r['trigger_level'], r['trigger_date'], r['trigger_close'],
             r['vol_multiple'], r['days_q_to_trigger']))
        p('      ENTRY            : %s @ Rs%s (next-day open)   still %s%% below the left rim'
          % (r['entry_date'], r['entry_open'], r['dist_to_left_rim_pct']))
        p('      shape            : R2=%s vertex=%s no-V=%s volRatio=%s obvGain=%s'
          % (r['fit_r2'], r['vertex_frac'], r['flat_frac'], r['vol_ratio'], r['obv_gain']))
        p('      [INFO ONLY] fwd60=%s%% fwd120=%s%% fwd250=%s%% | ST(7,3) exit %s ret %s%% bars %s | MFE %s%% MAE %s%%'
          % (r['info_fwd60_pct'], r['info_fwd120_pct'], r['info_fwd250_pct'],
             r['info_st73_exit_date'] or 'still open', r['info_st73_ret_pct'],
             r['info_st73_bars_held'], r['info_mfe_250_pct'], r['info_mae_250_pct']))

    p('')
    p('--- next 12 by pattern_quality (context only) ---')
    for r in ranked[:12]:
        p('  %-14s %s  score %s  ceiling Rs%-9s vol %sx  depth %s%%  R2 %s  tv20 %s cr'
          % (r['symbol'], r['trigger_date'], r['pattern_quality'], r['trigger_level'],
             r['vol_multiple'], r['depth_pct'], r['fit_r2'], r['tv20_cr_at_trigger']))

    import math
    vals = sorted(x for x in (f(r['info_fwd250_pct']) for r in ded)
                  if x is not None and math.isfinite(x))
    if vals:
        p('')
        p('--- INFORMATION ONLY: +250-bar return from the fill, all %d events ---' % len(vals))
        p('  NOT a backtest: no costs, no taxes, no sizing, no slot limit, no market gate,')
        p('  and a survivorship-biased universe.')
        p('  median %+.1f%%  mean %+.1f%%  p10 %+.1f%%  p90 %+.1f%%  frac>0 %.0f%%'
          % (vals[len(vals) // 2], sum(vals) / len(vals), vals[int(.1 * len(vals))],
             vals[int(.9 * len(vals))], 100.0 * sum(1 for x in vals if x > 0) / len(vals)))

    OUT.write_text('\n'.join(out), encoding='utf-8')
    print('\nwrote %s' % OUT)


if __name__ == '__main__':
    main()
