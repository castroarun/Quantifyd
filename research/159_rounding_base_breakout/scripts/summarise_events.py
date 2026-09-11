"""
research/159 — summarise the rounding-base event table.

De-duplicates across window modes, prints per-year counts, runs the KMEW replication
check, and selects the short-list for Arun's manual chart verification.

SELECTION IS BLIND TO OUTCOME. The only ranking input is `pattern_quality`, which is
computed from bars <= the breakout day (see STATUS 3.8). Every `info_*` column
(forward returns, SuperTrend outcome, MFE/MAE) is printed for context but is NEVER
read by the selection code below. This is the hindsight Arun explicitly forbade.
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / 'results'
CSV_IN = RES / 'rounding_base_events.csv'
OUT = RES / 'summary.txt'

MODE_PREF = {'VAR': 0, 'L250': 1, 'L180': 2, 'L120': 3}


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
    p('research/159 — ROUNDING BASE (SAUCER) + VOLUME ACCUMULATION -> RIM BREAKOUT')
    p('Causal detector event summary.  Source: %s' % CSV_IN.name)
    p('=' * 100)
    p('raw events (symbol x window-mode): %d' % len(rows))

    # ---- de-duplicate: one row per (symbol, breakout_date); keep the best mode -------
    best = {}
    for r in rows:
        k = (r['symbol'], r['breakout_date'])
        cur = best.get(k)
        if cur is None or MODE_PREF[r['mode']] < MODE_PREF[cur['mode']]:
            best[k] = r
    ded = sorted(best.values(), key=lambda r: r['breakout_date'])
    p('de-duplicated events (symbol x breakout date): %d' % len(ded))
    p('distinct symbols: %d' % len({r['symbol'] for r in ded}))

    obv = [r for r in ded if r['obv_filter_pass'] == '1']
    p('of which pass the OBV/volume accumulation filter (OBV_ON): %d  (%.0f%%)'
      % (len(obv), 100.0 * len(obv) / max(len(ded), 1)))
    p('ipo_short_window events (young listings, declared exception): %d'
      % sum(1 for r in ded if r['ipo_short_window'] == '1'))

    # ---- per-year counts -------------------------------------------------------------
    p('')
    p('--- EVENTS PER YEAR (de-duplicated) ---')
    cy = Counter(r['breakout_date'][:4] for r in ded)
    cyo = Counter(r['breakout_date'][:4] for r in obv)
    p('%-6s %8s %8s   %s' % ('year', 'all', 'OBV_ON', ''))
    for y in sorted(cy):
        p('%-6s %8d %8d   %s' % (y, cy[y], cyo[y], '#' * min(cy[y], 60)))
    p('%-6s %8d %8d' % ('TOTAL', sum(cy.values()), sum(cyo.values())))
    yrs = len(cy)
    p('mean %.1f events/year across %d years with at least one event' % (sum(cy.values()) / yrs, yrs))

    p('')
    p('--- BY WINDOW MODE (raw, before de-dup) ---')
    for m, n in Counter(r['mode'] for r in rows).most_common():
        p('  %-5s %d' % (m, n))

    # ---- KMEW replication check ------------------------------------------------------
    p('')
    p('--- KMEW REPLICATION CHECK (Arun\'s own example; expected breakout ~Sep-2025) ---')
    km = [r for r in rows if r['symbol'] == 'KMEW']
    if not km:
        p('  *** FAIL — detector produced NO event for KMEW ***')
    for r in sorted(km, key=lambda r: MODE_PREF[r['mode']]):
        p('  mode=%-5s q=%s  rim=%s @ %s  trough=%s @ %s  depth=%s%%  baselen=%s'
          % (r['mode'], r['base_qualify_date'], r['rim_level'], r['rim_date'],
             r['trough_close'], r['trough_date'], r['depth_pct'], r['base_len_bars']))
        p('        BREAKOUT %s close %s | fill(a) next open %s on %s | fill(b) buy-stop %s'
          % (r['breakout_date'], r['breakout_close'], r['fill_a_nextopen'],
             r['fill_a_nextopen_date'], r['fill_b_buystop']))
        p('        R2=%s vertex=%s flat=%s volratio=%s obv_pass=%s ipo_short=%s score=%s'
          % (r['fit_r2'], r['vertex_frac'], r['flat_frac'], r['vol_ratio'],
             r['obv_filter_pass'], r['ipo_short_window'], r['pattern_quality']))
        p('        [info only] fwd250=%s%%  ST(7,3) exit %s ret %s%% after %s bars'
          % (r['info_fwd250_pct'], r['info_st73_exit_date'], r['info_st73_ret_pct'],
             r['info_st73_bars_held']))

    # ---- fill mechanic: does the choice matter? --------------------------------------
    gaps = [f(r['gap_pct_a_vs_rim']) for r in ded if f(r['gap_pct_a_vs_rim']) is not None]
    if gaps:
        gaps.sort()
        p('')
        p('--- FILL MECHANIC: next-day open vs the frozen rim (%d events) ---' % len(gaps))
        p('  median %+.2f%%   p10 %+.2f%%   p90 %+.2f%%   (positive = open above the rim)'
          % (gaps[len(gaps) // 2], gaps[int(0.1 * len(gaps))], gaps[int(0.9 * len(gaps))]))

    # ---- SHORT-LIST ------------------------------------------------------------------
    # Pre-registered selection rule, uses pattern_quality ONLY:
    #   pool  = de-duplicated events that pass OBV accumulation, have a real wait between
    #           the base-qualify day and the breakout (days_q_to_breakout >= 1), have an
    #           actual fill, and are old enough that a chart shows the aftermath.
    #   pick  = greedily by pattern_quality, at most one per symbol, forcing a different
    #           calendar year and a different liquidity bucket for each pick.
    p('')
    p('--- SHORT-LIST FOR MANUAL VERIFICATION (ranked by pattern_quality ONLY) ---')

    def bucket(r):
        tv = f(r['tv20_cr_at_breakout'], 0.0) or 0.0
        return 'small(<Rs10cr)' if tv < 10 else ('mid(Rs10-50cr)' if tv < 50 else 'large(>=Rs50cr)')

    pool = [r for r in ded
            if r['obv_filter_pass'] == '1'
            and int(r['days_q_to_breakout'] or 0) >= 1
            and f(r['fill_a_nextopen']) is not None
            and r['breakout_date'] < '2025-09-01']
    bysym = {}
    for r in pool:
        k = r['symbol']
        if k not in bysym or f(r['pattern_quality'], 0) > f(bysym[k]['pattern_quality'], 0):
            bysym[k] = r
    ranked = sorted(bysym.values(), key=lambda r: -f(r['pattern_quality'], 0))
    p('eligible pool: %d events / %d distinct symbols' % (len(pool), len(bysym)))

    pgaps = sorted(x for x in (f(r['gap_pct_a_vs_rim']) for r in pool) if x is not None)
    if pgaps:
        p('  fill gap WITHIN this clean pool: median %+.2f%%  p10 %+.2f%%  p90 %+.2f%%'
          % (pgaps[len(pgaps) // 2], pgaps[int(0.1 * len(pgaps))], pgaps[int(0.9 * len(pgaps))]))

    picks, used_yr, used_bk = [], set(), set()
    for r in ranked:
        y, b = r['breakout_date'][:4], bucket(r)
        if y in used_yr or b in used_bk:
            continue
        picks.append(r); used_yr.add(y); used_bk.add(b)
        if len(picks) == 3:
            break
    for i, r in enumerate(picks, 1):
        p('')
        p('  [%d] %s   mode=%s   pattern_quality=%s   liquidity=%s'
          % (i, r['symbol'], r['mode'], r['pattern_quality'], bucket(r)))
        p('      left rim   : %s  @ Rs %s' % (r['rim_date'], r['rim_level']))
        p('      trough     : %s  @ Rs %s   (depth %s%%, base %s bars, trough at %s of base)'
          % (r['trough_date'], r['trough_close'], r['depth_pct'],
             r['base_len_bars'], r['trough_pos']))
        p('      base qualif: %s  (close first back within 5%% of the rim; rim frozen here)'
          % r['base_qualify_date'])
        p('      BREAKOUT   : %s  close Rs %s   (%s bars after qualify)'
          % (r['breakout_date'], r['breakout_close'], r['days_q_to_breakout']))
        p('      entry fills: (a) next-day open Rs %s on %s   (b) buy-stop at rim Rs %s'
          % (r['fill_a_nextopen'], r['fill_a_nextopen_date'], r['fill_b_buystop']))
        p('      shape      : R2=%s  curvature=%s  vertex=%s  no-V=%s  vol_ratio=%s  obv_gain=%s'
          % (r['fit_r2'], r['fit_curvature'], r['vertex_frac'], r['flat_frac'],
             r['vol_ratio'], r['obv_gain']))
        p('      [INFO ONLY, not used to select] fwd60=%s%% fwd120=%s%% fwd250=%s%% | '
          'ST(7,3) exit %s ret %s%% bars %s | MFE250 %s%% MAE250 %s%%'
          % (r['info_fwd60_pct'], r['info_fwd120_pct'], r['info_fwd250_pct'],
             r['info_st73_exit_date'] or 'still open', r['info_st73_ret_pct'],
             r['info_st73_bars_held'], r['info_mfe_250_pct'], r['info_mae_250_pct']))

    # ---- ADDENDUM: longest-base example ------------------------------------------
    # Arun's own two examples had ~6 and ~9 month bases (~120 and ~190 trading bars).
    # The pre-registered rule above does not constrain base length, and its top pick came
    # in at 91 bars. This ADDITIONAL pick is the highest pattern_quality event in the same
    # eligible pool with a base of >= 150 bars, so he also has a long saucer to eyeball.
    # Still ranked by pattern_quality only; no info_* column is consulted.
    longs = [r for r in ranked if int(r['base_len_bars']) >= 150
             and r['symbol'] not in {q['symbol'] for q in picks}]
    if longs:
        r = longs[0]
        p('')
        p('--- ADDENDUM: longest-base example (base >= 150 bars, same blind ranking) ---')
        p('  [4] %s   mode=%s   pattern_quality=%s   liquidity=%s'
          % (r['symbol'], r['mode'], r['pattern_quality'], bucket(r)))
        p('      left rim   : %s  @ Rs %s' % (r['rim_date'], r['rim_level']))
        p('      trough     : %s  @ Rs %s   (depth %s%%, base %s bars, trough at %s of base)'
          % (r['trough_date'], r['trough_close'], r['depth_pct'],
             r['base_len_bars'], r['trough_pos']))
        p('      base qualif: %s' % r['base_qualify_date'])
        p('      BREAKOUT   : %s  close Rs %s   (%s bars after qualify)'
          % (r['breakout_date'], r['breakout_close'], r['days_q_to_breakout']))
        p('      entry fills: (a) next-day open Rs %s on %s   (b) buy-stop at rim Rs %s'
          % (r['fill_a_nextopen'], r['fill_a_nextopen_date'], r['fill_b_buystop']))
        p('      shape      : R2=%s  vertex=%s  no-V=%s  vol_ratio=%s  obv_gain=%s  tv20=%s cr'
          % (r['fit_r2'], r['vertex_frac'], r['flat_frac'], r['vol_ratio'],
             r['obv_gain'], r['tv20_cr_at_breakout']))
        p('      [INFO ONLY] fwd60=%s%% fwd120=%s%% fwd250=%s%% | ST(7,3) exit %s ret %s%% bars %s'
          % (r['info_fwd60_pct'], r['info_fwd120_pct'], r['info_fwd250_pct'],
             r['info_st73_exit_date'] or 'still open', r['info_st73_ret_pct'],
             r['info_st73_bars_held']))

    # ---- runners-up, for context -----------------------------------------------------
    p('')
    p('--- next 10 by pattern_quality (context only) ---')
    for r in ranked[:14]:
        p('  %-14s %s  score %s  depth %s%%  R2 %s  base %s bars  tv20 %s cr'
          % (r['symbol'], r['breakout_date'], r['pattern_quality'], r['depth_pct'],
             r['fit_r2'], r['base_len_bars'], r['tv20_cr_at_breakout']))

    # ---- honest context on the information columns -----------------------------------
    import math
    vals = [f(r['info_fwd250_pct']) for r in ded]
    vals = [x for x in vals if x is not None and math.isfinite(x)]
    if vals:
        vals.sort()
        p('')
        p('--- INFORMATION ONLY: distribution of +250-bar return from fill (a), all %d events ---'
          % len(vals))
        p('  This is NOT a backtest: no costs, no taxes, no position sizing, no slot limit,')
        p('  no market gate, and it is measured on a survivorship-biased universe.')
        p('  median %+.1f%%  mean %+.1f%%  p10 %+.1f%%  p90 %+.1f%%  frac>0 %.0f%%'
          % (vals[len(vals) // 2], sum(vals) / len(vals), vals[int(0.1 * len(vals))],
             vals[int(0.9 * len(vals))], 100.0 * sum(1 for x in vals if x > 0) / len(vals)))

    OUT.write_text('\n'.join(out), encoding='utf-8')
    print('\nwrote %s' % OUT)


if __name__ == '__main__':
    main()
