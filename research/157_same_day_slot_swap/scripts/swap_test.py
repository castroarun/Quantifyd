# -*- coding: utf-8 -*-
"""Does a slot freed by today's exit have to wait until tomorrow?

r/142's per-bar loop runs ENTRIES then EXITS, so a name that stops out at today's close
frees its slot only for tomorrow. Nothing documents that as a decision. This runs the same
book both ways off one shared set of frames, so the only difference between the two arms is
the order of two blocks.

FIDELITY CONTROL FIRST: the forked simulate, run in control order, must reproduce the
untouched bluesky_replay.simulate exactly. A fork that has drifted would make every other
number here meaningless, so the study refuses to continue if the curves differ.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
STUDY = ROOT / 'research' / '142_bananapatterns_replication' / 'scripts'
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(ROOT))

import bluesky_replay as br            # noqa: E402  (path set above)

START, END = '2020-01-01', '2026-09-05'
RS_MIN, TRAIL_SMA = 70, 15             # the adopted live spec: trail-15, 16 slots, no gate
SLOTS, SIZE_PCT, STOP, COST = 16, 0.0625, 0.08, 0.0025


def simulate2(exits_first, days_idx, dates, C, H, O, ATH, S50, RS, TVp, TRIG, weak_arr,
              cost=COST, stop=STOP, slots=SLOTS, size_pct=SIZE_PCT, fill_close=False):
    """A fork of br.simulate whose ONLY addition is the block-order switch.

    Kept deliberately close to the original, line for line, so the diff that matters is
    visible: `exits_first` decides whether do_exits() runs before or after do_entries().
    Selection is RS-ranked and deterministic, so there is no rng and no seed.
    """
    cash = float(br.CAPITAL)
    positions = []            # (col, entry_i, buy, qty)
    trades = []
    equity = np.empty(len(days_idx), dtype=float)
    passed_up = 0

    for k, i in enumerate(days_idx):

        def do_entries():
            nonlocal cash, positions, passed_up
            if weak_arr[i]:
                return
            cand = np.nonzero(TRIG[i])[0]
            if not len(cand):
                return
            mtm = sum(q * (C[i, c] if not np.isnan(C[i, c]) else b)
                      for c, _, b, q in positions)
            eq = cash + mtm
            cand = cand[np.argsort(-np.nan_to_num(RS[i, cand]))]      # sel='rs'
            for c in cand:
                if len(positions) >= slots:
                    passed_up += 1
                    continue
                piv = float(ATH[i, c])
                # An exit happens at the CLOSE. Buying the replacement at the OPEN of
                # that same bar spends money the book has not got yet and fills a slot
                # that is still occupied -- look-ahead, worth +12.6pp of pure foresight
                # when I first ran it. A same-day swap has to fill at the close.
                fill = float(C[i, c]) if fill_close else max(piv, float(O[i, c]))
                if not np.isfinite(fill) or fill <= 0:
                    passed_up += 1
                    continue
                size = size_pct * eq
                qty = int(size / fill)
                if qty < 1 or cash < qty * fill * (1 + cost):
                    passed_up += 1
                    continue
                cash -= qty * fill * (1 + cost)
                positions.append((c, i, fill, qty))

        def do_exits():
            nonlocal cash, positions
            still = []
            for c, ei, b, q in positions:
                cl = C[i, c]
                if np.isnan(cl):
                    still.append((c, ei, b, q))
                    continue
                reason = None
                if cl <= b * (1 - stop):
                    reason = 'stop_8pct'
                elif i > ei and not np.isnan(S50[i, c]) and cl < S50[i, c]:
                    reason = 'trail_50d'
                if reason:
                    cash += q * float(cl) * (1 - cost)
                    trades.append((c, ei, i, b, float(cl), reason))
                else:
                    still.append((c, ei, b, q))
            positions = still

        if exits_first:
            do_exits()
            do_entries()
        else:
            do_entries()
            do_exits()

        mtm = sum(q * (C[i, c] if not np.isnan(C[i, c]) else b)
                  for c, _, b, q in positions)
        equity[k] = cash + mtm

    last = days_idx[-1]
    for c, ei, b, q in positions:
        cl = C[last, c]
        trades.append((c, ei, last, b, float(cl) if not np.isnan(cl) else b, 'open_marked'))
    return equity, trades, passed_up


def build():
    """Frames and signals, exactly as bluesky_replay.main builds them."""
    base_start = (pd.Timestamp(START) - pd.Timedelta(days=550)).strftime('%Y-%m-%d')
    w = br.load_frames(base_start, trail_sma=TRAIL_SMA)
    close, high, open_, athcp, sma50, tv20 = (w[k] for k in
                                              ('close', 'high', 'open', 'athcp', 'sma50', 'tv20'))
    etf_cols = [c for c in close.columns if br.ETF_RE.search(c)]
    tv_prev = tv20.shift(1)
    prev_close = close.shift(1)
    eligible = tv_prev >= br.TV_FLOOR
    eligible[etf_cols] = False

    r63 = close / close.shift(63) - 1
    r126 = close / close.shift(126) - 1
    r189 = close / close.shift(189) - 1
    r252 = close / close.shift(252) - 1
    score = (2 * r63 + r126 + r189 + r252).where(eligible)
    rs = (score.rank(axis=1, pct=True) * 100).shift(1)

    setup = (prev_close < athcp) & (prev_close >= 0.8 * athcp) & eligible & (rs >= RS_MIN)
    trig = setup & (close > athcp) & athcp.notna()

    dates = close.index
    days_idx = np.array([i for i, d in enumerate(dates) if START <= str(d.date()) <= END])
    return (days_idx, dates, close.values, high.values, open_.values, athcp.values,
            sma50.values, rs.values, tv_prev.values, trig.fillna(False).values,
            np.zeros(len(dates), dtype=bool), dates[days_idx])


def main():
    t0 = time.time()
    print('building frames...', flush=True)
    (days_idx, dates, C, H, O, ATH, S50, RS, TVp, TRIG, weak, used) = build()
    print('%d trading days, %d signals (%.0fs)'
          % (len(days_idx), int(TRIG[days_idx].sum()), time.time() - t0), flush=True)

    # ---- fidelity control: the fork must reproduce the untouched original ----
    print('\ncontrol: forked simulate vs bluesky_replay.simulate ...', flush=True)
    eq_ref, tr_ref, _ = br.simulate(0, 'rs', days_idx, dates, C, H, O, ATH, S50, RS, TVp,
                                    TRIG, weak, True, COST, stop=STOP, slots=SLOTS,
                                    size_pct=SIZE_PCT)
    eq_a, tr_a, pa = simulate2(False, days_idx, dates, C, H, O, ATH, S50, RS, TVp, TRIG, weak)
    same = np.allclose(eq_ref, eq_a, rtol=0, atol=1e-6) and len(tr_ref) == len(tr_a)
    print('   equity identical: %s | trades %d vs %d' % (same, len(tr_ref), len(tr_a)),
          flush=True)
    if not same:
        print('\nFORK IS NOT FAITHFUL - stopping. Nothing below would mean anything.')
        print('   ref final %.0f   fork final %.0f' % (eq_ref[-1], eq_a[-1]))
        raise SystemExit(2)

    # ---- the INVALID pair, kept because the gap to the honest pair IS the lesson ----
    print(chr(10) + '[INVALID, look-ahead] exits-first, entries still filling at the OPEN',
          flush=True)
    eq_x, tr_x, _ = simulate2(True, days_idx, dates, C, H, O, ATH, S50, RS, TVp, TRIG, weak)
    sx, _ = br.stats_from(eq_x, used, tr_x, br.CAPITAL)
    s_ctl, _ = br.stats_from(eq_a, used, tr_a, br.CAPITAL)
    print('   CAGR %.2f%% vs control %.2f%% -- FORESIGHT, not edge: it sells at the close '
          'and spends the proceeds at that morning open.' % (sx['cagr'], s_ctl['cagr']),
          flush=True)

    # ---- the honest pair: both fill at the CLOSE, ordering the only difference ----
    print(chr(10) + 'honest pair, both filling at the CLOSE ...', flush=True)
    eq_a, tr_a, pa = simulate2(False, days_idx, dates, C, H, O, ATH, S50, RS, TVp, TRIG,
                               weak, fill_close=True)
    eq_b, tr_b, pb = simulate2(True, days_idx, dates, C, H, O, ATH, S50, RS, TVp, TRIG,
                               weak, fill_close=True)

    sa, ea = br.stats_from(eq_a, used, tr_a, br.CAPITAL)
    sb, eb = br.stats_from(eq_b, used, tr_b, br.CAPITAL)

    print('\n%-26s %14s %14s %10s' % ('', 'A next-day', 'B same-day', 'B - A'))
    for k, lbl, f in (('cagr', 'CAGR %', '%.2f'), ('dd', 'Max drawdown %', '%.2f'),
                      ('n', 'Trades', '%d'), ('win', 'Win rate %', '%.1f'),
                      ('mean', 'Mean per trade %', '%.2f'), ('x', 'Growth multiple', '%.2f')):
        va, vb = sa[k], sb[k]
        d = ('%+.2f' % (vb - va)) if k != 'n' else ('%+d' % (vb - va))
        print('%-26s %14s %14s %10s' % (lbl, f % va, f % vb, d))
    ca = sa['cagr'] / abs(sa['dd']) if sa['dd'] else 0
    cb = sb['cagr'] / abs(sb['dd']) if sb['dd'] else 0
    print('%-26s %14.2f %14.2f %10s' % ('Calmar', ca, cb, '%+.2f' % (cb - ca)))
    print('%-26s %14d %14d' % ('Signals passed up', pa, pb))

    print('\nper year (net %%):\n%-6s %10s %10s %8s' % ('year', 'A', 'B', 'B-A'))
    yrs = sorted(set(sa['yearly']) | set(sb['yearly']))
    for y in yrs:
        va, vb = sa['yearly'].get(y), sb['yearly'].get(y)
        print('%-6s %10s %10s %8s' % (y, va, vb,
              ('%+.1f' % (vb - va)) if (va is not None and vb is not None) else '-'))
    wins = sum(1 for y in yrs if sa['yearly'].get(y) is not None
               and sb['yearly'].get(y, 0) > sa['yearly'][y])
    print('\nB beat A in %d of %d years' % (wins, len(yrs)))

    out = ROOT / 'research' / '157_same_day_slot_swap' / 'results'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'A_next_day': pd.Series(eq_a, index=used),
                  'B_same_day': pd.Series(eq_b, index=used)}).to_csv(out / 'curves.csv')
    print('\ncurves -> %s (%.0fs total)' % (out / 'curves.csv', time.time() - t0))


if __name__ == '__main__':
    main()
