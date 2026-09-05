"""P1/P2 — VCP entry-pivot fingerprint + exit-convention forensics (research/151).

For each of the 40 ground-truth trades from bananapatterns.com's VCP screen:

ENTRY:
  * data sanity: is `buy` inside the entry day's [low, high]?  scale ratio close/buy
  * pivot search: max(high) and max(close) over window w ending d bars before entry,
    for w in WINDOWS, d in GAPS  ->  which (w, basis, d) reproduces `buy` exactly?
  * swing-high search: last confirmed k-bar fractal swing high strictly before entry
    (k in 2,3,5,7), and the highest such swing high in the last L bars
  * ATH-close / ATH-high anchors (the Blue Sky pivot, for contrast)

EXIT:
  * replay stop (7% and 8%, intraday-touch vs close-basis) x trail-50 (exit at the
    signal close / next open / next close) -> which convention reproduces their
    (exit_date, sell) on the 23 closed trades

Read-only. Output: results/p1_entry_fingerprint.csv, results/p1_exit_conventions.csv
and a console summary.
"""
import csv
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
DB = ROOT / 'backtest_data' / 'market_data.db'
GT = STUDY / 'data' / 'vcp_trades_groundtruth.csv'

WINDOWS = [5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 150, 200, 252, 9999]
GAPS = [1, 2, 3, 4, 5, 6, 8, 10]
FRACTALS = [2, 3, 5, 7]
EXACT = 0.0015          # 0.15% == "the same level"
NEAR = 0.010            # 1.0% == "matched"


def load_bars(db, sym):
    df = pd.read_sql_query(
        "select date, open, high, low, close, volume from market_data_unified "
        "where symbol=? and timeframe='day' order by date", db, params=(sym,))
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'].str[:10])
    df = df.drop_duplicates('date').set_index('date').sort_index()
    return df


def fractal_swing_highs(high, k):
    """Indices i where high[i] == max(high[i-k .. i+k]). Confirmed only at i+k."""
    n = len(high)
    out = []
    for i in range(k, n - k):
        seg = high[i - k:i + k + 1]
        if high[i] >= seg.max():
            out.append(i)
    return out


def main():
    db = sqlite3.connect(str(DB))
    gt = list(csv.DictReader(open(GT)))
    rows = []
    exit_rows = []
    missing = []

    for t in gt:
        s, ed_s = t['symbol'], t['entry_date']
        buy = float(t['buy'])
        df = load_bars(db, s)
        if df.empty:
            missing.append((s, 'no_data'))
            continue
        ed = pd.Timestamp(ed_s)
        if ed not in df.index:
            missing.append((s, f'no_bar_{ed_s}'))
            continue
        i = df.index.get_loc(ed)
        eday = df.iloc[i]
        hist = df.iloc[:i]
        if len(hist) < 60:
            missing.append((s, 'short_history'))
            continue

        inside = float(eday['low']) <= buy <= float(eday['high'])
        scale = float(eday['close']) / buy

        # --- grid pivot search
        best = None
        exact_hits = []
        H = hist['high'].values
        C = hist['close'].values
        for d in GAPS:
            if d > len(hist):
                break
            endH = H[:len(H) - (d - 1)]
            endC = C[:len(C) - (d - 1)]
            for w in WINDOWS:
                for basis, arr in (('H', endH), ('C', endC)):
                    seg = arr[-w:] if w < len(arr) else arr
                    if not len(seg):
                        continue
                    piv = float(seg.max())
                    diff = abs(buy - piv) / piv
                    if diff <= EXACT:
                        exact_hits.append((w, basis, d, piv, diff))
                    if best is None or diff < best[4]:
                        best = (w, basis, d, piv, diff)

        # --- fractal swing highs
        fr = {}
        for k in FRACTALS:
            idxs = [j for j in fractal_swing_highs(H, k) if j + k < len(H)]
            if idxs:
                last_sh = float(H[idxs[-1]])
                # highest confirmed swing high in the last 120 bars
                recent = [H[j] for j in idxs if j >= len(H) - 120]
                base_sh = float(max(recent)) if recent else np.nan
                fr[k] = (last_sh, base_sh,
                         abs(buy - last_sh) / last_sh,
                         abs(buy - base_sh) / base_sh if recent else np.nan)
            else:
                fr[k] = (np.nan, np.nan, np.nan, np.nan)

        ath_h = float(H.max())
        ath_c = float(C.max())
        prev = hist.iloc[-1]

        rows.append(dict(
            symbol=s, entry_date=ed_s, buy=buy,
            inside_day_range=inside, scale_ratio=round(scale, 4),
            day_open=float(eday['open']), day_high=float(eday['high']),
            day_low=float(eday['low']), day_close=float(eday['close']),
            prev_close=float(prev['close']), prev_high=float(prev['high']),
            open_above_buy=float(eday['open']) > buy,
            best_w=best[0], best_basis=best[1], best_gap=best[2],
            best_pivot=round(best[3], 2), best_diff_pct=round(best[4] * 100, 4),
            n_exact_hits=len(exact_hits),
            exact_min_w=min([h[0] for h in exact_hits]) if exact_hits else '',
            exact_bases=''.join(sorted({h[1] for h in exact_hits})) if exact_hits else '',
            fr3_last=round(fr[3][0], 2), fr3_last_diff=round(fr[3][2] * 100, 3),
            fr3_base=round(fr[3][1], 2), fr3_base_diff=round(fr[3][3] * 100, 3),
            fr5_last=round(fr[5][0], 2), fr5_last_diff=round(fr[5][2] * 100, 3),
            fr5_base=round(fr[5][1], 2), fr5_base_diff=round(fr[5][3] * 100, 3),
            vs_ath_high_pct=round((buy - ath_h) / ath_h * 100, 2),
            vs_ath_close_pct=round((buy - ath_c) / ath_c * 100, 2),
        ))

        # --- exit conventions (closed trades only)
        if t['status'] == 'closed' and t['exit_date']:
            true_xd = pd.Timestamp(t['exit_date'])
            true_px = float(t['sell'])
            sma50 = df['close'].rolling(50).mean()
            fwd = df.iloc[i:]
            for stop_pct in (0.07, 0.08):
                for smode in ('I', 'C'):
                    for tmode in ('SC', 'NO', 'NC'):
                        stop_px = buy * (1 - stop_pct)
                        xd = xpx = reason = None
                        dates = list(fwd.index)
                        for j, d in enumerate(dates):
                            r = fwd.iloc[j]
                            if smode == 'I' and r['low'] <= stop_px:
                                xd, xpx, reason = d, stop_px, 'stop'
                                break
                            if smode == 'C' and r['close'] <= stop_px:
                                xd, xpx, reason = d, float(r['close']), 'stop'
                                break
                            sv = sma50.get(d, np.nan)
                            if j > 0 and pd.notna(sv) and r['close'] < sv:
                                if tmode == 'SC':
                                    xd, xpx, reason = d, float(r['close']), 'trail'
                                elif tmode == 'NO' and j + 1 < len(dates):
                                    xd, xpx, reason = dates[j + 1], float(fwd.iloc[j + 1]['open']), 'trail'
                                elif tmode == 'NC' and j + 1 < len(dates):
                                    xd, xpx, reason = dates[j + 1], float(fwd.iloc[j + 1]['close']), 'trail'
                                if xd is not None:
                                    break
                        exit_rows.append(dict(
                            symbol=s, entry_date=ed_s, stop_pct=stop_pct,
                            stop_mode=smode, trail_mode=tmode,
                            true_exit=t['exit_date'], true_px=true_px,
                            got_exit=str(xd.date()) if xd is not None else '',
                            got_px=round(xpx, 2) if xpx else '',
                            date_match=(xd is not None and xd == true_xd),
                            px_match=(xpx is not None and abs(xpx - true_px) / true_px <= NEAR),
                            reason=reason or ''))

    db.close()
    edf = pd.DataFrame(rows)
    edf.to_csv(STUDY / 'results' / 'p1_entry_fingerprint.csv', index=False)
    xdf = pd.DataFrame(exit_rows)
    xdf.to_csv(STUDY / 'results' / 'p1_exit_conventions.csv', index=False)

    print(f'\n=== P1 ENTRY FINGERPRINT — {len(edf)}/{len(gt)} trades with usable data ===')
    if missing:
        print('MISSING:', missing)
    print(f'buy inside entry-day range: {int(edf.inside_day_range.sum())}/{len(edf)}')
    print(f'scale ratio outside 0.8-1.3 (split defect?): '
          f'{list(edf.loc[(edf.scale_ratio<0.8)|(edf.scale_ratio>1.3), "symbol"])}')
    print(f'entry-day OPEN above buy (fill inflation if booked at pivot): '
          f'{int(edf.open_above_buy.sum())}/{len(edf)}')
    print(f'\nbest-pivot diff distribution (%):')
    print(edf.best_diff_pct.describe().round(3).to_string())
    print(f'exact (<={EXACT*100:.2f}%) reproduced by SOME (w,basis,gap): '
          f'{int((edf.best_diff_pct<=EXACT*100).sum())}/{len(edf)}')
    print(f'within 1%: {int((edf.best_diff_pct<=1.0).sum())}/{len(edf)}')
    print('\nbest (w, basis) frequency among trades matched <=0.15%:')
    ok = edf[edf.best_diff_pct <= EXACT * 100]
    if len(ok):
        print(ok.groupby(['best_w', 'best_basis']).size().sort_values(ascending=False).head(15).to_string())
    print('\nfractal-swing-high match rates:')
    for k, col in ((3, 'fr3'), (5, 'fr5')):
        for kind in ('last', 'base'):
            c = f'{col}_{kind}_diff'
            print(f'  k={k} {kind:4s}: exact {int((edf[c]<=EXACT*100).sum()):2d}/{len(edf)}  '
                  f'<=1% {int((edf[c]<=1.0).sum()):2d}/{len(edf)}  median {edf[c].median():.2f}%')
    print('\nvs ATH (blue-sky pivot) — median buy vs prior ATH high %: '
          f'{edf.vs_ath_high_pct.median():.2f}, close %: {edf.vs_ath_close_pct.median():.2f}')
    print(f'  trades at/above prior ATH close: {int((edf.vs_ath_close_pct>=-0.15).sum())}/{len(edf)}')

    if len(xdf):
        print('\n=== P2 EXIT CONVENTIONS (closed trades) ===')
        g = xdf.groupby(['stop_pct', 'stop_mode', 'trail_mode']).agg(
            n=('date_match', 'size'), date_hits=('date_match', 'sum'),
            px_hits=('px_match', 'sum')).reset_index()
        g['both'] = xdf.groupby(['stop_pct', 'stop_mode', 'trail_mode']).apply(
            lambda d: int((d.date_match & d.px_match).sum())).values
        print(g.sort_values('both', ascending=False).to_string(index=False))


if __name__ == '__main__':
    main()
