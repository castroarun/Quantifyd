"""
Phase-1 forensic replay: do BananaPatterns' published trades reproduce from their
stated rules on our data?

For each ground-truth trade (data/trades_groundtruth.csv, transcribed from Arun's
screenshots of bananapatterns.com):

ENTRY — "At the pivot" (buy-stop at the pivot; screen says pivot = the ATH):
  - buy price vs prior highs (ATH / 252d / 100d / 50d / 20d, strictly before entry
    date): which definition matches to <=0.25%?
  - did the entry day actually trade through the pivot (high >= buy)?
  - fill realism: open > buy means a real buy-stop fills at the open, not the pivot
  - distance from ATH the day before entry (screen: "within 20% of the pivot")
  - 126d return as a crude RS proxy (universe percentile deferred to phase 2)

EXIT — "-8% cut" + "trail 50-day" replayed under convention grid:
  stop:  I = intraday touch (low <= buy*0.92, fill at stop px)
         C = close basis   (close <= buy*0.92, fill at close)
  trail: SC = exit at the close that broke the 50-SMA
         NO = exit next day's open
         NC = exit next day's close
  -> which combo reproduces their (exit_date, sell)? per-trade deltas for each.

Read-only on the DB. Output: results/trade_match.csv + console summary.
"""
import csv
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]          # .../quantifyd
STUDY = Path(__file__).resolve().parents[1]         # .../137_bananapatterns_replication
DB = ROOT / 'backtest_data' / 'market_data.db'
GT_NAME = sys.argv[1] if len(sys.argv) > 1 else 'trades_groundtruth.csv'
GT_CSV = STUDY / 'data' / GT_NAME
OUT_CSV = STUDY / 'results' / f"trade_match_{GT_NAME.replace('trades_groundtruth', 'gt').replace('.csv', '')}.csv"

PRICE_TOL = 0.010      # 1.0% = "match"; we also report the raw delta
PIVOT_TOL = 0.0025     # 0.25% to call the buy price "equal to" a prior high
STOP_PCT = 0.08

STOP_MODES = ['I', 'C']
TRAIL_MODES = ['SC', 'NO', 'NC']


def load_bars(db, symbol):
    q = ("select date, open, high, low, close, volume from market_data_unified "
         "where symbol=? and timeframe='day' order by date")
    df = pd.read_sql_query(q, db, params=(symbol,))
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'].str[:10])
    df = df.drop_duplicates('date').set_index('date').sort_index()
    df['sma50'] = df['close'].rolling(50).mean()
    return df


def replay_exit(bars, entry_date, buy, stop_mode, trail_mode):
    """Walk forward from entry date; first of (-8% stop, 50-SMA trail) wins.
    Returns (exit_date, exit_px, reason) or (None, None, 'no_exit')."""
    stop_px = buy * (1 - STOP_PCT)
    fwd = bars.loc[bars.index >= entry_date]
    dates = list(fwd.index)
    for i, d in enumerate(dates):
        row = fwd.iloc[i]
        # stop check (skip pathological same-day stop only if pivot > stop, always true)
        if stop_mode == 'I' and row['low'] <= stop_px:
            # on the entry day a fill at the pivot then a slide to -8% intraday is
            # possible; honour it
            return d, stop_px, 'stop_8pct'
        if stop_mode == 'C' and row['close'] <= stop_px:
            return d, row['close'], 'stop_8pct'
        # trail: close below 50-SMA (need SMA available)
        if i > 0 and pd.notna(row['sma50']) and row['close'] < row['sma50']:
            if trail_mode == 'SC':
                return d, row['close'], 'trail_50d'
            if i + 1 < len(dates):
                nxt = fwd.iloc[i + 1]
                px = nxt['open'] if trail_mode == 'NO' else nxt['close']
                return dates[i + 1], px, 'trail_50d'
            return d, row['close'], 'trail_50d'
    return None, None, 'no_exit'


def pct(a, b):
    return (a - b) / b * 100.0 if (a is not None and b) else None


def main():
    db = sqlite3.connect(str(DB))
    gt = list(csv.DictReader(open(GT_CSV)))
    print(f'DB: {DB}')
    print(f'{len(gt)} ground-truth trades\n')

    # ---- coverage check (playbook §3) ----
    syms = sorted({t['symbol'] for t in gt})
    missing = []
    print('--- coverage ---')
    for s in syms:
        r = db.execute("select min(date),max(date),count(*) from market_data_unified "
                       "where symbol=? and timeframe='day'", (s,)).fetchone()
        print(f'{s:12s} {r[0]} .. {r[1]}  rows={r[2]}')
        if not r[2]:
            missing.append(s)
    if missing:
        print(f'\nMISSING SYMBOLS ({len(missing)}): {missing} — reported, not dropped\n')

    bars_cache = {s: load_bars(db, s) for s in syms if s not in missing}

    rows_out = []
    combo_hits = {(sm, tm): 0 for sm in STOP_MODES for tm in TRAIL_MODES}
    n_closed = 0

    for t in gt:
        s = t['symbol']
        out = {k: t[k] for k in ('symbol', 'entry_date', 'exit_date', 'buy', 'sell',
                                 'return_pct', 'exit_reason', 'status')}
        if s in missing or bars_cache[s].empty:
            out['note'] = 'NO DATA'
            rows_out.append(out)
            continue
        bars = bars_cache[s]
        ed = pd.Timestamp(t['entry_date'])
        buy = float(t['buy'])

        if ed not in bars.index:
            near = bars.index[bars.index.searchsorted(ed):][:1]
            out['note'] = f'entry date not in data (next bar: {near[0].date() if len(near) else "none"})'
            rows_out.append(out)
            continue

        # ---- entry analysis ----
        hist = bars.loc[bars.index < ed]
        eday = bars.loc[ed]
        if len(hist) < 20:
            out['note'] = f'only {len(hist)} bars before entry'
        piv = {
            'ATH': hist['high'].max() if len(hist) else None,
            'H252': hist['high'].tail(252).max() if len(hist) else None,
            'H100': hist['high'].tail(100).max() if len(hist) else None,
            'H50': hist['high'].tail(50).max() if len(hist) else None,
            'H20': hist['high'].tail(20).max() if len(hist) else None,
        }
        for k, v in piv.items():
            out[f'buy_vs_{k}_pct'] = round(pct(buy, v), 3) if v else None
        matches = [k for k, v in piv.items()
                   if v and abs(buy - v) / v <= PIVOT_TOL]
        out['pivot_match'] = '+'.join(matches) if matches else 'NONE'
        out['eday_traded_through'] = bool(eday['high'] >= buy >= eday['low'])
        out['eday_open_above_buy'] = bool(eday['open'] > buy * 1.001)  # fantasy fill flag
        out['eday_open'] = round(float(eday['open']), 2)
        out['eday_close'] = round(float(eday['close']), 2)
        out['buy_vs_eday_close_pct'] = round(pct(buy, float(eday['close'])), 2)
        prev_close = float(hist['close'].iloc[-1]) if len(hist) else None
        ath = piv['ATH']
        out['prev_close_from_ATH_pct'] = round((ath - prev_close) / ath * 100, 2) if (ath and prev_close) else None
        if len(hist) >= 127:
            out['ret_126d_pct'] = round(pct(prev_close, float(hist['close'].iloc[-127])), 1)

        # ---- exit replay (closed trades) ----
        if t['status'] == 'closed':
            n_closed += 1
            gt_xd = pd.Timestamp(t['exit_date'])
            gt_sell = float(t['sell'])
            best = None
            for sm in STOP_MODES:
                for tm in TRAIL_MODES:
                    xd, xp, xr = replay_exit(bars, ed, buy, sm, tm)
                    if xd is None:
                        continue
                    dd = abs((xd - gt_xd).days)
                    dp = abs(pct(xp, gt_sell))
                    key = f'{sm}_{tm}'
                    out[f'x_{key}'] = f'{xd.date()}|{xp:.2f}|{xr}'
                    hit = dd <= 3 and dp <= PRICE_TOL * 100 and xr == t['exit_reason']
                    if hit:
                        combo_hits[(sm, tm)] += 1
                    score = (0 if hit else 1, dd, dp)
                    if best is None or score < best[0]:
                        best = (score, sm, tm, xd, xp, xr, dd, dp)
            if best:
                _, sm, tm, xd, xp, xr, dd, dp = best
                out['best_combo'] = f'{sm}_{tm}'
                out['best_exit_date'] = str(xd.date())
                out['best_exit_px'] = round(xp, 2)
                out['best_reason'] = xr
                out['exit_date_delta_d'] = dd
                out['exit_px_delta_pct'] = round(dp, 2)
                out['exit_match'] = bool(dd <= 3 and dp <= PRICE_TOL * 100
                                         and xr == t['exit_reason'])
        else:
            # open trade: their return implies a year-end mark
            implied = buy * (1 + float(t['return_pct']) / 100)
            ye = bars.loc[bars.index <= pd.Timestamp('2025-12-31')]
            if len(ye):
                ye_close = float(ye['close'].iloc[-1])
                out['ye_close_ours'] = round(ye_close, 2)
                out['ye_close_theirs_implied'] = round(implied, 2)
                out['ye_mark_delta_pct'] = round(pct(ye_close, implied), 2)

        rows_out.append(out)

    # ---- write + summarize ----
    OUT_CSV.parent.mkdir(exist_ok=True)
    fieldnames = []
    for r in rows_out:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print('\n--- ENTRY: pivot fingerprint ---')
    for r in rows_out:
        if 'pivot_match' in r:
            print(f"{r['symbol']:12s} {r['entry_date']} buy={r['buy']:>9s} "
                  f"pivot={r['pivot_match']:10s} vsATH={r.get('buy_vs_ATH_pct')}% "
                  f"through={r.get('eday_traded_through')} "
                  f"openAbove={r.get('eday_open_above_buy')} "
                  f"fromATH(20%rule)={r.get('prev_close_from_ATH_pct')}%")

    print('\n--- EXIT: convention scoreboard (matches / closed trades) ---')
    for (sm, tm), n in sorted(combo_hits.items(), key=lambda x: -x[1]):
        print(f'stop={sm} trail={tm}: {n}/{n_closed}')

    print('\n--- EXIT: per-trade best ---')
    for r in rows_out:
        if r.get('status') == 'closed' and 'best_combo' in r:
            print(f"{r['symbol']:12s} {r['entry_date']} their: {r['exit_date']} @{r['sell']} "
                  f"({r['exit_reason']})  ours[{r['best_combo']}]: {r['best_exit_date']} "
                  f"@{r['best_exit_px']} ({r['best_reason']}) "
                  f"dd={r['exit_date_delta_d']}d dp={r['exit_px_delta_pct']}% "
                  f"{'MATCH' if r.get('exit_match') else 'X'}")

    n_entry_ok = sum(1 for r in rows_out if r.get('pivot_match') not in (None, 'NONE'))
    n_exit_ok = sum(1 for r in rows_out if r.get('exit_match'))
    print(f'\nSUMMARY: entries with a pivot fingerprint: {n_entry_ok}/{len(rows_out)} | '
          f'exit matches: {n_exit_ok}/{n_closed} | no-data symbols: {len(missing)}')
    print(f'Wrote {OUT_CSV}')


if __name__ == '__main__':
    main()
