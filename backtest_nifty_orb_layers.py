"""
Phase 1c: Layered variant sweep.

Tests:
  Layer A: Lenient vs strict abort (re-entry-into-OR handling during wait)
  Layer B: Wider OR (15 / 30 / 45 / 60 min)
  Layer C: Asymmetric RSI thresholds (long vs short)
  Layer D: K wait + gap exclusion combined
  Layer E: RSI as EXIT trigger (take every break, exit on adverse RSI)

Metrics added:
  trades_per_year    — annualized signal count
  wins_per_year      — annualized win count
  abort_events       — # of candles where price was inside OR during the wait window
                        (in lenient mode these don't kill the trade, just measured)

Sample: NIFTY50 5-min, 455 sessions = 1.81 years
"""

import sqlite3
from datetime import time as dtime

import numpy as np
import pandas as pd

DB = 'backtest_data/market_data.db'
SYMBOL = 'NIFTY50'
SESSION_OPEN = dtime(9, 15)
SESSION_CLOSE = dtime(15, 30)
NO_ENTRY_AFTER = dtime(14, 0)

YEARS_IN_SAMPLE = 455 / 252  # ~1.81 years


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def load_df():
    conn = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT date, open, high, low, close FROM market_data_unified "
        "WHERE symbol=? AND timeframe='5minute' ORDER BY date",
        conn, params=[SYMBOL])
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    for c in ['open', 'high', 'low', 'close']:
        df[c] = df[c].astype(float)
    df = df.between_time(SESSION_OPEN, SESSION_CLOSE)
    df['rsi5m'] = rsi(df['close'], 14)
    return df


def run_variant(df, *, or_min, rsi_long=None, rsi_short=None,
                wait_k=0, abort_mode='lenient', gap_max=None,
                rsi_exit_long=None, rsi_exit_short=None):
    """
    or_min: 5/10/15/30/45/60
    rsi_long, rsi_short: entry thresholds (None = no filter on that side)
    wait_k: wait window in 5-min candles (0 = strict same-candle confirmation)
    abort_mode: 'strict' = first re-entry kills; 'lenient' = keep watching
    gap_max: max |gap_pct| tolerated (None = no gap filter)
    rsi_exit_long: if set, exit long if rsi5m falls below this on any later candle
    rsi_exit_short: if set, exit short if rsi5m rises above this on any later candle
    """
    total_min = 9 * 60 + 15 + or_min
    or_end_time = dtime(total_min // 60, total_min % 60)

    trades = []
    raw_breaks = 0
    skipped_gap = 0
    aborted_strict = 0          # only used when abort_mode=='strict'
    abort_events_total = 0      # total candles where price was inside OR during wait
    abort_recovered = 0         # lenient: signal was inside OR mid-wait but eventually entered
    never_confirmed = 0
    confirmed_imm = 0
    confirmed_delayed = 0
    skipped_no_rsi_filter_late = 0  # entry pushed past 14:00 cutoff after wait

    days = df.index.normalize().unique()
    prev_close = None

    for day in days:
        sess = df[df.index.normalize() == day]
        if len(sess) < 10:
            continue
        sess_open = float(sess.iloc[0]['open'])
        gap_pct = ((sess_open / prev_close - 1) * 100) if prev_close else 0.0
        prev_close = float(sess.iloc[-1]['close'])

        if gap_max is not None and abs(gap_pct) > gap_max:
            skipped_gap += 1
            continue

        or_bars = sess.between_time(SESSION_OPEN, or_end_time)
        if len(or_bars) < 1:
            continue
        or_high = or_bars['high'].max()
        or_low = or_bars['low'].min()

        post = sess.between_time(or_end_time, SESSION_CLOSE).iloc[1:]
        if len(post) < 2:
            continue

        # Find first break
        break_i, break_dir = None, None
        for i, (ts, row) in enumerate(post.iterrows()):
            if row['close'] > or_high:
                break_i, break_dir = i, 'LONG'
                break
            if row['close'] < or_low:
                break_i, break_dir = i, 'SHORT'
                break
        if break_i is None:
            continue
        if post.index[break_i].time() >= NO_ENTRY_AFTER:
            continue

        raw_breaks += 1

        # Determine entry threshold for this direction (None = no filter)
        rsi_thresh = rsi_long if break_dir == 'LONG' else rsi_short

        def rsi_confirms(rval):
            if rsi_thresh is None:
                return True
            if pd.isna(rval):
                return False
            return rval > rsi_thresh if break_dir == 'LONG' else rval < rsi_thresh

        def price_outside_or(close_p):
            return close_p > or_high if break_dir == 'LONG' else close_p < or_low

        # Wait window scan
        entry_i = None
        had_abort_event = False
        abort_events_in_window = 0
        for k in range(wait_k + 1):
            scan_i = break_i + k
            if scan_i >= len(post):
                break
            row = post.iloc[scan_i]
            outside = price_outside_or(row['close'])
            if not outside and k > 0:
                abort_events_in_window += 1
                had_abort_event = True
                if abort_mode == 'strict':
                    break
                # lenient: continue scanning
                continue
            if outside and rsi_confirms(row['rsi5m']):
                entry_i = scan_i
                if k == 0:
                    confirmed_imm += 1
                else:
                    confirmed_delayed += 1
                if had_abort_event:
                    abort_recovered += 1
                break

        abort_events_total += abort_events_in_window
        if abort_mode == 'strict' and entry_i is None and had_abort_event:
            aborted_strict += 1

        if entry_i is None:
            never_confirmed += 1
            continue

        if post.index[entry_i].time() >= NO_ENTRY_AFTER:
            skipped_no_rsi_filter_late += 1
            continue

        entry_ts = post.index[entry_i]
        entry_row = post.iloc[entry_i]
        wait_min = (entry_ts - post.index[break_i]).total_seconds() / 60

        # Outcome scan
        rest = post.iloc[entry_i + 1:]
        sl_hit = False
        sl_time = None
        rsi_exit_hit = False
        rsi_exit_time = None
        for rts, rrow in rest.iterrows():
            # opposite-OR breach (canonical SL)
            if break_dir == 'LONG' and rrow['close'] < or_low:
                sl_hit = True
                sl_time = rts
                break
            if break_dir == 'SHORT' and rrow['close'] > or_high:
                sl_hit = True
                sl_time = rts
                break
            # RSI-based exit (Layer E)
            if break_dir == 'LONG' and rsi_exit_long is not None:
                if not pd.isna(rrow['rsi5m']) and rrow['rsi5m'] < rsi_exit_long:
                    rsi_exit_hit = True
                    rsi_exit_time = rts
                    break
            if break_dir == 'SHORT' and rsi_exit_short is not None:
                if not pd.isna(rrow['rsi5m']) and rrow['rsi5m'] > rsi_exit_short:
                    rsi_exit_hit = True
                    rsi_exit_time = rts
                    break

        # Outcome label:
        # - If SL hit -> LOSS
        # - If RSI-exit hit (no SL) -> LOSS_RSI_EXIT (still a loss in option-strangle terms; price moved adversely enough)
        # - Else -> WIN
        outcome = 'LOSS' if sl_hit else ('LOSS_RSI' if rsi_exit_hit else 'WIN')

        trades.append({
            'date': day.date(),
            'direction': break_dir,
            'gap_pct': round(gap_pct, 3),
            'break_ts': post.index[break_i],
            'entry_ts': entry_ts,
            'wait_min': wait_min,
            'had_abort_event': had_abort_event,
            'outcome': outcome,
            'sl_hit': sl_hit,
            'sl_time': sl_time,
            'rsi_exit_hit': rsi_exit_hit,
            'mins_to_sl': (sl_time - entry_ts).total_seconds() / 60 if sl_hit else None,
            'mins_to_rsi_exit': (rsi_exit_time - entry_ts).total_seconds() / 60 if rsi_exit_hit else None,
        })

    n = len(trades)
    df_t = pd.DataFrame(trades) if n else pd.DataFrame()
    wins = df_t[df_t['outcome'] == 'WIN'] if n else df_t
    losses_sl = df_t[df_t['outcome'] == 'LOSS'] if n else df_t
    losses_rsi = df_t[df_t['outcome'] == 'LOSS_RSI'] if n else df_t

    return {
        'N_taken': n,
        'WR_pct': round(len(wins) / n * 100, 1) if n else 0.0,
        'Trades_per_year': round(n / YEARS_IN_SAMPLE, 1),
        'Wins_per_year': round(len(wins) / YEARS_IN_SAMPLE, 1),
        'SL_losses': len(losses_sl),
        'RSI_losses': len(losses_rsi),
        'Imm_conf': confirmed_imm,
        'Delayed_conf': confirmed_delayed,
        'Abort_events': abort_events_total,
        'Abort_recovered': abort_recovered,
        'Aborted_strict': aborted_strict,
        'Never_conf': never_confirmed,
        'Skipped_gap': skipped_gap,
        'Med_wait_min_delayed': round(float(df_t.loc[df_t['wait_min'] > 0, 'wait_min'].median()), 1)
                                 if n and (df_t['wait_min'] > 0).any() else None,
        'Med_min_to_SL': round(float(losses_sl['mins_to_sl'].median()), 0) if len(losses_sl) else None,
        'Raw_breaks': raw_breaks,
    }


def fmt_row(label, r):
    return {
        'variant': label,
        'N': r['N_taken'],
        'WR%': r['WR_pct'],
        'Tr/yr': r['Trades_per_year'],
        'Win/yr': r['Wins_per_year'],
        'SLloss': r['SL_losses'],
        'RSIloss': r['RSI_losses'],
        'AbortEv': r['Abort_events'],
        'AbortRec': r['Abort_recovered'],
        'AbortStr': r['Aborted_strict'],
        'NeverConf': r['Never_conf'],
        'SkipGap': r['Skipped_gap'],
        'MedWait': r['Med_wait_min_delayed'],
        'MedSLmin': r['Med_min_to_SL'],
    }


def main():
    print("Loading NIFTY50 5-min...")
    df = load_df()
    n_sess = df.index.normalize().nunique()
    print(f"  {len(df)} bars, {n_sess} sessions, ~{YEARS_IN_SAMPLE:.2f} years\n")

    rows = []

    # ============================================================
    # Layer A: Strict vs lenient abort, on best base config
    # Base: OR30m, RSI>60/<40, K=12
    # ============================================================
    rows.append(fmt_row(
        "A. base OR30 RSI60/40 K=12 STRICT abort",
        run_variant(df, or_min=30, rsi_long=60, rsi_short=40, wait_k=12, abort_mode='strict')))
    rows.append(fmt_row(
        "A. base OR30 RSI60/40 K=12 LENIENT abort",
        run_variant(df, or_min=30, rsi_long=60, rsi_short=40, wait_k=12, abort_mode='lenient')))

    # ============================================================
    # Layer B: Wider OR (15/30/45/60), RSI60/40, K=12, lenient
    # ============================================================
    for w in [15, 30, 45, 60]:
        rows.append(fmt_row(
            f"B. OR{w}m RSI60/40 K=12 lenient",
            run_variant(df, or_min=w, rsi_long=60, rsi_short=40, wait_k=12, abort_mode='lenient')))

    # ============================================================
    # Layer C: Asymmetric RSI thresholds (OR30, K=12, lenient)
    # ============================================================
    asym_pairs = [
        (50, 40),  # loose long, std short
        (55, 40),
        (55, 35),
        (60, 40),  # symmetric mid
        (60, 35),
        (60, 45),  # very loose short (compensate weakness)
        (65, 35),  # symmetric tight
        (65, 40),
        (50, 35),
    ]
    for lo, hi in asym_pairs:
        rows.append(fmt_row(
            f"C. OR30 RSI L>{lo} S<{hi} K=12 lenient",
            run_variant(df, or_min=30, rsi_long=lo, rsi_short=hi, wait_k=12, abort_mode='lenient')))

    # Single-side variants (long-only / short-only)
    rows.append(fmt_row(
        "C. OR30 SHORT-only RSI<40 K=12 lenient",
        run_variant(df, or_min=30, rsi_long=999, rsi_short=40, wait_k=12, abort_mode='lenient')))
    rows.append(fmt_row(
        "C. OR30 LONG-only RSI>60 K=12 lenient",
        run_variant(df, or_min=30, rsi_long=60, rsi_short=-999, wait_k=12, abort_mode='lenient')))

    # ============================================================
    # Layer D: K wait + gap exclusion combined (OR30, RSI60/40, lenient)
    # ============================================================
    for gmax in [None, 1.0, 0.5]:
        rows.append(fmt_row(
            f"D. OR30 RSI60/40 K=12 lenient gap<={gmax}",
            run_variant(df, or_min=30, rsi_long=60, rsi_short=40, wait_k=12, abort_mode='lenient', gap_max=gmax)))

    # ============================================================
    # Layer E: RSI as EXIT (no entry filter, but RSI exit on adverse turn)
    # ============================================================
    # Take every break (no entry RSI filter), exit on opposite OR OR adverse RSI
    for ex_lo, ex_hi in [(45, 55), (40, 60), (35, 65), (30, 70)]:
        rows.append(fmt_row(
            f"E. OR30 NO entry filter, RSI exit L<{ex_lo} S>{ex_hi}",
            run_variant(df, or_min=30, wait_k=0, abort_mode='lenient',
                        rsi_exit_long=ex_lo, rsi_exit_short=ex_hi)))
    # E variant with entry filter ALSO applied
    for ex_lo, ex_hi in [(45, 55), (40, 60)]:
        rows.append(fmt_row(
            f"E. OR30 RSI60/40 entry + RSI exit L<{ex_lo} S>{ex_hi} K=12",
            run_variant(df, or_min=30, rsi_long=60, rsi_short=40, wait_k=12, abort_mode='lenient',
                        rsi_exit_long=ex_lo, rsi_exit_short=ex_hi)))

    # ============================================================
    # Print
    # ============================================================
    res = pd.DataFrame(rows)
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 70)

    print("=" * 200)
    print("LAYERED VARIANT SWEEP — NIFTY ORB")
    print(f"Annualized over {YEARS_IN_SAMPLE:.2f}-year sample. Lenient abort = price re-entering OR doesn't kill the watch.")
    print("=" * 200)
    print(res.to_string(index=False))

    res.to_csv('nifty_orb_layered_sweep.csv', index=False)

    # Top 10 by Wins/year
    print("\n" + "-" * 100)
    print("TOP 10 by Wins/year (= EV-volume proxy)")
    print("-" * 100)
    top = res.sort_values('Win/yr', ascending=False).head(10)
    print(top[['variant', 'N', 'WR%', 'Tr/yr', 'Win/yr', 'SLloss', 'RSIloss']].to_string(index=False))

    print("\nSaved: nifty_orb_layered_sweep.csv")


if __name__ == '__main__':
    main()
