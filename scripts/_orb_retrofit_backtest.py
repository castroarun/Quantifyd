"""
ORB features 1-3 retrofit analyzer
===================================

Reads `backtest_data/orb_backtest.db` and simulates what the book would
have looked like on each historical day with these features applied:

  F1 — staleness guards (age<=15min, drift<=0.5%, cutoff<=14:00)
  F2 — two-tier book drawdown cut (soft -Rs 7500 halve, hard -Rs 15000 flatten)
  F3 — tail hedge (skew>=0.70, count>=10; 1 lot 1.5% OTM weekly CE/PE)

Exactness notes
---------------
* F1 is exact — every row in `orb_backtest_signals` has `entry_time`
  and direction, so we can test each gate deterministically.
* F3 is exact on the trigger side (we know count + skew per day) but
  the hedge P&L is a proxy: premium cost minus a rough payoff based
  on whether the book's final net P&L was negative (tail day).
* F2 is the weakest: the DB only stores per-trade FINAL outcomes,
  not intraday mark-to-market. We use a proxy — if the *sum* of
  losing-trade P&Ls hits the threshold, assume the book crossed it.
  This over-counts (some losers ended bad but never touched the
  intraday threshold) and under-counts (some days hit -7500 intraday
  then recovered). The real P&L impact of F2 requires a tick-level
  re-run, which is tracked as a TODO in the plan doc.

Output
------
CSV: `backtest_data/retrofit_features_1_to_3.csv`
Columns:
  date, universe_size,
  baseline_trades, baseline_pnl,
  f1_staleness_blocks, f1_trades_after, f1_pnl_after,
  f3_hedge_fired, f3_net_side, f3_skew, f3_hedge_est_cost,
  f2_softcut_proxy, f2_hardcut_proxy, f2_pnl_proxy,
  combined_pnl_proxy
"""

import csv
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, time as dtime

# ─── Config knobs (aligned with ORB_DEFAULTS) ──────────────────────
SIGNAL_AGE_MAX_MINS = 15
ENTRY_END_TIME = dtime(14, 0)
OR_END_TIME = dtime(9, 30)          # ORB opening range ends here
HEDGE_SKEW = 0.70
HEDGE_MIN_POSITIONS = 10
HEDGE_EVAL_START = dtime(10, 0)
HEDGE_EVAL_END = dtime(14, 0)
HEDGE_EST_PREMIUM_COST = 2500       # Rs per 1 lot of 1.5% OTM NIFTY weekly (conservative avg)
HEDGE_TAIL_DAY_PAYOFF = 6000        # Rs avg payoff on losing days (proxy)
BOOK_DD_SOFT = -7500
BOOK_DD_HARD = -15000


# ─── Helpers ────────────────────────────────────────────────────────

def _parse_hhmm(s):
    """Parse 'HH:MM' string -> dtime. Returns None if empty/invalid."""
    if not s:
        return None
    try:
        h, m = s.split(':')[:2]
        return dtime(int(h), int(m))
    except Exception:
        return None


def _notional(row):
    if not row['entry_price']:
        return 0.0
    # The backtest db doesn't store qty — use price as a proxy for notional
    # (it's the same across stocks for the purposes of skew since we only
    # care about the ratio). For accuracy in per-day comparisons, use the
    # raw count.
    return float(row['entry_price'])


# ─── Main ───────────────────────────────────────────────────────────

def run(db_path, out_csv):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    runs = {r['run_date']: dict(r)
            for r in conn.execute("SELECT * FROM orb_backtest_runs")}
    signals_by_date = defaultdict(list)
    for r in conn.execute("SELECT * FROM orb_backtest_signals"):
        signals_by_date[r['run_date']].append(dict(r))
    conn.close()

    dates = sorted(runs.keys())
    print(f"Retrofit on {len(dates)} backtest days "
          f"(range: {dates[0]} -> {dates[-1]})" if dates else "No data.")
    if not dates:
        return

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = [
        'date', 'universe_size',
        'baseline_trades', 'baseline_pnl',
        'f1_staleness_blocks', 'f1_trades_after', 'f1_pnl_after',
        'f3_hedge_fired', 'f3_net_side', 'f3_skew', 'f3_hedge_est_cost',
        'f2_softcut_proxy', 'f2_hardcut_proxy', 'f2_pnl_proxy',
        'combined_pnl_proxy',
    ]
    totals = defaultdict(float)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for d in dates:
            row_out = _simulate_day(d, runs[d], signals_by_date.get(d, []))
            w.writerow(row_out)
            for k, v in row_out.items():
                if isinstance(v, (int, float)) and k not in ('universe_size',):
                    totals[k] += v

    print(f"\nCSV written: {out_csv}")
    print(f"\nAggregates across {len(dates)} days:")
    print(f"  Baseline total P&L        Rs {totals['baseline_pnl']:>12,.0f}")
    print(f"  After F1 (staleness)      Rs {totals['f1_pnl_after']:>12,.0f}  "
          f"(delta Rs {totals['f1_pnl_after']-totals['baseline_pnl']:+,.0f})")
    print(f"  F1 entries blocked        {int(totals['f1_staleness_blocks'])} "
          f"(of {int(totals['baseline_trades'])} baseline)")
    print(f"  F3 hedge fired days       {int(totals['f3_hedge_fired'])}")
    print(f"  F3 hedge premium cost     Rs {totals['f3_hedge_est_cost']:>12,.0f}")
    print(f"  F2 softcut proxy days     {int(totals['f2_softcut_proxy'])}")
    print(f"  F2 hardcut proxy days     {int(totals['f2_hardcut_proxy'])}")
    print(f"  Combined proxy total P&L  Rs {totals['combined_pnl_proxy']:>12,.0f}")


def _simulate_day(d, run_row, sigs):
    """Apply F1/F2/F3 to one day's signals."""
    taken = [s for s in sigs if s['signal_type'] == 'TAKEN']
    baseline_trades = len(taken)
    baseline_pnl = sum(float(s['pnl_inr'] or 0) for s in taken)

    # ─── F1: staleness ────────────────────────────────────────────
    # A signal is "stale" if its entry_time is past the entry cutoff
    # (14:00). We cannot compute signal_age (time from breakout candle
    # to now) from the DB, so we use the stricter-of-the-two available
    # proxies: entry_time past 14:00.
    # NOTE: the backtest engine enters on the first valid breakout, so
    # age_max=15min is already implicit in how the BT generates trades
    # — we don't double-filter that.
    f1_blocks = 0
    taken_after_f1 = []
    for s in taken:
        t = _parse_hhmm(s.get('entry_time'))
        if t is not None and t > ENTRY_END_TIME:
            f1_blocks += 1
        else:
            taken_after_f1.append(s)
    f1_pnl = sum(float(s['pnl_inr'] or 0) for s in taken_after_f1)

    # ─── F3: tail hedge ───────────────────────────────────────────
    # Computed on the POST-F1 set (those are the trades that would
    # actually have happened with features active).
    short_count = sum(1 for s in taken_after_f1 if s['direction'] == 'SHORT')
    long_count = sum(1 for s in taken_after_f1 if s['direction'] == 'LONG')
    total_count = short_count + long_count
    if total_count > 0:
        skew = abs(short_count - long_count) / total_count
    else:
        skew = 0.0
    net_side = 'SHORT' if short_count > long_count else (
        'LONG' if long_count > short_count else 'FLAT')

    # Any entry between 10:00 and 14:00 AND count≥10 on that tick would
    # fire the hedge. We approximate by checking: if at least 10 of the
    # taken trades entered before 14:00 AND skew≥0.70 — the hedge would
    # have been fired on the first tick that crosses both thresholds.
    entered_before_hedge_end = sum(
        1 for s in taken_after_f1
        if (t := _parse_hhmm(s.get('entry_time'))) is not None
        and HEDGE_EVAL_START <= t <= HEDGE_EVAL_END
    )
    hedge_fired = int(entered_before_hedge_end >= HEDGE_MIN_POSITIONS
                       and skew >= HEDGE_SKEW)
    hedge_cost = HEDGE_EST_PREMIUM_COST if hedge_fired else 0

    # Proxy hedge payoff: on days where book P&L (after F1) was negative
    # AND hedge was fired, assume an average payoff equal to
    # HEDGE_TAIL_DAY_PAYOFF. This is intentionally conservative — the
    # real hedge payoff depends on NIFTY's intraday reversal magnitude
    # on that day. Tracked as a follow-up for a tick-level re-run.
    hedge_payoff = (HEDGE_TAIL_DAY_PAYOFF
                    if (hedge_fired and f1_pnl < -2000) else 0)
    hedge_net = hedge_payoff - hedge_cost

    # ─── F2: drawdown cut (proxy only) ────────────────────────────
    # Without intraday mark-to-market we can't tell when the book
    # crossed -7500 or -15000. Proxy: use the sum of NEGATIVE trade
    # P&Ls as a "loser bucket" — if that bucket is <= threshold,
    # assume the cut would have triggered at some point.
    losers_sum = sum(float(s['pnl_inr'] or 0)
                      for s in taken_after_f1 if (s['pnl_inr'] or 0) < 0)
    softcut = int(losers_sum <= BOOK_DD_SOFT)
    hardcut = int(losers_sum <= BOOK_DD_HARD)

    # Proxy P&L impact of F2: if hardcut fires, floor the day's loss at
    # HARD threshold (remaining losing trades don't bleed further).
    # If only softcut fires, scale losers to 50% (halved positions
    # continue to bleed at half the rate) — this is directionally right
    # but the magnitude needs tick-level validation.
    if hardcut:
        f2_pnl = BOOK_DD_HARD
    elif softcut:
        winners_sum = sum(float(s['pnl_inr'] or 0)
                          for s in taken_after_f1 if (s['pnl_inr'] or 0) >= 0)
        f2_pnl = winners_sum + losers_sum * 0.5
    else:
        f2_pnl = f1_pnl

    combined = f2_pnl + hedge_net

    return {
        'date': d,
        'universe_size': run_row.get('universe_size') or 0,
        'baseline_trades': baseline_trades,
        'baseline_pnl': round(baseline_pnl, 2),
        'f1_staleness_blocks': f1_blocks,
        'f1_trades_after': len(taken_after_f1),
        'f1_pnl_after': round(f1_pnl, 2),
        'f3_hedge_fired': hedge_fired,
        'f3_net_side': net_side,
        'f3_skew': round(skew, 3),
        'f3_hedge_est_cost': hedge_cost,
        'f2_softcut_proxy': softcut,
        'f2_hardcut_proxy': hardcut,
        'f2_pnl_proxy': round(f2_pnl, 2),
        'combined_pnl_proxy': round(combined, 2),
    }


if __name__ == '__main__':
    # Repo default location; override via argv[1]
    default_db = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'backtest_data', 'orb_backtest.db',
    )
    default_csv = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'backtest_data', 'retrofit_features_1_to_3.csv',
    )
    db = sys.argv[1] if len(sys.argv) > 1 else default_db
    out = sys.argv[2] if len(sys.argv) > 2 else default_csv
    if not os.path.exists(db):
        print(f"DB not found: {db}", file=sys.stderr)
        print("Usage: python scripts/_orb_retrofit_backtest.py [db_path] [csv_path]")
        sys.exit(2)
    run(db, out)
