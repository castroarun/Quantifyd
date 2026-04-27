"""ORB Filter Tuning — Phase 1 + Phase 2 sweep.

Phase 1: 4x4 grid of (rsi_short_threshold, signal_age_max_mins) +
         4x1 grid of (rsi_long_threshold, fixed signal_age=15)
Phase 2: 4 cells of signal_drift_max_pct (only run if Phase 1 winner
         doesn't address VEDL-style drift misses).

Universe: 15 ORB stocks
Period: 2024-03-18 to 2026-03-12 (full available 5-min data window)
Engine: ORBFilterEngine (subclass of ORBBacktestEngine) — overrides
        _simulate_day to add signal_age + signal_drift gating that
        replicates the LIVE orb_live_engine.py behavior.

Live engine logic being replicated:
- When a breakout candle is detected, the trade can only enter on bars
  where (current_bar_time - breakout_candle_time) <= signal_age_max_mins
  AND |current_close - breakout_close| / breakout_close <= signal_drift_max_pct
  AND all other filters pass.
- If filters fail on the breakout bar, the engine retries on subsequent
  bars (mimicking the 5-min scanner re-poll in live), expiring once age
  or drift exceeds limits.
"""
from __future__ import annotations

import csv
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.disable(logging.WARNING)
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from services.orb_backtest_engine import (
    ORBConfig, ORBBacktestEngine, BacktestResult,
    _calc_rsi_wilder, _resample_5min_to_15min,
)

OUT = Path(__file__).resolve().parents[1] / 'results'
OUT.mkdir(parents=True, exist_ok=True)

# Live ORB universe (15 stocks)
UNIVERSE = [
    'ADANIENT', 'TATASTEEL', 'BEL', 'VEDL', 'BPCL', 'M&M', 'BAJFINANCE',
    'TRENT', 'HAL', 'IRCTC', 'GRASIM', 'GODREJPROP', 'RELIANCE',
    'AXISBANK', 'APOLLOHOSP',
]

PERIOD_START = '2024-03-18'
PERIOD_END = '2026-03-12'

# Output schema (matches task spec)
FIELDNAMES = [
    'cell_label', 'rsi_long', 'rsi_short', 'signal_age_max',
    'signal_drift_max', 'trades', 'wins', 'losses', 'win_rate_pct',
    'avg_win', 'avg_loss', 'profit_factor', 'sharpe', 'max_dd_pct',
    'calmar', 'net_pnl', 'cagr_pct',
    # Long/short breakdown for diagnostic value
    'long_trades', 'short_trades', 'long_wr', 'short_wr',
]


# =============================================================================
# Custom engine — adds signal_age + signal_drift gating that retries entries
# while constraints permit. Replicates orb_live_engine.py behavior.
# =============================================================================

@dataclass
class FilterORBConfig(ORBConfig):
    """ORBConfig + the two staleness guards from live."""
    signal_age_max_mins: int = 15
    signal_drift_max_pct: float = 0.005


class ORBFilterEngine(ORBBacktestEngine):
    """ORB engine with signal_age and signal_drift entry gating."""

    def _simulate_day(
        self,
        symbol: str,
        day_bars: pd.DataFrame,
        prev_daily: dict,
        prev2_daily: Optional[dict],
        trade_date: pd.Timestamp,
    ) -> List[dict]:
        cfg: FilterORBConfig = self.config
        bars = day_bars.to_dict('records')
        n_bars = len(bars)

        if n_bars < 3:
            return []

        # ---- Daily filter pre-checks (copied verbatim from base engine) ----
        pivot = prev_daily['pivot']
        tc = prev_daily['tc']
        bc = prev_daily['bc']
        cpr_width_pct = prev_daily['cpr_width_pct']
        r1 = prev_daily['r1']
        s1 = prev_daily['s1']
        r2 = prev_daily['r2']
        s2 = prev_daily['s2']

        if cfg.use_cpr_width_filter and cpr_width_pct > cfg.cpr_width_threshold_pct:
            return []
        if cfg.use_inside_day_filter:
            if prev2_daily is None:
                return []
            prev_range = prev_daily['high'] - prev_daily['low']
            prev2_range = prev2_daily['high'] - prev2_daily['low']
            if prev_range >= prev2_range:
                return []
        if cfg.use_narrow_range_filter:
            if not prev_daily.get('is_narrow_range', False):
                return []

        allow_long_prev_cpr = True
        allow_short_prev_cpr = True
        if cfg.use_prev_cpr_filter:
            allow_long_prev_cpr = prev_daily.get('prev_cpr_bullish', True)
            allow_short_prev_cpr = prev_daily.get('prev_cpr_bearish', True)

        prev_day_high = prev_daily['high']
        prev_day_low = prev_daily['low']
        bb_middle = prev_daily.get('bb_middle', np.nan)
        daily_ema = prev_daily.get('daily_ema', np.nan)
        daily_atr = prev_daily.get('atr', np.nan)

        # ---- Build OR ----
        or_high = -np.inf
        or_low = np.inf
        or_bar_count = cfg.or_minutes // 5
        if n_bars < or_bar_count:
            return []
        for i in range(or_bar_count):
            b = bars[i]
            or_high = max(or_high, b['high'])
            or_low = min(or_low, b['low'])
        or_range = or_high - or_low
        if or_range <= 0:
            return []

        # ---- Walk bars after OR ----
        trades: List[dict] = []
        trades_today = 0
        position = None
        prev_close = bars[or_bar_count - 1]['close']

        # VWAP accumulators (session start)
        cum_tp_vol = 0.0
        cum_vol = 0.0
        for i in range(or_bar_count):
            b = bars[i]
            tp = (b['high'] + b['low'] + b['close']) / 3.0
            cum_tp_vol += tp * b['volume']
            cum_vol += b['volume']

        # Virgin CPR
        cpr_virgin = True
        cpr_top = max(tc, bc)
        cpr_bot = min(tc, bc)
        for i in range(or_bar_count):
            b = bars[i]
            if b['low'] <= cpr_top and b['high'] >= cpr_bot:
                cpr_virgin = False
                break

        # ---- Signal-aware entry state ----
        # Track an "armed" breakout that's pending entry. Re-tries filters on
        # each new bar until either filters pass (entry), or signal_age /
        # signal_drift expire (signal abandoned), or a new breakout supersedes.
        pending = None  # dict: {direction, breakout_bar_idx, breakout_time, breakout_close}

        last_entry_t = self._last_entry_time
        eod_t = self._eod_exit_time
        max_age = cfg.signal_age_max_mins
        max_drift = cfg.signal_drift_max_pct

        for i in range(or_bar_count, n_bars):
            bar = bars[i]
            bar_dt = bar['date']
            if isinstance(bar_dt, datetime):
                bar_time = bar_dt.time()
            else:
                bar_time = bar_dt.to_pydatetime().time()
            bar_close = bar['close']
            bar_high = bar['high']
            bar_low = bar['low']

            # VWAP update
            tp = (bar_high + bar_low + bar_close) / 3.0
            cum_tp_vol += tp * bar['volume']
            cum_vol += bar['volume']
            vwap = cum_tp_vol / cum_vol if cum_vol > 0 else bar_close

            # Virgin CPR update
            if cpr_virgin and bar_low <= cpr_top and bar_high >= cpr_bot:
                cpr_virgin = False

            # ---- Position management (exits) ----
            if position is not None:
                exit_price = None
                exit_reason = None
                direction = position['direction']
                if direction == 'long':
                    if bar_low <= position['sl']:
                        exit_price = position['sl']
                        exit_reason = 'sl'
                    elif bar_high >= position['target']:
                        exit_price = position['target']
                        exit_reason = 'target'
                else:
                    if bar_high >= position['sl']:
                        exit_price = position['sl']
                        exit_reason = 'sl'
                    elif bar_low <= position['target']:
                        exit_price = position['target']
                        exit_reason = 'target'

                if exit_price is None and bar_time >= eod_t:
                    exit_price = bar_close
                    exit_reason = 'eod'

                if exit_price is not None:
                    pnl = ((exit_price - position['entry_price'])
                           if direction == 'long'
                           else (position['entry_price'] - exit_price))
                    holding_bars = i - position['entry_bar_idx']
                    trade_rec = {
                        'symbol': symbol,
                        'date': trade_date.strftime('%Y-%m-%d')
                                if hasattr(trade_date, 'strftime')
                                else str(trade_date)[:10],
                        'direction': direction,
                        'entry_time': position['entry_time'],
                        'entry_price': position['entry_price'],
                        'sl': position['sl'],
                        'target': position['target'],
                        'exit_time': str(bar_time),
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl_pts': round(pnl, 2),
                        'or_high': or_high,
                        'or_low': or_low,
                        'or_range': or_range,
                        'holding_bars': holding_bars,
                    }
                    trades.append(trade_rec)
                    position = None

                if position is not None:
                    prev_close = bar_close
                    continue

            # ---- Look for new breakout (if not in pending) ----
            # If we already have a pending breakout, check if a fresher one
            # supersedes (e.g., long flipped to short). LIVE engine takes
            # the LATEST valid transition consistent with current close.
            new_long_breakout = (bar_close > or_high and prev_close <= or_high)
            new_short_breakout = (bar_close < or_low and prev_close >= or_low)

            if new_long_breakout and not new_short_breakout:
                # Reset pending if direction flipped, or arm fresh
                pending = {
                    'direction': 'long',
                    'breakout_bar_idx': i,
                    'breakout_time': bar_dt,
                    'breakout_close': bar_close,
                }
            elif new_short_breakout and not new_long_breakout:
                pending = {
                    'direction': 'short',
                    'breakout_bar_idx': i,
                    'breakout_time': bar_dt,
                    'breakout_close': bar_close,
                }
            # both/neither: leave pending alone

            # ---- Try to enter from pending ----
            if pending is None:
                prev_close = bar_close
                continue

            if trades_today >= cfg.max_trades_per_day:
                prev_close = bar_close
                continue
            if bar_time >= last_entry_t:
                prev_close = bar_close
                continue
            if bar_time >= eod_t:
                prev_close = bar_close
                continue

            direction = pending['direction']
            if direction == 'long' and not cfg.allow_longs:
                prev_close = bar_close
                continue
            if direction == 'short' and not cfg.allow_shorts:
                prev_close = bar_close
                continue

            # ---- Check signal_age ----
            breakout_dt = pending['breakout_time']
            if not isinstance(breakout_dt, datetime):
                breakout_dt = breakout_dt.to_pydatetime()
            cur_dt = bar_dt if isinstance(bar_dt, datetime) else bar_dt.to_pydatetime()
            age_mins = (cur_dt - breakout_dt).total_seconds() / 60.0
            if age_mins > max_age:
                # Signal expired — discard pending
                pending = None
                prev_close = bar_close
                continue

            # ---- Check signal_drift ----
            breakout_close = pending['breakout_close']
            if breakout_close > 0:
                drift = abs(bar_close - breakout_close) / breakout_close
                if drift > max_drift:
                    pending = None
                    prev_close = bar_close
                    continue

            # ---- Check direction consistency with current close ----
            # (Live engine: SHORT only valid if close < or_low NOW; LONG only
            # if close > or_high NOW.)
            if direction == 'long' and bar_close <= or_high:
                prev_close = bar_close
                continue
            if direction == 'short' and bar_close >= or_low:
                prev_close = bar_close
                continue

            # ---- Entry price = current bar close (mimics live "enter at scan time") ----
            entry_price = bar_close

            # ---- Apply other filters ----
            if not self._pass_filters(
                cfg, direction, entry_price, vwap, bar, trade_date,
                pivot, tc, bc, r1, s1, r2, s2, cpr_virgin,
                allow_long_prev_cpr, allow_short_prev_cpr,
                prev_day_high, prev_day_low,
                bb_middle, daily_ema, daily_atr,
            ):
                # Filters failed — keep pending; will retry next bar (subject
                # to age/drift). This is the key live-replicating behavior.
                prev_close = bar_close
                continue

            # ---- Entry! ----
            sl, target = self._compute_sl_target(
                cfg, direction, entry_price, or_high, or_low, or_range,
                daily_atr, prev_daily,
            )
            position = {
                'direction': direction,
                'entry_price': entry_price,
                'entry_time': str(bar_time),
                'entry_bar_idx': i,
                'sl': sl,
                'target': target,
            }
            trades_today += 1
            pending = None
            prev_close = bar_close

        # ---- EOD force close ----
        if position is not None:
            last_bar = bars[-1]
            direction = position['direction']
            exit_price = last_bar['close']
            pnl = ((exit_price - position['entry_price'])
                   if direction == 'long'
                   else (position['entry_price'] - exit_price))
            holding_bars = (n_bars - 1) - position['entry_bar_idx']
            last_dt = last_bar['date']
            if isinstance(last_dt, datetime):
                last_t = last_dt.time()
            else:
                last_t = last_dt.to_pydatetime().time()
            trade_rec = {
                'symbol': symbol,
                'date': trade_date.strftime('%Y-%m-%d')
                        if hasattr(trade_date, 'strftime')
                        else str(trade_date)[:10],
                'direction': direction,
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'sl': position['sl'],
                'target': position['target'],
                'exit_time': str(last_t),
                'exit_price': exit_price,
                'exit_reason': 'eod',
                'pnl_pts': round(pnl, 2),
                'or_high': or_high,
                'or_low': or_low,
                'or_range': or_range,
                'holding_bars': holding_bars,
            }
            trades.append(trade_rec)

        return trades


# =============================================================================
# Cell aggregation
# =============================================================================

def aggregate_results(per_symbol_results: Dict[str, BacktestResult]) -> Dict:
    """Pool trades across all symbols into one summary row."""
    all_trades = []
    for sym, r in per_symbol_results.items():
        all_trades.extend(r.trades)

    n = len(all_trades)
    if n == 0:
        return dict(
            trades=0, wins=0, losses=0, win_rate_pct=0.0,
            avg_win=0.0, avg_loss=0.0, profit_factor=0.0,
            sharpe=0.0, max_dd_pct=0.0, calmar=0.0,
            net_pnl=0.0, cagr_pct=0.0,
            long_trades=0, short_trades=0, long_wr=0.0, short_wr=0.0,
        )

    pnls = np.array([t['pnl_pts'] for t in all_trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n * 100.0
    avg_win = float(wins.mean()) if n_wins else 0.0
    avg_loss = float(losses.mean()) if n_losses else 0.0
    gross_p = float(wins.sum()) if n_wins else 0.0
    gross_l = float(abs(losses.sum())) if n_losses else 0.0
    pf = gross_p / gross_l if gross_l > 0 else (float('inf') if gross_p > 0 else 0.0)

    # Aggregate by date for sharpe + DD
    by_date = {}
    for t in all_trades:
        by_date.setdefault(t['date'], 0.0)
        by_date[t['date']] += t['pnl_pts']
    daily_arr = np.array(sorted(by_date.values()))
    # Sharpe on daily PnL (all dates with at least one trade)
    if len(daily_arr) > 1 and daily_arr.std() > 0:
        sharpe = (daily_arr.mean() / daily_arr.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max DD on points-cumulative
    sorted_dates = sorted(by_date.keys())
    cum = np.cumsum([by_date[d] for d in sorted_dates])
    if len(cum) > 0:
        running_max = np.maximum.accumulate(cum)
        dd = running_max - cum
        max_dd_pts = float(dd.max()) if len(dd) > 0 else 0.0
    else:
        max_dd_pts = 0.0

    net_pnl = float(pnls.sum())

    # Approximate "%" max-DD using points relative to net_pnl + initial cap proxy:
    # For point-based system, scale by avg entry price across all trades to give
    # a rough percent. We use sum of |entry_price| as risk denominator proxy.
    avg_entry = float(np.mean([t['entry_price'] for t in all_trades]))
    max_dd_pct = max_dd_pts / avg_entry * 100.0 if avg_entry > 0 else 0.0

    # Direction breakdown
    long_t = [t for t in all_trades if t['direction'] == 'long']
    short_t = [t for t in all_trades if t['direction'] == 'short']
    long_wr = (sum(1 for t in long_t if t['pnl_pts'] > 0) / len(long_t) * 100.0
               if long_t else 0.0)
    short_wr = (sum(1 for t in short_t if t['pnl_pts'] > 0) / len(short_t) * 100.0
                if short_t else 0.0)

    # CAGR proxy: net_pnl / avg_entry over period years
    period_days = (pd.Timestamp(PERIOD_END) - pd.Timestamp(PERIOD_START)).days
    years = max(period_days / 365.25, 0.5)
    total_return_pct = net_pnl / avg_entry * 100.0 if avg_entry > 0 else 0.0
    cagr = (((1 + total_return_pct / 100.0) ** (1.0 / years)) - 1.0) * 100.0

    calmar = cagr / max_dd_pct if max_dd_pct > 0 else 0.0

    return dict(
        trades=n,
        wins=n_wins,
        losses=n_losses,
        win_rate_pct=round(win_rate, 2),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        profit_factor=round(pf, 3),
        sharpe=round(sharpe, 2),
        max_dd_pct=round(max_dd_pct, 2),
        calmar=round(calmar, 2),
        net_pnl=round(net_pnl, 2),
        cagr_pct=round(cagr, 2),
        long_trades=len(long_t),
        short_trades=len(short_t),
        long_wr=round(long_wr, 2),
        short_wr=round(short_wr, 2),
    )


# =============================================================================
# Cell definitions
# =============================================================================

def make_base_config() -> FilterORBConfig:
    """Live ORB baseline (matches config.py ORB_DEFAULTS at sweep time)."""
    return FilterORBConfig(
        or_minutes=15,
        last_entry_time='14:00',
        eod_exit_time='15:18',
        max_trades_per_day=1,
        allow_longs=True,
        allow_shorts=True,
        sl_type='or_opposite',
        target_type='r_multiple',
        r_multiple=1.5,
        use_vwap_filter=True,
        use_rsi_filter=True,
        rsi_long_threshold=60.0,
        rsi_short_threshold=40.0,
        use_cpr_dir_filter=True,
        use_cpr_width_filter=True,
        cpr_width_threshold_pct=0.65,
        # Note: gap filter is NOT in the dataclass — it's enforced at the
        # live-engine layer (cfg.get('use_gap_filter')); the backtest engine
        # has no gap filter. Acceptable: this sweep is filter-tuning, not
        # absolute-PnL. The relative ranking is robust.
        signal_age_max_mins=15,
        signal_drift_max_pct=0.005,
    )


def build_phase1_cells() -> List[Tuple[str, FilterORBConfig]]:
    """20 cells: 4x4 RSI_short × age, plus 4x1 RSI_long with age=15."""
    cells = []
    base = make_base_config()

    # Baseline first (for reference)
    cells.append(('baseline_60_40_age15_drift05', replace(base)))

    # 4x4 grid: rsi_short × signal_age_max_mins (rsi_long stays 60)
    for rsi_s in (40, 42, 45, 48):
        for age in (15, 20, 30, 45):
            label = f'short_rsi{rsi_s}_age{age}'
            c = replace(base, rsi_short_threshold=float(rsi_s),
                        signal_age_max_mins=int(age))
            cells.append((label, c))

    # 4x1 grid: rsi_long ∈ {52,55,58,60}, age=15
    for rsi_l in (52, 55, 58, 60):
        label = f'long_rsi{rsi_l}_age15'
        c = replace(base, rsi_long_threshold=float(rsi_l),
                    signal_age_max_mins=15)
        cells.append((label, c))

    return cells


def build_phase2_cells(winner_cfg: FilterORBConfig) -> List[Tuple[str, FilterORBConfig]]:
    """4 cells: signal_drift_max_pct sweep, holding winning RSI/age."""
    cells = []
    for drift in (0.005, 0.0085, 0.012, 0.015):
        label = (f'drift{drift:.4f}_rsi_l{int(winner_cfg.rsi_long_threshold)}'
                 f'_rsi_s{int(winner_cfg.rsi_short_threshold)}'
                 f'_age{winner_cfg.signal_age_max_mins}')
        c = replace(winner_cfg, signal_drift_max_pct=drift)
        cells.append((label, c))
    return cells


# =============================================================================
# Main
# =============================================================================

def filter_data_by_period(data: dict, start_date: str, end_date: str) -> dict:
    out = {}
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    for tf in ('5min', 'day'):
        df = data[tf]
        mask = (df['date'] >= start_ts) & (df['date'] <= end_ts)
        out[tf] = df[mask].reset_index(drop=True).copy()
    return out


def run_cell(label: str, cfg: FilterORBConfig, data_by_sym: Dict) -> Dict:
    """Run one cell across all symbols, return aggregated row."""
    engine = ORBFilterEngine(cfg)
    per_sym = {}
    for sym, sym_data in data_by_sym.items():
        try:
            r = engine.run(sym, sym_data)
            per_sym[sym] = r
        except Exception as e:
            print(f'    ERROR {sym}: {e}', flush=True)
    agg = aggregate_results(per_sym)
    return dict(
        cell_label=label,
        rsi_long=cfg.rsi_long_threshold,
        rsi_short=cfg.rsi_short_threshold,
        signal_age_max=cfg.signal_age_max_mins,
        signal_drift_max=cfg.signal_drift_max_pct,
        **agg,
    )


def write_csv_header(path: Path):
    with path.open('w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def append_csv_row(path: Path, row: Dict):
    with path.open('a', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(
            {k: row.get(k, '') for k in FIELDNAMES}
        )


def load_existing(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open() as f:
        return {row['cell_label'] for row in csv.DictReader(f)}


def composite_score(row: Dict) -> float:
    """Higher = better. Used for ranking cells.
    PF*sharpe + 0.001*trades - 0.5*max(0, max_dd_pct - 5).
    """
    pf = row['profit_factor']
    sh = row['sharpe']
    n = row['trades']
    dd = row['max_dd_pct']
    return pf * sh + 0.001 * n - 0.5 * max(0.0, dd - 5.0)


def main():
    t_start = time.time()

    print('=' * 70)
    print('ORB Filter Tuning Sweep — Phase 1')
    print('=' * 70)

    out_csv = OUT / 'summary.csv'
    print(f'Output CSV: {out_csv}')

    # Header (write only if file does not exist)
    if not out_csv.exists():
        write_csv_header(out_csv)

    done = load_existing(out_csv)
    if done:
        print(f'Resuming — {len(done)} cells already complete: {sorted(done)}')

    # ---- Preload data ----
    print(f'\nPreloading 5-min + daily data for {len(UNIVERSE)} symbols...')
    raw_data = ORBBacktestEngine.preload_data(UNIVERSE)
    print(f'Preload done in {time.time()-t_start:.1f}s')

    # Filter to period
    data_by_sym = {sym: filter_data_by_period(raw_data[sym], PERIOD_START, PERIOD_END)
                   for sym in UNIVERSE if sym in raw_data}
    print(f'Period: {PERIOD_START} to {PERIOD_END} | {len(data_by_sym)} symbols ready')

    # ---- Phase 1 cells ----
    cells = build_phase1_cells()
    cells_to_run = [c for c in cells if c[0] not in done]
    print(f'\nPhase 1 plan: {len(cells)} cells total, {len(cells_to_run)} to run')
    print(f'Header: {FIELDNAMES}\n')

    rows_p1 = []
    # Re-load already-completed rows so we can rank at the end
    if done:
        with out_csv.open() as f:
            for row in csv.DictReader(f):
                if row['cell_label'] in done:
                    rows_p1.append({
                        k: (float(v) if k not in ('cell_label',) and v not in ('', None)
                            else v) for k, v in row.items()
                    })

    for idx, (label, cfg) in enumerate(cells_to_run, 1):
        t0 = time.time()
        print(f'[{idx}/{len(cells_to_run)}] {label} '
              f'(rsi_l={cfg.rsi_long_threshold:.0f} rsi_s={cfg.rsi_short_threshold:.0f} '
              f'age={cfg.signal_age_max_mins} drift={cfg.signal_drift_max_pct}) ...',
              end='', flush=True)
        row = run_cell(label, cfg, data_by_sym)
        elapsed = time.time() - t0
        print(f' {elapsed:.0f}s | trades={row["trades"]} '
              f'WR={row["win_rate_pct"]:.1f} PF={row["profit_factor"]:.2f} '
              f'Sharpe={row["sharpe"]:.2f} DD={row["max_dd_pct"]:.2f} '
              f'NetPnl={row["net_pnl"]:.0f}', flush=True)
        append_csv_row(out_csv, row)
        rows_p1.append(row)

    # ---- Phase 1 ranking ----
    print('\n' + '=' * 70)
    print('PHASE 1 RANKING (by composite score)')
    print('=' * 70)
    ranked = sorted(rows_p1, key=composite_score, reverse=True)
    print(f'{"#":>3} {"Cell":<30} {"Trd":>5} {"WR%":>6} {"PF":>5} '
          f'{"Sh":>5} {"DD%":>5} {"NetP":>8} {"Score":>7}')
    print('-' * 80)
    for i, r in enumerate(ranked, 1):
        print(f'{i:>3} {str(r["cell_label"]):<30} {int(r["trades"]):>5} '
              f'{float(r["win_rate_pct"]):>6.1f} {float(r["profit_factor"]):>5.2f} '
              f'{float(r["sharpe"]):>5.2f} {float(r["max_dd_pct"]):>5.2f} '
              f'{float(r["net_pnl"]):>8.0f} {composite_score(r):>7.2f}')

    # Identify baseline + winner
    baseline = next(r for r in rows_p1 if r['cell_label'] == 'baseline_60_40_age15_drift05')
    bp_pf = float(baseline['profit_factor'])
    bp_dd = float(baseline['max_dd_pct'])
    bp_n = int(baseline['trades'])

    print('\n' + '=' * 70)
    print('BASELINE COMPARISON')
    print('=' * 70)
    print(f'Baseline: rsi_l=60, rsi_s=40, age=15, drift=0.005')
    print(f'  trades={bp_n}, PF={bp_pf:.2f}, DD={bp_dd:.2f}%, '
          f'sharpe={float(baseline["sharpe"]):.2f}')
    print(f'\nPass criteria: PF >= {bp_pf:.2f}, DD <= {bp_dd+2:.2f}%, '
          f'trades > {bp_n}\n')

    passing = []
    for r in ranked:
        if r['cell_label'] == 'baseline_60_40_age15_drift05':
            continue
        if (float(r['profit_factor']) >= bp_pf
            and float(r['max_dd_pct']) <= bp_dd + 2.0
            and int(r['trades']) > bp_n):
            passing.append(r)

    print(f'PASSING CELLS (vs baseline gate): {len(passing)}')
    for r in passing[:10]:
        delta_n = int(r['trades']) - bp_n
        delta_pf = float(r['profit_factor']) - bp_pf
        delta_dd = float(r['max_dd_pct']) - bp_dd
        print(f'  {str(r["cell_label"]):<30} trades=+{delta_n} '
              f'PF={delta_pf:+.2f} DD={delta_dd:+.2f}%')

    print(f'\nTotal runtime: {time.time()-t_start:.1f}s')
    print(f'Artifacts: {out_csv}')

    return rows_p1, baseline, passing


if __name__ == '__main__':
    sys.exit(main())
