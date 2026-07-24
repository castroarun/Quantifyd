#!/usr/bin/env python3
"""RSI momentum-regime SLOT-BASED PORTFOLIO engine (research/72, Phase C).

N equal-capital slots. Long-only, flat=cash. Per name:
  ENTRY: RSI(len) close >= entry_th  -> eligible to buy at that close if a slot is free
  EXIT : RSI(len) close <  exit_th   -> sell at that close, free the slot

Daily loop over the union trading calendar:
  1. mark open positions to market (close-to-close)
  2. process EXITS (RSI<exit) -> free slot, charge exit cost
  3. process ENTRIES: candidates = RSI>=entry not held; if more than free slots,
     rank by RSI desc, take strongest; buy at close, allocate NAV/N, charge entry cost.

Idle cash earns 0% (note: 6% yield would only help). Reports GROSS and NET.
Reuses rsi_regime_engine (E) for rsi/metrics/per_year/load.
"""
import sys, csv, os
from pathlib import Path
import numpy as np
import pandas as pd

SD = Path(__file__).resolve().parent
sys.path.insert(0, str(SD))
import rsi_regime_engine as E


def load_universe_prices(symbols, timeframe='day', min_start_year=2015, min_rows=1500):
    """Load each symbol's df once. Keep only names with data from <=start-of-2015
    and >= min_rows day-rows. Returns (price_data dict, rsi cache dict, qualified list)."""
    price = {}
    for s in symbols:
        df = E.load(s, timeframe=timeframe)
        if df.empty:
            continue
        # drop bad prints
        df = df[df['close'] > 0]
        if len(df) < min_rows:
            continue
        if df.index[0] > pd.Timestamp(f'{min_start_year}-12-31'):
            # must have data reaching back to at least end-2015 to be a modern-period proxy
            continue
        price[s] = df
    return price


def run_portfolio(price_data, entry_th, exit_th, rsi_len, N, cost_bps=15.0,
                  start=None, end=None):
    """Slot-based portfolio. Returns (gross_eqs, net_eqs, stats)."""
    # union calendar
    all_dates = sorted(set().union(*[set(df.index) for df in price_data.values()]))
    all_dates = pd.DatetimeIndex(all_dates)
    if start: all_dates = all_dates[all_dates >= pd.Timestamp(start)]
    if end:   all_dates = all_dates[all_dates <= pd.Timestamp(end)]

    # Precompute RSI + close series per name.
    # close_map = REAL bars only (used for entry/exit decisions and buy/sell price).
    # ff_map    = forward-filled to union calendar (used for mark-to-market only), so a
    #             held name with no print on a special/half-day session is carried at its
    #             last real close instead of vanishing from NAV.
    rsi_map = {}
    close_map = {}
    ff_map = {}
    for s, df in price_data.items():
        rsi_map[s] = E.rsi(df['close'], rsi_len)
        close_map[s] = df['close']
        ff_map[s] = df['close'].reindex(all_dates).ffill()

    rt = cost_bps / 10000.0  # charged on each side's notional

    # Two parallel books: gross (no cost) and net (cost). Same trade decisions.
    # State per book: cash, positions {sym: (shares, )}. We track shares so
    # mark-to-market is exact close-to-close.
    def new_book():
        return dict(cash=1.0, pos={})  # pos: sym -> shares (in NAV units)

    gross = new_book()
    net = new_book()

    def nav(book, dt):
        v = book['cash']
        for s, sh in book['pos'].items():
            px = ff_map[s].get(dt, np.nan)  # carry-forward price for MTM
            if not np.isnan(px):
                v += sh * px
        return v

    gross_curve = []
    net_curve = []
    n_trades = 0  # count round trips on net book (entries)

    prev_dt = None
    for dt in all_dates:
        # ---- 1. positions are marked automatically via shares*price in nav() ----
        # ---- 2. EXITS ----
        for book, is_net in ((gross, False), (net, True)):
            to_exit = []
            for s in list(book['pos'].keys()):
                rv = rsi_map[s].get(dt, np.nan)
                px = close_map[s].get(dt, np.nan)
                if np.isnan(px) or px <= 0:
                    continue  # no valid print today; hold (skip name-day)
                if not np.isnan(rv) and rv < exit_th:
                    to_exit.append((s, px))
            for s, px in to_exit:
                sh = book['pos'].pop(s)
                proceeds = sh * px
                if is_net:
                    proceeds *= (1 - rt)  # exit cost
                book['cash'] += proceeds

        # ---- 3. ENTRIES ----
        for book, is_net in ((gross, False), (net, True)):
            free = N - len(book['pos'])
            if free <= 0:
                continue
            # candidates: RSI>=entry, not held, valid price today
            cands = []
            for s in price_data.keys():
                if s in book['pos']:
                    continue
                rv = rsi_map[s].get(dt, np.nan)
                px = close_map[s].get(dt, np.nan)
                if np.isnan(rv) or np.isnan(px) or px <= 0:
                    continue
                if rv >= entry_th:
                    cands.append((rv, s, px))
            if not cands:
                continue
            cands.sort(key=lambda x: x[0], reverse=True)  # strongest RSI first
            take = cands[:free]
            cur_nav = nav(book, dt)
            alloc = cur_nav / N
            for rv, s, px in take:
                if alloc <= 0 or book['cash'] < alloc * 0.999:
                    # allocate min(alloc, cash) to avoid going negative
                    alloc_use = min(alloc, book['cash'])
                else:
                    alloc_use = alloc
                if alloc_use <= 0:
                    continue
                cost_notional = alloc_use
                shares = alloc_use / px
                if is_net:
                    # entry cost: buy fewer shares effectively (charge on notional)
                    shares = (alloc_use * (1 - rt)) / px
                    n_trades += 1
                book['pos'][s] = book['pos'].get(s, 0.0) + shares
                book['cash'] -= alloc_use

        gross_curve.append((dt, nav(gross, dt)))
        net_curve.append((dt, nav(net, dt)))
        prev_dt = dt

    gross_eqs = pd.Series(dict(gross_curve))
    net_eqs = pd.Series(dict(net_curve))
    yrs = (net_eqs.index[-1] - net_eqs.index[0]).days / 365.25
    trades_per_yr = n_trades / yrs if yrs > 0 else np.nan
    return gross_eqs, net_eqs, dict(n_entries=n_trades, trades_per_yr=round(trades_per_yr, 1),
                                    n_days=len(all_dates))
