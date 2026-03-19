#!/usr/bin/env python3
"""Debug Trade #1: ITC LONG 2024-04-26 — show exactly why this trade was taken."""

import sys, logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
sys.path.insert(0, str(Path(__file__).parent))
logging.disable(logging.WARNING)

from services.cpr_intraday_engine import CPRIntradayEngine, CPRIntradayConfig

config = CPRIntradayConfig(
    start_date='2024-01-01',
    end_date='2025-10-27',
    narrow_cpr_threshold=0.5,
    cpr_proximity_pct=3.0,
    max_wick_pct=30.0,
    st_period=7,
    st_multiplier=4.0,
)

print("Loading data...")
daily_data, five_min_data = CPRIntradayEngine.preload_data(
    symbols=['ITC'], start_date='2024-01-01', end_date='2025-10-27'
)

engine = CPRIntradayEngine(config, preloaded_daily=daily_data, preloaded_5min=five_min_data)

# ---- WEEKLY CPR for week of 2024-04-22 (Mon) ----
# Need prior week's (Apr 15-19) OHLC to compute CPR
itc_daily = daily_data['ITC'].copy()
itc_daily.index = pd.to_datetime(itc_daily.index if isinstance(itc_daily.index[0], str) else itc_daily['date'] if 'date' in itc_daily.columns else itc_daily.index)

# Get daily data and compute weekly CPR
print("\n" + "="*80)
print("TRADE #1 DEEP DIVE: ITC LONG on 2024-04-26 (Friday)")
print("="*80)

# Weekly CPR calculation
print("\n--- STEP 1: Weekly CPR (from prior week Apr 15-19) ---")
# Filter prior week
if 'date' in itc_daily.columns:
    itc_daily['dt'] = pd.to_datetime(itc_daily['date'])
else:
    itc_daily['dt'] = itc_daily.index

prior_week = itc_daily[(itc_daily['dt'] >= '2024-04-15') & (itc_daily['dt'] <= '2024-04-19')]
print(f"Prior week daily candles (Apr 15-19):")
for _, row in prior_week.iterrows():
    print(f"  {str(row['dt'])[:10]}: O={row['open']:.2f} H={row['high']:.2f} L={row['low']:.2f} C={row['close']:.2f}")

week_high = prior_week['high'].max()
week_low = prior_week['low'].min()
week_close = prior_week.iloc[-1]['close']

pivot = (week_high + week_low + week_close) / 3
bc = (week_high + week_low) / 2
tc = (pivot - bc) + pivot
cpr_width = abs(tc - bc)
cpr_width_pct = (cpr_width / pivot) * 100

print(f"\nWeekly CPR:")
print(f"  Week H={week_high:.2f}, L={week_low:.2f}, C={week_close:.2f}")
print(f"  Pivot = {pivot:.2f}")
print(f"  TC = {tc:.2f}")
print(f"  BC = {bc:.2f}")
print(f"  CPR Width = {cpr_width:.2f} ({cpr_width_pct:.3f}%)")
print(f"  Threshold = 0.5% => {'PASS (narrow)' if cpr_width_pct < 0.5 else 'FAIL (too wide)'}")

# ---- Monday's first 30-min candle ----
print("\n--- STEP 2: Monday's First 30-min Candle (bias setting) ---")
itc_5min = five_min_data['ITC'].copy()
if 'date' in itc_5min.columns:
    itc_5min['dt'] = pd.to_datetime(itc_5min['date'])
else:
    itc_5min['dt'] = pd.to_datetime(itc_5min.index)

# Week of Apr 22 — find Monday
# Apr 22 is Monday
mon_candles = itc_5min[(itc_5min['dt'] >= '2024-04-22 09:15:00') & (itc_5min['dt'] < '2024-04-22 09:45:00')]
print(f"Monday 2024-04-22, 5-min candles (9:15-9:40):")
for _, row in mon_candles.iterrows():
    print(f"  {str(row['dt'])}: O={row['open']:.2f} H={row['high']:.2f} L={row['low']:.2f} C={row['close']:.2f}")

if len(mon_candles) > 0:
    # Aggregate to 30-min candle
    candle_o = mon_candles.iloc[0]['open']
    candle_h = mon_candles['high'].max()
    candle_l = mon_candles['low'].min()
    candle_c = mon_candles.iloc[-1]['close']

    print(f"\n30-min candle (9:15-9:45):")
    print(f"  O={candle_o:.2f} H={candle_h:.2f} L={candle_l:.2f} C={candle_c:.2f}")

    # Clean candle check
    body = abs(candle_c - candle_o)
    total_range = candle_h - candle_l
    if body > 0 and total_range > 0:
        if candle_c >= candle_o:  # Green
            upper_wick = candle_h - candle_c
            lower_wick = candle_o - candle_l
        else:  # Red
            upper_wick = candle_h - candle_o
            lower_wick = candle_c - candle_l

        max_wick = max(upper_wick, lower_wick)
        wick_pct = (max_wick / body) * 100 if body > 0 else 999

        direction = "GREEN (LONG bias)" if candle_c > candle_o else "RED (SHORT bias)"
        print(f"  Direction: {direction}")
        print(f"  Body = {body:.2f}, Upper wick = {upper_wick:.2f}, Lower wick = {lower_wick:.2f}")
        print(f"  Max wick/body = {wick_pct:.1f}%")
        print(f"  Threshold = 30% => {'PASS (clean)' if wick_pct <= 30 else 'FAIL (too wicky)'}")

        # Proximity check
        dist_to_pivot = abs(candle_c - pivot) / pivot * 100
        dist_to_tc = abs(candle_c - tc) / tc * 100
        dist_to_bc = abs(candle_c - bc) / bc * 100
        min_dist = min(dist_to_pivot, dist_to_tc, dist_to_bc)
        nearest = 'Pivot' if min_dist == dist_to_pivot else ('TC' if min_dist == dist_to_tc else 'BC')

        print(f"\n  Proximity to CPR:")
        print(f"    Close={candle_c:.2f} vs Pivot={pivot:.2f} (dist={dist_to_pivot:.3f}%)")
        print(f"    Close={candle_c:.2f} vs TC={tc:.2f} (dist={dist_to_tc:.3f}%)")
        print(f"    Close={candle_c:.2f} vs BC={bc:.2f} (dist={dist_to_bc:.3f}%)")
        print(f"    Nearest: {nearest} at {min_dist:.3f}%")
        print(f"    Threshold = 3.0% => {'PASS' if min_dist <= 3.0 else 'FAIL'}")
else:
    print("  No 5-min data found for Monday Apr 22!")
    # Check if Apr 22 was a holiday — try Apr 23
    print("  Checking if it was a trading holiday...")
    tue_candles = itc_5min[(itc_5min['dt'] >= '2024-04-22') & (itc_5min['dt'] < '2024-04-27')]
    if len(tue_candles) > 0:
        first_day = tue_candles.iloc[0]['dt']
        print(f"  First trading day this week: {first_day}")

# ---- SuperTrend on 5-min for Apr 26 ----
print("\n--- STEP 3: SuperTrend on 5-min (Trade Day: Apr 26) ---")

# Get 5-min data around the trade day with enough history for ST calculation
st_data = itc_5min[(itc_5min['dt'] >= '2024-04-01') & (itc_5min['dt'] <= '2024-04-26 15:30:00')].copy()
st_data = st_data.set_index('dt') if 'dt' in st_data.columns else st_data

# Calculate SuperTrend manually
st_result = engine.calculate_supertrend(st_data.copy(), period=7, multiplier=4.0)

# Show the flips on Apr 26
apr26_st = st_result[st_result.index >= '2024-04-26 09:15:00']
print(f"SuperTrend on 2024-04-26 (first few candles):")
for idx, row in apr26_st.head(15).iterrows():
    flip = ""
    if row.get('st_flip_up', False):
        flip = " *** FLIP UP (BUY SIGNAL) ***"
    elif row.get('st_flip_down', False):
        flip = " *** FLIP DOWN (SELL SIGNAL) ***"
    st_dir = row.get('st_direction', '?')
    st_val = row.get('supertrend', 0)
    print(f"  {idx} | C={row['close']:.2f} | ST={st_val:.2f} | Dir={'UP' if st_dir == 1 else 'DOWN'}{flip}")

# Show the entry candle specifically
print(f"\n--- STEP 4: Entry & Exit ---")
entry_candles = itc_5min[(itc_5min['dt'] >= '2024-04-26 09:15:00') & (itc_5min['dt'] <= '2024-04-26 09:15:00')]
if len(entry_candles) > 0:
    ec = entry_candles.iloc[0]
    print(f"Entry candle: {ec['dt']} | O={ec['open']:.2f} H={ec['high']:.2f} L={ec['low']:.2f} C={ec['close']:.2f}")

exit_candles = itc_5min[(itc_5min['dt'] >= '2024-04-26 15:20:00') & (itc_5min['dt'] <= '2024-04-26 15:20:00')]
if len(exit_candles) > 0:
    xc = exit_candles.iloc[0]
    print(f"Exit candle:  {xc['dt']} | O={xc['open']:.2f} H={xc['high']:.2f} L={xc['low']:.2f} C={xc['close']:.2f}")
    print(f"Direction: LONG | Entry ~{382.24:.2f} (with slippage) | Exit ~{380.41:.2f} | PnL: -2,970")

# Check if bias was set on Monday or another day
print(f"\n--- STEP 5: Bias Timeline for This Week ---")
week_5min = itc_5min[(itc_5min['dt'] >= '2024-04-22') & (itc_5min['dt'] < '2024-04-27')]
if len(week_5min) > 0:
    trading_days = sorted(week_5min['dt'].dt.date.unique())
    print(f"Trading days this week: {[str(d) for d in trading_days]}")

    # Show first 30-min candle for the first trading day (bias day)
    first_day = str(trading_days[0])
    bias_candles = week_5min[(week_5min['dt'] >= f'{first_day} 09:15:00') & (week_5min['dt'] < f'{first_day} 09:45:00')]
    if len(bias_candles) > 0:
        bo = bias_candles.iloc[0]['open']
        bh = bias_candles['high'].max()
        bl = bias_candles['low'].min()
        bc_val = bias_candles.iloc[-1]['close']
        print(f"Bias-setting day: {first_day}")
        print(f"  30-min candle: O={bo:.2f} H={bh:.2f} L={bl:.2f} C={bc_val:.2f}")
        print(f"  {'GREEN => LONG bias' if bc_val > bo else 'RED => SHORT bias'}")
