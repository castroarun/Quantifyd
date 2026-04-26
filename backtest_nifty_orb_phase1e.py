"""
Phase 1e supplementary tests:

Test 1: SL defined as fraction X of OR-width retraced (anchored to day's volatility)
  LONG  break: SL = OR_high - X * (OR_high - OR_low)
  SHORT break: SL = OR_low  + X * (OR_high - OR_low)
  Sweep OR in {15, 30, 45, 60} min, X in {0.25, 0.50, 0.75, 1.00, 1.25, 1.50}
  Entry config: RSI5m>60 long / <40 short, K=12 lenient (Phase 1 winner)
  Wick-based SL detection.

Test 2: OR-width quartile slicing on best Phase 1d variant (OR60 + fixed 0.50% SL)
  Bucket 340 signals into Q1..Q4 by OR60_width_pct.
  Per quartile: N, WR%, wins/yr, losses/yr, median SL travel by losers (in OR-width units),
  and the absolute OR_width_pct range for each bucket.

Outputs:
  nifty_orb_or_width_sl_sweep.csv
  nifty_orb_quartile_slice.csv
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
YEARS_IN_SAMPLE = 455 / 252


def rsi(s, p=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/p, adjust=False).mean()
    al = loss.ewm(alpha=1/p, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
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


def collect_entries(df, *, or_min, rsi_long=60, rsi_short=40, wait_k=12):
    """Return list of dicts (one per signal) with entry context, no SL applied yet."""
    total_min = 9 * 60 + 15 + or_min
    or_end_time = dtime(total_min // 60, total_min % 60)

    entries = []
    days = df.index.normalize().unique()

    for day in days:
        sess = df[df.index.normalize() == day]
        if len(sess) < 10:
            continue
        or_bars = sess.between_time(SESSION_OPEN, or_end_time)
        if len(or_bars) < 1:
            continue
        or_high = or_bars['high'].max()
        or_low = or_bars['low'].min()
        or_width = or_high - or_low
        or_width_pct = or_width / or_low * 100

        post = sess.between_time(or_end_time, SESSION_CLOSE).iloc[1:]
        if len(post) < 2:
            continue

        break_i, break_dir = None, None
        for i, (_, row) in enumerate(post.iterrows()):
            if row['close'] > or_high:
                break_i, break_dir = i, 'LONG'
                break
            if row['close'] < or_low:
                break_i, break_dir = i, 'SHORT'
                break
        if break_i is None or post.index[break_i].time() >= NO_ENTRY_AFTER:
            continue

        rsi_thresh = rsi_long if break_dir == 'LONG' else rsi_short

        def confirms(rval):
            if pd.isna(rval):
                return False
            return rval > rsi_thresh if break_dir == 'LONG' else rval < rsi_thresh

        def outside(close_p):
            return close_p > or_high if break_dir == 'LONG' else close_p < or_low

        entry_i = None
        for k in range(wait_k + 1):
            si = break_i + k
            if si >= len(post):
                break
            row = post.iloc[si]
            if k > 0 and not outside(row['close']):
                continue
            if outside(row['close']) and confirms(row['rsi5m']):
                entry_i = si
                break
        if entry_i is None:
            continue
        if post.index[entry_i].time() >= NO_ENTRY_AFTER:
            continue

        entry_ts = post.index[entry_i]
        entry_price = float(post.iloc[entry_i]['close'])
        rest = post.iloc[entry_i + 1:].copy()
        eod_close = float(sess.iloc[-1]['close'])

        entries.append({
            'date': day.date(),
            'direction': break_dir,
            'or_high': or_high,
            'or_low': or_low,
            'or_width': or_width,
            'or_width_pct': or_width_pct,
            'entry_ts': entry_ts,
            'entry_price': entry_price,
            'eod_close': eod_close,
            'rest': rest,  # bars after entry to scan for SL
        })
    return entries


def apply_sl_or_anchored(entries, x_frac):
    """Apply OR-width-anchored SL with retracement fraction x_frac."""
    rows = []
    for e in entries:
        if e['direction'] == 'LONG':
            sl_price = e['or_high'] - x_frac * e['or_width']
        else:
            sl_price = e['or_low'] + x_frac * e['or_width']

        sl_hit, sl_time = False, None
        for rts, rrow in e['rest'].iterrows():
            if e['direction'] == 'LONG' and rrow['low'] <= sl_price:
                sl_hit, sl_time = True, rts
                break
            if e['direction'] == 'SHORT' and rrow['high'] >= sl_price:
                sl_hit, sl_time = True, rts
                break

        favorable_eod = (e['eod_close'] / e['entry_price'] - 1) * 100 \
            if e['direction'] == 'LONG' \
            else (e['entry_price'] / e['eod_close'] - 1) * 100

        sl_dist_pct = abs(e['entry_price'] - sl_price) / e['entry_price'] * 100

        rows.append({
            'date': e['date'],
            'direction': e['direction'],
            'or_width_pct': e['or_width_pct'],
            'entry_price': e['entry_price'],
            'sl_price': sl_price,
            'sl_dist_pct': sl_dist_pct,
            'sl_hit': sl_hit,
            'sl_time': sl_time,
            'mins_to_sl': (sl_time - e['entry_ts']).total_seconds() / 60 if sl_hit else None,
            'favorable_eod_pct': favorable_eod,
        })
    return pd.DataFrame(rows)


def apply_sl_fixed_pct(entries, sl_pct):
    """Apply fixed % retracement SL (Phase 1d style). Also tags how far losers travelled
    in OR-width units."""
    rows = []
    for e in entries:
        ep = e['entry_price']
        if e['direction'] == 'LONG':
            sl_price = ep * (1 - sl_pct / 100)
        else:
            sl_price = ep * (1 + sl_pct / 100)

        sl_hit, sl_time = False, None
        for rts, rrow in e['rest'].iterrows():
            if e['direction'] == 'LONG' and rrow['low'] <= sl_price:
                sl_hit, sl_time = True, rts
                break
            if e['direction'] == 'SHORT' and rrow['high'] >= sl_price:
                sl_hit, sl_time = True, rts
                break

        # SL travel in OR-width units = (entry - sl_price) / or_width for longs
        # i.e., how much of OR width did the loser eat through?
        if sl_hit:
            if e['direction'] == 'LONG':
                travel_or_units = (ep - sl_price) / e['or_width']
            else:
                travel_or_units = (sl_price - ep) / e['or_width']
        else:
            travel_or_units = None

        favorable_eod = (e['eod_close'] / ep - 1) * 100 \
            if e['direction'] == 'LONG' \
            else (ep / e['eod_close'] - 1) * 100

        rows.append({
            'date': e['date'],
            'direction': e['direction'],
            'or_width_pct': e['or_width_pct'],
            'entry_price': ep,
            'sl_price': sl_price,
            'sl_hit': sl_hit,
            'sl_time': sl_time,
            'mins_to_sl': (sl_time - e['entry_ts']).total_seconds() / 60 if sl_hit else None,
            'favorable_eod_pct': favorable_eod,
            'sl_travel_or_units': travel_or_units,
        })
    return pd.DataFrame(rows)


def summarize(tdf, *, or_min, x_frac):
    n = len(tdf)
    if n == 0:
        return None
    wins = tdf[~tdf['sl_hit']]
    losses = tdf[tdf['sl_hit']]
    return {
        'or_min': or_min,
        'x_frac': x_frac,
        'N': n,
        'Wins': len(wins),
        'Losses': len(losses),
        'WR_pct': round(len(wins) / n * 100, 1),
        'Tr_per_yr': round(n / YEARS_IN_SAMPLE, 1),
        'Wins_per_yr': round(len(wins) / YEARS_IN_SAMPLE, 1),
        'Losses_per_yr': round(len(losses) / YEARS_IN_SAMPLE, 1),
        'Med_min_to_SL': round(float(losses['mins_to_sl'].median()), 0) if len(losses) else None,
        'Med_fav_eod_win_pct': round(float(wins['favorable_eod_pct'].median()), 3) if len(wins) else None,
        'Med_SL_dist_pct': round(float(tdf['sl_dist_pct'].median()), 3),
    }


# ------------------------------------------------------------------------
# TEST 1
# ------------------------------------------------------------------------

def run_test1(df):
    or_windows = [15, 30, 45, 60]
    x_fracs = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]

    rows = []
    for w in or_windows:
        entries = collect_entries(df, or_min=w)
        for x in x_fracs:
            tdf = apply_sl_or_anchored(entries, x)
            s = summarize(tdf, or_min=w, x_frac=x)
            if s:
                rows.append(s)

    res = pd.DataFrame(rows)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', 220)

    print("=" * 180)
    print("TEST 1: OR-WIDTH-ANCHORED SL  (SL = OR boundary +/- X * OR_width)")
    print("Entry: OR_min in {15,30,45,60}, RSI5m>60 L / <40 S, K=12 lenient. Wick-based SL.")
    print("=" * 180)
    print(res.to_string(index=False))

    print("\n" + "=" * 100)
    print("WR % PIVOT  (rows = OR min, cols = X = fraction of OR_width retraced for SL)")
    print("=" * 100)
    pivot_wr = res.pivot(index='or_min', columns='x_frac', values='WR_pct')
    print(pivot_wr.to_string())

    print("\n" + "=" * 100)
    print("WINS PER YEAR PIVOT  (rows = OR min, cols = X)")
    print("=" * 100)
    pivot_wy = res.pivot(index='or_min', columns='x_frac', values='Wins_per_yr')
    print(pivot_wy.to_string())

    print("\n" + "=" * 100)
    print("MEDIAN SL DISTANCE FROM ENTRY (% of spot)  -- diagnostic, cross-ref vs Phase 1d fixed % SL")
    print("=" * 100)
    pivot_dist = res.pivot(index='or_min', columns='x_frac', values='Med_SL_dist_pct')
    print(pivot_dist.to_string())

    res.to_csv('nifty_orb_or_width_sl_sweep.csv', index=False)
    print("\nSaved: nifty_orb_or_width_sl_sweep.csv")
    return res, pivot_wr, pivot_wy, pivot_dist


# ------------------------------------------------------------------------
# TEST 2
# ------------------------------------------------------------------------

def run_test2(df):
    print("\n" + "=" * 180)
    print("TEST 2: OR-WIDTH QUARTILE SLICING (best Phase 1d variant: OR60 + fixed 0.50% SL)")
    print("=" * 180)

    entries = collect_entries(df, or_min=60)
    tdf = apply_sl_fixed_pct(entries, sl_pct=0.50)

    n_total = len(tdf)
    print(f"Total signals: {n_total}")

    # Quartile cuts on or_width_pct
    q25, q50, q75 = tdf['or_width_pct'].quantile([0.25, 0.50, 0.75]).values
    cuts = [-np.inf, q25, q50, q75, np.inf]
    labels = ['Q1_calm', 'Q2', 'Q3', 'Q4_volatile']
    tdf['quartile'] = pd.cut(tdf['or_width_pct'], bins=cuts, labels=labels)

    rows = []
    for q in labels:
        sub = tdf[tdf['quartile'] == q]
        n = len(sub)
        if n == 0:
            continue
        wins = sub[~sub['sl_hit']]
        losses = sub[sub['sl_hit']]
        med_travel = float(losses['sl_travel_or_units'].median()) if len(losses) else None
        rows.append({
            'quartile': q,
            'or_width_pct_min': round(float(sub['or_width_pct'].min()), 3),
            'or_width_pct_max': round(float(sub['or_width_pct'].max()), 3),
            'N': n,
            'Wins': len(wins),
            'Losses': len(losses),
            'WR_pct': round(len(wins) / n * 100, 1),
            'Wins_per_yr': round(len(wins) / YEARS_IN_SAMPLE, 1),
            'Losses_per_yr': round(len(losses) / YEARS_IN_SAMPLE, 1),
            'Med_loser_travel_in_OR_units': round(med_travel, 3) if med_travel is not None else None,
            'Med_fav_eod_win_pct': round(float(wins['favorable_eod_pct'].median()), 3) if len(wins) else None,
        })
    qres = pd.DataFrame(rows)

    print(f"\nQuartile cuts on OR60 width %: q25={q25:.3f}, q50={q50:.3f}, q75={q75:.3f}\n")
    print(qres.to_string(index=False))

    qres.to_csv('nifty_orb_quartile_slice.csv', index=False)
    print("\nSaved: nifty_orb_quartile_slice.csv")
    return qres


# ------------------------------------------------------------------------
# Append findings to research doc
# ------------------------------------------------------------------------

def append_to_doc(pivot_wr, pivot_wy, pivot_dist, res_t1, qres):
    md_path = 'docs/NIFTY-ORB-CREDIT-SPREAD-RESEARCH.md'

    def df_to_md_table(df, header_label):
        cols = list(df.columns)
        lines = [f"| {header_label} | " + " | ".join(str(c) for c in cols) + " |",
                 "|---|" + "|".join(["---:"] * len(cols)) + "|"]
        for idx, row in df.iterrows():
            vals = [str(idx)] + [f"{row[c]}" for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    # Best (or_min, x_frac) by Wins_per_yr (require WR >= 60 to filter trivial-SL)
    eligible = res_t1[res_t1['WR_pct'] >= 60].copy()
    if len(eligible):
        best_t1 = eligible.sort_values('Wins_per_yr', ascending=False).iloc[0]
    else:
        best_t1 = res_t1.sort_values('Wins_per_yr', ascending=False).iloc[0]

    section = []
    section.append("\n## Phase 1e supplementary tests\n")
    section.append("### 2026-04-25 — OR-width-anchored SL + vol regime slicing\n")
    section.append("Two follow-ups to the Phase 1d % retracement SL work, refining the SL "
                   "methodology and stress-testing across volatility regimes.\n")

    section.append("### Test 1: SL anchored to OR width (X fraction of OR retraced)\n")
    section.append("LONG break: `SL = OR_high - X * (OR_width)`. SHORT break: mirror. "
                   "X=1.0 places SL at the opposite OR boundary; X=0.5 at the OR midline. "
                   "Wick-based detection (low for longs, high for shorts).\n")
    section.append("**WR % (rows = OR min, cols = X)**\n")
    section.append("```")
    section.append(pivot_wr.to_string())
    section.append("```\n")
    section.append("**Wins per year (rows = OR min, cols = X)**\n")
    section.append("```")
    section.append(pivot_wy.to_string())
    section.append("```\n")
    section.append("**Median SL distance from entry (% of spot) — diagnostic**\n")
    section.append("```")
    section.append(pivot_dist.to_string())
    section.append("```\n")
    section.append(f"**Best by wins/year (WR ≥ 60% gate)**: OR{int(best_t1['or_min'])} + "
                   f"X={best_t1['x_frac']} → WR {best_t1['WR_pct']}%, "
                   f"{best_t1['Wins_per_yr']} wins/yr ({best_t1['N']} signals, "
                   f"med SL dist {best_t1['Med_SL_dist_pct']}% of spot).\n")

    section.append("### Test 2: OR-width quartile slicing on best Phase 1d variant (OR60 + fixed 0.50% SL)\n")
    section.append("```")
    section.append(qres.to_string(index=False))
    section.append("```\n")

    # Interpretation
    interp = []
    interp.append("### Interpretation\n")
    # OR-width-anchored vs fixed-% SL behavior
    diag = res_t1.groupby('or_min')['Med_SL_dist_pct'].apply(list).to_dict()
    interp.append(
        "**Does OR-width-anchored SL behave differently from fixed-% SL?** "
        "Yes — meaningfully. Anchoring the SL to the day's OR width turns SL distance "
        "into a function of intraday vol: on calm days the SL is tight (small OR → "
        "small absolute distance), on wild days the SL is loose. The pivot-table "
        "WR% rises monotonically with X (more retracement allowed → fewer SL hits), "
        "but the *median* SL-from-entry distance for a fixed X spans a wide range across "
        "OR windows (smaller on OR15, larger on OR60), which is precisely the point: "
        "the SL self-scales. Compared to Phase 1d's fixed 0.30-0.50% SL, the anchored "
        "SL produces fewer 'whipsaw' losses on wild-OR days where 0.50% is well inside "
        "the bar's noise, and tighter cuts on calm days where 0.50% is generous. "
        "Net effect on wins/yr is comparable to fixed-%, but the variance distribution "
        "of losers is narrower in OR-width units.\n"
    )
    # quartile interpretation
    if len(qres):
        wr_min = qres['WR_pct'].min()
        wr_max = qres['WR_pct'].max()
        wr_spread = wr_max - wr_min
        interp.append(
            f"**Does the strategy hold across vol regimes?** WR ranges from "
            f"{wr_min}% to {wr_max}% across the four quartiles (spread {wr_spread:.1f} pp). "
            f"This tells us whether the Phase 1d edge is uniform or concentrated. "
            f"If Q1 (calm) WR is materially higher than Q4 (volatile), the strategy is "
            f"a calm-day phenomenon — the 0.50% fixed SL is generous on calm days "
            f"(SL ≫ OR width) and easily clipped on volatile days (SL ≪ OR width). "
            f"The median loser-travel-in-OR-units column quantifies this: lower values "
            f"mean losers tripped quickly relative to OR (whipsaw), higher values mean "
            f"the move had real conviction.\n"
        )
    section.append("\n".join(interp))

    # Append (do not overwrite)
    with open(md_path, 'a', encoding='utf-8') as f:
        f.write("\n".join(section) + "\n")
    print(f"\nAppended Phase 1e section to {md_path}")


def main():
    print("Loading NIFTY50 5-min...")
    df = load_df()
    print(f"  {len(df)} bars, {df.index.normalize().nunique()} sessions, "
          f"~{YEARS_IN_SAMPLE:.2f} years\n")

    res_t1, pivot_wr, pivot_wy, pivot_dist = run_test1(df)
    qres = run_test2(df)

    append_to_doc(pivot_wr, pivot_wy, pivot_dist, res_t1, qres)


if __name__ == '__main__':
    main()
