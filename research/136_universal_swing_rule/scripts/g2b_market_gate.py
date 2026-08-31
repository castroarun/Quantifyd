"""G2b: the same book, with a MARKET brake instead of only a per-stock one.

G2 failed: every variant had a worse Sharpe than NIFTYBEES and drawdowns of
-42% to -55% against the index -36%, and 2015-2020 badly underperformed. The
per-trade edge is real but the trades cluster - everything is oversold at once -
so a 10-slot book is one leveraged bet on the market entered while it falls. The
stock-level 200-DMA does not help: in a crash a stock is above its own average
right up until you buy it.

This adds the brake that works on the momentum book already running: no NEW
entries while NIFTYBEES is below its own 200-DMA. Open positions still run to
their exit. One condition, no fitting, and it is the third study to point at
the same mechanism.

Original notes follow.
G2: run the signal as an actual book and compare it to owning the index.

G1c established the entry: RSI(2)<10 filtered to close>200-DMA, whose excess over
a date-matched random pick rises monotonically with horizon (+16 bps at 2 days to
+32 bps at 30), t 6.3-12.6 across 174,562 signals. Neither the unfiltered signal
nor a deeper oversold threshold shows that shape, so the trend filter is what
carries it.

Per-trade excess is not spendable. This runs the book — fixed slots, equal
notional, both legs charged, one position per name — and asks whether the result
beats NIFTYBEES, and whether it beats the same machinery fed random entries.

Everything is looked up through dicts built once; the first version scanned the
frame inside the loop and was hopeless.

Guards: entry at the next open, trailing features only, costs swept, random-entry
control on identical plumbing, and the period split for walk-forward.

Read-only.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logging.disable(logging.WARNING)

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
OUT = ROOT / 'research' / '136_universal_swing_rule' / 'results'
OUT.mkdir(parents=True, exist_ok=True)

START, END = '2015-01-01', '2026-08-29'
MIN_HISTORY, MIN_TURNOVER = 250, 5e7
CAPITAL, SLOTS = 1_000_000.0, 10
HOLDS = (10, 15, 20, 30)
COSTS = (20, 30)
SEED = 20260831

print('loading...', flush=True)
conn = sqlite3.connect(f"file:{ROOT / 'backtest_data' / 'market_data.db'}?mode=ro", uri=True)
df = pd.read_sql_query(
    "SELECT symbol, substr(date,1,10) d, open, close, volume FROM market_data_unified "
    "WHERE timeframe='day' AND date >= ? AND date <= ? ORDER BY symbol, date",
    conn, params=(START, END))
bench = pd.read_sql_query(
    "SELECT substr(date,1,10) d, close FROM market_data_unified WHERE timeframe='day' "
    "AND symbol='NIFTYBEES' AND date >= ? AND date <= ? ORDER BY date",
    conn, params=(START, END))
conn.close()

df['d'] = pd.to_datetime(df['d'])
df = df.sort_values(['symbol', 'd'])
g = df.groupby('symbol', sort=False)

df['turn20'] = g.apply(lambda x: (x['close'] * x['volume']).rolling(20, min_periods=20).median(),
                       include_groups=False).reset_index(level=0, drop=True)
df['bars'] = g.cumcount()
df['next_open'] = g['open'].shift(-1)
df['sma200'] = g['close'].transform(lambda s: s.rolling(200, min_periods=200).mean())


def _rsi2(s):
    dd = s.diff()
    up = dd.clip(lower=0).rolling(2).mean()
    dn = (-dd.clip(upper=0)).rolling(2).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


df['rsi2'] = g['close'].transform(_rsi2)
df['eligible'] = ((df['bars'] >= MIN_HISTORY) & (df['turn20'] >= MIN_TURNOVER)
                  & df['next_open'].notna() & (df['next_open'] > 0))
df['signal'] = df['eligible'] & (df['rsi2'] < 10) & (df['close'] > df['sma200'])
print(f'  {len(df):,} rows | {int(df.signal.sum()):,} signals', flush=True)

# ── one-time dict indexes: every loop lookup below is O(1) ──────────────────
OPEN = {(s, d): float(o) for s, d, o in zip(df.symbol, df.d, df.open) if o == o}
NOPEN = {(s, d): float(o) for s, d, o in zip(df.symbol, df.d, df.next_open) if o == o}
el = df[df.eligible]
dates = sorted(el.d.unique())
dates = [pd.Timestamp(x) for x in dates]
ELIG = {d: list(v) for d, v in el.groupby('d')['symbol']}
sg = df[df.signal]
SIG = {d: list(v) for d, v in sg.groupby('d')['symbol']}
RSI = {(s, d): float(r) for s, d, r in zip(sg.symbol, sg.d, sg.rsi2) if r == r}
print(f'  {len(dates):,} trading days indexed\n', flush=True)


def run(hold: int, cost_bps: float, random_entries: bool, seed: int = SEED):
    rng = np.random.default_rng(seed)
    cash, pos, curve = CAPITAL, [], []
    fee = cost_bps / 10000.0

    for i, d in enumerate(dates):
        keep = []
        for p in pos:
            if i >= p['exit_i'] and (p['symbol'], d) in OPEN:
                cash += p['qty'] * OPEN[(p['symbol'], d)] * (1 - fee)
            else:
                keep.append(p)
        pos = keep

        free = SLOTS - len(pos)
        if free > 0 and RISK_ON.get(d, False):   # no new entries in a falling market
            if random_entries:
                avail = ELIG.get(d, [])
                cands = list(rng.choice(avail, size=min(free, len(avail)), replace=False)) \
                    if avail else []
            else:
                cands = SIG.get(d, [])
                if len(cands) > free:
                    # most oversold first — no lookahead, no parameter
                    cands = sorted(cands, key=lambda s: RSI.get((s, d), 99))[:free]
            held = {p['symbol'] for p in pos}
            for sym in cands:
                if free <= 0 or sym in held:
                    continue
                px = NOPEN.get((sym, d))
                if not px:
                    continue
                mtm = sum(q['qty'] * OPEN.get((q['symbol'], d), q['entry']) for q in pos)
                alloc = min(cash, (cash + mtm) / SLOTS)
                if alloc < 1000:
                    continue
                cash -= alloc * (1 + fee)
                pos.append(dict(symbol=sym, qty=alloc / px, entry=px,
                                exit_i=min(i + hold, len(dates) - 1)))
                held.add(sym)
                free -= 1

        mtm = sum(p['qty'] * OPEN.get((p['symbol'], d), p['entry']) for p in pos)
        curve.append((d, cash + mtm))
    return pd.DataFrame(curve, columns=['d', 'equity']).set_index('d')['equity']


def metrics(eq):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dd = float((eq / eq.cummax() - 1).min())
    r = eq.pct_change().dropna()
    return dict(cagr=100 * cagr, maxdd=100 * dd,
                calmar=cagr / abs(dd) if dd else np.nan,
                sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() else np.nan,
                final=eq.iloc[-1])


bench['d'] = pd.to_datetime(bench['d'])
bs = bench.set_index('d')['close']
# market brake: NIFTYBEES above its own trailing 200-DMA, shifted so today's
# decision uses yesterday's close only
_mkt = bs.rolling(200, min_periods=200).mean()
RISK_ON = ((bs > _mkt).shift(1).fillna(False)).to_dict()
bs = bs[(bs.index >= dates[0]) & (bs.index <= dates[-1])]
bm = metrics(bs / bs.iloc[0] * CAPITAL) if len(bs) > 100 else None

print('=' * 92)
print(f'{SLOTS}-slot book | equal notional | Rs {CAPITAL:,.0f} | entry next open | '
      f'{dates[0]:%Y-%m-%d} -> {dates[-1]:%Y-%m-%d}')
print('=' * 92)
print(f"{'hold':>5} {'cost':>6} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>7} {'Sharpe':>7} "
      f"{'final':>13}    random-entry book")
print('-' * 92)

rows = []
for hold in HOLDS:
    for cb in COSTS:
        eq = run(hold, cb, False)
        m = metrics(eq)
        rm = metrics(run(hold, cb, True))
        rows.append(dict(hold=hold, cost_bps=cb, **m,
                         rand_cagr=rm['cagr'], rand_calmar=rm['calmar']))
        print(f"{hold:>4}d {cb:>5}b {m['cagr']:>7.2f}% {m['maxdd']:>7.1f}% {m['calmar']:>7.2f} "
              f"{m['sharpe']:>7.2f} {m['final']:>13,.0f}    "
              f"{rm['cagr']:>6.2f}% / Cal {rm['calmar']:.2f}", flush=True)
        eq.to_csv(OUT / f'g2b_equity_h{hold}_c{cb}.csv')

pd.DataFrame(rows).to_csv(OUT / 'g2b_market_gate.csv', index=False)
if bm:
    print('-' * 92)
    print(f"NIFTYBEES buy & hold      CAGR {bm['cagr']:>6.2f}%  MaxDD {bm['maxdd']:>6.1f}%  "
          f"Calmar {bm['calmar']:.2f}  Sharpe {bm['sharpe']:.2f}")

print('\n=== WALK-FORWARD (hold 20d, 30 bps) ===', flush=True)
eq = run(20, 30, False)
for label, lo, hi in (('2015-2020', '2015-01-01', '2020-12-31'),
                      ('2021-2026', '2021-01-01', '2026-12-31')):
    seg = eq[(eq.index >= lo) & (eq.index <= hi)]
    b = bs[(bs.index >= lo) & (bs.index <= hi)]
    if len(seg) > 100:
        m = metrics(seg)
        bb = metrics(b / b.iloc[0] * CAPITAL) if len(b) > 100 else None
        print(f"  {label}  CAGR {m['cagr']:>6.2f}%  MaxDD {m['maxdd']:>6.1f}%  "
              f"Calmar {m['calmar']:>5.2f}" +
              (f"   | NIFTYBEES CAGR {bb['cagr']:>6.2f}%  MaxDD {bb['maxdd']:>6.1f}%" if bb else ''),
              flush=True)

print('\n=== PER-YEAR (hold 20d, 30 bps) ===', flush=True)
yr = eq.resample('YE').last()
prev = CAPITAL
for ts, v in yr.items():
    by = bs[bs.index.year == ts.year]
    br = 100 * (by.iloc[-1] / by.iloc[0] - 1) if len(by) > 20 else float('nan')
    print(f'  {ts.year}  book {100 * (v / prev - 1):>+7.2f}%   NIFTYBEES {br:>+7.2f}%', flush=True)
    prev = v

print(f'\nwrote {OUT}/g2_portfolio.csv')
