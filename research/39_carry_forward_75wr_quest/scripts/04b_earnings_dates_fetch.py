"""Fetch real earnings announcement dates from yfinance for the F&O universe.

Source: yfinance .NS tickers — get_earnings_dates() returns up to 25 rows
        of historical + upcoming earnings with EPS estimate, actual, surprise%.

Output: results/04b_earnings_dates.csv
        Columns: symbol, announcement_date, eps_estimate, eps_actual, surprise_pct

Notes:
- yfinance times are in US/Eastern. We convert to IST date (announcement
  date in India for next-day-open trading).
- yfinance only goes back ~6 years; we'll fetch what's available and use
  it from 2018-01 (the earliest the F&O universe has daily data anyway).
- Errors are logged but do not abort. Empty results are fine — we just
  skip that symbol.
- Caches per-symbol in case of restart.
"""

from __future__ import annotations

import csv
import os
import sys
import time
import traceback

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _engine_daily_my as E  # type: ignore

try:
    import yfinance as yf
except ImportError:
    print('yfinance not installed. pip install yfinance', flush=True)
    sys.exit(1)


RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results',
))
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'logs',
))
os.makedirs(LOG_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(RESULTS_DIR, '04b_earnings_dates.csv')
LOG_PATH = os.path.join(LOG_DIR, '04b_earnings_fetch.log')

# yfinance suffix for NSE stocks
YF_SUFFIX = '.NS'

# Some symbols differ between Kite and yfinance — apply substitutions
YF_OVERRIDES = {
    'M&M': 'M%26M.NS',  # ampersand encoding sometimes works; fallback below
    # 'BAJAJ-AUTO' is fine
}

# Symbols where the standard .NS lookup fails (try BSE .BO as fallback)
# We'll discover at runtime.

FIELDS = ['symbol', 'announcement_date', 'eps_estimate', 'eps_actual', 'surprise_pct']


def _yf_symbol(sym: str) -> str:
    """Map our universe symbol to yfinance ticker."""
    # M&M: yfinance handles M%26M.NS sometimes, but M&M.NS also works
    if sym == 'M&M':
        return 'M%26M.NS'
    return f'{sym}{YF_SUFFIX}'


def _log(msg: str) -> None:
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}\n')
    print(msg, flush=True)


def fetch_one(symbol: str, retries: int = 2) -> pd.DataFrame:
    """Fetch earnings dates for a single symbol, returning a normalized DataFrame."""
    yf_sym = _yf_symbol(symbol)
    last_err = None
    for attempt in range(retries + 1):
        try:
            t = yf.Ticker(yf_sym)
            ed = t.get_earnings_dates(limit=40)  # yfinance caps around 24 anyway
            if ed is None or ed.empty:
                # Try .BO fallback
                if attempt == 0 and not yf_sym.endswith('.BO'):
                    yf_sym = symbol + '.BO'
                    continue
                return pd.DataFrame(columns=FIELDS)
            ed = ed.copy()
            # index is "Earnings Date" tz-aware (US/Eastern)
            ed = ed.reset_index()
            # column names per yfinance: 'Earnings Date', 'EPS Estimate', 'Reported EPS', 'Surprise(%)'
            date_col = 'Earnings Date'
            est_col = 'EPS Estimate'
            act_col = 'Reported EPS'
            sur_col = 'Surprise(%)'
            for c in (date_col, est_col, act_col, sur_col):
                if c not in ed.columns:
                    ed[c] = pd.NaT if c == date_col else float('nan')
            # Convert ET -> IST (we just want the date in IST). Add 9.5h to ET -> UTC then to IST
            # Simpler: take .dt.tz_convert('Asia/Kolkata') if tz-aware
            try:
                if pd.api.types.is_datetime64_any_dtype(ed[date_col]):
                    if getattr(ed[date_col].dt, 'tz', None) is not None:
                        ed[date_col] = ed[date_col].dt.tz_convert('Asia/Kolkata')
                    # else: naive, leave as-is
            except Exception:
                pass
            ed['announcement_date'] = pd.to_datetime(ed[date_col]).dt.tz_localize(None).dt.normalize()
            out = pd.DataFrame({
                'symbol': symbol,
                'announcement_date': ed['announcement_date'],
                'eps_estimate': pd.to_numeric(ed[est_col], errors='coerce'),
                'eps_actual': pd.to_numeric(ed[act_col], errors='coerce'),
                'surprise_pct': pd.to_numeric(ed[sur_col], errors='coerce'),
            })
            # Drop rows where actual is missing (future earnings, no use) for OOS calibration?
            # Actually keep all — actual=NaN means upcoming or unreported; we can still
            # use the announcement_date for past events. Filter at use-time.
            return out
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5)
                continue
            _log(f'  [{symbol}] FAILED after {retries+1} attempts: {e}')
            return pd.DataFrame(columns=FIELDS)
    return pd.DataFrame(columns=FIELDS)


def main():
    t0 = time.time()
    _log(f'[main] starting earnings fetch for F&O universe -> {OUTPUT_CSV}')

    syms = E.list_daily_universe(E.FNO_UNIVERSE, start='2018-01-01',
                                  end='2026-03-19', min_rows=1500)
    _log(f'[main] {len(syms)} F&O symbols with sufficient daily data')

    # Resume support: skip symbols already in CSV
    done = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            existing = pd.read_csv(OUTPUT_CSV)
            done = set(existing['symbol'].unique().tolist())
            _log(f'[main] resume: {len(done)} symbols already in CSV')
        except Exception:
            done = set()

    write_header = not os.path.exists(OUTPUT_CSV)
    f = open(OUTPUT_CSV, 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if write_header:
        w.writeheader()

    total_rows = 0
    for i, sym in enumerate(syms, 1):
        if sym in done:
            _log(f'[{i}/{len(syms)}] {sym} SKIP (already fetched)')
            continue
        try:
            df = fetch_one(sym)
        except Exception as e:
            _log(f'[{i}/{len(syms)}] {sym} EXCEPTION: {e}')
            continue
        n_rows = len(df)
        n_actual = df['eps_actual'].notna().sum() if n_rows else 0
        for _, row in df.iterrows():
            d = row['announcement_date']
            if pd.isna(d):
                continue
            w.writerow({
                'symbol': sym,
                'announcement_date': pd.Timestamp(d).strftime('%Y-%m-%d'),
                'eps_estimate': row['eps_estimate'] if pd.notna(row['eps_estimate']) else '',
                'eps_actual': row['eps_actual'] if pd.notna(row['eps_actual']) else '',
                'surprise_pct': row['surprise_pct'] if pd.notna(row['surprise_pct']) else '',
            })
            total_rows += 1
        f.flush()
        _log(f'[{i}/{len(syms)}] {sym}: {n_rows} dates ({n_actual} with actuals) | total={total_rows}')
        # Throttle ~0.5s between yfinance calls
        time.sleep(0.6)

    f.close()
    _log(f'[main] DONE in {time.time()-t0:.0f}s | total_rows={total_rows}')


if __name__ == '__main__':
    main()
