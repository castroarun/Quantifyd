# -*- coding: utf-8 -*-
"""Load the research/160 Screener fundamentals into the shared database.

    backtest_data/fundamentals.db          <- the central store, next to market_data.db

Why a separate file and not a table in market_data.db: that file is 31 GB and is opened by
every live trading service every day; a research-grade fundamentals store should not ride
inside it. Everything here is derived from `research/160_quality_growth_near_ath/results/`
(Screener page cache + the point-in-time panel) and is fully rebuildable from the scripts in
that folder. Idempotent: every table uses INSERT OR REPLACE on its natural key, so a re-run
after a re-fetch simply updates rows.

Tables
------
screener_annual      one row per (ticker, fy_end, source): the raw annual P&L / balance-sheet
                     lines as Screener shows them TODAY (restated, not as-reported).
screener_quarterly   one row per (ticker, quarter_end, source): sales, operating profit,
                     OPM %, net profit. Screener carries ~13 quarters, so history starts ~2023.
screener_top_ratios  today's headline ratios (market cap, price, face value, ROE, ROCE ...)
                     keyed by fetch date. A denominator/sanity source, NEVER point-in-time.
screener_meta        per ticker: which page answered, HTTP codes, depth, fetch time, note.
symbol_map           market_data.db symbol <-> Screener ticker (series suffix stripped).
features_pit_monthly the POINT-IN-TIME panel: one row per (decision date = 1st of month,
                     symbol) holding only figures FILED before that date (FY usable from
                     year-end + lag_months; quarter usable from quarter-end + 60 days).
                     This is the table a backtest should read. Column meanings are in
                     research/160_quality_growth_near_ath/results/features_pit_monthly_README.md
                     and mirrored in the `schema_notes` table.
schema_notes         free-text documentation rows (table, column, note) so the DB explains itself.

Point-in-time rule, restated because it is the whole point: a backtest must join
features_pit_monthly on the decision date and use ONLY that row. Reading screener_annual by
fy_end directly in a backtest re-introduces look-ahead unless you apply the filing lag yourself.
"""
import glob
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone, timedelta

import pandas as pd

ROOT = '/home/arun/quantifyd'
RES = os.path.join(ROOT, 'research/160_quality_growth_near_ath/results')
DB = os.path.join(ROOT, 'backtest_data/fundamentals.db')
LAG_MONTHS = 4          # the panel on disk was built with the default 4-month lag
IST = timezone(timedelta(hours=5, minutes=30))

ANNUAL_FIELDS = ['sales', 'expenses', 'operating_profit', 'opm_pct', 'other_income', 'interest',
                 'depreciation', 'profit_before_tax', 'net_profit', 'eps', 'dividend_payout_pct',
                 'equity_capital', 'reserves', 'borrowings', 'other_liabilities', 'total_assets',
                 'roce_pct']
QUARTER_FIELDS = ['sales', 'operating_profit', 'opm_pct', 'net_profit']

DDL = f"""
CREATE TABLE IF NOT EXISTS screener_annual (
    ticker TEXT NOT NULL, fy_end TEXT NOT NULL, source TEXT NOT NULL,
    {', '.join(f'{f} REAL' for f in ANNUAL_FIELDS)},
    fetched_at TEXT,
    PRIMARY KEY (ticker, fy_end, source));
CREATE INDEX IF NOT EXISTS ix_annual_fy ON screener_annual (fy_end);

CREATE TABLE IF NOT EXISTS screener_quarterly (
    ticker TEXT NOT NULL, quarter_end TEXT NOT NULL, source TEXT NOT NULL,
    {', '.join(f'{f} REAL' for f in QUARTER_FIELDS)},
    fetched_at TEXT,
    PRIMARY KEY (ticker, quarter_end, source));
CREATE INDEX IF NOT EXISTS ix_quarterly_q ON screener_quarterly (quarter_end);

CREATE TABLE IF NOT EXISTS screener_top_ratios (
    ticker TEXT NOT NULL, fetch_date TEXT NOT NULL, key TEXT NOT NULL, value REAL, raw TEXT,
    PRIMARY KEY (ticker, fetch_date, key));

CREATE TABLE IF NOT EXISTS screener_meta (
    ticker TEXT PRIMARY KEY, source TEXT, http_json TEXT, fetched_at TEXT, note TEXT,
    n_fy INTEGER, n_q INTEGER);

CREATE TABLE IF NOT EXISTS symbol_map (
    symbol TEXT PRIMARY KEY, screener_ticker TEXT, first_date TEXT, last_date TEXT,
    n_rows INTEGER, active INTEGER);
CREATE INDEX IF NOT EXISTS ix_symbol_map_ticker ON symbol_map (screener_ticker);

CREATE TABLE IF NOT EXISTS schema_notes (
    tbl TEXT NOT NULL, col TEXT NOT NULL, note TEXT, PRIMARY KEY (tbl, col));

CREATE TABLE IF NOT EXISTS load_log (
    loaded_at TEXT, source_dir TEXT, n_annual INTEGER, n_quarterly INTEGER, n_top INTEGER,
    n_meta INTEGER, n_map INTEGER, n_panel INTEGER, lag_months INTEGER);
"""

NOTES = [
    ('screener_annual', '*', 'Raw annual rows exactly as screener.in shows them at fetch time: RESTATED figures, not as-reported. Rs crore except opm_pct/roce_pct/dividend_payout_pct (%) and eps (Rs). fy_end = 31-Mar of the fiscal year. source = consolidated|standalone (both kept when both were fetched).'),
    ('screener_annual', 'roce_pct', "Screener's own ROCE row. Absent (NULL) for banks/NBFCs - capital employed is not meaningful for lenders. Never computed here."),
    ('screener_annual', 'borrowings', 'Absent (NULL) for lenders: Screener lumps their debt into other_liabilities, so any D/E test excludes the financial sector by construction.'),
    ('screener_quarterly', '*', 'Quarterly results table: ~13 quarters per page, so history starts around mid-2023. Rs crore; opm_pct in %.'),
    ('screener_top_ratios', '*', "TODAY's headline block at fetch time (market_cap, current_price, face_value, roe, roce, ...). A denominator / sanity source only. NEVER usable point-in-time."),
    ('features_pit_monthly', '*', 'POINT-IN-TIME panel: decision date = 1st of month; a fiscal year is usable from year-end + lag_months (4), a quarter from quarter-end + 60 days. Backtests join on `date` and use only that row; the consumer forward-fills until the next date.'),
    ('features_pit_monthly', 'n_fy_usable', 'Filed fiscal years at that date. >= 4 is the has_data condition (a genuine 3-year growth rate needs 4 annual points). Coverage jumps 7% -> 87% in Aug-2018 when FY2018 is filed - the honest start of any fundamentals window.'),
    ('features_pit_monthly', 'sales_g3', '3-year CAGR % from the 4th-last to the latest usable FY; NULL when the base is <= 0.'),
    ('features_pit_monthly', 'profit_g3', 'As sales_g3, on net_profit.'),
    ('features_pit_monthly', 'roe_avg3', 'Mean of the last 3 usable years of net_profit / (equity_capital + reserves) x100.'),
    ('features_pit_monthly', 'roce_latest', "Screener's ROCE row for the latest usable FY; NULL for lenders (is_lender = 1)."),
    ('features_pit_monthly', 'de_latest', 'borrowings / (equity_capital + reserves) for the latest usable FY; NULL for lenders.'),
    ('features_pit_monthly', 'opm_slope3', 'OLS slope of annual OPM % over the last 3 usable FY, in pp per year. opm_range3 = max-min pp, opm_min3 = min.'),
    ('features_pit_monthly', 'opm_q_slope8', 'OLS slope of quarterly OPM % over the last 8 usable quarters, pp per QUARTER; opm_q_std8 = sample sigma. Only populated from ~2024.'),
    ('features_pit_monthly', 'neg3', '1 if any negative Sales or Net Profit in the last 3 usable FY.'),
    ('features_pit_monthly', 'shares_pit', "equity_capital / face_value = share count in CRORES (face value is today's; a denominator, not a feature)."),
    ('features_pit_monthly', 'mcap_pit', 'shares_pit x last close at/before the decision date, Rs crore. Reconciled to Screener market cap: median |error| 0.7%, 95.6% within 10%.'),
    ('features_pit_monthly', 'mcap_scale_suspect', '1 when a >45% single-day close collapse (unadjusted split/bonus in market_data.db) lies in the symbol FUTURE relative to this row - the mcap level is fine, its history is on the old price scale.'),
    ('features_pit_monthly', 'is_lender', 'No ROCE on any usable year while ROE is computable -> bank/NBFC.'),
    ('symbol_map', '*', 'market_data.db symbol (series suffix intact, e.g. MODISONLTD-BE) -> Screener ticker (suffix stripped). Built from research/160 universe.csv: all day-timeframe symbols minus funds (etf_exclusions.json) minus stubs (<250 rows or nothing after 2015).'),
    ('*', '*', 'Source and rebuild: research/160_quality_growth_near_ath/scripts/{screener_fetch_full,build_pit_panel,load_fundamentals_db}.py. Screener KEEPS delisted companies (101 of 102 stopped series had pages), so the fundamentals leg carries little survivorship of its own; the price DB (102 stopped series in 2,158) is the real gap. Study: /app/backtest/quality-growth-near-ath-research160.'),
]


def num(x):
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def main():
    con = sqlite3.connect(DB)
    con.executescript(DDL)
    con.executemany('INSERT OR REPLACE INTO schema_notes VALUES (?,?,?)', NOTES)

    files = sorted(glob.glob(os.path.join(RES, 'screener_cache', '*.json')))
    n_a = n_q = n_t = n_m = 0
    for f in files:
        d = json.load(open(f))
        t = d.get('ticker') or os.path.basename(f)[:-5]
        src = d.get('source') or 'none'
        fa = d.get('fetched_at')
        for fy, row in (d.get('annual') or {}).items():
            con.execute(
                f"INSERT OR REPLACE INTO screener_annual (ticker, fy_end, source, {', '.join(ANNUAL_FIELDS)}, fetched_at) "
                f"VALUES (?,?,?,{','.join('?' * len(ANNUAL_FIELDS))},?)",
                [t, fy, src] + [num(row.get(k)) for k in ANNUAL_FIELDS] + [fa])
            n_a += 1
        for q, row in (d.get('quarterly') or {}).items():
            con.execute(
                f"INSERT OR REPLACE INTO screener_quarterly (ticker, quarter_end, source, {', '.join(QUARTER_FIELDS)}, fetched_at) "
                f"VALUES (?,?,?,{','.join('?' * len(QUARTER_FIELDS))},?)",
                [t, q, src] + [num(row.get(k)) for k in QUARTER_FIELDS] + [fa])
            n_q += 1
        fdate = (fa or '')[:10]
        for k, v in (d.get('top') or {}).items():
            con.execute('INSERT OR REPLACE INTO screener_top_ratios VALUES (?,?,?,?,?)',
                        (t, fdate, k, num(v), None if num(v) is not None else str(v)))
            n_t += 1
        con.execute('INSERT OR REPLACE INTO screener_meta VALUES (?,?,?,?,?,?,?)',
                    (t, src, json.dumps(d.get('http')), fa, d.get('note'), d.get('n_fy'), d.get('n_q')))
        n_m += 1
    con.commit()

    u = pd.read_csv(os.path.join(RES, 'universe.csv'))
    cols = [c for c in ['symbol', 'screener_ticker', 'first_date', 'last_date', 'n_rows', 'active'] if c in u.columns]
    u = u[cols].copy()
    if 'active' in u:
        u['active'] = u['active'].astype(int)
    u.to_sql('symbol_map', con, if_exists='replace', index=False)
    con.execute('CREATE INDEX IF NOT EXISTS ix_symbol_map_ticker ON symbol_map (screener_ticker)')
    n_map = len(u)

    p = pd.read_csv(os.path.join(RES, 'features_pit_monthly.csv.gz'))
    for c in ('is_lender', 'neg3', 'mcap_scale_suspect'):
        if c in p:
            p[c] = p[c].astype(bool).astype(int)
    p['lag_months'] = LAG_MONTHS
    p.to_sql('features_pit_monthly', con, if_exists='replace', index=False)
    con.execute('CREATE UNIQUE INDEX IF NOT EXISTS ix_pit_date_sym ON features_pit_monthly (date, symbol)')
    con.execute('CREATE INDEX IF NOT EXISTS ix_pit_sym ON features_pit_monthly (symbol)')
    n_p = len(p)

    con.execute('INSERT INTO load_log VALUES (?,?,?,?,?,?,?,?,?)',
                (datetime.now(IST).isoformat(timespec='seconds'), RES, n_a, n_q, n_t, n_m, n_map, n_p, LAG_MONTHS))
    con.commit()
    con.execute('VACUUM')
    con.close()
    print(f'loaded -> {DB}\n  screener_annual {n_a}\n  screener_quarterly {n_q}\n  screener_top_ratios {n_t}\n'
          f'  screener_meta {n_m}\n  symbol_map {n_map}\n  features_pit_monthly {n_p}\n  size {os.path.getsize(DB)/1e6:.1f} MB')


if __name__ == '__main__':
    sys.exit(main())
