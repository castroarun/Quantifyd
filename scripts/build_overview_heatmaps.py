#!/usr/bin/env python3
"""Build the two heatmaps for /app/overview.

  books  × days — every book's daily net P&L for the last quarter
  stocks × days — the Nifty-500 universe's daily returns, with sector

Written as a static JSON (static/app/overview_heatmaps.json) so the page picks it
up without a backend restart, the same way the other regen jobs publish. READ
ONLY: opens every DB mode=ro, imports no engine, writes nothing but its own file.

Run: venv/bin/python3 scripts/build_overview_heatmaps.py [--days 65]
"""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BD = ROOT / 'backtest_data'
OUT = ROOT / 'static' / 'app' / 'overview_heatmaps.json'
DAYS = 65

for i, a in enumerate(sys.argv):
    if a == '--days' and i + 1 < len(sys.argv):
        DAYS = max(20, min(250, int(sys.argv[i + 1])))

# ─────────────────────────── books × days ───────────────────────────
# (label, db, table, date column, pnl column, optional filter)
BOOKS = [
    ('NAS · NIFTY',      'nas_atm_trading.db',        'nas_atm_trades', 'exit_time', 'net_pnl', None),
    ('NAS · ATM2',       'nas_atm2_trading.db',       'nas_atm_trades', 'exit_time', 'net_pnl', None),
    ('NAS · ATM4',       'nas_atm4_trading.db',       'nas_atm_trades', 'exit_time', 'net_pnl', None),
    ('NAS · 916 ATM',    'nas_916_atm_trading.db',    'nas_atm_trades', 'exit_time', 'net_pnl', None),
    ('NAS · 916 ATM2',   'nas_916_atm2_trading.db',   'nas_atm_trades', 'exit_time', 'net_pnl', None),
    ('NAS · 916 ATM4',   'nas_916_atm4_trading.db',   'nas_atm_trades', 'exit_time', 'net_pnl', None),
    ('SENSEX · ATM',     'sensex_atm_trading.db',     'nas_atm_trades', 'exit_time', 'net_pnl', None),
    ('SENSEX · ATM2',    'sensex_atm2_trading.db',    'nas_atm_trades', 'exit_time', 'net_pnl', None),
    ('SENSEX · ATM4',    'sensex_atm4_trading.db',    'nas_atm_trades', 'exit_time', 'net_pnl', None),
    ('ORB Cash',         'orb_trading.db',            'orb_positions',  'exit_time', 'pnl_inr', "status='CLOSED'"),
    ('N500M',            'n500m_trading.db',          'n500m_positions', 'exit_time', 'pnl_inr', "status!='OPEN'"),
    ('I75WR',            'intraday_75wr.db',          'i75_positions',  'exit_time', 'pnl_inr', "status='CLOSED'"),
    ('KC6',              'kc6_trading.db',            'kc6_trades',     'exit_date', 'pnl_abs', None),
    ('Momentum ₹3L',     'momentum_paper.db',         'mp_closed',      'exit_date', 'net_pnl', None),
    ('Breakout ₹10L',    'breakout_paper.db',         'bp_closed',      'exit_date', 'net_pnl', None),
    ('HA 2-Green',       'ha_paper.db',               'hap_fills',      'ts',        'pnl',     'pnl IS NOT NULL'),
    ('OHOL',             'ohol_paper.db',             'ohp_fills',      'ts',        'pnl',     'pnl IS NOT NULL'),
    ('ORB Revival',      'orb_paper.db',              'obp_fills',      'ts',        'pnl',     'pnl IS NOT NULL'),
]


def book_rows():
    rows, all_days = [], set()
    for label, db, table, dcol, pcol, where in BOOKS:
        path = BD / db
        if not path.exists():
            continue
        try:
            con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
            cols = [c[1] for c in con.execute(f'PRAGMA table_info({table})')]
            if dcol not in cols or pcol not in cols:
                con.close()
                continue
            q = (f'SELECT substr({dcol},1,10) d, SUM({pcol}) p FROM {table} '
                 f'WHERE {dcol} IS NOT NULL' + (f' AND {where}' if where else '') +
                 ' GROUP BY 1 ORDER BY 1')
            day_pnl = {d: float(p or 0) for d, p in con.execute(q) if d}
            con.close()
        except Exception as e:
            print(f'  [warn] {db}: {e}')
            continue
        if not day_pnl:
            continue
        rows.append({'label': label, 'days': day_pnl})
        all_days |= set(day_pnl)
    dates = sorted(all_days)[-DAYS:]
    out = []
    for r in rows:
        vals = [round(r['days'].get(d, 0.0)) for d in dates]
        if not any(vals):
            continue
        out.append({'label': r['label'], 'v': vals,
                    'total': round(sum(vals)), 'days': sum(1 for v in vals if v)})
    out.sort(key=lambda r: -r['total'])
    return dates, out


# ─────────────────────────── stocks × days ──────────────────────────
def stock_rows():
    meta = {}
    lst = ROOT / 'data' / 'nifty500_list.csv'
    if lst.exists():
        with lst.open(encoding='utf-8-sig') as fh:
            for row in csv.DictReader(fh):
                meta[row['Symbol'].strip()] = (row['Sector'] or 'Other').strip()
    db = BD / 'market_data.db'
    if not db.exists():
        return [], []
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    last = con.execute("SELECT MAX(date) FROM market_data_unified WHERE timeframe='day'").fetchone()[0]
    rows = con.execute(
        "SELECT symbol, date, close FROM market_data_unified "
        "WHERE timeframe='day' AND date >= date(?, ?) ORDER BY symbol, date",
        (last, f'-{DAYS + 8} day')).fetchall()
    con.close()

    series = defaultdict(list)
    for s, d, c in rows:
        if c:
            series[s].append((d[:10], float(c)))

    all_days = sorted({d for v in series.values() for d, _ in v})[-DAYS:]
    dayset = set(all_days)
    out = []
    for sym, pts in series.items():
        if sym not in meta or len(pts) < 5:
            continue
        by = dict(pts)
        prev, vals = None, []
        for d in all_days:
            c = by.get(d)
            if c is None or prev is None:
                vals.append(None)
            else:
                vals.append(round((c / prev - 1) * 100, 2))
            if c is not None:
                prev = c
        if sum(1 for v in vals if v is not None) < len(all_days) * 0.5:
            continue          # too gappy to colour honestly
        live = [v for v in vals if v is not None]
        out.append({'s': sym, 'sec': meta[sym], 'v': vals,
                    'sum': round(sum(live), 1)})
    out.sort(key=lambda r: (r['sec'], -r['sum']))
    return all_days, out


def main():
    bdates, books = book_rows()
    sdates, stocks = stock_rows()
    payload = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'books': {'dates': bdates, 'rows': books},
        'stocks': {'dates': sdates, 'rows': stocks},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(',', ':')))
    print(f'books  : {len(books)} rows × {len(bdates)} days')
    print(f'stocks : {len(stocks)} rows × {len(sdates)} days')
    print(f'wrote {OUT} ({OUT.stat().st_size // 1024} KB)')
    for r in books[:5]:
        print(f"   {r['label']:16s} {r['total']:>10,}  ({r['days']} active days)")


if __name__ == '__main__':
    main()
