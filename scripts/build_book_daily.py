"""Day-by-day trade history for the per-book pages.

The ORB Cash, N500M and MST pages each showed only today: today's positions,
today's closed trades, today's signals. The book record strip said "94 trades,
last 8d ago" but there was nowhere to see those trades, so a book that had been
running for months looked like a book that had done nothing.

This projects each book's own positions table into days: the day's realised
P&L, how many trades, how many won, a running cumulative, and the trades
themselves so a day can be opened up.

Written to static/app/book_daily.json, which Flask serves as a static file —
so the pages pick it up without restarting the service, which matters because
the trading day is the one time you actually want to look at this.

READ-ONLY. Every database opens mode=ro and no engine is imported. It cannot
affect live or paper trading (standing rule, .claude/CLAUDE.md).

Run: venv/bin/python3 scripts/build_book_daily.py
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BD = ROOT / 'backtest_data'
OUT = ROOT / 'static' / 'app' / 'book_daily.json'

# book -> how to read its trades.
#   db, table, date col, pnl col, symbol col, WHERE for 'closed',
#   and the columns worth showing in the expanded day
BOOKS = {
    'orb': {
        'db': 'orb_trading.db', 'table': 'orb_positions',
        'date': 'trade_date', 'pnl': 'pnl_inr', 'symbol': 'instrument',
        'where': "status='CLOSED'",
        'mode_col': 'paper_mode',          # 1 = paper, 0 = live money
        'cols': ['direction', 'qty', 'entry_price', 'exit_price', 'entry_time',
                 'exit_time', 'exit_reason', 'conviction_grade'],
    },
    'n500m': {
        'db': 'n500m_trading.db', 'table': 'n500m_positions',
        'date': 'trade_date', 'pnl': 'pnl_inr', 'symbol': 'symbol',
        'where': "status='CLOSED'",
        'mode_col': 'mode',                # text: 'PAPER' / 'LIVE'
        'cols': ['direction', 'signal_type', 'timeframe', 'qty', 'entry_price',
                 'exit_price', 'entry_time', 'exit_time', 'exit_reason', 'exit_policy'],
    },
    'mst': {
        'db': 'mst_trading.db', 'table': 'mst_positions',
        'date': 'entry_time', 'pnl': 'pnl_inr', 'symbol': 'tradingsymbol',
        'where': "status='CLOSED'",
        'mode_col': 'paper_mode',
        'cols': ['side', 'leg_role', 'strike', 'option_type', 'qty', 'entry_price',
                 'exit_price', 'entry_time', 'exit_time', 'exit_reason', 'week_label'],
    },
}


def _mode_label(raw, col: str) -> str:
    """Normalise the two conventions (int flag, text) into one label."""
    if raw is None:
        return 'unknown'
    if col == 'mode':
        return str(raw).lower() or 'unknown'
    return 'paper' if int(raw or 0) else 'live'


def build_book(key: str, spec: dict) -> dict:
    path = BD / spec['db']
    if not path.exists():
        return {'days': [], 'summary': None, 'error': f"{spec['db']} not found"}

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({spec['table']})")}
        wanted = [c for c in ([spec['date'], spec['pnl'], spec['symbol'], spec['mode_col']]
                              + spec['cols']) if c in have]
        sel = ', '.join(f'"{c}"' for c in dict.fromkeys(wanted))
        rows = [dict(r) for r in conn.execute(
            f"SELECT {sel} FROM {spec['table']} WHERE {spec['where']} "
            f"ORDER BY \"{spec['date']}\""
        )]
    finally:
        conn.close()

    days: dict[str, dict] = {}
    for r in rows:
        day = str(r.get(spec['date']) or '')[:10]
        if not day:
            continue
        pnl = r.get(spec['pnl'])
        pnl = float(pnl) if pnl is not None else 0.0
        mode = _mode_label(r.get(spec['mode_col']), spec['mode_col'])

        b = days.setdefault(day, {
            'date': day, 'trades': 0, 'wins': 0, 'losses': 0,
            'pnl': 0.0, 'modes': set(), 'rows': [],
        })
        b['trades'] += 1
        b['pnl'] += pnl
        b['wins' if pnl > 0 else 'losses'] += 1
        b['modes'].add(mode)
        b['rows'].append({
            'symbol': r.get(spec['symbol']),
            'pnl': round(pnl, 2),
            'mode': mode,
            **{c: r.get(c) for c in spec['cols'] if c in r},
        })

    out, cum = [], 0.0
    for day in sorted(days):
        b = days[day]
        cum += b['pnl']
        modes = sorted(b['modes'])
        out.append({
            'date': day,
            'trades': b['trades'],
            'wins': b['wins'],
            'losses': b['losses'],
            'pnl': round(b['pnl'], 2),
            'cum': round(cum, 2),
            # a day is 'mixed' only if the book genuinely traded both ways that day
            'mode': modes[0] if len(modes) == 1 else 'mixed',
            'rows': b['rows'],
        })
    out.reverse()      # newest first, the way every other page reads

    priced = [d for d in out if d['trades']]
    wins = sum(d['wins'] for d in priced)
    total = sum(d['trades'] for d in priced)
    return {
        'days': out,
        'summary': {
            'days': len(out),
            'trades': total,
            'wins': wins,
            'win_rate': round(100.0 * wins / total, 1) if total else None,
            'net': round(sum(d['pnl'] for d in out), 2),
            'best_day': max((d['pnl'] for d in out), default=None),
            'worst_day': min((d['pnl'] for d in out), default=None),
            'green_days': sum(1 for d in out if d['pnl'] > 0),
            'first': out[-1]['date'] if out else None,
            'last': out[0]['date'] if out else None,
        },
    }


def main() -> None:
    feed = {'generated_at': datetime.now().isoformat(timespec='seconds'), 'books': {}}
    for key, spec in BOOKS.items():
        try:
            feed['books'][key] = build_book(key, spec)
        except Exception as e:                       # one bad book must not kill the feed
            feed['books'][key] = {'days': [], 'summary': None, 'error': str(e)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(feed, indent=1, default=str), encoding='utf-8')
    tmp.replace(OUT)                                  # atomic: never serve a half-written feed

    for key, b in feed['books'].items():
        s = b.get('summary')
        if s:
            print(f"{key:8} {s['days']:>3} days  {s['trades']:>4} trades  "
                  f"net {s['net']:>12,.0f}  {s['first']} -> {s['last']}")
        else:
            print(f"{key:8} -- {b.get('error')}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == '__main__':
    main()
