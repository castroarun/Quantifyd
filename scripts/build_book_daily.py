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
        'capital': 300000, 'buying_power': 1500000, 'costs_modelled': False,
        'basis': '\u20b93L deposit \u00b7 MIS 5x \u2192 \u20b915L buying power \u00b7 5 concurrent',
        'db': 'orb_trading.db', 'table': 'orb_positions',
        'date': 'trade_date', 'pnl': 'pnl_inr', 'symbol': 'instrument',
        'where': "status='CLOSED'",
        'mode_col': 'paper_mode',          # 1 = paper, 0 = live money
        'order_col': 'kite_entry_order_id',
        'cols': ['direction', 'qty', 'entry_price', 'exit_price', 'entry_time',
                 'exit_time', 'exit_reason', 'conviction_grade'],
    },
    'n500m': {
        # risk-sized, not capital-sized: qty = 3000 / (entry - stop)
        'capital': None, 'buying_power': None, 'costs_modelled': False,
        'basis': 'risk-sized \u00b7 \u20b93,000 risk/trade \u00b7 5 concurrent',
        'db': 'n500m_trading.db', 'table': 'n500m_positions',
        'date': 'trade_date', 'pnl': 'pnl_inr', 'symbol': 'symbol',
        'where': "status='CLOSED'",
        'mode_col': 'mode', 'order_col': 'kite_entry_order_id',
        'cols': ['direction', 'signal_type', 'timeframe', 'qty', 'entry_price',
                 'exit_price', 'entry_time', 'exit_time', 'exit_reason', 'exit_policy'],
    },
    'i75wr': {
        'capital': 300000, 'buying_power': None, 'costs_modelled': False,
        'basis': '3 configs · ₹3L each · ₹3,000 risk/trade · 5 concurrent combined',
        'db': 'intraday_75wr.db', 'table': 'i75_positions',
        'date': 'trade_date', 'pnl': 'pnl_inr', 'symbol': 'instrument',
        'where': "status='CLOSED'",
        'mode_col': 'paper_mode', 'order_col': 'kite_entry_order_id',
        'cols': ['system_id', 'direction', 'qty', 'entry_price', 'exit_price',
                 'entry_time', 'exit_time', 'exit_reason'],
    },
    'momentum-3l': {
        'capital': 300000, 'buying_power': None, 'costs_modelled': True,
        'basis': '\u20b93L live \u00b7 costs and STCG modelled',
        'db': 'momentum_paper.db', 'table': 'mp_closed',
        'date': 'exit_date', 'pnl': 'net_pnl', 'symbol': 'symbol',
        'where': '1=1', 'mode_col': None, 'mode_fixed': 'live',
        'order_col': None,
        'cols': ['entry_date', 'entry_price', 'exit_price', 'qty', 'holding_days',
                 'gross_pnl', 'cost', 'stcg_tax', 'reason'],
    },
    'breakout-paper': {
        'capital': 1000000, 'buying_power': None, 'costs_modelled': True,
        'basis': '\u20b910L paper \u00b7 costs and settlement modelled',
        'db': 'breakout_paper.db', 'table': 'bp_closed',
        'date': 'exit_date', 'pnl': 'net_pnl', 'symbol': 'symbol',
        'where': '1=1', 'mode_col': None, 'mode_fixed': 'paper',
        'order_col': None,
        'cols': ['entry_date', 'entry_price', 'exit_price', 'qty', 'holding_days',
                 'gross_pnl', 'cost', 'reason'],
    },
    'pairs': {
        'capital': 1000000, 'buying_power': None, 'costs_modelled': True,
        'basis': 'market-neutral pairs \u00b7 costs modelled',
        'db': 'pair_trading.db', 'table': 'pair_trades',
        'date': 'exit_date', 'pnl': 'net_pnl_inr', 'symbol': 'pair_name',
        'where': '1=1', 'mode_col': 'paper_mode', 'order_col': None,
        'cols': ['direction', 'entry_date', 'days_held', 'entry_z', 'exit_z',
                 'gross_pnl_inr', 'cost_inr', 'exit_reason'],
    },
    'orb-index': {
        'capital': None, 'buying_power': None, 'costs_modelled': True,
        'basis': 'NIFTY strangle \u00b7 costs modelled',
        'db': 'strangle_trading.db', 'table': 'strangle_trades',
        'date': 'exit_date', 'pnl': 'net_pnl', 'symbol': 'variant_id',
        'where': '1=1', 'mode_col': None, 'mode_fixed': 'paper',
        'order_col': None,
        'cols': ['direction', 'spot_at_entry', 'spot_at_exit', 'ce_strike', 'pe_strike',
                 'gross_pnl', 'costs', 'hold_minutes', 'exit_reason'],
    },
    'kc6': {
        'capital': None, 'buying_power': None, 'costs_modelled': None,
        'basis': 'KC6 mean reversion \u00b7 paper throughout',
        'db': 'kc6_trading.db', 'table': 'kc6_trades',
        'date': 'exit_date', 'pnl': 'pnl_abs', 'symbol': 'symbol',
        'where': '1=1', 'mode_col': None, 'mode_fixed': 'paper',
        'order_col': None,
        'cols': ['entry_date', 'entry_price', 'exit_price', 'qty', 'hold_days',
                 'pnl_pct', 'exit_reason'],
    },
    'ha-paper': {
        'capital': 2000000, 'buying_power': None, 'costs_modelled': None,
        'basis': '\u20b920L paper \u00b7 Heikin-Ashi 2-green',
        'db': 'ha_paper.db', 'table': 'hap_fills',
        'date': 'ts', 'pnl': 'pnl', 'symbol': 'symbol',
        'where': 'pnl IS NOT NULL', 'mode_col': None, 'mode_fixed': 'paper',
        'order_col': None,
        'cols': ['side', 'price', 'qty', 'reason'],
    },
    'ohol-paper': {
        'capital': None, 'buying_power': None, 'costs_modelled': None,
        'basis': 'open-high / open-low \u00b7 1 lot',
        'db': 'ohol_paper.db', 'table': 'ohp_fills',
        'date': 'ts', 'pnl': 'pnl', 'symbol': 'symbol',
        'where': 'pnl IS NOT NULL', 'mode_col': None, 'mode_fixed': 'paper',
        'order_col': None,
        'cols': ['side', 'price', 'lots', 'lot_size', 'reason'],
    },
    'orb-paper': {
        'capital': 1000000, 'buying_power': None, 'costs_modelled': None,
        'basis': '\u20b910L paper \u00b7 research/89 ORB revival',
        'db': 'orb_paper.db', 'table': 'obp_fills',
        'date': 'ts', 'pnl': 'pnl', 'symbol': 'symbol',
        'where': 'pnl IS NOT NULL', 'mode_col': None, 'mode_fixed': 'paper',
        'order_col': None,
        'cols': ['side', 'price', 'qty', 'reason'],
    },
    'fnoms-paper': {
        'capital': None, 'buying_power': None, 'costs_modelled': None,
        'basis': 'F&O multi-signal \u00b7 paper',
        'db': 'fnoms_paper.db', 'table': 'fms_fills',
        'date': 'ts', 'pnl': 'pnl', 'symbol': 'symbol',
        'where': 'pnl IS NOT NULL', 'mode_col': None, 'mode_fixed': 'paper',
        'order_col': None,
        'cols': ['system', 'side', 'price', 'qty', 'reason'],
    },
    'mst': {
        'capital': None, 'buying_power': None, 'costs_modelled': False,
        'basis': 'NIFTY options \u00b7 margin-based',
        'db': 'mst_trading.db', 'table': 'mst_positions',
        'date': 'entry_time', 'pnl': 'pnl_inr', 'symbol': 'tradingsymbol',
        'where': "status='CLOSED'",
        'mode_col': 'paper_mode', 'order_col': 'order_id',
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
        wanted = [c for c in ([spec['date'], spec['pnl'], spec['symbol'],
                               spec.get('mode_col'), spec.get('order_col'),
                               'qty', 'entry_price', 'price', 'lots', 'lot_size']
                              + spec['cols']) if c and c in have]
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
        mode = (_mode_label(r.get(spec['mode_col']), spec['mode_col'])
                if spec.get('mode_col') else spec.get('mode_fixed', 'unknown'))

        # Did this order actually reach the broker? A 'PAPER-' id is simulated;
        # a real Kite id means the order was placed for real. Stronger evidence
        # than the paper_mode flag, which is only what the engine believed.
        oid = str(r.get(spec.get('order_col')) or '')
        origin = 'sim' if (not oid or oid.upper().startswith('PAPER')) else 'broker'

        # what the trade actually put to work at entry
        try:
            px = r.get('entry_price')
            if px in (None, ''):
                px = r.get('price')          # fill logs name it plainly
            units = r.get('qty')
            if units in (None, ''):
                units = float(r.get('lots') or 0) * float(r.get('lot_size') or 0) or None
            deployed = abs(float(units or 0) * float(px or 0))
        except (TypeError, ValueError):
            deployed = 0.0

        b = days.setdefault(day, {
            'date': day, 'trades': 0, 'wins': 0, 'losses': 0,
            'pnl': 0.0, 'deployed': 0.0, 'modes': set(), 'origins': set(), 'rows': [],
        })
        b['trades'] += 1
        b['pnl'] += pnl
        b['deployed'] += deployed
        b['wins' if pnl > 0 else 'losses'] += 1
        b['modes'].add(mode)
        b['origins'].add(origin)
        b['rows'].append({
            'symbol': r.get(spec['symbol']),
            'pnl': round(pnl, 2),
            'deployed': round(deployed, 2),
            'mode': mode,
            'origin': origin,
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
            'deployed': round(b['deployed'], 2),
            'cum': round(cum, 2),
            # a day is 'mixed' only if the book genuinely traded both ways that day
            'mode': modes[0] if len(modes) == 1 else 'mixed',
            'origin': (sorted(b['origins'])[0] if len(b['origins']) == 1 else 'mixed'),
            'rows': b['rows'],
        })
    out.reverse()      # newest first, the way every other page reads

    priced = [d for d in out if d['trades']]
    wins = sum(d['wins'] for d in priced)
    total = sum(d['trades'] for d in priced)
    # per-day totals are turnover (positions are opened and closed intraday),
    # so the meaningful size figure is the largest SINGLE position, not the sum
    # Is the record distinguishable from luck? t = mean / (sd / sqrt n) on
    # per-trade P&L. And does it rest on a handful of trades?
    trade_pnls = [r['pnl'] for d in out for r in d['rows'] if r.get('pnl') is not None]
    t_stat = top3 = None
    if len(trade_pnls) >= 5:
        m = sum(trade_pnls) / len(trade_pnls)
        var = sum((x - m) ** 2 for x in trade_pnls) / (len(trade_pnls) - 1)
        if var > 0:
            t_stat = round(m / ((var ** 0.5) / (len(trade_pnls) ** 0.5)), 2)
        net_all = sum(trade_pnls)
        if net_all > 0:
            top3 = round(100.0 * sum(sorted(trade_pnls, reverse=True)[:3]) / net_all, 1)

    day_totals = [d['deployed'] for d in out if d['deployed']]
    positions = [r['deployed'] for d in out for r in d['rows'] if r.get('deployed')]
    return {
        'days': out,
        'summary': {
            'capital': spec.get('capital'),
            'buying_power': spec.get('buying_power'),
            'basis': spec.get('basis'),
            'max_position': round(max(positions), 2) if positions else None,
            'busiest_day': round(max(day_totals), 2) if day_totals else None,
            'avg_trade': round(sum(positions) / len(positions), 2) if positions else None,
            'days': len(out),
            'trades': total,
            'wins': wins,
            'win_rate': round(100.0 * wins / total, 1) if total else None,
            'net': round(sum(d['pnl'] for d in out), 2),
            'best_day': max((d['pnl'] for d in out), default=None),
            'worst_day': min((d['pnl'] for d in out), default=None),
            'green_days': sum(1 for d in out if d['pnl'] > 0),
            'costs_modelled': spec.get('costs_modelled'),
            't_stat': t_stat,
            'top3_share': top3,
            'broker_trades': sum(1 for d in out for r in d['rows'] if r.get('origin') == 'broker'),
            'sim_trades': sum(1 for d in out for r in d['rows'] if r.get('origin') == 'sim'),
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
