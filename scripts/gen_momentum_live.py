"""Bake True North's live marks to a static file, once a minute in market hours.

WHY. /api/momentum-paper/state is the only thing the True North page can render from,
and get_state() does a live Kite quote() AND a ~1000x1600 pandas pivot ON THE REQUEST
PATH. Measured during market hours on 2026-09-07: 0.58s, then 3.57s on the very next
call. Open Alpha renders in 2ms because a cron bakes its numbers and the page just
fetches a file. This gives True North the same, without touching the engine.

WHAT IT IS NOT. This is a DISPLAY mirror, not a second source of truth. It never writes
book state, never places an order, and the page still fetches the real API for
everything else — this file only lets the prices paint immediately and keep ticking.
If it is stale or missing the page falls back to the API exactly as before.

Cron: * 9-15 * * 1-5 (every minute, market hours) + one post-close run.
Out:  static/app/momentum_live.json    Log: /tmp/momentum_live.log
"""
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'static' / 'app' / 'momentum_live.json'



def donchian_stop(sym, n=15):
    """The 15-day-low stop: min of the n closes BEFORE the latest bar.

    Mirrors momentum_paper._donchian_low, which slices [-n-1:-1] off the close series -
    i.e. it excludes the most recent bar. Read straight from SQLite so the per-minute
    bake never touches the pandas pivot that makes the state endpoint slow.
    """
    import sqlite3
    try:
        con = sqlite3.connect(
            'file:%s?mode=ro' % (ROOT / 'backtest_data' / 'market_data.db'), uri=True)
        try:
            rows = [r[0] for r in con.execute(
                "SELECT close FROM market_data_unified WHERE symbol=? AND timeframe='day' "
                "AND close > 0 ORDER BY date DESC LIMIT ?", (sym, n + 1))]
        finally:
            con.close()
    except Exception:
        return None
    # 1dp, matching momentum_paper's own rounding - two pages must not show two stops
    return round(min(rows[1:]), 1) if len(rows) > n else None



def _realized():
    """Net realised P&L on True North's closed trades. Same figure the state API reports."""
    import sqlite3
    try:
        con = sqlite3.connect(
            'file:%s?mode=ro' % (ROOT / 'backtest_data' / 'momentum_paper.db'), uri=True)
        try:
            v = con.execute('select sum(net_pnl) from mp_closed').fetchone()[0]
        finally:
            con.close()
        return round(v or 0)
    except Exception:
        return 0


def main():
    from services import momentum_paper as mp

    pos = mp._positions()
    syms = sorted(pos)
    if not syms:
        print('no positions')
        return

    kite = mp._kite()
    q = {}
    for i in range(0, len(syms), 200):
        try:
            q.update(kite.quote(['NSE:' + s for s in syms[i:i + 200]]))
        except Exception as e:
            print('quote batch failed:', e)

    rows = []
    tot_val = tot_pnl = 0.0
    for s in syms:
        p = pos[s]
        d = q.get('NSE:' + s, {})
        ltp = d.get('last_price')
        prev = (d.get('ohlc') or {}).get('close')
        entry = float(p.get('entry_price') or 0)
        qty = float(p.get('qty') or 0)
        if not ltp:
            ltp = entry
        val = qty * ltp
        pnl = qty * (ltp - entry) if entry else 0.0
        tot_val += val
        tot_pnl += pnl
        _stop = donchian_stop(s)
        rows.append(dict(
            stop=_stop,
            to_stop_pct=(round((ltp / _stop - 1) * 100, 1) if (_stop and ltp) else None),
            symbol=s, qty=qty, entry_price=round(entry, 2), ltp=round(float(ltp), 2),
            prev_close=round(float(prev), 2) if prev else None,
            day_move_pct=round((ltp / prev - 1) * 100, 2) if prev else None,
            value=round(val), pnl=round(pnl),
            pnl_pct=round((ltp / entry - 1) * 100, 2) if entry else None))

    cash = float(mp._get('cash', 0.0) or 0.0)
    try:
        swept = float(mp._sweep_value())
    except Exception:
        swept = 0.0
    nav = tot_val + cash + swept
    capital = float(mp._get('capital', 0.0) or 0.0)

    ui = dict(updated=str(datetime.now()), asof=str(date.today()),
              positions=rows, value=round(tot_val), cash=round(cash),
              swept=round(swept), nav=round(nav), capital=round(capital),
              pnl=round(tot_pnl),
              pnl_pct=round(100 * tot_pnl / (tot_val - tot_pnl), 2) if tot_val - tot_pnl else 0,
              n=len(rows), slots=mp.CFG['n_hold'],
              # realised P&L on closed trades: one SUM, so the desk does not have to call
              # the state endpoint (10-16s) just to complete its P&L breakdown
              realized=_realized(),
              # what the swept cash has actually earned: ETF value less its cost
              interest=round(swept - float(mp._get('sweep_cost', 0.0) or 0.0)) if swept else 0,
              source='gen_momentum_live')
    tmp = OUT.with_suffix('.json.tmp')
    json.dump(ui, open(tmp, 'w'), indent=1, default=str)
    os.replace(tmp, OUT)
    print(f'{datetime.now():%H:%M:%S} baked {len(rows)} positions, '
          f'value Rs {tot_val:,.0f} P&L {tot_pnl:+,.0f} nav Rs {nav:,.0f}')


if __name__ == '__main__':
    main()
