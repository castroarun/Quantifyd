"""Open Alpha REAL book — state, marks, and the EOD-faithful exit checker.

Seeded 04-Sep-2026 (Arun's explicit go, ahead of the Dec-5 soak gate — logged
override). Rules mirror the paper spec: -8% hard stop on CLOSE, 15-SMA trail on
CLOSE (entry-day trail exempt). Real execution is manual-assisted for now:

  mode `mark`  : refresh static/app/oa_real.json from live quotes (page display)
  mode `check` : 15:18 IST close-proxy check — if price is below the stop or the
                 15-SMA trail, raise a desktop alert with the exact sell order.
                 ALERT-ONLY: this script never places orders.
  mode `seed`  : build state from today's executed CNC orders (one-off)

State: backtest_data/oa_real_state.json   Feed: /tmp/nas_alert_feed.log (popups)
"""
import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'backtest_data' / 'oa_real_state.json'
UI = ROOT / 'static' / 'app' / 'oa_real.json'
FEED = Path('/tmp/nas_alert_feed.log')
STOP_PCT = 0.08
TRAIL_N = 15
SYMS16 = ['INDSWFTLAB', 'SETL', 'WELCORP', 'SHILPAMED', 'KMEW', 'SBCL', 'IOLCP',
          'SPORTKING', 'IRISDOREME', 'INOXINDIA', 'MANINDS', 'SSWL', 'ENTERO',
          'NITINSPIN', 'TMB', 'KTKBANK']


def _kite():
    from kiteconnect import KiteConnect
    api_key = [l.split('=', 1)[1].strip() for l in open(ROOT / '.env')
               if l.startswith('KITE_API_KEY')][0]
    tok = json.load(open(ROOT / 'backtest_data' / 'access_token.json'))
    k = KiteConnect(api_key=api_key)
    k.set_access_token(tok.get('access_token') or tok.get('token'))
    return k


def _alert(title, body, urgency='critical'):
    with open(FEED, 'a') as f:
        f.write(json.dumps(dict(ts=str(datetime.now()), book='OA-REAL', urgency=urgency,
                                title=title, body=body)) + '\n')


def _sma15(kite, syms, live):
    """SMA15 close-proxy per symbol: last 14 DB closes + today's live price."""
    import sqlite3
    con = sqlite3.connect(str(ROOT / 'backtest_data' / 'market_data.db'))
    out = {}
    for s in syms:
        rows = [r[0] for r in con.execute(
            "SELECT close FROM market_data_unified WHERE symbol=? AND timeframe='day' "
            "ORDER BY date DESC LIMIT 14", (s,))]
        if len(rows) == 14 and live.get(s):
            out[s] = (sum(rows) + live[s]) / 15.0
    con.close()
    return out


def seed():
    kite = _kite()
    fills = {}
    for o in kite.orders():
        if (o['status'] == 'COMPLETE' and o['transaction_type'] == 'BUY'
                and o['product'] == 'CNC' and o['tradingsymbol'] in SYMS16):
            f = fills.setdefault(o['tradingsymbol'], dict(qty=0, value=0.0))
            f['qty'] += o['filled_quantity']
            f['value'] += o['filled_quantity'] * o['average_price']
    positions = []
    invested = 0.0
    for s in SYMS16:
        f = fills.get(s)
        if not f or f['qty'] == 0:
            print(f'WARNING: no fill for {s}')
            continue
        buy = f['value'] / f['qty']
        invested += f['value']
        positions.append(dict(symbol=s, qty=f['qty'], buy=round(buy, 2),
                              entry_date=str(date.today()),
                              stop=round(buy * (1 - STOP_PCT), 2), src='real'))
    st = dict(book='OA-REAL', seeded=str(datetime.now()), positions=positions,
              invested=round(invested, 0),
              note='Seeded 04-Sep-2026 from Arun-executed CNC fills (top-16 by RS of the '
                   'day\'s 21 triggered candidates). LIQUIDCASE 1757u sold to fund. '
                   'Deliberate override of the Dec-5 soak gate. Exits manual-assisted: '
                   '15:18 checker alerts; no automated selling yet.',
              trades=[])
    json.dump(st, open(STATE, 'w'), indent=1)
    print(f'seeded {len(positions)} positions, invested Rs {invested:,.0f}')
    mark()


def mark():
    kite = _kite()
    st = json.load(open(STATE))
    syms = [p['symbol'] for p in st['positions']]
    q = {}
    for i in range(0, len(syms), 25):
        q.update(kite.quote(['NSE:' + s for s in syms[i:i+25]]))
    live = {s: q.get('NSE:' + s, {}).get('last_price') for s in syms}
    smas = _sma15(kite, syms, live)
    rows, tot_val, tot_pnl = [], 0.0, 0.0
    for p in st['positions']:
        lp = live.get(p['symbol'])
        oh = q.get('NSE:' + p['symbol'], {}).get('ohlc', {})
        prev = oh.get('close')
        val = p['qty'] * lp if lp else p['qty'] * p['buy']
        pnl = p['qty'] * (lp - p['buy']) if lp else 0.0
        tot_val += val
        tot_pnl += pnl
        sma = smas.get(p['symbol'])
        days_held = (date.today() - date.fromisoformat(p['entry_date'])).days
        rows.append(dict(**p, ltp=lp, days=days_held,
                         day_move_pct=round((lp / prev - 1) * 100, 2) if lp and prev else None,
                         value=round(val), pnl=round(pnl),
                         pnl_pct=round((lp / p['buy'] - 1) * 100, 2) if lp else None,
                         trail=round(sma, 2) if sma else None,
                         to_stop_pct=round((lp / p['stop'] - 1) * 100, 1) if lp else None,
                         to_trail_pct=round((lp / sma - 1) * 100, 1) if lp and sma else None))
    cash = float(st.get('cash', 0.0))
    nav = tot_val + cash
    for r in rows:
        r['weight'] = round(100 * r['value'] / nav, 1) if nav else 0
    # append the daily nav point on the post-close mark (>= 16:00 IST)
    if datetime.now().hour >= 16:
        nc = st.setdefault('navcurve', [])
        today_s = str(date.today())
        nc[:] = [x for x in nc if x['d'] != today_s]
        nc.append(dict(d=today_s, nav=round(nav)))
        json.dump(st, open(STATE, 'w'), indent=1)
    realized = sum(t.get('net_pnl', 0) for t in st.get('trades', []))
    ui = dict(updated=str(datetime.now()), positions=rows, invested=st['invested'],
              value=round(tot_val), cash=round(cash), nav=round(nav),
              pnl=round(tot_pnl), realized=round(realized),
              pnl_pct=round(100 * tot_pnl / st['invested'], 2) if st['invested'] else 0,
              inception='04-Sep-2026', navcurve=st.get('navcurve', []),
              note=st['note'], trades=st.get('trades', []))
    json.dump(ui, open(UI, 'w'), indent=1)
    print(f"marked {len(rows)} positions: value Rs {tot_val:,.0f} P&L {tot_pnl:+,.0f}")


def check():
    """15:18 close-proxy rule check. Alert-only."""
    kite = _kite()
    st = json.load(open(STATE))
    syms = [p['symbol'] for p in st['positions']]
    q = {}
    for i in range(0, len(syms), 25):
        q.update(kite.quote(['NSE:' + s for s in syms[i:i+25]]))
    live = {s: q.get('NSE:' + s, {}).get('last_price') for s in syms}
    smas = _sma15(kite, syms, live)
    today = str(date.today())
    hits = []
    for p in st['positions']:
        lp = live.get(p['symbol'])
        if not lp:
            continue
        if lp <= p['stop']:
            hits.append((p, lp, f"below -8% stop {p['stop']}"))
        elif p['entry_date'] != today and smas.get(p['symbol']) and lp < smas[p['symbol']]:
            hits.append((p, lp, f"below 15-SMA trail {smas[p['symbol']]:.2f}"))
    if not hits:
        _alert('OA-REAL 15:18 check: all clear', f'{len(syms)} positions, no exits due', 'low')
        print('all clear')
    for p, lp, why in hits:
        msg = (f"SELL {p['symbol']} x{p['qty']} CNC (limit ~{lp:.2f}) — {why}. "
               f"Entry {p['buy']}, now {lp} ({(lp/p['buy']-1)*100:+.1f}%). Place before 15:30.")
        _alert(f"OA-REAL EXIT DUE: {p['symbol']}", msg)
        print('EXIT DUE:', msg)


if __name__ == '__main__':
    {'seed': seed, 'mark': mark, 'check': check}[sys.argv[1] if len(sys.argv) > 1 else 'mark']()
