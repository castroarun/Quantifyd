#!/usr/bin/env python3
"""Manual ATM4-roll replica manager (2026-07-29, user-directed) — v2.

Manages EXACTLY my two replica legs, decoupled from netted positions (the
77500 PE is SHARED with the live ATM2 book, so position-qty is -80; I must
never act on that). I track my own fixed 40-qty legs by LTP, and detect a leg
closing via MY broker SL order id. On exit I BUY exactly 40 per open leg —
never the netted qty — so ATM2's leg is untouched.

Exit: net MTM (my 40+40, by LTP) <= -Rs5,000, or 15:15 EOD (hard 15:29).
Each leg also has its own broker-resting SL as the catastrophe backstop.
Only ever BUYs to cover; never opens. Idempotent; exits once flat.
"""
import sys, os, time, json, datetime
sys.path.insert(0, '/home/arun/quantifyd')
os.chdir('/home/arun/quantifyd')
from services.kite_service import get_kite

EXCH, PROD, VAR = 'BFO', 'MIS', 'regular'
LEGS = [
    {'t': 'SENSEX26JUL77500PE', 'entry': 173.25, 'qty': 40, 'sl': '260729150924499', 'open': True},
    {'t': 'SENSEX26JUL77800CE', 'entry': 152.15, 'qty': 40, 'sl': '260729150924503', 'open': True},
]
NET_STOP, EOD, HARD = -5000.0, '15:15', '15:29'
STATUS = '/home/arun/quantifyd/logs/replica_monitor_status.json'
LOG = '/home/arun/quantifyd/logs/replica_monitor.log'


def log(m):
    line = f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {m}'
    with open(LOG, 'a') as f:
        f.write(line + '\n')
    print(line, flush=True)


def sl_status(k, oid):
    try:
        h = k.order_history(oid) or []
        return h[-1].get('status', '?') if h else '?'
    except Exception as e:
        return 'ERR:' + str(e)[:40]


def ltps(k, tsyms):
    try:
        q = k.quote(['BFO:' + t for t in tsyms])
        return {t: (q.get('BFO:' + t) or {}).get('last_price') for t in tsyms}
    except Exception as e:
        log('quote err ' + str(e)); return {t: None for t in tsyms}


def cancel_sl(k, oid):
    try:
        k.cancel_order(variety=VAR, order_id=oid); log(f'cancelled SL {oid}')
    except Exception as e:
        log(f'cancel SL {oid} skipped ({str(e)[:40]})')


def close_leg(k, L):
    try:
        oid = k.place_order(variety=VAR, exchange=EXCH, tradingsymbol=L['t'],
                            transaction_type='BUY', quantity=L['qty'],
                            product=PROD, order_type='MARKET')
        log(f"CLOSE {L['t']} qty={L['qty']} oid={oid}")
    except Exception as e:
        log(f"CLOSE FAIL {L['t']}: {e}")


def main():
    k = get_kite()
    log(f'monitor v2 START — my legs only (40 each), net_stop={NET_STOP}, eod={EOD}')
    while True:
        now = datetime.datetime.now(); hm = now.strftime('%H:%M')
        try:
            # detect legs already closed by their own broker SL
            for L in LEGS:
                if L['open'] and sl_status(k, L['sl']) == 'COMPLETE':
                    L['open'] = False; log(f"{L['t']} closed by broker SL")
            openL = [L for L in LEGS if L['open']]
            if not openL:
                with open(STATUS, 'w') as f:
                    json.dump({'ts': str(now), 'state': 'FLAT'}, f, indent=1)
                log('all my legs flat -> monitor exit'); break
            lp = ltps(k, [L['t'] for L in openL])
            net = sum((L['entry'] - lp[L['t']]) * L['qty'] for L in openL if lp.get(L['t']) is not None)
            reason = 'NET_STOP_5000' if net <= NET_STOP else ('HARD_EOD' if hm >= HARD else ('EOD_1515' if hm >= EOD else None))
            with open(STATUS, 'w') as f:
                json.dump({'ts': str(now), 'net_mtm': round(net), 'net_stop': NET_STOP,
                           'action': reason or 'HOLD',
                           'legs': [{'t': L['t'], 'entry': L['entry'], 'ltp': lp.get(L['t']), 'qty': L['qty']} for L in openL]},
                          f, indent=1, default=str)
            log('net=%.0f %s -> %s' % (net, ' '.join(f"{L['t'][-4:]}={lp.get(L['t'])}" for L in openL), reason or 'HOLD'))
            if reason:
                for L in openL:
                    cancel_sl(k, L['sl'])
                    close_leg(k, L)
                    L['open'] = False
                log(f'exited all my legs on {reason}'); break
        except Exception as e:
            log(f'cycle err: {e}')
        time.sleep(30)
    log('monitor END')


if __name__ == '__main__':
    main()
