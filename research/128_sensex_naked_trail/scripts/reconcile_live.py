#!/usr/bin/env python3
"""research/128 stage 3 - reconcile the replay harness against the 12 REAL live episodes.

For every SENSEX naked-survivor leg that the deployed (broken) trail actually closed
(exit_reason='SL_HIT' AND notes LIKE '%ST_TRAIL%'), pull its real 1-minute premium path
from the recorded chain and replay every arm on it. Reports:
  * harness fidelity: replayed INCUMBENT exit vs the ACTUAL live exit
  * what BE_ONLY / HOLD / the recommended arm would have booked on each episode

READ-ONLY on all DBs.  Outputs results/live_reconciliation.csv
"""
import csv, json, os, sys, sqlite3
sys.path.insert(0, '/home/arun/quantifyd')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sweep_arms import sim, leg_cost_per_lot, build_arms, LOT

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
CHAIN = '/home/arun/quantifyd/backtest_data/options_data.db'
EOD_M = 15 * 60 + 15
ENTRY_M = 9 * 60 + 16

RECOMMENDED = os.environ.get('R128_ARM', 'CEIL_p10_m2.5_N2_SEED')


def hm2m(h):
    return int(h[:2]) * 60 + int(h[3:5])


def m2hm(m):
    return '%02d:%02d' % (m // 60, m % 60)


def live_episodes():
    out = []
    for db, sysname in (('sensex_atm_trading', 'ATM'), ('sensex_atm4_trading', 'ATM4')):
        c = sqlite3.connect('file:/home/arun/quantifyd/backtest_data/%s.db?mode=ro' % db, uri=True)
        cols = [x[1] for x in c.execute('PRAGMA table_info(nas_atm_positions)')]
        for r in c.execute("SELECT * FROM nas_atm_positions WHERE exit_reason='SL_HIT' "
                           "AND notes LIKE '%ST_TRAIL%' ORDER BY id"):
            d = dict(zip(cols, r))
            d['sysname'] = sysname
            out.append(d)
    return out


def leg_path(c, day, expiry, strike, side, m_from, m_to):
    rows = c.execute(
        "SELECT snapshot_time, ltp FROM option_chain WHERE symbol='SENSEX' AND expiry_date=? "
        "AND strike=? AND instrument_type=? AND snapshot_time>=? AND snapshot_time<? "
        "AND ltp IS NOT NULL ORDER BY snapshot_time", (expiry, float(strike), side, day, day + 'z')).fetchall()
    out = []
    seen = set()
    for st, ltp in rows:
        mi = hm2m(st[11:16])
        if mi < m_from or mi > m_to or mi in seen or not ltp or ltp <= 0:
            continue
        seen.add(mi)
        out.append((mi, round(ltp, 2)))
    return out


def main():
    c = sqlite3.connect('file:%s?mode=ro' % CHAIN, uri=True)
    arms = {a['label']: a for a in build_arms()}
    want = ['INCUMBENT', 'BE_ONLY', 'HOLD_EOD', RECOMMENDED]
    fields = ['sys', 'day', 'weekday', 'tsym', 'lots', 'entry', 'naked_hm',
              'live_exit_hm', 'live_exit_px', 'live_net_lot', 'live_net_book',
              'replay_inc_hm', 'replay_inc_px', 'fidelity',
              'be_hm', 'be_px', 'be_net_lot',
              'hold_px', 'hold_net_lot',
              'rec_hm', 'rec_px', 'rec_reason', 'rec_net_lot',
              'rec_minus_live_book', 'be_minus_live_book', 'path_end_hm']
    rows_out = []
    for p in live_episodes():
        day = p['entry_time'][:10]
        lots = int(p['qty']) // LOT
        # when did it become naked? = the sibling leg's exit time in the same strangle
        dbname = 'sensex_atm_trading' if p['sysname'] == 'ATM' else 'sensex_atm4_trading'
        cc = sqlite3.connect('file:/home/arun/quantifyd/backtest_data/%s.db?mode=ro' % dbname, uri=True)
        cols = [x[1] for x in cc.execute('PRAGMA table_info(nas_atm_positions)')]
        sib = [dict(zip(cols, r)) for r in cc.execute(
            'SELECT * FROM nas_atm_positions WHERE strangle_id=? AND id<>?',
            (p['strangle_id'], p['id']))]
        exits = [s['exit_time'] for s in sib if s['exit_time']]
        naked_m = max(hm2m(x[11:16]) for x in exits) if exits else hm2m(p['entry_time'][11:16])
        path = leg_path(c, day, p['expiry_date'], p['strike'], p['instrument_type'],
                        naked_m, EOD_M)
        pre = leg_path(c, day, p['expiry_date'], p['strike'], p['instrument_type'],
                       ENTRY_M, naked_m - 1)
        if len(path) < 5:
            print('SKIP %s %s: no chain path' % (p['sysname'], day))
            continue
        ep = dict(eid='%s_%s' % (p['sysname'], day), system=p['sysname'], day=day,
                  weekday='', dte=0, side=p['instrument_type'], strike=p['strike'],
                  entry=float(p['entry_price']), naked_m=naked_m,
                  naked_px=path[0][1], entry_m=ENTRY_M, pre=pre, path=path)
        E = ep['entry']
        live_px = float(p['exit_price'])
        live_net = (E - live_px) * LOT - leg_cost_per_lot(E, live_px, LOT, 'INCUMBENT')
        res = {}
        for lbl in want:
            m, px, reason = sim(ep, arms[lbl])
            res[lbl] = (m, px, reason,
                        (E - px) * LOT - leg_cost_per_lot(E, px, LOT, reason))
        im, ipx, _ir, _in = res['INCUMBENT']
        fid = 'EXACT' if abs(im - hm2m(p['exit_time'][11:16])) <= 1 else (
            'CLOSE' if abs(im - hm2m(p['exit_time'][11:16])) <= 10 else 'DIVERGE')
        rows_out.append(dict(
            sys=p['sysname'], day=day, weekday='', tsym=p['tradingsymbol'], lots=lots,
            entry=round(E, 2), naked_hm=m2hm(naked_m),
            live_exit_hm=p['exit_time'][11:16], live_exit_px=live_px,
            live_net_lot=round(live_net), live_net_book=round(live_net * lots),
            replay_inc_hm=m2hm(im), replay_inc_px=round(ipx, 2), fidelity=fid,
            be_hm=m2hm(res['BE_ONLY'][0]), be_px=round(res['BE_ONLY'][1], 2),
            be_net_lot=round(res['BE_ONLY'][3]),
            hold_px=round(res['HOLD_EOD'][1], 2), hold_net_lot=round(res['HOLD_EOD'][3]),
            rec_hm=m2hm(res[RECOMMENDED][0]), rec_px=round(res[RECOMMENDED][1], 2),
            rec_reason=res[RECOMMENDED][2], rec_net_lot=round(res[RECOMMENDED][3]),
            rec_minus_live_book=round((res[RECOMMENDED][3] - live_net) * lots),
            be_minus_live_book=round((res['BE_ONLY'][3] - live_net) * lots),
            path_end_hm=m2hm(path[-1][0])))

    with open(os.path.join(RES, 'live_reconciliation.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print('episodes: %d   recommended arm: %s' % (len(rows_out), RECOMMENDED))
    tot_live = sum(r['live_net_book'] for r in rows_out)
    tot_rec = sum(r['live_net_book'] + r['rec_minus_live_book'] for r in rows_out)
    tot_be = sum(r['live_net_book'] + r['be_minus_live_book'] for r in rows_out)
    print('fidelity: %s' % {k: sum(1 for r in rows_out if r['fidelity'] == k)
                            for k in ('EXACT', 'CLOSE', 'DIVERGE')})
    print('booked (net, at traded lots)  LIVE %+d | BE_ONLY %+d | %s %+d'
          % (tot_live, tot_be, RECOMMENDED, tot_rec))
    for r in rows_out:
        print('%-5s %s %-20s x%d E=%7.2f naked %s | live %s @%7.2f net/lot %+6d | '
              'BE %s @%7.2f %+6d | REC %s @%7.2f %-9s %+6d | hold @%7.2f %+6d'
              % (r['sys'], r['day'], r['tsym'], r['lots'], r['entry'], r['naked_hm'],
                 r['live_exit_hm'], r['live_exit_px'], r['live_net_lot'],
                 r['be_hm'], r['be_px'], r['be_net_lot'],
                 r['rec_hm'], r['rec_px'], r['rec_reason'], r['rec_net_lot'],
                 r['hold_px'], r['hold_net_lot']))


if __name__ == '__main__':
    main()
