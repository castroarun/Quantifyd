#!/usr/bin/env python3
"""research/128 stage 1 - synthesise SENSEX naked-survivor episodes.

Replays the DEPLOYED SENSEX ATM and ATM4 rules over every usable chain day and emits, for
each day/system that produced a naked survivor, the survivor's 1-minute premium path from
the moment it became naked to the 15:15 EOD squareoff.

READ-ONLY on backtest_data/options_data.db.
Outputs: results/episodes.csv  (metadata, one row per episode)
         results/paths.jsonl   (premium path per episode)
"""
import sqlite3, csv, json, os
from datetime import date, timedelta

Q = '/home/arun/quantifyd/'
CHAIN = Q + 'backtest_data/options_data.db'
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
LOG = os.path.join(RES, 'build_episodes.log')

LOT = 20            # SENSEX lot (option_chain.lot_size is WRONG - research/119)
STEP = 100
LEG_SL = 0.30
ROLL_MIN_OTM = 50
ENTRY_M = 9 * 60 + 16
EOD_M = 15 * 60 + 15
WD = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']


def log(m):
    with open(LOG, 'a') as f:
        f.write(m + '\n')
    print(m, flush=True)


def hm2m(h):
    return int(h[:2]) * 60 + int(h[3:5])


def m2hm(m):
    return '%02d:%02d' % (m // 60, m % 60)


def trading_dte(day, exp):
    d0, d1 = date.fromisoformat(day), date.fromisoformat(exp)
    n, d = 0, d0
    while d < d1:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def all_days(c):
    return [r[0] for r in c.execute(
        "SELECT DISTINCT substr(snapshot_time,1,10) d FROM option_chain "
        "WHERE symbol='SENSEX' ORDER BY d")
        if date.fromisoformat(r[0]).weekday() < 5]


def load_day(c, day):
    rows = c.execute(
        "SELECT snapshot_time, expiry_date, strike, instrument_type, ltp, underlying_spot "
        "FROM option_chain WHERE symbol='SENSEX' AND snapshot_time>=? AND snapshot_time<? "
        "AND ltp IS NOT NULL", (day, day + 'z')).fetchall()
    if not rows:
        return None, 'no rows'
    last_snap = max(r[0] for r in rows)
    if last_snap[11:16] < '15:15':
        return None, 'partial session (last %s)' % last_snap[11:16]
    exps = sorted({e for (_, e, _, _, _, _) in rows if e and e >= day})
    if not exps:
        return None, 'no expiry'
    fexp = exps[0]
    spot, chain = {}, {}
    for st, e, k, it, ltp, sp in rows:
        mi = hm2m(st[11:16])
        if sp and mi not in spot:
            spot[mi] = sp
        if e != fexp:
            continue
        chain.setdefault(mi, {}).setdefault(k, {})[it] = ltp
    if len(set(spot.values())) < 50:
        return None, 'frozen chain (holiday guard)'
    mins = sorted(m for m in chain if ENTRY_M <= m <= EOD_M)
    if len(mins) < 200:
        return None, 'thin (%d mins)' % len(mins)
    return (fexp, spot, chain, mins), None


def px(chain, mi, K, side):
    d = chain.get(mi)
    if not d:
        return None
    v = d.get(K)
    if not v:
        return None
    return v.get(side)


def last_px(chain, mi, K, side, back=5):
    for j in range(0, back + 1):
        p = px(chain, mi - j, K, side)
        if p is not None and p > 0:
            return p
    return None


def path_of(chain, m0, m1, K, side):
    out = []
    for mi in range(m0, m1 + 1):
        p = px(chain, mi, K, side)
        if p is not None and p > 0:
            out.append((mi, round(p, 2)))
    return out


def find_roll_strike(chain, mi, spot, side, target):
    """Mirror NasAtm4Executor._find_roll_strike on the recorded chain."""
    atm = int(round(spot / STEP)) * STEP
    best = (None, None, float('inf'))
    for i in range(15):
        strike = atm + i * STEP if side == 'CE' else atm - i * STEP
        if side == 'CE' and strike - spot < ROLL_MIN_OTM:
            continue
        if side == 'PE' and spot - strike < ROLL_MIN_OTM:
            continue
        prem = last_px(chain, mi, strike, side)
        if prem is None or prem <= 0:
            continue
        diff = abs(prem - target)
        if diff < best[2]:
            best = (strike, prem, diff)
        elif best[0] is not None and prem < target:
            break
    return best[0], best[1]


def K_ok(chain, spot, m):
    sp = spot.get(m)
    if not sp:
        return False
    K = int(round(sp / STEP)) * STEP
    d = chain.get(m, {}).get(K)
    return bool(d and 'CE' in d and 'PE' in d)


def replay_day(day, fexp, spot, chain, mins, dte):
    eps = []
    m0 = next((m for m in mins if m >= ENTRY_M and K_ok(chain, spot, m)), None)
    if m0 is None:
        return eps, 'no entry minute'
    sp0 = spot[m0]
    K = int(round(sp0 / STEP)) * STEP
    ce0, pe0 = px(chain, m0, K, 'CE'), px(chain, m0, K, 'PE')
    if not ce0 or not pe0:
        return eps, 'no ATM quote at entry'
    if dte in (0,):     # SENSEX_ATM_DEFAULTS leg_sl_disabled_dtes=(0,)
        return eps, 'DTE0 - no per-leg stop, no survivor route'

    legs = {'CE': dict(K=K, entry=ce0, sl=ce0 * (1 + LEG_SL)),
            'PE': dict(K=K, entry=pe0, sl=pe0 * (1 + LEG_SL))}

    t1 = None
    for mi in range(m0 + 1, EOD_M + 1):
        for side in ('CE', 'PE'):
            L = legs[side]
            p = px(chain, mi, L['K'], side)
            if p is not None and p >= L['sl']:
                t1 = (mi, side, p)
                break
        if t1:
            break
    if not t1:
        return eps, 'no leg SL all day (both held to EOD)'

    smi, sside, _sp = t1
    osd = 'PE' if sside == 'CE' else 'CE'
    O = legs[osd]

    # ---- ATM: survivor naked from the FIRST SL ----
    surv_p = last_px(chain, smi, O['K'], osd)
    pth = path_of(chain, smi, EOD_M, O['K'], osd)
    pre = path_of(chain, m0, smi - 1, O['K'], osd)
    if surv_p and len(pth) >= 5:
        eps.append(dict(system='ATM', day=day, expiry=fexp, dte=dte, pre=pre, entry_m=m0,
                        weekday=WD[date.fromisoformat(day).weekday()],
                        side=osd, strike=O['K'], entry=round(O['entry'], 2),
                        naked_m=smi, naked_hm=m2hm(smi),
                        naked_px=round(surv_p, 2), spot0=round(sp0, 2),
                        stopped_side=sside, n_path=len(pth), path=pth))

    # ---- ATM4: roll on first SL; survivor naked from the SECOND SL ----
    price_x = surv_p
    spx = spot.get(smi) or spot.get(smi - 1)
    if price_x and spx:
        nk, np_ = find_roll_strike(chain, smi, spx, sside, price_x)
        if nk and np_:
            rolled = dict(K=nk, entry=np_, sl=max(price_x, np_) * 1.3)
            O4 = dict(O)
            O4['sl'] = price_x * 1.3
            t2 = None
            for mi in range(smi + 1, EOD_M + 1):
                for L, side in ((rolled, sside), (O4, osd)):
                    p = px(chain, mi, L['K'], side)
                    if p is not None and p >= L['sl']:
                        t2 = (mi, side)
                        break
                if t2:
                    break
            if t2:
                s2mi, s2side = t2
                SV, svside = (O4, osd) if s2side == sside else (rolled, sside)
                surv2 = last_px(chain, s2mi, SV['K'], svside)
                pth2 = path_of(chain, s2mi, EOD_M, SV['K'], svside)
                pre2 = path_of(chain, m0, s2mi - 1, SV['K'], svside)
                if surv2 and len(pth2) >= 5:
                    eps.append(dict(system='ATM4', day=day, expiry=fexp, dte=dte,
                                    pre=pre2, entry_m=m0,
                                    weekday=WD[date.fromisoformat(day).weekday()],
                                    side=svside, strike=SV['K'],
                                    entry=round(SV['entry'], 2),
                                    naked_m=s2mi, naked_hm=m2hm(s2mi),
                                    naked_px=round(surv2, 2), spot0=round(sp0, 2),
                                    stopped_side=s2side, n_path=len(pth2), path=pth2))
    return eps, None


def main():
    os.makedirs(RES, exist_ok=True)
    open(LOG, 'w').close()
    c = sqlite3.connect('file:%s?mode=ro' % CHAIN, uri=True)
    days = all_days(c)
    log('SENSEX candidate days: %d (%s .. %s)' % (len(days), days[0], days[-1]))
    fg = ['system', 'day', 'weekday', 'expiry', 'dte', 'side', 'strike', 'entry',
          'naked_hm', 'naked_m', 'naked_px', 'spot0', 'stopped_side', 'n_path']
    fout = open(os.path.join(RES, 'episodes.csv'), 'w', newline='')
    w = csv.DictWriter(fout, fieldnames=fg)
    w.writeheader()
    pj = open(os.path.join(RES, 'paths.jsonl'), 'w')
    kept = nep = 0
    for day in days:
        d, why = load_day(c, day)
        if not d:
            log('  SKIP %s: %s' % (day, why))
            continue
        fexp, spot, chain, mins = d
        dte = trading_dte(day, fexp)
        kept += 1
        eps, why = replay_day(day, fexp, spot, chain, mins, dte)
        if why:
            log('  %s dte=%d -> %s' % (day, dte, why))
            continue
        for e in eps:
            eid = '%s_%s' % (e['system'], e['day'])
            w.writerow({k: e[k] for k in fg})
            pj.write(json.dumps(dict(eid=eid, **{k: e[k] for k in
                     ('system', 'day', 'weekday', 'dte', 'side', 'strike', 'entry',
                      'naked_m', 'naked_px', 'entry_m', 'pre', 'path')})) + '\n')
            nep += 1
        log('  %s dte=%d wd=%s -> %d episode(s) %s' % (
            day, dte, WD[date.fromisoformat(day).weekday()], len(eps),
            ','.join('%s@%s' % (x['system'], x['naked_hm']) for x in eps)))
        fout.flush(); pj.flush()
    log('DONE: usable days=%d episodes=%d' % (kept, nep))


if __name__ == '__main__':
    main()
