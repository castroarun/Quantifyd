#!/usr/bin/env python3
"""
research/115_pcr_oi_adjustments -- G2: does ACTING on a PCR/OI warning beat DOING NOTHING?

G1c already killed the whole PCR / dOI / OI-wall / max-pain family on information grounds.
This script answers the question in rupees anyway, because that is the form Arun asked it
in, and because a clean rupee table is harder to argue with than a correlation.

BOOK MODELLED (our live construction):
  short 1 ATM straddle at 09:16 on the nearest expiry, exit at 15:15.
  NIFTY lot 75, SENSEX lot 10 (from the recorded chain).

COSTS  (per leg-side, per lot):
  NIFTY : 0.5 pt slippage x 75 = Rs 37.5  + Rs 30 brokerage/charges = Rs 67.5
  SENSEX: 1.0 pt slippage x 10 = Rs 10.0  + Rs 30                   = Rs 40.0
  EVERY arm here pays exactly 4 leg-sides (2 in, 2 out) -- exiting early or exiting one
  leg early does NOT add leg-sides.  So this comparison gives the adjustment rules the
  MOST FAVOURABLE possible treatment: zero incremental cost.  If they still lose to HOLD,
  the conclusion is not a cost artifact.

ARMS
  HOLD          baseline -- carry both legs to the time exit (research/114's champion)
  EXIT_ALL      on trigger, close the straddle
  EXIT_LEG      on trigger, close only the losing (threatened) leg, carry the other
  STOP12/STOP13 on trigger, arm a combined stop at 1.2x / 1.3x the entry premium

CONTROLS
  RANDOM_<n>    trigger at a uniformly random minute, 20 deterministic draws averaged --
                the date-matched / random-entry control this repo requires.  Any arm that
                does not beat its own random-trigger twin is reacting to noise.
  MOVE_0.3/0.5  a PURE PRICE trigger (|spot move| from entry) -- the move-stop we already
                run live, included as a reference point with no option information in it.
"""
import csv, os, math, json, random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, '..', 'results'))
MASTER = os.path.join(OUT, 'features_master.csv')

EXIT_BY = '15:15'
COST_PER_LEGSIDE = {'NIFTY': 67.5, 'SENSEX': 40.0}
LEGSIDES = 4
N_RANDOM = 20

# ---- PRE-REGISTERED TRIGGER LIST (counted in the multiple-testing accounting) --------
def triggers():
    T = []
    for x in (1.2, 1.4, 1.6):
        T.append(('pcr_oi_all>%.1f' % x, lambda r, x=x: r['pcr_oi_all'] is not None and r['pcr_oi_all'] > x))
    for x in (0.6, 0.8):
        T.append(('pcr_oi_all<%.1f' % x, lambda r, x=x: r['pcr_oi_all'] is not None and r['pcr_oi_all'] < x))
    for x in (0.05, 0.10):
        T.append(('d_pcr_oi_15>%.2f' % x, lambda r, x=x: r['d_pcr_oi_all_15'] is not None and r['d_pcr_oi_all_15'] > x))
        T.append(('d_pcr_oi_15<-%.2f' % x, lambda r, x=x: r['d_pcr_oi_all_15'] is not None and r['d_pcr_oi_all_15'] < -x))
        T.append(('doi_net_15>%.2f' % x, lambda r, x=x: r['doi_net_15'] is not None and r['doi_net_15'] > x))
        T.append(('doi_net_15<-%.2f' % x, lambda r, x=x: r['doi_net_15'] is not None and r['doi_net_15'] < -x))
    for x in (0.02, 0.05):
        T.append(('doi_atm_15<-%.2f' % x, lambda r, x=x: r['doi_atm_15'] is not None and r['doi_atm_15'] < -x))
    T.append(('abs_doi_atm_15>0.05', lambda r: r['doi_atm_15'] is not None and abs(r['doi_atm_15']) > 0.05))
    for x in (0.001, 0.002):
        T.append(('wall_near<%.3f' % x, lambda r, x=x: (r['dist_wall_ce'] is not None and r['dist_wall_pe'] is not None
                  and min(abs(r['dist_wall_ce']), abs(r['dist_wall_pe'])) < x)))
        T.append(('abs_mp_drift_15>%.3f' % x, lambda r, x=x: r['mp_drift_15'] is not None and abs(r['mp_drift_15']) > x))
    for x in (1.0, 2.0):
        T.append(('d_atm_iv_15>%.1f' % x, lambda r, x=x: r['d_atm_iv_15'] is not None and r['d_atm_iv_15'] > x))
    # pure-price reference (contains NO option information)
    for x in (0.003, 0.005):
        T.append(('MOVE>%.1f%%' % (x * 100), lambda r, x=x: abs(r['spot'] / r['_spot0'] - 1.0) > x))
    return T


def fnum(s):
    if s in ('', None):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load():
    days = defaultdict(list)
    with open(MASTER) as f:
        for r in csv.DictReader(f):
            days[(r['date'], r['venue'])].append(r)
    out = {}
    for u, rows in days.items():
        rows = [r for r in rows if r['minute'] <= EXIT_BY]
        rr = []
        for r in rows:
            d = {k: fnum(r[k]) for k in r if k not in ('date', 'venue', 'minute')}
            d['minute'] = r['minute']
            rr.append(d)
        if len(rr) < 120:
            continue
        good = [r for r in rr if r['ce0'] is not None and r['pe0'] is not None]
        if len(good) < 120:
            continue
        s0 = good[0]
        for r in rr:
            r['_spot0'] = s0['spot']
        out[u] = dict(rows=good, ce_e=s0['ce0'], pe_e=s0['pe0'],
                      prem_e=s0['ce0'] + s0['pe0'], lot=s0['lot'] or 0, dte=s0['dte'])
    return out


def pnl(day, venue, trig_idx, action):
    """gross rupee P&L per lot for one arm.  trig_idx None => never triggered."""
    rows, ce_e, pe_e, lot = day['rows'], day['ce_e'], day['pe_e'], day['lot']
    last = rows[-1]
    if trig_idx is None or action == 'HOLD':
        return ((ce_e - last['ce0']) + (pe_e - last['pe0'])) * lot
    t = rows[trig_idx]
    if action == 'EXIT_ALL':
        return ((ce_e - t['ce0']) + (pe_e - t['pe0'])) * lot
    if action == 'EXIT_LEG':
        if (t['ce0'] - ce_e) >= (t['pe0'] - pe_e):     # call side is the threatened one
            return ((ce_e - t['ce0']) + (pe_e - last['pe0'])) * lot
        return ((ce_e - last['ce0']) + (pe_e - t['pe0'])) * lot
    if action.startswith('STOP'):
        mult = 1.2 if action == 'STOP12' else 1.3
        for r in rows[trig_idx:]:
            if (r['ce0'] + r['pe0']) >= mult * (ce_e + pe_e):
                return ((ce_e - r['ce0']) + (pe_e - r['pe0'])) * lot
        return ((ce_e - last['ce0']) + (pe_e - last['pe0'])) * lot
    raise ValueError(action)


def stats(a):
    n = len(a)
    m = sum(a) / n
    v = sum((x - m) ** 2 for x in a) / (n - 1) if n > 1 else 0.0
    s = sorted(a)
    return dict(n=n, mean=m, t=(m / math.sqrt(v / n) if v > 0 else 0.0),
                win=100.0 * sum(1 for x in a if x > 0) / n,
                p05=s[int(0.05 * (n - 1))], med=s[n // 2], worst=s[0], best=s[-1])


def main():
    data = load()
    print('day-venue units usable: %d' % len(data), flush=True)
    TRIG = triggers()
    ACTIONS = ('EXIT_ALL', 'EXIT_LEG', 'STOP12', 'STOP13')

    # baseline
    base = {}
    for u, d in data.items():
        c = COST_PER_LEGSIDE[u[1]] * LEGSIDES
        base[u] = pnl(d, u[1], None, 'HOLD') - c

    rows_out = []
    for venue in ('NIFTY', 'SENSEX', 'BOTH'):
        us = [u for u in data if venue == 'BOTH' or u[1] == venue]
        b = [base[u] for u in us]
        st = stats(b)
        rows_out.append(dict(venue=venue, trigger='-', action='HOLD', **{k: round(v, 2) for k, v in st.items()},
                             fired_pct=100.0, diff_vs_hold=0.0, t_diff=0.0, beats_random=''))

    for tname, tfn in TRIG:
        # trigger index per unit + the random twin
        tidx, ridx = {}, {}
        for u, d in data.items():
            hit = None
            for i, r in enumerate(d['rows']):
                try:
                    if tfn(r):
                        hit = i
                        break
                except (TypeError, KeyError):
                    pass
            tidx[u] = hit
            rng = random.Random(('%s|%s|%s' % (tname, u[0], u[1])).__hash__() & 0xffffffff)
            ridx[u] = [rng.randrange(len(d['rows'])) if hit is not None else None
                       for _ in range(N_RANDOM)]

        for action in ACTIONS:
            for venue in ('NIFTY', 'SENSEX', 'BOTH'):
                us = [u for u in data if venue == 'BOTH' or u[1] == venue]
                if not us:
                    continue
                arm, diff, rnd = [], [], []
                fired = 0
                for u in us:
                    d = data[u]
                    c = COST_PER_LEGSIDE[u[1]] * LEGSIDES
                    p = pnl(d, u[1], tidx[u], action) - c
                    arm.append(p); diff.append(p - base[u])
                    if tidx[u] is not None:
                        fired += 1
                        rp = sum(pnl(d, u[1], j, action) - c for j in ridx[u]) / N_RANDOM
                    else:
                        rp = base[u]
                    rnd.append(rp - base[u])
                st = stats(arm)
                sd = stats(diff)
                sr = stats(rnd)
                rows_out.append(dict(
                    venue=venue, trigger=tname, action=action,
                    n=st['n'], mean=round(st['mean'], 1), t=round(st['t'], 2),
                    win=round(st['win'], 1), p05=round(st['p05'], 1), med=round(st['med'], 1),
                    worst=round(st['worst'], 1), best=round(st['best'], 1),
                    fired_pct=round(100.0 * fired / len(us), 1),
                    diff_vs_hold=round(sd['mean'], 1), t_diff=round(sd['t'], 2),
                    random_diff=round(sr['mean'], 1),
                    beats_random=int(sd['mean'] > sr['mean'])))
        print('  %s done' % tname, flush=True)

    head = ['venue', 'trigger', 'action', 'n', 'fired_pct', 'mean', 't', 'diff_vs_hold',
            't_diff', 'random_diff', 'beats_random', 'win', 'med', 'p05', 'worst', 'best']
    with open(os.path.join(OUT, 'g2_rule_bakeoff.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=head, extrasaction='ignore')
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    both = [r for r in rows_out if r['venue'] == 'BOTH']
    both.sort(key=lambda r: -(r['diff_vs_hold'] if r['action'] != 'HOLD' else 1e9))
    print('\n=== POOLED (both venues), net Rs per lot per day, vs HOLD ===')
    print('%-22s %-9s %8s %8s %7s %8s %6s %8s' %
          ('trigger', 'action', 'mean', 'vs HOLD', 't', 'rand', 'fire%', 'win%'))
    for r in both[:30]:
        print('%-22s %-9s %8.1f %8.1f %7.2f %8.1f %6.1f %8.1f' % (
            r['trigger'], r['action'], r['mean'], r['diff_vs_hold'], r['t_diff'],
            r.get('random_diff', 0), r['fired_pct'], r['win']))
    nb = [r for r in both if r['action'] != 'HOLD']
    better = [r for r in nb if r['diff_vs_hold'] > 0]
    sigb = [r for r in better if r['t_diff'] >= 3.9 and r['beats_random'] == 1]
    print('\narms tested: %d ; beating HOLD at all: %d ; beating HOLD with t>=3.9 AND '
          'beating their random twin: %d' % (len(nb), len(better), len(sigb)))
    for r in sigb:
        print('   SURVIVOR: %s %s  +%.1f/lot t %.2f' % (r['trigger'], r['action'],
                                                        r['diff_vs_hold'], r['t_diff']))
    json.dump(dict(arms=len(nb), better=len(better), significant=len(sigb),
                   hold_pooled=[r for r in rows_out if r['action'] == 'HOLD']),
              open(os.path.join(OUT, 'g2_summary.json'), 'w'), indent=2, default=str)


if __name__ == '__main__':
    main()
