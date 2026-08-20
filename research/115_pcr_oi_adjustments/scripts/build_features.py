#!/usr/bin/env python3
"""
research/115_pcr_oi_adjustments -- minute-level CAUSAL feature build.

For every (trading day, venue) we build one row per minute containing
  * the PCR / OI-change / wall / max-pain / IV family of signals, each using ONLY
    information timestamped <= that minute, and
  * the forward outcomes that matter to a SHORT STRADDLE book:
      fret_h    signed forward return of the underlying
      fabs_h    |forward return|
      fdprem_h  forward change in the premium of the straddle we are actually short
                (fixed strike K0 = the 09:16 ATM strike, i.e. what our live books hold)

Written per day to results/features/<VENUE>_<DATE>.csv so the run is resumable and
crash-safe.  READ-ONLY on the database.

PRE-REGISTERED SIGNAL LIST (see RESULTS.md multiple-testing accounting) -- 22 signals:
  pcr_oi_all pcr_oi_atm5 pcr_oi_atm10 pcr_vol_all pcr_vol_atm5
  d_pcr_oi_all_15 d_pcr_oi_atm5_15
  doi_ce_5 doi_ce_15 doi_ce_30 doi_pe_5 doi_pe_15 doi_pe_30
  doi_net_15 doi_atm_15 doi_wing_15
  dist_wall_ce dist_wall_pe wall_squeeze
  mp_dev mp_drift_15 mp_drift_30
  atm_iv d_atm_iv_15 skew            (3 IV cross-checks, flagged separately)
  d_prem_15                          (NON-OI momentum control)
horizons h in {5,15,30,60} minutes.
"""
import sqlite3, os, csv, sys, math
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DB = os.path.join(ROOT, 'backtest_data', 'options_data.db')
OUT = os.path.abspath(os.path.join(HERE, '..', 'results'))
FEAT = os.path.join(OUT, 'features')
os.makedirs(FEAT, exist_ok=True)

VENUES = ('NIFTY', 'SENSEX')
HORIZONS = (5, 15, 30, 60)
ENTRY_HHMM = '09:16'
LAST_HHMM = '15:20'

SIGNALS = ['pcr_oi_all', 'pcr_oi_atm5', 'pcr_oi_atm10', 'pcr_vol_all', 'pcr_vol_atm5',
           'd_pcr_oi_all_15', 'd_pcr_oi_atm5_15',
           'doi_ce_5', 'doi_ce_15', 'doi_ce_30', 'doi_pe_5', 'doi_pe_15', 'doi_pe_30',
           'doi_net_15', 'doi_atm_15', 'doi_wing_15',
           'dist_wall_ce', 'dist_wall_pe', 'wall_squeeze',
           'mp_dev', 'mp_drift_15', 'mp_drift_30',
           'atm_iv', 'd_atm_iv_15', 'skew', 'd_prem_15']

FIELDS = (['date', 'venue', 'minute', 'dte', 'spot', 'atm', 'k0', 'prem', 'prem_entry',
           'ce0', 'pe0', 'lot'] + SIGNALS +
          ['fret_%d' % h for h in HORIZONS] +
          ['fabs_%d' % h for h in HORIZONS] +
          ['fdprem_%d' % h for h in HORIZONS])


def conn():
    c = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
    c.execute('PRAGMA query_only=ON')
    return c


def day_list(c):
    days = defaultdict(set)
    for sym, d in c.execute(
            "select symbol, substr(snapshot_time,1,10) d from download_log "
            "where symbol in ('NIFTY','SENSEX') group by symbol, d"):
        days[sym].add(d)
    return days


def dnum(s):
    return int(s[:4]) * 372 + int(s[5:7]) * 31 + int(s[8:10])


def maxpain(strikes, ce_oi, pe_oi):
    """Strike minimising total writer payout -- classic max pain."""
    best, bestv = None, None
    for k in strikes:
        # payout to option BUYERS if spot settles at k
        tot = 0.0
        for j in strikes:
            if k > j:
                tot += ce_oi.get(j, 0) * (k - j)
            if k < j:
                tot += pe_oi.get(j, 0) * (j - k)
        if bestv is None or tot < bestv:
            bestv, best = tot, k
    return best


def build_day(c, sym, d, w):
    lo, hi = d + 'T00:00:00', d + 'T23:59:59'
    rows = list(c.execute(
        "select snapshot_time, expiry_date, strike, instrument_type, ltp, oi, volume, iv, "
        "underlying_spot, lot_size from option_chain "
        "where symbol=? and snapshot_time>=? and snapshot_time<=? order by snapshot_time",
        (sym, lo, hi)))
    if not rows:
        return 0
    exps = sorted({r[1] for r in rows})
    near = exps[0]
    dte = dnum(near) - dnum(d)
    rows = [r for r in rows if r[1] == near]
    if not rows:
        return 0

    # minute -> strike -> {'CE':..,'PE':..}
    by_min = defaultdict(lambda: defaultdict(dict))
    spot_of = {}
    lot = None
    for st, ex, k, t, ltp, oi, vol, iv, sp, ls in rows:
        m = st[11:16]
        by_min[m][k][t] = (ltp, oi, vol, iv)
        if sp:
            spot_of[m] = sp
        if ls:
            lot = ls
    minutes = sorted(m for m in by_min if spot_of.get(m))
    minutes = [m for m in minutes if ENTRY_HHMM <= m <= LAST_HHMM]
    if len(minutes) < 120:
        return 0

    # fixed strike universe = strikes present (both legs) in EVERY minute -> no composition drift
    common = None
    for m in minutes:
        s = {k for k, v in by_min[m].items() if 'CE' in v and 'PE' in v}
        common = s if common is None else (common & s)
    common = sorted(common or [])
    if len(common) < 11:
        return 0
    step = min(common[i + 1] - common[i] for i in range(len(common) - 1))

    # entry strike K0 = ATM at 09:16 (what the live books short)
    m0 = minutes[0]
    k0 = min(common, key=lambda k: abs(k - spot_of[m0]))
    prem_entry = None

    # ---- pass 1: per-minute state ------------------------------------------------
    state = {}
    for m in minutes:
        sp = spot_of[m]
        ch = by_min[m]
        ce_oi, pe_oi, ce_vol, pe_vol = {}, {}, {}, {}
        for k in common:
            ce_oi[k] = ch[k]['CE'][1] or 0
            pe_oi[k] = ch[k]['PE'][1] or 0
            ce_vol[k] = ch[k]['CE'][2] or 0
            pe_vol[k] = ch[k]['PE'][2] or 0
        atm = min(common, key=lambda k: abs(k - sp))
        ai = common.index(atm)

        def win(n):
            return common[max(0, ai - n): ai + n + 1]

        w5, w10 = win(5), win(10)
        wall_ce = max(common, key=lambda k: ce_oi[k])
        wall_pe = max(common, key=lambda k: pe_oi[k])
        atm_iv_vals = [x for x in (ch[atm]['CE'][3], ch[atm]['PE'][3]) if x]
        lo_i = common[max(0, ai - 3)]
        hi_i = common[min(len(common) - 1, ai + 3)]
        skew_p = ch[lo_i]['PE'][3]
        skew_c = ch[hi_i]['CE'][3]
        prem = None
        if ch[k0]['CE'][0] is not None and ch[k0]['PE'][0] is not None:
            prem = ch[k0]['CE'][0] + ch[k0]['PE'][0]
        state[m] = dict(
            spot=sp, atm=atm, prem=prem,
            ce0=ch[k0]['CE'][0], pe0=ch[k0]['PE'][0],
            sum_ce_oi=sum(ce_oi.values()), sum_pe_oi=sum(pe_oi.values()),
            sum_ce_oi5=sum(ce_oi[k] for k in w5), sum_pe_oi5=sum(pe_oi[k] for k in w5),
            sum_ce_oi10=sum(ce_oi[k] for k in w10), sum_pe_oi10=sum(pe_oi[k] for k in w10),
            sum_ce_vol=sum(ce_vol.values()), sum_pe_vol=sum(pe_vol.values()),
            sum_ce_vol5=sum(ce_vol[k] for k in w5), sum_pe_vol5=sum(pe_vol[k] for k in w5),
            oi_atm=sum(ce_oi[k] + pe_oi[k] for k in win(2)),
            oi_wing=sum(ce_oi[k] + pe_oi[k] for k in common) - sum(ce_oi[k] + pe_oi[k] for k in win(5)),
            wall_ce=wall_ce, wall_pe=wall_pe,
            mp=maxpain(common, ce_oi, pe_oi),
            atm_iv=(sum(atm_iv_vals) / len(atm_iv_vals)) if atm_iv_vals else None,
            skew=(skew_p - skew_c) if (skew_p and skew_c) else None,
        )
    prem_entry = state[minutes[0]]['prem']
    if not prem_entry:
        for m in minutes[:10]:
            if state[m]['prem']:
                prem_entry = state[m]['prem']
                break
    if not prem_entry:
        return 0

    idx = {m: i for i, m in enumerate(minutes)}
    n = 0

    def lag(i, h):
        j = i - h
        return state[minutes[j]] if j >= 0 else None

    for i, m in enumerate(minutes):
        s = state[m]
        if s['prem'] is None:
            continue
        r = dict(date=d, venue=sym, minute=m, dte=dte, spot=s['spot'], atm=s['atm'],
                 k0=k0, prem=s['prem'], prem_entry=prem_entry, ce0=s['ce0'], pe0=s['pe0'], lot=lot)
        sp = s['spot']

        def sd(a, b):
            return (a / b) if b else None

        r['pcr_oi_all'] = sd(s['sum_pe_oi'], s['sum_ce_oi'])
        r['pcr_oi_atm5'] = sd(s['sum_pe_oi5'], s['sum_ce_oi5'])
        r['pcr_oi_atm10'] = sd(s['sum_pe_oi10'], s['sum_ce_oi10'])
        r['pcr_vol_all'] = sd(s['sum_pe_vol'], s['sum_ce_vol'])
        r['pcr_vol_atm5'] = sd(s['sum_pe_vol5'], s['sum_ce_vol5'])
        l15, l5, l30 = lag(i, 15), lag(i, 5), lag(i, 30)
        r['d_pcr_oi_all_15'] = (r['pcr_oi_all'] - sd(l15['sum_pe_oi'], l15['sum_ce_oi'])) \
            if (l15 and r['pcr_oi_all'] is not None and sd(l15['sum_pe_oi'], l15['sum_ce_oi']) is not None) else None
        r['d_pcr_oi_atm5_15'] = (r['pcr_oi_atm5'] - sd(l15['sum_pe_oi5'], l15['sum_ce_oi5'])) \
            if (l15 and r['pcr_oi_atm5'] is not None and sd(l15['sum_pe_oi5'], l15['sum_ce_oi5']) is not None) else None
        # dOI normalised by the lagged level -> unit-free, comparable across venues/days
        for h, L in ((5, l5), (15, l15), (30, l30)):
            r['doi_ce_%d' % h] = ((s['sum_ce_oi'] - L['sum_ce_oi']) / L['sum_ce_oi']) if (L and L['sum_ce_oi']) else None
            r['doi_pe_%d' % h] = ((s['sum_pe_oi'] - L['sum_pe_oi']) / L['sum_pe_oi']) if (L and L['sum_pe_oi']) else None
        r['doi_net_15'] = (r['doi_pe_15'] - r['doi_ce_15']) if (r['doi_pe_15'] is not None and r['doi_ce_15'] is not None) else None
        r['doi_atm_15'] = ((s['oi_atm'] - l15['oi_atm']) / l15['oi_atm']) if (l15 and l15['oi_atm']) else None
        r['doi_wing_15'] = ((s['oi_wing'] - l15['oi_wing']) / l15['oi_wing']) if (l15 and l15['oi_wing']) else None
        r['dist_wall_ce'] = (s['wall_ce'] - sp) / sp
        r['dist_wall_pe'] = (s['wall_pe'] - sp) / sp
        r['wall_squeeze'] = (s['wall_ce'] - s['wall_pe']) / sp
        r['mp_dev'] = (s['mp'] - sp) / sp
        r['mp_drift_15'] = ((s['mp'] - l15['mp']) / sp) if l15 else None
        r['mp_drift_30'] = ((s['mp'] - l30['mp']) / sp) if l30 else None
        r['atm_iv'] = s['atm_iv']
        r['d_atm_iv_15'] = (s['atm_iv'] - l15['atm_iv']) if (l15 and s['atm_iv'] and l15['atm_iv']) else None
        r['skew'] = s['skew']
        r['d_prem_15'] = (s['prem'] - l15['prem']) if (l15 and l15['prem']) else None

        for h in HORIZONS:
            j = i + h
            if j < len(minutes):
                f = state[minutes[j]]
                r['fret_%d' % h] = f['spot'] / sp - 1.0
                r['fabs_%d' % h] = abs(f['spot'] / sp - 1.0)
                r['fdprem_%d' % h] = (f['prem'] - s['prem']) if f['prem'] is not None else None
            else:
                r['fret_%d' % h] = r['fabs_%d' % h] = r['fdprem_%d' % h] = None
        w.writerow(r)
        n += 1
    return n


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    c = conn()
    days = day_list(c)
    master = open(os.path.join(OUT, 'features_master.csv'), 'w', newline='')
    mw = csv.DictWriter(master, fieldnames=FIELDS, extrasaction='ignore')
    mw.writeheader()
    tot = 0
    for sym in VENUES:
        if only and sym != only:
            continue
        for d in sorted(days.get(sym, [])):
            n = build_day(c, sym, d, mw)
            tot += n
            master.flush()
            print('%s %s -> %d rows (total %d)' % (sym, d, n, tot), flush=True)
    master.close()
    print('DONE rows=%d' % tot)


if __name__ == '__main__':
    main()
