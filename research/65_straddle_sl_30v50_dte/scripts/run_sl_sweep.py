#!/usr/bin/env python3
"""Short ATM NIFTY straddle: 30% vs 50% per-leg SL by DTE, on the per-minute option chain.
Cheap probe (~38 trading days, Apr-Jun 2026). Gross + net. Two exit policies. Run on VPS."""
import sqlite3, csv, os, sys
from datetime import datetime, date

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'backtest_data', 'options_data.db')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUT, exist_ok=True)
LOT = 65
ENTRY_HHMM = '09:20'        # nearest snapshot to this
EOD_HHMM = '15:15'
COST_PT = 4.2               # base net cost per straddle in premium-points (~Rs275/65)

con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row

def trading_days():
    rows = con.execute("SELECT DISTINCT date(snapshot_time) d FROM underlying_spot ORDER BY d").fetchall()
    return [r['d'] for r in rows]

def entry_snapshot(day):
    # nearest snapshot at/after ENTRY on this day (must be before 09:35 or skip). NIFTY only.
    r = con.execute("""SELECT snapshot_time, spot_price FROM underlying_spot
        WHERE symbol='NIFTY' AND date(snapshot_time)=? AND time(snapshot_time)>=? ORDER BY snapshot_time LIMIT 1""",
        (day, ENTRY_HHMM)).fetchone()
    if not r: return None
    if r['snapshot_time'][11:16] > '09:35': return None
    return r['snapshot_time'], r['spot_price']

def nearest_expiry(day):
    r = con.execute("""SELECT MIN(expiry_date) e FROM option_chain
        WHERE symbol='NIFTY' AND date(snapshot_time)=? AND expiry_date>=?""", (day, day)).fetchone()
    return r['e'] if r else None

def leg_series(day, strike, expiry, itype, t0):
    rows = con.execute("""SELECT snapshot_time st, ltp FROM option_chain
        WHERE symbol='NIFTY' AND date(snapshot_time)=? AND strike=? AND expiry_date=? AND instrument_type=?
          AND snapshot_time>=? AND time(snapshot_time)<=?
        ORDER BY snapshot_time""", (day, strike, expiry, itype, t0, EOD_HHMM)).fetchall()
    # collapse to one ltp per minute
    out = []
    seen = set()
    for r in rows:
        m = r['st'][:16]
        if m in seen or r['ltp'] is None: continue
        seen.add(m); out.append((r['st'], float(r['ltp'])))
    return out

def simulate(day):
    es = entry_snapshot(day)
    if not es: return None
    t0, spot = es
    if not spot: return None
    atm = round(spot/50)*50
    exp = nearest_expiry(day)
    if not exp: return None
    dte = (date.fromisoformat(exp) - date.fromisoformat(day)).days
    ce = leg_series(day, atm, exp, 'CE', t0)
    pe = leg_series(day, atm, exp, 'PE', t0)
    if len(ce) < 5 or len(pe) < 5: return None
    ce0, pe0 = ce[0][1], pe[0][1]
    if ce0 <= 0 or pe0 <= 0: return None
    # align by minute
    pem = {p[0][:16]: p[1] for p in pe}
    cem = {c[0][:16]: c[1] for c in ce}
    mins = sorted(set(cem) & set(pem))
    if len(mins) < 5: return None

    def run(sl, policy):
        # policy: 'both' = exit both on first breach; 'survivor' = exit breached leg, keep other to EOD
        ce_sl = ce0*(1+sl) if sl else None
        pe_sl = pe0*(1+sl) if sl else None
        ce_exit = pe_exit = None
        for m in mins[1:]:
            c, p = cem[m], pem[m]
            if policy == 'both':
                if sl and (c >= ce_sl or p >= pe_sl):
                    ce_exit, pe_exit = c, p; break
            else:  # survivor
                if sl and ce_exit is None and c >= ce_sl: ce_exit = c
                if sl and pe_exit is None and p >= pe_sl: pe_exit = p
                if ce_exit is not None and pe_exit is not None: break
        last = mins[-1]
        if ce_exit is None: ce_exit = cem[last]
        if pe_exit is None: pe_exit = pem[last]
        gross_pts = (ce0 - ce_exit) + (pe0 - pe_exit)   # short both
        return gross_pts

    res = {'day': day, 'dte': dte, 'atm': atm, 'spot': round(spot,1), 'ce0': ce0, 'pe0': pe0,
           'straddle': round(ce0+pe0,1), 'n_min': len(mins)}
    for pol in ('both', 'survivor'):
        for sl, tag in ((0.30, '30'), (0.50, '50'), (None, 'NO')):
            res['g_%s_%s' % (pol, tag)] = round(run(sl, pol)*LOT, 0)               # gross Rs
            res['n_%s_%s' % (pol, tag)] = round((run(sl, pol)-COST_PT)*LOT, 0)     # net Rs
    return res

def main():
    rows = []
    for day in trading_days():
        try:
            r = simulate(day)
        except Exception as e:
            print('  skip %s: %s' % (day, e)); continue
        if r:
            rows.append(r); print('  %s DTE=%s ATM=%s straddle=%.0f net30b=%.0f net50b=%.0f'
                                   % (r['day'], r['dte'], r['atm'], r['straddle'], r['n_both_30'], r['n_both_50']))
    if not rows: print('NO ROWS'); return
    cols = list(rows[0].keys())
    with open(os.path.join(OUT, 'sl_dte_sweep.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    # aggregate by DTE (exit-both policy, net)
    print('\n=== EXIT-BOTH, NET Rs/straddle by DTE ===')
    print('%-4s %4s | %8s %8s %8s | %6s %6s' % ('DTE','n','net30','net50','netNO','win30','win50'))
    by = {}
    for r in rows: by.setdefault(r['dte'], []).append(r)
    tot = {'30':[], '50':[], 'NO':[]}
    for dte in sorted(by):
        g = by[dte]; n = len(g)
        a = lambda k: sum(x[k] for x in g)/n
        w = lambda k: 100*sum(1 for x in g if x[k] > 0)/n
        for t in ('30','50','NO'): tot[t] += [x['n_both_'+t] for x in g]
        print('%-4s %4d | %8.0f %8.0f %8.0f | %5.0f%% %5.0f%%'
              % (dte, n, a('n_both_30'), a('n_both_50'), a('n_both_NO'), w('n_both_30'), w('n_both_50')))
    print('-'*52)
    print('%-4s %4d | %8.0f %8.0f %8.0f | %5.0f%% %5.0f%%' % ('ALL', len(rows),
          sum(tot['30'])/len(rows), sum(tot['50'])/len(rows), sum(tot['NO'])/len(rows),
          100*sum(1 for v in tot['30'] if v>0)/len(rows), 100*sum(1 for v in tot['50'] if v>0)/len(rows)))
    print('\n=== TOTALS (sum net Rs, exit-both) ===')
    for t in ('30','50','NO'): print('  SL%s: total net Rs %.0f' % (t, sum(tot[t])))
    print('\n(survivor policy is in the CSV: n_survivor_30/50/NO)')

if __name__ == '__main__':
    main()
