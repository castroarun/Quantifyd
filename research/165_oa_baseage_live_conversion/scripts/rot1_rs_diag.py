# -*- coding: utf-8 -*-
"""research/165 - WHICH rs252 does the live book compute? A diagnostic, not a gate.

research/170's engine reads rs252 off research/164's PANEL: every symbol's raw close
forward-filled onto one master trading calendar, so "252 bars ago" means 252 calendar
sessions. The live scanner has no panel. Two live-implementable readings exist:

  OWN   252 of the symbol's OWN daily bars
  CAL   the same forward-fill onto a master calendar derived from the database, 252 sessions

They differ for any name that missed a session in the last year. This measures both against
the study's frozen `rs252` column in `events164.csv` - which is the number the published
result was computed from - and says which one the live code should use.

Read-only. Usage:
    python3 research/165_oa_baseage_live_conversion/scripts/rot1_rs_diag.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
# panel164.pkl pickles a `bt_core.Panel`, so bt_core must be importable by that bare name
sys.path.insert(0, str(ROOT / 'research/170_qs_leeway_and_baseage_best_entrant/scripts'))
from services import oa_baseage as spec                                    # noqa: E402
from services import oa_baseage_entry as live                              # noqa: E402

PANEL_PKL = ROOT / 'research/164_baseage_slots_sizing/results/panel164.pkl'
EV164 = ROOT / 'research/164_baseage_slots_sizing/results/events164.csv'
LOOK = 252


def db_calendar(con):
    return [r[0][:10] for r in con.execute(
        "SELECT DISTINCT substr(date,1,10) FROM market_data_unified WHERE timeframe='day' "
        "ORDER BY 1")]


def main():
    panel = pickle.load(open(PANEL_PKL, 'rb'))
    con = spec.connect()
    cal_db = db_calendar(con)
    cal_p = list(panel.cal)
    print('panel calendar : %d sessions, %s .. %s' % (len(cal_p), cal_p[0], cal_p[-1]))
    print('DB   calendar  : %d sessions, %s .. %s' % (len(cal_db), cal_db[0], cal_db[-1]))
    lo, hi = cal_p[0], cal_p[-1]
    cal_db_w = [d for d in cal_db if lo <= d <= hi]
    print('DB calendar clipped to the panel window: %d sessions; identical to the panel: %s'
          % (len(cal_db_w), cal_db_w == cal_p))
    if cal_db_w != cal_p:
        a, b = set(cal_db_w), set(cal_p)
        print('  in DB not in panel: %d  e.g. %s' % (len(a - b), sorted(a - b)[:5]))
        print('  in panel not in DB: %d  e.g. %s' % (len(b - a), sorted(b - a)[:5]))

    ev = pd.read_csv(EV164, dtype={'trigger_date': str})
    ev = ev[ev.rs252 > -1e8]           # the study's own "no 12-month history" sentinel
    print('\nevents with a study rs252: %d of the frozen list' % len(ev))

    pos_db = {d: i for i, d in enumerate(cal_db_w)}
    rows = []
    for sym, g in ev.groupby('symbol'):
        d = live._raw_closes(con, sym)
        dates = sorted({str(x)[:10] for (x,) in con.execute(
            "SELECT date FROM market_data_unified WHERE symbol=? AND timeframe='day' "
            "AND volume>0 AND close>0", (sym,))})
        if len(d) != len(dates):
            continue
        own_pos = {dt: i for i, dt in enumerate(dates)}
        # CAL: forward-fill onto the master calendar, exactly as bt_core.Panel does
        arr = np.full(len(cal_db_w), np.nan)
        for dt, c in zip(dates, d):
            k = pos_db.get(dt)
            if k is not None:
                arr[k] = c
        arr = pd.Series(arr).ffill().to_numpy()
        for _, r in g.iterrows():
            t = r['trigger_date']
            own = np.nan
            k = own_pos.get(t)
            if k is not None and k >= LOOK and d[k - LOOK] > 0:
                own = 100.0 * (d[k] / d[k - LOOK] - 1.0)
            cal = np.nan
            k2 = pos_db.get(t)
            if k2 is not None and k2 >= LOOK and np.isfinite(arr[k2 - LOOK]) \
                    and arr[k2 - LOOK] > 0 and np.isfinite(arr[k2]):
                cal = 100.0 * (arr[k2] / arr[k2 - LOOK] - 1.0)
            lv = live.rs252(con, sym, t)
            rows.append((sym, t, float(r['rs252']), own, cal,
                         np.nan if lv is None else lv))
    con.close()

    df = pd.DataFrame(rows, columns=['symbol', 'trigger_date', 'study', 'own', 'cal', 'live'])
    for k in ('own', 'cal', 'live'):
        d = (df[k] - df['study']).abs()
        ok = (d < 0.005) | (~np.isfinite(df[k]) & ~np.isfinite(df['study']))
        print('\n%s vs the study rs252 over %d events' % (k.upper(), len(df)))
        print('  exact (within 0.005pp)  : %6d  (%.2f%%)' % (ok.sum(), 100.0 * ok.mean()))
        print('  missing where study has : %6d' % int((~np.isfinite(df[k])).sum()))
        print('  median |difference|     : %.4f pp' % float(np.nanmedian(d)))
        print('  p95    |difference|     : %.4f pp' % float(np.nanpercentile(d.dropna(), 95)))

    # what actually matters: does the ranking of a contested day change?
    print('\nRANKING - the only thing the rule reads')
    for key in ('own', 'cal', 'live'):
        same = tot = 0
        for _, g in df.groupby('trigger_date'):
            if len(g) < 2:
                continue
            tot += 1
            a = g.sort_values(['study', 'symbol'], ascending=[False, True]).iloc[0]['symbol']
            b = g.sort_values([key, 'symbol'], ascending=[False, True]).iloc[0]['symbol']
            same += (a == b)
        print('  %s: top-ranked name identical on %d of %d multi-signal days (%.2f%%)'
              % (key.upper(), same, tot, 100.0 * same / max(tot, 1)))
    df.to_csv(HERE.parent / 'results' / 'rot1_rs_diag.csv', index=False)


if __name__ == '__main__':
    main()
