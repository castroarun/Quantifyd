"""SENSEX ATM straddle — build the recorded-chain study NIFTY already has.

research/139. The NIFTY analysis in research/138 was only possible because
static/app/options_study.json holds a pre-aggregated 5-min ATM straddle premium
path per recorded day. There is no SENSEX equivalent, so this builds one from
the same source (options_data.db, symbol='SENSEX', 92 days from 2026-04-20).

Mechanic held identical to the NIFTY study so the two are comparable:
  sell the ATM straddle at the first snapshot at/after 09:16, ATM = spot rounded
  to the nearest 100 (SENSEX strike step), nearest expiry. Track the COMBINED
  premium (CE+PE) every 5 minutes to 15:20.

Two SENSEX-specific things that must not be copied from NIFTY:
  * strike step is 100, not 50
  * expiry is THURSDAY, not Tuesday, so the weekday->DTE map is different:
    Mon=DTE3, Tue=DTE2, Wed=DTE1, Thu=DTE0, Fri=DTE4
  * lot is 20, not 65. A "10 lot" SENSEX position is qty 200 and is NOT the same
    notional as 10 NIFTY lots (qty 650) — the per-DTE comparison INSIDE SENSEX is
    exact, the cross-venue one is indicative only.

DTE is computed in TRADING days (the NIFTY study's convention), not calendar, so
weekends do not push Wed/Thu off the 0-4 scale.

Writes static/app/sensex_options_study.json in the same shape as the NIFTY file.
Read-only against the database.
"""
from __future__ import annotations

import json, sqlite3, sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DB = ROOT / 'backtest_data' / 'options_data.db'
OUT = ROOT / 'static' / 'app' / 'sensex_options_study.json'
STEP, LOT = 100, 20
WD = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
con.execute('PRAGMA temp_store=MEMORY')

days = [r[0] for r in con.execute(
    "SELECT DISTINCT substr(snapshot_time,1,10) FROM option_chain "
    "WHERE symbol='SENSEX' ORDER BY 1")]
print(f'{len(days)} recorded SENSEX days: {days[0]} -> {days[-1]}', flush=True)


def trading_dte(day: str, expiry: str) -> int:
    d, e = date.fromisoformat(day), date.fromisoformat(expiry[:10])
    n, cur = 0, d + timedelta(days=1)
    while cur <= e:
        if cur.weekday() < 5:
            n += 1
        cur += timedelta(days=1)
    return n


out_days = []
for day in days:
    rows = con.execute(
        "SELECT snapshot_time, tradingsymbol, strike, instrument_type, ltp, expiry_date, "
        "underlying_spot FROM option_chain WHERE symbol='SENSEX' "
        "AND substr(snapshot_time,1,10)=? AND ltp IS NOT NULL AND ltp>0", (day,)).fetchall()
    if not rows:
        continue
    # nearest expiry on or after the day
    exps = sorted({r[5][:10] for r in rows if r[5] and r[5][:10] >= day})
    if not exps:
        continue
    exp = exps[0]
    rows = [r for r in rows if r[5] and r[5][:10] == exp]
    if not rows:
        continue

    # entry snapshot: first at/after 09:16
    times = sorted({r[0] for r in rows})
    ent_t = next((t for t in times if t[11:16] >= '09:16'), None)
    if not ent_t:
        continue
    spot0 = next((r[6] for r in rows if r[0] == ent_t and r[6]), None)
    if not spot0:
        continue
    atm = int(round(float(spot0) / STEP) * STEP)

    # per-(time) combined premium for the ATM pair
    ce = {r[0]: r[4] for r in rows if int(r[2]) == atm and r[3] == 'CE'}
    pe = {r[0]: r[4] for r in rows if int(r[2]) == atm and r[3] == 'PE'}
    if ent_t not in ce or ent_t not in pe:
        continue

    series, seen = [], set()
    for t in times:
        if t[11:16] < '09:16' or t[11:16] > '15:30':
            continue
        if t not in ce or t not in pe:
            continue
        hm = t[11:16]
        bucket = hm[:4] + ('0' if hm[4] < '5' else '5')     # 5-min buckets
        if bucket in seen:
            continue
        seen.add(bucket)
        series.append([hm, round(float(ce[t]) + float(pe[t]), 2)])
    if len(series) < 5:
        continue

    ent = series[0][1]
    out_days.append({
        'date': day, 'weekday': WD[date.fromisoformat(day).weekday()],
        'dte': trading_dte(day, exp), 'expiry': exp, 'atm': atm,
        'entry': ent, 'close': series[-1][1],
        'high': max(p for _, p in series), 'low': min(p for _, p in series),
        'spot_open': round(float(spot0)), 'series': series,
    })
    print(f'  {day} {WD[date.fromisoformat(day).weekday()]} DTE{out_days[-1]["dte"]} '
          f'ATM {atm} entry {ent} bars {len(series)}', flush=True)

con.close()
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({'generated_at': datetime.now().isoformat()[:16],
                           'underlying': 'SENSEX', 'lot': LOT, 'step': STEP,
                           'n_days': len(out_days), 'days': out_days}), encoding='utf-8')
print(f'\nwrote {OUT} — {len(out_days)} days')
by = defaultdict(int)
for d in out_days:
    by[(d['dte'], d['weekday'])] += 1
for k in sorted(by):
    print(f'  DTE{k[0]} {k[1]}: {by[k]} days')
