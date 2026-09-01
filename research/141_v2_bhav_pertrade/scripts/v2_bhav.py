"""research/141 — V2 iron fly rebuilt on real bhavcopy, with per-trade output.

Arun's reading of the CPR numbers:
    AlgoTest (with stop+PT):  +Rs8.1L  -> +Rs11.0L
    ours (EOD, no stop/PT):   +Rs6.93L -> +Rs13.79L
    "algotest SL is based on probably 1min close data, and ours is EOD, waiting
     until almost EOD and acting significantly increase the profits"

That is a real and testable hypothesis: if the unmanaged book ends HIGHER than the
managed one on the same filtered subset, the intraday stop is costing money. But
the published pair cannot settle it, because the two rows differ in TWO ways at
once - engine AND management. This isolates management on ONE engine.

It also exists because neither engine stored per-trade data. research/60's results
hold PNGs and a JSON; there are no AlgoTest trade CSVs in the repo. So win/loss
STREAKS and a year-by-year HEATMAP are not computable from what we have - they have
to be recomputed from the chain, which is what this does. For the AlgoTest side they
remain unavailable until the platform CSVs are exported.

## Construction (the live engine's spec, from services/v2_ironfly_api.py)

  ENTRY   when flat: the 2nd-nearest NIFTY weekly, requiring DTE >= 4 calendar days.
          ATM = nearest 50 to the entry-day OPEN (causal - the look-ahead audit's
          correction; picking ATM from the close tripled results and hid drawdown).
  LEGS    SELL ATM CE + ATM PE, BUY wings at +/-2.0% of ATM = short iron fly.
  GATE    India VIX >= 13.0 at entry.
  EXIT    the last trading day before expiry, at the CLOSE ("roll at DTE<=1").
  COSTS   Rs20/order x 4 legs + 0.25% slippage on premium (the live CFG basis).
  SIZE    10 lots = qty 650.

## The four arms

  A  hold to DTE-1                          <- the existing reproduction
  B  + 2% underlying move-stop
  C  A + skip when prior-day CPR < 0.10%
  D  B + skip when prior-day CPR < 0.10%

A vs B isolates the STOP on one engine. C vs D does the same on the filtered subset.
That is the comparison Arun's hypothesis needs and the published table cannot give.

## What EOD can and cannot see - stated before any result

The 2% stop is on the UNDERLYING, and NIFTY daily OHLC is available, so the DAY a
2% excursion happened is known exactly. What is NOT known is the price at the
moment of breach - this exits at that day's CLOSE instead. That is neither
systematically optimistic nor pessimistic, but it is an approximation and every
arm-B/D number carries it. The 40% profit target is NOT modelled at all: it is a
premium trigger that can fire and reverse within a day, and daily closes cannot
see it. So arms B and D are "stop only", not the full live rule set.

**No stopless configuration is being recommended.** Arm A exists as the control
that isolates what the stop does, exactly as the nostop column did in research/139.

Read-only against market_data.db.
"""
from __future__ import annotations

import json, sqlite3, statistics as st, sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DB = ROOT / 'backtest_data' / 'market_data.db'
OUT = ROOT / 'research' / '141_v2_bhav_pertrade' / 'results'
OUT.mkdir(parents=True, exist_ok=True)

LOT, LOTS = 65, 10
QTY = LOT * LOTS                 # 650
WING_PCT = 0.020
VIX_FLOOR = 13.0
MIN_DTE = 4
BROK_PER_LEG = 20.0
SLIP_PCT = 0.0025
CPR_MIN = 0.10                   # skip when prior-day CPR width % < this

con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
con.execute('PRAGMA temp_store=MEMORY')
con.execute('PRAGMA cache_size=-400000')

print('loading NIFTY daily OHLC...', flush=True)
nifty = {}
for d, o, h, l, cl in con.execute(
        "SELECT date, open, high, low, close FROM market_data_unified "
        "WHERE symbol='NIFTY50' AND timeframe='day' ORDER BY date"):
    if cl:
        nifty[d[:10]] = (float(o or cl), float(h or cl), float(l or cl), float(cl))
assert len(nifty) > 3000, len(nifty)
print(f'  {len(nifty)} days {min(nifty)} -> {max(nifty)}', flush=True)

print('loading India VIX...', flush=True)
vix = {r[0][:10]: float(r[1]) for r in con.execute(
    "SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day'")
    if r[1]}
print(f'  {len(vix)} vix days', flush=True)

print('loading NIFTY option bhavcopy (traded contracts only)...', flush=True)
opt, avail, exp_by_day = {}, defaultdict(list), defaultdict(set)
n = 0
for td, ex, k, ot, op, cl, ct in con.execute(
        "SELECT trade_date, expiry_date, strike, option_type, open, close, contracts "
        "FROM nse_options_bhav WHERE symbol='NIFTY' AND contracts > 0 AND close > 0"):
    td, ex = td[:10], ex[:10]
    opt[(td, ex, float(k), ot)] = (float(op or cl), float(cl))
    avail[(td, ex, ot)].append(float(k))
    exp_by_day[td].add(ex)
    n += 1
    if n % 2_000_000 == 0:
        print(f'  {n:,}...', flush=True)
con.close()
for kk in avail:
    avail[kk].sort()
print(f'  {n:,} traded rows / {len(exp_by_day)} days', flush=True)

days = sorted(d for d in exp_by_day if d in nifty)
dayset = set(days)


def cpr_width_pct(prev):
    o, h, l, c = nifty[prev]
    piv = (h + l + c) / 3.0
    bc = (h + l) / 2.0
    tc = 2 * piv - bc
    return abs(tc - bc) / c * 100.0


def snap(td, ex, ot, target, lo=None, hi=None):
    ks = avail.get((td, ex, ot))
    if not ks:
        return None
    cnd = [k for k in ks if (lo is None or k >= lo) and (hi is None or k <= hi)]
    return min(cnd, key=lambda k: abs(k - target)) if cnd else None


def second_weekly(dstr):
    """Expiry choice is the documented ambiguity - both readings are swept."""
    import os as _os
    exps = sorted(exp_by_day[dstr])
    fut = [e for e in exps if e > dstr]
    if not fut:
        return None
    if _os.environ.get('NEAREST') == '1':
        return fut[0]
    return fut[1] if len(fut) >= 2 else None


def run(use_stop, use_cpr):
    trades, i = [], 0
    while i < len(days):
        dstr = days[i]
        if dstr < START:
            i += 1; continue
        if vix.get(dstr, 0) < VIX_FLOOR:
            i += 1; continue
        if use_cpr:
            prev = days[i - 1] if i else None
            if not prev or cpr_width_pct(prev) < CPR_MIN:
                i += 1; continue
        ex = second_weekly(dstr)
        if not ex:
            i += 1; continue
        if ENTRY_TDTE:
            tdte = sum(1 for d in days if dstr < d < ex) + 1
            if tdte != ENTRY_TDTE:
                i += 1; continue
        elif (date.fromisoformat(ex) - date.fromisoformat(dstr)).days < MIN_DTE:
            i += 1; continue

        op = nifty[dstr][0]                       # causal: ATM from the OPEN
        atm = snap(dstr, ex, 'CE', op)
        atmp = snap(dstr, ex, 'PE', op)
        if atm is None or atmp is None or atm != atmp:
            atm = atmp = snap(dstr, ex, 'CE', op)
            if atm is None or (dstr, ex, atm, 'PE') not in opt:
                i += 1; continue
        wc = snap(dstr, ex, 'CE', atm * (1 + WING_PCT), lo=atm + 50)
        wp = snap(dstr, ex, 'PE', atm * (1 - WING_PCT), hi=atm - 50)
        if wc is None or wp is None:
            i += 1; continue
        legs = [(atm, 'CE', -1), (atm, 'PE', -1), (wc, 'CE', +1), (wp, 'PE', +1)]
        if any((dstr, ex, k, ot) not in opt for k, ot, _ in legs):
            i += 1; continue

        # exit day = last trading day strictly before expiry
        hold = [d for d in days if dstr < d < ex]
        if not hold:
            i += 1; continue
        exit_day, stopped = hold[-1], False
        if use_stop:
            es = nifty[dstr][3]
            for d in hold:
                _, h, l, _ = nifty[d]
                if abs(h - es) / es >= 0.02 or abs(l - es) / es >= 0.02:
                    exit_day, stopped = d, True
                    break
        if any((exit_day, ex, k, ot) not in opt for k, ot, _ in legs):
            nxt = [d for d in hold if d <= exit_day
                   and all((d, ex, k, ot) in opt for k, ot, _ in legs)]
            if not nxt:
                i += 1; continue
            exit_day = nxt[-1]

        credit = sum(-sg * opt[(dstr, ex, k, ot)][0] for k, ot, sg in legs)
        exitv = sum(-sg * opt[(exit_day, ex, k, ot)][1] for k, ot, sg in legs)
        gross = (credit - exitv) * QTY
        slip = SLIP_PCT * (abs(credit) + abs(exitv)) * QTY
        pnl = round(gross - BROK_PER_LEG * 4 - slip)

        # per-leg detail. SELL: (entry-exit); BUY: (exit-entry) -> -sg*(entry-exit)
        legdet, legsum = [], 0.0
        for k, ot, sg in legs:
            en = opt[(dstr, ex, k, ot)][0]
            xt = opt[(exit_day, ex, k, ot)][1]
            lp = -sg * (en - xt) * QTY
            legsum += lp
            legdet.append(dict(side='SELL' if sg < 0 else 'BUY', type=ot, strike=k,
                               qty=QTY, entry=round(en, 2), exit=round(xt, 2),
                               pnl=round(lp)))
        assert abs(legsum - gross) < 1.0, (legsum, gross)   # legs must reconstruct the trade

        trades.append(dict(entry=dstr, exit=exit_day, expiry=ex, atm=atm, wc=wc, wp=wp,
                           vix=round(vix.get(dstr, 0), 2), credit=round(credit, 2),
                           exitv=round(exitv, 2), stopped=stopped, pnl=pnl,
                           gross=round(gross), costs=round(BROK_PER_LEG * 4 + slip),
                           spot_entry=round(nifty[dstr][3], 2),
                           spot_exit=round(nifty[exit_day][3], 2),
                           year=dstr[:4], month=dstr[:7], legs=legdet))
        i = days.index(exit_day) + 1
    return trades


def streaks(v):
    bw = bl = cw = cl_ = 0
    for x in v:
        if x > 0:
            cw += 1; cl_ = 0
        else:
            cl_ += 1; cw = 0
        bw, bl = max(bw, cw), max(bl, cl_)
    return bw, bl


def dd(v):
    cum = peak = w = 0.0
    for x in v:
        cum += x; peak = max(peak, cum); w = min(w, cum - peak)
    return w


import os
NEAREST = os.environ.get('NEAREST') == '1'
ENTRY_TDTE = int(os.environ.get('ENTRY_TDTE', '0'))   # 0 = any day with DTE>=4
START = os.environ.get('START', '2011-01-01')
print(f'\nvariant: NEAREST={NEAREST} ENTRY_TDTE={ENTRY_TDTE} START={START}')

ARMS = [('A  hold to DTE-1 (control)', False, False),
        ('B  + 2% move-stop', True, False),
        ('C  A + CPR>=0.10% filter', False, True),
        ('D  B + CPR>=0.10% filter', True, True)]

allres = {}
print(f'\n{"arm":28} {"n":>4} {"net":>12} {"mean":>9} {"win%":>6} {"maxDD":>12} '
      f'{"Calmar":>7} {"Wstk":>5} {"Lstk":>5} {"stops":>6}')
print('-' * 108)
for name, us, uc in ARMS:
    tr = run(us, uc)
    v = [t['pnl'] for t in tr]
    if not v:
        print(f'{name:28} no trades'); continue
    yrs = (date.fromisoformat(tr[-1]['entry']) - date.fromisoformat(tr[0]['entry'])).days / 365.25
    cal = (sum(v) / yrs) / abs(dd(v)) if dd(v) else float('nan')
    bw, bl = streaks(v)
    print(f'{name:28} {len(v):>4} {sum(v):>12,.0f} {sum(v)/len(v):>9,.0f} '
          f'{100*sum(1 for x in v if x>0)/len(v):>5.0f}% {dd(v):>12,.0f} {cal:>7.2f} '
          f'{bw:>5} {bl:>5} {sum(1 for t in tr if t["stopped"]):>6}')
    allres[name] = tr

print('\nPER-MONTH NET (month-of-year x year, the monthly heatmap source)')
for name, tr in allres.items():
    by = defaultdict(float)
    for t in tr:
        by[t['month']] += t['pnl']
    print(f'  {name}: {len(by)} months')

print('\nPER-YEAR NET (the heatmap source)')
yrs = sorted({t['year'] for tr in allres.values() for t in tr})
print(f'  {"arm":28}' + ''.join(f'{y:>11}' for y in yrs))
for name, tr in allres.items():
    by = defaultdict(float)
    for t in tr:
        by[t['year']] += t['pnl']
    print(f'  {name:28}' + ''.join(f'{by.get(y, 0):>11,.0f}' for y in yrs))

(OUT / 'arms.json').write_text(json.dumps(
    {k: v for k, v in allres.items()}, indent=1), encoding='utf-8')
print(f'\nwrote {OUT}/arms.json')
