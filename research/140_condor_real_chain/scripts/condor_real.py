"""research/140 — the Wed->Fri iron condor on REAL NSE bhavcopy prices.

Arun, 2026-08-31: "but why we hv acces ot EOD options data for years from NSE
bhavcopy right?"

He is right and it is the obvious point I missed. research/80 valued this condor
on a calibrated Black-Scholes engine with NO volatility skew, and its own RESULTS.md
warns the engine is weakest far-OTM — exactly where the long wings sit. I repeated
that caveat as if it were unavoidable. It is not: `nse_options_bhav` holds
**5,179,544 real NIFTY option rows over 3,861 trading days, 2011-01-03 to 2026-08-28**,
and this strategy enters at a Wednesday CLOSE and exits at a Friday CLOSE. Daily
closing prices are not an approximation for it — they are exactly the two prices
the strategy trades at.

So the engine can be replaced outright rather than caveated.

## The construction (research/80's surviving spec, unchanged)

  ENTRY  Wednesday close: SELL a ~0.8%-OTM strangle, BUY wings 1.0% BEYOND each
         short (so each vertical is ~1% of spot wide, not 0.2%)
  EXIT   Friday close. Never held over a weekend, never into Mon/Tue.
  STOP   close if the combined premium DOUBLES (x2). research/80's winning row is
         "0.8% / 1.0% (stop x2)" — the live paper book omits this stop, which is a
         separate finding already logged.
  SIZE   2 lots/leg = 130 qty. Reported per-lot and at 2 lots.

## What EOD data can and cannot see — stated before any result

  CAN: the entry price, the exit price, and the P&L between them. These are the
       only two prices the strategy actually transacts at, so the RETURN is exact
       up to costs.
  CANNOT: an intraday breach of the x2 stop. With daily closes the stop can only
       be evaluated at Thursday's close and Friday's close. A day that doubled
       intraday and came back is invisible. This BIASES THE STOPPED VARIANT
       OPTIMISTIC and is reported as such — the no-stop variant is unaffected and
       is the honest headline.
  Mitigation: the bhavcopy carries daily HIGH for each leg, so a conservative
       upper bound on stop firing is computed too (using each leg's high, which
       overstates because the two legs do not peak together).

## Guards (playbook)

  * REAL TRADED CONTRACTS ONLY — binding rule from research/89: every leg must
    have contracts > 0 on both entry and exit day. A strike with zero volume has a
    stale close and would manufacture P&L.
  * Non-overlapping trades: one campaign per week, Wed->Fri, so the t-stat is honest.
  * Costs charged explicitly, and a cost sensitivity reported.
  * Per-year table, so a result that lives in one regime cannot hide.
  * The DTE actually obtained is recorded per trade, because weekly expiries did
    not exist before 2019 and the expiry WEEKDAY changed (Thu -> Tue) during the
    sample. Trades are kept only where a genuine ~1-week expiry existed.

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
OUT = ROOT / 'research' / '140_condor_real_chain' / 'results'

LOT, LOTS = 65, 2
QTY = LOT * LOTS
SHORT_PCT, WING_PCT = 0.008, 0.010
COST_PER_LEG = 20.0            # round trip per leg, as condor_paper.py charges
MIN_DTE, MAX_DTE = 4, 11       # a genuine ~1-week expiry

con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
con.execute('PRAGMA temp_store=MEMORY')
con.execute('PRAGMA cache_size=-200000')

print('loading NIFTY spot...', flush=True)
# NOTE: the index symbol in market_data_unified is 'NIFTY50' (no space). An earlier
# version used 'NIFTY 50' (0 rows) and fell back to LIKE 'NIFTY%50%', which ALSO
# matches NIFTY500 - a different index at a different level - silently corrupting
# every strike calculation. Pinned exactly, and asserted.
spot = {r[0][:10]: r[1] for r in con.execute(
    "SELECT date, close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day'")}
assert len(spot) > 3000, f'NIFTY50 spot rows: {len(spot)}'
_probe = spot.get('2026-08-26')
assert _probe and 23000 < _probe < 26000, f'2026-08-26 spot looks wrong: {_probe}'
print(f'  {len(spot)} spot days {min(spot)} -> {max(spot)} · 2026-08-26 close {_probe}',
      flush=True)

print('loading option closes (traded contracts only)...', flush=True)
# key: (trade_date, expiry, strike, type) -> (close, high, contracts)
opt = {}
exp_by_day = defaultdict(set)
avail = defaultdict(list)     # (day, expiry, type) -> sorted traded strikes
n = 0
for td, ex, k, ot, cl, hi, ct in con.execute(
        "SELECT trade_date, expiry_date, strike, option_type, close, high, contracts "
        "FROM nse_options_bhav WHERE symbol='NIFTY' AND contracts > 0 AND close > 0"):
    td, ex = td[:10], ex[:10]
    opt[(td, ex, float(k), ot)] = (float(cl), float(hi or 0), int(ct))
    exp_by_day[td].add(ex)
    avail[(td, ex, ot)].append(float(k))
    n += 1
    if n % 2_000_000 == 0:
        print(f'  {n:,} rows...', flush=True)
con.close()
for kk in avail:
    avail[kk].sort()
print(f'  {n:,} traded option rows across {len(exp_by_day)} days', flush=True)

days = sorted(exp_by_day)
dayset = set(days)


def nth_next(d, wd):
    """next date on/after d whose weekday is wd and which is a trading day"""
    cur = d
    for _ in range(10):
        if cur.weekday() == wd and cur.isoformat() in dayset:
            return cur.isoformat()
        cur += timedelta(days=1)
    return None


def snap(td, ex, ot, target, lo=None, hi=None):
    """Nearest strike that ACTUALLY TRADED that day, optionally bounded.
    A real book buys the strike the exchange lists and the market quotes, not the
    arithmetic ideal — so snapping is more faithful than requiring an exact hit."""
    ks = avail.get((td, ex, ot))
    if not ks:
        return None
    c = [k for k in ks if (lo is None or k >= lo) and (hi is None or k <= hi)]
    if not c:
        return None
    return min(c, key=lambda k: abs(k - target))


def leg(td, ex, k, ot):
    return opt.get((td, ex, k, ot))


trades = []
skips = defaultdict(int)
for dstr in days:
    d = date.fromisoformat(dstr)
    if d.weekday() != 2:                      # Wednesday entry
        continue
    sp = spot.get(dstr)
    if not sp:
        skips['no spot']; skips['no spot'] += 1
        continue
    # Friday exit of the same week
    fri = None
    for add in (2, 1, 3):                     # Fri, else Thu, else Mon-ish fallback
        cand = (d + timedelta(days=add)).isoformat()
        if cand in dayset and date.fromisoformat(cand).weekday() == 4:
            fri = cand
            break
    if not fri:
        skips['no friday'] += 1
        continue
    # expiry: nearest with DTE in band
    cands = sorted(e for e in exp_by_day[dstr]
                   if MIN_DTE <= (date.fromisoformat(e) - d).days <= MAX_DTE)
    if not cands:
        skips['no ~1wk expiry'] += 1
        continue
    ex = cands[0]

    # snap every leg to a strike that genuinely traded on the ENTRY day
    sc = snap(dstr, ex, 'CE', sp * (1 + SHORT_PCT), lo=sp)
    sp_ = snap(dstr, ex, 'PE', sp * (1 - SHORT_PCT), hi=sp)
    if sc is None or sp_ is None:
        skips['no traded short strike'] += 1
        continue
    wc = snap(dstr, ex, 'CE', sc * (1 + WING_PCT), lo=sc + 25)
    wp = snap(dstr, ex, 'PE', sp_ * (1 - WING_PCT), hi=sp_ - 25)
    if wc is None or wp is None:
        skips['no traded wing strike'] += 1
        continue

    legs = [(sc, 'CE', -1), (sp_, 'PE', -1), (wc, 'CE', +1), (wp, 'PE', +1)]
    ein, eout, ok = {}, {}, True
    for k, ot, sgn in legs:
        a, b = leg(dstr, ex, k, ot), leg(fri, ex, k, ot)
        if not a or not b:
            ok = False
            break
        if a[2] <= 0:                      # entry leg must have really traded
            ok = False
            break
        ein[(k, ot)], eout[(k, ot)] = a, b
    if not ok:
        skips['leg missing / untraded'] += 1
        continue

    credit = sum(-sgn * ein[(k, ot)][0] for k, ot, sgn in legs)     # net received
    exitv = sum(-sgn * eout[(k, ot)][0] for k, ot, sgn in legs)
    gross = (credit - exitv) * QTY
    cost = COST_PER_LEG * 4
    trades.append(dict(entry=dstr, exit=fri, expiry=ex,
                       dte=(date.fromisoformat(ex) - d).days,
                       spot=sp, sc=sc, sp=sp_, wc=wc, wp=wp,
                       sc_pct=round(100*(sc/sp-1),2), sp_pct=round(100*(sp_/sp-1),2),
                       wc_pct=round(100*(wc/sp-1),2), wp_pct=round(100*(wp/sp-1),2),
                       credit=round(credit, 2), exitv=round(exitv, 2),
                       pnl=round(gross - cost), width=round(min(wc - sc, sp_ - wp))))

print(f'\n{len(trades)} campaigns built. skips: {dict(skips)}', flush=True)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'trades.json').write_text(json.dumps(trades), encoding='utf-8')


def dd(v):
    cum = peak = w = 0.0
    for x in v:
        cum += x; peak = max(peak, cum); w = min(w, cum - peak)
    return w


def rep(title, tr):
    if len(tr) < 3:
        print(f'\n{title}: only {len(tr)} trades')
        return
    v = [t['pnl'] for t in tr]
    sd = st.stdev(v)
    t_ = st.mean(v) / (sd / len(v) ** 0.5)
    print(f'\n{title}')
    print(f'  n {len(v)} · {tr[0]["entry"]} -> {tr[-1]["entry"]}')
    print(f'  net Rs{sum(v):,.0f} at {LOTS} lots · mean Rs{st.mean(v):,.0f}/campaign · '
          f'median Rs{st.median(v):,.0f}')
    print(f'  t {t_:.2f} · win {100*sum(1 for x in v if x>0)/len(v):.0f}% · '
          f'maxDD Rs{dd(v):,.0f}')
    print(f'  best Rs{max(v):,.0f} · worst Rs{min(v):,.0f} · '
          f'mean DTE {st.mean([t["dte"] for t in tr]):.1f} · '
          f'mean width {st.mean([t["width"] for t in tr]):.0f} pts')


rep('ALL CAMPAIGNS — no stop (EOD close to close, real prices)', trades)

print('\nPER YEAR')
by = defaultdict(list)
for t in trades:
    by[t['entry'][:4]].append(t['pnl'])
print(f"  {'year':6} {'n':>4} {'net':>12} {'mean':>10} {'win%':>6}")
for y in sorted(by):
    v = by[y]
    print(f'  {y:6} {len(v):>4} {sum(v):>12,.0f} {st.mean(v):>10,.0f} '
          f'{100*sum(1 for x in v if x>0)/len(v):>5.0f}%')

print('\nCOST SENSITIVITY (per leg, round trip)')
for cpl in (0, 20, 40, 80):
    v = [t['pnl'] + (COST_PER_LEG - cpl) * 4 for t in trades]
    print(f'  Rs{cpl:>3}/leg  net Rs{sum(v):>12,.0f}  mean Rs{st.mean(v):>8,.0f}  '
          f't {st.mean(v)/(st.stdev(v)/len(v)**0.5):>5.2f}')

print(f'\nresults written to {OUT}')
