"""Forward paper books for the front-weekly iron fly — arms C and D, gated and ungated.

Arun: "running both, so that i can see how both does in different market conditions"
and "only reconstruct the trades that are to be open now from its entry date and
point in time".

So this does NOT backfill 92 days of synthetic history. It works out which cycle
would be OPEN right now, reconstructs that one position from its real entry
timestamp on the recorded chain, and marks it to the latest quote. From here it
extends forward each run.

## Why four books, not two

Arms C and D differ only in the stop. Both also carry a VIX >= 13 entry gate, and
India VIX is currently ~11 — so run faithfully, BOTH WOULD BE DORMANT and there
would be nothing to compare. Rather than quietly drop the gate to make the page
look busy, each arm runs twice:

    FLY_D_GATED    CPR skip + 2% move-stop + 40% target, VIX >= 13   (faithful)
    FLY_C_GATED    CPR skip + NO stop,                    VIX >= 13   (faithful control)
    FLY_D_OPEN     same as D, no VIX gate                             (trades now)
    FLY_C_OPEN     same as C, no VIX gate                             (trades now)

That is the comparison Arun asked for and one more besides: what the VIX gate is
worth, measured forward instead of assumed.

## The construction (research/141's front-weekly build)

  entry   09:20, 4 trading days before the FRONT weekly expiry
  legs    SELL ATM CE + ATM PE, BUY wings at +/-2.0% of ATM, snapped to traded strikes
  exit    1 trading day before expiry, or the stop, or the target
  size    10 lots = qty 650; Rs20/leg + 0.25% slippage

The stopless arms are CONTROLS carried forward to measure what the stop does. No
stopless book is proposed for real money.

Writes backtest_data/fly_paper_state.json. Read-only against options_data.db.
"""
from __future__ import annotations

import json, sqlite3, sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
OPT = ROOT / 'backtest_data' / 'options_data.db'
MKT = ROOT / 'backtest_data' / 'market_data.db'
STATE = ROOT / 'backtest_data' / 'fly_paper_state.json'

LOT, LOTS = 65, 10
QTY = LOT * LOTS
WING_PCT, ENTRY_TDTE, EXIT_TDTE = 0.020, 4, 1
ENTRY_AT, EXIT_AT = '09:20', '15:15'
BROK, SLIP = 20.0, 0.0025
VIX_FLOOR, CPR_MIN = 13.0, 0.10

BOOKS = {
    'FLY_D_GATED': dict(name='Fly D · CPR + stop (VIX-gated)', cpr=True, stop=0.02, vix=True),
    'FLY_C_GATED': dict(name='Fly C · CPR, no stop (VIX-gated)', cpr=True, stop=None, vix=True),
    'FLY_D_OPEN': dict(name='Fly D · CPR + stop (ungated)', cpr=True, stop=0.02, vix=False),
    'FLY_C_OPEN': dict(name='Fly C · CPR, no stop (ungated)', cpr=True, stop=None, vix=False),
}
PT = 0.40

oc = sqlite3.connect(f'file:{OPT}?mode=ro', uri=True)
mk = sqlite3.connect(f'file:{MKT}?mode=ro', uri=True)


def trading_days():
    return [r[0] for r in oc.execute(
        "SELECT DISTINCT substr(snapshot_time,1,10) FROM option_chain "
        "WHERE symbol='NIFTY' ORDER BY 1")]


DAYS = trading_days()
TODAY = DAYS[-1]


def front_expiry(day):
    r = oc.execute("SELECT MIN(expiry_date) FROM option_chain WHERE symbol='NIFTY' "
                   "AND substr(snapshot_time,1,10)=? AND expiry_date>?", (day, day)).fetchone()
    return r[0][:10] if r and r[0] else None


def snap_quote(day, hhmm, expiry, strike, typ):
    """First quote at/after hhmm for that leg. None if the strike never traded."""
    r = oc.execute(
        "SELECT ltp FROM option_chain WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? "
        "AND substr(snapshot_time,12,5)>=? AND expiry_date LIKE ?||'%' AND strike=? "
        "AND instrument_type=? AND ltp>0 ORDER BY snapshot_time LIMIT 1",
        (day, hhmm, expiry, strike, typ)).fetchone()
    return float(r[0]) if r else None


_QCACHE = {}


def last_quote(day, expiry, strike, typ):
    """Cached: the same leg must mark at ONE instant across all books, or C and D
    appear to differ for reasons that are nothing to do with their rules."""
    ck = (day, expiry, strike, typ)
    if ck in _QCACHE:
        return _QCACHE[ck]
    r = oc.execute(
        "SELECT ltp, snapshot_time FROM option_chain WHERE symbol='NIFTY' "
        "AND substr(snapshot_time,1,10)=? AND expiry_date LIKE ?||'%' AND strike=? "
        "AND instrument_type=? AND ltp>0 ORDER BY snapshot_time DESC LIMIT 1",
        (day, expiry, strike, typ)).fetchone()
    _QCACHE[ck] = (float(r[0]), r[1]) if r else (None, None)
    return _QCACHE[ck]


def traded_strikes(day, expiry, typ):
    return sorted(float(r[0]) for r in oc.execute(
        "SELECT DISTINCT strike FROM option_chain WHERE symbol='NIFTY' "
        "AND substr(snapshot_time,1,10)=? AND expiry_date LIKE ?||'%' AND instrument_type=? "
        "AND ltp>0", (day, expiry, typ)))


def spot_at(day, hhmm):
    r = oc.execute("SELECT underlying_spot FROM option_chain WHERE symbol='NIFTY' "
                   "AND substr(snapshot_time,1,10)=? AND substr(snapshot_time,12,5)>=? "
                   "AND underlying_spot IS NOT NULL ORDER BY snapshot_time LIMIT 1",
                   (day, hhmm)).fetchone()
    return float(r[0]) if r else None


def spot_now():
    r = oc.execute("SELECT underlying_spot, snapshot_time FROM option_chain WHERE symbol='NIFTY' "
                   "AND underlying_spot IS NOT NULL ORDER BY snapshot_time DESC LIMIT 1").fetchone()
    return (float(r[0]), r[1]) if r else (None, None)


def vix_on(day):
    """Latest India VIX close at or before `day`, with its own date so staleness shows."""
    r = mk.execute("SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' "
                   "AND timeframe='day' AND date<=? ORDER BY date DESC LIMIT 1",
                   (day + 'T23:59',)).fetchone()
    return (float(r[1]), r[0][:10]) if r else (None, None)


def cpr_ok(day):
    """Prior-day CPR width as a % of today's open. True = wide enough to trade."""
    rows = [r for r in mk.execute(
        "SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' "
        "AND timeframe='day' AND date<=? ORDER BY date DESC LIMIT 2", (day + 'T23:59',))]
    if len(rows) < 2:
        return None, None
    cur, prev = rows[0], rows[1]
    _, o, h, l, c = prev
    piv = (h + l + c) / 3.0
    bc = (h + l) / 2.0
    tc = 2 * piv - bc
    w = abs(tc - bc) / cur[1] * 100.0
    return (w >= CPR_MIN), round(w, 3)


# ---------------------------------------------------------------- the open cycle
exp = front_expiry(TODAY)
# Sessions before expiry must be counted from the EXPIRY back, on the calendar —
# the recorded set only holds days that have already happened, so counting back
# from today lands on the wrong dates whenever expiry is still ahead.
# (Exchange holidays are not modelled; they are rare and shift entry by one day.)
def _sessions_before(exp_iso, n):
    d, out = date.fromisoformat(exp_iso), []
    while len(out) < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            out.append(d.isoformat())
    return out            # out[0] = 1 TD before, out[n-1] = n TD before


_back = _sessions_before(exp, max(ENTRY_TDTE, EXIT_TDTE))
entry_day = _back[ENTRY_TDTE - 1]
exit_day = _back[EXIT_TDTE - 1]
open_now = bool(entry_day and entry_day <= TODAY < (exit_day or ''))

print(f'front weekly expiry {exp}')
print(f'  entry day  {entry_day}   ({ENTRY_TDTE} TD before)')
print(f'  exit  day  {exit_day}   ({EXIT_TDTE} TD before)')
print(f'  today      {TODAY}  ->  a cycle should be OPEN: {open_now}\n')

vix, vix_as = vix_on(entry_day or TODAY)
ok_cpr, cprw = cpr_ok(entry_day or TODAY)
print(f'gates at entry: VIX {vix} (as of {vix_as})  ·  prior-day CPR {cprw}%\n')

sp_now, sp_at = spot_now()
out = {'generated_at': datetime.now().isoformat()[:19], 'day': TODAY, 'expiry': exp,
       'entry_day': entry_day, 'exit_day': exit_day, 'spot_now': sp_now, 'spot_at': sp_at,
       'vix': vix, 'vix_asof': vix_as, 'cpr_pct': cprw, 'books': {}}

for key, b in BOOKS.items():
    rec = {'name': b['name'], 'cpr': b['cpr'], 'stop': b['stop'], 'vix_gate': b['vix']}
    if not open_now or not entry_day:
        rec.update(state='FLAT', reason='no cycle open right now')
        out['books'][key] = rec
        continue
    if b['vix'] and (vix is None or vix < VIX_FLOOR):
        rec.update(state='GATED', reason=f'VIX {vix} < {VIX_FLOOR} floor at entry')
        out['books'][key] = rec
        continue
    if b['cpr'] and ok_cpr is False:
        rec.update(state='GATED', reason=f'prior-day CPR {cprw}% < {CPR_MIN}%')
        out['books'][key] = rec
        continue

    sp = spot_at(entry_day, ENTRY_AT)
    if not sp:
        rec.update(state='NO DATA', reason='no spot at entry')
        out['books'][key] = rec
        continue

    ce_ks, pe_ks = traded_strikes(entry_day, exp, 'CE'), traded_strikes(entry_day, exp, 'PE')
    if not ce_ks or not pe_ks:
        rec.update(state='NO DATA', reason='no traded strikes at entry')
        out['books'][key] = rec
        continue
    atm = min(ce_ks, key=lambda k: abs(k - sp))
    wc = min([k for k in ce_ks if k > atm] or [atm], key=lambda k: abs(k - atm * (1 + WING_PCT)))
    wp = min([k for k in pe_ks if k < atm] or [atm], key=lambda k: abs(k - atm * (1 - WING_PCT)))

    legs, ok = [], True
    for strike, typ, side in ((atm, 'CE', 'SELL'), (atm, 'PE', 'SELL'),
                              (wc, 'CE', 'BUY'), (wp, 'PE', 'BUY')):
        en = snap_quote(entry_day, ENTRY_AT, exp, strike, typ)
        lt, at = last_quote(TODAY, exp, strike, typ)
        if en is None or lt is None:
            ok = False
            break
        sgn = 1 if side == 'SELL' else -1
        legs.append(dict(side=side, type=typ, strike=strike, qty=QTY, entry=round(en, 2),
                         ltp=round(lt, 2), at=at, pnl=round(sgn * (en - lt) * QTY)))
    if not ok:
        rec.update(state='NO DATA', reason='a leg never traded')
        out['books'][key] = rec
        continue

    credit = sum((1 if l['side'] == 'SELL' else -1) * l['entry'] for l in legs)
    mark = sum((1 if l['side'] == 'SELL' else -1) * l['ltp'] for l in legs)
    gross = (credit - mark) * QTY
    costs = BROK * 4 + SLIP * (abs(credit) + abs(mark)) * QTY
    pnl = round(gross - costs)

    # would the stop or target already have fired?
    hit, why = None, None
    if b['stop'] is not None and sp_now:
        if abs(sp_now - sp) / sp >= b['stop']:
            hit, why = 'STOP', f"underlying moved {100*abs(sp_now-sp)/sp:.2f}% >= {b['stop']:.0%}"
    if hit is None and credit and (credit - mark) / credit >= PT:
        hit, why = 'TARGET', f'captured {100*(credit-mark)/credit:.0f}% of credit'

    rec.update(state=('CLOSED (' + hit + ')') if hit else 'OPEN',
               reason=why or f'held since {entry_day} 09:20',
               entry_day=entry_day, entry_spot=round(sp, 2), atm=atm, wings=[wp, wc],
               credit=round(credit, 2), mark=round(mark, 2), pnl=pnl,
               move_pct=(round(100 * (sp_now - sp) / sp, 2) if sp_now else None),
               legs=legs)
    out['books'][key] = rec

STATE.write_text(json.dumps(out, indent=1), encoding='utf-8')
oc.close(); mk.close()

print(f"{'book':34} {'state':18} {'credit':>8} {'mark':>8} {'P&L':>10}  why")
print('-' * 108)
for k, r in out['books'].items():
    print(f"{r['name']:34} {r.get('state',''):18} {str(r.get('credit','—')):>8} "
          f"{str(r.get('mark','—')):>8} {str(r.get('pnl','—')):>10}  {r.get('reason','')}")
print(f'\nwrote {STATE}')
