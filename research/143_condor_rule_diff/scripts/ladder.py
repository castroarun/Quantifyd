#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/143 - WHY the condor flipped sign: the engine, or the rules?

Arun: "if our earlier backtest says very positive calmar 1.63, win so and so, how
come now we are saying loss making? ensure if both rules are same first."

He is right to insist, and the answer is that they are NOT the same rules. I reported
research/140 as though it re-tested research/80's condor on real prices and found the
engine wrong by Rs1,073 a campaign. It did not. Four things changed at once and I
attributed the whole gap to one of them.

  1 ANCHOR   r/80 enters when expiry is exactly 6 CALENDAR days out and exits when it
             is <=4. NIFTY expiry was THURSDAY until Sep-2025, so for most of the
             sample that is FRIDAY -> MONDAY, carrying a WEEKEND, held DTE6 -> DTE3.
             r/140 enters WEDNESDAY and exits FRIDAY, never over a weekend, taking the
             nearest expiry in a DTE 4-11 band (mean 7.7) -- so in the Thursday era it
             holds DTE8 -> DTE6. A different part of the decay curve entirely.
  2 STOP     r/80's headline closes if the combined premium DOUBLES, checked on
             5-MINUTE bars. r/140 ran NO stop, matching what the paper book does.
  3 INSTRUMENT  r/80 prices a synthetic weekly expiry every Thursday from 2015. NIFTY
             WEEKLY options did not exist until Feb-2019 -- our own bhavcopy shows 14
             distinct expiries a year through 2018, then 56 in 2019. Four of its
             "11/12 positive years" traded a contract that was never listed.
  4 PRICING  Black-Scholes off India VIX with a per-DTE IV multiplier and NO SKEW,
             against real traded closes.

This script holds everything constant but one difference at a time, so each can be
priced separately instead of blamed collectively.

  A   BS   · r80 anchor · stop x2 · 2015-26   <- the published +880/campaign
  B   BS   · r80 anchor · no stop · 2015-26   A vs B  = what the STOP is worth
  B19 BS   · r80 anchor · no stop · 2019-26   B vs B19 = the pre-weekly FICTION
  C   REAL · r80 anchor · no stop · 2019-26   B19 vs C = the ENGINE
  D   REAL · Wed->Fri   · no stop · 2019-26   C vs D  = the ANCHOR
  E   REAL · Wed->Fri   · no stop · 2011-26   <- research/140 as published

All arms: 0.8% shorts, wings 1.0% beyond each short, 2 lots (130 qty), 4 legs.
"""
import json, math, sqlite3, statistics as st, sys
from collections import defaultdict
from datetime import date, datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = '/home/arun/quantifyd'
DB = ROOT + '/backtest_data/market_data.db'
QTY, BROK, SLIP = 130, 20, 0.01
SHORT_PCT, WING_PCT = 0.008, 0.010
MIN_DTE, MAX_DTE = 4, 11

CAL = json.load(open(ROOT + '/research/80_farDTE_rescue/results/engine_calib.json'))['iv_mult_by_dte']


def ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs(S, K, T, iv, kind):
    if T <= 0 or iv <= 0:
        return max(0.0, (S - K) if kind == 'CE' else (K - S))
    d1 = (math.log(S / K) + 0.5 * iv * iv * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    return S * ncdf(d1) - K * ncdf(d2) if kind == 'CE' else K * ncdf(-d2) - S * ncdf(-d1)


def ivm(dd_):
    d = int(round(dd_))
    if str(d) in CAL:
        return CAL[str(d)]
    ks = sorted(int(k) for k in CAL)
    return CAL[str(min(ks, key=lambda k: abs(k - d)))]


def synth_expiry(d):
    """r/80's assumption: a weekly expiry every Thursday (Tuesday after Sep-2025)."""
    t = 1 if d >= date(2025, 9, 1) else 3
    a = (t - d.weekday()) % 7
    return d if a == 0 else d + timedelta(days=a)


print('loading...', flush=True)
con = sqlite3.connect('file:' + DB + '?mode=ro', uri=True)
con.execute('PRAGMA cache_size=-200000')
spot_d = {r[0][:10]: r[1] for r in con.execute(
    "SELECT date, close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day'")}
assert len(spot_d) > 3000, len(spot_d)
vix_d = {r[0][:10]: r[1] for r in con.execute(
    "SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day'")}
bars5 = defaultdict(list)
for ts, cl in con.execute("SELECT date, close FROM market_data_unified "
                          "WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date"):
    bars5[str(ts)[:10]].append((str(ts), cl))
vix5 = {str(t): v for t, v in con.execute(
    "SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='5minute'")}

opt, exp_by_day, avail = {}, defaultdict(set), defaultdict(list)
for td, ex, k, ot, cl in con.execute(
        "SELECT trade_date, expiry_date, strike, option_type, close FROM nse_options_bhav "
        "WHERE symbol='NIFTY' AND contracts > 0 AND close > 0"):
    td, ex = td[:10], ex[:10]
    opt[(td, ex, float(k), ot)] = float(cl)
    exp_by_day[td].add(ex)
    avail[(td, ex, ot)].append(float(k))
con.close()
for kk in avail:
    avail[kk].sort()
odayset = set(exp_by_day)
print('  spot %d | bhav days %d | 5min days %d' % (len(spot_d), len(odayset), len(bars5)), flush=True)


def snap(td, ex, ot, target, lo=None, hi=None):
    ks = avail.get((td, ex, ot))
    if not ks:
        return None
    cnd = [k for k in ks if (lo is None or k >= lo) and (hi is None or k <= hi)]
    return min(cnd, key=lambda k: abs(k - target)) if cnd else None


def campaigns_r80(dayset, expiries_for):
    """entry when an expiry is exactly 6 calendar days out; exit when it is <= 4"""
    out = []
    ds = sorted(dayset)
    idx = {d: i for i, d in enumerate(ds)}
    for dstr in ds:
        d = date.fromisoformat(dstr)
        exps = [e for e in expiries_for(dstr) if (date.fromisoformat(e) - d).days == 6]
        if not exps:
            continue
        ex = exps[0]
        ed = date.fromisoformat(ex)
        xd = None
        for s in ds[idx[dstr] + 1:]:
            if (ed - date.fromisoformat(s)).days <= 4:
                xd = s
                break
        if xd:
            out.append((dstr, xd, ex))
    return out


def campaigns_wedfri(dayset, expiries_for):
    out = []
    for dstr in sorted(dayset):
        d = date.fromisoformat(dstr)
        if d.weekday() != 2:
            continue
        fri = None
        for add in (2, 1, 3):
            cand = (d + timedelta(days=add)).isoformat()
            if cand in dayset and date.fromisoformat(cand).weekday() == 4:
                fri = cand
                break
        if not fri:
            continue
        cands = sorted(e for e in expiries_for(dstr)
                       if MIN_DTE <= (date.fromisoformat(e) - d).days <= MAX_DTE)
        if not cands:
            continue
        out.append((dstr, fri, cands[0]))
    return out


def run_real(anchor, lo_year, hi_year='2026'):
    tr = []
    for dstr, xd, ex in anchor(odayset, lambda d: exp_by_day[d]):
        if not (lo_year <= dstr[:4] <= hi_year):
            continue
        sp0 = spot_d.get(dstr)
        if not sp0:
            continue
        sc = snap(dstr, ex, 'CE', sp0 * (1 + SHORT_PCT), lo=sp0)
        spk = snap(dstr, ex, 'PE', sp0 * (1 - SHORT_PCT), hi=sp0)
        if sc is None or spk is None:
            continue
        wc = snap(dstr, ex, 'CE', sc * (1 + WING_PCT), lo=sc + 25)
        wp = snap(dstr, ex, 'PE', spk * (1 - WING_PCT), hi=spk - 25)
        if wc is None or wp is None:
            continue
        legs = [(sc, 'CE', -1), (spk, 'PE', -1), (wc, 'CE', 1), (wp, 'PE', 1)]
        cin = cout = 0.0
        ok = True
        for k, ot, sgn in legs:
            a, b = opt.get((dstr, ex, k, ot)), opt.get((xd, ex, k, ot))
            if a is None or b is None:
                ok = False
                break
            cin += -sgn * a
            cout += -sgn * b
        if not ok:
            continue
        tr.append(dict(year=dstr[:4], entry=dstr, exit=xd, expiry=ex,
                       dte=(date.fromisoformat(ex) - date.fromisoformat(dstr)).days,
                       pnl=(cin - cout) * QTY - 4 * BROK))
    return tr


def run_bs(anchor, lo_year, stop_mult, hi_year='2026'):
    """r/80's engine: BS off VIX, synthetic weekly expiry, 5-min stop check."""
    d5 = set(bars5)
    ds = sorted(d5)
    idx = {d: i for i, d in enumerate(ds)}
    tr = []
    for dstr, xd, ex in anchor(d5, lambda s: [synth_expiry(date.fromisoformat(s)).isoformat()]):
        if not (lo_year <= dstr[:4] <= hi_year):
            continue
        e = date.fromisoformat(ex)
        ts0, S0 = bars5[dstr][-1]
        v0 = vix5.get(ts0, vix_d.get(dstr))
        if not v0:
            continue

        def T_yr(ts):
            return max((datetime(e.year, e.month, e.day, 15, 30)
                        - datetime.fromisoformat(ts)).total_seconds() / (365 * 24 * 3600), 1e-6)

        T0 = T_yr(ts0)
        iv0 = (v0 / 100) * ivm(T0 * 365)
        Kc = round(S0 * (1 + SHORT_PCT) / 50) * 50
        Kp = round(S0 * (1 - SHORT_PCT) / 50) * 50
        Kcw = round(S0 * (1 + SHORT_PCT + WING_PCT) / 50) * 50
        Kpw = round(S0 * (1 - SHORT_PCT - WING_PCT) / 50) * 50
        cred = (bs(S0, Kc, T0, iv0, 'CE') + bs(S0, Kp, T0, iv0, 'PE')
                - bs(S0, Kcw, T0, iv0, 'CE') - bs(S0, Kpw, T0, iv0, 'PE'))
        if cred < 2:
            continue
        val = cred
        stopped = False
        for s in ds[idx[dstr] + 1: idx[xd] + 1]:
            for ts, Sx in bars5[s]:
                vv = vix5.get(ts, vix_d.get(s, v0))
                Tx = T_yr(ts)
                ivx = (vv / 100) * ivm(Tx * 365)
                val = (bs(Sx, Kc, Tx, ivx, 'CE') + bs(Sx, Kp, Tx, ivx, 'PE')
                       - bs(Sx, Kcw, Tx, ivx, 'CE') - bs(Sx, Kpw, Tx, ivx, 'PE'))
                if stop_mult and val >= cred * stop_mult:
                    stopped = True
                    break
            if stopped:
                break
        tr.append(dict(year=dstr[:4], entry=dstr, exit=xd, expiry=ex, stopped=stopped,
                       dte=(e - date.fromisoformat(dstr)).days,
                       pnl=(cred * (1 - SLIP) - val * (1 + SLIP)) * QTY - 4 * BROK))
    return tr


def dd_(v):
    cum = peak = worst = 0.0
    for x in v:
        cum += x
        peak = max(peak, cum)
        worst = min(worst, cum - peak)
    return worst


def rep(lab, tr, note=''):
    v = [t['pnl'] for t in tr]
    if len(v) < 3:
        print('%-38s n=%d (too few)' % (lab, len(v)))
        return
    m, sd = st.mean(v), st.stdev(v)
    se = sd / math.sqrt(len(v))
    ys = sorted({t['year'] for t in tr})
    ann = sum(v) / len(ys)
    mdd = dd_(v)
    posy = sum(1 for y in ys if sum(t['pnl'] for t in tr if t['year'] == y) > 0)
    stp = [t['stopped'] for t in tr if 'stopped' in t]
    stxt = (str(round(100 * sum(stp) / len(stp))) + '%') if stp else '-'
    print('%-38s %4d %+8.0f %+9.0f %4.0f%% %+10.0f %6.2f %+6.2f %2d/%-2d %5s  %s'
          % (lab, len(v), m, ann, 100 * sum(1 for x in v if x > 0) / len(v), mdd,
             ann / abs(mdd) if mdd else 0, m / se, posy, len(ys), stxt, note))


print('')
print('=' * 120)
print('  DECOMPOSITION - one difference at a time.  2 lots (130 qty), 0.8% shorts, 1.0% wings')
print('=' * 120)
print('%-38s %4s %8s %9s %5s %10s %6s %6s %5s %5s'
      % ('arm', 'n', 'mean', 'ann', 'win', 'maxDD', 'Calmar', 't', '+yrs', 'stop'))
print('-' * 120)

A = run_bs(campaigns_r80, '2015', 2.0)
rep('A   BS   r80-anchor  stop x2  2015-26', A, '<- the published headline')
B = run_bs(campaigns_r80, '2015', None)
rep('B   BS   r80-anchor  NO stop  2015-26', B, '<- A vs B = the STOP')
B19 = run_bs(campaigns_r80, '2019', None)
rep('B19 BS   r80-anchor  NO stop  2019-26', B19, '<- B vs B19 = pre-weekly FICTION')
C = run_real(campaigns_r80, '2019')
rep('C   REAL r80-anchor  NO stop  2019-26', C, '<- B19 vs C = the ENGINE')
D = run_real(campaigns_wedfri, '2019')
rep('D   REAL Wed->Fri    NO stop  2019-26', D, '<- C vs D = the ANCHOR')
E = run_real(campaigns_wedfri, '2011')
rep('E   REAL Wed->Fri    NO stop  2011-26', E, '<- research/140 as published')
print('-' * 120)
print('')

WD = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for lab, arm in (('B  r80 anchor (BS)', B), ('C  r80 anchor (real)', C), ('D  Wed->Fri (real)', D)):
    wd = defaultdict(int)
    for t in arm:
        wd[WD[date.fromisoformat(t['entry']).weekday()]] += 1
    hold = st.mean([(date.fromisoformat(t['exit']) - date.fromisoformat(t['entry'])).days
                    for t in arm]) if arm else 0
    print('  %-22s entry %s | mean DTE %.1f | mean hold %.1f cal days'
          % (lab, dict(wd), st.mean([t['dte'] for t in arm]), hold))

json.dump(dict(A=A, B=B, B19=B19, C=C, D=D, E=E),
          open(ROOT + '/research/143_condor_rule_diff/results/ladder.json', 'w'))
print('')
print('wrote results/ladder.json')
