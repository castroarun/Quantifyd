#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NWV weekly-view structure bake-off — jade lizard vs iron condor, REAL option EOD
================================================================================
Replays the live NWV engine over 2020-02 -> 2026-07 Mondays; for each view bucket
(BEAR / BULL / NEUTRAL) builds candidate weekly credit structures on the NEXT-week
expiry, prices them from real NSE bhav EOD (`nse_options_bhav`), applies EOD exit
policies (pessimistic gap fills), charges per-leg costs, requires real traded
volume on every leg (r/89 binding rule).

Structures include the user's live 2026-07-27 construction (pivot-anchored
put-heavy asymmetric condor / jade lizard) and the June-study fixed-offset ICs.
"""
import sys, os, csv, sqlite3, statistics
from datetime import date, datetime, timedelta
from collections import defaultdict, Counter

sys.path.insert(0, '/home/arun/quantifyd')
sys.path.insert(0, '/home/arun/quantifyd/research')
from nwv_phase1_regime import (
    build_view, daily_close, mondays, week_sessions, expiries_after, opt_eod,
    seg_hlc, round_to_strike,
    VIEW_BEARISH, VIEW_NTB, VIEW_BULLISH, VIEW_NTBULL,
)
from services.nwv_engine import VIEW_NEUTRAL, compute_pivots

LOT, LOTS = 65, 10
COST_PT_PER_LEG = 0.5          # per leg, per side (entry + exit)
START, END = date(2020, 2, 3), date(2026, 7, 13)
OUT = '/home/arun/quantifyd/research/94_nwv_jade_lizard_ic/results'
os.makedirs(OUT, exist_ok=True)

DB = '/home/arun/quantifyd/backtest_data/market_data.db'
con2 = sqlite3.connect(DB)
c2 = con2.cursor()

def traded(td, exp, k, ot):
    r = c2.execute("SELECT contracts FROM nse_options_bhav WHERE symbol='NIFTY' "
                   "AND trade_date=? AND expiry_date=? AND strike=? AND option_type=?",
                   (td, exp, float(k), ot)).fetchone()
    return bool(r and r[0] and r[0] > 0)

def r50d(x): return int(x // 50) * 50
def r50u(x): return int(-(-x // 50)) * 50
def r100(x): return round_to_strike(x, 100)

def structures(spot, piv):
    s1, r1, pp = piv['s1'], piv['r1'], piv['pp']
    r2 = piv.get('r2') or pp + (piv['_h'] - piv['_l'])
    s2 = piv.get('s2') or pp - (piv['_h'] - piv['_l'])
    # pivot anchors with degenerate-week guards (Monday gaps beyond the pivot)
    sp  = min(r50d(s1), r50d(spot - 100))            # short put below spot, ~S1
    scw = max(r50d(r2), r50u(spot + 200))            # wide short call, ~R2
    sct = max(r50u(r1), r50u(spot + 100))            # tight short call, ~R1
    spf = min(r50d(s2), r50d(spot - 200))            # far short put, ~S2 (bear mirror)
    S = {
        # fixed-offset family (June study, 100-strikes)
        'ic_neutral': [(-1, r100(spot-250), 'PE'), (1, r100(spot-250)-200, 'PE'),
                       (-1, r100(spot+250), 'CE'), (1, r100(spot+250)+200, 'CE')],
        'ic_lock-50': [(-1, r100(spot-300), 'PE'), (1, r100(spot-300)-200, 'PE'),
                       (-1, r100(spot+200), 'CE'), (1, r100(spot+200)+200, 'CE')],
        'ic_bull':    [(-1, r100(spot-100), 'PE'), (1, r100(spot-100)-200, 'PE'),
                       (-1, r100(spot+400), 'CE'), (1, r100(spot+400)+200, 'CE')],
        'jl_fix250':  [(-1, r100(spot-250), 'PE'),
                       (-1, r100(spot+250), 'CE'), (1, r100(spot+250)+200, 'CE')],
        # pivot-anchored family (user's 2026-07-27 construction, 50-strikes)
        'pv_userJL':  [(-1, sp, 'PE'), (1, sp-550, 'PE'),
                       (-1, scw, 'CE'), (1, scw+200, 'CE')],
        'pv_trueJL':  [(-1, sp, 'PE'),
                       (-1, scw, 'CE'), (1, scw+200, 'CE')],
        'pv_condorR1':[(-1, sp, 'PE'), (1, sp-550, 'PE'),
                       (-1, sct, 'CE'), (1, sct+200, 'CE')],
        'pv_bearMir': [(-1, sct, 'CE'), (1, sct+550, 'CE'),
                       (-1, spf, 'PE'), (1, spf-200, 'PE')],
    }
    return S

EXITS = ('pt50_stop1x', 'pt50_stop2x', 'hold')

def summ(p):
    p = [x for x in p if x is not None]
    if not p:
        return None
    n = len(p); w = [x for x in p if x > 0]
    gw = sum(x for x in p if x > 0); gl = -sum(x for x in p if x < 0)
    sp = sorted(p)
    return dict(n=n, winpct=100*len(w)/n, avg=sum(p)/n, tot=sum(p),
                pf=(gw/gl) if gl else 999.0, worst=min(p),
                p5=sp[max(0, int(0.05*n)-1)] if n >= 5 else min(p))

def main():
    res = defaultdict(list)                     # (bucket, struct, exit) -> [pnl]
    by_year = defaultdict(list)                 # (bucket, struct, exit, year) -> [pnl]
    vcount = Counter(); skip = Counter()
    jl_cred = []                                # (credit, call_wing) for pv_trueJL

    for mon in mondays(START, END):
        v = build_view(mon)
        if not v:
            continue
        vcount[v['view']] += 1
        if v['view'] == VIEW_BEARISH or v['view'] == VIEW_NTB:
            bucket = 'BEAR'
        elif v['view'] == VIEW_BULLISH or v['view'] == VIEW_NTBULL:
            bucket = 'BULL'
        elif v['view'] == VIEW_NEUTRAL:
            bucket = 'NEU'
        else:
            skip['view_ignore'] += 1
            continue
        ms, spot = v['ms'], v['spot']
        year = ms[:4]
        sess = week_sessions(ms)
        if len(sess) < 2:
            skip['short_week'] += 1
            continue
        # full pivot set from prior-week H/L/C
        pw = seg_hlc((mon - timedelta(days=7)).isoformat(),
                     (mon - timedelta(days=3)).isoformat())
        if not pw:
            skip['no_prior_week'] += 1
            continue
        ph, pl, pc = pw
        piv = dict(compute_pivots(ph, pl, pc))
        piv['_h'], piv['_l'] = ph, pl
        # next-week expiry: first expiry >= 6 days out
        exps = [e for e in expiries_after(ms)
                if (datetime.strptime(e, '%Y-%m-%d').date() - mon).days >= 6]
        if not exps:
            skip['no_expiry'] += 1
            continue
        expiry = exps[0]

        def leg_px(d, k, ot):
            return opt_eod(d, expiry, k, ot)

        def legs_val(d, legs):
            tot = 0.0
            for q, k, ot in legs:
                px = leg_px(d, k, ot)
                if px is None:
                    return None
                tot += q * px
            return tot

        # ---- incumbent debit spread (BEAR/BULL only) ----
        if bucket in ('BEAR', 'BULL'):
            ot = 'PE' if bucket == 'BEAR' else 'CE'
            L = r100(spot); M = L - 200 if bucket == 'BEAR' else L + 200
            dl = [(1, L, ot), (-1, M, ot)]
            if all(traded(ms, expiry, k, o) for _, k, o in dl):
                entry = legs_val(ms, dl)
                if entry is not None and entry > 0:
                    debit = entry
                    max_gain = 200 - debit
                    cost = 2 * len(dl) * COST_PT_PER_LEG
                    pt_done = None; last = None
                    for d in sess:
                        val = legs_val(d, dl)
                        if val is None:
                            continue
                        if pt_done is None and (val - debit) >= 0.60 * max_gain:
                            pt_done = (val - debit - cost) * LOT * LOTS
                        last = val
                    if last is not None:
                        hold_pnl = (last - debit - cost) * LOT * LOTS
                        res[(bucket, 'debit', 'hold')].append(hold_pnl)
                        by_year[(bucket, 'debit', 'hold', year)].append(hold_pnl)
                        pt_pnl = pt_done if pt_done is not None else hold_pnl
                        res[(bucket, 'debit', 'pt60')].append(pt_pnl)
                        by_year[(bucket, 'debit', 'pt60', year)].append(pt_pnl)

        # ---- credit structures ----
        for name, legs in structures(spot, piv).items():
            if not all(traded(ms, expiry, k, o) for _, k, o in legs):
                skip[f'untraded_{name}'] += 1
                continue
            entry_lv = legs_val(ms, legs)
            if entry_lv is None:
                skip[f'noprice_{name}'] += 1
                continue
            credit = -entry_lv
            if credit <= 0:
                skip[f'nocredit_{name}'] += 1
                continue
            if name == 'pv_trueJL':
                jl_cred.append(credit)
            cost = 2 * len(legs) * COST_PT_PER_LEG
            cap = credit * LOT * LOTS
            fills = {}                        # exit policy -> pnl (gross pts MTM)
            last_lv = None
            for d in sess:
                lv = legs_val(d, legs)
                if lv is None:
                    continue
                last_lv = lv
                mtm = (lv - entry_lv) * LOT * LOTS
                if 'pt50_stop1x' not in fills and (mtm >= 0.50*cap or mtm <= -1.0*cap):
                    fills['pt50_stop1x'] = mtm
                if 'pt50_stop2x' not in fills and (mtm >= 0.50*cap or mtm <= -2.0*cap):
                    fills['pt50_stop2x'] = mtm
            if last_lv is None:
                skip[f'nomtm_{name}'] += 1
                continue
            fri = (last_lv - entry_lv) * LOT * LOTS
            for ex in EXITS:
                pnl = fills.get(ex, fri) if ex != 'hold' else fri
                pnl -= cost * LOT * LOTS
                res[(bucket, name, ex)].append(pnl)
                by_year[(bucket, name, ex, year)].append(pnl)

    # ---- report ----
    print(f"view distribution ({sum(vcount.values())} replayed Mondays):")
    for vv, n in vcount.most_common():
        print(f"  {vv:20s} {n:4d}")
    print("\nskips:", dict(skip))
    if jl_cred:
        med = statistics.median(jl_cred)
        cov = 100 * len([c for c in jl_cred if c >= 200]) / len(jl_cred)
        print(f"\npv_trueJL credit: median {med:.0f} pts vs 200-pt call wing -> "
              f"upside-risk-free in {cov:.0f}% of weeks")

    rows = []
    for bucket in ('BULL', 'BEAR', 'NEU'):
        print(f"\n================ {bucket}-view weeks (net of {COST_PT_PER_LEG}pt/leg/side, "
              f"{LOTS} lots) ================")
        keys = sorted({(s, e) for (b, s, e) in res if b == bucket})
        for s, e in keys:
            m = summ(res[(bucket, s, e)])
            if not m:
                continue
            rows.append(dict(bucket=bucket, struct=s, exit=e, **m))
            print(f"  {s:12s} {e:12s} n={m['n']:3d} win%={m['winpct']:5.1f} "
                  f"avg=Rs{m['avg']:8,.0f} tot=Rs{m['tot']:10,.0f} PF={m['pf']:5.2f} "
                  f"worst=Rs{m['worst']:9,.0f} p5=Rs{m['p5']:9,.0f}")

    with open(f'{OUT}/jl_ic_ranking.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['bucket','struct','exit','n','winpct',
                                          'avg','tot','pf','worst','p5'])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(f'{OUT}/jl_ic_by_year.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bucket','struct','exit','year','n','avg','tot'])
        for (b, s, e, y), p in sorted(by_year.items()):
            w.writerow([b, s, e, y, len(p), round(sum(p)/len(p)), round(sum(p))])
    print(f"\nwrote {OUT}/jl_ic_ranking.csv and jl_ic_by_year.csv")

if __name__ == '__main__':
    main()
