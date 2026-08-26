#!/usr/bin/env python3
"""research/128 stage 2 - sweep naked-survivor trail designs over the synthesised episodes.

Reads results/paths.jsonl (from build_episodes.py) and replays each ARM over every episode
at 1-minute poll resolution. Net-of-cost with the MEASURED outcome-aware cost model
(research/122 stage_a_alldays.cost_per_lot), reduced to ONE leg.

Outputs: results/arm_episode.csv  (arm x episode, the heavy file)
         results/arm_summary.csv  (one row per arm)
         results/roundtrip.csv    (per-episode round-trip statistics)
"""
import csv, json, os, sys, math, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
sys.path.insert(0, '/home/arun/quantifyd')

LOT = 20
EOD_M = 15 * 60 + 15
BAR = 5                      # the live trail builds 5-minute premium candles

# --- MEASURED cost model (research/122, 443 real live leg-sides) -------------
NLOTS_REF = 10
SLIP_ENTRY = 0.0
SLIP_TIME = 0.178
SLIP_STOP = 6.548
URGENT = {'TRAIL', 'BE', 'TP', 'INCUMBENT', 'GIVEBACK', 'RANDOM'}


def leg_cost_per_lot(sell_prem, buy_prem, lot, reason, slip_mult=1.0):
    """Round-trip cost of ONE short option leg, Rs per lot."""
    sell = sell_prem * lot
    buy = buy_prem * lot
    tot = sell + buy
    brok = 40.0 / NLOTS_REF                 # Rs20 x 2 orders, spread over the study lot count
    stt = 0.001 * sell                      # STT on the sell side premium
    txn = 0.0003503 * tot
    ipft = 0.0000050 * tot
    sebi = 0.0000010 * tot
    stamp = 0.00003 * buy
    gst = 0.18 * (brok + txn + ipft + sebi)
    slip = (SLIP_ENTRY + (SLIP_STOP if reason in URGENT else SLIP_TIME)) * slip_mult
    return brok + stt + txn + ipft + sebi + stamp + gst + slip * lot


# ---------------------------------------------------------------- candles ---
def bars_from(polls, bar=BAR):
    """Build completed OHLC bars from (minute, price) polls. Returns (bars, cur)."""
    bars, cur, bkt = [], None, None
    for m, p in polls:
        b = (m // bar) * bar
        if b != bkt:
            if cur is not None:
                bars.append(cur)
            cur = dict(open=p, high=p, low=p, close=p)
            bkt = b
        else:
            cur['high'] = max(cur['high'], p)
            cur['low'] = min(cur['low'], p)
            cur['close'] = p
    return bars, cur, bkt


def short_trail(candles, period, mult):
    """NasAtm4Executor.compute_short_trailing_stop - ratcheting upper band for a SHORT."""
    n = len(candles)
    if n < period + 1:
        return None
    high = np.array([c['high'] for c in candles], dtype=float)
    low = np.array([c['low'] for c in candles], dtype=float)
    close = np.array([c['close'] for c in candles], dtype=float)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    hl2 = (high + low) / 2.0
    upper = hl2 + mult * atr
    stop = np.zeros(n)
    stop[period - 1] = upper[period - 1]
    for i in range(period, n):
        cand = min(upper[i], stop[i - 1])
        stop[i] = upper[i] if close[i] > cand else cand
    return float(stop[-1])


import pandas as pd
from services.technical_indicators import calc_supertrend as _repo_st


def tv_supertrend(candles, period, mult):
    """EXACTLY what services/sensex_naked_trail.py does today: the repo's TradingView
    band-locked SuperTrend on the 5-min premium candles. Its value can sit BELOW the
    premium (direction==1 -> lower band) - that is the live self-trigger bug."""
    if len(candles) < period + 2:
        return None
    try:
        st, _d = _repo_st(pd.DataFrame(candles), period, mult)
        v = float(st.iloc[-1])
    except Exception:
        return None
    if not (v == v) or v <= 0:
        return None
    return round(v, 1)


# ------------------------------------------------------------------- arms ---
def sim(ep, arm):
    """Replay one arm over one episode. Returns (exit_minute, exit_price, reason)."""
    E = ep['entry']
    path = ep['path']
    kind = arm['kind']

    if kind == 'HOLD':
        return path[-1][0], path[-1][1], 'EOD'

    if kind == 'BE':
        for m, p in path:
            if p >= E:
                return m, p, 'BE'
        return path[-1][0], path[-1][1], 'EOD'

    if kind == 'TP':
        x = arm['x']
        for m, p in path:
            if p >= E:
                return m, p, 'BE'
            if p <= x * E:
                return m, p, 'TP'
        return path[-1][0], path[-1][1], 'EOD'

    if kind in ('GIVEBACK', 'GIVEBACK_PTS'):
        best = None
        N = arm.get('N', 1)
        breach = 0
        for m, p in path:
            stop = E if best is None else min(
                best * (1 + arm['g']) if kind == 'GIVEBACK' else best + arm['g'], E)
            if p >= stop:
                breach += 1
                if breach >= N:
                    return m, p, 'GIVEBACK' if best is not None and stop < E else 'BE'
            else:
                breach = 0
            best = p if best is None else min(best, p)
        return path[-1][0], path[-1][1], 'EOD'

    if kind == 'INCUMBENT':
        # exactly today's behaviour: TV SuperTrend(7,3) on 5-min premium candles built from
        # the naked moment forward; value written into sl_price; fires on live >= sl.
        period, mult = arm.get('period', 7), arm.get('mult', 3.0)
        bars, cur, bkt = [], None, None
        for m, p in path:
            b = (m // BAR) * BAR
            if b != bkt:
                if cur is not None:
                    bars.append(cur)
                    bars = bars[-80:]
                cur = dict(open=p, high=p, low=p, close=p)
                bkt = b
            else:
                cur['high'] = max(cur['high'], p)
                cur['low'] = min(cur['low'], p)
                cur['close'] = p
            seq = bars + [cur]
            st = tv_supertrend(seq, period, mult) if len(seq) >= period + 2 else None
            sl = st if (st is not None and st < E) else E
            if p >= sl:
                return m, p, 'INCUMBENT'
        return path[-1][0], path[-1][1], 'EOD'

    if kind == 'CEIL':
        period, mult, N, seed = arm['period'], arm['mult'], arm['N'], arm['seed']
        clamp = arm.get('clamp', 1)
        if seed:
            bars, cur, bkt = bars_from(ep.get('pre') or [], BAR)
            nbkt = (ep['naked_m'] // BAR) * BAR
            if cur is not None and bkt != nbkt:
                bars = bars + [cur]      # completed before the naked bucket
                cur, bkt = None, None
        else:
            bars, cur, bkt = [], None, None
        st_val = short_trail(bars, period, mult) if len(bars) >= period + 1 else None
        breach = 0
        for m, p in path:
            b = (m // BAR) * BAR
            if b != bkt:
                if cur is not None:
                    bars.append(cur)
                    bars = bars[-200:]
                    st_val = short_trail(bars, period, mult) if len(bars) >= period + 1 else None
                cur = dict(open=p, high=p, low=p, close=p)
                bkt = b
            else:
                cur['high'] = max(cur['high'], p)
                cur['low'] = min(cur['low'], p)
                cur['close'] = p
            if st_val is None:
                stop = E
            else:
                stop = min(st_val, E) if clamp else st_val
            if p > stop or (stop >= E and p >= E):
                breach += 1
                if breach >= N:
                    return m, p, 'TRAIL' if (st_val is not None and st_val < E) else 'BE'
            else:
                breach = 0
        return path[-1][0], path[-1][1], 'EOD'

    raise ValueError(kind)


def build_arms():
    arms = []
    arms.append(dict(label='INCUMBENT', kind='INCUMBENT', family='null', period=7, mult=3.0))
    arms.append(dict(label='BE_ONLY', kind='BE', family='null'))
    arms.append(dict(label='HOLD_EOD', kind='HOLD', family='null'))
    for period in (5, 7, 10, 14):
        for mult in (2.0, 2.5, 3.0, 3.5, 4.0):
            for N in (1, 2, 3, 5):
                for seed in (0, 1):
                    arms.append(dict(
                        label='CEIL_p%d_m%.1f_N%d_%s' % (period, mult, N,
                                                         'SEED' if seed else 'COLD'),
                        kind='CEIL', family='CEIL', period=period, mult=mult, N=N, seed=seed))
    for N in (1, 2, 3):
        for seed in (0, 1):
            arms.append(dict(
                label='CEILNC_p7_m3.0_N%d_%s' % (N, 'SEED' if seed else 'COLD'),
                kind='CEIL', family='CEIL_NOCLAMP', period=7, mult=3.0, N=N,
                seed=seed, clamp=0))
    for g in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        arms.append(dict(label='GIVEBACK_%d%%' % round(g * 100), kind='GIVEBACK',
                         family='GIVEBACK', g=g, N=1))
    for g in (10, 15, 20, 25, 30, 40):
        arms.append(dict(label='GIVEBACK_%dpt' % g, kind='GIVEBACK_PTS',
                         family='GIVEBACK_PTS', g=float(g), N=1))
    for x in (0.70, 0.60, 0.50, 0.40, 0.30, 0.20):
        arms.append(dict(label='TP_%d%%' % round(x * 100), kind='TP', family='TP', x=x))
    return arms


def roundtrip_stats(ep):
    E = ep['entry']
    path = ep['path']
    prices = [p for _, p in path]
    mn = min(prices)
    imn = prices.index(mn)
    after = prices[imn:]
    mx_after = max(after)
    mx_all = max(prices)
    dec = (E - mn)
    rec = (mx_after - mn) / dec if dec > 0 else 0.0
    return dict(
        eid=ep['eid'], system=ep['system'], day=ep['day'], weekday=ep['weekday'],
        dte=ep['dte'], side=ep['side'], entry=E,
        naked_hm='%02d:%02d' % (ep['naked_m'] // 60, ep['naked_m'] % 60),
        naked_px=ep['naked_px'], min_px=mn, min_hm='%02d:%02d' % (path[imn][0] // 60, path[imn][0] % 60),
        eod_px=prices[-1], max_after_min=mx_after, max_all=mx_all,
        decay_to_pct=round(100.0 * mn / E, 1),
        recovery_frac=round(rec, 4),
        rt_full=int(mx_after >= E),
        rt_75=int(rec >= 0.75), rt_50=int(rec >= 0.50), rt_25=int(rec >= 0.25),
        best_pnl_lot=round((E - mn) * LOT), eod_pnl_lot=round((E - prices[-1]) * LOT),
        naked_pnl_lot=round((E - ep['naked_px']) * LOT))


def main():
    random.seed(20260826)
    eps = [json.loads(l) for l in open(os.path.join(RES, 'paths.jsonl'))]
    print('episodes: %d' % len(eps), flush=True)

    # ---- round-trip statistics -------------------------------------------
    rt = [roundtrip_stats(e) for e in eps]
    with open(os.path.join(RES, 'roundtrip.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rt[0].keys()))
        w.writeheader()
        w.writerows(rt)
    print('roundtrip.csv written', flush=True)

    # ---- random-exit null -------------------------------------------------
    NDRAW = 400
    rand_net = {}
    for e in eps:
        E = e['entry']
        vals = []
        for _ in range(NDRAW):
            m, p = random.choice(e['path'])
            vals.append((E - p) * LOT - leg_cost_per_lot(E, p, LOT, 'RANDOM'))
        rand_net[e['eid']] = float(np.mean(vals))

    arms = build_arms()
    print('arms: %d' % len(arms), flush=True)
    fe = open(os.path.join(RES, 'arm_episode.csv'), 'w', newline='')
    we = csv.DictWriter(fe, fieldnames=['arm', 'family', 'eid', 'system', 'day', 'weekday',
                                        'dte', 'side', 'entry', 'naked_hm', 'exit_hm',
                                        'exit_px', 'reason', 'gross_lot', 'cost_lot', 'net_lot',
                                        'hold_min'])
    we.writeheader()
    per_arm = {}
    for a in arms:
        rows = []
        for e in eps:
            m, p, reason = sim(e, a)
            E = e['entry']
            gross = (E - p) * LOT
            cost = leg_cost_per_lot(E, p, LOT, reason)
            net = gross - cost
            rows.append((e, m, p, reason, gross, cost, net))
            we.writerow(dict(arm=a['label'], family=a['family'], eid=e['eid'],
                             system=e['system'], day=e['day'], weekday=e['weekday'],
                             dte=e['dte'], side=e['side'], entry=E,
                             naked_hm='%02d:%02d' % (e['naked_m'] // 60, e['naked_m'] % 60),
                             exit_hm='%02d:%02d' % (m // 60, m % 60), exit_px=round(p, 2),
                             reason=reason, gross_lot=round(gross), cost_lot=round(cost),
                             net_lot=round(net), hold_min=m - e['naked_m']))
        per_arm[a['label']] = (a, rows)
        fe.flush()
    fe.close()
    print('arm_episode.csv written', flush=True)

    # ---- summary ----------------------------------------------------------
    base_net = {e['eid']: n for (e, _m, _p, _r, _g, _c, n) in per_arm['BE_ONLY'][1]}
    inc_net = {e['eid']: n for (e, _m, _p, _r, _g, _c, n) in per_arm['INCUMBENT'][1]}
    days = sorted({e['day'] for e in eps})
    split = days[int(len(days) * 0.6)]

    fs = open(os.path.join(RES, 'arm_summary.csv'), 'w', newline='')
    cols = ['arm', 'family', 'n', 'total_net_lot', 'mean_net_lot', 'median_net_lot',
            'sd_net_lot', 't_stat', 'win_rate', 'worst_lot', 'best_lot',
            'fire_rate', 'mean_hold_min',
            'vs_be_mean', 'vs_be_t', 'vs_inc_mean', 'vs_inc_t', 'vs_rand_mean',
            'is_mean', 'oos_mean', 'oos_n', 'oos_total',
            'dte1_mean', 'dte2_mean', 'dte3p_mean',
            'atm_mean', 'atm4_mean']
    ws = csv.DictWriter(fs, fieldnames=cols)
    ws.writeheader()
    for lbl, (a, rows) in per_arm.items():
        net = np.array([r[6] for r in rows], dtype=float)
        hold = np.array([r[1] - r[0]['naked_m'] for r in rows], dtype=float)
        fired = np.array([1.0 if r[3] != 'EOD' else 0.0 for r in rows])
        dbe = np.array([r[6] - base_net[r[0]['eid']] for r in rows])
        dic = np.array([r[6] - inc_net[r[0]['eid']] for r in rows])
        dra = np.array([r[6] - rand_net[r[0]['eid']] for r in rows])
        isk = np.array([r[0]['day'] < split for r in rows])

        def t(x):
            return float(np.mean(x) / (np.std(x, ddof=1) / math.sqrt(len(x)))) if len(x) > 1 and np.std(x, ddof=1) > 0 else 0.0

        def sub(mask):
            v = net[mask]
            return round(float(np.mean(v))) if len(v) else ''
        dte = np.array([r[0]['dte'] for r in rows])
        sysa = np.array([r[0]['system'] for r in rows])
        ws.writerow(dict(
            arm=lbl, family=a['family'], n=len(rows),
            total_net_lot=round(float(net.sum())), mean_net_lot=round(float(net.mean()), 1),
            median_net_lot=round(float(np.median(net)), 1),
            sd_net_lot=round(float(net.std(ddof=1)), 1), t_stat=round(t(net), 2),
            win_rate=round(100.0 * float((net > 0).mean()), 1),
            worst_lot=round(float(net.min())), best_lot=round(float(net.max())),
            fire_rate=round(100.0 * float(fired.mean()), 1),
            mean_hold_min=round(float(hold.mean()), 1),
            vs_be_mean=round(float(dbe.mean()), 1), vs_be_t=round(t(dbe), 2),
            vs_inc_mean=round(float(dic.mean()), 1), vs_inc_t=round(t(dic), 2),
            vs_rand_mean=round(float(dra.mean()), 1),
            is_mean=sub(isk), oos_mean=sub(~isk), oos_n=int((~isk).sum()),
            oos_total=round(float(net[~isk].sum())),
            dte1_mean=sub(dte == 1), dte2_mean=sub(dte == 2), dte3p_mean=sub(dte >= 3),
            atm_mean=sub(sysa == 'ATM'), atm4_mean=sub(sysa == 'ATM4')))
    fs.close()
    print('arm_summary.csv written', flush=True)
    with open(os.path.join(RES, 'random_null.json'), 'w') as f:
        json.dump(dict(mean_random_net_lot=float(np.mean(list(rand_net.values()))),
                       ndraw=NDRAW, per_ep=rand_net), f, indent=1)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
