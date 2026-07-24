#!/usr/bin/env python3
"""research/90 G2: pessimistic fills, post-stop actions, indicator exits, iron condor.

Extends run_g1_daily_sweep.py. See NIFTY_STRANGLE_G2_PESSFILL_DAILY_SWEEP_STATUS.md
for locked mechanics. Key deltas vs G1:
 - stop triggers use option daily OPEN/HIGH: gap-open >= level -> both legs fill
   at open; intraday high >= level -> stopped leg at level, other at close
 - post-stop arms: A0 flat-both | leg (ride survivor w/ own stop) | roll (re-sell
   stopped side at same %OTM from parity spot, once; 2nd stop -> flat all)
 - indicator exits (replace premium stop): ATR14 pctile 80/90, ADX14 20/25,
   VIX +10% day jump, VIX >= 1.25x entry -> exit both at NEXT day's OPEN
 - iron condor: wings at entry close (M +-500, W +-200), closed with the shorts
"""
import csv
import datetime as dt
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict

BASE = "/home/arun/quantifyd"
RESDIR = os.path.join(BASE, "research/90_nifty_strangle_rules/results")
DB = os.path.join(BASE, "backtest_data/market_data.db")
START = "2019-01-01"
MIN_CONTRACTS = 50
MIN_LEG_PTS = 1.5
COST_PCT = 0.005
COST_FLAT = 0.15
P_GRID = {"M": [0.020, 0.025, 0.030], "W": [0.008, 0.012, 0.016]}
P_FIX = {"M": 0.025, "W": 0.012}
TIME_EXIT_DTE = {"M": 2, "W": 1}
DTE_RANGE = {"M": (15, 40), "W": (3, 10)}
WING = {"M": 500, "W": 200}

os.makedirs(RESDIR, exist_ok=True)


def log(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_kite_daily(token, cache_name):
    path = os.path.join(RESDIR, cache_name)
    if os.path.exists(path):
        with open(path) as f:
            return {r["date"]: r for r in csv.DictReader(f)}
    sys.path.insert(0, BASE)
    from kiteconnect import KiteConnect
    import config as qcfg
    tok = json.load(open(os.path.join(BASE, "backtest_data/access_token.json")))
    kite = KiteConnect(api_key=getattr(qcfg, "KITE_API_KEY", None) or tok.get("api_key"))
    kite.set_access_token(tok["access_token"])
    rows = []
    for y0, y1 in [(2015, 2019), (2019, 2023), (2023, 2027)]:
        d = kite.historical_data(token, f"{y0}-01-01", f"{min(y1, 2026)}-12-31", "day")
        rows += [{"date": str(b["date"])[:10], "open": b["open"], "high": b["high"],
                  "low": b["low"], "close": b["close"]} for b in d]
        time.sleep(0.5)
    seen, out = set(), []
    for r in rows:
        if r["date"] not in seen:
            seen.add(r["date"])
            out.append(r)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "open", "high", "low", "close"])
        w.writeheader()
        w.writerows(out)
    return {r["date"]: r for r in out}


def wilder_indicators(nifty_daily):
    """Return dicts: atr_pct[date] (pctile of ATR14 vs trailing 252), adx[date]."""
    ds = sorted(nifty_daily)
    H = [float(nifty_daily[d]["high"]) for d in ds]
    L = [float(nifty_daily[d]["low"]) for d in ds]
    C = [float(nifty_daily[d]["close"]) for d in ds]
    n = len(ds)
    tr = [H[0] - L[0]] + [max(H[i] - L[i], abs(H[i] - C[i - 1]), abs(L[i] - C[i - 1]))
                          for i in range(1, n)]
    pdm = [0.0] + [max(H[i] - H[i - 1], 0) if (H[i] - H[i - 1]) > (L[i - 1] - L[i]) else 0
                   for i in range(1, n)]
    ndm = [0.0] + [max(L[i - 1] - L[i], 0) if (L[i - 1] - L[i]) > (H[i] - H[i - 1]) else 0
                   for i in range(1, n)]

    def wsmooth(xs, per=14):
        out, s = [], None
        for i, x in enumerate(xs):
            if i < per:
                out.append(None)
                s = (s or 0) + x
                if i == per - 1:
                    out[-1] = s
            else:
                s = s - s / per + x
                out.append(s)
        return out

    trs, pdms, ndms = wsmooth(tr), wsmooth(pdm), wsmooth(ndm)
    atr = [None if trs[i] is None else trs[i] / 14 for i in range(n)]
    dx = []
    for i in range(n):
        if trs[i] is None or trs[i] == 0:
            dx.append(None)
            continue
        pdi = 100 * pdms[i] / trs[i]
        ndi = 100 * ndms[i] / trs[i]
        dx.append(100 * abs(pdi - ndi) / (pdi + ndi) if (pdi + ndi) else None)
    adx_l, s, cnt = [None] * n, None, 0
    for i in range(n):
        if dx[i] is None:
            continue
        if s is None:
            cnt += 1
            s = (s or 0) if cnt == 1 else s
            s = dx[i] if cnt == 1 else s + dx[i]
            if cnt < 14:
                s = s if cnt == 1 else s
                continue
            s = s / 14
        else:
            s = (s * 13 + dx[i]) / 14
        adx_l[i] = s
    atr_pct, adx = {}, {}
    hist = []
    for i, d in enumerate(ds):
        if adx_l[i] is not None:
            adx[d] = adx_l[i]
        if atr[i] is not None:
            hist.append(atr[i])
            window = hist[-252:]
            atr_pct[d] = 100.0 * sum(1 for x in window if x <= atr[i]) / len(window)
    return atr_pct, adx


def load_chain():
    log("loading nse_options_bhav (NIFTY) with OHLC...")
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute(
        "SELECT trade_date, expiry_date, option_type, strike, open, high, close, "
        "settle_price, contracts FROM nse_options_bhav "
        "WHERE symbol='NIFTY' AND trade_date>='2018-12-01'")
    data, days_set, exp_set = {}, set(), set()
    for td, ed, ot, k, o, h, cl, sp, c in cur:
        p = cl if (cl or 0) > 0 else (sp or 0)
        if p <= 0:
            continue
        data[(td, ed, ot, float(k))] = (o or p, max(h or p, p), p, c or 0)
        days_set.add(td)
        exp_set.add(ed)
    con.close()
    log(f"loaded {len(data)} rows, {len(days_set)} days, {len(exp_set)} expiries")
    return data, sorted(days_set), sorted(exp_set)


def build_day_idx(data):
    idx = defaultdict(lambda: defaultdict(lambda: {"CE": {}, "PE": {}}))
    for (td, ed, ot, k), v in data.items():
        idx[td][ed][ot][k] = v  # (open, high, close, contracts)
    return idx


def calc_spot(day_idx, day, expiries):
    for e in expiries:
        if e >= day:
            ch = day_idx.get(day, {}).get(e)
            if not ch or not ch["CE"] or not ch["PE"]:
                continue
            common = set(ch["CE"]) & set(ch["PE"])
            if not common:
                continue
            k = min(common, key=lambda s: abs(ch["CE"][s][2] - ch["PE"][s][2]))
            return k + ch["CE"][k][2] - ch["PE"][k][2]
    return None


def pick_strike(day_idx, day, exp, ot, target, min_pts=MIN_LEG_PTS, min_ctr=MIN_CONTRACTS):
    ch = day_idx.get(day, {}).get(exp, {}).get(ot, {})
    for k in sorted(ch, key=lambda s: abs(s - target))[:6]:
        if ch[k][3] >= min_ctr and ch[k][2] >= min_pts:
            return k
    return None


def build_series(day_idx, days, entry, exp, pe_k, ce_k):
    i0 = days.index(entry)
    ed = dt.date.fromisoformat(exp)
    out, prev = [], None
    for d in days[i0:]:
        if d > exp:
            break
        ch = day_idx.get(d, {}).get(exp, {})
        pe = ch.get("PE", {}).get(pe_k)
        ce = ch.get("CE", {}).get(ce_k)
        if prev:
            pe = pe or (prev[1][2], prev[1][2], prev[1][2], 0)
            ce = ce or (prev[2][2], prev[2][2], prev[2][2], 0)
        if pe is None or ce is None:
            return None
        prev = (d, pe, ce)
        out.append((d, pe, ce, (ed - dt.date.fromisoformat(d)).days))
    return out


def friction(entry_prem, exit_prem):
    return COST_PCT * (entry_prem + exit_prem) + COST_FLAT


def stop_fill(bar, level):
    """bar=(open,high,close,ctr). Return fill px if stop touched (pessimistic), else None."""
    o, h, c, _ = bar
    if o >= level:
        return o
    if h >= level:
        return level
    return None


def sim_core(series, pe0, ce0, arm, pt, stop, gb):
    """A0 flat-both with pessimistic stop fills. Returns exit_day, gross, reason."""
    credit = pe0 + ce0
    peak = 0.0
    sl_pe = stop * pe0 if stop else None
    sl_ce = stop * ce0 if stop else None
    for j, (d, pe, ce, cdte) in enumerate(series):
        if j == 0:
            continue
        if stop:
            fp, fc = stop_fill(pe, sl_pe), stop_fill(ce, sl_ce)
            if fp is not None or fc is not None:
                gap = (pe[0] >= sl_pe) or (ce[0] >= sl_ce)
                if gap:
                    exit_prem = pe[0] + ce[0]
                elif fp is not None:
                    exit_prem = fp + ce[2]
                else:
                    exit_prem = pe[2] + fc
                return d, credit - exit_prem, "STOP", exit_prem
        comb = pe[2] + ce[2]
        profit = credit - comb
        peak = max(peak, profit)
        if gb and peak >= 0.25 * credit and profit <= 0.5 * peak:
            return d, profit, "GIVEBACK", comb
        if pt and profit >= pt * credit:
            return d, profit, "PT", comb
        if cdte <= TIME_EXIT_DTE[arm]:
            return d, profit, "TIME", comb
    d, pe, ce, _ = series[-1]
    comb = pe[2] + ce[2]
    return d, credit - comb, "EXPIRY", comb


def sim_leg_or_roll(series, pe0, ce0, arm, pt, stop, action, day_idx, days, exp, spots, p):
    """Post-stop arms. Returns exit_day, gross, reason, exit_prem_total, entry_prem_total."""
    credit = pe0 + ce0
    entry_prem = credit
    sl = {"PE": stop * pe0, "CE": stop * ce0}
    live = {"PE": pe0, "CE": ce0}   # entry credit per live leg
    realized = 0.0
    rolled = False
    exit_prem_sum = 0.0
    j = 0
    while j < len(series) - 1:
        j += 1
        d, pe, ce, cdte = series[j]
        bars = {"PE": pe, "CE": ce}
        # stop priority first (resting orders), then close-based exits
        hit = None
        for side in ("PE", "CE"):
            if side in live:
                f = stop_fill(bars[side], sl[side])
                if f is not None:
                    hit = (side, f)
                    break
        if hit:
            side, fill = hit
            realized += live[side] - fill
            exit_prem_sum += fill
            del live[side]
            if not live:
                return d, realized, "STOP2_FLAT", exit_prem_sum, entry_prem
            if action == "roll" and not rolled:
                rolled = True
                spot_d = spots.get(d)
                if spot_d:
                    tgt = spot_d * (1 - p) if side == "PE" else spot_d * (1 + p)
                    nk = pick_strike(day_idx, d, exp, side, tgt)
                    if nk:
                        nbar = day_idx[d][exp][side][nk]
                        live[side] = nbar[2]
                        entry_prem += nbar[2]
                        sl[side] = stop * nbar[2]
                        # swap this side's bars to the rolled strike from j+1 onward;
                        # index-based while loop sees the patched list
                        series = _patch_series(series, j, day_idx, days, exp, (side, nk))
            # 'leg' action: survivor just keeps running
            continue
        # close-based exits on remaining live legs
        comb = sum(bars[s][2] for s in live)
        profit = sum(live[s] for s in live) - comb + realized
        if pt and profit >= pt * credit:
            return d, profit, "PT", exit_prem_sum + comb, entry_prem
        if cdte <= TIME_EXIT_DTE[arm]:
            return d, profit, "TIME", exit_prem_sum + comb, entry_prem
    d, pe, ce, _ = series[-1]
    bars = {"PE": pe, "CE": ce}
    comb = sum(bars[s][2] for s in live)
    return d, sum(live[s] for s in live) - comb + realized, "EXPIRY", exit_prem_sum + comb, entry_prem


def _patch_series(series, from_j, day_idx, days, exp, new_strike):
    """Replace one side's bars from index from_j+1 onward with the rolled strike's bars."""
    side, nk = new_strike
    out = list(series)
    prev_px = None
    for idx in range(from_j, len(out)):
        d, pe, ce, cdte = out[idx]
        bar = day_idx.get(d, {}).get(exp, {}).get(side, {}).get(nk)
        if bar is None:
            px = prev_px if prev_px is not None else (pe[2] if side == "PE" else ce[2])
            bar = (px, px, px, 0)
        prev_px = bar[2]
        if side == "PE":
            out[idx] = (d, bar, ce, cdte)
        else:
            out[idx] = (d, pe, bar, cdte)
    return out


def sim_indicator(series, pe0, ce0, arm, pt, mech, atr_pct, adx, vix, entry_vix):
    credit = pe0 + ce0
    pending_exit = False
    for j, (d, pe, ce, cdte) in enumerate(series):
        if j == 0:
            continue
        if pending_exit:
            exit_prem = pe[0] + ce[0]  # next day's open
            return d, credit - exit_prem, "IND", exit_prem
        comb = pe[2] + ce[2]
        profit = credit - comb
        if pt and profit >= pt * credit:
            return d, profit, "PT", comb
        if cdte <= TIME_EXIT_DTE[arm]:
            return d, profit, "TIME", comb
        sig = False
        if mech == "ATR80":
            sig = atr_pct.get(d, 0) >= 80
        elif mech == "ATR90":
            sig = atr_pct.get(d, 0) >= 90
        elif mech == "ADX20":
            sig = adx.get(d, 0) >= 20
        elif mech == "ADX25":
            sig = adx.get(d, 0) >= 25
        elif mech == "VIXJ10":
            i = None
            v = vix.get(d)
            pv = vix.get(series[j - 1][0])
            sig = bool(v and pv and float(v["close"]) >= 1.10 * float(pv["close"]))
        elif mech == "VIXL25":
            v = vix.get(d)
            sig = bool(v and entry_vix and float(v["close"]) >= 1.25 * entry_vix)
        if sig:
            pending_exit = True
    d, pe, ce, _ = series[-1]
    comb = pe[2] + ce[2]
    return d, credit - comb, "EXPIRY", comb


def main():
    t0 = time.time()
    nifty_daily = fetch_kite_daily(256265, "cache_nifty_daily.csv")
    vix_daily = fetch_kite_daily(264969, "cache_vix_daily.csv")
    atr_pct, adx = wilder_indicators(nifty_daily)
    log(f"indicators: atr_pct {len(atr_pct)}d, adx {len(adx)}d")

    data, days, expiries = load_chain()
    day_idx = build_day_idx(data)
    monthlies = set()
    last = {}
    for e in expiries:
        last[e[:7]] = e
    monthlies = set(last.values())

    spots = {}
    for d in days:
        if d >= START:
            s = calc_spot(day_idx, d, expiries)
            if s:
                spots[d] = s
    log(f"spots for {len(spots)} days")

    cycles = []
    for arm in ("W", "M"):
        exps = [e for e in expiries if e >= START and (arm == "W" or e in monthlies)]
        prev = None
        for e in exps:
            if prev:
                nxt = [d for d in days if d > prev and d in spots]
                if nxt:
                    entry = nxt[0]
                    cdte = (dt.date.fromisoformat(e) - dt.date.fromisoformat(entry)).days
                    lo, hi = DTE_RANGE[arm]
                    if lo <= cdte <= hi and entry < e:
                        cycles.append((arm, entry, e))
            prev = e
    log(f"cycles: {len(cycles)} (W={sum(1 for c in cycles if c[0]=='W')})")

    out_path = os.path.join(RESDIR, "g2_cycles.csv")
    fields = ["family", "arm", "entry", "expiry", "p", "mech", "action", "pt", "stop",
              "gb", "credit", "exit_day", "exit_reason", "gross", "net", "vix", "year"]
    f = open(out_path, "w", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    nrows, skips = 0, 0

    for ci, (arm, entry, exp) in enumerate(cycles):
        spot = spots[entry]
        vrow = vix_daily.get(entry)
        entry_vix = float(vrow["close"]) if vrow else None
        for p in P_GRID[arm]:
            pe_k = pick_strike(day_idx, entry, exp, "PE", spot * (1 - p))
            ce_k = pick_strike(day_idx, entry, exp, "CE", spot * (1 + p))
            if not pe_k or not ce_k or ce_k <= pe_k:
                skips += 1
                continue
            series = build_series(day_idx, days, entry, exp, pe_k, ce_k)
            if not series or len(series) < 2:
                skips += 1
                continue
            pe0 = series[0][1][2]
            ce0 = series[0][2][2]
            credit = pe0 + ce0

            def emit(family, mech, action, pt, stop, gb, xd, gross, reason, exit_prem, entry_prem=None):
                nonlocal nrows
                ep = entry_prem if entry_prem is not None else credit
                net = gross - friction(ep, exit_prem)
                w.writerow(dict(family=family, arm=arm, entry=entry, expiry=exp, p=p,
                                mech=mech, action=action, pt=pt, stop=stop, gb=gb,
                                credit=round(credit, 2), exit_day=xd, exit_reason=reason,
                                gross=round(gross, 2), net=round(net, 2),
                                vix=entry_vix or "", year=entry[:4]))
                nrows += 1

            # --- family: core (pessimistic fills), full G1 grid
            for pt in (0.4, 0.5, 0.6, None):
                for stop in (1.5, 2.0, 2.5, None):
                    for gb in (0, 1):
                        xd, gross, reason, xprem = sim_core(series, pe0, ce0, arm, pt, stop, gb)
                        emit("core", "PS", "A0", pt, stop, gb, xd, gross, reason, xprem)

            if abs(p - P_FIX[arm]) < 1e-9:
                # --- family: post-stop actions
                for stop in (1.5, 2.0, 2.5):
                    for pt in (0.5, None):
                        for action in ("leg", "roll"):
                            xd, gross, reason, xprem, eprem = sim_leg_or_roll(
                                list(series), pe0, ce0, arm, pt, stop, action,
                                day_idx, days, exp, spots, p)
                            emit("poststop", "PS", action, pt, stop, 0, xd, gross, reason, xprem, eprem)
                # --- family: indicator exits
                for mech in ("ATR80", "ATR90", "ADX20", "ADX25", "VIXJ10", "VIXL25"):
                    for pt in (0.5, None):
                        xd, gross, reason, xprem = sim_indicator(
                            series, pe0, ce0, arm, pt, mech, atr_pct, adx, vix_daily, entry_vix)
                        emit("indicator", mech, "A0", pt, None, 0, xd, gross, reason, xprem)
                # --- family: iron condor
                wpe = pick_strike(day_idx, entry, exp, "PE", pe_k - WING[arm], min_pts=0.5, min_ctr=0)
                wce = pick_strike(day_idx, entry, exp, "CE", ce_k + WING[arm], min_pts=0.5, min_ctr=0)
                if wpe and wce and wpe < pe_k and wce > ce_k:
                    wseries = build_series(day_idx, days, entry, exp, wpe, wce)
                    if wseries and len(wseries) == len(series):
                        wcost = wseries[0][1][2] + wseries[0][2][2]
                        for stop in (2.0, None):
                            for pt in (0.5, None):
                                xd, gross, reason, xprem = sim_core(series, pe0, ce0, arm, pt, stop, 0)
                                widx = next(i for i, s in enumerate(series) if s[0] == xd)
                                wexit = wseries[widx][1][2] + wseries[widx][2][2]
                                cg = gross - wcost + wexit
                                emit("condor", "PS", "IC", pt, stop, 0, xd, cg, reason,
                                     xprem + wexit, credit + wcost)
        if (ci + 1) % 50 == 0:
            f.flush()
            log(f"cycle {ci+1}/{len(cycles)} rows={nrows} skips={skips}")
    f.close()
    log(f"SIM DONE rows={nrows} skips={skips} in {time.time()-t0:.0f}s")

    # ---------------- aggregate ----------------
    import statistics as st
    rows = list(csv.DictReader(open(out_path)))
    for r in rows:
        r["net"] = float(r["net"])
    key_fields = ("family", "arm", "p", "mech", "action", "pt", "stop", "gb")
    cfgs = defaultdict(list)
    for r in rows:
        cfgs[tuple(r[k] for k in key_fields)].append(r)

    rank_path = os.path.join(RESDIR, "g2_ranking.csv")
    yr_path = os.path.join(RESDIR, "g2_yearly.csv")
    rf = open(rank_path, "w", newline="")
    yf = open(yr_path, "w", newline="")
    rfields = list(key_fields) + ["n", "net_mean", "net_t", "win_rate", "worst", "p5",
                                  "post22_mean", "post22_worst"]
    rw = csv.DictWriter(rf, fieldnames=rfields)
    rw.writeheader()
    yw = csv.DictWriter(yf, fieldnames=list(key_fields) + ["year", "n", "net_mean"])
    yw.writeheader()
    summary = []
    for key, rs in cfgs.items():
        nets = [r["net"] for r in rs]
        n = len(nets)
        mu = st.mean(nets)
        sd = st.stdev(nets) if n > 1 else 0
        post = [r["net"] for r in rs if r["year"] >= "2022"]
        row = dict(zip(key_fields, key))
        row.update(n=n, net_mean=round(mu, 2),
                   net_t=round(mu / (sd / n ** 0.5), 2) if sd else 0,
                   win_rate=round(100 * sum(1 for x in nets if x > 0) / n, 1),
                   worst=round(min(nets), 1), p5=round(sorted(nets)[max(0, n // 20)], 1),
                   post22_mean=round(st.mean(post), 2) if post else "",
                   post22_worst=round(min(post), 1) if post else "")
        rw.writerow(row)
        summary.append(row)
        for y in sorted(set(r["year"] for r in rs)):
            ys = [r["net"] for r in rs if r["year"] == y]
            yrow = dict(zip(key_fields, key))
            yrow.update(year=y, n=len(ys), net_mean=round(st.mean(ys), 2))
            yw.writerow(yrow)
    rf.close()
    yf.close()

    def show(title, sel, sort_key="net_t"):
        log(f"=== {title} ===")
        for s in sorted(sel, key=lambda x: -float(x[sort_key] or 0))[:10]:
            log(f"  {s['family']}/{s['arm']}/p{s['p']}/{s['mech']}/{s['action']}/pt{s['pt']}/"
                f"stop{s['stop']} n={s['n']} mean={s['net_mean']} t={s['net_t']} "
                f"worst={s['worst']} p5={s['p5']} post22={s['post22_mean']} "
                f"post22worst={s['post22_worst']}")

    show("CORE pessimistic, M p=0.025, no gb",
         [s for s in summary if s["family"] == "core" and s["arm"] == "M"
          and s["p"] == "0.025" and s["gb"] == "0"])
    show("CORE pessimistic, W p=0.012, no gb",
         [s for s in summary if s["family"] == "core" and s["arm"] == "W"
          and s["p"] == "0.012" and s["gb"] == "0"])
    show("POST-STOP actions", [s for s in summary if s["family"] == "poststop"])
    show("INDICATOR exits", [s for s in summary if s["family"] == "indicator"])
    show("IRON CONDOR", [s for s in summary if s["family"] == "condor"])
    log(f"ALL DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
