#!/usr/bin/env python3
"""research/90 G3: Arun's weekly spec — Monday entry, next-week expiry (DTE 6-12),
strikes by PREMIUM TARGET (Rs 15/20/25/30 per leg), stop 1.5x/2.0x pessimistic,
roll-away once (re-targeted to same premium rule), 2nd stop -> flat.
Comparator: day-after-expiry entry with the same premium targeting.
See MONDAY_20RS_STRANGLE_WEEKLY_SWEEP_STATUS.md.
"""
import csv
import datetime as dt
import os
import sqlite3
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
DTE_LO, DTE_HI = 6, 12
AFTEREXP_DTE = (3, 10)
TIME_EXIT_DTE = 1
T_GRID = [15, 20, 25, 30]
STOP_GRID = [1.5, 2.0]
PT_GRID = [None, 0.4, 0.5, 0.6, 0.7, 0.8]


def log(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_chain():
    log("loading nse_options_bhav (NIFTY) OHLC...")
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
    log(f"loaded {len(data)} rows")
    return data, sorted(days_set), sorted(exp_set)


def build_day_idx(data):
    idx = defaultdict(lambda: defaultdict(lambda: {"CE": {}, "PE": {}}))
    for (td, ed, ot, k), v in data.items():
        idx[td][ed][ot][k] = v
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


def pick_by_premium(day_idx, day, exp, ot, target_prem, spot):
    """OTM strike whose close premium is nearest target; liquid; >=MIN_LEG_PTS."""
    ch = day_idx.get(day, {}).get(exp, {}).get(ot, {})
    best, bdiff = None, None
    for k, v in ch.items():
        if ot == "PE" and k >= spot:
            continue
        if ot == "CE" and k <= spot:
            continue
        if v[3] < MIN_CONTRACTS or v[2] < MIN_LEG_PTS:
            continue
        diff = abs(v[2] - target_prem)
        if bdiff is None or diff < bdiff:
            best, bdiff = k, diff
    if best is not None and bdiff is not None and bdiff <= target_prem:  # sanity
        return best
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


def stop_fill(bar, level):
    o, h, c, _ = bar
    if o >= level:
        return o
    if h >= level:
        return level
    return None


def sim_roll(series, pe0, ce0, pt, stop, target_prem, day_idx, days, exp, spots):
    credit = pe0 + ce0
    sl = {"PE": stop * pe0, "CE": stop * ce0}
    live = {"PE": pe0, "CE": ce0}
    realized, rolled, exit_prem_sum = 0.0, False, 0.0
    entry_prem = credit
    j = 0
    while j < len(series) - 1:
        j += 1
        d, pe, ce, cdte = series[j]
        bars = {"PE": pe, "CE": ce}
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
            if not rolled:
                rolled = True
                spot_d = spots.get(d)
                if spot_d:
                    nk = pick_by_premium(day_idx, d, exp, side, target_prem, spot_d)
                    if nk:
                        nbar = day_idx[d][exp][side][nk]
                        live[side] = nbar[2]
                        entry_prem += nbar[2]
                        sl[side] = stop * nbar[2]
                        series = _patch(series, j, day_idx, exp, (side, nk))
            continue
        comb = sum(bars[s][2] for s in live)
        profit = sum(live[s] for s in live) - comb + realized
        if pt and profit >= pt * credit:
            return d, profit, "PT", exit_prem_sum + comb, entry_prem
        if cdte <= TIME_EXIT_DTE:
            return d, profit, "TIME", exit_prem_sum + comb, entry_prem
    d, pe, ce, _ = series[-1]
    bars = {"PE": pe, "CE": ce}
    comb = sum(bars[s][2] for s in live)
    return d, sum(live[s] for s in live) - comb + realized, "EXPIRY", exit_prem_sum + comb, entry_prem


def _patch(series, from_j, day_idx, exp, new_strike):
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
        out[idx] = (d, bar, ce, cdte) if side == "PE" else (d, pe, bar, cdte)
    return out


def main():
    t0 = time.time()
    data, days, expiries = load_chain()
    day_idx = build_day_idx(data)
    spots = {}
    for d in days:
        if d >= START:
            s = calc_spot(day_idx, d, expiries)
            if s:
                spots[d] = s

    # entry days
    entries = {"monday": [], "afterexp": []}
    prev_wk = None
    for d in days:
        if d < START or d not in spots:
            continue
        wk = dt.date.fromisoformat(d).isocalendar()[:2]
        if wk != prev_wk:
            entries["monday"].append(d)
            prev_wk = wk
    prev = None
    for e in expiries:
        if prev:
            nxt = [d for d in days if d > prev and d in spots]
            if nxt:
                entries["afterexp"].append(nxt[0])
        prev = e
    entries["afterexp"] = sorted(set(entries["afterexp"]))

    def target_expiry(entry, lo, hi):
        d0 = dt.date.fromisoformat(entry)
        for e in expiries:
            if e > entry:
                cdte = (dt.date.fromisoformat(e) - d0).days
                if lo <= cdte <= hi:
                    return e, cdte
                if cdte > hi:
                    return None, None
        return None, None

    out_path = os.path.join(RESDIR, "g3_cycles.csv")
    fields = ["mode", "entry", "expiry", "dte", "T", "pt", "stop", "pe_k", "ce_k",
              "credit", "otm_pct", "exit_day", "exit_reason", "gross", "net", "year"]
    f = open(out_path, "w", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    nrows, skips = 0, 0

    for mode, elist in entries.items():
        lo, hi = (DTE_LO, DTE_HI) if mode == "monday" else AFTEREXP_DTE
        for entry in elist:
            exp, cdte = target_expiry(entry, lo, hi)
            if not exp:
                skips += 1
                continue
            spot = spots[entry]
            for T in T_GRID:
                pe_k = pick_by_premium(day_idx, entry, exp, "PE", T, spot)
                ce_k = pick_by_premium(day_idx, entry, exp, "CE", T, spot)
                if not pe_k or not ce_k:
                    skips += 1
                    continue
                series = build_series(day_idx, days, entry, exp, pe_k, ce_k)
                if not series or len(series) < 2:
                    skips += 1
                    continue
                pe0, ce0 = series[0][1][2], series[0][2][2]
                credit = pe0 + ce0
                otm = round(100 * min(spot - pe_k, ce_k - spot) / spot, 2)
                for stop in STOP_GRID:
                    for pt in PT_GRID:
                        xd, gross, reason, xprem, eprem = sim_roll(
                            list(series), pe0, ce0, pt, stop, T, day_idx, days, exp, spots)
                        net = gross - (COST_PCT * (eprem + xprem) + COST_FLAT)
                        w.writerow(dict(mode=mode, entry=entry, expiry=exp, dte=cdte,
                                        T=T, pt=pt, stop=stop, pe_k=pe_k, ce_k=ce_k,
                                        credit=round(credit, 2), otm_pct=otm,
                                        exit_day=xd, exit_reason=reason,
                                        gross=round(gross, 2), net=round(net, 2),
                                        year=entry[:4]))
                        nrows += 1
    f.close()
    log(f"SIM DONE rows={nrows} skips={skips} in {time.time()-t0:.0f}s")

    import statistics as st
    rows = list(csv.DictReader(open(out_path)))
    for r in rows:
        r["net"] = float(r["net"])
    cfgs = defaultdict(list)
    for r in rows:
        cfgs[(r["mode"], r["T"], r["pt"], r["stop"])].append(r)
    rank_path = os.path.join(RESDIR, "g3_ranking.csv")
    yf = open(os.path.join(RESDIR, "g3_yearly.csv"), "w", newline="")
    yw = csv.DictWriter(yf, fieldnames=["mode", "T", "pt", "stop", "year", "n", "net_mean"])
    yw.writeheader()
    rf = open(rank_path, "w", newline="")
    rfields = ["mode", "T", "pt", "stop", "n", "net_mean", "net_t", "win_rate", "worst",
               "p5", "post22_mean", "post22_worst", "avg_credit", "avg_otm", "avg_dte", "roll_rate"]
    rw = csv.DictWriter(rf, fieldnames=rfields)
    rw.writeheader()
    summary = []
    for key, rs in cfgs.items():
        nets = [r["net"] for r in rs]
        n = len(nets)
        mu = st.mean(nets)
        sd = st.stdev(nets) if n > 1 else 0
        post = [r["net"] for r in rs if r["year"] >= "2022"]
        row = dict(mode=key[0], T=key[1], pt=key[2], stop=key[3], n=n,
                   net_mean=round(mu, 2),
                   net_t=round(mu / (sd / n ** 0.5), 2) if sd else 0,
                   win_rate=round(100 * sum(1 for x in nets if x > 0) / n, 1),
                   worst=round(min(nets), 1), p5=round(sorted(nets)[max(0, n // 20)], 1),
                   post22_mean=round(st.mean(post), 2) if post else "",
                   post22_worst=round(min(post), 1) if post else "",
                   avg_credit=round(st.mean([float(r["credit"]) for r in rs]), 1),
                   avg_otm=round(st.mean([float(r["otm_pct"]) for r in rs]), 2),
                   avg_dte=round(st.mean([float(r["dte"]) for r in rs]), 1),
                   roll_rate=round(100 * sum(1 for r in rs if r["exit_reason"] in
                                             ("STOP2_FLAT",) or "STOP" in r["exit_reason"]) / n, 1))
        rw.writerow(row)
        summary.append(row)
        for y in sorted(set(r["year"] for r in rs)):
            ys = [r["net"] for r in rs if r["year"] == y]
            yw.writerow(dict(mode=key[0], T=key[1], pt=key[2], stop=key[3],
                             year=y, n=len(ys), net_mean=round(st.mean(ys), 2)))
    rf.close()
    yf.close()

    for mode in ("monday", "afterexp"):
        log(f"=== {mode} (sorted by t) ===")
        for s in sorted([x for x in summary if x["mode"] == mode],
                        key=lambda x: -float(x["net_t"])):
            log(f"  T{s['T']} pt{s['pt']} stop{s['stop']} n={s['n']} mean={s['net_mean']} "
                f"t={s['net_t']} win={s['win_rate']} worst={s['worst']} p5={s['p5']} "
                f"post22={s['post22_mean']}/{s['post22_worst']} credit={s['avg_credit']} "
                f"otm={s['avg_otm']}% dte={s['avg_dte']}")
    log(f"ALL DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
