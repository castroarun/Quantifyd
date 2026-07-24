#!/usr/bin/env python3
"""research/90 G1: NIFTY rules-based short strangle — daily-granularity exit-rule sweep.

Real NSE bhav EOD option data (nse_options_bhav), 2019->2026. A0 arm only (no
adjustments): fixed entry day, fixed %OTM strikes, pre-committed exits.
Gates (VIX / weekly CPR width) recorded per cycle, evaluated in aggregation.
See NIFTY_STRANGLE_RULES_DAILY_SWEEP_STATUS.md for the locked design.
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
MIN_CONTRACTS = 50          # binding liquidity filter (research/89)
MIN_LEG_PTS = 1.5
COST_PCT = 0.005            # friction: 0.5% of (entry+exit premium)
COST_FLAT = 0.15            # pts flat per cycle (~Rs.100 at 650 qty)
PT_GRID = [0.40, 0.50, 0.60, None]
STOP_GRID = [1.5, 2.0, 2.5, None]
GB_GRID = [True, False]
P_GRID = {"M": [0.020, 0.025, 0.030], "W": [0.008, 0.012, 0.016]}
TIME_EXIT_DTE = {"M": 2, "W": 1}
DTE_RANGE = {"M": (15, 40), "W": (3, 10)}

os.makedirs(RESDIR, exist_ok=True)


def log(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------- Kite daily caches (NIFTY OHLC for CPR, VIX) ----------------
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


def weekly_cpr_terciles(nifty_daily):
    """Prior-week CPR width pct per week-start date -> tercile (1=narrow) vs trailing 52w."""
    by_week = defaultdict(list)
    for ds, r in sorted(nifty_daily.items()):
        d = dt.date.fromisoformat(ds)
        by_week[d.isocalendar()[:2]].append((ds, float(r["high"]), float(r["low"]), float(r["close"])))
    weeks = sorted(by_week)
    width = {}
    for wk in weeks:
        rows = by_week[wk]
        H = max(r[1] for r in rows); L = min(r[2] for r in rows); C = rows[-1][3]
        P = (H + L + C) / 3.0; BC = (H + L) / 2.0; TC = 2 * P - BC
        width[wk] = abs(TC - BC) / C * 100.0
    terc = {}
    for i, wk in enumerate(weeks):
        if i == 0:
            continue
        hist = [width[w] for w in weeks[max(0, i - 52):i]]
        prev_w = width[weeks[i - 1]]
        if len(hist) < 20:
            terc[wk] = 0  # unknown
        else:
            hs = sorted(hist)
            t1, t2 = hs[len(hs) // 3], hs[2 * len(hs) // 3]
            terc[wk] = 1 if prev_w <= t1 else (2 if prev_w <= t2 else 3)
    return terc  # keyed by (iso_year, iso_week) of the ENTRY week; value from PRIOR week width


# ---------------- Load option data ----------------
def load_chain():
    log("loading nse_options_bhav (NIFTY)...")
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute(
        "SELECT trade_date, expiry_date, option_type, strike, close, settle_price, contracts "
        "FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date>='2018-12-01'")
    px, ctr, days_set, exp_set = {}, {}, set(), set()
    n = 0
    for td, ed, ot, k, cl, sp, c in cur:
        p = cl if (cl or 0) > 0 else (sp or 0)
        if p <= 0:
            continue
        k = float(k)
        px[(td, ed, ot, k)] = p
        ctr[(td, ed, ot, k)] = c or 0
        days_set.add(td)
        exp_set.add(ed)
        n += 1
    con.close()
    log(f"loaded {n} usable rows, {len(days_set)} trade days, {len(exp_set)} expiries")
    return px, ctr, sorted(days_set), sorted(exp_set)


def classify_monthlies(expiries):
    last = {}
    for e in expiries:
        last[e[:7]] = e  # max expiry per YYYY-MM (sorted input)
    return set(last.values())


def build_day_chain_index(px):
    """day -> expiry -> {'CE': {strike: px}, 'PE': {strike: px}}"""
    idx = defaultdict(lambda: defaultdict(lambda: {"CE": {}, "PE": {}}))
    for (td, ed, ot, k), p in px.items():
        idx[td][ed][ot][k] = p
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
            k = min(common, key=lambda s: abs(ch["CE"][s] - ch["PE"][s]))
            return k + ch["CE"][k] - ch["PE"][k]
    return None


def nearest_liquid_strike(day_idx, ctr, day, exp, ot, target):
    ch = day_idx.get(day, {}).get(exp, {}).get(ot, {})
    cands = sorted(ch, key=lambda k: abs(k - target))
    for k in cands[:6]:
        if ctr.get((day, exp, ot, k), 0) >= MIN_CONTRACTS and ch[k] >= MIN_LEG_PTS:
            return k
    return None


# ---------------- Simulation ----------------
def simulate_cycle(day_idx, days, entry_day, exp, pe_k, ce_k, pe0, ce0, arm):
    """Return daily series [(day, pe, ce, cal_dte)] entry->exp inclusive, ffilled."""
    i0 = days.index(entry_day)
    series = []
    pe, ce = pe0, ce0
    ed = dt.date.fromisoformat(exp)
    for d in days[i0:]:
        if d > exp:
            break
        ch = day_idx.get(d, {}).get(exp, {})
        pe = ch.get("PE", {}).get(pe_k, pe)
        ce = ch.get("CE", {}).get(ce_k, ce)
        series.append((d, pe, ce, (ed - dt.date.fromisoformat(d)).days))
    return series


def eval_exits(series, credit, pe0, ce0, arm, pt, stop, gb):
    peak = 0.0
    for j, (d, pe, ce, cdte) in enumerate(series):
        if j == 0:
            continue
        comb = pe + ce
        profit = credit - comb
        peak = max(peak, profit)
        if stop and (pe >= stop * pe0 or ce >= stop * ce0):
            return d, comb, "STOP", cdte
        if gb and peak >= 0.25 * credit and profit <= 0.5 * peak:
            return d, comb, "GIVEBACK", cdte
        if pt and profit >= pt * credit:
            return d, comb, "PT", cdte
        if cdte <= TIME_EXIT_DTE[arm]:
            return d, comb, "TIME", cdte
    d, pe, ce, cdte = series[-1]
    return d, pe + ce, "EXPIRY", cdte


def main():
    t0 = time.time()
    nifty_daily = fetch_kite_daily(256265, "cache_nifty_daily.csv")
    vix_daily = fetch_kite_daily(264969, "cache_vix_daily.csv")
    log(f"kite caches: nifty {len(nifty_daily)}d, vix {len(vix_daily)}d")
    cpr_terc = weekly_cpr_terciles(nifty_daily)

    px, ctr, days, expiries = load_chain()
    day_idx = build_day_chain_index(px)
    monthlies = classify_monthlies(expiries)
    near_exps = [e for e in expiries if e >= START]

    # spot proxy per day (only days we need: all trading days >= START)
    spots = {}
    for d in days:
        if d >= START:
            s = calc_spot(day_idx, d, expiries)
            if s:
                spots[d] = s
    log(f"spot proxy built for {len(spots)} days; sample {list(spots.items())[:2]}")

    # cycle entry days
    cycles = []  # (arm, entry_day, target_exp)
    for arm in ("W", "M"):
        exps = [e for e in near_exps if (arm == "W" or e in monthlies)]
        prev = None
        for e in exps:
            if prev:
                nxt = [d for d in days if d > prev and d >= START and d in spots]
                if nxt:
                    entry = nxt[0]
                    cdte = (dt.date.fromisoformat(e) - dt.date.fromisoformat(entry)).days
                    lo, hi = DTE_RANGE[arm]
                    if lo <= cdte <= hi and entry < e:
                        cycles.append((arm, entry, e))
            prev = e
    nW = sum(1 for c in cycles if c[0] == "W")
    log(f"cycles: {len(cycles)} (W={nW}, M={len(cycles)-nW})")

    out_path = os.path.join(RESDIR, "g1_cycles.csv")
    fields = ["arm", "entry", "expiry", "p", "pe_k", "ce_k", "pe0", "ce0", "credit",
              "pt", "stop", "gb", "exit_day", "exit_reason", "exit_dte", "days_held",
              "gross", "net", "vix", "cpr_terc", "max_ratio", "peak_profit", "spot0"]
    f = open(out_path, "w", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    skipped, done = 0, 0

    for ci, (arm, entry, exp) in enumerate(cycles):
        spot = spots[entry]
        d0 = dt.date.fromisoformat(entry)
        vix = vix_daily.get(entry, {}).get("close", "")
        terc = cpr_terc.get(d0.isocalendar()[:2], 0)
        for p in P_GRID[arm]:
            pe_k = nearest_liquid_strike(day_idx, ctr, entry, exp, "PE", spot * (1 - p))
            ce_k = nearest_liquid_strike(day_idx, ctr, entry, exp, "CE", spot * (1 + p))
            if not pe_k or not ce_k or ce_k <= pe_k:
                skipped += 1
                continue
            pe0 = day_idx[entry][exp]["PE"][pe_k]
            ce0 = day_idx[entry][exp]["CE"][ce_k]
            credit = pe0 + ce0
            series = simulate_cycle(day_idx, days, entry, exp, pe_k, ce_k, pe0, ce0, arm)
            if len(series) < 2:
                skipped += 1
                continue
            max_ratio = max(max(s[1] / pe0, s[2] / ce0) for s in series[1:])
            peak_prof = max(credit - (s[1] + s[2]) for s in series[1:])
            for pt in PT_GRID:
                for stop in STOP_GRID:
                    for gb in GB_GRID:
                        xd, xcomb, xreason, xdte = eval_exits(series, credit, pe0, ce0, arm, pt, stop, gb)
                        gross = credit - xcomb
                        net = gross - (COST_PCT * (credit + xcomb) + COST_FLAT)
                        held = (dt.date.fromisoformat(xd) - d0).days
                        w.writerow(dict(arm=arm, entry=entry, expiry=exp, p=p,
                                        pe_k=pe_k, ce_k=ce_k, pe0=round(pe0, 2), ce0=round(ce0, 2),
                                        credit=round(credit, 2), pt=pt, stop=stop, gb=int(gb),
                                        exit_day=xd, exit_reason=xreason, exit_dte=xdte,
                                        days_held=held, gross=round(gross, 2), net=round(net, 2),
                                        vix=vix, cpr_terc=terc, max_ratio=round(max_ratio, 2),
                                        peak_profit=round(peak_prof, 2), spot0=round(spot, 1)))
                        done += 1
        if (ci + 1) % 25 == 0:
            f.flush()
            log(f"cycle {ci+1}/{len(cycles)} | rows={done} skips={skipped}")
    f.close()
    log(f"SIM DONE rows={done} skipped_cells={skipped} in {time.time()-t0:.0f}s")

    # ---------------- Aggregate ----------------
    import statistics as st
    rows = list(csv.DictReader(open(out_path)))
    for r in rows:
        r["net"] = float(r["net"]); r["gross"] = float(r["gross"])
        r["credit"] = float(r["credit"]); r["year"] = r["entry"][:4]
    cfgs = defaultdict(list)
    for r in rows:
        cfgs[(r["arm"], r["p"], r["pt"], r["stop"], r["gb"])].append(r)
    rank_path = os.path.join(RESDIR, "g1_ranking.csv")
    rf = open(rank_path, "w", newline="")
    rfields = ["arm", "p", "pt", "stop", "gb", "n", "net_mean", "net_t", "win_rate",
               "worst", "p5", "net_sum", "net_post22_mean", "post22_years_pos",
               "net_2x_cost_mean", "avg_hold"]
    rw = csv.DictWriter(rf, fieldnames=rfields)
    rw.writeheader()
    summary = []
    for key, rs in cfgs.items():
        nets = [r["net"] for r in rs]
        n = len(nets)
        mu = st.mean(nets)
        sd = st.stdev(nets) if n > 1 else 0
        tstat = mu / (sd / n ** 0.5) if sd else 0
        post = [r["net"] for r in rs if r["year"] >= "2022"]
        ypos = sum(1 for y in set(r["year"] for r in rs if r["year"] >= "2022")
                   if st.mean([q["net"] for q in rs if q["year"] == y]) > 0)
        net2x = st.mean([r["gross"] - 2 * (r["gross"] - r["net"]) for r in rs])
        row = dict(arm=key[0], p=key[1], pt=key[2], stop=key[3], gb=key[4], n=n,
                   net_mean=round(mu, 2), net_t=round(tstat, 2),
                   win_rate=round(100 * sum(1 for x in nets if x > 0) / n, 1),
                   worst=round(min(nets), 1), p5=round(sorted(nets)[max(0, n // 20)], 1),
                   net_sum=round(sum(nets), 0),
                   net_post22_mean=round(st.mean(post), 2) if post else "",
                   post22_years_pos=ypos,
                   net_2x_cost_mean=round(net2x, 2),
                   avg_hold=round(st.mean([float(r["days_held"]) for r in rs]), 1))
        rw.writerow(row)
        summary.append(row)
    rf.close()

    log("=== TOP 12 by post-2022 net mean (n>=30) ===")
    top = sorted([s for s in summary if s["n"] >= 30 and s["net_post22_mean"] != ""],
                 key=lambda s: -float(s["net_post22_mean"]))[:12]
    for s in top:
        log(str(s))
    log("=== BASELINES (hold-to-expiry: pt=None stop=None gb=0) ===")
    for s in summary:
        if s["pt"] in ("", "None") and s["stop"] in ("", "None") and str(s["gb"]) == "0":
            log(str(s))
    # gate analysis on reference config M/0.025/0.5/2.0/gb1 and W/0.012/0.5/2.0/gb1
    for arm, p in (("M", "0.025"), ("W", "0.012")):
        ref = [r for r in rows if r["arm"] == arm and r["p"] == p and r["pt"] == "0.5"
               and r["stop"] == "2.0" and r["gb"] == "1"]
        if not ref:
            continue
        log(f"=== GATES on ref {arm} p={p} PT50 stop2.0 gb ===")
        lo = [r["net"] for r in ref if r["vix"] and float(r["vix"]) < 16]
        hi = [r["net"] for r in ref if r["vix"] and float(r["vix"]) >= 16]
        log(f"VIX<16: n={len(lo)} mean={st.mean(lo):.2f} | VIX>=16: n={len(hi)} mean={st.mean(hi) if hi else 0:.2f}")
        for t in (1, 2, 3):
            tt = [r["net"] for r in ref if r["cpr_terc"] == str(t)]
            if tt:
                log(f"CPR tercile {t} (1=narrow): n={len(tt)} mean={st.mean(tt):.2f}")
    log(f"ALL DONE in {time.time()-t0:.0f}s. Outputs: g1_cycles.csv, g1_ranking.csv")


if __name__ == "__main__":
    main()
