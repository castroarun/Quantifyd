#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/121 S4 - the OPTIONS SAMPLE: rupee truth for the three live non-expiry
TimeB cells, the full stop ladder, and the day-level PCR / OI features.

This sample is CONFIRMATION ONLY. It is ~16 days per window; nothing is fitted here.

Cells (from backtest_data/csl_paper_config.json, frozen 2026-08-13):
  NIFTY  Mon  13:00-14:00  SL20   (CSL_TIMEB_NIFTY DTE1)
  SENSEX Wed  10:30-12:00  SL20   (CSL_TIMEB_SENSEX DTE1)
  NIFTY  Fri  10:00-12:00  SL20   (CSL_TIMEB_NIFTY DTE2)

For every recorded day of the right venue (the live weekday is flagged, but ALL
weekdays are replayed so a pooled, larger-n view exists too) we sell the ATM
straddle at the window's start minute and walk the 1-minute chain forward, booking:
  * the live rule (combined SL 20% of credit)
  * a percentage stop ladder  25/20/15/12/10/8/6 %
  * a rupee-cap ladder        2500/2000/1600/1400/1200/1000/800 per lot
  * NOSTOP
and recording MAE, the underlying excursion, and the premium-rise-vs-underlying-move
pairs used to translate a % stop into an underlying move on the long sample.

Costs: NIFTY 0.5 pt/leg-side x 65 + Rs30/leg-side = Rs 250/lot round trip;
       SENSEX 1.0 pt/leg-side x 20 + Rs30/leg-side = Rs 200/lot round trip.

Holiday guard (research/120): reject any day with < 50 distinct underlying prints.
READ-ONLY on options_data.db.
"""
import sqlite3, csv, os, math
from datetime import date

CHAIN = "/home/arun/quantifyd/backtest_data/options_data.db"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

VENUE = {"NIFTY": dict(lot=65, step=50, slip=0.5),
         "SENSEX": dict(lot=20, step=100, slip=1.0)}
CHG = 30.0
LEG_SIDES = 4
EXCLUDE_DAYS = {"2026-08-21"}
SESS_END = "15:20"

CELLS = [
    ("MON_NIFTY_DTE1", "NIFTY", "13:00", "14:00", 0),
    ("WED_SENSEX_DTE1", "SENSEX", "10:30", "12:00", 2),
    ("FRI_NIFTY_DTE2", "NIFTY", "10:00", "12:00", 4),
]

PCT_STOPS = [25, 20, 15, 12, 10, 8, 6]
RS_CAPS = [2500, 2000, 1600, 1400, 1200, 1000, 800]

FEATCOLS = ["cpr_today", "cpr_prev", "wcpr_this", "wcpr_prev", "gap_pct", "gap_abs",
            "pdr_pct", "pdr_rel", "atr14_pct", "ret_prev", "vix_open", "vix_prevclose",
            "vix_chg_oc_pct", "vix_chg_oc_pts", "vix_chg_cc_pct", "vix_chg_cc_pts"]


def hm2m(h):
    return int(h[:2]) * 60 + int(h[3:5])


def m2hm(m):
    return "%02d:%02d" % (m // 60, m % 60)


def load_features(sym):
    p = os.path.join(RES, "daily_features_%s.csv" % sym)
    out = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            out[r["day"]] = r
    return out


def rec_days(c, sym):
    q = ("SELECT DISTINCT substr(snapshot_time,1,10) d FROM download_log "
         "WHERE symbol=? ORDER BY d")
    return [r[0] for r in c.execute(q, (sym,)) if r[0] not in EXCLUDE_DAYS]


def load_day(c, sym, day):
    q = ("SELECT snapshot_time, expiry_date, strike, instrument_type, ltp, underlying_spot, "
         "oi, volume FROM option_chain WHERE symbol=? AND snapshot_time>=? AND snapshot_time<? "
         "AND ltp IS NOT NULL")
    rows = c.execute(q, (sym, day, day + "z")).fetchall()
    if not rows:
        return None
    exps = sorted({e for (_, e, _, _, _, _, _, _) in rows if e and e >= day})
    if not exps:
        return None
    fexp = exps[0]
    spot, chain, oi = {}, {}, {}
    for st, e, k, it, ltp, sp, o, vol in rows:
        mi = hm2m(st[11:16])
        if sp and mi not in spot:
            spot[mi] = sp
        if e != fexp:
            continue
        chain.setdefault(mi, {}).setdefault(k, {})[it] = ltp
        oi.setdefault(mi, {}).setdefault(k, {})[it] = (o or 0, vol or 0)
    if len(set(spot.values())) < 50:
        return None          # frozen chain = exchange holiday (research/120)
    ch2 = {}
    for mi, ks in chain.items():
        ch2[mi] = {k: (v["CE"], v["PE"]) for k, v in ks.items() if "CE" in v and "PE" in v}
    return fexp, spot, ch2, oi


def oi_features(oi_at_min, spot_px, step):
    """Day-level option-book features at one minute. Returns dict."""
    if not oi_at_min:
        return {}
    atm = round(spot_px / step) * step
    ce_oi = {k: v.get("CE", (0, 0))[0] for k, v in oi_at_min.items()}
    pe_oi = {k: v.get("PE", (0, 0))[0] for k, v in oi_at_min.items()}
    ce_vol = {k: v.get("CE", (0, 0))[1] for k, v in oi_at_min.items()}
    pe_vol = {k: v.get("PE", (0, 0))[1] for k, v in oi_at_min.items()}
    tot_ce, tot_pe = sum(ce_oi.values()), sum(pe_oi.values())
    tv_ce, tv_pe = sum(ce_vol.values()), sum(pe_vol.values())
    near = [k for k in oi_at_min if abs(k - atm) <= 5 * step]
    a_ce = sum(ce_oi.get(k, 0) for k in near); a_pe = sum(pe_oi.get(k, 0) for k in near)
    wings = [k for k in oi_at_min if abs(k - atm) > 5 * step]
    w_ce = sum(ce_oi.get(k, 0) for k in wings); w_pe = sum(pe_oi.get(k, 0) for k in wings)
    f = {}
    f["pcr_oi_all"] = round(tot_pe / tot_ce, 4) if tot_ce > 0 else ""
    f["pcr_oi_atm"] = round(a_pe / a_ce, 4) if a_ce > 0 else ""
    f["pcr_vol_all"] = round(tv_pe / tv_ce, 4) if tv_ce > 0 else ""
    f["oi_total"] = tot_ce + tot_pe
    f["oi_atm"] = a_ce + a_pe
    f["oi_wing"] = w_ce + w_pe
    f["oi_atm_share"] = round((a_ce + a_pe) / (tot_ce + tot_pe), 4) if (tot_ce + tot_pe) > 0 else ""
    ce_wall = max(ce_oi, key=lambda k: ce_oi[k]) if ce_oi else None
    pe_wall = max(pe_oi, key=lambda k: pe_oi[k]) if pe_oi else None
    f["dist_ce_wall_bp"] = round((ce_wall - spot_px) / spot_px * 1e4, 1) if ce_wall else ""
    f["dist_pe_wall_bp"] = round((spot_px - pe_wall) / spot_px * 1e4, 1) if pe_wall else ""
    if ce_wall and pe_wall:
        f["wall_span_bp"] = round((ce_wall - pe_wall) / spot_px * 1e4, 1)
    else:
        f["wall_span_bp"] = ""
    # max pain: strike minimising total payoff to option BUYERS
    ks = sorted(oi_at_min)
    best, bestv = None, None
    for s in ks:
        pay = sum(ce_oi.get(k, 0) * max(0.0, s - k) for k in ks) + \
              sum(pe_oi.get(k, 0) * max(0.0, k - s) for k in ks)
        if bestv is None or pay < bestv:
            best, bestv = s, pay
    f["maxpain"] = best
    f["maxpain_drift_bp"] = round((spot_px - best) / spot_px * 1e4, 1) if best else ""
    return f


def replay(chain, spot, K, m0, m1):
    """Walk the window minute by minute; return the path summary."""
    if m0 not in chain or K not in chain[m0]:
        return None
    ce0, pe0 = chain[m0][K]
    credit = ce0 + pe0
    if credit <= 0:
        return None
    s0 = spot.get(m0)
    path = []          # (minute, combined, underlying excursion bp)
    for mi in range(m0 + 1, m1 + 1):
        d = chain.get(mi)
        if not d or K not in d:
            continue
        ce, pe = d[K]
        sp = spot.get(mi)
        exc = abs(sp - s0) / s0 * 1e4 if (sp and s0) else 0.0
        path.append((mi, ce + pe, exc))
    if not path:
        return None
    return dict(credit=credit, s0=s0, path=path)


def book(credit, path, lot, cost, stop_comb):
    """Exit at the first minute combined >= stop_comb, else at the last minute."""
    for mi, comb, _e in path:
        if stop_comb is not None and comb >= stop_comb:
            return (credit - comb) * lot - cost, mi, comb, "SL"
    mi, comb, _e = path[-1]
    return (credit - comb) * lot - cost, mi, comb, "TIME"


HEAD = (["cell", "venue", "day", "dow", "is_live_dow", "expiry", "dte_cal", "start", "end",
         "strike", "spot0", "credit", "credit_pct_spot", "lot", "cost",
         "mae_rs", "und_exc_bp", "und_net_bp", "prem_rise_max_pct",
         "net_NOSTOP", "exit_NOSTOP"]
        + ["net_SL%d" % s for s in PCT_STOPS] + ["fired_SL%d" % s for s in PCT_STOPS]
        + ["net_RC%d" % s for s in RS_CAPS] + ["fired_RC%d" % s for s in RS_CAPS]
        + ["pre_move_bp", "pre_range_bp"]
        + ["pcr_oi_all", "pcr_oi_atm", "pcr_vol_all", "oi_atm_share",
           "dist_ce_wall_bp", "dist_pe_wall_bp", "wall_span_bp", "maxpain_drift_bp",
           "d_pcr_oi_all", "d_oi_total_pct", "d_oi_atm_pct", "d_oi_wing_pct"]
        + FEATCOLS)


def main():
    os.makedirs(RES, exist_ok=True)
    c = sqlite3.connect("file:%s?mode=ro" % CHAIN, uri=True)
    feats = {"NIFTY": load_features("NIFTY50"), "SENSEX": load_features("SENSEX")}
    m_end = hm2m(SESS_END)

    fo = open(os.path.join(RES, "options_sample.csv"), "w", newline="")
    w = csv.DictWriter(fo, fieldnames=HEAD); w.writeheader()
    fc = open(os.path.join(RES, "prem_vs_move.csv"), "w", newline="")
    wc = csv.writer(fc); wc.writerow(["cell", "venue", "day", "und_exc_bp", "prem_rise_pct"])

    for venue in ("NIFTY", "SENSEX"):
        V = VENUE[venue]
        lot, step, slip = V["lot"], V["step"], V["slip"]
        cost = LEG_SIDES * slip * lot + LEG_SIDES * CHG
        days = rec_days(c, venue)
        cache = {}
        prev_oi = {}          # day -> (feature dict at 15:15, fexp)
        for day in days:
            d = load_day(c, venue, day)
            if not d:
                print("  %s %s SKIP (holiday/frozen or no data)" % (venue, day), flush=True)
                continue
            cache[day] = d
        ordered = [x for x in days if x in cache]
        for i, day in enumerate(ordered):
            fexp, spot, chain, oi = cache[day]
            dte_cal = (date.fromisoformat(fexp) - date.fromisoformat(day)).days
            dow = date.fromisoformat(day).weekday()
            fr = feats[venue].get(day, {})
            mins = sorted(chain)
            dayopen_m = min(mins) if mins else None
            # previous recorded day's END-OF-DAY book state, same expiry -> overnight deltas
            prev = None
            if i >= 1:
                pd_ = ordered[i - 1]
                pexp, pspot, pchain, poi = cache[pd_]
                if pexp == fexp:
                    pm = [m for m in sorted(poi) if m <= hm2m("15:20")]
                    if pm:
                        mlast = pm[-1]
                        psp = pspot.get(mlast)
                        if psp:
                            prev = oi_features(poi[mlast], psp, step)
            for cname, cven, s, e, live_dow in CELLS:
                if cven != venue:
                    continue
                m0n, m1 = hm2m(s), hm2m(e)
                if m0n not in chain:
                    cand = [m for m in mins if m0n <= m <= m0n + 10]
                    if not cand:
                        continue
                    m0 = min(cand)
                else:
                    m0 = m0n
                sp0 = spot.get(m0)
                if not sp0:
                    continue
                K = round(sp0 / step) * step
                r = replay(chain, spot, K, m0, min(m1, m_end))
                if not r:
                    continue
                credit, path = r["credit"], r["path"]
                mae_pts = max(0.0, max(cb for _m, cb, _x in path) - credit)
                und_exc = max(x for _m, _cb, x in path)
                und_net = (spot.get(path[-1][0], sp0) - sp0) / sp0 * 1e4
                row = dict(cell=cname, venue=venue, day=day, dow=dow,
                           is_live_dow=1 if dow == live_dow else 0,
                           expiry=fexp, dte_cal=dte_cal, start=m2hm(m0), end=m2hm(path[-1][0]),
                           strike=K, spot0=round(sp0, 2), credit=round(credit, 2),
                           credit_pct_spot=round(credit / sp0 * 100, 3), lot=lot, cost=round(cost),
                           mae_rs=round(mae_pts * lot), und_exc_bp=round(und_exc, 1),
                           und_net_bp=round(und_net, 1),
                           prem_rise_max_pct=round(mae_pts / credit * 100, 2))
                n0, x0, _, _ = book(credit, path, lot, cost, None)
                row["net_NOSTOP"] = round(n0); row["exit_NOSTOP"] = m2hm(x0)
                for st in PCT_STOPS:
                    nv, xm, xc, why = book(credit, path, lot, cost, credit * (1 + st / 100.0))
                    row["net_SL%d" % st] = round(nv); row["fired_SL%d" % st] = 1 if why == "SL" else 0
                for rc in RS_CAPS:
                    nv, xm, xc, why = book(credit, path, lot, cost, credit + rc / lot)
                    row["net_RC%d" % rc] = round(nv); row["fired_RC%d" % rc] = 1 if why == "SL" else 0
                # causal intraday context at the window start
                pre = [m for m in mins if m < m0]
                if pre and dayopen_m is not None:
                    o_px = spot.get(dayopen_m)
                    ps = [spot[m] for m in pre if m in spot]
                    row["pre_move_bp"] = round(abs(sp0 - o_px) / o_px * 1e4, 1) if o_px else ""
                    row["pre_range_bp"] = round((max(ps) - min(ps)) / sp0 * 1e4, 1) if ps else ""
                else:
                    row["pre_move_bp"] = ""; row["pre_range_bp"] = ""
                # option-book features at the entry minute
                oim = oi.get(m0) or oi.get(min(oi, key=lambda m: abs(m - m0))) if oi else None
                of = oi_features(oim, sp0, step) if oim else {}
                for k in ("pcr_oi_all", "pcr_oi_atm", "pcr_vol_all", "oi_atm_share",
                          "dist_ce_wall_bp", "dist_pe_wall_bp", "wall_span_bp", "maxpain_drift_bp"):
                    row[k] = of.get(k, "")
                if prev and of:
                    try:
                        row["d_pcr_oi_all"] = round(float(of["pcr_oi_all"]) - float(prev["pcr_oi_all"]), 4)
                    except Exception:
                        row["d_pcr_oi_all"] = ""
                    for a, b in (("d_oi_total_pct", "oi_total"), ("d_oi_atm_pct", "oi_atm"),
                                 ("d_oi_wing_pct", "oi_wing")):
                        try:
                            row[a] = round((of[b] - prev[b]) / prev[b] * 100.0, 2) if prev[b] else ""
                        except Exception:
                            row[a] = ""
                else:
                    for a in ("d_pcr_oi_all", "d_oi_total_pct", "d_oi_atm_pct", "d_oi_wing_pct"):
                        row[a] = ""
                for k in FEATCOLS:
                    row[k] = fr.get(k, "")
                w.writerow(row)
                # premium-rise vs underlying-move pairs (for the long-sample translation)
                run = 0.0
                for _m, cb, x in path:
                    rise = (cb - credit) / credit * 100.0
                    if rise > run:
                        run = rise
                        wc.writerow([cname, venue, day, round(x, 1), round(rise, 2)])
            fo.flush(); fc.flush()
        print("%s: %d usable days" % (venue, len(ordered)), flush=True)
    fo.close(); fc.close()
    print("DONE")


if __name__ == "__main__":
    main()
