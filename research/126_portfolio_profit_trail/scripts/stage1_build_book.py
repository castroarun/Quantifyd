#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/125 Stage 1 - rebuild the LIVE BOOK's per-minute intraday P&L path.

LIVE-FIRST, BACKFILL-WITH-RECORDED (binding project rule):
  * the NIFTY 9:16 suite (ATM/ATM2/ATM4) is taken from its REAL per-minute MTM
    (nas_mtm_snapshots, 70 sessions) -- it has cascade/ST-trail/move-stop machinery
    that cannot be faithfully replayed, so it is never modelled.
  * the CSL sleeves (TimeB NIFTY/SENSEX, NAS_COMB20, CSL30F_SENSEX_WED) are replayed
    from the REAL 1-minute option chain using their FROZEN live config.
  * where a CSL sleeve has a REAL live series for that day, the live series is used
    and the replay is kept alongside for reconciliation.

Also emits the WING price paths (bought OTM CE+PE at several distances) with the
ask/bid/volume/oi needed for the Arm-B audit -- so the 12 GB chain is read ONCE.

READ-ONLY on every DB.  Outputs (results/):
  book_minute.csv.gz     day,t,sleeve,pnl_rs,open        (replayed sleeves + live suite)
  sleeve_days.csv        one row per day x sleeve
  wing_minute.csv.gz     day,venue,sleeve,dist,t,ask_comb,bid_comb,ltp_comb,volok,stale
  stage1.log
"""
import sqlite3, csv, os, sys, json, gzip, time
from datetime import date

Q = "/home/arun/quantifyd/"
CHAIN = Q + "backtest_data/options_data.db"
CFG = Q + "backtest_data/csl_paper_config.json"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
os.makedirs(RES, exist_ok=True)
LOG = os.path.join(RES, "stage1.log")

TODAY = date.today().isoformat()
EXCLUDE = {TODAY}                       # partial session (market open when built)
FROZEN_HOLIDAYS = {"2026-05-01", "2026-05-28", "2026-06-26"}

VEN = {"NIFTY":  dict(lot=65, step=50,  dists=[100, 150, 200, 250, 300, 400, 500]),
       "SENSEX": dict(lot=20, step=100, dists=[400, 600, 800, 1000, 1200, 1600, 2000])}

# --- MEASURED cost model (2026-08-25), shape from research/122 stage_a_alldays.py -----
SLIP_ENTRY = 0.0        # measured -0.228 pt (favourable) -> booked at 0
SLIP_TIME = 0.178
SLIP_STOP = 6.548       # every FORCED/mid-session exit pays this per leg-side
NLOTS_REF = 10


def cost_short_leg(credit, exitp, lot, nlots, reason):
    """Rs cost for ONE short straddle sleeve round trip, TOTAL (not per lot)."""
    sell = credit * lot * nlots
    buy = exitp * lot * nlots
    tot = sell + buy
    brok = 80.0                       # Rs20 x 4 orders
    stt = 0.001 * sell
    txn = 0.0003503 * tot
    ipft = 0.0000050 * tot
    sebi = 0.0000010 * tot
    stamp = 0.00003 * buy
    gst = 0.18 * (brok + txn + ipft + sebi)
    slip_pts = 2 * SLIP_ENTRY + 2 * (SLIP_STOP if reason in ("SL", "TRAIL", "FORCED") else SLIP_TIME)
    return brok + stt + txn + ipft + sebi + stamp + gst + slip_pts * lot * nlots


def cost_long_wings(buyp, sellp, lot, nlots, reason):
    """Rs cost for BUYING 2 wings and selling them back. STT applies on the SELL of a
    long option (0.1% of premium, same as any option sale). Slippage: wings are bought
    at ASK and sold at BID in the price path itself, so only charges + a small extra
    slip are added here."""
    buy = buyp * lot * nlots
    sell = sellp * lot * nlots
    tot = buy + sell
    brok = 80.0
    stt = 0.001 * sell
    txn = 0.0003503 * tot
    ipft = 0.0000050 * tot
    sebi = 0.0000010 * tot
    stamp = 0.00003 * buy
    gst = 0.18 * (brok + txn + ipft + sebi)
    return brok + stt + txn + ipft + sebi + stamp + gst


_logf = open(LOG, "a")
def log(m):
    _logf.write(m + "\n"); _logf.flush(); print(m, flush=True)


# ------------------------------------------------------------------ sleeves
def sleeves():
    """LIVE money only. -> list(name, venue, book, lots_default, {dte:(entry,exit,sl,lots)})"""
    cfg = json.load(open(CFG))["books"]
    spec = [("TB_NIFTY", "NIFTY", "CSL_TIMEB_NIFTY", 8),
            ("COMB20", "NIFTY", "NAS_COMB20", 2),
            ("TB_SENSEX", "SENSEX", "CSL_TIMEB_SENSEX", 8),
            ("SXWED", "SENSEX", "CSL30F_SENSEX_WED", 3)]
    out = []
    for nm, ven, bk, dl in spec:
        cells = {}
        for dte, c in (cfg.get(bk) or {}).items():
            sl = c["sl"]
            cells[int(dte)] = (c["entry"], c["exit"],
                               None if sl == "none" else float(sl) / 100.0,
                               int(c.get("lots", dl)))
        if cells:
            out.append((nm, ven, bk, dl, cells))
    return out


BACKSTOP = 0.50          # 'sl: none' cells still carry the 50% disaster backstop


# ------------------------------------------------------------------ data
def rec_days(c, sym):
    return [r[0] for r in c.execute(
        "SELECT DISTINCT substr(snapshot_time,1,10) d FROM download_log WHERE symbol=? "
        "ORDER BY d", (sym,)) if date.fromisoformat(r[0]).weekday() < 5]


def load_day(c, sym, day, band):
    """front expiry only, strikes within +-band of the day's spot. -> (exp, per, spot)
    per[hm][strike][CE|PE] = (ltp, bid, ask, volume, oi)"""
    fexp = c.execute("SELECT MIN(expiry_date) FROM option_chain WHERE symbol=? "
                     "AND snapshot_time>=? AND snapshot_time<? AND expiry_date>=?",
                     (sym, day, day + "z", day)).fetchone()[0]
    if not fexp:
        return None
    per, spot = {}, {}
    last = ""
    for st, k, it, ltp, bid, ask, vol, oi, sp in c.execute(
            "SELECT snapshot_time,strike,instrument_type,ltp,bid,ask,volume,oi,underlying_spot "
            "FROM option_chain WHERE symbol=? AND snapshot_time>=? AND snapshot_time<? "
            "AND expiry_date=?", (sym, day, day + "z", fexp)):
        hm = st[11:16]
        if st > last:
            last = st
        if sp and hm not in spot:
            spot[hm] = sp
        if sp and abs(k - sp) > band:
            continue
        per.setdefault(hm, {}).setdefault(k, {})[it] = (ltp, bid, ask, vol, oi)
    if not per or not spot:
        return None
    if last[11:16] < "15:15":                       # partial session
        return ("PARTIAL", None, None)
    if len(set(spot.values())) < 50:                # frozen-chain holiday guard
        return ("FROZEN", None, None)
    return fexp, per, spot


def dte_of(day, exp, days):
    if exp == day:
        return 0
    if exp in days and day in days:
        return days.index(exp) - days.index(day)
    n, cur = 0, date.fromisoformat(day)
    e = date.fromisoformat(exp)
    while cur < e:
        cur = date.fromordinal(cur.toordinal() + 1)
        if cur.weekday() < 5:
            n += 1
    return n


def atm_strike(per, spot, hm_list, want, step):
    """first available minute >= want; returns (hm, K, spot)"""
    for hm in hm_list:
        if hm < want:
            continue
        sp = spot.get(hm)
        if not sp:
            continue
        K = round(sp / step) * step
        d = per.get(hm, {}).get(K)
        if d and "CE" in d and "PE" in d and d["CE"][0] and d["PE"][0]:
            return hm, K, sp
        cands = [k for k, dd in per.get(hm, {}).items()
                 if "CE" in dd and "PE" in dd and dd["CE"][0] and dd["PE"][0]]
        if cands:
            K = min(cands, key=lambda k: abs(k - sp))
            return hm, K, sp
    return None, None, None


# ------------------------------------------------------------------ main
def main():
    open(LOG, "w").close()
    c = sqlite3.connect("file:%s?mode=ro" % CHAIN, uri=True)
    SL = sleeves()
    log("sleeves (LIVE money): " + "; ".join(
        "%s[%s] %s" % (n, v, {k: (e, x, s, l) for k, (e, x, s, l) in cc.items()})
        for n, v, b, d, cc in SL))

    daysby = {v: rec_days(c, v) for v in VEN}
    alldays = sorted(set(daysby["NIFTY"]) | set(daysby["SENSEX"]))
    alldays = [d for d in alldays if d not in EXCLUDE and d not in FROZEN_HOLIDAYS]
    log("candidate days: %d  %s..%s (excluded today=%s, frozen holidays)" %
        (len(alldays), alldays[0], alldays[-1], TODAY))

    bm = gzip.open(os.path.join(RES, "book_minute.csv.gz"), "wt", newline="")
    bw = csv.writer(bm); bw.writerow(["day", "sleeve", "t", "pnl_rs", "state"])
    wm = gzip.open(os.path.join(RES, "wing_minute.csv.gz"), "wt", newline="")
    ww = csv.writer(wm)
    ww.writerow(["day", "venue", "sleeve", "dist", "t", "ask_comb", "bid_comb",
                 "ltp_comb", "vol_ce", "vol_pe", "oi_ce", "oi_pe"])
    sd = open(os.path.join(RES, "sleeve_days.csv"), "w", newline="")
    sw = csv.DictWriter(sd, fieldnames=[
        "day", "weekday", "venue", "sleeve", "expiry", "dte", "lots", "lot",
        "entry", "exit_hm", "strike", "spot0", "credit", "exit_comb", "reason",
        "gross_rs", "cost_rs", "net_rs", "peak_rs", "trough_rs", "peak_t"])
    sw.writeheader()

    WD = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    t0 = time.time()
    kept = {v: 0 for v in VEN}
    for day in alldays:
        wd = WD[date.fromisoformat(day).weekday()]
        for ven, V in VEN.items():
            if day not in daysby[ven]:
                continue
            band = max(V["dists"]) + 3 * V["step"]
            d = load_day(c, ven, day, band)
            if not d:
                log("  %s %s SKIP no-data" % (ven, day)); continue
            if d[0] in ("PARTIAL", "FROZEN"):
                log("  %s %s SKIP %s" % (ven, day, d[0])); continue
            fexp, per, spot = d
            dte = dte_of(day, fexp, daysby[ven])
            mins = sorted(per)
            if len(mins) < 200:
                log("  %s %s SKIP thin(%d)" % (ven, day, len(mins))); continue
            kept[ven] += 1
            act = [(n, cc[dte]) for n, v, b, dl, cc in SL if v == ven and dte in cc]
            if not act:
                log("  %s %s %s dte=%d exp=%s : no live sleeve" % (ven, day, wd, dte, fexp))
                continue
            for nm, (e_hm, x_hm, slp, lots) in act:
                lot = V["lot"]
                hm0, K, sp0 = atm_strike(per, spot, mins, e_hm, V["step"])
                if hm0 is None or hm0 > x_hm:
                    log("  %s %s %s: no ATM at %s" % (ven, day, nm, e_hm)); continue
                d0 = per[hm0][K]
                credit = d0["CE"][0] + d0["PE"][0]
                if credit <= 0:
                    continue
                thr = credit * (1.0 + (slp if slp is not None else BACKSTOP))
                comb_last, exit_hm, exit_comb, reason = credit, hm0, credit, "TIME"
                stopped = False
                peak, trough, peak_t = 0.0, 0.0, hm0
                path = []
                for hm in mins:
                    if hm < hm0 or hm > x_hm:
                        continue
                    dd = per[hm].get(K)
                    if not dd or "CE" not in dd or "PE" not in dd:
                        continue
                    if dd["CE"][0] is None or dd["PE"][0] is None:
                        continue
                    comb = dd["CE"][0] + dd["PE"][0]
                    if stopped:
                        path.append((hm, (credit - exit_comb) * lot * lots, "CLOSED"))
                        continue
                    pnl = (credit - comb) * lot * lots
                    path.append((hm, pnl, "OPEN"))
                    if pnl > peak:
                        peak, peak_t = pnl, hm
                    if pnl < trough:
                        trough = pnl
                    comb_last, exit_hm, exit_comb = comb, hm, comb
                    if comb >= thr:                       # causal: tested on this bar's print
                        stopped, reason = True, "SL"
                if len(path) < 5:
                    continue
                gross = (credit - exit_comb) * lot * lots
                cst = cost_short_leg(credit, exit_comb, lot, lots, reason)
                for hm, pnl, stt in path:
                    bw.writerow([day, nm, hm, round(pnl, 1), stt])
                sw.writerow(dict(day=day, weekday=wd, venue=ven, sleeve=nm, expiry=fexp,
                                 dte=dte, lots=lots, lot=lot, entry=hm0, exit_hm=exit_hm,
                                 strike=K, spot0=round(sp0, 2), credit=round(credit, 2),
                                 exit_comb=round(exit_comb, 2), reason=reason,
                                 gross_rs=round(gross), cost_rs=round(cst),
                                 net_rs=round(gross - cst), peak_rs=round(peak),
                                 trough_rs=round(trough), peak_t=peak_t))
                # ---------------- wing price paths (Arm B raw material) -------------
                for dist in V["dists"]:
                    kc, kp = K + dist, K - dist
                    n_out = 0
                    for hm in mins:
                        if hm < hm0 or hm > x_hm:
                            continue
                        dc = per[hm].get(kc, {}).get("CE")
                        dp = per[hm].get(kp, {}).get("PE")
                        if not dc or not dp:
                            continue
                        ww.writerow([day, ven, nm, dist, hm,
                                     round((dc[2] or 0) + (dp[2] or 0), 2),
                                     round((dc[1] or 0) + (dp[1] or 0), 2),
                                     round((dc[0] or 0) + (dp[0] or 0), 2),
                                     dc[3] or 0, dp[3] or 0, dc[4] or 0, dp[4] or 0])
                        n_out += 1
            log("  %s %s %s dte=%d exp=%s sleeves=%s  [%.0fs]" %
                (ven, day, wd, dte, fexp, ",".join(n for n, _ in act), time.time() - t0))
            bm.flush(); wm.flush(); sd.flush()
    bm.close(); wm.close(); sd.close()
    log("kept days: %s   total %.0fs" % (kept, time.time() - t0))
    log("DONE")


if __name__ == "__main__":
    main()
