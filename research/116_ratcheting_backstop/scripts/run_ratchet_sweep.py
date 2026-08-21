#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/116 - Static vs ratcheting backstop on the live short-straddle books.

ONE varied axis: the defence (the stop level and how/whether it moves).
Everything else frozen to the live constructions in backtest_data/csl_paper_config.json:
venue, per-DTE entry/exit window, per-DTE combined-SL %, ATM strike, front expiry.

Data: options_data.db :: option_chain, 1-minute snapshots, READ-ONLY.
Costs: 0.5 pt/leg-side NIFTY, 1.0 SENSEX, plus Rs30/leg-side/lot. 4 leg-sides per straddle.

Causal discipline: at every bar we test the stop level carried in from PRIOR bars, then
update the ratchet/peak with this bar's data. No bar can trigger a stop it just set.
"""
import sqlite3, csv, os, sys, json
from datetime import date

Q = "/home/arun/quantifyd/"
CHAIN = Q + "backtest_data/options_data.db"
CFG = Q + "backtest_data/csl_paper_config.json"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
DETAIL = os.path.join(RES, "ratchet_detail.csv")
LOG = os.path.join(RES, "run.log")

EXCLUDE_DAYS = {"2026-08-21"}      # today - market still open, partial series

VENUE = {
    "NIFTY":  dict(lot=65,  step=50,  slip=0.5),
    "SENSEX": dict(lot=20,  step=100, slip=1.0),
}
CHG = 30.0        # Rs per leg-side per lot
BACKSTOP = 1.5    # the 50% disaster backstop that exists even on "sl: none" cells


def log(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
    print(m, flush=True)


# ---------------------------------------------------------------- constructions
def constructions():
    """Frozen live cells: (name, venue, book, {dte: (entry_hm, exit_hm, sl_or_None)})."""
    cfg = json.load(open(CFG))["books"]
    out = []
    for name, venue, book in [
        ("COMB_NIFTY",  "NIFTY",  "NAS_COMB20"),
        ("TIMEB_NIFTY", "NIFTY",  "CSL_TIMEB_NIFTY"),
        ("COMB_SENSEX", "SENSEX", "CSL30F_SENSEX"),
        ("TIMEB_SENSEX", "SENSEX", "CSL_TIMEB_SENSEX"),
    ]:
        cells = {}
        for dte, c in cfg[book].items():
            sl = c["sl"]
            sl = None if sl == "none" else float(sl) / 100.0
            cells[int(dte)] = (c["entry"], c["exit"], sl)
        out.append((name, venue, book, cells))
    return out


# ---------------------------------------------------------------- data
def trading_days(c, sym):
    return [r[0] for r in c.execute(
        "SELECT DISTINCT substr(snapshot_time,1,10) d FROM download_log "
        "WHERE symbol=? ORDER BY d", (sym,))]


def dte_of(day, expiry, days):
    """Trading days to expiry, using the recorded calendar where possible."""
    if expiry == day:
        return 0
    if expiry in days:
        return days.index(expiry) - days.index(day)
    d0, d1 = date.fromisoformat(day), date.fromisoformat(expiry)
    n, cur = 0, d0
    while cur < d1:
        cur = date.fromordinal(cur.toordinal() + 1)
        if cur.weekday() < 5:
            n += 1
    return n


def load_day(c, sym, day):
    """-> (expiry, {hm: {strike: {CE,PE}}}, {hm: spot}) for the front expiry only."""
    fexp = c.execute(
        "SELECT MIN(expiry_date) FROM option_chain WHERE symbol=? AND snapshot_time LIKE ? "
        "AND expiry_date>=?", (sym, day + "%", day)).fetchone()[0]
    if not fexp:
        return None
    per, spot = {}, {}
    for stime, k, it, ltp, sp in c.execute(
            "SELECT snapshot_time, strike, instrument_type, ltp, underlying_spot "
            "FROM option_chain WHERE symbol=? AND snapshot_time LIKE ? AND expiry_date=? "
            "AND ltp IS NOT NULL", (sym, day + "%", fexp)):
        hm = stime[11:16]
        per.setdefault(hm, {}).setdefault(k, {})[it] = ltp
        if sp and hm not in spot:
            spot[hm] = sp
    return (fexp, per, spot) if per else None


def build_series(per, spot, step, entry_hm, exit_hm):
    """ATM straddle minute series inside the window -> (K, credit, entry_hm, [(hm, comb)])."""
    mins = sorted(h for h in per if entry_hm <= h <= exit_hm)
    if not mins:
        return None
    e_hm = mins[0]
    sp = spot.get(e_hm) or next((spot[h] for h in sorted(spot) if h >= e_hm), None)
    if not sp:
        return None
    K = round(sp / step) * step
    d0 = per[e_hm].get(K, {})
    if "CE" not in d0 or "PE" not in d0:
        cands = [k for k, d in per[e_hm].items() if "CE" in d and "PE" in d]
        if not cands:
            return None
        K = min(cands, key=lambda k: abs(k - sp))
        d0 = per[e_hm][K]
    credit = d0["CE"] + d0["PE"]
    ser = []
    for hm in mins:
        d = per[hm].get(K, {})
        if "CE" in d and "PE" in d:
            ser.append((hm, d["CE"] + d["PE"]))
    if len(ser) < 10 or credit <= 0:
        return None
    return K, credit, e_hm, ser


# ---------------------------------------------------------------- the defence variants
def variants():
    """name -> cfg. THE one varied axis."""
    v = [("NO_DEFENCE", {"none": True}),
         ("STATIC", {})]
    for t in (0.50, 0.60, 0.70):
        v.append(("BE_CLAMP_%d" % (t * 100), {"be": t}))
    for k in (1.3, 1.5, 1.75, 2.0, 2.5):
        v.append(("RATCHET_K%s" % ("%.2f" % k).rstrip("0").rstrip("."), {"k": k}))
    for g in (30, 50):
        v.append(("GIVEBACK_%d" % g, {"gb": g / 100.0, "arm": 0.15}))
    for r in (1000, 2000, 3000):
        v.append(("RS_GB_%d" % r, {"rs": r}))
    v.append(("TIME_RATCHET_MID", {"k": 1.5, "after_mid": True}))
    v.append(("HYBRID_BE60_GB50", {"be": 0.60, "gb": 0.50, "arm": 0.15}))
    return v


def replay(cfg, credit, sl, ser, lot):
    """-> (exit_hm, exit_comb, reason, peak_gross_rs_per_lot_up_to_exit)."""
    static_stop = credit * min(1 + sl, BACKSTOP) if sl is not None else credit * BACKSTOP
    stop = None if cfg.get("none") else static_stop
    mid_hm = ser[len(ser) // 2][0]
    peak_pts = 0.0          # peak open profit in points, from PRIOR bars only
    armed_pct = armed_rs = False
    for hm, comb in ser:
        # ---- 1. test the defence carried in from prior bars (causal) ----
        if stop is not None and comb >= stop:
            return hm, comb, "STOP", peak_pts * lot
        if cfg.get("gb") and armed_pct and (credit - comb) <= peak_pts * (1 - cfg["gb"]):
            return hm, comb, "GIVEBACK", peak_pts * lot
        if cfg.get("rs") and armed_rs and (credit - comb) * lot <= peak_pts * lot - cfg["rs"]:
            return hm, comb, "RS_GIVEBACK", peak_pts * lot
        # ---- 2. update state with THIS bar ----
        pnl = credit - comb
        if pnl > peak_pts:
            peak_pts = pnl
        if cfg.get("gb") and peak_pts >= cfg["arm"] * credit:
            armed_pct = True
        if cfg.get("rs") and peak_pts * lot >= cfg["rs"]:
            armed_rs = True
        if cfg.get("be") and comb <= cfg["be"] * credit and stop is not None:
            stop = min(stop, credit)
        if cfg.get("k") and stop is not None:
            if (not cfg.get("after_mid")) or hm >= mid_hm:
                stop = min(stop, cfg["k"] * comb)
    hm, comb = ser[-1]
    return hm, comb, "WINDOW", peak_pts * lot


# ---------------------------------------------------------------- main
def main():
    os.makedirs(RES, exist_ok=True)
    open(LOG, "w").close()
    c = sqlite3.connect("file:%s?mode=ro" % CHAIN, uri=True)
    cons = constructions()
    vs = variants()
    log("constructions: %s" % ", ".join(n for n, _, _, _ in cons))
    log("variants (%d): %s" % (len(vs), ", ".join(n for n, _ in vs)))

    f = open(DETAIL, "w", newline="")
    w = csv.DictWriter(f, fieldnames=[
        "construction", "venue", "book", "day", "dow", "dte", "expiry", "strike",
        "entry_hm", "credit", "variant", "exit_hm", "exit_comb", "reason",
        "gross", "net", "peak_gross", "giveback"])
    w.writeheader()

    for venue in ("NIFTY", "SENSEX"):
        V = VENUE[venue]
        lot = V["lot"]
        cost = 4 * V["slip"] * lot + 4 * CHG      # Rs per lot, round trip, 2 legs
        days = trading_days(c, venue)
        vcons = [x for x in cons if x[1] == venue]
        done = 0
        for day in days:
            if day in EXCLUDE_DAYS:
                continue
            d = load_day(c, venue, day)
            if not d:
                log("%s %s: no chain" % (venue, day))
                continue
            fexp, per, spot = d
            dte = dte_of(day, fexp, days)
            dow = date.fromisoformat(day).strftime("%a")
            for name, _, book, cells in vcons:
                if dte not in cells:
                    continue
                e_win, x_win, sl = cells[dte]
                b = build_series(per, spot, V["step"], e_win, x_win)
                if not b:
                    log("%s %s %s: no series" % (venue, day, name))
                    continue
                K, credit, e_hm, ser = b
                for vname, vcfg in vs:
                    hm, comb, reason, peak = replay(vcfg, credit, sl, ser, lot)
                    gross = (credit - comb) * lot
                    net = gross - cost
                    w.writerow(dict(construction=name, venue=venue, book=book, day=day,
                                    dow=dow, dte=dte, expiry=fexp, strike=K,
                                    entry_hm=e_hm, credit=round(credit, 2),
                                    variant=vname, exit_hm=hm,
                                    exit_comb=round(comb, 2), reason=reason,
                                    gross=round(gross), net=round(net),
                                    peak_gross=round(peak),
                                    giveback=round(peak - gross)))
            done += 1
            if done % 20 == 0:
                log("  %s %d/%d days" % (venue, done, len(days)))
                f.flush()
    f.close()
    log("detail written: %s" % DETAIL)


if __name__ == "__main__":
    main()
