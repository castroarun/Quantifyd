#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AS-IS (rules-true) reconstruction after a manual close-all — 2026-08-18.

For every LIVE leg reconciled MANUAL_CLOSE_RECONCILED today, continue the system's own
exit rules on the recorded 1-min option chain from the reconcile moment to EOD, and
write the outcome back into that system's DB as a PAPER row (status CLOSED, notes
link the live leg) — so the Trade Book shows the as-is track per system and future
rules-vs-actual reviews have the record.

Today's reconciled state (14:18):
  ATM   : naked CE 24250 short @42.7 entry — ST(7,3) BE-clamped trail -> EOD 15:15
  ATM2  : CE 24250 @42.2 + PE 24250 @44.6 — Rs2,500/lot rupee MTM stop -> EOD 15:15
  ATM4  : naked CE 24250 short @42.1 entry — ST(7,3) trail -> EOD 15:15
Also: sync the COMB rightful record (csl STATE) into the PUB json + mark the frozen
csl_paper_live book CLOSED so the UI reflects it.

Run AFTER close (>= 15:35 IST). Idempotent (skips if AS_IS rows exist today).
Trail note: ST(7,3) rebuilt on 5-min candles aggregated from 1-min chain LTPs —
an approximation of the live tick trail, stated in the row notes.
"""
import sys, json, sqlite3, shutil
from datetime import datetime

sys.path.insert(0, "/home/arun/quantifyd")
TODAY = "2026-08-18"
CHAIN = "/home/arun/quantifyd/backtest_data/options_data.db"
QTY = 130
EOD = "15:15"

def log(m):
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), m), flush=True)

def chain_series(sym_strike, inst, start_hm):
    c = sqlite3.connect("file:%s?mode=ro" % CHAIN, uri=True)
    rows = c.execute(
        "SELECT substr(snapshot_time,12,5) hm, ltp FROM option_chain "
        "WHERE symbol='NIFTY' AND strike=? AND instrument_type=? AND expiry_date=? "
        "AND snapshot_time LIKE ? AND ltp IS NOT NULL ORDER BY snapshot_time",
        (sym_strike, inst, TODAY, TODAY + "%")).fetchall()
    c.close()
    seen = {}
    for hm, ltp in rows:
        seen[hm] = ltp          # last print per minute
    return [(hm, seen[hm]) for hm in sorted(seen) if start_hm <= hm <= EOD]

def st_trail_exit(series, arm_hm, entry_price):
    """ST(7,3) short-premium trail on 5-min candles built from 1-min closes,
    breakeven-clamped (never above entry). Returns (exit_hm, exit_px, reason)."""
    def candles(upto_idx):
        out = {}
        for hm, px in series[:upto_idx + 1]:
            key = hm[:4] + str(int(hm[4]) // 5 * 5)
            o = out.setdefault(key, [px, px, px, px])
            o[1] = max(o[1], px); o[2] = min(o[2], px); o[3] = px
        return [dict(open=v[0], high=v[1], low=v[2], close=v[3]) for v in out.values()]
    from services.nas_atm4_executor import NasAtm4Executor
    for i, (hm, px) in enumerate(series):
        if hm < arm_hm:
            continue
        cs = candles(i)
        if len(cs) < 8:
            continue
        st, _ = NasAtm4Executor.compute_short_trailing_stop(cs, 7, 3.0)
        if st is None:
            continue
        stop = min(st, entry_price)          # BE clamp
        if px > stop:
            return hm, px, "ASIS_ST_EXIT"
    return series[-1][0], series[-1][1], "ASIS_EOD"

def insert_paper(dbf, table, legs, note):
    conn = sqlite3.connect(dbf)
    cur = conn.cursor()
    for L in legs:
        cur.execute(
            "INSERT INTO %s (strangle_id, leg, tradingsymbol, exchange, transaction_type,"
            " instrument_type, qty, strike, expiry_date, entry_price, entry_time,"
            " exit_price, exit_time, exit_reason, sl_price, signal_type,"
            " adjustment_count, status, notes, mode, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)" % table,
            (L["sid"], L["leg"], L["tsym"], "NFO", "SELL", L["leg"], QTY, L["strike"],
             TODAY, L["entry"], TODAY + "T09:16:00", L["exit"],
             TODAY + "T" + L["exit_hm"] + ":00", L["reason"], L.get("sl"),
             "ASIS_RECON", 0, "CLOSED", note, "paper",
             datetime.now().isoformat(), datetime.now().isoformat()))
    conn.commit(); conn.close()

def already(dbf, table):
    conn = sqlite3.connect(dbf)
    n = conn.execute("SELECT count(*) FROM %s WHERE signal_type='ASIS_RECON' "
                     "AND created_at LIKE ?" % table, (TODAY + "%",)).fetchone()[0]
    conn.close(); return n > 0

B = "/home/arun/quantifyd/backtest_data/"
NOTE = ("AS-IS rules-true continuation after MANUAL_CLOSE_RECONCILED 14:18 "
        "(replayed on recorded 1-min chain; ST trail approximated on 5-min candles "
        "from 1-min closes). For rules-vs-actual review, not money.")

ce = chain_series(24250, "CE", "14:18")
pe = chain_series(24250, "PE", "14:18")
if not ce or not pe:
    log("no chain data after 14:18 - abort"); sys.exit(1)

# ATM: naked CE, ST trail (armed since 10:47 -> rebuild trail over full series from 10:47)
ce_full = chain_series(24250, "CE", "10:47")
hm1, px1, r1 = st_trail_exit(ce_full, "14:18", 42.7)
db1 = B + "nas_916_atm_trading.db"
if not already(db1, "nas_atm_positions"):
    insert_paper(db1, "nas_atm_positions",
                 [dict(sid=990818, leg="CE", tsym="NIFTY2681824250CE", strike=24250,
                       entry=42.7, exit=px1, exit_hm=hm1, reason=r1, sl=None)], NOTE)
    log("ATM  as-is: CE exit %s @ %.2f (%s)  pnl %+d" % (hm1, px1, r1, round((42.7 - px1) * QTY)))
else:
    log("ATM  as-is rows already present - skip")

# ATM2: both legs, rupee stop Rs2500/lot x2 => exit when comb >= entry_comb + 5000/130
entry_comb2 = 42.2 + 44.6
thr2 = entry_comb2 + 5000.0 / QTY
comb = {}
for hm, v in ce:
    comb.setdefault(hm, [None, None])[0] = v
for hm, v in pe:
    comb.setdefault(hm, [None, None])[1] = v
hm2, r2 = EOD, "ASIS_EOD"
ce2 = ce[-1][1]; pe2 = pe[-1][1]
for hm in sorted(comb):
    a, b = comb[hm]
    if a is None or b is None:
        continue
    if a + b >= thr2:
        hm2, r2, ce2, pe2 = hm, "ASIS_RUPEE_STOP", a, b
        break
db2 = B + "nas_916_atm2_trading.db"
if not already(db2, "nas_atm_positions"):
    insert_paper(db2, "nas_atm_positions",
                 [dict(sid=990818, leg="CE", tsym="NIFTY2681824250CE", strike=24250,
                       entry=42.2, exit=ce2, exit_hm=hm2, reason=r2, sl=None),
                  dict(sid=990818, leg="PE", tsym="NIFTY2681824250PE", strike=24250,
                       entry=44.6, exit=pe2, exit_hm=hm2, reason=r2, sl=None)], NOTE)
    log("ATM2 as-is: exit %s (%s) CE %.2f PE %.2f  pnl %+d"
        % (hm2, r2, ce2, pe2, round(((42.2 - ce2) + (44.6 - pe2)) * QTY)))
else:
    log("ATM2 as-is rows already present - skip")

# ATM4: naked CE, same trail
hm4, px4, r4 = st_trail_exit(ce_full, "14:18", 42.1)
db4 = B + "nas_916_atm4_trading.db"
if not already(db4, "nas_atm_positions"):
    insert_paper(db4, "nas_atm_positions",
                 [dict(sid=990818, leg="CE", tsym="NIFTY2681824250CE", strike=24250,
                       entry=42.1, exit=px4, exit_hm=hm4, reason=r4, sl=None)], NOTE)
    log("ATM4 as-is: CE exit %s @ %.2f (%s)  pnl %+d" % (hm4, px4, r4, round((42.1 - px4) * QTY)))
else:
    log("ATM4 as-is rows already present - skip")

# CSL30F_SENSEX (paper, frozen OPEN at the 14:16 daemon kill): rightful record.
def _csl30f_sensex():
    K, CRED, QTYS, LOTS, EXP = 77500, 575.2, 60, 3, "2026-08-20"
    THR = CRED * 1.30
    STATE = "/home/arun/quantifyd/backtest_data/csl_paper_state.json"
    stx = json.load(open(STATE))
    if any(r.get("book") == "CSL30F_SENSEX" and r.get("day") == TODAY for r in stx.get("records", [])):
        log("CSL30F_SENSEX: already recorded"); return
    c = sqlite3.connect("file:%s?mode=ro" % CHAIN, uri=True)
    rows = c.execute(
        "SELECT substr(snapshot_time,12,5) hm, instrument_type, ltp FROM option_chain "
        "WHERE symbol='SENSEX' AND strike=? AND expiry_date=? AND snapshot_time LIKE ? "
        "AND ltp IS NOT NULL ORDER BY snapshot_time", (K, EXP, TODAY + "%")).fetchall()
    c.close()
    per = {}
    for hm, it, ltp in rows:
        per.setdefault(hm, {})[it] = ltp
    streak, exit_hm, exit_comb, reason = 0, None, None, "TIME_EXIT"
    last_comb = None
    for hm in sorted(per):
        if hm < "09:17":
            continue
        d = per[hm]
        if "CE" not in d or "PE" not in d:
            continue
        comb = d["CE"] + d["PE"]
        last_comb = (hm, comb)
        if hm >= "15:20":
            exit_hm, exit_comb = hm, comb; break
        if streak >= 2:
            exit_hm, exit_comb, reason = hm, comb, "SL_DWELL"; break
        streak = streak + 1 if comb >= THR else 0
    if exit_comb is None:
        if not last_comb:
            log("CSL30F_SENSEX: no chain data - skip"); return
        exit_hm, exit_comb = last_comb
    pnl = round((CRED - exit_comb) * QTYS - 160)
    shutil.copy(STATE, STATE + ".bak_sx30f")
    stx.setdefault("records", []).append({
        "day": TODAY, "book": "CSL30F_SENSEX", "sym": "SENSEX", "dte": 2,
        "cfg": "09:16->15:20 SL30", "strike": K, "expiry": EXP, "credit": CRED,
        "entry_ts": "09:16:08", "exit_ts": exit_hm + ":00", "exit_comb": round(exit_comb, 2),
        "reason": reason, "pnl": pnl, "series": [], "lots": LOTS, "qty": QTYS,
        "source": "PAPER",
        "recon": "RIGHTFUL reconstruction: daemon killed 14:16 (manual close-all watcher) with this paper book frozen OPEN; replayed to its rules exit on the recorded 1-min SENSEX chain."})
    stx.setdefault("cum", {})["CSL30F_SENSEX"] = round(stx["cum"].get("CSL30F_SENSEX", 0) + pnl)
    tmp = STATE + ".tmp2"
    json.dump(stx, open(tmp, "w"))
    shutil.move(tmp, STATE)
    log("CSL30F_SENSEX rightful: %s @ %.1f -> pnl %+d" % (reason, exit_comb, pnl))
_csl30f_sensex()

# COMB: sync rightful record (STATE) -> PUB + close the frozen live book in the UI json
STATE = B + "csl_paper_state.json"
PUB = "/home/arun/quantifyd/static/app/csl_paper.json"
LIVEJ = "/home/arun/quantifyd/static/app/csl_paper_live.json"
try:
    stx = json.load(open(STATE))
    if any(r.get("book") == "NAS_COMB20" and r.get("day") == TODAY for r in stx.get("records", [])):
        shutil.copy(PUB, PUB + ".bak_asis")
        json.dump(stx, open(PUB, "w"))
        lv = json.load(open(LIVEJ))
        changed = False
        for bk in ("NAS_COMB20", "CSL30F_SENSEX"):
            if lv.get("books", {}).get(bk, {}).get("state") == "OPEN":
                lv["books"][bk]["state"] = "CLOSED"; changed = True
        if changed:
            json.dump(lv, open(LIVEJ, "w"))
        log("COMB: rightful record synced to PUB; frozen live book marked CLOSED")
    else:
        log("COMB: rightful record not in STATE yet (recorder still waiting?) - rerun me after 15:25")
except Exception as ex:
    log("COMB sync err: %s" % ex)
log("done")
