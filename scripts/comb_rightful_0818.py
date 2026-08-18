#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""COMB rightful paper record for 2026-08-18 (daemon killed 14:16 after Arun's manual
close-all; standing instruction: record the rules-true book, no sign-off needed).

Book: NAS_COMB20 DTE0 — entered 09:16:08 K=24250, credit 86.95 (CE 38.35 + PE 48.60),
rules: combined-SL 25% (dwell-confirmed like the daemon: exit on the poll AFTER two
consecutive polls >= threshold) else TIME_EXIT 15:20. Tracks live LTPs every 30s from
here; the morning segment (09:16->now) never breached SL — verified from the recorded
1-min chain before launch (see /tmp/comb_rightful.log). Appends a PAPER record with a
recon note. Idempotent."""
import sys, json, time, shutil
from datetime import datetime, date
sys.path.insert(0, "/home/arun/quantifyd")

STATE = "/home/arun/quantifyd/backtest_data/csl_paper_state.json"
LOG = "/tmp/comb_rightful.log"
CE, PE = "NIFTY2681824250CE", "NIFTY2681824250PE"
TODAY = "2026-08-18"
CREDIT, CE0, PE0, K = 86.95, 38.35, 48.60, 24250
SL_THR = CREDIT * 1.25          # DTE0 SL25
QTY, LOTS, COST = 130, 2, 160

def log(m):
    line = "[%s] %s" % (datetime.now().strftime("%H:%M:%S"), m)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

st = json.load(open(STATE))
if any(r.get("book") == "NAS_COMB20" and r.get("day") == TODAY for r in st.get("records", [])):
    log("record already present - exit"); sys.exit(0)

# 1) verify the morning segment from the recorded 1-min chain: any SL-dwell before now?
import sqlite3
db = sqlite3.connect("file:/home/arun/quantifyd/backtest_data/options_data.db?mode=ro", uri=True)
rows = db.execute(
    "SELECT substr(snapshot_time,12,5) hm, instrument_type, ltp FROM option_chain "
    "WHERE symbol='NIFTY' AND strike=? AND snapshot_time LIKE ? AND instrument_type IN ('CE','PE') "
    "AND expiry_date=? AND ltp IS NOT NULL ORDER BY snapshot_time", (K, TODAY + "%", TODAY)).fetchall()
per = {}
for hm, it, ltp in rows:
    per.setdefault(hm, {})[it] = ltp
streak, exited, exit_hm, exit_comb = 0, False, None, None
for hm in sorted(per):
    if hm < "09:17":
        continue
    d = per[hm]
    if "CE" not in d or "PE" not in d:
        continue
    comb = d["CE"] + d["PE"]
    if streak >= 2:                      # dwell confirmed on the two prior minutes
        exited, exit_hm, exit_comb = True, hm, comb
        break
    streak = streak + 1 if comb >= SL_THR else 0
log("morning replay: %d minutes, SL(%.1f) dwell-exit=%s" % (len(per), SL_THR, exit_hm or "no"))

def record(exit_comb, exit_ts, reason):
    shutil.copy(STATE, STATE + ".bak_rightful_" + datetime.now().strftime("%H%M%S"))
    stx = json.load(open(STATE))
    if any(r.get("book") == "NAS_COMB20" and r.get("day") == TODAY for r in stx.get("records", [])):
        log("race: already recorded"); return
    pnl = round((CREDIT - exit_comb) * QTY - COST)
    stx.setdefault("records", []).append({
        "day": TODAY, "book": "NAS_COMB20", "sym": "NIFTY", "dte": 0,
        "cfg": "09:16->15:20 SL25", "strike": K, "expiry": TODAY, "credit": CREDIT,
        "entry_ts": "09:16:08", "exit_ts": exit_ts, "exit_comb": round(exit_comb, 2),
        "reason": reason, "pnl": pnl, "series": [], "lots": LOTS, "qty": QTY,
        "source": "PAPER",
        "recon": ("RIGHTFUL reconstruction 2026-08-18: Arun closed the whole account "
                  "manually at ~14:16 and the pre-guard CSL daemon was auto-killed by "
                  "the manual-close watcher. This records COMB per its own rules "
                  "(hold to SL25-dwell or 15:20 TIME_EXIT) for rules-vs-actual review."),
    })
    stx.setdefault("cum", {})["NAS_COMB20"] = round(stx["cum"].get("NAS_COMB20", 0) + pnl)
    tmp = STATE + ".tmp"
    json.dump(stx, open(tmp, "w"))
    shutil.move(tmp, STATE)
    log("RECORDED %s exit_comb=%.2f pnl=%+d cum=%+d" % (reason, exit_comb, pnl, stx["cum"]["NAS_COMB20"]))

if exited:
    record(exit_comb, exit_hm + ":00", "SL_DWELL")
    sys.exit(0)

# 2) live segment: poll LTPs every 30s until SL-dwell or 15:20
from services.kite_service import get_kite
k = get_kite()
log("live tracking armed (streak=%d from morning tail)" % streak)
while True:
    hm = datetime.now().strftime("%H:%M")
    try:
        q = k.ltp(["NFO:" + CE, "NFO:" + PE])
        comb = q["NFO:" + CE]["last_price"] + q["NFO:" + PE]["last_price"]
    except Exception as ex:
        log("ltp err: %s" % str(ex)[:60]); time.sleep(30); continue
    if hm >= "15:20":
        record(comb, datetime.now().strftime("%H:%M:%S"), "TIME_EXIT"); break
    if streak >= 2:
        record(comb, datetime.now().strftime("%H:%M:%S"), "SL_DWELL"); break
    streak = streak + 1 if comb >= SL_THR else 0
    time.sleep(30)
