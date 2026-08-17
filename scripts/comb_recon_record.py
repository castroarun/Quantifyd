# -*- coding: utf-8 -*-
"""Record COMB (NAS_COMB20) RIGHTFUL trade for 2026-08-17 as a paper record.

COMB entered a 24300 straddle @ 09:16 (credit 133.35) and, per its rules, holds to the
15:20 TIME_EXIT (its combined-SL 173.4 never hit). Its CE was force-closed early by the
portfolio stop at 12:46, but this reconstruction records COMB *as its own rules say it
should have run* -- hold both legs to 15:20 and mark-to-market there. Marked source=PAPER
with a `recon` note so a later rules-vs-actual assessment can tell it apart.

Waits until 15:20:05 IST, captures the real 24300 CE+PE marks, appends to csl_paper_state.json
(backup + cum update + idempotent). Self-contained; needs no permissions to finalize.
"""
import sys, json, time, shutil
from datetime import datetime, date
sys.path.insert(0, "/home/arun/quantifyd")

STATE = "/home/arun/quantifyd/backtest_data/csl_paper_state.json"
LOG   = "/tmp/comb_recon.log"
CE    = "NIFTY2681824300CE"
PE    = "NIFTY2681824300PE"
BOOK  = "NAS_COMB20"
TODAY = "2026-08-17"

def log(m):
    line = "[%s] %s" % (datetime.now().strftime("%H:%M:%S"), m)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

# ---- idempotency: bail if a NAS_COMB20 record already exists for today ----
st = json.load(open(STATE))
if any(r.get("book") == BOOK and r.get("day") == TODAY for r in st.get("records", [])):
    log("NAS_COMB20 record for %s already present -- nothing to do." % TODAY)
    sys.exit(0)

# ---- wait until 15:20:05 IST (VPS clock is IST) ----
log("recorder armed; waiting for 15:20:05 IST to mark COMB's rightful TIME_EXIT")
while datetime.now().strftime("%H:%M:%S") < "15:20:05":
    time.sleep(2)

# ---- capture the real 15:20 marks ----
from services.kite_service import get_kite
k = get_kite()
q = k.ltp(["NFO:%s" % CE, "NFO:%s" % PE])
ce = float(q["NFO:%s" % CE]["last_price"])
pe = float(q["NFO:%s" % PE]["last_price"])
exit_comb = round(ce + pe, 2)

credit, qty, lots, cost = 133.35, 130, 2, 160
pnl = round((credit - exit_comb) * qty - cost)

rec = {
    "day": TODAY, "book": BOOK, "sym": "NIFTY", "dte": 1,
    "cfg": "09:16->15:20 SL30", "strike": 24300, "expiry": "2026-08-18",
    "credit": credit, "entry_ts": "09:16:07",
    "exit_ts": datetime.now().strftime("%H:%M:%S"),
    "exit_comb": exit_comb, "reason": "TIME_EXIT",
    "pnl": pnl, "series": [], "lots": lots, "qty": qty, "source": "PAPER",
    "recon": ("RIGHTFUL reconstruction: COMB per its own rules (24300 straddle @09:16 "
              "credit 133.35, hold to 15:20 TIME_EXIT; SL 173.4 never hit). CE was "
              "force-closed early ~102 by the account-level portfolio stop @12:46, "
              "unknown to COMB; this record marks the clean rules outcome for later "
              "rules-vs-actual assessment."),
}

# ---- backup + append + cum update, atomic write ----
shutil.copy(STATE, STATE + ".bak_recon_%s" % datetime.now().strftime("%H%M%S"))
st = json.load(open(STATE))  # reload fresh in case of any change
if any(r.get("book") == BOOK and r.get("day") == TODAY for r in st.get("records", [])):
    log("race: NAS_COMB20 record appeared -- skipping.")
    sys.exit(0)
st.setdefault("records", []).append(rec)
st.setdefault("cum", {})[BOOK] = round(st["cum"].get(BOOK, 0) + pnl)
tmp = STATE + ".tmp"
json.dump(st, open(tmp, "w"))
shutil.move(tmp, STATE)

log("RECORDED COMB rightful TIME_EXIT: exit_comb=%.2f (CE %.2f + PE %.2f) credit %.2f "
    "-> P&L %+d (%d lots) source=PAPER; cum[%s]=%+d"
    % (exit_comb, ce, pe, credit, pnl, lots, BOOK, st["cum"][BOOK]))
