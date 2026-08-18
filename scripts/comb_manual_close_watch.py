#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-day watcher (2026-08-18): today's running CSL daemon predates the broker-truth
guard (it loads code only at its 09:12 start). If Arun manually closes the COMB legs
at the broker while COMB's book is still OPEN, this watcher kills the daemon BEFORE
its SL/15:20 exit can fire phantom BUY orders. Self-expires at 15:26 IST.
From tomorrow the guard inside csl_paper_exec.py makes this watcher unnecessary."""
import sys, json, time, subprocess
from datetime import datetime
sys.path.insert(0, "/home/arun/quantifyd")

LIVE = "/home/arun/quantifyd/static/app/csl_paper_live.json"
LOG = "/tmp/comb_watch.log"

def log(m):
    line = "[%s] %s" % (datetime.now().strftime("%H:%M:%S"), m)
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

from services.kite_service import get_kite
k = get_kite()
log("watcher armed: kills csl_paper_exec if COMB legs leave the broker while book OPEN")

while datetime.now().strftime("%H:%M") < "15:26":
    try:
        b = json.load(open(LIVE)).get("books", {}).get("NAS_COMB20", {})
        if b.get("state") != "OPEN" or not b.get("ce_sym"):
            log("COMB no longer OPEN (state=%s) - watcher done" % b.get("state"))
            break
        ce_sym, pe_sym = b["ce_sym"], b["pe_sym"]
        shorts = {ce_sym: 0, pe_sym: 0}
        for p in k.positions().get("net", []):
            if p.get("tradingsymbol") in shorts and p.get("product") == "MIS":
                shorts[p["tradingsymbol"]] = max(0, -int(p.get("quantity") or 0))
        if shorts[ce_sym] == 0 and shorts[pe_sym] == 0:
            log("MANUAL CLOSE DETECTED: broker holds no short %s / %s but COMB book is OPEN -> killing csl_paper_exec" % (ce_sym, pe_sym))
            subprocess.run(["pkill", "-f", "csl_paper_exec.py"])
            log("daemon killed - COMB cannot fire a phantom exit today. Record COMB manually if needed.")
            break
    except Exception as ex:
        log("watch err (tolerated): %s" % str(ex)[:80])
    time.sleep(30)
log("watcher exit")
