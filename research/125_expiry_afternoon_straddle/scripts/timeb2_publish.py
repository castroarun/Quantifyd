"""TIMEB2 live publisher - tails /tmp/timeb2_live.log, polls leg LTPs while OPEN,
writes static/app/timeb2_live.json every ~10s so /app/nas can render the one-shot
book live. Self-terminates at 14:45."""
import sys, re, json, time
from datetime import datetime
sys.path.insert(0, "/home/arun/quantifyd/research/111_sensex_manual_gmt".replace("gmt", "mgmt") + "/scripts")
sys.path.insert(0, "/home/arun/quantifyd/research/111_sensex_manual_mgmt/scripts")
sys.path.insert(0, "/home/arun/quantifyd")
import csl_paper_exec as X

LOG = "/tmp/timeb2_live.log"
PUB = "/home/arun/quantifyd/static/app/timeb2_live.json"
QTY = 520
k = None

while True:
    now = datetime.now().strftime("%H:%M")
    if now >= "14:45":
        break
    d = {"ts": datetime.now().strftime("%H:%M:%S"), "status": "ARMED",
         "window": "13:15-14:30", "sl": "CSL30", "lots": 8, "qty": QTY}
    try:
        t = open(LOG).read()
        m = re.search(r"legs (\S+CE) / (\S+PE)", t)
        if m: d["ce"], d["pe"] = m.group(1), m.group(2)
        m = re.search(r"OPEN \[LIVE\] credit ([\d.]+) \(CE ([\d.]+) \+ PE ([\d.]+)\) \| SL trigger ([\d.]+)", t)
        if m:
            d.update(status="OPEN", credit=float(m.group(1)), ce_fill=float(m.group(2)),
                     pe_fill=float(m.group(3)), sl_trigger=float(m.group(4)))
        m = re.search(r"\[(\d\d:\d\d):\d\d\] EXIT trigger: (\w+)", t)
        if m: d.update(status="EXITING", reason=m.group(2), exit_ts=m.group(1))
        m = re.search(r"EXIT (\S+CE) buy fill ([\d.]+)", t)
        if m: d["exit_ce"] = float(m.group(2))
        m = re.search(r"EXIT (\S+PE) buy fill ([\d.]+)", t)
        if m: d["exit_pe"] = float(m.group(2))
        m = re.search(r"DONE \[LIVE\] (\w+) credit ([\d.]+) -> debit ([\d.]+) \| P&L ([+-]?\d+)", t)
        if m:
            d.update(status="DONE", reason=m.group(1), credit=float(m.group(2)),
                     debit=float(m.group(3)), pnl=int(m.group(4)))
        if "ABORT" in t:
            am = re.search(r"ABORT[^\n]*", t)
            d.update(status="ABORTED", note=am.group(0)[-80:] if am else "")
        if d["status"] == "OPEN" and d.get("ce"):
            try:
                if k is None: k = X.kite()
                q = k.ltp(["NFO:" + d["ce"], "NFO:" + d["pe"]])
                comb = q["NFO:" + d["ce"]]["last_price"] + q["NFO:" + d["pe"]]["last_price"]
                d["comb"] = round(comb, 2)
                d["open_pnl"] = round((d["credit"] - comb) * QTY)
            except Exception as ex:
                d["ltp_err"] = str(ex)[:50]
        json.dump(d, open(PUB, "w"))
    except FileNotFoundError:
        pass
    except Exception as ex:
        try: json.dump({"status": "PUB_ERR", "err": str(ex)[:80]}, open(PUB, "w"))
        except Exception: pass
    if d.get("status") == "DONE" or d.get("status") == "ABORTED":
        break
    time.sleep(10)
print("publisher done")
