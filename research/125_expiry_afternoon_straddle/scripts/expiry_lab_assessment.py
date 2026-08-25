"""research/125 - EXPIRY-AFTERNOON LAB assessment (cron Tue+Thu 16:05 IST).
Re-runs the expiry-afternoon sweep on all recorded expiry days, re-scores the frozen
winner slots, compares TimeB2 live days vs the model, appends a PERMANENT run-history
row, and publishes static/app/straddles/expiry_lab.json for the /app/straddles section."""
import json, os, subprocess, sys
from datetime import datetime

Q = "/home/arun/quantifyd"
SWEEP = Q + "/research/125_expiry_afternoon_straddle/scripts/expiry_afternoon_sweep.py"
SWEEP_OUT = Q + "/research/125_expiry_afternoon_straddle/results/expiry_afternoon.json"
LIVE = Q + "/research/125_expiry_afternoon_straddle/results/timeb2_live_days.json"
PUB = Q + "/static/app/straddles/expiry_lab.json"

# the frozen watch-list (research/125, 2026-08-25): slot -> deployed/considered size
WINNERS = [
    ("NIFTY", "13:15", "14:30", 30, 8),     # TimeB2 live slot (Tuesdays)
    ("NIFTY", "13:15", "14:45", 30, 8),
    ("SENSEX", "13:30", "14:15", "none", 8),
    ("SENSEX", "13:30", "15:00", 30, 8),
]
REF = [("NIFTY", "13:45", "15:00"), ("SENSEX", "13:45", "15:00")]  # AlgoTest original
BASIS = {"NIFTY": 10, "SENSEX": 5}

r = subprocess.run([Q + "/venv/bin/python3", SWEEP], capture_output=True, text=True, timeout=1800)
if r.returncode != 0:
    print("SWEEP FAILED:\n" + r.stdout[-800:] + r.stderr[-800:]); sys.exit(1)

d = json.load(open(SWEEP_OUT))
cells = {(c["sym"], c["entry"], c["exit"], c["sl"]): c for c in d["cells"]}

prev = {}
if os.path.exists(PUB):
    prev = json.load(open(PUB))

winners, flags = [], []
for sym, e, xx, sl, lots in WINNERS:
    c = cells.get((sym, e, xx, sl))
    if not c:
        flags.append("MISSING cell %s %s-%s SL%s" % (sym, e, xx, sl)); continue
    f = lots / BASIS[sym]
    row = {"sym": sym, "entry": e, "exit": xx, "sl": sl, "lots": lots,
           "mean": round(c["mean"] * f), "mean_lot": round(c["mean"] / BASIS[sym]),
           "win": c["win"], "maxdd": round(c["maxdd"] * f), "ratio": c["ratio"], "n": c["n"],
           "series": [[dd, round(v * f)] for dd, v in c["series"]]}
    # drift vs the previous published run
    for pw in (prev.get("winners") or []):
        if (pw["sym"], pw["entry"], pw["exit"], str(pw["sl"])) == (sym, e, xx, str(sl)):
            if pw.get("mean", 0) > 0 and row["mean"] < 0.6 * pw["mean"]:
                flags.append("DRIFT %s %s-%s: mean %+d vs prev %+d" % (sym, e, xx, row["mean"], pw["mean"]))
            break
    if c["ratio"] is not None and c["ratio"] < 5:
        flags.append("WEAK %s %s-%s ratio %.1f < 5" % (sym, e, xx, c["ratio"]))
    winners.append(row)

ref = [{"sym": s, "entry": e, "exit": xx, "sl": c["sl"], "mean_lot": round(c["mean"] / BASIS[s]),
        "win": c["win"], "ratio": c["ratio"]}
       for s, e, xx in REF for c in [max((cells[k] for k in cells if k[:3] == (s, e, xx)),
                                         key=lambda c: c["ratio"] or -9, default=None)] if c]

live_days = json.load(open(LIVE)) if os.path.exists(LIVE) else []
# from 2026-08-26 TimeB2 is the daemon book CSL_TIMEB2_LIVE - its trades live in the
# CSL day records, not the one-shot file
try:
    st = json.load(open(Q + "/backtest_data/csl_paper_state.json"))
    recs = st.get("records") or (st if isinstance(st, list) else [])
    for r in recs:
        if isinstance(r, dict) and r.get("book") == "CSL_TIMEB2_LIVE":
            live_days.append({"day": r.get("day"), "status": "DONE", "reason": r.get("reason"),
                              "credit": r.get("credit"), "pnl": r.get("pnl"),
                              "qty": r.get("qty"), "lots": r.get("lots")})
except Exception:
    pass
# live-vs-model: match each DONE live day to the TimeB2 model cell's same-day value
model = {dd: v for dd, v in (winners[0]["series"] if winners else [])}
lvm = []
for ld in live_days:
    if ld.get("status", "").startswith("DONE") and ld.get("pnl") is not None:
        lvm.append({"day": ld["day"], "live": ld["pnl"], "model": model.get(ld["day"]),
                    "reason": ld.get("reason"), "credit": ld.get("credit")})

hist = prev.get("runs") or []
verdict = "OK" if not flags else "; ".join(flags[:4])
hist.append({"at": datetime.now().strftime("%Y-%m-%d %H:%M"),
             "days": {s: d["meta"][s]["expiry_days"] for s in d["meta"]},
             "top": {"%s %s-%s SL%s" % (w["sym"], w["entry"], w["exit"], w["sl"]):
                     "%+d/d %d%% r%s" % (w["mean"], w["win"], w["ratio"]) for w in winners},
             "verdict": verdict})

json.dump({"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "meta": d["meta"], "cost_model": d.get("cost_model"),
           "winners": winners, "algotest_ref": ref, "calm": d.get("calm"),
           "live_days": live_days, "live_vs_model": lvm, "runs": hist},
          open(PUB, "w"))
print("expiry_lab.json published | verdict: %s | live days tracked: %d" % (verdict, len(lvm)))
