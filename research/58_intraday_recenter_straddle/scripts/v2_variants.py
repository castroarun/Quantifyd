"""research/58 — V2 straddle VARIANT LAB. Sweeps move-stop x wings x PT over the SAME
recorded chain (via v2_curves.py) so all management variants are studied in parallel,
refreshed daily by the post-close cron. Writes variants.json (ranked) for /app/straddles."""
import os, json, subprocess, datetime
from pathlib import Path

ROOT = Path("/home/arun/quantifyd")
S = ROOT / "research/58_intraday_recenter_straddle"
OUTS = [ROOT / "static/app/straddles", ROOT / "frontend/public/straddles"]
PY = str(ROOT / "venv/bin/python")

# (label, move%, wings pts, PT%) — move<=0 => no stop (naked roll control); wings=0 => naked
GRID = [
    ("naked · roll only (no stop)", 0,   0,   40),
    ("naked · stop 1.0%",           1.0, 0,   40),
    ("naked · stop 1.5%",           1.5, 0,   40),
    ("naked · stop 2.0%",           2.0, 0,   40),
    ("naked · stop 2.5%",           2.5, 0,   40),
    ("iron-fly · stop 1.5%",        1.5, 500, 40),
    ("iron-fly · stop 2.0%",        2.0, 500, 40),
    ("iron-fly · stop 2.5%",        2.5, 500, 40),
]

def maxdd(nets):
    cum = peak = m = 0
    for x in nets:
        cum += x; peak = max(peak, cum); m = min(m, cum - peak)
    return m

variants = []
for label, move, wings, pt in GRID:
    env = dict(os.environ, V2_MOVE=str(move), V2_WINGS=str(wings), V2_PT=str(pt), PYTHONPATH=".")
    subprocess.run([PY, str(S / "scripts/v2_curves.py")], cwd=str(ROOT), env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    d = json.load(open(S / "results/v2_positional.json"))
    tr = d["trades"]; pnl = [t["pnl"] for t in tr]
    n = len(pnl); wins = sum(1 for x in pnl if x > 0)
    reasons = {}
    for t in tr:
        reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1
    variants.append({
        "label": label, "move": move, "wings": wings, "pt": pt,
        "net": sum(pnl), "mean": round(sum(pnl) / n) if n else 0,
        "win": round(100 * wins / n) if n else 0,
        "best": max(pnl) if pnl else 0, "worst": min(pnl) if pnl else 0,
        "maxdd": round(maxdd(pnl)), "trades": n,
        "calmar": round(sum(pnl) / abs(maxdd(pnl)), 2) if maxdd(pnl) < 0 else None,
        "reasons": reasons, "book_curve": d["book_curve"],
    })

variants.sort(key=lambda v: v["net"], reverse=True)
payload = {
    "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "lots": 10, "lot": 65,
    "note": "recorded-chain replay since 2026-04-20 · NET incl wings · 10 lots · SIGNAL not validated (single regime)",
    "variants": variants,
}
for o in OUTS:
    o.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(o / "variants.json", "w"))
print("wrote variants.json (%d variants):" % len(variants))
for v in variants:
    print("  %-30s net=%+d  win=%d%%  maxDD=%d  calmar=%s" % (v["label"], v["net"], v["win"], v["maxdd"], v["calmar"]))
