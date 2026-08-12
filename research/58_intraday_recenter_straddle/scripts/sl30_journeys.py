"""V1 intraday one-and-done + 30% COMBINED-PREMIUM stop-loss — daily journeys + backtest stats.
Computed from options_study.json (full 5-min ATM straddle premium path per recorded day) — the
SAME data the Opt-Study Peak+% analysis used, so it's instant (no chain re-query). Rule: sell ATM
~09:20; exit ONCE when the combined premium rises >=30% above entry (a 30%-of-credit loss); else
hold to the last bar. 10 lots x 65 = qty 650, -160 round-trip. Writes v1_sl30.json (app + public)."""
import json, os

ROOT = "/home/arun/quantifyd"
STUDY = os.path.join(ROOT, "static/app/options_study.json")
OUTS = [os.path.join(ROOT, "static/app/straddles/v1_sl30.json"),
        os.path.join(ROOT, "frontend/public/straddles/v1_sl30.json")]
QTY = 650; COST = 160; SL = 30.0

d = json.load(open(STUDY))
per_day, trades, book, cum = {}, [], [], 0
for dy in sorted(d["days"], key=lambda x: x["date"]):
    s = dy.get("series") or []
    bars = [(b[0], b[1]) for b in s if b[1]]
    ent = next((p for (h, p) in bars if h >= "09:20"), bars[0][1] if bars else None)
    if not ent or len(bars) < 2:
        continue
    thr = (1 + SL / 100.0) * ent          # premium level that triggers the 30% stop
    series, stopped, final, exit_t, peak = [], False, None, None, 0.0
    for (h, p) in bars:
        if h < "09:20":
            continue
        peak = max(peak, (p / ent - 1) * 100)
        mtm = round((ent - p) * QTY - COST)   # short straddle MTM (credit - premium)
        if not stopped:
            series.append([h, mtm])
            if p >= thr:
                final, stopped, exit_t = mtm, True, h
        else:
            series.append([h, final])
    if len(series) < 2:
        continue
    ys = [v for _, v in series]
    if final is None:
        final = ys[-1]
    day = dy["date"]
    per_day[day] = {"series": series, "exit": ({"time": exit_t, "pnl": final} if stopped else None),
                    "final": final, "stopped": stopped, "low": min(ys), "high": max(ys),
                    "dte": dy["dte"], "strike": dy.get("atm"), "credit": round(ent, 1),
                    "peak_pct": round(peak, 1)}
    cum += final
    trades.append({"day": day, "dte": dy["dte"], "final": final, "stopped": stopped,
                   "exit_time": exit_t, "peak_pct": round(peak, 1)})
    book.append([day, cum])

fins = [t["final"] for t in trades]
n = len(fins); wins = sum(1 for v in fins if v > 0); slhit = sum(1 for t in trades if t["stopped"])
c = pk = mdd = 0
for v in fins:
    c += v; pk = max(pk, c); mdd = min(mdd, c - pk)
by_dte = {}
for k in range(5):
    sub = [t for t in trades if t["dte"] == k]
    if sub:
        f = [t["final"] for t in sub]; sh = sum(1 for t in sub if t["stopped"])
        c2 = p2 = dd2 = 0
        for v in f:
            c2 += v; p2 = max(p2, c2); dd2 = min(dd2, c2 - p2)
        by_dte[k] = {"n": len(sub), "total": sum(f), "mean": round(sum(f) / len(f)),
                     "win": round(100 * sum(1 for v in f if v > 0) / len(f)),
                     "sl_hit_pct": round(100 * sh / len(sub)), "maxdd": round(dd2)}
stats = {"total": sum(fins), "n": n, "mean": round(sum(fins) / n) if n else 0,
         "win": round(100 * wins / n) if n else 0, "sl_hit_pct": round(100 * slhit / n) if n else 0,
         "maxdd": round(mdd), "best": max(fins) if fins else 0, "worst": min(fins) if fins else 0,
         "by_dte": by_dte}
out = {"version": "V1 one-and-done + 30% combined-premium SL (NIFTY, recorded chain)",
       "index": "NIFTY", "sl_pct": SL, "lots": 10, "lot": 65, "days": sorted(per_day),
       "per_day": per_day, "trades": trades, "book_curve": book, "stats": stats}
for o in OUTS:
    os.makedirs(os.path.dirname(o), exist_ok=True)
    json.dump(out, open(o, "w"))
print("SL30: %d days | total %+d | mean %+d | win %d%% | SL-hit %d%% | maxDD %d | worst %d" % (
    n, stats["total"], stats["mean"], stats["win"], stats["sl_hit_pct"], stats["maxdd"], stats["worst"]))
print("by DTE:", by_dte)
