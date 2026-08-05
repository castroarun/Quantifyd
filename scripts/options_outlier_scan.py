#!/usr/bin/env python3
"""Options-data outlier + drift scan (program #4). Reads the daily straddle summaries in
static/app/options_study.json and flags: (a) OUTLIER days — the straddle expanded / barely decayed,
or a big underlying move (the days that hurt a short-straddle book); (b) DRIFT — is the median decay
softening recently (edge weakening). Writes static/app/options_outliers.json for the reports page.
Cron: daily post-close. Re-assess-as-we-go, automated."""
import json
import os
import datetime

ROOT = "/home/arun/quantifyd"
SRC = ROOT + "/static/app/options_study.json"
OUT = ROOT + "/static/app/options_outliers.json"


def median(xs):
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def main():
    now = datetime.datetime.now()
    if not os.path.exists(SRC):
        json.dump({"generated": now.strftime("%Y-%m-%d %H:%M:%S"), "note": "no options_study.json"}, open(OUT, "w"))
        return
    days = (json.load(open(SRC)) or {}).get("days") or []
    rows = [d for d in days if d.get("entry") and d.get("decay_pct") is not None]
    if len(rows) < 15:
        json.dump({"generated": now.strftime("%Y-%m-%d %H:%M:%S"), "note": "insufficient data", "n_days": len(rows)}, open(OUT, "w"))
        return
    decay = [d["decay_pct"] for d in rows]
    moves = [abs(d.get("spot_move") or 0) for d in rows]
    dmed = median(decay)
    mad = median([abs(x - dmed) for x in decay]) or 1.0
    mp90 = sorted(moves)[int(0.9 * len(moves))]

    outliers = []
    for d in rows:
        z = (d["decay_pct"] - dmed) / (1.4826 * mad)
        flags = []
        if d["decay_pct"] <= 0:
            flags.append("straddle expanded — no decay")
        elif z <= -2:
            flags.append("unusually low decay")
        elif z >= 2.5:
            flags.append("exceptional decay")
        if abs(d.get("spot_move") or 0) >= mp90:
            flags.append("big underlying move %+d" % (d.get("spot_move") or 0))
        if flags:
            outliers.append({"date": d["date"], "weekday": d.get("weekday"), "dte": d.get("dte"),
                             "decay_pct": d["decay_pct"], "spot_move": d.get("spot_move"), "flags": flags})

    recent = decay[-10:]
    rmed = median(recent)
    if rmed < dmed - 2:
        read = "decay SOFTENING — short-straddle edge weaker recently (%.1f%% vs %.1f%% median)" % (rmed, dmed)
    elif rmed > dmed + 2:
        read = "decay stronger recently (%.1f%% vs %.1f%%)" % (rmed, dmed)
    else:
        read = "decay stable (%.1f%% recent vs %.1f%% median)" % (rmed, dmed)

    out = {
        "generated": now.strftime("%Y-%m-%d %H:%M:%S"), "n_days": len(rows),
        "median_decay_pct": round(dmed, 1), "median_abs_move": round(median(moves)),
        "drift": read, "n_outliers": len(outliers), "outliers": outliers[-15:],
    }
    json.dump(out, open(OUT, "w"), separators=(",", ":"))
    print("outliers: %d flagged over %d days | drift: %s" % (len(outliers), len(rows), read))


if __name__ == "__main__":
    main()
