#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Restate the NAS daily history on a uniform 2-lot basis (display-only).

WHY: the traded size changed repeatedly -- 1, 2, 5, 10 lots, across two different NIFTY lot
sizes (75 until 2026-05-15, 65 from 2026-05-18). So the daily-P&L history is NOT comparable
across time: the amplitude jumps ~10x on 2026-06-25 purely because the paper book went from
1 lot (65 qty) to 10 lots (650 qty). That reads as a regime change. It is a size change.

WHAT THIS DOES NOT DO: it does not rewrite a single P&L value, and it does not touch the
position ledger. `nas_*_positions` is the audit record of orders that really went to the
exchange. Instead we annotate each (system, day) with the size it actually traded and a
`scale` factor to restate it at 2 lots. The UI multiplies by `scale`; the raw numbers stay
intact and recoverable.

    lots  = qty / lot_size(day)          lot_size = 75 before 2026-05-18, else 65
    scale = 2 / lots                     (2 lots = the uniform live size)

Writes into each static/snapshots/nas_mtm_<day>.json:
    systems[k]['qty'], ['lots'], ['scale'], ['mode']
and into index.json, per day:
    'combined_last_2lot'   (sum of system.last * scale)

Idempotent. Backs up static/snapshots first.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parents[1] / "static").exists() \
    else Path("/home/arun/quantifyd")
SNAP = ROOT / "static" / "snapshots"

TARGET_LOTS = 2
LOT_SWITCH = "2026-05-18"   # NIFTY lot 75 -> 65 (verified: every qty divides exactly either side)

# snapshot system key -> (db file, positions table)
DBS = {
    "nas":          ("nas_trading.db",          "nas_positions"),
    "nas-atm":      ("nas_atm_trading.db",      "nas_atm_positions"),
    "nas-atm2":     ("nas_atm2_trading.db",     "nas_atm_positions"),
    "nas-atm4":     ("nas_atm4_trading.db",     "nas_atm_positions"),
    "nas-916-otm":  ("nas_916_otm_trading.db",  "nas_positions"),
    "nas-916-atm":  ("nas_916_atm_trading.db",  "nas_atm_positions"),
    "nas-916-atm2": ("nas_916_atm2_trading.db", "nas_atm_positions"),
    "nas-916-atm4": ("nas_916_atm4_trading.db", "nas_atm_positions"),
}


def lot_size(day: str) -> int:
    return 75 if day < LOT_SWITCH else 65


def day_size(sys_key: str, day: str):
    """Dominant (qty, mode) that system traded that day -> (qty, lots, scale, mode)."""
    spec = DBS.get(sys_key)
    if not spec:
        return None
    p = ROOT / "backtest_data" / spec[0]
    if not p.exists():
        return None
    try:
        c = sqlite3.connect(str(p), timeout=5)
        rows = c.execute(
            f"SELECT qty, mode, COUNT(*) n FROM {spec[1]} "
            f"WHERE date(entry_time)=? AND qty IS NOT NULL AND qty>0 "
            f"GROUP BY qty, mode ORDER BY n DESC", (day,)).fetchall()
        c.close()
    except Exception:
        return None
    if not rows:
        return None
    qty, mode, _ = rows[0]
    lots = qty / lot_size(day)
    if lots <= 0:
        return None
    return int(qty), round(lots, 4), round(TARGET_LOTS / lots, 6), (mode or "paper")


def _combined_2lot(systems: dict) -> dict:
    """Whole-book curve with every system restated at 2 lots.

    Same aggregation as dump_nas_mtm._combined_curve (union of snapshot times,
    forward-fill each system's most recent value, sum) with system.scale applied.
    """
    all_ts = sorted({p[0] for v in systems.values() for p in (v.get("points") or [])})
    if not all_ts:
        return {"points": [], "last": 0.0, "n": 0}
    idx = {k: 0 for k in systems}
    cur = {k: 0.0 for k in systems}
    out = []
    for ts in all_ts:
        tot = 0.0
        for k, v in systems.items():
            pts = v.get("points") or []
            sc = float(v.get("scale") or 1.0)
            i = idx[k]
            while i < len(pts) and pts[i][0] <= ts:
                cur[k] = float(pts[i][1] or 0.0) * sc
                i += 1
            idx[k] = i
            tot += cur[k]
        out.append([ts, round(tot, 2)])
    return {"points": out, "last": out[-1][1] if out else 0.0, "n": len(out)}


def main():
    if not SNAP.exists():
        print("no snapshots dir:", SNAP)
        return
    bak = SNAP.parent / ("snapshots_backup_" + date.today().isoformat())
    if not bak.exists():
        shutil.copytree(SNAP, bak)
        print("backup ->", bak)

    idx_path = SNAP / "index.json"
    idx = json.loads(idx_path.read_text())

    changed = 0
    for d in idx.get("days", []):
        day = d["date"]
        fp = SNAP / d["file"]
        if not fp.exists():
            continue
        doc = json.loads(fp.read_text())
        systems = doc.get("systems", {})
        combined_2lot = 0.0
        for k, sysdoc in systems.items():
            info = day_size(k, day)
            if info is None:
                sysdoc["scale"] = 1.0
                sysdoc["lots"] = None
                combined_2lot += float(sysdoc.get("last") or 0)
                continue
            qty, lots, scale, mode = info
            sysdoc["qty"] = qty
            sysdoc["lots"] = lots
            sysdoc["scale"] = scale
            sysdoc["mode"] = mode
            combined_2lot += float(sysdoc.get("last") or 0) * scale
        # Rebuild the whole-book curve on the 2-lot basis. Mirrors
        # dump_nas_mtm._combined_curve: union of ts, forward-fill each system, sum --
        # but each system's value scaled to 2 lots first. On a mixed day (2-lot live +
        # 10-lot paper) the scale differs per system, so a single day-level ratio would
        # distort the intraday shape. This does not.
        doc["combined_2lot"] = _combined_2lot(systems)
        fp.write_text(json.dumps(doc, separators=(",", ":")))
        d["combined_last_2lot"] = round(float(doc["combined_2lot"]["last"]), 2)
        changed += 1

    idx["normalized_to_lots"] = TARGET_LOTS
    idx_path.write_text(json.dumps(idx, indent=2))
    print("annotated %d day files" % changed)

    print("\n%-12s %12s %12s   sizes traded" % ("DAY", "AS-TRADED", "PER-2-LOT"))
    for d in idx["days"][:14]:
        fp = SNAP / d["file"]
        doc = json.loads(fp.read_text())
        sizes = sorted({("%s:%sL" % (k.replace("nas-", ""), int(v["lots"])))
                        for k, v in doc.get("systems", {}).items()
                        if v.get("lots") and float(v.get("last") or 0) != 0})
        print("%-12s %12s %12s   %s" % (
            d["date"],
            format(d["combined_last"], "+,.0f"),
            format(d.get("combined_last_2lot", 0), "+,.0f"),
            " ".join(sizes) or "-"))


if __name__ == "__main__":
    main()
