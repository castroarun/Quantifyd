"""One-shot NAS state probe — reads each variant's DB and prints counts.

Usage: python3 scripts/nas_state_probe.py
"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DBS = [
    ("Sq OTM",   "backtest_data/nas_trading.db",          "nas_positions"),
    ("Sq ATM",   "backtest_data/nas_atm_trading.db",      "nas_atm_positions"),
    ("Sq ATM2",  "backtest_data/nas_atm2_trading.db",     "nas_atm_positions"),
    ("Sq ATM4",  "backtest_data/nas_atm4_trading.db",     "nas_atm_positions"),
    ("916 OTM",  "backtest_data/nas_916_otm_trading.db",  "nas_positions"),
    ("916 ATM",  "backtest_data/nas_916_atm_trading.db",  "nas_atm_positions"),
    ("916 ATM2", "backtest_data/nas_916_atm2_trading.db", "nas_atm_positions"),
    ("916 ATM4", "backtest_data/nas_916_atm4_trading.db", "nas_atm_positions"),
]

print("Variant     ACTIVE  PENDING  FAILED   open strangles (legs)")
print("-" * 78)

grand_active = 0
unbalanced = []

for name, path, tab in DBS:
    full = ROOT / path
    if not full.exists():
        print(f"{name:<10} MISSING DB")
        continue
    con = sqlite3.connect(str(full))
    counts = {}
    q1 = f"SELECT status, COUNT(*) FROM {tab} WHERE date(entry_time)=date('now','localtime') GROUP BY status"
    for status, n in con.execute(q1):
        counts[status] = n
    by_sid = {}
    q2 = f"SELECT strangle_id, leg, tradingsymbol FROM {tab} WHERE status='ACTIVE'"
    for sid, leg, tsym in con.execute(q2):
        by_sid.setdefault(sid, []).append((leg, tsym))
    leg_summary = []
    for sid in sorted(by_sid):
        legs = sorted(by_sid[sid])
        leg_str = "+".join(l[0] for l in legs)
        leg_summary.append(f"#{sid}({leg_str})")
        has_ce = any(l[0] == "CE" for l in legs)
        has_pe = any(l[0] == "PE" for l in legs)
        if not (has_ce and has_pe):
            unbalanced.append((name, sid, legs))
    a = counts.get("ACTIVE", 0)
    p = counts.get("PENDING", 0)
    f = counts.get("FAILED", 0)
    legs_col = " ".join(leg_summary)
    print(f"{name:<10}  {a:>5}   {p:>5}   {f:>5}   {legs_col}")
    grand_active += a
    con.close()

print()
print(f"GRAND TOTAL ACTIVE legs across 8 variants: {grand_active}")
if unbalanced:
    print()
    print("UNBALANCED STRANGLES (missing CE or PE):")
    for v, sid, legs in unbalanced:
        print(f"  {v} #{sid}: {legs}")
else:
    print("All open strangles are balanced (CE + PE both ACTIVE).")
