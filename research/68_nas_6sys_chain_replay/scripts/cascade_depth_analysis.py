"""Cascade-depth analysis (research/68) — find the optimal/max number of re-centers for the
ATM2 move-stop systems. Each day per system the strangles form one cascade chain: seq 0 =
the original entry, seq 1 = the 1st re-center (after a 0.4% move-stop), seq 2 = the 2nd, etc.
This reads the ATM2 paper trades, derives the cascade sequence from the per-day entry order,
and aggregates realized P&L per cascade level across ALL paper days. Run anytime — it fills
in as the paper book accumulates, and the level where the marginal avg P&L turns negative is
where to cap. Pure read-only. (Cap is currently max_reentries=5/day.)"""
import sqlite3, collections
from pathlib import Path

ROOT = Path("/home/arun/quantifyd")
DBS = [("nas_atm2_trading.db", "Squeeze-ATM2"), ("nas_916_atm2_trading.db", "916-ATM2")]
QTY = 65  # 1 lot, paper

# agg[seq] across both systems; per[lab][seq] per system
agg = collections.defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
per = collections.defaultdict(lambda: collections.defaultdict(lambda: {"n": 0, "pnl": 0.0}))
days_seen = set()

for dbf, lab in DBS:
    p = ROOT / "backtest_data" / dbf
    if not p.exists():
        continue
    c = sqlite3.connect(str(p)); c.row_factory = sqlite3.Row
    legs = c.execute("SELECT strangle_id, entry_price, exit_price, entry_time, date(entry_time) d "
                     "FROM nas_atm_positions WHERE exit_price IS NOT NULL AND strangle_id IS NOT NULL").fetchall()
    strangles = collections.defaultdict(lambda: {"pnl": 0.0, "et": None, "d": None})
    for r in legs:
        s = strangles[r["strangle_id"]]
        s["pnl"] += (r["entry_price"] - r["exit_price"]) * QTY
        if s["et"] is None or r["entry_time"] < s["et"]:
            s["et"] = r["entry_time"]
        s["d"] = r["d"]
    byday = collections.defaultdict(list)
    for sid, s in strangles.items():
        byday[s["d"]].append((s["et"], s["pnl"]))
        days_seen.add(s["d"])
    for d, lst in byday.items():
        lst.sort()
        for seq, (et, pnl) in enumerate(lst):
            a = agg[seq]; a["n"] += 1; a["pnl"] += pnl; a["wins"] += (1 if pnl > 0 else 0)
            per[lab][seq]["n"] += 1; per[lab][seq]["pnl"] += pnl

print("=== ATM2 CASCADE-DEPTH P&L (paper, 1 lot = 65/leg) — %d paper days ===" % len(days_seen))
print("seq #0 = original entry; #1+ = re-centers after a 0.4%% move-stop. Cap = 5/day.\n")
print("%-8s %7s %11s %10s %7s | %12s" % ("cascade", "trades", "total Rs", "avg Rs", "win%", "cum total Rs"))
cum = 0.0
for seq in sorted(agg):
    a = agg[seq]; avg = a["pnl"] / a["n"] if a["n"] else 0; cum += a["pnl"]
    print("  #%-5d %7d %11.0f %10.0f %6.0f%% | %12.0f" % (seq, a["n"], a["pnl"], avg, 100 * a["wins"] / max(a["n"], 1), cum))
print("\nPer system (avg Rs by cascade seq):")
for lab in per:
    row = "  %-12s" % lab + " ".join("#%d:%+.0f(n%d)" % (sq, per[lab][sq]["pnl"] / max(per[lab][sq]["n"], 1), per[lab][sq]["n"]) for sq in sorted(per[lab]))
    print(row)
print("\n>> Where the marginal cascade's AVG turns negative (and stays) = the optimal cap.")
print("   Sparse now (paper book just started); fills in over weeks.")
