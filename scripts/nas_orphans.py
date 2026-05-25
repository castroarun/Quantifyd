"""Show the 3 unbalanced strangles with exact symbol/qty + the rejected siblings."""
import sqlite3
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

print(f"{'Variant':<10} {'Strangle':<10} {'OPEN leg (BUY to close)':<40} {'Qty':>5}   {'REJECTED sibling':<35}")
print("-" * 110)
for name, path, tab in DBS:
    full = ROOT / path
    if not full.exists():
        continue
    con = sqlite3.connect(str(full))
    sids = [r[0] for r in con.execute(
        f"SELECT DISTINCT strangle_id FROM {tab} "
        f"WHERE date(entry_time)=date('now','localtime') AND strangle_id IS NOT NULL"
    )]
    for sid in sorted(sids):
        legs = con.execute(
            f"SELECT leg, tradingsymbol, qty, entry_price, status FROM {tab} "
            f"WHERE strangle_id=? AND date(entry_time)=date('now','localtime') "
            f"ORDER BY id",
            (sid,),
        ).fetchall()
        actives = [l for l in legs if l[4] == 'ACTIVE']
        if len(actives) == 1:
            # Unbalanced
            survivor = actives[0]
            survivor_sym = survivor[1]
            survivor_qty = survivor[2]
            survivor_leg = survivor[0]
            other = [l for l in legs if l[4] != 'ACTIVE']
            other_str = ', '.join(f"{l[1]} ({l[4]})" for l in other) if other else "none"
            print(f"{name:<10} #{sid:<9} {survivor_sym:<40} {survivor_qty:>5}   {other_str:<35}")
    con.close()
