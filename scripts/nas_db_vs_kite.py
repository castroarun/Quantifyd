"""Read-only: compare DB ACTIVE legs vs Kite net positions. No trading.

Surfaces any DB row marked ACTIVE that doesn't correspond to a real open
short on Kite (stale ACTIVE), and any Kite short that's missing from DB.
"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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

# Step 1: collect all DB ACTIVE legs
db_active = []  # (variant, sid, leg, tsym, qty, db_path, row_id)
for name, path, tab in DBS:
    full = ROOT / path
    if not full.exists():
        continue
    con = sqlite3.connect(str(full))
    rows = con.execute(
        f"SELECT id, strangle_id, leg, tradingsymbol, qty, transaction_type "
        f"FROM {tab} WHERE status='ACTIVE'"
    ).fetchall()
    for row_id, sid, leg, tsym, qty, txn in rows:
        db_active.append((name, sid, leg, tsym, qty, txn))
    con.close()

# Step 2: collect Kite net positions (short option positions only)
from services.kite_service import get_kite
kite = get_kite()
kite_net = kite.positions().get('net', [])
kite_short = {}  # tradingsymbol -> qty (negative = short)
for p in kite_net:
    if p.get('exchange') == 'NFO' and p.get('quantity') != 0:
        kite_short[p['tradingsymbol']] = p['quantity']

# Step 3: cross-check
print(f"DB ACTIVE legs: {len(db_active)}")
print(f"Kite open NFO positions: {len(kite_short)}")
print()

# Aggregate DB short qty per symbol
db_short = {}
for name, sid, leg, tsym, qty, txn in db_active:
    sign = -1 if txn == 'SELL' else +1
    db_short[tsym] = db_short.get(tsym, 0) + sign * qty

print(f"{'Symbol':<30s} {'DB qty':>10s} {'Kite qty':>10s} {'Match':>10s}")
print("-" * 65)
all_symbols = sorted(set(db_short) | set(kite_short))
mismatches = []
for tsym in all_symbols:
    d = db_short.get(tsym, 0)
    k = kite_short.get(tsym, 0)
    match = "OK" if d == k else "MISMATCH"
    if d != k:
        mismatches.append((tsym, d, k))
    print(f"{tsym:<30s} {d:>10d} {k:>10d} {match:>10s}")

print()
if mismatches:
    print(f"!! {len(mismatches)} SYMBOL(S) DIFFER between DB and Kite — see above.")
    print("    Most likely cause: DB marked ACTIVE but Kite order rejected/cancelled later.")
else:
    print("OK — DB ACTIVE quantity matches Kite net position for every symbol.")
