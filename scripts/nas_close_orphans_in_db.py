"""Mark the 3 known orphan CE legs as CLOSED in DB only (user closes on Kite).

Touches EXACTLY 3 rows:
  Sq OTM  #26  (nas_trading.db)
  Sq ATM  #40  (nas_atm_trading.db)
  Sq ATM4 #18  (nas_atm4_trading.db)

The orphans were created when sibling PE orders got REJECTED for insufficient
funds at the exchange. We do not have the Kite buyback price yet — exit_price
stays NULL; the user fills the CE on Kite separately. P&L stats for these 3
will read 0/blank in the DB until manually stitched.
"""
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (DB path, table, strangle_id, leg, expected_tradingsymbol, expected_qty)
TARGETS = [
    ("backtest_data/nas_trading.db",       "nas_positions",     26, "CE", "NIFTY26MAY24250CE", 130),
    ("backtest_data/nas_atm_trading.db",   "nas_atm_positions", 40, "CE", "NIFTY26MAY24000CE",  65),
    ("backtest_data/nas_atm4_trading.db",  "nas_atm_positions", 18, "CE", "NIFTY26MAY24000CE",  65),
]

EXIT_REASON = "MANUAL_USER_CLOSE_OFFAPP"
NOTE = "Closed off-app by user; sibling PE rejected for low funds"

print("=== BEFORE ===")
for path, tab, sid, leg, exp_sym, exp_qty in TARGETS:
    full = ROOT / path
    con = sqlite3.connect(str(full))
    row = con.execute(
        f"SELECT id, status, tradingsymbol, qty, entry_price FROM {tab} "
        f"WHERE strangle_id=? AND leg=? AND status='ACTIVE'",
        (sid, leg),
    ).fetchone()
    if row is None:
        print(f"  {path} #{sid}{leg}: no matching ACTIVE row — already closed? Skipping.")
    else:
        row_id, status, tsym, qty, entry = row
        ok_sym = (tsym == exp_sym)
        ok_qty = (qty == exp_qty)
        flag = "✓" if (ok_sym and ok_qty) else "!! MISMATCH"
        print(f"  {path:<40s} row={row_id:>4d} #{sid} {leg} {tsym} qty={qty} entry={entry} {flag}")
    con.close()

ans = input("\nProceed with UPDATE on these 3 rows? (type YES): ").strip()
if ans != "YES":
    print("Aborted. No changes made.")
    raise SystemExit(1)

print()
print("=== UPDATING ===")
total = 0
for path, tab, sid, leg, exp_sym, exp_qty in TARGETS:
    full = ROOT / path
    con = sqlite3.connect(str(full))
    cur = con.execute(
        f"UPDATE {tab} SET status='CLOSED', exit_reason=?, "
        f"exit_time=datetime('now','localtime'), notes=? "
        f"WHERE strangle_id=? AND leg=? AND status='ACTIVE' "
        f"AND tradingsymbol=? AND qty=?",
        (EXIT_REASON, NOTE, sid, leg, exp_sym, exp_qty),
    )
    print(f"  {path:<40s} #{sid} {leg}: {cur.rowcount} row(s) updated")
    total += cur.rowcount
    con.commit()
    con.close()

print()
print(f"=== AFTER ===  total rows updated: {total}")
for path, tab, sid, leg, exp_sym, exp_qty in TARGETS:
    full = ROOT / path
    con = sqlite3.connect(str(full))
    row = con.execute(
        f"SELECT id, status, exit_reason, exit_time, notes FROM {tab} "
        f"WHERE strangle_id=? AND leg=? AND tradingsymbol=?",
        (sid, leg, exp_sym),
    ).fetchone()
    if row:
        print(f"  {path:<40s} #{sid} {leg}: status={row[1]} reason={row[2]} exit_time={row[3]}")
    con.close()
