"""GO-LIVE RESET — wipe the paper book and re-base it for a Rs3L LIVE allocation.

Run ONLY after 15:30 IST, on the same restart that loads the Rs3L shared-account config.

Why a reset is mandatory: the paper book holds 6 positions worth ~Rs15.2L that DO NOT EXIST at the
broker, on a Rs20L paper capital. Carrying that ledger into a Rs3L live account would make the book
believe it owns stock it does not, and every Donchian stop would try to sell shares that (if they exist
at all) are Arun's personal holdings.

Safety: this script does NOT arm live mode and places NO orders. It only re-bases the ledger.
Pass --confirm to actually write; without it, it prints what it would do.
"""
import sys, json
sys.path.insert(0, "/home/arun/quantifyd")
from datetime import date
from services import momentum_paper as mp

CONFIRM = "--confirm" in sys.argv
NEW_CAPITAL = 300_000

mp.init_db()
st = mp.get_state()
c = mp._conn()
pos = mp._positions()
print("=" * 72)
print(f"GO-LIVE RESET  ({'WRITING' if CONFIRM else 'DRY RUN — pass --confirm to apply'})")
print("=" * 72)
print(f"\nCURRENT (paper) BOOK")
print(f"  mode              {st['mode']}   live_armed={mp._get('live_armed','0')}")
print(f"  capital           Rs{st['capital']:,}")
print(f"  NAV               Rs{st['nav']:,}")
print(f"  positions         {len(pos)}  -> {', '.join(sorted(pos)) if pos else '(none)'}")
print(f"  cash              Rs{st['cash']:,}")
print(f"  closed trades     {len(st.get('closed', []))}")
print(f"\nWILL BECOME")
print(f"  capital           Rs{NEW_CAPITAL:,}  (live, inside the shared account)")
print(f"  positions         0   (paper holdings cleared — they never existed at the broker)")
print(f"  cash              Rs{NEW_CAPITAL:,}")
print(f"  mode              PAPER  (live arming is a SEPARATE, deliberate step)")
print(f"  product           {mp.CFG['order_product']}   hedge={mp.CFG['hedge_enabled']}   "
      f"shared_account={mp.CFG['shared_account']}")

if not CONFIRM:
    print("\nDRY RUN — nothing written. Re-run with --confirm after 15:30 IST.")
    sys.exit(0)

# archive the paper history so it is not lost
arch = f"/home/arun/quantifyd/backtest_data/momentum_paper_archive_{date.today().isoformat()}.json"
json.dump(dict(state=st, positions={k: dict(v) for k, v in pos.items()}), open(arch, "w"), default=str)
print(f"\narchived paper history -> {arch}")

c.execute("DELETE FROM mp_positions")
c.execute("DELETE FROM mp_fills")
c.execute("DELETE FROM mp_closed")
c.execute("DELETE FROM mp_nav")
try:
    c.execute("DELETE FROM mp_hedge")
    c.execute("DELETE FROM mp_hedge_closed")
except Exception:
    pass
c.commit()
mp._set("capital", float(NEW_CAPITAL))
mp._set("cash", float(NEW_CAPITAL))
mp._set("interest_earned", 0.0)
mp._set("sweep_units", 0.0)
mp._set("live_mode", "0")          # stays PAPER until explicitly armed
mp._set("live_armed", "0")
mp._set("seeded", "")              # allow a fresh seed
mp._set("inception", date.today().isoformat())
print("ledger re-based to Rs3L, positions cleared, mode left at PAPER (unarmed).")
print("\nNEXT STEPS (deliberate, separate):")
print("  1. verify: curl -s localhost:5000/api/momentum-paper/state")
print("  2. arm:    curl -X POST localhost:5000/api/momentum-paper/toggle-mode \\")
print("               -H 'Content-Type: application/json' \\")
print("               -d '{\"live\":true,\"arm\":true,\"capital\":300000}'")
print("  3. seed during market hours tomorrow (~15:00 IST) -> 8 real BUY orders of ~Rs37.5k")
