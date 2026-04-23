"""One-shot: market-exit the 3 ORB losing positions whose 14:30 strict
breakeven SL failed because the would-be SL trigger was on the wrong
side of LTP. Equivalent to the backtest's 'SL fires at entry' outcome.
"""
import sys
sys.path.insert(0, '/home/arun/quantifyd')

from services.orb_live_engine import ORBLiveEngine
from config import ORB_DEFAULTS

engine = ORBLiveEngine(ORB_DEFAULTS)

TARGETS = ['TATASTEEL', 'HAL', 'BAJFINANCE']

for sym in TARGETS:
    positions = engine.db.get_open_positions(instrument=sym)
    if not positions:
        print(f'{sym}: no open position in DB — skip')
        continue
    ltps = engine.get_live_ltp([sym])
    ltp = ltps.get(sym)
    if ltp is None:
        print(f'{sym}: no LTP — skip')
        continue
    for pos in positions:
        print(f'{sym}: exiting {pos["direction"]} qty={pos["qty"]} @ market '
              f'(LTP {ltp}, entry {pos["entry_price"]})')
        engine.place_exit_order(pos, ltp, 'V9T_LOCK50_BE_FORCED')
