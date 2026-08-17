#!/usr/bin/env python3
"""Turn ORB cash back ON in PAPER mode, without resurrecting the noise.

It was switched off 2026-08-10 because it wasn't traded, its 5-minutely
margin alerts spammed, and it competed for cash the momentum book now uses.
In paper mode none of that should apply — but the margin gate was never
paper-aware: with real available margin near zero it would block every
paper signal and fire a high-priority alert per skip. So:

  1. config.py  -> enabled=True (paper_trading_mode already True)
  2. engine     -> skip the margin gate and its alert entirely in paper mode
"""
from pathlib import Path

# ---- 1. config: Off -> Paper ----
cfg = Path('/home/arun/quantifyd/config.py')
s = cfg.read_text()
old = """    'enabled': False,                   # 2026-08-10: OFF — not traded, and its 5-minutely
                                        # margin alerts were spamming while it competed for
                                        # the cash the momentum book now uses."""
new = """    'enabled': True,                    # 2026-08-17: PAPER, daily (Arun). Real cash is not
                                        # touched, so the 08-10 reasons (margin-alert spam,
                                        # competing for the momentum book's cash) no longer
                                        # apply — the margin gate is paper-aware below."""
assert old in s, 'ORB enabled block not found'
cfg.write_text(s.replace(old, new, 1))
print('config.py: ORB cash -> PAPER, enabled')

# ---- 2. engine: make the margin gate paper-aware ----
eng = Path('/home/arun/quantifyd/services/orb_live_engine.py')
e = eng.read_text()

old_gate = """                if available is not None and available < min_balance_required:
                    logger.warning(
                        f"[ORB] {sym}: LOW FUNDS — available Rs {available:.0f} < "
                        f"required Rs {min_balance_required:.0f} (1.2x of {alloc}). SKIPPING TRADE."
                    )"""
new_gate = """                # Paper mode spends no real cash: a near-zero broker balance
                # must not block paper signals or raise margin alerts.
                _paper = bool(self.cfg.get('paper_trading_mode', True))

                if (not _paper) and available is not None and available < min_balance_required:
                    logger.warning(
                        f"[ORB] {sym}: LOW FUNDS — available Rs {available:.0f} < "
                        f"required Rs {min_balance_required:.0f} (1.2x of {alloc}). SKIPPING TRADE."
                    )"""
assert old_gate in e, 'margin gate not found'
e = e.replace(old_gate, new_gate, 1)

old_alert = """    def _check_fund_alert(self):
        \"\"\"Check if available margin is below 1.2x allocation threshold.\"\"\"
        if not self._last_margin:
            return None"""
new_alert = """    def _check_fund_alert(self):
        \"\"\"Check if available margin is below 1.2x allocation threshold.

        Silent in paper mode — no real margin is consumed there, so the
        alert only produced noise (the reason ORB was switched off 08-10).
        \"\"\"
        if self.cfg.get('paper_trading_mode', True):
            return None
        if not self._last_margin:
            return None"""
assert old_alert in e, 'fund alert fn not found'
e = e.replace(old_alert, new_alert, 1)
eng.write_text(e)
print('orb_live_engine.py: margin gate + fund alert are now paper-aware')
