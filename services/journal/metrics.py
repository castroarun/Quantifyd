"""
Journal-level trade metrics: R-multiple, hold-time, gross/net P&L.

MAE/MFE computation against market_data_unified is a Phase-2 concern;
this module focuses on what we can compute purely from the trade row.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional


def parse_ts(ts) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    s = str(ts)
    # Try common formats
    for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d'):
        try:
            return datetime.strptime(s.split('+')[0], fmt)
        except ValueError:
            continue
    return None


def hold_minutes(entry_time, exit_time) -> Optional[int]:
    a = parse_ts(entry_time)
    b = parse_ts(exit_time)
    if not a or not b:
        return None
    delta = b - a
    return max(0, int(delta.total_seconds() // 60))


def gross_pnl(direction: str, qty: int, entry_price: float, exit_price: float) -> Optional[float]:
    if entry_price is None or exit_price is None or qty is None:
        return None
    diff = (exit_price - entry_price) if (direction or 'LONG').upper() == 'LONG' else (entry_price - exit_price)
    return float(diff) * float(qty)


def initial_risk(direction: str, qty: int, entry_price: float, sl_price: Optional[float]) -> Optional[float]:
    if sl_price is None or entry_price is None or qty is None:
        return None
    risk_per_share = abs(entry_price - sl_price)
    if risk_per_share == 0:
        return None
    return float(risk_per_share) * float(qty)


def r_multiple(pnl_net: Optional[float], risk: Optional[float]) -> Optional[float]:
    if pnl_net is None or risk is None or risk == 0:
        return None
    return round(float(pnl_net) / float(risk), 3)
