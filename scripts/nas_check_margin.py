"""Probe Kite margins + active option legs not yet subscribed to ticker."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.kite_service import get_kite

kite = get_kite()
m = kite.margins(segment='equity')
print("EQUITY segment:")
print(f"  net available cash: Rs {m.get('net', 0):,.2f}")
print(f"  available live balance: Rs {m.get('available', {}).get('live_balance', 0):,.2f}")
print(f"  available collateral: Rs {m.get('available', {}).get('collateral', 0):,.2f}")
print(f"  utilised debits: Rs {m.get('utilised', {}).get('debits', 0):,.2f}")
print(f"  utilised exposure: Rs {m.get('utilised', {}).get('exposure', 0):,.2f}")
print(f"  utilised m2m_unrealised: Rs {m.get('utilised', {}).get('m2m_unrealised', 0):,.2f}")

print()
print("Current Kite positions (NFO):")
pos = kite.positions().get('net', [])
for p in pos:
    if p.get('exchange') == 'NFO' and p.get('quantity') != 0:
        print(f"  {p['tradingsymbol']:<25s} qty={p['quantity']:>4d} avg_price={p.get('average_price',0):.2f} pnl={p.get('pnl',0):.2f}")
