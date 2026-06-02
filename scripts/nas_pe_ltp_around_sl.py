"""Pull 1-min OHLC of NIFTY26MAY24000PE around the SL fires to estimate
whether the SL hit was a single-tick spike or a sustained move.

Read-only — uses kite.historical_data() against the option symbol.
"""
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.kite_service import get_kite

kite = get_kite()

# Resolve the instrument token for NIFTY26MAY24000PE
instruments = kite.instruments('NFO')
target = None
for inst in instruments:
    if (inst.get('name') == 'NIFTY' and inst.get('strike') == 24000
        and inst.get('instrument_type') == 'PE'
        and str(inst.get('expiry')).startswith('2026-05')):
        target = inst
        break

if not target:
    print("Could not resolve NIFTY26MAY24000PE")
    sys.exit(1)

token = target['instrument_token']
print(f"Resolved {target['tradingsymbol']} -> token={token} expiry={target['expiry']}")
print()

# Get 1-min candles for 12:40 - 12:55 IST today
today = datetime.now().strftime('%Y-%m-%d')
candles = kite.historical_data(
    instrument_token=token,
    from_date=f'{today} 12:30:00',
    to_date=f'{today} 12:55:00',
    interval='minute',
)

# SL trigger reference points from journal
sl_events = [
    ('12:48:55', 'ATM2', 125.06, 125.45),
    ('12:48:57', 'ATM',  127.53, 127.55),
    ('12:50:09', 'ATM',  129.03, 129.55),
    ('12:50:09', 'ATM2', 128.77, 129.55),
]

print(f"{'Time':<10} {'O':>8} {'H':>8} {'L':>8} {'C':>8}  Notes")
print('-' * 70)
for c in candles:
    t = c['date'].strftime('%H:%M:%S')
    note = ''
    for stime, variant, sl, ltp_at_fire in sl_events:
        if stime[:5] == t[:5]:
            note += f' SL[{variant}]={sl:.2f} ltp_at_fire={ltp_at_fire:.2f};'
    print(f"{t:<10} {c['open']:>8.2f} {c['high']:>8.2f} {c['low']:>8.2f} {c['close']:>8.2f}  {note}")
