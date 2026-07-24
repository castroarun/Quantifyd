"""Validate: fetch daily bars LIVE from Kite (in-memory, no DB) and recompute the 20-day high.
If donch comes out ~current (not the stale 1050), the live-fetch fix is correct."""
import pandas as pd
from datetime import datetime, timedelta
from services.kite_service import get_kite

kite = get_kite()
insts = kite.instruments("NSE")
tok = {i['tradingsymbol']: i['instrument_token'] for i in insts}
to_d = datetime.now()
from_d = to_d - timedelta(days=120)
for sym in ("CARBORUNIV", "AIAENG"):
    t = tok.get(sym)
    if not t:
        print(sym, "no token"); continue
    data = kite.historical_data(t, from_d, to_d, 'day')
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df = df.sort_values('date').set_index('date')
    donch = float(df['high'].iloc[-21:-1].max())   # 20-day high excl today
    print(f"{sym}: live bars={len(df)} | latest {df.index[-1].date()} close {df['close'].iloc[-1]:.1f} "
          f"| LIVE 20-day high = {donch:.1f}  (stale DB said 1050/4110)")
