"""Probe: what NIFTY futures history can we actually get, and what's the real basis?
Reads creds from .env + access_token.json (never prints secrets). Also tests NSE bhavcopy reach."""
import os, sys, json, datetime as dt

ROOT = "/home/arun/quantifyd"
# load .env
env = {}
for line in open(os.path.join(ROOT, ".env")):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1); env[k.strip()] = v.strip()
API_KEY = env.get("KITE_API_KEY", "")
tok = json.load(open(os.path.join(ROOT, "backtest_data", "access_token.json")))["access_token"]

from kiteconnect import KiteConnect
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(tok)

try:
    prof = kite.profile()
    print("Kite OK, user:", prof.get("user_shortname", "?"))
except Exception as e:
    print("Kite auth FAILED:", str(e)[:120]); sys.exit(0)

# find NIFTY futures contracts
insts = kite.instruments("NFO")
nifty_fut = [i for i in insts if i["name"] == "NIFTY" and i["instrument_type"] == "FUT"]
nifty_fut.sort(key=lambda x: x["expiry"])
print(f"\nNIFTY FUT contracts available: {len(nifty_fut)}")
for i in nifty_fut[:4]:
    print(f"  {i['tradingsymbol']}  expiry={i['expiry']}  token={i['instrument_token']}")

# spot token
nse = kite.instruments("NSE")
spot = [i for i in nse if i["tradingsymbol"] == "NIFTY 50"]
spot_tok = spot[0]["instrument_token"] if spot else 256265
print("NIFTY 50 spot token:", spot_tok)

# how far back does history go for the near-month future?
near = nifty_fut[0]
today = dt.date(2026, 7, 8)
for yrs_back in [0.25, 1, 3, 6, 11]:
    frm = today - dt.timedelta(days=int(yrs_back * 365))
    try:
        h = kite.historical_data(near["instrument_token"], frm, today, "day")
        if h:
            print(f"  near-fut hist from {frm}: {len(h)} bars, first={h[0]['date'].date()} last={h[-1]['date'].date()}")
        else:
            print(f"  near-fut hist from {frm}: EMPTY")
    except Exception as e:
        print(f"  near-fut hist from {frm}: ERR {str(e)[:80]}")

# current basis: near future close vs spot close (last ~5 days)
try:
    fh = kite.historical_data(near["instrument_token"], today - dt.timedelta(days=10), today, "day")
    sh = kite.historical_data(spot_tok, today - dt.timedelta(days=10), today, "day")
    sd = {x["date"].date(): x["close"] for x in sh}
    print("\nRecent basis (near-month future vs spot):")
    for x in fh[-5:]:
        d = x["date"].date(); s = sd.get(d)
        if s:
            b = 100 * (x["close"] / s - 1)
            print(f"  {d}  fut={x['close']:.1f} spot={s:.1f}  basis={b:+.2f}%")
except Exception as e:
    print("basis calc ERR:", str(e)[:100])

# NSE bhavcopy reachability (a COVID-crash date) — old archive format
print("\n--- NSE bhavcopy reachability test (2020-03-24 COVID crash) ---")
import urllib.request
url = "https://archives.nseindia.com/content/historical/DERIVATIVES/2020/MAR/fo24MAR2020bhav.csv.zip"
req = urllib.request.Request(url, headers={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "*/*", "Referer": "https://www.nseindia.com/"})
try:
    r = urllib.request.urlopen(req, timeout=15)
    data = r.read()
    print(f"  OLD archive: HTTP {r.status}, {len(data)} bytes  -> {'OK' if len(data)>1000 else 'too small'}")
except Exception as e:
    print("  OLD archive ERR:", str(e)[:120])
url2 = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_20260707_F_0000.csv.zip"
req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
try:
    r2 = urllib.request.urlopen(req2, timeout=15)
    print(f"  NEW UDiFF (recent): HTTP {r2.status}, {len(r2.read())} bytes")
except Exception as e:
    print("  NEW UDiFF ERR:", str(e)[:120])
print("DONE")
