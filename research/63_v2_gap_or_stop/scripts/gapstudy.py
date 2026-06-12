"""Gap-day frequency + underlying-behaviour study for the V2 gap-OR stop overlay.
Read-only. NIFTY daily from Kite (tok 256265). No options data needed for this first-order read."""
import sys, json, os
from datetime import date
sys.path.insert(0, "/home/arun/quantifyd")
from kiteconnect import KiteConnect

ak = os.environ.get("KITE_API_KEY")
try:
    import config; ak = ak or getattr(config, "KITE_API_KEY", None)
except Exception:
    pass
tokj = json.load(open("/home/arun/quantifyd/backtest_data/access_token.json"))
at = tokj.get("access_token") if isinstance(tokj, dict) else tokj
ak = ak or (tokj.get("api_key") if isinstance(tokj, dict) else None)
kite = KiteConnect(api_key=ak); kite.set_access_token(at)

# daily history (years allowed for 'day' interval) — chunk by year to respect API limits
import datetime as dt
rows = []
start = dt.date(2019, 1, 1)
while start < dt.date.today():
    end = min(start + dt.timedelta(days=365), dt.date.today())
    try:
        rows += kite.historical_data(256265, start, end, "day")
    except Exception as e:
        print("chunk fail", start, e)
    start = end + dt.timedelta(days=1)
# dedupe by date
seen = {}
for r in rows:
    seen[r["date"].date()] = r
bars = [seen[k] for k in sorted(seen)]
print(f"NIFTY daily bars: {len(bars)}  {bars[0]['date'].date()} -> {bars[-1]['date'].date()}")

TH = 0.02
gaps = []
for i in range(1, len(bars)):
    pc = bars[i - 1]["close"]; o = bars[i]["open"]
    g = (o - pc) / pc
    if abs(g) >= TH:
        b = bars[i]
        # for a SHORT fly: gap-up -> adverse = further UP (high-open); favourable = revert down (open-low / close)
        ext_adverse = (b["high"] - o) / o if g > 0 else (o - b["low"]) / o      # how much further the gap ran
        revert = (o - b["close"]) / o if g > 0 else (b["close"] - o) / o          # close back toward entry (+ = reverted)
        closed_inside = abs((b["close"] - pc) / pc) < TH                          # day closed back inside prior-close +-2%
        gaps.append(dict(date=str(b["date"].date()), gap_pct=round(g * 100, 2),
                         dir="UP" if g > 0 else "DN",
                         further_run_pct=round(ext_adverse * 100, 2),
                         revert_to_close_pct=round(revert * 100, 2),
                         closed_inside_band=closed_inside))

n = len(bars) - 1
print(f"\n>={TH*100:.0f}% gap-open days: {len(gaps)} / {n} sessions = {100*len(gaps)/n:.2f}%  (~{len(gaps)/((bars[-1]['date'].date()-bars[0]['date'].date()).days/365):.1f}/yr)")
ups = [x for x in gaps if x["dir"] == "UP"]; dns = [x for x in gaps if x["dir"] == "DN"]
print(f"  gap-UP: {len(ups)}   gap-DN: {len(dns)}")
import statistics as st
def summ(name, lst):
    if not lst: return
    fr = [x["further_run_pct"] for x in lst]; rv = [x["revert_to_close_pct"] for x in lst]
    inside = sum(1 for x in lst if x["closed_inside_band"])
    print(f"\n  [{name}] n={len(lst)}")
    print(f"    further run beyond open (adverse for short fly): median {st.median(fr):.2f}%  max {max(fr):.2f}%")
    print(f"    revert to close (favourable):                    median {st.median(rv):.2f}%")
    print(f"    closed back INSIDE prior-close +-2% band:        {inside}/{len(lst)} = {100*inside/len(lst):.0f}%")
summ("ALL gaps", gaps); summ("gap-UP", ups); summ("gap-DN", dns)
print("\n  --- every gap day ---")
for x in gaps:
    print(f"    {x['date']}  {x['dir']} {x['gap_pct']:+.2f}%  further+{x['further_run_pct']:.2f}%  revert {x['revert_to_close_pct']:+.2f}%  inside={x['closed_inside_band']}")
