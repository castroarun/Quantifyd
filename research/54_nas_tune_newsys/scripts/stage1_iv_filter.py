"""research/54 Stage 1 — IV-LEVEL FILTER on the real recorded NIFTY chain.

Question: does entry ATM-IV level separate winning from losing short-straddle days?
Classic short-vol edge = "sell when vol is rich." Never tested in research/52.

Engine reused from research/52/scripts/scan.py (real-chain replay). Short ATM straddle,
enter 09:20, ±0.4% underlying-move stop (research/52 finding #3), time-exit 14:45.
Per day record (dte, pnl, entry ATM-IV, opening-range%). Then:
  - corr(entry ATM-IV, pnl)  [single honest signal, all days]
  - P&L by IV tercile (all-DTE; flag DTE confound + tiny n)
  - combined 1-DTE + high-IV + tight-open real-chain number
28-29 days => SIGNAL. Net of ₹80/leg. Read gradient + tail, not peak.
"""
import sqlite3
from pathlib import Path
from datetime import time as dtime
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/arun/quantifyd"); BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results"; OUT.mkdir(parents=True, exist_ok=True)
OPT = ROOT / "backtest_data" / "options_data.db"
LOT = 65; QTY = LOT * 2; BROK = 80
ENTRY = dtime(9, 20); TIMEEXIT = dtime(14, 45); EOD = dtime(15, 15)
oc = sqlite3.connect(str(OPT))
oc.execute("CREATE INDEX IF NOT EXISTS idx_oc_tsym_time ON option_chain(tradingsymbol, snapshot_time)")
DAYS = [r[0] for r in oc.execute("SELECT DISTINCT substr(snapshot_time,1,10) FROM option_chain WHERE symbol='NIFTY' ORDER BY 1")]

_cache = {}
def load(day):
    if day in _cache: return _cache[day]
    df = pd.read_sql_query(
        "SELECT snapshot_time,tradingsymbol,strike,instrument_type,ltp,iv,expiry_date,underlying_spot "
        "FROM option_chain WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? AND ltp IS NOT NULL", oc, params=[day])
    if df.empty: _cache[day] = None; return None
    df["t"] = pd.to_datetime(df["snapshot_time"])
    exps = sorted(df["expiry_date"].unique()); fut = [e for e in exps if e >= day]
    exp = fut[0] if fut else exps[-1]
    df = df[df["expiry_date"] == exp]
    dte = (pd.to_datetime(exp).date() - pd.to_datetime(day).date()).days
    spot = df.groupby("t")["underlying_spot"].first().sort_index()
    chain = {}
    for ts, g in df.groupby("tradingsymbol"):
        g = g.sort_values("t")
        chain[ts] = (g["t"].values, g["ltp"].values, g["iv"].values, g["strike"].iloc[0], g["instrument_type"].iloc[0])
    _cache[day] = (dte, spot, chain); return _cache[day]

def at(chain, ts, t, idx=1):
    if ts not in chain: return None
    ta, la, iva, _, _ = chain[ts]; i = np.searchsorted(ta, np.datetime64(t), side="right") - 1
    if i < 0: return None
    arr = la if idx == 1 else iva; v = arr[i]
    return float(v) if v and v > 0 else None

def tsym(chain, strike, typ):
    for ts, (_, _, _, st, ty) in chain.items():
        if int(st) == int(strike) and ty == typ: return ts
    return None

def opening_range_pct(spot_s):
    op = spot_s[(spot_s.index.time >= dtime(9, 15)) & (spot_s.index.time <= dtime(9, 30))]
    if len(op) < 2: return None
    return (op.max() - op.min()) / op.iloc[0] * 100

def sim(day, stop_mode="move", move_pct=0.4, sl_mult=1.3):
    d = load(day)
    if d is None: return None
    dte, spot_s, chain = d
    times = [t for t in spot_s.index if ENTRY <= t.time() <= EOD]
    if not times: return None
    t0 = times[0]; spot0 = float(spot_s.loc[t0]); atm = round(spot0 / 50) * 50
    legs, ivs = [], []
    for typ in ("CE", "PE"):
        ts = tsym(chain, atm, typ); p = at(chain, ts, t0, 1) if ts else None
        iv = at(chain, ts, t0, 2) if ts else None
        if p: legs.append({"ts": ts, "entry": p, "sl": p * sl_mult, "open": True})
        if iv: ivs.append(iv)
    if len(legs) < 2: return None
    atm_iv = float(np.mean(ivs)) if ivs else None
    pnl = 0.0
    for t in times:
        if not any(l["open"] for l in legs): break
        force = t.time() >= TIMEEXIT
        moved = abs(float(spot_s.loc[t]) - spot0) / spot0 * 100 >= move_pct
        for lg in legs:
            if not lg["open"]: continue
            p = at(chain, lg["ts"], t, 1)
            if p is None: continue
            hit = force or (stop_mode == "move" and moved) or (stop_mode == "prem" and p >= lg["sl"])
            if hit:
                pnl += (lg["entry"] - p) * QTY - BROK; lg["open"] = False
    lastt = times[-1]
    for lg in legs:
        if lg["open"]:
            p = at(chain, lg["ts"], lastt, 1) or lg["entry"]
            pnl += (lg["entry"] - p) * QTY - BROK
    return dict(day=day, dte=dte, pnl=pnl, atm_iv=atm_iv, orpct=opening_range_pct(spot_s), spot0=spot0)

print("running stage1 IV-filter over %d NIFTY days..." % len(DAYS), flush=True)
rows = []
for day in DAYS:
    r = sim(day, "move", 0.4)
    if r and r["atm_iv"]: rows.append(r)
    print("  %s dte=%s pnl=%s iv=%s" % (day, r["dte"] if r else "-",
          round(r["pnl"]) if r else "-", round(r["atm_iv"], 1) if r and r["atm_iv"] else "-"), flush=True)
df = pd.DataFrame(rows)
df.to_csv(OUT / "stage1_iv_perday.csv", index=False)

corr_all = df["atm_iv"].corr(df["pnl"])
ormed = df["orpct"].median()
df["ivbucket"] = pd.qcut(df["atm_iv"], 3, labels=["low", "mid", "high"])
tab = df.groupby("ivbucket")["pnl"].agg(["count", "sum", "mean"])
d1 = df[df["dte"] <= 1]
corr_1 = d1["atm_iv"].corr(d1["pnl"]) if len(d1) > 2 else float("nan")

# combined real-chain stacks (tiny n, flagged)
def book(sub): a = sub["pnl"].values; return (len(a), round(a.sum()), round(a.mean()) if len(a) else 0, round(a.min()) if len(a) else 0)
all_book = book(df)
oneDTE = book(df[df["dte"] <= 1])
hi_iv = book(df[df["ivbucket"] == "high"])
tight = book(df[df["orpct"] < ormed])
stack = book(df[(df["dte"] <= 1) & (df["atm_iv"] >= df["atm_iv"].median()) & (df["orpct"] < ormed)])

L = ["# research/54 Stage 1 — IV-level filter (real NIFTY chain, %d days, ±0.4%% move stop)\n" % len(df),
     "Short ATM straddle, enter 09:20, ±0.4%% underlying-move stop, exit 14:45. Net ₹80/leg. "
     "**28-29 days => SIGNAL.** IV = entry ATM (avg CE+PE) implied vol from the chain.\n",
     "## Headline signal",
     "- **corr(entry ATM-IV, day P&L) = %.2f** (all %d days); 1-DTE-only corr = %.2f (n=%d)" % (
         corr_all, len(df), corr_1, len(d1)),
     "- Positive corr => richer entry vol -> better short-straddle day (the 'sell rich vol' edge).\n",
     "## P&L by entry-IV tercile (all-DTE; flag DTE confound + small n)",
     "| IV bucket | n | total ₹ | ₹/day |", "|---|---|---|---|"]
for b in ["low", "mid", "high"]:
    if b in tab.index: L.append("| %s | %d | %d | %d |" % (b, tab.loc[b, "count"], tab.loc[b, "sum"], tab.loc[b, "mean"]))
L += ["\n## Real-chain books (n / total ₹ / ₹-day / worst-day) — tiny n, SIGNAL",
      "| book | n | total ₹ | ₹/day | worst ₹ |", "|---|---|---|---|---|",
      "| all days | %d | %d | %d | %d |" % all_book,
      "| 1-DTE only | %d | %d | %d | %d |" % oneDTE,
      "| high-IV tercile | %d | %d | %d | %d |" % hi_iv,
      "| tight-open only | %d | %d | %d | %d |" % tight,
      "| **1-DTE + IV>=med + tight-open** | %d | %d | %d | %d |" % stack,
      "\n- DTE dominates short-vol P&L (finding #1), so the all-DTE tercile mixes DTEs — read corr + the stacked book.",
      "- If high-IV total > low-IV AND corr>0, the IV filter adds to the stack. Confirm as the recorder grows."]
(OUT / "RESULTS_iv_filter.md").write_text("\n".join(L), encoding="utf-8")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sc = ax[0].scatter(df["atm_iv"], df["pnl"], c=df["dte"], cmap="viridis", s=40)
ax[0].axhline(0, color="#999"); ax[0].set_xlabel("entry ATM IV %"); ax[0].set_ylabel("day P&L ₹")
ax[0].set_title("IV vs P&L (color=DTE)  corr=%.2f" % corr_all); fig.colorbar(sc, ax=ax[0], label="DTE")
vals = [tab.loc[b, "sum"] if b in tab.index else 0 for b in ["low", "mid", "high"]]
ax[1].bar(["low", "mid", "high"], vals, color=["#d33" if v < 0 else "#0a6" for v in vals])
ax[1].axhline(0, color="#333"); ax[1].set_title("Total ₹ by entry-IV tercile (all-DTE)")
fig.suptitle("research/54 Stage 1 — IV-level filter, real NIFTY chain (%d days, SIGNAL)" % len(df))
fig.tight_layout(); fig.savefig(OUT / "stage1_iv_filter.png", dpi=110, bbox_inches="tight")
print("DONE  corr_all=%.2f  buckets=%s" % (corr_all, vals), flush=True)
oc.close()
