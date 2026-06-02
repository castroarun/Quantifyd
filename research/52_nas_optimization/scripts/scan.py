"""Part (a): DISCIPLINED one-factor sensitivity of a short straddle/strangle on the
recorded NIFTY chain. NOT a brute-force grid — we vary ONE axis at a time around a
sensible base and read the GRADIENT (monotonic > peak). DTE is a search axis (0..6).

Base: enter 09:20, sell ATM CE+PE (lots=2), per-leg SL = entry x 1.3, on a leg-SL the
surviving leg switches to ST(7,2) exit, else time-exit 14:45. All real chain premiums.
28 days => SIGNAL not validation; multiple-testing caveat (each axis has few cells, and
each P&L is one 28-day path).
"""
import sqlite3, math
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

def st_up(ps, period=7, mult=2.0):
    o = ps.resample("5min").agg(["first", "max", "min", "last"]).dropna()
    if len(o) < period + 1: return pd.Series(False, index=o.index)
    o.columns = ["open", "high", "low", "close"]; pc = o["close"].shift(1)
    tr = pd.concat([o["high"]-o["low"], (o["high"]-pc).abs(), (o["low"]-pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean(); hl2 = (o["high"]+o["low"])/2
    up = hl2 - mult*atr; dn = hl2 + mult*atr
    d = pd.Series(index=o.index, dtype=bool); trend = True; fu = fl = np.nan
    for i in range(len(o)):
        c = o["close"].iloc[i]
        if np.isnan(atr.iloc[i]): d.iloc[i] = True; continue
        fu = up.iloc[i] if (np.isnan(fu) or up.iloc[i] > fu or o["close"].iloc[i-1] < fu) else fu
        fl = dn.iloc[i] if (np.isnan(fl) or dn.iloc[i] < fl or o["close"].iloc[i-1] > fl) else fl
        if trend and c < fu: trend = False
        elif (not trend) and c > fl: trend = True
        d.iloc[i] = trend
    return d

_cache = {}
def load(day):
    if day in _cache: return _cache[day]
    df = pd.read_sql_query("SELECT snapshot_time,tradingsymbol,strike,instrument_type,ltp,expiry_date,underlying_spot "
                           "FROM option_chain WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? AND ltp IS NOT NULL",
                           oc, params=[day])
    if df.empty: _cache[day] = None; return None
    df["t"] = pd.to_datetime(df["snapshot_time"])
    exps = sorted(df["expiry_date"].unique()); fut = [e for e in exps if e >= day]
    exp = fut[0] if fut else exps[-1]
    df = df[df["expiry_date"] == exp]
    dte = (pd.to_datetime(exp).date() - pd.to_datetime(day).date()).days
    spot = df.groupby("t")["underlying_spot"].first().sort_index()
    chain = {ts: (g.sort_values("t")["t"].values, g.sort_values("t")["ltp"].values, g["strike"].iloc[0], g["instrument_type"].iloc[0])
             for ts, g in df.groupby("tradingsymbol")}
    _cache[day] = (dte, spot, chain); return _cache[day]

def prem(chain, ts, t):
    if ts not in chain: return None
    ta, la, _, _ = chain[ts]; i = np.searchsorted(ta, np.datetime64(t), side="right") - 1
    if i < 0: return None
    v = la[i]; return float(v) if v and v > 0 else None

def tsym(chain, strike, typ):
    for ts, (_, _, st, ty) in chain.items():
        if int(st) == int(strike) and ty == typ: return ts
    return None

def sim(day, strike_off=0, sl_mult=1.3, exit_mode="ST"):
    d = load(day)
    if d is None: return None
    dte, spot_s, chain = d
    times = [t for t in spot_s.index if t.time() >= ENTRY and t.time() <= EOD]
    if not times: return None
    t0 = times[0]; spot = float(spot_s.loc[t0]); atm = round(spot/50)*50
    legs = []
    for typ, sgn in (("CE", 1), ("PE", -1)):
        k = atm + sgn*strike_off*50
        ts = tsym(chain, k, typ); p = prem(chain, ts, t0) if ts else None
        if p: legs.append({"ts": ts, "typ": typ, "entry": p, "sl": p*sl_mult, "open": True, "naked": False})
    if len(legs) < 2: return None
    stc = {}
    pnl = 0.0
    for t in times:
        if not any(l["open"] for l in legs): break
        force = t.time() >= TIMEEXIT
        for lg in legs:
            if not lg["open"]: continue
            p = prem(chain, lg["ts"], t)
            if p is None: continue
            if force or (exit_mode == "eod" and t.time() >= EOD):
                pnl += (lg["entry"]-p)*QTY - BROK; lg["open"] = False; continue
            if exit_mode == "time" and force:
                pnl += (lg["entry"]-p)*QTY - BROK; lg["open"] = False; continue
            if lg["naked"]:
                if lg["ts"] not in stc:
                    ta, la, _, _ = chain[lg["ts"]]; stc[lg["ts"]] = st_up(pd.Series(la, index=pd.DatetimeIndex(ta)).sort_index())
                s = stc[lg["ts"]]; sub = s[s.index <= t]
                if len(sub) and bool(sub.iloc[-1]):
                    pnl += (lg["entry"]-p)*QTY - BROK; lg["open"] = False
                continue
            if p >= lg["sl"]:
                pnl += (lg["entry"]-p)*QTY - BROK; lg["open"] = False
                if exit_mode == "ST":
                    for o2 in legs:
                        if o2["open"]: o2["naked"] = True
    # close any residual at last time
    lastt = times[-1]
    for lg in legs:
        if lg["open"]:
            p = prem(chain, lg["ts"], lastt) or lg["entry"]
            pnl += (lg["entry"]-p)*QTY - BROK
    return dte, pnl

def run_axis(label, variants, key):
    """variants: list of (name, kwargs). Aggregate P&L over ALL days per variant."""
    out = []
    for name, kw in variants:
        tot = 0.0; n = 0; bydte = {}
        for day in DAYS:
            r = sim(day, **kw)
            if r is None: continue
            dte, p = r; tot += p; n += 1; bydte[dte] = bydte.get(dte, 0)+p
        out.append((name, tot, n, bydte))
    return label, key, out

print("running sensitivity axes...")
# Base: strike_off=0 (ATM), sl_mult=1.3, exit=ST
axes = []
# DTE axis: base params, but report P&L split by DTE (one sim pass)
tot_by_dte = {}
for day in DAYS:
    r = sim(day, 0, 1.3, "ST")
    if r: tot_by_dte[r[0]] = tot_by_dte.get(r[0], 0) + r[1]
axes.append(("DTE-at-entry (base ATM/1.3/ST)", "dte", [(f"{k}DTE", v, 0, {}) for k, v in sorted(tot_by_dte.items())]))
axes.append(run_axis("Strike offset (OTM strikes from ATM)", [("ATM", dict(strike_off=0)), ("1-OTM", dict(strike_off=1)), ("2-OTM", dict(strike_off=2)), ("3-OTM", dict(strike_off=3))], "strike"))
axes.append(run_axis("SL multiple (x entry premium)", [("1.2x", dict(sl_mult=1.2)), ("1.3x", dict(sl_mult=1.3)), ("1.5x", dict(sl_mult=1.5)), ("2.0x", dict(sl_mult=2.0)), ("none", dict(sl_mult=99))], "sl"))
axes.append(run_axis("Exit mode", [("time-1445", dict(exit_mode="time")), ("EOD-1515", dict(exit_mode="eod")), ("SL->ST(7,2)", dict(exit_mode="ST"))], "exit"))

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
for a, (label, key, out) in zip(ax.flat, axes):
    names = [o[0] for o in out]; vals = [o[1] for o in out]
    a.bar(names, vals, color=["#0a6" if v >= 0 else "#d33" for v in vals])
    a.axhline(0, color="#333", lw=.8); a.set_title(label, fontsize=10)
    a.tick_params(axis="x", rotation=30)
    for i, v in enumerate(vals): a.text(i, v, f"{v/1000:.1f}k", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
fig.suptitle("Part (a) — one-factor sensitivity, ATM short straddle on recorded chain (lots=2, 28d SIGNAL only)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "scan_sensitivity.png", dpi=110, bbox_inches="tight")
print("WROTE", OUT / "scan_sensitivity.png")

L = ["# Part (a) — one-factor sensitivity (short straddle on recorded NIFTY chain)\n",
     "Base: enter 09:20, ATM CE+PE, lots=2, per-leg SL 1.3x, SL->naked ST(7,2), time-exit 14:45. "
     "Vary ONE axis at a time. **28 days => SIGNAL, read the gradient (monotonic > peak), not the peak.**\n"]
for label, key, out in axes:
    L.append(f"## {label}")
    L.append("| variant | net ₹ |"); L.append("|---|---|")
    for name, tot, n, bd in out: L.append(f"| {name} | {tot:,.0f} |")
    L.append("")
L.append("- DTE axis confirms where the edge concentrates. Monotonic responses (e.g. P&L rising as DTE->1) "
         "are more trustworthy than isolated peaks. Multiple-testing: few cells/axis, each is one 28-day path.")
(OUT / "RESULTS_scan.md").write_text("\n".join(L), encoding="utf-8")
print("WROTE", OUT / "RESULTS_scan.md")
oc.close()
