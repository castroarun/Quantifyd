"""research/57 G3a (FIXED) — INTRADAY move-stop frequency. Global elapsed-minutes throttle (was buggy
minute-of-day, which stopped checking after the entry day). Recipe move1.5+PT40, exit by 1 DTE."""
import sqlite3
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT = Path("/home/arun/quantifyd"); OUT = ROOT / "research/57_positional_straddle_biweekly/results"
OPT = ROOT / "backtest_data" / "options_data.db"
LOT = 65; BROK = 80; ROLL_DTE = 1; MOVE = 1.5; PT = 40
oc = sqlite3.connect(str(OPT))
oc.execute("CREATE INDEX IF NOT EXISTS idx_oc_lk ON option_chain(expiry_date,strike,instrument_type,snapshot_time)")
EXP = {}
for day, exp in oc.execute("SELECT DISTINCT substr(snapshot_time,1,10), expiry_date FROM option_chain WHERE symbol='NIFTY'"):
    if exp >= day: EXP.setdefault(day, set()).add(exp)
EXP = {d: sorted(s) for d, s in EXP.items()}; DAYS = sorted(EXP)
SPOTS = {}
for st, sp in oc.execute("SELECT snapshot_time, spot_price FROM underlying_spot WHERE symbol='NIFTY' AND spot_price>0"):
    SPOTS[st[:16]] = float(sp)
def dte(E, day):
    return (datetime.strptime(E, "%Y-%m-%d").date() - datetime.strptime(day, "%Y-%m-%d").date()).days
def ltp1(strike, ot, E, day, hhmm):
    r = oc.execute("SELECT ltp FROM option_chain WHERE expiry_date=? AND strike=? AND instrument_type=? AND snapshot_time>=? AND snapshot_time<=? AND symbol='NIFTY' AND ltp>0 ORDER BY snapshot_time DESC LIMIT 1",
                   (E, strike, ot, day + "T00:00:00", day + "T" + hhmm + ":59")).fetchone()
    return float(r[0]) if r and r[0] else None
def series(strike, ot, E, lo, hi):
    return {st[:16]: float(v) for st, v in oc.execute(
        "SELECT snapshot_time, ltp FROM option_chain WHERE expiry_date=? AND strike=? AND instrument_type=? AND symbol='NIFTY' AND snapshot_time>=? AND snapshot_time<=? AND ltp>0",
        (E, strike, ot, lo, hi))}
def gmin(t):  # global minutes from a fixed epoch
    return datetime.strptime(t[:16], "%Y-%m-%dT%H:%M").timestamp() / 60.0

print("loading trades + per-minute series...", flush=True)
TRADES = []
for d0 in DAYS:
    exps = EXP.get(d0, [])
    if len(exps) < 2: continue
    E = exps[1]; spot0 = SPOTS.get(d0 + "T09:20")
    if not spot0:
        cand = sorted(k for k in SPOTS if k[:10] == d0 and k[11:] >= "09:15")
        spot0 = SPOTS[cand[0]] if cand else None
    if not spot0: continue
    K = round(spot0 / 50) * 50
    ce0 = ltp1(K, "CE", E, d0, "09:20"); pe0 = ltp1(K, "PE", E, d0, "09:20")
    if not ce0 or not pe0: continue
    credit = ce0 + pe0
    carry = [d for d in DAYS if d >= d0 and dte(E, d) >= ROLL_DTE]
    if len(carry) < 2: continue
    ce_s = series(K, "CE", E, d0 + "T09:20:00", carry[-1] + "T15:30:00")
    pe_s = series(K, "PE", E, d0 + "T09:20:00", carry[-1] + "T15:30:00")
    tl = sorted(t for t in ce_s if t in pe_s and t in SPOTS and t >= d0 + "T09:20")
    if len(tl) < 5: continue
    TRADES.append(dict(spot0=spot0, credit=credit, tl=tl, ce=ce_s, pe=pe_s))
print("trades:", len(TRADES), flush=True)

def run(interval, eod_only=False):
    out = []
    for tr in TRADES:
        spot0, credit, tl = tr["spot0"], tr["credit"], tr["tl"]
        res = None; last = -1e18
        for t in tl:
            islast = (t == tl[-1])
            if eod_only:
                if not (t[11:16] >= "15:18" and t[11:16] <= "15:22") and not islast:
                    continue
            else:
                g = gmin(t)
                if g - last < interval and not islast:
                    continue
                last = g
            sp = SPOTS.get(t); ce = tr["ce"].get(t); pe = tr["pe"].get(t)
            if sp is None or ce is None or pe is None: continue
            mtm = (credit - (ce + pe)) * LOT
            if abs(sp - spot0) / spot0 * 100 >= MOVE or mtm >= PT / 100.0 * credit * LOT or islast:
                res = mtm; break
        if res is None:
            t = tr["tl"][-1]; res = (credit - (tr["ce"][t] + tr["pe"][t])) * LOT
        out.append(res - 2 * BROK)
    return np.array(out)

def stat(a): return "mean=%+d median=%+d win%%=%d worst=%d std=%d" % (round(a.mean()), round(np.median(a)), round(100*(a>0).mean()), a.min(), round(a.std()))
print("=== INTRADAY move-stop frequency (recipe move1.5+PT40) ===", flush=True)
L = ["# research/57 G3a (FIXED) — intraday move-stop frequency (recipe move1.5+PT40, %d trades)\n" % len(TRADES),
     "Stop+PT checked intraday at each tf (global-minute throttle). Exit = straddle premium at trigger minute. Net Rs80/leg. **30d SIGNAL.**\n",
     "| check tf | mean | median | win% | worst | std |", "|---|---|---|---|---|---|"]
for tf, name, eod in [(1, "1-min (continuous)", False), (5, "5-min", False), (10, "10-min", False), (15, "15-min", False), (0, "EOD 15:20 only", True)]:
    a = run(tf, eod_only=eod)
    L.append("| %s | %+d | %+d | %+d | %d | %d |" % (name, round(a.mean()), round(np.median(a)), round(100*(a>0).mean()), a.min(), round(a.std())))
    print("  %-20s %s" % (name, stat(a)), flush=True)
L += ["\n## Read", "- Finer tf fires nearer the 1.5% line (less overshoot) -> smaller worst; too fine may whipsaw.",
      "- vs EOD-only: shows how much the intraday stop improves the tail. 30d SIGNAL."]
(OUT / "RESULTS_g3a_intraday.md").write_text("\n".join(L), encoding="utf-8")
print("G3a DONE", flush=True)
oc.close()
