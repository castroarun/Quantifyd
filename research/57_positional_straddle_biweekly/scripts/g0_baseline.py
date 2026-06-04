"""research/57 G0 — baseline positional bi-weekly short straddle + EOD overnight wings (real NIFTY chain).

Each trading day: SELL ATM straddle in the 2nd-NEAREST weekly expiry at 09:20, carry until that
expiry's DTE<=1, marking real premiums at 15:20 daily. Overnight wings: BUY ATM±W far-OTM at 15:20,
SELL at 09:20 next day (gap protection). Report per-trade net (straddle-only vs +wings), worst, theta.
30 days => SIGNAL; overlapping daily entries are correlated (flagged). Net Rs80/leg.
"""
import sqlite3
from pathlib import Path
from datetime import datetime, date
import numpy as np, pandas as pd

ROOT = Path("/home/arun/quantifyd"); OUT = ROOT / "research/57_positional_straddle_biweekly/results"
OUT.mkdir(parents=True, exist_ok=True)
OPT = ROOT / "backtest_data" / "options_data.db"
LOT = 65; BROK = 80; WING = 500; ROLL_DTE = 1
oc = sqlite3.connect(str(OPT))
oc.execute("CREATE INDEX IF NOT EXISTS idx_oc_lk ON option_chain(expiry_date,strike,instrument_type,snapshot_time)")
DAYS = [r[0] for r in oc.execute("SELECT DISTINCT substr(snapshot_time,1,10) FROM option_chain WHERE symbol='NIFTY' ORDER BY 1")]

def expiries_on(day):
    return sorted({r[0] for r in oc.execute(
        "SELECT DISTINCT expiry_date FROM option_chain WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? AND expiry_date>=?",
        (day, day))})

def spot_at(day, hhmm):
    r = oc.execute("SELECT underlying_spot FROM option_chain WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? "
                   "AND substr(snapshot_time,12,5)<=? AND underlying_spot IS NOT NULL ORDER BY snapshot_time DESC LIMIT 1",
                   (day, hhmm)).fetchone()
    return float(r[0]) if r and r[0] else None

def ltp(strike, otype, expiry, day, hhmm):
    r = oc.execute("SELECT ltp FROM option_chain WHERE symbol='NIFTY' AND strike=? AND instrument_type=? AND expiry_date=? "
                   "AND substr(snapshot_time,1,10)=? AND substr(snapshot_time,12,5)<=? AND ltp>0 ORDER BY snapshot_time DESC LIMIT 1",
                   (strike, otype, expiry, day, hhmm)).fetchone()
    return float(r[0]) if r and r[0] else None

def dte(expiry, day):
    return (datetime.strptime(expiry, "%Y-%m-%d").date() - datetime.strptime(day, "%Y-%m-%d").date()).days

def run_trade(d0):
    exps = expiries_on(d0)
    if len(exps) < 2: return None
    E = exps[1]                      # 2nd-nearest weekly = bi-weekly contract
    spot0 = spot_at(d0, "09:20")
    if not spot0: return None
    K = round(spot0 / 50) * 50
    ce0 = ltp(K, "CE", E, d0, "09:20"); pe0 = ltp(K, "PE", E, d0, "09:20")
    if not ce0 or not pe0: return None
    credit = ce0 + pe0
    # carry days: from d0 to the day E's dte hits ROLL_DTE
    carry = [d for d in DAYS if d >= d0 and dte(E, d) >= ROLL_DTE]
    if len(carry) < 2: return None
    d_exit = carry[-1]
    ce_x = ltp(K, "CE", E, d_exit, "15:20") or ce0; pe_x = ltp(K, "PE", E, d_exit, "15:20") or pe0
    straddle_gross = (credit - (ce_x + pe_x)) * LOT - 2 * BROK
    # daily straddle MTM (15:20) -> worst adverse + path
    mtm = []
    for d in carry:
        c = ltp(K, "CE", E, d, "15:20"); p = ltp(K, "PE", E, d, "15:20")
        if c and p: mtm.append((credit - (c + p)) * LOT)
    worst_mtm = min(mtm) if mtm else 0
    # EOD wings overlay: each carry night buy K+W CE & K-W PE at 15:20, sell 09:20 next day
    wing_pnl = 0.0; gap_saves = 0
    for i in range(len(carry) - 1):
        d, dn = carry[i], carry[i + 1]
        wc_b = ltp(K + WING, "CE", E, d, "15:20"); wp_b = ltp(K - WING, "PE", E, d, "15:20")
        wc_s = ltp(K + WING, "CE", E, dn, "09:20"); wp_s = ltp(K - WING, "PE", E, dn, "09:20")
        if wc_b and wc_s: wing_pnl += (wc_s - wc_b) * LOT - 2 * BROK
        if wp_b and wp_s: wing_pnl += (wp_s - wp_b) * LOT - 2 * BROK
        # overnight straddle gap (15:20 d -> 09:20 dn)
        c1 = ltp(K, "CE", E, d, "15:20"); p1 = ltp(K, "PE", E, d, "15:20")
        c2 = ltp(K, "CE", E, dn, "09:20"); p2 = ltp(K, "PE", E, dn, "09:20")
        if c1 and p1 and c2 and p2:
            ov = ((c1 + p1) - (c2 + p2))  # straddle gain overnight (decay +, gap -)
            if ov < -30: gap_saves += 1
    return dict(entry=d0, expiry=E, exit=d_exit, days=len(carry), K=K, credit=round(credit, 1),
                straddle_net=round(straddle_gross), worst_mtm=round(worst_mtm),
                wing_pnl=round(wing_pnl), with_wings=round(straddle_gross + wing_pnl), gap_nights=gap_saves)

print("running G0 over %d entry days..." % len(DAYS), flush=True)
rows = []
for d0 in DAYS:
    r = run_trade(d0)
    if r: rows.append(r)
    if r: print("  %s E=%s %dd credit=%.0f straddle=%+d wings=%+d net=%+d worst=%d" % (
        r["entry"], r["expiry"][5:], r["days"], r["credit"], r["straddle_net"], r["wing_pnl"], r["with_wings"], r["worst_mtm"]), flush=True)
T = pd.DataFrame(rows)
T.to_csv(OUT / "g0_trades.csv", index=False)

def stat(col):
    a = T[col].values
    return "n=%d  total=%+d  mean=%+d  median=%+d  win%%=%d  worst=%d" % (
        len(a), a.sum(), round(a.mean()), round(np.median(a)), round(100 * (a > 0).mean()), a.min())

L = ["# research/57 G0 — baseline bi-weekly short straddle + EOD wings (real NIFTY chain, %d trades)\n" % len(T),
     "Short ATM straddle in 2nd-nearest weekly, carry to DTE<=%d, EOD ±%dpt wings overnight. 09:20 entry, 15:20 marks. Net Rs80/leg. **30d SIGNAL, overlapping daily entries (correlated).**\n" % (ROLL_DTE, WING),
     "## Per-trade P&L", "- **straddle-only:** " + stat("straddle_net"), "- **straddle + EOD wings:** " + stat("with_wings"),
     "- wings total P&L: %+d (cost of overnight protection across all trades)" % T["wing_pnl"].sum(),
     "- avg days held: %.1f  |  worst single-trade MTM: %d  |  trades with a gap-down night: %d" % (
         T["days"].mean(), T["worst_mtm"].min(), (T["gap_nights"] > 0).sum()),
     "\n## Read", "- Does the naked short bi-weekly straddle decay net-positive (theta) before management?",
     "- Do the EOD wings cost more than the gap protection they buy (compare straddle-only vs +wings worst & total)?",
     "- 30d = SIGNAL; daily entries overlap (same expiry cycles) -> treat as directional, not validated."]
(OUT / "RESULTS_g0.md").write_text("\n".join(L), encoding="utf-8")
print("\nG0 DONE", flush=True)
print("straddle-only:", stat("straddle_net"), flush=True)
print("with-wings   :", stat("with_wings"), flush=True)
print("wings total  :", T["wing_pnl"].sum(), flush=True)
oc.close()
