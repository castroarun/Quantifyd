"""research/58 V2 — POSITIONAL bi-weekly straddle backtest + per-trade daily MTM curves (for the app).
Sequential book: 09:20 ATM straddle in 2nd-nearest weekly; carry; exit on 1.5% move-stop OR PT-40%
OR DTE<=1 (roll); on exit RE-ENTER same day (stay short); +/-500pt overnight wings (buy 15:20 /
sell 09:20) tracked separately. 10 lots. Emits per-trade daily PnL series + book cumulative."""
import sqlite3, json
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT = Path("/home/arun/quantifyd")
OUT = ROOT / "research/58_intraday_recenter_straddle/results"; OUT.mkdir(parents=True, exist_ok=True)
OPT = ROOT / "backtest_data" / "options_data.db"
LOT = 65; LOTS = 10; QTY = LOT * LOTS; COST = 2 * 80
MOVE = 1.5; PT = 40; ROLL_DTE = 1; WING = 500

oc = sqlite3.connect(str(OPT))
oc.execute("CREATE INDEX IF NOT EXISTS idx_oc_lk ON option_chain(expiry_date,strike,instrument_type,snapshot_time)")
EXP = {}
for day, exp in oc.execute("SELECT DISTINCT substr(snapshot_time,1,10), expiry_date FROM option_chain WHERE symbol='NIFTY'"):
    if exp >= day: EXP.setdefault(day, set()).add(exp)
EXP = {d: sorted(s) for d, s in EXP.items()}
DAYS = sorted(EXP)
SPOTS = {}
for st, sp in oc.execute("SELECT snapshot_time, spot_price FROM underlying_spot WHERE symbol='NIFTY' AND spot_price>0 ORDER BY snapshot_time"):
    SPOTS.setdefault(st[:10], []).append((st[11:16], float(sp)))

def spot_at(day, hhmm):
    arr = SPOTS.get(day)
    if not arr: return None
    c = [s for (t, s) in arr if t <= hhmm]
    return c[-1] if c else (arr[0][1] if arr else None)
def ltp(strike, ot, E, day, hhmm):
    lo = day + "T00:00:00"; hi = day + "T" + hhmm + ":59"
    r = oc.execute("SELECT ltp FROM option_chain WHERE expiry_date=? AND strike=? AND instrument_type=? AND snapshot_time>=? AND snapshot_time<=? AND symbol='NIFTY' AND ltp>0 ORDER BY snapshot_time DESC LIMIT 1",
                   (E, strike, ot, lo, hi)).fetchone()
    return float(r[0]) if r and r[0] else None
def dte(E, day):
    return (datetime.strptime(E, "%Y-%m-%d").date() - datetime.strptime(day, "%Y-%m-%d").date()).days

trades = []          # each: {entry_day, exit_day, strike, expiry, exit_reason, pnl, wing_pnl, series:[[day,cum]]}
book_cum, book_curve = 0, []
i = 0
while i < len(DAYS):
    d0 = DAYS[i]
    exps = EXP.get(d0, [])
    if len(exps) < 2: i += 1; continue
    E = exps[1]                                   # 2nd-nearest weekly
    s0 = spot_at(d0, "09:20")
    if not s0: i += 1; continue
    K = round(s0 / 50) * 50
    ce0 = ltp(K, "CE", E, d0, "09:20"); pe0 = ltp(K, "PE", E, d0, "09:20")
    if not ce0 or not pe0: i += 1; continue
    credit = ce0 + pe0
    series, wing_pnl, exit_reason, exit_day = [], 0.0, "open", d0
    j = i
    while j < len(DAYS):
        d = DAYS[j]
        if dte(E, d) < ROLL_DTE: exit_reason = "roll_dte"; exit_day = d; break
        c = ltp(K, "CE", E, d, "15:20"); p = ltp(K, "PE", E, d, "15:20"); sp = spot_at(d, "15:20")
        if not (c and p and sp):
            j += 1; continue
        mtm = (credit - (c + p)) * QTY
        series.append([d, round(mtm + wing_pnl)])
        moved = abs(sp - s0) / s0 * 100 >= MOVE
        profit = mtm >= PT / 100.0 * credit * QTY
        if moved or profit:
            exit_reason = "move_stop" if moved else "profit_target"; exit_day = d; break
        # carry overnight -> wings: buy 15:20 today, sell 09:20 next day
        if j + 1 < len(DAYS):
            nd = DAYS[j + 1]
            wc = ltp(K + WING, "CE", E, d, "15:20"); wp = ltp(K - WING, "PE", E, d, "15:20")
            wc2 = ltp(K + WING, "CE", E, nd, "09:20"); wp2 = ltp(K - WING, "PE", E, nd, "09:20")
            if wc and wp and wc2 and wp2:
                wing_pnl += ((wc2 - wc) + (wp2 - wp)) * QTY - COST   # long wings: gain if they rise overnight
        j += 1
    else:
        exit_day = DAYS[-1]; exit_reason = "data_end"
    cF = ltp(K, "CE", E, exit_day, "15:20"); pF = ltp(K, "PE", E, exit_day, "15:20")
    final = (((credit - (cF + pF)) * QTY) if (cF and pF) else (series[-1][1] - wing_pnl if series else 0)) - COST + wing_pnl
    trades.append({"entry_day": d0, "exit_day": exit_day, "strike": K, "expiry": E,
                   "exit_reason": exit_reason, "pnl": round(final), "wing_pnl": round(wing_pnl),
                   "series": series})
    book_cum += round(final); book_curve.append([exit_day, book_cum])
    # immediate re-enter -> next sequence starts the day AFTER this exit
    i = DAYS.index(exit_day) + 1

json.dump({"version": "V2 positional bi-weekly", "move_stop": MOVE, "pt": PT, "wings": WING,
           "lots": LOTS, "lot": LOT, "trades": trades, "book_curve": book_curve}, open(OUT / "v2_positional.json", "w"))
a = np.array([t["pnl"] for t in trades])
wa = np.array([t["wing_pnl"] for t in trades])
print("=== V2 POSITIONAL (1.5%% + PT40 + wings + re-enter, 10 lots) ===")
print("trades=%d  total=%+d  mean/trade=%+d  median=%+d  win%%=%d  worst=%d  best=%d  wing_total=%+d" % (
    len(a), a.sum(), round(a.mean()), round(np.median(a)), round(100*(a>0).mean()), a.min(), a.max(), wa.sum()))
print("exit reasons:", {r: sum(1 for t in trades if t["exit_reason"] == r) for r in set(t["exit_reason"] for t in trades)})
oc.close()
