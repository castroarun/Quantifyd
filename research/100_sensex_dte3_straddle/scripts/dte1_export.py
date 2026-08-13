# -*- coding: utf-8 -*-
"""DTE-1 expiry-eve short straddle: NIFTY 2019->now (7yr) + SENSEX 2024->now. Real bhav OPEN->CLOSE
on the day before weekly expiry. Exports per-trade JSON for the tearsheet + prints 7yr/per-year summary."""
import sqlite3, datetime as dt, json, statistics
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=30000")
vix = {d: cl for d, cl in c.execute("SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day' AND close>0")}
COST, SLIP = 160, 0.003

def nifty_spot():
    return {d: cl for d, cl in c.execute("SELECT date, close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day' AND close>0")}
def sensex_spot():
    sp = defaultdict(list)
    for d, u in c.execute("SELECT trade_date, underlying FROM bse_options_bhav WHERE symbol='SENSEX' AND underlying>0"):
        sp[d].append(u)
    return {d: statistics.median(v) for d, v in sp.items()}

def leg(table, sym, d, E, K, ot):
    r = c.execute(f"SELECT open,close,open_interest{',lot' if table=='bse_options_bhav' else ''} FROM {table} WHERE symbol=? AND trade_date=? AND expiry_date=? AND strike=? AND option_type=?",
                  (sym, d, E, K, ot)).fetchone()
    if not r: return None
    o, cl, oi = r[0], r[1], r[2]
    lot = r[3] if table == 'bse_options_bhav' else 1
    return (o if o and o > 0 else cl, cl, oi or 0, lot or 1)

def run(table, sym, spot, STEP, QTY_fixed, FROM):
    exps = sorted({r[0] for r in c.execute(f"SELECT DISTINCT expiry_date FROM {table} WHERE symbol=? AND expiry_date>=?", (sym, FROM))})
    tdates = sorted({r[0] for r in c.execute(f"SELECT DISTINCT trade_date FROM {table} WHERE symbol=? AND trade_date>=?", (sym, FROM))})
    tset = set(tdates)
    def dte(E, d): return (dt.date.fromisoformat(E) - dt.date.fromisoformat(d)).days
    rows = []
    for E in exps:
        target = dt.date.fromisoformat(E) - dt.timedelta(days=1)
        d = None
        for k in range(4):
            cd = (target - dt.timedelta(days=k)).isoformat()
            if cd in tset: d = cd; break
        if not d or dte(E, d) < 1 or d not in spot: continue
        K = round(spot[d]/STEP)*STEP
        ce = leg(table, sym, d, E, K, "CE"); pe = leg(table, sym, d, E, K, "PE")
        if not ce or not pe or ce[2] < 100 or pe[2] < 100: continue
        hold = [x for x in tdates if d <= x <= E and dte(E, x) >= 1]; xd = hold[-1] if hold else d
        cf = leg(table, sym, xd, E, K, "CE"); pf = leg(table, sym, xd, E, K, "PE")
        if not cf or not pf: continue
        QTY = QTY_fixed if QTY_fixed else ce[3]*10
        pnl = round((ce[0]+pe[0]-(cf[1]+pf[1]))*QTY - COST - SLIP*(ce[0]+pe[0]+cf[1]+pf[1])*QTY)
        rows.append({"exit": xd, "pnl": pnl, "vix": round(vix.get(d, 0), 1)})
    rows.sort(key=lambda r: r["exit"])
    return rows

def summ(rows, lbl):
    p = [r["pnl"] for r in rows]; n = len(p)
    m = sum(p)/n; sd = (sum((x-m)**2 for x in p)/n)**0.5
    cum = pk = mdd = 0
    for x in p: cum += x; pk = max(pk, cum); mdd = min(mdd, cum-pk)
    print(f"{lbl}: n={n} total={round(sum(p)):+,} mean={round(m):+,} win={100*sum(1 for x in p if x>0)/n:.1f}% M/SD={m/sd:.2f} worst={min(p):+,} maxDD={round(mdd):+,}")
    yr = defaultdict(list)
    for r in rows: yr[r["exit"][:4]].append(r["pnl"])
    for y in sorted(yr):
        q = yr[y]; mm = sum(q)/len(q)
        print(f"   {y}: n={len(q):>2} total={round(sum(q)):>+10,} win={100*sum(1 for x in q if x>0)/len(q):>4.0f}% worst={min(q):>+9,}")

nif = run("nse_options_bhav", "NIFTY", nifty_spot(), 50, 650, "2019-01-01")
sen = run("bse_options_bhav", "SENSEX", sensex_spot(), 100, None, "2024-01-01")
summ(nif, "NIFTY DTE-1 7yr (QTY650)")
summ(sen, "SENSEX DTE-1 (QTY200)")
json.dump({"nifty": nif, "sensex": sen}, open("/tmp/dte1_curves.json", "w"))
print("exported", len(nif), "NIFTY +", len(sen), "SENSEX trades -> /tmp/dte1_curves.json")
c.close()
