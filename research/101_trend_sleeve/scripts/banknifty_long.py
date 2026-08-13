# -*- coding: utf-8 -*-
"""LONG BankNifty MONTHLY options (no shorts): buy ATM straddle out to 5% OTM strangles, ~30 DTE
entry, carried to expiry (intrinsic settlement). This is a long-vol book — the natural complement
to a short straddle. Real nse_options_bhav BANKNIFTY premiums; ATM from entry-day OPEN (no look-ahead)."""
import sqlite3, datetime as dt
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
SYM, STEP, SLIP, LOT = "BANKNIFTY", 100, 0.003, 15   # BNF lot 15; report per 1 lot
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=30000")
bnf = {d: (o, cl) for d, o, cl in c.execute("SELECT date,open,close FROM market_data_unified WHERE symbol='BANKNIFTY' AND timeframe='day' AND close>0")}
bdays = sorted(bnf)
allexp = sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol=? AND expiry_date>='2016-06-01'", (SYM,))})
# monthly = last expiry of each calendar month
bymon = defaultdict(list)
for e in allexp: bymon[e[:7]].append(e)
monthly = sorted(max(v) for v in bymon.values())
tdates = sorted({r[0] for r in c.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol=? AND trade_date>='2016-06-01'", (SYM,))})
tset = set(tdates)
def leg_open(d, E, K, ot):
    r = c.execute("SELECT open,close,open_interest FROM nse_options_bhav WHERE symbol=? AND trade_date=? AND expiry_date=? AND strike=? AND option_type=?", (SYM, d, E, K, ot)).fetchone()
    if not r: return None
    o, cl, oi = r
    return (o if o and o > 0 else cl, oi or 0)

def run(pct):
    trades = []
    for E in monthly:
        if E not in bnf: continue                       # need expiry-day spot for settlement
        d = None                                        # nearest trade day on/before E-28 (~30 DTE)
        for k in range(8):
            cd = (dt.date.fromisoformat(E) - dt.timedelta(days=28 + k)).isoformat()
            if cd in tset and cd in bnf: d = cd; break
        if not d: continue
        atm = round(bnf[d][0] / STEP) * STEP            # ATM from entry-day OPEN
        Kc = round(atm * (1 + pct / 100) / STEP) * STEP
        Kp = round(atm * (1 - pct / 100) / STEP) * STEP
        ce = leg_open(d, E, Kc, "CE"); pe = leg_open(d, E, Kp, "PE")
        if not ce or not pe or ce[1] < 50 or pe[1] < 50: continue
        cost = ce[0] + pe[0]
        sexp = bnf[E][1]                                # expiry-day close = settlement spot
        payoff = max(sexp - Kc, 0) + max(Kp - sexp, 0)  # intrinsic at expiry
        pnl_pts = payoff - cost - SLIP * cost           # entry slippage on the buy
        trades.append((E, round(pnl_pts * LOT), cost, payoff))
    return trades

def rep(tr, lbl):
    p = [x[1] for x in tr]; n = len(p); tot = sum(p); m = tot / n
    sd = (sum((x - m) ** 2 for x in p) / n) ** 0.5
    wins = [x for x in p if x > 0]
    best = max(p); worst = min(p)
    print(f"{lbl}: n={n} total={tot:+,} mean/mo={round(m):+,} win={round(100*len(wins)/n)}% M/SD={m/sd if sd else 0:+.2f} best={best:+,} worst={worst:+,}")
    # biggest winning months (the crash/big-move payoffs = the cushion)
    big = sorted(tr, key=lambda t: -t[1])[:4]
    print("   biggest gains: " + ", ".join(f"{t[0]} +{t[1]:,}" for t in big))

print("LONG BankNifty monthly options, 1 lot (qty 15), ~30 DTE entry -> carried to expiry, 2016-2026")
print("(long premium = defined max loss = the premium paid; profits on big moves)\n")
for pct in [0, 1, 2, 3, 4, 5]:
    lbl = "ATM straddle " if pct == 0 else f"{pct}% strangle  "
    rep(run(pct), lbl)
c.close()
