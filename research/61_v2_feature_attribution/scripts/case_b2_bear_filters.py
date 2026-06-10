"""CASE B (bear side) — can filters rescue DOWN-break continuation?
The inside-week DOWN-break had only 43% continuation (n=14 — too few to mine).
So test the user's filters on the FULL down-break sample (n~156, credible n):
  - daily RSI(14) <= 35 / <= 30 (oversold confirmation)
  - prior-day CPR in the MIDDLE (not wide, not narrow)
  - break on DAY-1 of the trade week (proxy for an early/first-candle confirmed break)
Report continuation% + median forward move per filter, on ALL down-breaks (credible) and on
the inside-week down subset (flagged: tiny). A filter must beat the 41% base AND have a
mechanism to matter. Read-only, daily bars.
"""
import json
import numpy as np, pandas as pd
from kiteconnect import KiteConnect
import config

tok = json.load(open("backtest_data/access_token.json"))
k = KiteConnect(api_key=config.KITE_API_KEY); k.set_access_token(tok["access_token"])
recs = []
for a, b in [("2017-01-01","2021-06-30"),("2021-07-01","2026-06-09")]:
    for c in k.historical_data(256265, a, b, "day"):
        recs.append((c["date"].strftime("%Y-%m-%d"), c["open"], c["high"], c["low"], c["close"]))
df = (pd.DataFrame(recs, columns=["date","o","h","l","c"]).drop_duplicates("date")
      .set_index("date").sort_index()); df.index = pd.to_datetime(df.index)
def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    return 100 - 100/(1 + up.ewm(alpha=1/n, adjust=False).mean()/dn.ewm(alpha=1/n, adjust=False).mean())
df["rsi"] = rsi(df["c"])
piv = (df.h.shift(1)+df.l.shift(1)+df.c.shift(1))/3
df["cprw"] = (2*piv-(df.h.shift(1)+df.l.shift(1))/2 - (df.h.shift(1)+df.l.shift(1))/2).abs()/df.o*100
df["wk"] = df.index.to_period("W-FRI")
wk = df.groupby("wk").agg(h=("h","max"), l=("l","min"))
wk["inside"] = (wk.h < wk.h.shift(1)) & (wk.l > wk.l.shift(1))
periods = list(wk.index)
dpos = {d: i for i, d in enumerate(df.index)}

# collect DOWN-breaks (first daily close < prior-week low) for every trade week
def down_breaks(trade_weeks):
    out = []
    for per in trade_weeks:
        i = periods.index(per)
        if i < 1: continue
        lvl = wk.l.iloc[i-1]; days = df[df.wk == per]
        if len(days) == 0: continue
        broke = None
        for n, (dt, row) in enumerate(days.iterrows()):
            if row.c > wk.h.iloc[i-1]: break              # up-break first -> not a down setup
            if row.c < lvl:
                broke = (dt, row.c, n == 0, row.rsi, row.cprw); break
        if broke:
            dt, bc, day1, r, cw = broke
            p = dpos[dt]; wend = days.iloc[-1].c
            seg = df.iloc[p:min(p+6, len(df))]
            cont_we = (bc - wend)/bc*100                  # +ve = continued DOWN to wk end
            cont_5 = (bc - df.iloc[min(p+5, len(df)-1)].c)/bc*100
            out.append(dict(cont_we=cont_we, cont_5=cont_5, day1=day1, rsi=r, cprw=cw))
    return pd.DataFrame(out)

allw = [periods[i] for i in range(1, len(periods))]
insidew = [periods[i] for i in range(1, len(periods)) if bool(wk.inside.iloc[i-1])]
A = down_breaks(allw); I = down_breaks(insidew)
# CPR mid = middle two quartiles of the full down-break sample
qlo, qhi = A.cprw.quantile(0.25), A.cprw.quantile(0.75)

def report(D, tag):
    if len(D) == 0: print(f"\n[{tag}] no down-breaks"); return
    def line(name, mask):
        s = D[mask]
        if len(s) == 0: return f"   {name:<22} n=0"
        return (f"   {name:<22} n={len(s):>3} | down-contin%={100*(s.cont_we>0).mean():4.0f}% "
                f"| median to wk-end={s.cont_we.median():+5.2f}% to +5TD={s.cont_5.median():+5.2f}%")
    print(f"\n=== [{tag}] DOWN-breaks (n={len(D)}) — base continuation {100*(D.cont_we>0).mean():.0f}% ===")
    print(line("ALL down-breaks", D.index == D.index))
    print(line("RSI<=35", D.rsi <= 35))
    print(line("RSI<=30", D.rsi <= 30))
    print(line("RSI>35 (contrast)", D.rsi > 35))
    print(line("CPR mid (Q2-Q3)", (D.cprw >= qlo) & (D.cprw <= qhi)))
    print(line("CPR wide (>Q3)", D.cprw > qhi))
    print(line("break on day-1", D.day1))
    print(line("break after day-1", ~D.day1))
    print(line("RSI<=35 & day-1", (D.rsi <= 35) & D.day1))
    print(line("RSI<=35 & CPR mid", (D.rsi <= 35) & (D.cprw >= qlo) & (D.cprw <= qhi)))

report(A, "ALL weeks (credible n)")
report(I, "INSIDE-week subset (TINY n — directional only)")
print("\nNote: NSE index has structural UP drift -> down-breaks fight the drift (why base is ~41%).")
print("A filter must beat base meaningfully on the CREDIBLE sample AND survive on inside-weeks to matter.")
print("\nDONE.")
