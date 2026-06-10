"""CASE B — G0 underlying probe: is there a tradable CONTINUATION edge after a
weekly inside-candle breakout? (precondition for any inside-week debit-spread strategy)

Full NIFTY weekly history (not just V2 cycles). For each week whose PRIOR week was an
inside week, in the trade week look for the FIRST daily close that breaks the inside
week's high (up) or low (down); then measure forward continuation from that breakout
close to (a) week-end and (b) +5 trading days. Compare vs the unconditional breakout
base-rate. Read-only, daily bars only — no options, no orders.
"""
import json
import numpy as np, pandas as pd
from kiteconnect import KiteConnect
import config

tok = json.load(open("backtest_data/access_token.json"))
k = KiteConnect(api_key=config.KITE_API_KEY); k.set_access_token(tok["access_token"])
recs = []
for a, b in [("2017-01-01", "2021-06-30"), ("2021-07-01", "2026-06-09")]:
    for c in k.historical_data(256265, a, b, "day"):
        recs.append((c["date"].strftime("%Y-%m-%d"), c["o" if False else "open"], c["high"], c["low"], c["close"]))
df = (pd.DataFrame(recs, columns=["date", "o", "h", "l", "c"])
      .drop_duplicates("date").set_index("date").sort_index())
df.index = pd.to_datetime(df.index)
df["wk"] = df.index.to_period("W-FRI")
wk = df.groupby("wk").agg(o=("o","first"), h=("h","max"), l=("l","min"), c=("c","last"))
wk["inside"] = (wk.h < wk.h.shift(1)) & (wk.l < 1e18) & (wk.l > wk.l.shift(1))
periods = list(wk.index)
dpos = {d: i for i, d in enumerate(df.index)}
dlist = list(df.index)

def study(trade_weeks, label):
    up = []; dn = []; nobreak = 0; n = 0
    for per in trade_weeks:
        i = periods.index(per)
        if i < 1: continue
        prev = wk.iloc[i-1]
        up_lvl, dn_lvl = prev.h, prev.l
        days = df[df.wk == per]
        if len(days) == 0: continue
        n += 1
        brk = None
        for dt, row in days.iterrows():
            if row.c > up_lvl: brk = ("up", dt, row.c); break
            if row.c < dn_lvl: brk = ("dn", dt, row.c); break
        if brk is None: nobreak += 1; continue
        side, bdt, bc = brk
        # forward to week-end
        wend = days.iloc[-1].c
        r_we = (wend - bc)/bc*100
        # forward to +5 TD beyond breakout day
        p = dpos[bdt]
        f5 = df.iloc[min(p+5, len(df)-1)].c
        r_5 = (f5 - bc)/bc*100
        # max favourable / adverse over next 5 TD in break direction
        seg = df.iloc[p:min(p+6, len(df))]
        if side == "up":
            mfe = (seg.h.max() - bc)/bc*100; mae = (seg.l.min() - bc)/bc*100
            up.append((r_we, r_5, mfe, mae))
        else:
            mfe = (bc - seg.l.min())/bc*100; mae = (bc - seg.h.max())/bc*100  # favourable = down
            dn.append((r_we, r_5, mfe, mae))
    def agg(rows, side):
        if not rows: return f"  {side}: n=0"
        a = np.array(rows)
        # continuation return is signed in break direction: up -> +ret good; dn -> -ret good
        cont_we = a[:,0] if side == "up" else -a[:,0]
        cont_5 = a[:,1] if side == "up" else -a[:,1]
        winwe = (cont_we > 0).mean()*100
        return (f"  {side}-break n={len(rows):>3} | contin% (to wk-end)={winwe:4.0f}% | "
                f"median cont to wk-end={np.median(cont_we):+5.2f}% to +5TD={np.median(cont_5):+5.2f}% | "
                f"median MFE={np.median(a[:,2]):4.2f}% MAE={np.median(a[:,3]):+5.2f}%")
    print(f"\n=== {label}: trade-weeks n={n}, no-break={nobreak} ({100*nobreak/max(n,1):.0f}%) ===")
    print(agg(up, "up")); print(agg(dn, "dn"))
    return up, dn

# trade weeks = weeks whose PRIOR week was inside
post_inside = [periods[i] for i in range(1, len(periods)) if bool(wk.inside.iloc[i-1])]
all_weeks = [periods[i] for i in range(1, len(periods))]
print(f"Total weeks: {len(periods)}, inside weeks: {int(wk.inside.sum())}, "
      f"post-inside trade weeks: {len(post_inside)}")
study(post_inside, "POST-INSIDE-WEEK breakout (the Case-B setup)")
study(all_weeks, "ALL weeks breakout (unconditional control)")

# crude debit-spread viability: of post-inside UP-breaks, how many reach +1.0% / +1.5%
i = [periods.index(p) for p in post_inside]
ups = []
for per in post_inside:
    ii = periods.index(per); prev = wk.iloc[ii-1]; days = df[df.wk == per]
    brk = None
    for dt, row in days.iterrows():
        if row.c > prev.h: brk = ("up", dt, row.c); break
        if row.c < prev.l: brk = ("dn", dt, row.c); break
    if brk and brk[0] == "up":
        p = dpos[brk[1]]; seg = df.iloc[p:min(p+6, len(df))]
        ups.append((seg.h.max()-brk[2])/brk[2]*100)
if ups:
    ups = np.array(ups)
    print(f"\nPost-inside UP-breaks: n={len(ups)} | reach +1.0% within 5TD: {(ups>=1.0).mean()*100:.0f}% | "
          f"+1.5%: {(ups>=1.5).mean()*100:.0f}% | +2.0%: {(ups>=2.0).mean()*100:.0f}%")
    print("(a call debit spread needs the up-move to clear the net debit ~ +1-1.5% to profit)")
print("\nDONE.")
