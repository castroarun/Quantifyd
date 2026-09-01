"""Is V2's VIX floor a filter, or an off-switch? And what does the alternative look like?

The shadow-skip ledger answers a question I did not ask. I proposed checking
whether the live CPR/inside-week gate is the one the look-ahead audit rejected for
the next-week fly. But the ledger shows the CPR gate is NEVER REACHED: all 28
recorded skips read `vix<13.0`, and every compression-log row carries
vix_regime='below_floor' with would_enter=0.

So the binding constraint is CFG['vix_floor'] = 13.0, an ABSOLUTE India VIX level.

That matters because the 45-DTE book in this same repo deliberately does NOT use an
absolute level - it uses "India VIX percentile rank > 25 (vs previous 252 sessions)",
a RELATIVE measure. An absolute threshold does not travel across volatility regimes:
if the whole VIX distribution shifts down, an absolute floor silently becomes an
off-switch rather than a filter, and the book stops producing forward evidence
without anything appearing to be wrong.

This measures exactly that: how much of history each rule would admit, in the recent
regime versus the long one.

Read-only.
"""
import sqlite3, statistics as st, sys
from datetime import date

DB = "/home/arun/quantifyd/backtest_data/market_data.db"
c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

rows = []
for sym in ("INDIA VIX", "INDIAVIX", "INDIA_VIX"):
    rows = c.execute("SELECT date, close FROM market_data_unified "
                     "WHERE symbol=? AND timeframe='day' ORDER BY date", (sym,)).fetchall()
    if rows:
        print(f"VIX series: symbol={sym!r}, {len(rows)} days "
              f"{rows[0][0][:10]} -> {rows[-1][0][:10]}")
        break
if not rows:
    cands = [r[0] for r in c.execute(
        "SELECT DISTINCT symbol FROM market_data_unified WHERE symbol LIKE '%VIX%'")]
    print("no VIX series found; candidates:", cands)
    sys.exit(0)

d = [(r[0][:10], float(r[1])) for r in rows if r[1]]
vals = [v for _, v in d]
FLOOR = 13.0

print(f"\nVIX distribution, full history ({len(vals)} days)")
qs = [0, 5, 10, 25, 50, 75, 90, 100]
srt = sorted(vals)
for q in qs:
    print(f"  p{q:<3} {srt[min(len(srt)-1, int(q/100*len(srt)))]:>6.2f}")

def frac(sub, thr):
    return 100.0 * sum(1 for v in sub if v >= thr) / len(sub) if sub else 0.0

print(f"\nHOW OFTEN VIX >= {FLOOR} (the live absolute floor)")
for label, cut in (("all history", None), ("2024+", "2024-01-01"),
                   ("2025+", "2025-01-01"), ("2026 only", "2026-01-01")):
    sub = [v for dt, v in d if cut is None or dt >= cut]
    if sub:
        print(f"  {label:12} n={len(sub):>5}  passes {frac(sub, FLOOR):>5.1f}%  "
              f"median {st.median(sub):>5.2f}  max {max(sub):>6.2f}")

# percentile-rank rule, the one the 45-DTE book uses
print("\nPERCENTILE-RANK RULE (rank vs previous 252 sessions), as /app/straddle45 uses")
ranks = []
for i in range(252, len(d)):
    win = [v for _, v in d[i - 252:i]]
    r = 100.0 * sum(1 for v in win if v < d[i][1]) / len(win)
    ranks.append((d[i][0], d[i][1], r))
for thr in (25, 40, 50, 75):
    for label, cut in (("all", None), ("2026", "2026-01-01")):
        sub = [r for dt, v, r in ranks if cut is None or dt >= cut]
        if sub:
            p = 100.0 * sum(1 for r in sub if r > thr) / len(sub)
            print(f"  rank>{thr:<3} {label:5} n={len(sub):>5}  passes {p:>5.1f}%")

print("\n2026 SO FAR — what each rule admits")
sub26 = [(dt, v, r) for dt, v, r in ranks if dt >= "2026-01-01"]
if sub26:
    print(f"  days                       {len(sub26)}")
    print(f"  VIX >= 13.0 (live rule)    {sum(1 for _,v,_ in sub26 if v>=13.0)}")
    print(f"  VIX rank > 25              {sum(1 for _,_,r in sub26 if r>25)}")
    print(f"  VIX rank > 50              {sum(1 for _,_,r in sub26 if r>50)}")
    print(f"  VIX max in 2026            {max(v for _,v,_ in sub26):.2f}")

print("\nWHEN DID V2 LAST TRADE?")
v = sqlite3.connect("file:/home/arun/quantifyd/backtest_data/v2_ironfly_trading.db?mode=ro", uri=True)
for r in v.execute("SELECT system, MAX(day), COUNT(*) FROM v2_positions GROUP BY system"):
    print(f"  {r[0]:10} last entry {r[1]}  ({r[2]} positions ever)")
last = v.execute("SELECT MAX(day) FROM v2_positions").fetchone()[0]
if last:
    gap = (date.today() - date.fromisoformat(last)).days
    print(f"  -> {gap} calendar days since the last entry of any kind")
