"""research/153 smoke test: build ctx, count signals per config family, run one book."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ipo_replay as ir

t0 = time.time()
ctx = ir.Ctx()
print(f"ctx built in {time.time()-t0:.0f}s", flush=True)
dates = ctx.dates
print("panel:", dates[0].date(), "->", dates[-1].date(), ctx.C.shape)

print("\n=== SIGNAL COUNTS (whole panel) ===")
print(f"{'age_m':>6}{'L':>5}{'depth':>7}{'rs_policy':>11}{'signals':>9}{'symbols':>9}")
for age in (3, 6, 12, 24):
    for L in (25, 40, 60):
        for dep in (0.25, 0.35, 0.50):
            for pol in ("off", "relaxed", "short", "strict"):
                trig, piv, lo = ir.build_trigger(ctx, max_age_m=age, min_bars=max(25, L),
                                                 L=L, max_depth=dep, rs_policy=pol)
                n = int(trig.sum()); ns = int((trig.sum(axis=0) > 0).sum())
                if L == 40 and dep == 0.35:
                    print(f"{age:6d}{L:5d}{dep:7.2f}{pol:>11}{n:9d}{ns:9d}")
                elif pol == "relaxed":
                    print(f"{age:6d}{L:5d}{dep:7.2f}{pol:>11}{n:9d}{ns:9d}")

print("\n=== ONE BOOK (age12, L40, depth .35, relaxed RS, 8 slots, 18.75%) ===")
trig, piv, lo = ir.build_trigger(ctx, max_age_m=12, min_bars=40, L=40,
                                 max_depth=0.35, rs_policy="relaxed")
sma = ctx.sma(50)
days = np.array([i for i, d in enumerate(dates) if "2006-01-01" <= str(d.date()) <= "2026-09-04"])
for seed in (1, 2, 3):
    t = time.time()
    eq, tr, pu, inv = ir.simulate_ipo(seed, days, dates, ctx.C, ctx.O, piv, lo, sma,
                                      ctx.RSF, ctx.TVp, trig, ctx.NOWEAK,
                                      cost=0.0025, stop=0.08, slots=8, size_pct=0.1875)
    st, e = ir.stats_from(eq, dates[days], tr, invested=inv)
    print(f"  seed {seed}: {st['x']:7.2f}x CAGR {st['cagr']:6.2f}% DD {st['dd']:7.2f}% "
          f"n {st['n']:4d} tpy {st['tpy']:5.1f} win {st['win']:4.1f}% mean {st['mean']:+6.2f}% "
          f"inv {st['invested_pct']:4.0f}% passed {pu} [{time.time()-t:.0f}s]", flush=True)
print("DONE")
