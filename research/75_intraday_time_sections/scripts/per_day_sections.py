"""research/75 — per-day section breakdown + shape assessment: is the intraday sectioning pattern
consistent day-to-day, or do different days have a different lull? Reads section_metrics.csv."""
import pandas as pd, numpy as np
from pathlib import Path

OUT = Path("/home/arun/quantifyd/research/75_intraday_time_sections/results")
R = pd.read_csv(OUT / "section_metrics.csv")
order = ["OPEN", "MORN", "LULL", "AFT", "CLOSE"]
rng = R.pivot_table(index=["day", "dte"], columns="section", values="nifty_range_per_hr").reindex(columns=order)
dec = R.pivot_table(index=["day", "dte"], columns="section", values="prem_decay_per_hr").reindex(columns=order)

print("=== PER-DAY  NIFTY range/hr by section  (calmest = lowest, marked *) ===")
print("day         dte " + "".join("%7s" % s for s in order) + "   calmest")
calm = {}
for (day, dte), row in rng.iterrows():
    v = row.values
    if np.isnan(v).all():
        continue
    c = order[int(np.nanargmin(v))]
    calm[c] = calm.get(c, 0) + 1
    cells = "".join((("%6.0f*" % x) if s == c else ("%7.0f" % x)) if not np.isnan(x) else "     --" for s, x in zip(order, v))
    print("%s  %d %s   %s" % (day, dte, cells, c))

tot = sum(calm.values())
print("\n=== WHICH section is calmest (least movement) — distribution across %d days ===" % tot)
for s in order:
    print("  %-6s calmest on %2d days (%2.0f%%)" % (s, calm.get(s, 0), 100 * calm.get(s, 0) / tot))

# slowest-decay section per day (decay is negative; slowest = closest to 0 = max)
calmd = {}
for (day, dte), row in dec.iterrows():
    v = row.values
    if np.isnan(v).all():
        continue
    s = order[int(np.nanargmax(v))]
    calmd[s] = calmd.get(s, 0) + 1
print("\n=== WHICH section has the SLOWEST premium decay — distribution ===")
for s in order:
    print("  %-6s slowest-decay on %2d days (%2.0f%%)" % (s, calmd.get(s, 0), 100 * calmd.get(s, 0) / max(sum(calmd.values()), 1)))

print("\n=== CONSISTENCY: each section's range/hr across days (mean +/- std, min-max) ===")
for s in order:
    v = R[R.section == s]["nifty_range_per_hr"].dropna()
    print("  %-6s mean %3.0f  std %3.0f  range [%3.0f .. %3.0f]" % (s, v.mean(), v.std(), v.min(), v.max()))

# classify each day's shape
def shape(row):
    v = row.reindex(order)
    if v.isna().all():
        return "n/a"
    calmest = order[int(np.nanargmin(v.values))]
    if calmest == "LULL":
        return "textbook (lull calmest)"
    if calmest in ("OPEN", "MORN"):
        return "morning-calm"
    return "afternoon-calm"
sh = rng.apply(shape, axis=1).value_counts()
print("\n=== DAY-SHAPE classification (by calmest section) ===")
for k, v in sh.items():
    print("  %-26s %d days" % (k, v))
