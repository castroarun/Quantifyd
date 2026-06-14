from pathlib import Path
p = Path("/home/arun/quantifyd/research/64_factor_index_rotation/results/RESULTS.md")
t = p.read_text()
note = """# CONSISTENCY FIX (2026-06-14) - whole page re-based to one canonical window

Earlier drafts mixed a 2016-26 window (which dropped a flat 2015 warm-up year, inflating
Momentum+Gold+Nasdaq to 22.1% / Calmar 1.77) with a full 2015-26 window elsewhere - the
same book showed 19-22% / 1.5-1.8 in different tables. Every figure is now computed by ONE
script (`canonical.py`) on the full 2015-26 window, inverse-vol, net 20bps:

- **WINNER = Value + Gold + Nasdaq: CAGR 17.4%, MaxDD -9.5%, Calmar 1.83** (best clean sleeve).
- Momentum + Gold + Nasdaq: 20.0% / -12.5% / 1.60 (higher return, deeper DD).
- Nifty + Gold + Nasdaq (research/63, equal): 17.6% / -11.3% / 1.57.
- Factor-only (Value+Momentum+Alpha, no assets): 13.8% / -24.9% / 0.55 (diversifying across
  factors fails - they are ~0.8 correlated).

Featured sleeve switched Momentum -> Value (Value has the best Calmar on the consistent
window). Factsheet regenerated for Value. The conclusions are unchanged: factors are mostly
the same Nifty bet; the value is a single equity-sleeve swap + Gold + Nasdaq; one factor not two.

---

"""
if "CONSISTENCY FIX (2026-06-14)" not in t:
    p.write_text(note + t); print("RESULTS.md updated with consistency fix")
else:
    print("already")
