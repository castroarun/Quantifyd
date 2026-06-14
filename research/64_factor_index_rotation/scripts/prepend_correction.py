from pathlib import Path
p = Path("/home/arun/quantifyd/research/64_factor_index_rotation/results/RESULTS.md")
t = p.read_text()
note = """# CORRECTION (2026-06-14) - Quality & Low-Vol factor index data were CORRUPT

A post-hoc data-quality check (triggered by an implausible Low-Vol vol) found the Kite
INDEX series for **Quality (150% daily vol, +472% single-day print)** and **Low-Vol
(308% daily vol)** are corrupted by bad prints. Consequently:

- **RETRACTED:** the "Low-Vol is the lone diversifier (0.42-0.47 corr)" claim and the
  factor-basket numbers that used Quality/Low-Vol (e.g. "5factors equal 0.76",
  "Mom+LowVol 0.76"). Those were bad-data artifacts.
- **Clean factors (Momentum/Value/Alpha/Nifty) are ~0.8 correlated** - even MORE than the
  0.65 first reported; the bad data noise had deflated the correlation. The core
  conclusion (factors are mostly the same Nifty bet; diversifying across them fails)
  HOLDS and strengthens.
- **Replace-Nifty (user question), CLEAN, sleeve+Gold+Nasdaq inverse-vol, 2015-26:**
  Value **Calmar 1.77** (DD -9.5%) > Quality 1.74 (cleaned-indicative) > Momentum 1.53 >
  Alpha 1.46 ~ Nifty 1.46. Use ONE factor, not two. Value is the lowest-DD clean pick;
  Momentum wins the shorter 2016-26 window.
- **Headline G2 winner (Momentum+Gold+Nasdaq) uses CLEAN data and STANDS.**
- Clean Low-Vol/Quality need the factor ETFs (LOWVOL1, NIFTYQLITY etc., 2022+ only).

Scripts: clean_rerun.py, replace_nifty_test.py. Clean CSVs:
factor_monthly_closes_CLEAN.csv, factor_corr_CLEAN.csv.

---

"""
if "CORRECTION (2026-06-14)" not in t:
    p.write_text(note + t)
    print("RESULTS.md corrected")
else:
    print("already corrected")
