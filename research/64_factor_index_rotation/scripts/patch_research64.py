"""Correct the published research/64 study after the Quality/LowVol data-corruption
finding. Targeted replacements in backtests.ts. Run on VPS. Idempotent-ish."""
from pathlib import Path
P = Path("/home/arun/quantifyd/frontend/src/data/backtests.ts")
ts = P.read_text()

if "DATA-INTEGRITY NOTE" in ts:
    print("already patched"); raise SystemExit

# --- A) verdict ---
old_verdict = "'Follow-on to the GTAA ETF study: apply the same switching / equal-weight / risk-parity toolkit to the Nifty single-factor indices (Momentum, Quality, Value, Low-Vol, Alpha). It does NOT transfer — the factors are mostly the same bet: mean cross-correlation 0.65 and 0.79–0.91 vs the Nifty itself, so equal-weighting them tops out at Calmar 0.76 (best pure-factor book = Momentum+Low-Vol, 17.4% CAGR but −22.9% DD). The real win is COMBINING: use the strongest single factor (Momentum) as the equity sleeve inside the GTAA asset trio and weight by inverse-vol → Momentum + Gold + Nasdaq (inverse-vol), monthly = Calmar 1.77, CAGR 22.1%, MaxDD −12.5%, cost-insensitive. That marginally beats the GTAA Nifty book (1.75) by upgrading the equity sleeve and taming Nasdaq’s 24% vol. Piling in ALL factors dilutes it to 1.18. STRATEGY candidate — an incremental upgrade to research/63, not a standalone factor edge; factor selection/diversification alone is a SIGNAL.'"
new_verdict = "'Follow-on to the GTAA ETF study: does the \"diversify, don’t select\" result transfer to the Nifty single factors (Momentum/Quality/Value/Low-Vol/Alpha)? No — on CLEAN data the factors are ~0.8 correlated to each other and to the Nifty (mostly the same equity bet), so diversifying across factors does not cut drawdown. DATA-INTEGRITY NOTE (2026-06-14): the Quality & Low-Vol Kite INDEX series were found corrupt (bad prints — 150% / 308% annualised daily vol) and are excluded/retracted, including the earlier \"Low-Vol is the lone diversifier\" claim. The real win is swapping the SINGLE equity sleeve of the GTAA trio from Nifty to one factor + Gold + Nasdaq, inverse-vol: the best CLEAN choice is the Value factor (Calmar 1.77, CAGR 16.8%, MaxDD −9.5%, 2015–26) — or Momentum (wins the 2016–26 window) — both beating the Nifty book (1.46–1.75). Use ONE factor, not two (a second adds correlated equity beta and crowds out Gold/Nasdaq). STRATEGY candidate — an incremental single-sleeve upgrade to research/63, not a standalone factor edge.'"
assert old_verdict in ts, "verdict anchor missing"
ts = ts.replace(old_verdict, new_verdict, 1)

# --- B) correlation table -> clean + add replace-Nifty table ---
old_corr = """        {
          title: 'Factor cross-correlation (monthly returns, 2010–26) — why diversification fails',
          columns: ['', 'Mom', 'Qual', 'Value', 'LowVol', 'Alpha'],
          rows: [
            ['Momentum', '1.00', '0.80', '0.76', '0.44', '0.90'],
            ['Quality', '0.80', '1.00', '0.85', '0.46', '0.76'],
            ['Value', '0.76', '0.85', '1.00', '0.42', '0.71'],
            ['LowVol', '0.44', '0.46', '0.42', '1.00', '0.42'],
            ['Alpha', '0.90', '0.76', '0.71', '0.42', '1.00'],
          ],
          highlightRows: [3],
          heatmap: true,
        },"""
new_corr = """        {
          title: 'Factor cross-correlation (monthly, CLEAN data) — factors are mostly the same Nifty bet',
          caption: 'CORRECTION: the Quality & Low-Vol Kite index series were CORRUPT (150% / 308% daily vol) and are excluded; clean factors below are ~0.8 correlated (even more than the 0.65 first reported with the bad data).',
          columns: ['', 'Mom', 'Value', 'Alpha', 'Nifty'],
          rows: [
            ['Momentum', '1.00', '0.77', '0.91', '0.84'],
            ['Value', '0.77', '1.00', '0.73', '0.89'],
            ['Alpha', '0.91', '0.73', '1.00', '0.80'],
            ['Nifty', '0.84', '0.89', '0.80', '1.00'],
          ],
          highlightRows: [],
          heatmap: true,
        },
        {
          title: 'Replace the Nifty sleeve with ONE factor (sleeve + Gold + Nasdaq, inverse-vol, 2015–26)',
          caption: 'User question. Clean factors only; Quality cleaned-indicative; Low-Vol index unusable (needs the ETF, 2022+). One factor beats two.',
          columns: ['Equity sleeve', 'CAGR', 'MaxDD', 'Calmar'],
          rows: [
            ['Value (clean) — best, lowest-DD', '16.8%', '−9.5%', '1.77'],
            ['Quality (cleaned-indicative)', '16.3%', '−9.4%', '1.74'],
            ['Momentum (clean)', '19.1%', '−12.5%', '1.53'],
            ['Alpha (clean)', '19.9%', '−13.7%', '1.46'],
            ['Nifty (baseline)', '16.4%', '−11.3%', '1.46'],
          ],
          highlightRows: [0],
          heatmap: false,
        },"""
assert old_corr in ts, "corr table anchor missing"
ts = ts.replace(old_corr, new_corr, 1)

# --- C) rejected[0] ---
old_rej = "'Pure factor diversification: equal-weight 5 factors = Calmar 0.76 — factors are 0.65 correlated, so equal-weighting them barely cuts the −23% drawdown. The research/63 \"diversify > select\" result does NOT transfer.',"
new_rej = "'Pure factor diversification: clean factors are ~0.8 correlated (mostly the same Nifty bet), so equal-weighting them does not cut drawdown — the research/63 \"diversify > select\" result does NOT transfer to factors.',"
assert old_rej in ts, "rejected anchor missing"
ts = ts.replace(old_rej, new_rej, 1)

# --- D) winner summary ---
old_sum = "summary: 'Best of 56 configs. The 1.7-Calmar tier requires the cross-asset diversifiers; given those, the Momentum factor is a better equity sleeve than plain Nifty and inverse-vol tames Nasdaq’s vol. Marginal over research/63 (1.77 vs 1.75) but structurally sound.',"
new_sum = "summary: 'The 1.7-Calmar tier requires the cross-asset diversifiers (Gold+Nasdaq); given those, ONE factor as the equity sleeve beats plain Nifty and inverse-vol tames Nasdaq’s vol. On a consistent clean 2015–26 window the VALUE factor sleeve is the lower-drawdown pick (Calmar 1.77, DD −9.5%); Momentum wins the 2016–26 window — both beat Nifty. Use one factor, not two.',"
assert old_sum in ts, "summary anchor missing"
ts = ts.replace(old_sum, new_sum, 1)

# --- E) data-integrity caveat (prepend) ---
cav_marker = "    caveats: [\n      'Period dependence (biggest): the combined book is 2016–26"
new_cav = ("    caveats: [\n      'DATA INTEGRITY CORRECTION (2026-06-14): the Kite INDEX series for the Quality and "
           "Low-Vol factors were found CORRUPT (bad prints — 150% and 308% annualised daily vol, single-day prints "
           "up to +472%). All Quality/Low-Vol index results are retracted, including the earlier "
           "\"Low-Vol is the lone diversifier (0.42–0.47 corr)\" claim, which was a bad-data artifact. Clean factors "
           "(Momentum/Value/Alpha/Nifty) are ~0.8 correlated. Clean Low-Vol/Quality require the factor ETFs (2022+).',\n"
           "      'Period dependence (biggest): the combined book is 2016–26")
assert cav_marker in ts, "caveat marker missing"
ts = ts.replace(cav_marker, new_cav, 1)

P.write_text(ts)
print("research/64 study corrected (verdict, corr table, replace-Nifty table, summary, caveat)")
