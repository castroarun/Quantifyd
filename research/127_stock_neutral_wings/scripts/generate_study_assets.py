#!/usr/bin/env python3
"""research/127 — build the /app/backtest study assets: tearsheet PNG + backtests.ts entry
(with per-year, per-symbol, margin model, robustness and the FULL trade log)."""
import json, math, sqlite3, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent          # /home/arun/quantifyd
sys.path.insert(0, str(ROOT / "research" / "89_short_monthly_straddle" / "scripts"))
sys.path.insert(0, str(ROOT / "research" / "_utilities"))
import engine as E

RESULTS = HERE.parent / "results"
COST = 0.005
SLUG = "stock-45dte-neutral-wings"

tr = pd.read_csv(RESULTS / "phase_b2_trades.csv")
c1 = tr[(tr.config == "C1_E45X21W7K25_noSL") & (tr.atm_vol >= 100) & (tr.wing_vol_min >= 10)].copy()
c1["net_pct"] = c1["gross_pct"] - COST * c1["turnover_pct"]
c1["entry_date"] = pd.to_datetime(c1["entry_date"])
c1 = c1.sort_values("entry_date").reset_index(drop=True)

eq = pd.read_csv(RESULTS / "phase_e_equity.csv", parse_dates=["date"])

# ---------------- tearsheet ----------------
from tearsheet import generate_tearsheet
strat = pd.Series(eq["equity"].values, index=eq["date"])
conn = sqlite3.connect(E.db_path())
nifty = E.load_daily("NIFTY50", conn)["close"]
bench = nifty.reindex(strat.index, method="ffill")
meta = {"bench": "NIFTY 50",
        "footer": "Backtest 2016-2026, real NSE bhavcopy EOD stock options. Net of 0.5%-of-premium costs. "
                  "Margin MODELED at 1.25x max-loss + 2% (~6.7% notional) — real SPAN unverified; at 2x margin CAGR ~20%. Idle cash 5%."}
generate_tearsheet(strat, bench, "Stock 45-21 DTE Winged Strangle (10 slots, modeled margin)",
                   meta=meta, out_dir=str(RESULTS / "tearsheet"))
import glob, shutil
pngs = glob.glob(str(RESULTS / "tearsheet" / "*.png"))
dest = ROOT / "frontend" / "public" / "stock45_wings_tearsheet.png"
if pngs: shutil.copy(pngs[0], dest); print("tearsheet ->", dest)

# ---------------- tables ----------------
def yearly_rows():
    rows = []
    eqy = {y: (1 + d["ret"]).prod() - 1 for y, d in eq.groupby(eq.date.dt.year)}
    for y, d in c1.groupby("year"):
        net = d["net_pct"]
        t = net.mean() / (net.std(ddof=1) / math.sqrt(len(net))) if len(net) > 1 else float("nan")
        rows.append([str(y), str(len(d)), f"{net.mean()*100:+.3f}%", f"{t:+.2f}" if np.isfinite(t) else "-",
                     f"{100*(net>0).mean():.0f}%", f"{np.percentile(net,5)*100:+.2f}%",
                     f"{eqy.get(y, float('nan'))*100:+.1f}%" if y in eqy else "-"])
    return rows

def symbol_rows():
    rows = []
    g = c1.groupby("symbol").agg(n=("net_pct","size"), net=("net_pct","mean"),
                                  win=("net_pct", lambda x: (x>0).mean()), vol=("atm_vol","mean"))
    g = g[g.n >= 5].sort_values("net", ascending=False)
    for s, r in g.iterrows():
        rows.append([s, str(int(r.n)), f"{r.net*100:+.3f}%", f"{r.win*100:.0f}%", f"{r.vol:,.0f}"])
    return rows

def trade_rows():
    rows = []
    for _, t in c1.iterrows():
        rows.append([t["entry_date"].date().isoformat(), t["exit_date"], t["symbol"], t["expiry"],
                     f"{t['S0']:,.0f}", f"{t['Ks_pe']:,.0f}/{t['Ks_ce']:,.0f}",
                     f"{t['Kp']:,.0f}/{t['Kc']:,.0f}", f"{t['credit_pct']*100:.2f}%",
                     t["exit_reason"], f"{t['gross_pct']*100:+.2f}%", f"{t['net_pct']*100:+.2f}%"])
    return rows

net = c1["net_pct"]; tstat = net.mean() / (net.std(ddof=1) / math.sqrt(len(net)))

UNIVERSE = ", ".join(sorted(tr["symbol"].unique()))

entry = {
  "slug": SLUG,
  "title": "Stock 45→21 DTE Winged Strangle — one universal ruleset across the F&O stock universe",
  "verdict": ("Can the NIFTY 45-DTE window (research/119) be transplanted to single stocks? YES — as a DEFINED-RISK winged strangle with a hard liquidity gate. "
    "One ruleset, zero per-stock tuning: at 45 DTE sell the ±2.5% monthly strangle, buy wings 7% of spot away, NO stop, 50% profit target, time-exit at 21 DTE; "
    "trade only when all four legs actually traded (ATM vol ≥100, wings ≥10). On 628 liquid trades / ~70 stocks / 2016→Aug-2026 (real NSE bhavcopy EOD): "
    "net +0.264% of spot per trade at 0.5%-of-premium costs, t=+5.06, win 64.8%. ROBUSTNESS (G3) PASSED: survives dropping the top-5 names (+0.199, t=3.49); "
    "positive in every era (2016-23 +0.213 t=2.48, 2024-26 +0.290 t=4.44, 2021-24 ex-hot-years +0.168 t=2.46); edge RISES monotonically with liquidity "
    "(vol≥50 +0.108 → ≥500 +0.435 — the opposite of a stale-quote artifact); parameter plateau not peak; and the DTE-WINDOW PLACEBO is decisive — the identical "
    "structure entered at 35 DTE earns +0.02 (t=0.9) and at 55 DTE +0.06 (t=0.5): the 45→21 theta window IS the edge. Next-session entry keeps t=3.53. "
    "REFUTED along the way: 30-DTE entry (net t=-9), every premium stop (no-SL wins; wings suffice), plain IV-rank gating (not monotone), and all price-action "
    "calm gates (ADX/BB-squeeze/CPR/trend-dist — marginal). VRP=IV/RV20 IS a clean monotone signal on the crude base config but adds nothing to the optimized "
    "composite. PORTFOLIO (10 slots, entries ranked by ATM volume): at MODELED margin of 1.25×max-loss+2% (~6.7% of notional) the dense era 2021-26 shows 38.5% "
    "CAGR / -21% MaxDD / Calmar 1.81 — but DO NOT trust that row: real SPAN+exposure for stock condors is unverified and likely higher. The stressed band is the "
    "honest claim: at 1.5× margin 26.3% CAGR / -14.1% / Calmar 1.86; at 2× margin 20.2% / -10.4% / Calmar 1.94. Monthly correlation to NIFTY is -0.09 and the "
    "book averaged +1.65%/month in the 11 months NIFTY fell >3% — true diversification for a NIFTY-heavy short-vol book. GATE TO GO LIVE: measure real basket "
    "margin (Kite margin API), then paper-book the top-liquidity tier."),
  "status": "COMPLETE",
  "date": "2026-08-25",
  "cardBlurb": ("The NIFTY 45→21-DTE theta window transfers to stocks — as an iron-condor-style winged strangle, one ruleset for the whole F&O universe. "
    "Net +0.264%/trade (t 5.06, 628 liquid trades, 2016-2026); DTE placebo proves the window; edge rises with liquidity. Portfolio 20-26% CAGR at stressed "
    "margin, corr to NIFTY -0.09. STRATEGY-candidate; real-margin check pending."),
  "cardStats": [
    {"label": "Verdict", "value": "STRATEGY-CANDIDATE (margin check pending)"},
    {"label": "Net/trade (liquid)", "value": "+0.264% S0 · t 5.06 · win 65%"},
    {"label": "Portfolio CAGR (2x-1x margin stress)", "value": "20-38% · Calmar 1.8-1.9"}],
  "systemRules": {
    "intro": "One universal ruleset — no per-stock tuning. Stock selection is purely the liquidity filter.",
    "sharedCoreTitle": "The C1 ruleset (applies identically to every stock)",
    "sharedCore": [
      {"k": "Entry", "v": "45 calendar days before the monthly stock-option expiry, at EOD close (rolled back to the prior session if needed, tolerance +5d)"},
      {"k": "Structure", "v": "SELL CE at nearest strike to spot+2.5% and PE at spot-2.5%; BUY wing CE/PE ~7% of spot beyond each short strike (nearest traded strike)"},
      {"k": "Liquidity gate", "v": "All 4 legs traded that day (contracts>0); ATM legs >=100 contracts; each wing >=10. No entry otherwise — no exceptions"},
      {"k": "Exits", "v": "FIRST of: profit target 50% of net credit · time exit at 21 DTE. NO premium stop (tested: every stop hurts; the wings are the risk cap)"},
      {"k": "Costs modeled", "v": "0.5% of premium turnover (slippage+txn proxy; no bid/ask data exists for stock options EOD) — sensitivity 0.25%/1% shown"},
      {"k": "Sizing (portfolio)", "v": "10 slots, entries each monthly cycle ranked by ATM volume; margin per position modeled 1.25x max-loss + 2% of notional; idle cash at 5% (liquid ETF)"}],
    "riskLayer": {
      "title": "What was optimized vs held fixed",
      "columns": ["Axis", "Swept", "Chosen", "Evidence"],
      "rows": [
        ["Entry DTE", "30/40/45/50/60", "45", "30-DTE net-NEGATIVE t=-9; placebo 35/55 ≈ zero — the window is the edge"],
        ["Exit DTE", "10/15/21/28", "21", "15-21 plateau, best t at 21"],
        ["Short strikes", "ATM / ±2.5% / ±5%", "±2.5%", "best t (3.62); ATM close; ±5% thins credit"],
        ["Wing width", "3/5/6/7/8/10% of spot", "7%", "monotone wider-better; 10% nets more with fatter tail (p05 -2.9% vs -2.0%)"],
        ["Stop", "150/200/300%/none", "NONE", "no-SL beats all stops (t 3.33 vs 1.6-3.3); wings cap risk"],
        ["Target", "50% / none", "50%", "removing it costs 0.03%/trade"]],
      "highlightRows": [0, 4]}},
  "system": {
    "intro": "Adapting research/119 (NIFTY 45-DTE short straddle, STRATEGY-candidate) to single stocks: stocks carry idiosyncratic overnight/news gap risk that indices diversify away, so wings are mandatory (defined risk) and liquidity is the hard screen — research/89 proved most stock-option EOD 'edges' are phantom fills in untraded strikes.",
    "rows": [
      {"k": "Data", "v": "nse_options_bhav — real NSE F&O bhavcopy EOD, 24.2M stock-option rows, 81 underlyings, 2016 → Aug-2026, volume+OI per strike"},
      {"k": "Spot / indicators", "v": "market_data_unified daily (used only for ATM anchor + normalization; the system needs NO price-action indicator)"},
      {"k": "Universe (data)", "v": "81 F&O stocks in the bhav table; effective TRADED universe after the liquidity gate ~70 names, concentrated in the most liquid tier"},
      {"k": "IV layer", "v": "per-stock daily ATM-IV series BS-inverted from straddle closes (results/iv_daily.csv, reusable) — used to TEST IV-rank/VRP gates, not in the final ruleset"},
      {"k": "Study lineage", "v": "research/119 (NIFTY 45-DTE) mechanism; research/89 liquidity discipline; research/127 = this study"}]},
  "conditions": {
    "intro": "Full 81-name data universe (today's F&O list — survivorship stated):",
    "rows": [
      {"k": "Universe", "v": UNIVERSE},
      {"k": "Period", "v": "2016-01 → 2026-08 (87 monthly cycles; liquid sample densifies from 2021 — pre-2021 only 1-2 tradeable names/cycle)"},
      {"k": "Configs tried", "v": "~31 across all phases (recorded for multiple-testing honesty); guards keep t>3 throughout"}]},
  "comparisons": [
    {"title": "Robustness gauntlet (G3) — net @0.5% cost, liquid sample",
     "columns": ["Attack", "Result", "t", "Verdict"],
     "rows": [
       ["C1 reference (n=628)", "+0.264% S0/trade", "+5.06", "—"],
       ["Drop top-3 names (ADANIPORTS/TATAMOTORS/TCS)", "+0.228%", "+4.12", "PASS"],
       ["Drop top-5 names", "+0.199%", "+3.49", "PASS"],
       ["2016-2023 only", "+0.213%", "+2.48", "PASS"],
       ["2024-2026 only", "+0.290%", "+4.44", "PASS"],
       ["2021-2024 (ex the strong 25/26)", "+0.168%", "+2.46", "PASS"],
       ["Liquidity vol>=50 / >=100 / >=200 / >=500", "+0.108 / +0.264 / +0.351 / +0.435", "3.1-5.1", "monotone UP — STRONG PASS"],
       ["Same structure at 35 DTE (placebo)", "+0.020%", "+0.93", "window is the edge"],
       ["Same structure at 55 DTE (placebo)", "+0.059%", "+0.54", "window is the edge"],
       ["Enter NEXT session (lag test)", "+0.158%", "+3.53", "PASS (no close-timing artifact)"]],
     "highlightRows": [6, 7, 8]},
    {"title": "Filters tested and their fate (entry gates on the liquid sample)",
     "columns": ["Filter", "With gate", "Without / opposite", "In ruleset?"],
     "rows": [
       ["VRP = IV/RV20 > 1.1 (on crude base)", "+0.395% t=4.1", "+0.13-0.17%", "NO — adds nothing to optimized composite"],
       ["IV rank > 0.5 (own 252d)", "not monotone (mid-rank best)", "-", "NO — refuted"],
       ["Realized-vol rank calm <0.33", "+0.190%", "+0.115% hot", "NO — marginal"],
       ["ADX<25 / BB-squeeze / CPR narrow / trend-dist / RSI-mid / NR7", "±0.02-0.08% differences", "-", "NO — the edge is structural, not timing"],
       ["Liquidity (all legs traded + vol thresholds)", "the whole edge", "phantom fills (r/89)", "YES — the only gate"]],
     "highlightRows": [4]},
    {"title": "Margin & sizing model (the honest weak point)",
     "columns": ["Assumption", "Margin %notional", "CAGR 21-26", "MaxDD", "Calmar", "Sharpe"],
     "rows": [
       ["Modeled: 1.25x max-loss + 2%", "~6.7%", "38.5%", "-21.2%", "1.81", "1.00"],
       ["x1.5 stress", "~10%", "26.3%", "-14.1%", "1.86", "0.93"],
       ["x2.0 stress (conservative claim)", "~13.4%", "20.2%", "-10.4%", "1.94", "0.87"]],
     "highlightRows": [2],
     "caption": "Avg max-loss per condor ~3.7% of notional (7% wing dist - ~3.3% credit). Real SPAN+exposure for stock condors is UNVERIFIED — the x1.5-x2 band is the claim until the Kite basket-margin check runs. Implied notional/slot at modeled margin is ~15x slot capital — capacity requires the top-liquidity tier."}],
  "results": {
    "metrics": [
      {"label": "Net / trade (liquid)", "value": "+0.264% S0", "hint": "628 trades, 0.5% cost", "tone": "pos"},
      {"label": "t-stat", "value": "5.06", "hint": "3.49 after dropping top-5 names", "tone": "pos"},
      {"label": "Win rate", "value": "64.8%", "hint": "89% of trades reach an orderly exit (target/time)"},
      {"label": "CAGR (2x margin)", "value": "20.2%", "hint": "38.5% at modeled margin — unverified", "tone": "pos"},
      {"label": "MaxDD (2x margin)", "value": "-10.4%", "hint": "-21.2% at modeled margin", "tone": "neg"},
      {"label": "Corr vs NIFTY", "value": "-0.09", "hint": "+1.65%/mo avg in NIFTY down>3% months", "tone": "pos"}],
    "tables": [
      {"title": "Year by year — trades (net @0.5%) and portfolio return (modeled margin)",
       "columns": ["Year", "Trades", "Net/trade", "t", "Win", "p05", "Portfolio yr"],
       "rows": yearly_rows(), "heatmap": True},
      {"title": "Per-symbol (n>=5, liquid sample) — the effective traded universe",
       "columns": ["Symbol", "Trades", "Net/trade", "Win", "Avg ATM vol"],
       "rows": symbol_rows(), "heatmap": True},
      {"title": "Full trade log — all 628 liquid C1 trades (net @0.5% cost)",
       "columns": ["Entry", "Exit", "Symbol", "Expiry", "Spot", "Shorts PE/CE", "Wings PE/CE", "Credit", "Exit via", "Gross", "Net"],
       "rows": trade_rows()}],
    "charts": [{"src": "/app/stock45_wings_tearsheet.png",
                "caption": "Client tearsheet — 10-slot portfolio at MODELED margin (6.7% of notional) vs NIFTY 50, monthly, 2016-2026. At 2x margin the equity curve compresses to ~20% CAGR / -10% DD. Idle cash at 5% (liquid ETF)."}]},
  "winners": [
    {"config": "C1 — 45→21 DTE ±2.5% strangle, 7% wings, no SL, TP50, liquidity-gated",
     "summary": "The r/119 theta window transfers to stocks; wings turn idiosyncratic gap risk into a capped, priced cost; liquidity gate keeps it real.",
     "metrics": [
       {"k": "Net/trade", "v": "+0.264% S0 (t 5.06, n 628)"},
       {"k": "Portfolio (2x-1x margin)", "v": "20-38% CAGR, Calmar 1.8-1.9"},
       {"k": "Diversification", "v": "corr NIFTY -0.09; +EV in crash months"}],
     "rejected": ["30-DTE entry (t=-9)", "any premium stop", "IV-rank gate", "price-action calm gates", "5% OTM shorts"]}],
  "caveats": [
    "MARGIN IS MODELED, NOT MEASURED. 1.25x max-loss + 2% (~6.7% notional) may understate real SPAN+exposure for stock condors; the x2 row (20.2% CAGR / -10.4% DD) is the conservative claim. Gate to live: real Kite basket-margin check. CAGR scales ~inversely with margin.",
    "Costs are a 0.5%-of-premium proxy — stock options have NO bid/ask history. Break-even ~1.9% of turnover on the composite. Non-top-tier names can be worse; start any live test on the most liquid tier only.",
    "No earnings calendar in the data: earnings gaps inside the hold ARE in the marks (wings cap them) but 'skip earnings cycles' is untested — likely a free improvement once a source exists.",
    "Survivorship: today's F&O list applied to the past; mitigated by the modern sub-period being the STRONGEST era. Pre-2021 the liquid universe is 1-2 names/cycle (portfolio years 2016-20 are noise).",
    "C1 was selected from ~31 configs — the raw t=5.06 is inflated by selection; the robustness gauntlet (drop-top-5 t=3.49, era splits t~2.5) is the deflated evidence.",
    "Bhav closes are settle-ish marks; untraded wing marks valued at 0 on exit (pessimistic for us). Entry at same-day close; next-session lag keeps t=3.53.",
    "87 monthly cycles, one macro regime (no 2008-style event; Mar-2020 thinly sampled). Worst in-sample month -9.9% at modeled margin; a multi-stock gap event could exceed it — max loss if ALL 10 slots hit max-loss simultaneously is ~-55% of capital (wings cap it there)."],
  "githubLinks": [
    {"label": "research/127 — scripts + RESULTS.md", "href": "https://github.com/castroarun/Quantifyd/tree/main/research/127_stock_neutral_wings"},
    {"label": "research/119 — the NIFTY 45-DTE parent study", "href": "https://github.com/castroarun/Quantifyd/tree/main/research/119_45dte_short_straddle"}],
  "projectPaths": [
    "research/127_stock_neutral_wings/STOCK_NEUTRAL_WINGED_STRADDLE_DAILY_SWEEP_STATUS.md",
    "research/127_stock_neutral_wings/results/RESULTS.md",
    "research/127_stock_neutral_wings/results/iv_daily.csv (per-stock daily ATM-IV series, reusable)"]}

ts = "  " + json.dumps(entry, indent=2, ensure_ascii=False).replace("\n", "\n  ") + ",\n"
frag = RESULTS / "study_entry.ts.txt"
frag.write_text(ts, encoding="utf-8")
print(f"entry fragment -> {frag}  ({len(ts)/1024:.0f} KB, trades {len(c1)})")

# append into backtests.ts just before the closing `];` of BACKTEST_STUDIES
bt = ROOT / "frontend" / "src" / "data" / "backtests.ts"
src = bt.read_text(encoding="utf-8")
if f"'{SLUG}'" in src or f'"{SLUG}"' in src:
    print("slug already present — NOT appending again")
else:
    idx = src.rfind("\n];")
    assert idx > 0, "array close not found"
    src = src[:idx] + "\n" + ts + src[idx:]
    bt.write_text(src, encoding="utf-8")
    print("appended to", bt)
