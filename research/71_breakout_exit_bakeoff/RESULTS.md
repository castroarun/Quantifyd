# RESULTS — MTF-Bullish Volume-Breakout Exit Bake-Off (research/71)

**VERDICT: STRATEGY (candidate).** An automatable version of the user's Chartink
MultiTimeFrame-bullish volume-breakout selection, exited with a **trailing stop (Donchian-20
or Supertrend 10,3), no profit target, a 20% catastrophe stop, and a NIFTY>200-DMA regime
gate, holding ~8 names**, produces **20.5% CAGR vs NIFTYBEES 11.6% (+9%/yr), Sharpe 0.71,
MaxDD −36% (index −60%), Calmar 0.57, 45.9× vs 9.4× over 20.5 years**, beating the index in
67% of years. The exit methodology question is answered decisively and robustly. It is a real,
tradeable momentum swing book — but a concentrated long-beta one with meaningful (~35%)
drawdowns, and the headline is optimistic due to survivorship bias (see caveats).

## The question, answered

**Which exit is best for these short-term breakout trades — fixed SL+target, trailing SL
(Donchian/Supertrend/EMA/chandelier), or fixed-days?**

1. **Trailing SL wins decisively.** On 20,804 tradeable breakouts, net per-trade @0.20%:
   ST(7,3) +4.43% · Donchian-20 +4.38% · EMA-50 +4.28% · ST(10,3) +4.27% · Chandelier-4×ATR
   +4.12% — all ≈ tied, PF ~1.8, avg hold ~6–7 weeks.
2. **Never use a profit target.** Every fixed-target config underperforms its no-target
   sibling (8%SL+15%tgt +0.9% vs 8%SL-no-tgt +4.9%). A target caps the fat right tail that
   *is* the edge.
3. **Hold weeks, not days.** 5-day holds / tight 2×ATR / EMA-10 ≈ 0. The breakout drift is
   slow; the money is in riding winners until the trail breaks. "Short-term" here = exit-driven
   swing (days-to-a-couple-months), NOT calendar-fixed and NOT buy-and-hold-forever.
4. **Fixed-days works but leaves money on the table.** Hold-40-days = +3.3%/trade, higher win
   rate (54%), positive median — the "smoothest feel" — but trailing earns more.
5. **Worst = tight fixed SL + quick target** (5%/10% ≈ 0). The opposite of intuition.

## Portfolio (G4) — exit × regime × concurrency, compounding, MTM

| Rank | Exit | Gate | Concurrent | CAGR | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|---|---|---|
| 1 | Donchian-20 | NIFTY>200DMA | 8 | 20.5% | 1.01 | −36.0% | 0.57 |
| 2 | Donchian-20 | gate | 10 | 19.1% | 0.99 | −34.2% | 0.56 |
| 3 | Donchian-20 | gate | 5 | 21.2% | 0.96 | −39.6% | 0.54 |
| 5 | Supertrend(10,3) | gate | 10 | 20.1% | 1.03 | −39.9% | 0.50 |
| — | Supertrend(10,3) | **no gate** | 10 | 15.3% | 0.78 | −48.5% | 0.32 |
| — | Donchian-20 | **no gate** | 5 | 11.6% | 0.57 | −70.5% | 0.16 |

- **The regime gate is decisive:** it ~halves MaxDD (−34% vs −55/−70%) AND raises CAGR
  (dodging bear-market bleed). Per-year 2025: −4% (gate) vs −44% (no gate); 2022: +20 vs −7.
  **Mandatory.** NIFTY>200-DMA on NIFTYBEES.
- **Donchian-20 ≈ Supertrend(10,3)** — statistically tied; Donchian slightly better Calmar,
  Supertrend slightly better Sharpe. Either is fine.
- **Concurrency 8–10 is the sweet spot** (5 = higher CAGR/higher DD; 15 = diluted).

## Honest caveats (read before trusting the headline)

1. **Survivorship bias (biggest).** Universe = symbols in today's DB — historical breakouts on
   names that later delisted/died are absent. Real returns are lower and real DD deeper than
   shown. The −36% DD and 20% CAGR are optimistic.
2. **Thin early years.** Few ₹5cr-median-turnover names existed 2006–2010; early-year returns
   (2007 +81%) sit on a small book — less reliable. The modern era (2015+) is more credible.
3. **Gross of tax.** ~45–50-day holds = short-term → 20% STCG. Net-of-tax CAGR ≈ 16–17%.
4. **Concentrated long-beta.** Beta 0.45, correlation 0.43 to NIFTY; all breakouts are one
   bet — the −36% DD is correlated-cluster risk (the seven-deadly-sins "single factor").
5. **Selection ranking is not optimal.** When more signals than slots, we ranked by today's
   %-run (user's workflow) — but G1 showed %-run ranking is *not* additive. A better ranker
   (base tightness, RS) is unexplored upside.
6. **No idle-cash yield** modeled (conservative offset, ~+? when <8 positions held).
7. Missing many of the user's exact microcaps (0 rows in DB); this is a *proxy* population on
   a mid/large-cap-skewed liquid universe. Data ends ~2026-05-15 for most names.

## Practical playbook (for the user's discretionary short-term trading)

- **Entry:** your Chartink MTF-bullish + near-52w-high + volume ≥ 2× scan, THEN keep only names
  with **20-day *median* turnover ≥ ₹5cr** and skip circuit-locked / gap-away (unfillable) days.
- **Exit:** **Donchian-20 lower-channel** *or* **Supertrend(10,3)** trailing stop, **NO profit
  target**, 20% catastrophe hard-stop. Ride ~6–7 weeks until the trail breaks.
- **Regime:** only take NEW entries when **NIFTY is above its 200-DMA** (this is the single
  biggest drawdown reducer).
- **Sizing:** ~8 concurrent positions, equal weight.
- Expect to be red on ~55% of trades and make it back on the occasional +70–140% runner.

## Next levers (unexplored upside)
- Better selection ranker than %-run (base tightness / RS-rank / distance-from-base).
- The user's "fine filters" to cut trade frequency to a tradeable cadence + possibly lower DD.
- A breadth-based gate (% of universe > 200DMA) vs the single-index gate.
- Point-in-time / delisting-aware universe to remove survivorship bias.
- Paper-forward soak (G5) before any capital.

*Reproducibility: DB snapshot ~2026-05-15 (CARTRADE to 06-30). Scripts g1_probe / g2_exit_bakeoff
/ g3_clean_bakeoff / g4_portfolio.py in research/71/scripts. Cost 0.20% RT, capital ₹10L,
window 2006–2026. Benchmark NIFTYBEES.*
