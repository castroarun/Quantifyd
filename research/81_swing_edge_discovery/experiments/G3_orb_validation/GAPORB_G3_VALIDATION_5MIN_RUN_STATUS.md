# G3 Validation — Gap-Up + ORB Long (≤4d hold) on NIFTY + 9 F&O Stocks

STATUS: DONE — G3 PASS (see Verdict)

EXP-G3 of research/81. The lead candidate earned this by F2 (index, IS t=3.0,
Val borderline pass) + A3 (stocks, t=2.87, 8/9 positive). G3's job is to
KILL it if it's overfit. Broad-universe + BANKNIFTY replication follows
post-backfill; OOS 2024+ stays locked throughout.

## Locked configs under test (no re-optimization anywhere in G3)

- **Index:** NIFTY, W=12, gap ≥ 0.25%, long first close > OR-high, stop =
  OR-low, exit close of session entry+3. Costs: futures-proxy, 1bp & 3bp.
- **Stocks:** 9 deep F&O names, W=6, same rule, stock-level gap.

## Plan (all pre-declared, IS = 2015-02→2021-09 unless stated)

1. **Parameter plateau (sensitivity, not selection):** W {4,6,9,12,15} ×
   gap_thr {0.10,0.20,0.25,0.35,0.50}% heatmap of net bps & t on IS, per
   instrument set. PASS = locked point sits on a plateau (neighbors within
   ~30% of its net bps, no sign flips); FAIL = lone spike → candidate dies.
   (Ledger: +50 sensitivity cells, flagged as robustness-scan not selection.)
2. **Walk-forward, no re-opt:** locked config rolled in 6-month windows
   2015-02→2023-12 (IS+Val; Val already consumed for family F, reused ONLY
   read-only for the locked config). PASS = ≥60% of half-years net-positive
   and no catastrophic half (< −60 bps avg/trade).
3. **Monte Carlo:** trade-order bootstrap (2000 iters, engine.metrics) on
   pooled locked-config trades → 95th-pct MaxDD and P(loss over 3y) at
   per-trade risk sizing. Report against brief gates (MC MaxDD ≤ 30% at
   1%-risk-per-trade equivalent scaling).
4. **Regime splits:** VIX terciles (causal trailing), NIFTY vs 200DMA
   (risk-on/off), per-year. Edge must not be single-regime.
5. **Super-winner guard (stocks):** drop the top-2 contributing symbols,
   recompute. PASS = still net-positive with t > 1.5.

## Status

| Date/time | Event | Notes |
|---|---|---|
| 2026-07-15 ~23:00 IST | Pre-registered | runner: `scripts/run_g3_orb_validation.py` |

## Findings

(after run)

## VERDICT (2026-07-15 ~23:15 IST): G3 PASS — STRATEGY CANDIDATE (pending G4 + OOS)

1. PLATEAU: index all 25 cells positive (19-47bps) with net RISING in gap
   threshold and ~flat in W — locked point on a broad plateau, not a spike.
   Stocks all 25 positive (14-22bps), flat. PASS.
2. WALK-FORWARD (no re-opt, 2015-2023): index 83% of half-years positive
   (worst -28.9bps avg/trade), stocks 89% (worst -19.3). PASS.
3. MONTE CARLO @1% risk/trade: index p95 MaxDD -23.2% (gate <=30% PASS),
   P(3y loss) 2.6%. Stocks p95 -53% — sizing constraint: stock sleeve must
   run ~0.5% risk/trade or capped concurrency -> G4 design input, not a kill.
4. REGIMES: index positive in ALL VIX terciles and BOTH 200DMA regimes
   (below-200DMA actually stronger, +41bps — bear-market gap-ups =
   short-covering continuation). Stocks positive everywhere, flat in VIX-high.
   Not a regime bet. PASS.
5. SUPER-WINNER GUARD: drop ICICIBANK+TCS -> +12.2bps t=2.2 on 2137 trades.
   PASS. (BHARTIARTL the lone negative name.)

Next: G4 portfolio construction (combined index+stocks book, NAV, sizing,
concurrency, correlation vs live books) -> USER CHECKPOINT -> single
pre-declared OOS touch (2024+). Broad-universe + BANKNIFTY replication
post-backfill remains queued as breadth confirmation.
STATUS: DONE
