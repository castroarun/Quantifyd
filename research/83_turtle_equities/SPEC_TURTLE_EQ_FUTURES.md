# SPEC — Turtle-EQ Long Book on Stock Futures (paper-first)

Source of validation: research/83 EXP-T2 arm C (2005–2023, gated, net of
futures-proxy costs): CAGR 14.4%, MaxDD −31.9%, Calmar 0.45 vs NIFTYBEES
13.5%/−59.7%/0.23. Standing caveat: family ~flat 2018–2023 ex-2020/21 —
REASON THIS IS PAPER-FIRST. Shorts excluded (closed at all horizons,
r/81+82+83).

## Universe & instruments

- NSE stock futures, current F&O list (~80 underlyings), near-month contract.
- Liquidity filter: 20d avg futures turnover ≥ ₹25 Cr; skip names in ban period.
- LONG ONLY.

## Entry (either system, evaluated on UNDERLYING daily closes)

- S1: close > max(high, 20 prior sessions) → buy next open.
- S2: close > max(high, 55 prior sessions) → buy next open.
- One position per underlying (S1/S2 signals on the same name share it).
- Gate: entries allowed only when NIFTYBEES prev close > its 200-DMA
  (existing positions RIDE the gate — exits below are the only exits).

## Exits (first hit wins)

1. Hard stop: entry_price − 2×N, N = ATR20 (Wilder-simple) at signal;
   GTT/SL-M at stop; gap-through fills at open (modeled and accepted).
2. Trailing channel: S1 position → close < min(low, 10 prior sessions);
   S2 position → close < min(low, 20 prior sessions); exit next open.
3. No profit target. No time stop.

## Sizing & risk (LOCKED — equal-notional; N-sizing tested and REJECTED)

- Equal-notional: target 12% of book equity per position, rounded to nearest
  whole lot (skip if 1 lot > 20% of equity).
- Max 8 concurrent positions; max gross notional 120% of equity.
- Book: ₹20–25L paper. Margin ≈ 18–22%/lot → ~₹10–16L margin at full load;
  keep ≥30% equity as unencumbered buffer (MTM calls in drawdowns are the
  liquidation risk of this wrapper — buffer is non-negotiable).
- No pyramiding in v1 (untested).

## Rolls

- Roll on T−2 before expiry: close near-month, open next-month at market,
  same lot count. Roll cost is real financing — do NOT count collateral
  yield as strategy alpha.

## Failure modes

- Missed fill/ban-period entry → skip, do not chase.
- Circuit/halt → exit at next available print; log deviation.
- Data outage → no new entries; existing GTT stops remain live.
- Kill-switch: flatten-all endpoint; also auto-halt new entries if book
  equity < 85% of trailing peak (review before re-arm).

## Live-vs-backtest tracking

- Log every fill vs model price (slippage budget 3 bps/side; alert >10 bps).
- Weekly: paper NAV vs the T2 arm-C backtest path; alert if tracking error
  drifts >2%/month. Verdict review after ~2 quarters of soak (the 2018-23
  flatness question is what the soak must answer).

## Pseudocode

    daily after close:
      update gate; for each open pos: check trail-exit for tomorrow-open
    daily 09:14:
      place/refresh GTT stops
    daily 09:15+1min:
      execute queued exits at open; then if gate ON and slots free:
        for signals from yesterday (S2 priority, then S1, gap% desc):
          size to 12% notional in lots; buy at open
    T-2 expiry: roll all
