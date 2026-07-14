# research/79 — NAS-OPT replayed on all 58 recorded chain days → **the DTE gate IS the strategy**

**Verdict: STRATEGY (as defined). The 0/1-DTE restriction is not a detail — it is the entire edge.
Run on every day, NAS-OPT makes +₹429/day. Run only on 0/1-DTE, it makes +₹1,578/day. The extra
days are not merely weaker; they are NEGATIVE (−₹441/day) and drag the total DOWN from +39,440 to
+24,871.** The all-days paper-trading switched on 2026-07-14 must stay tagged `observational` and
must NEVER be promoted into the system's result.

## Validated against the live book first (this is why it can be trusted)

11 of the 58 days were actually paper-traded live. Replay vs the real record:

| | |
|---|---|
| Same strikes on | **8 / 11** days → diff −435 to +97 (mostly < ₹300) |
| Live total | **+15,361** |
| Replay total | **+14,634** (within 5%) |
| Mean diff | **−66** · median −58 |

The 3 mismatched days (06-08, 06-22, 06-23) are NOT a bug: the 09:20 spot sat near a 25-pt
boundary, so ATM rounding flipped the strikes by 50. **That is a real fragility of NAS-OPT** — which
tick it reads at 09:20 can swing a day by thousands. Worth knowing; not fixable by better code.

## Headline (58 days, 2026-04-20 → 2026-07-14, 2 lots/leg, net of ₹80/leg)

| slice | n | total | mean | median | win% | worst |
|---|---|---|---|---|---|---|
| **SYSTEM (0/1-DTE)** | 25 | **+39,440** | **+1,578** | +2,024 | **68%** | −4,222 |
| OBSERVATIONAL (DTE≥2) | 33 | **−14,569** | −441 | −823 | 33% | −4,866 |
| ALL DAYS | 58 | +24,871 | +429 | −160 | 48% | −4,866 |

## The edge decays monotonically with DTE — and this is the whole story

| DTE | n | mean | win% |
|---|---|---|---|
| **0** | 12 | **+2,045** | 67% |
| **1** | 13 | **+1,147** | 69% |
| 4 | 11 | +494 | 55% |
| 5 | 11 | **−953** | 27% |
| 6 | 11 | **−865** | 18% |

Clean monotone decay from +2,045 → −865. Not a peak, not noise — a structure. Near expiry, theta is
large and the ±0.4% band is rarely hit; far from expiry, premium is fat, the band gets hit anyway,
and the move-stop is pure cost. (Weekday is just DTE in disguise: Tue=DTE0, Mon=DTE1, Wed=DTE6.)

## The move-stop is the cost centre — and it is concentrated in far-DTE

| | days | total | mean |
|---|---|---|---|
| Held to 14:45 | 22 | +63,281 | **+2,876** |
| Move-stop fired | 36 | −38,410 | **−1,067** |

But split by class: on SYSTEM days the move-stop costs only **−325/day** (14 days); on
OBSERVATIONAL days it costs **−1,539/day** (22 days). The stop is not broken — it is being asked to
protect a position that should never have been opened.

## Caveats (read before acting)

1. **58 days, one regime** (Apr–Jul 2026). No crash, no vol spike.
2. **ATM-rounding fragility** (above) makes day-level P&L genuinely noisy — 3 of 11 validation days
   flipped strikes on a knife-edge.
3. Fills are the recorded chain print at the trigger snapshot; only ₹80/leg brokerage is charged.
   No extra slippage modelled — a **cost sensitivity is owed** before this is called a STRATEGY at G4.
4. DTE2/DTE3 are absent from the sample (no such days in the window), so the decay curve has a gap.

## What this changes

- **Nothing about the live system.** The 0/1-DTE gate stays. It is the edge.
- **Confirms the 2026-07-14 decision** to keep all-day entries tagged `observational` — the data now
  says those days are EV-negative, so mixing them in would have quietly destroyed the record.
- The obvious follow-up: **is the ±0.4% move-stop worth keeping even on 0/1-DTE?** It costs
  −325/day there. Test: no-stop vs the band, 0/1-DTE only.
