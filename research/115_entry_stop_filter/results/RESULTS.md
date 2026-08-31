# RESULTS — Should a rebalance buy a name already below its own stop? (research/115)

**VERDICT: NO MEASURABLE PERFORMANCE DIFFERENCE between all three arms. Decide on operational
grounds, not returns. On those grounds ARM C (backfill deeper) wins, and it is what should be
live.**

## The trigger

On 2026-08-31 the live rebalance ranked RADICO 6th and ADANIENSOL 3rd, then refused both for
trading below their 15-day Donchian lows. RADICO had been SOLD by that same stop **eleven seconds
earlier** in the same run. The book finished 6/8 with two empty slots, and Arun asked why it did not
simply walk further down the ranked list for a name that qualifies.

## Three arms (engine, rules, costs, period all identical; run_lev62 at lev=1.0, 2006-2026)

| Arm | Behaviour | Provenance |
|---|---|---|
| A `none` | Buy the top-8 regardless of price vs stop | what research/62 and /104 actually validated |
| B `skip` | Refuse a name below its stop, leave the slot EMPTY | **shipped live 2026-08-26, never tested** |
| C `backfill` | Refuse it, then take the next qualifier from the 30-name pool | Arun's proposal |

## Full period

| Arm | CAGR | MaxDD | Sharpe | Calmar | blocked | backfilled | slots left empty |
|---|---|---|---|---|---|---|---|
| A none | 33.0% | -22.0% | 1.78 | 1.50 | 0 | 0 | 3 |
| B skip (live) | 33.3% | -23.8% | 1.76 | 1.40 | 52 | 0 | **43** |
| C backfill | 33.3% | -23.6% | 1.77 | 1.41 | 60 | 38 | **3** |

Read alone this says "the guard costs ~0.10 of Calmar". **That reading does not survive.**

## Per era — the ordering flips

| Era | A none | B skip | C backfill |
|---|---|---|---|
| 2006-2010 | 38.1% / -15.3% | 37.6% / -15.9% | 37.9% / **-15.0%** |
| 2011-2015 | 24.4% / -13.2% | **24.7%** / -13.2% | 24.0% / -13.2% |
| 2016-2020 | **29.6%** / -15.1% | 29.8% / -17.0% | 27.9% / **-17.6%** |
| 2021-2026 | 38.8% / -14.6% | 39.8% / -14.3% | **42.4%** / -14.2% |

C is the WORST arm in 2016-2020 and the BEST by 3.6pp in 2021-2026. A wins one era, B wins one, C
wins one. Over 52-60 blocked events in twenty years, that is noise, not edge. **No arm is reliably
better than the others on return or risk.**

## So decide on operational grounds

With performance a wash, the tie-breakers are real and one-sided:

1. **Same-day churn.** Arm A sells a name on its stop and can rebuy it in the same run, seconds
   later — RADICO on 2026-08-31 is the worked example. The backtest prices the round trip at 30bps
   and shrugs; a real book paying real charges, and a human reading the tradebook, should not.
2. **Deployment.** Arm B leaves 43 slots empty over the period; arm C leaves 3. Idle capital is a
   real cost the modelled 6.5% cash return flatters.
3. **Coherence.** It is indefensible for the entry rule to buy what the exit rule is about to sell.

C satisfies all three. B satisfies only the first. A satisfies none.

## Decision

**Adopt C.** Not because it earns more — it does not, reliably — but because at equal performance
it is the only arm that neither churns nor under-deploys.

Note what changes: entries may now come from below the top-8 (38 times in 20 years). The top-22
buffer was always a RETENTION rule; this makes the pool a shallow ENTRY fallback too. That is a
genuine, if small, departure from research/62, made deliberately and on an evidenced wash rather
than by assumption.

## Caveat

Effect sizes here are inside noise; do not quote the full-period Calmar difference as if it were a
finding. The honest claim is "no difference, so choose on mechanics", not "C is better".
