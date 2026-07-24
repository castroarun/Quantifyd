# Research 75 Phase 3 — Can per-stock risk controls REPLACE the index gate?

**VERDICT: NO. The NIFTYBEES-100EMA gate is irreplaceable. No combination of quality filter,
ATH-proximity, or per-stock MA/Donchian exits gets a gate-less book below −46% drawdown (vs the
gate's −29%). Stacking those filters ON TOP of the gate doesn't help either — the gate alone is
the whole risk story; the extra machinery only costs return.**

Midcap universe, RS-120 momentum, N=15, 2006–2026, daily-marked, net 0.3%.

| Risk control (gate OFF unless noted) | Net CAGR | MaxDD | Calmar | Stops |
|---|---|---|---|---|
| **Gate ON (baseline)** | 36.2% | **−29.2%** | **1.24** | 0 |
| No gate, nothing | 32.9% | −65.2% | 0.50 | 0 |
| + quality filter | 36.3% | −64.6% | 0.56 | 0 |
| + ATH-proximity (≥90% of ATH) | 23.1% | −58.4% | 0.39 | 0 |
| + SMA100 per-stock exit | 32.7% | −53.9% | 0.61 | 772 |
| + Donchian-15 exit (best gate-less) | 31.0% | −46.2% | 0.67 | 2120 |
| + quality + ATH + SMA100 | 25.6% | −60.7% | 0.42 | 475 |
| **gate ON + quality + ATH + SMA100 (full stack)** | 31.8% | −30.2% | 1.05 | 262 |

## Findings

1. **Quality filter does nothing for drawdown** (−64.6% vs −65.2% ungated). It is not a market-timer.
2. **ATH-proximity is actively harmful bolted onto the bare book** — slashes CAGR (36→23%) without
   fixing DD. (It only helps inside Aurum's fuller construction, not standalone.)
3. **Per-stock exits (SMA100 / Donchian) partially help but at brutal churn** (772–2120 stops, tax
   bleed) and still leave −46 to −54% DD — nowhere near the gate's −29%.
4. **Adding all filters on top of the gate** gives −30.2% (vs gate-alone −29.2%) at LOWER CAGR — no help.
5. **This reconciles the Aurum-vs-research/75 drawdown gap:** Aurum's shallower DD (−19–22%) does NOT
   come from quality/ATH filters (proven inert here) — it comes from research/41's always-on
   month-end MA-overlay + weekly gate, a different mechanic than these event-driven ports.

**Reconfirms research/41's core lesson:** a per-stock rule cannot be a market circuit-breaker —
stocks break one at a time, after they've already fallen. Collective (index-gate) exit beats N
idiosyncratic ones.

Runner: `scripts/run_phase3_combos.py`. Data: `ranking_v3.csv`.
