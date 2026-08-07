# Momentum-250 Book — Leverage × Concentration for Return (gated book, 2006–2026 daily)

STATUS: RUNNING (build + smoke, then N×LEV sweep)

## 1. The Ask
**Arun:** optimize the momentum-30 book for RETURN via futures leverage / regime filter / concentration.
**What we're testing:** The Nifty-250 momentum book (research/75: net ~32–35% CAGR, −32% DD, index-EMA cash gate = the whole risk story) already beats B&H. Can we raise RETURN further — without the drawdown becoming ruinous — by (a) LEVERAGE applied only while the gate is risk-on (borrow cost + margin-call modelled), (b) CONCENTRATION (fewer names), (c) a better momentum score (risk-adjusted). Honest: leverage a −32% DD book 2× and it can hit −64% or margin-call; the gate may or may not save it. Find the efficient frontier vs NIFTYBEES B&H.

## 2. The Base
- Engine: research/75 `run_nifty250_momentum.py` (daily-marked NAV, PIT top-250 universe, monthly rebalance, NIFTYBEES>EMA100 gate, cost 0.3% RT, cash 6.5%). A3 (drop EMA-stack) is return-best.
- New levers in `run_lev_conc.py`:
  - **Leverage `lev`** ∈ {1.0, 1.3, 1.6, 2.0}: deploy `lev×equity` in the top-N while gate risk-on; borrowed cash accrues **8% p.a.** financing (futures-basis / margin-funding proxy). In cash (gate off) → no leverage, 6.5% yield.
  - **Margin call:** if own-equity / gross-notional < 0.25 (≈ −30% gross adverse move at 2×), force-liquidate at marks → models ruin honestly.
  - **Concentration `N`** ∈ {5, 8, 12}.
  - **Score:** `radj_z` (risk-adjusted, best DD) primary; `ret252` control.
  - EMA-stack OFF (A3), index gate ON.
- **Success:** net CAGR vs NIFTYBEES B&H (11.6%/−60%DD) AND vs the unlevered book (34.7%/−32%); Calmar must not collapse; no margin-call ruin in-sample.

## 3. Plan — grid
- score=radj_z × N{5,8,12} × lev{1.0,1.3,1.6,2.0} = 12 cells + a few ret252 controls. Net-of-cost, daily-marked, 2006–2026. Post-tax20 + borrow-rate sensitivity (10%) on the winner only.

## 4. Status
| Time (IST) | Event | Notes |
|---|---|---|
| 2026-08-06 | Build run_lev_conc.py (leverage+margin-call+concentration) | imports research/75 engine |

## 5. Crash Recovery
- Runner: `research/104_momentum_leverage/scripts/run_lev_conc.py` (imports research/75 module by path). Writes `results/lev_conc.csv` incrementally, resumable (skips done configs). Rerun: `ssh arun@94.136.185.54 'cd /home/arun/quantifyd && nohup ./venv/bin/python3 research/104_momentum_leverage/scripts/run_lev_conc.py > /tmp/lev.log 2>&1 &'` then `tail -f /tmp/lev.log`.

## 6. Files
| File | Purpose | Commit? |
|---|---|---|
| `scripts/run_lev_conc.py` | Leverage×concentration runner | yes |
| `results/lev_conc.csv` | Per-config results | yes |
| `results/RESULTS.md` | Verdict | yes |

## 7. Findings — DONE 2026-08-06

**Leverage on the gated momentum book RAISES return and SURVIVES (0 margin calls / 20y incl 2008+2020).**
The index-EMA cash gate liquidates before drawdowns turn ruinous, so leverage that would wipe an ungated
book is survivable here. Net, daily-marked, 2006-2026. B&H = 11.7%/−60%/Cal 0.20.

Frontier (radj_z, N8):
| lev | CAGR | MaxDD | Sharpe | Calmar |
|---|---|---|---|---|
| 1.0 | 33.8% | −29.9% | 1.49 | 1.13 |
| 1.3 | 41.4% | −38.9% | 1.40 | 1.06 |
| 1.6 | 48.7% | −47.8% | 1.35 | 1.02 |
| 2.0 | 57.8% | −59.6% | 1.29 | 0.97 |

ret252 (plain 12m) N8: L1.0 37.5%/−34.5%/1.09 … L2.0 64.4%/−67.1%/0.96 — higher raw return, worse DD.

Reads: (1) return scales strongly with leverage, DD ~proportionally, Calmar drifts DOWN (no free lunch).
(2) N8 = concentration sweet spot (best Calmar vs N5/N12). (3) radj_z best risk-adjusted; ret252 highest raw.
**Recommended return upgrade: radj_z N8 L1.3–1.6 → 41–49% CAGR, −39 to −48% DD, Calmar ~1.0** (+7–15% CAGR
over unlevered). >1.6× is greed (Calmar<1, DD>−55%).

Caveats: multiples (8k–28k×) are 20y compounding fantasies — trust CAGR/DD/Calmar. −48% DD is real & brutal.
Margin-call model = 25% maint on DAILY marks; a gapping crash could force liquidation this understates (gap risk).
Financing 8% assumed. Same momentum edge magnified, not new alpha.

### Next lever: VOL-TARGETING
Scale leverage inversely to realized portfolio vol (target constant vol) → auto-deleverage in turbulence,
re-lever in calm. Standard way to make leverage risk-efficient; should lift the levered Calmar back toward 1.1+.

STATUS: DONE
