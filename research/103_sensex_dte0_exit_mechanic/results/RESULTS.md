# research/103 — SENSEX ATM2 Expiry-Day Exit Mechanic — RESULTS

**Verdict: G2 DIAGNOSTIC — the expiry-gamma trap is REAL, but the proposed ₹2,500 rupee-stop fix is
the WRONG fix. NO CHANGE to ATM2's 0.4% move-stop. The real lever is DTE0 sizing/participation, not
stop type.**

> ⚠️ **Modeled study.** No real SENSEX (BFO) option intraday exists — the straddle premium path is
> Black-Scholes-modeled off the SENSEX index 1-min + an assumed IV (10/12/15/18%). **Absolute P&L is
> unreliable** (every rule shows a modeled loss because the assumed IV underprices real expiry-day
> premium). The robust, IV-invariant outputs are (a) the **loss-when-stopped ratio across DTE** and
> (b) the **relative ranking of the exit rules** — both stable at every IV. Sample: 1,353 trading
> days 2021→2026 (240 DTE0 Thu, 275 DTE1 Wed, 833 DTE2+). Entry 09:16, square 15:25, 1 lot, net of
> ~₹200/lot cost.

## Finding 1 — the gamma trap is CONFIRMED

The *same* 0.4% move-stop's average **loss-when-triggered scales ~3.5× into expiry** (IV12%):

| DTE | mean loss when stopped | p95 loss | stop rate |
|---|---|---|---|
| **0 (Thu)** | **₹2,481/lot** | ₹3,626 | 82% |
| 1 (Wed) | ₹1,058 | ₹1,861 | 80% |
| 2+ | ₹708 | ₹1,347 | 73% |

Same shape at every IV. A fixed *spot-distance* stop (≈315 pts) is DTE-blind, so as gamma explodes
into expiry the same move crystallises a far bigger rupee loss, and DTE0's higher intraday vol trips
it more often. Arun's instinct — that DTE0 is where the move-stop bites hardest — is correct.

## Finding 2 — the ₹2,500 rupee stop does NOT fix DTE0 and HURTS normal days

Porting research/96's NIFTY rupee stop to SENSEX would be a **net negative** (mean net/lot, IV12%):

| Rule | DTE0 | DTE1 | DTE2+ |
|---|---|---|---|
| **MOVE_0.4% (current)** | **−2,016** | **−938** | **−632** |
| RUPEE_2500 | −2,150 | −1,462 | −1,146 |
| HOLD_EOD | −4,020 | −1,847 | −1,699 |

- On **DTE0** the rupee stop is ~tied with the move-stop (its ₹2,500 cap ≈ the loss a 0.4% move
  already makes at DTE0) → **no expiry-day improvement.**
- On **DTE2+** the rupee stop is **~2× worse** — it lets a normal-day loss run to ₹2,700 before
  cutting, while the move-stop cuts at ₹708. Its loss-when-stopped is a flat ~₹2,700 at *every* DTE
  (that's the point of a rupee stop) — which is exactly wrong on low-gamma days.

## Finding 3 — the current move-stop is the LEAST-BAD rule; tighter/faster beats looser/hold

Across every DTE bucket and IV, `MOVE_0.4%` has the best (least-negative) mean net; `HOLD_EOD` is
worst (a short straddle bleeds through adverse moves). So the move-stop is not a mechanic to replace
— if anything DTE0 wants a *tighter/faster* cut, not a looser rupee stop. (The "never hold"
strength is partly the IV artifact — at IV18% hold approaches breakeven on DTE1/2+ — so treat it as
directional, not absolute.)

## What it means for the live decision (fast-follow #2)

1. **Keep ATM2's 0.4% move-stop as-is.** The naive fix (₹2,500 rupee stop, ported from NIFTY) is
   refuted here — no DTE0 help, worse on normal days. This study's value is preventing a
   plausible-but-wrong change.
2. **The real DTE0 problem is participation/size, not stop type.** DTE0 losses dwarf DTE1/DTE2+
   under *every* rule (consistent with research/97's "DTE0 win 14%"). Levers worth a real study:
   smaller Thursday sizing, a *tighter* DTE0 stop, or skipping DTE0 and leaning DTE1.
3. **Cannot answer "should we trade SENSEX DTE0 at all" from a modeled straddle** — every rule is
   modeled-negative, but that's the IV artifact. That question needs **real BFO option intraday**
   (start recording the SENSEX chain intraday; today we only have daily snapshots).

**Gate:** does not clear anything to a live change → consistent with Option A (hold ATM2). Seven
sins: look-ahead none (causal bar-walk); cost netted; overfitting guarded (no param picked, IV
sensitivity shown, whole-sample); the binding open risk is **no real option data** (modeled premium)
— stated loudly, caps this at a diagnostic.
