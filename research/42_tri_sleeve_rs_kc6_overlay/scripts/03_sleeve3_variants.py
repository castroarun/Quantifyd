"""Sleeve 3 — short / hedge sleeve, THREE toggleable variants.

User decision (2026-05-17): build all three, the combined engine picks the
winner by post-tax Calmar contribution. None is hard-coded as "the" sleeve.

  A  COVERED_CALL  — write ~OTM monthly calls against the F&O-eligible
                     Sleeve-1 holdings. Pure income; caps a name's upside
                     beyond the call strike for that month. No extra capital.
  B  COLLAR        — A, and spend the call premium on a ~OTM protective
                     put on the same name. Near-zero-cost downside cap;
                     objective: push the −24.6% book DD toward ~−15%.
  C  SYST_SHORT    — independent short alpha: bear-put spread on the
                     WEAKEST-RS decile of the Sleeve-1 mid band (defined
                     risk), sized to a swept short-risk-% of book.

All overlays return a per-month P&L series in **Sleeve-1 NAV units**
(book starts at 1.0) so they compose additively with Sleeve-1 in the
combined engine (04). Option legs use the shared flat-IV BS pricer
(repo convention; STATUS-MD caveat C1). Covered-call assignment is
modelled as an upside cap at expiry intrinsic (caveat C4 — no live
chain, ignores early-assignment/pin).

Smoke-test (laptop snapshot; real sweep on VPS):
    python research/42_tri_sleeve_rs_kc6_overlay/scripts/03_sleeve3_variants.py
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

from config import COLLAR_DEFAULTS  # noqa: E402
from services.collar_engine import (  # noqa: E402
    bs_price, round_call_strike, round_put_strike, infer_strike_interval,
)

# Sleeve-1 driver (for the monthly timeline) + research/41 loaders.
_s1 = importlib.util.spec_from_file_location(
    "s1", str(HERE / "scripts" / "01_sleeve1_base_replay.py"))
s1 = importlib.util.module_from_spec(_s1); _s1.loader.exec_module(s1)
rs2 = s1.rs2

RISK_FREE = COLLAR_DEFAULTS.get("risk_free_rate", 0.065)


def _fno_lots():
    from services.data_manager import FNO_LOT_SIZES
    return {s: int(L) for s, L in FNO_LOT_SIZES.items() if L}


def run_sleeve3(timeline, close, variant="A", otm_pct=5.0, iv=0.25,
                short_risk_pct=0.05):
    """Per-month Sleeve-3 P&L in NAV units. Returns (pnl_series, legs_df).

    timeline : from s1.run_sleeve1 — list of monthly dicts with
               regime_on, total, holdings{sym:nav_value}, prices{sym:px},
               rs_rank (desc RS Series over the mid band).
    variant  : 'A' covered-call | 'B' collar | 'C' systematic-short
    Convention: P&L booked on the month it resolves (S0→S1 over one
    rebalance step). Positive = added to book NAV.
    """
    lots = _fno_lots()
    fno = set(lots)
    pnl = {}
    legs = []
    for i in range(len(timeline) - 1):
        cur, nxt = timeline[i], timeline[i + 1]
        dt0, dt1 = cur["date"], nxt["date"]
        t = max((dt1 - dt0).days, 1) / 365.0

        if variant in ("A", "B"):
            if not cur["regime_on"] or not cur["holdings"]:
                continue
            for sym, nav_val in cur["holdings"].items():
                if sym not in fno or sym not in cur["prices"]:
                    continue
                s0 = cur["prices"][sym]
                s1v = close[sym].loc[:dt1]
                if s0 <= 0 or s1v.empty:
                    continue
                s1p = float(s1v.iloc[-1])
                qty_nav = nav_val / s0          # NAV-unit "shares" held
                kc = round_call_strike(s0, otm_pct)
                call_prem = bs_price(s0, kc, t, RISK_FREE, iv, True)
                # covered-call: per-share (+premium − upside beyond Kc),
                # times the NAV-unit share count (do NOT re-divide by s0).
                cc = (call_prem - max(s1p - kc, 0.0)) * qty_nav
                add = cc
                if variant == "B":
                    kp = round_put_strike(s0, otm_pct)
                    put_cost = bs_price(s0, kp, t, RISK_FREE, iv, False)
                    put_payoff = max(kp - s1p, 0.0)
                    add += (-put_cost + put_payoff) * qty_nav
                # round-trip option cost proxy ~1% of premium notional
                add -= 0.01 * abs(call_prem) * qty_nav
                pnl[dt1] = pnl.get(dt1, 0.0) + add
                legs.append(dict(date=str(dt1), variant=variant, symbol=sym,
                                 s0=round(s0, 2), s1=round(s1p, 2),
                                 kc=kc, call_prem=round(call_prem, 2),
                                 nav_pnl=round(add, 6)))

        elif variant == "C":
            rr = cur.get("rs_rank")
            if rr is None or len(rr) == 0:
                continue
            book = cur["total"]
            # weakest-RS decile that is F&O-eligible (shortable)
            weak = [s for s in rr.sort_values().index if s in fno]
            n = max(1, len(rr) // 10)
            weak = weak[:n][:5]                 # cap concurrent shorts at 5
            if not weak:
                continue
            risk_budget = short_risk_pct * book
            per = risk_budget / len(weak)
            for sym in weak:
                s0v = close[sym].loc[:dt0]
                s1v = close[sym].loc[:dt1]
                if s0v.empty or s1v.empty:
                    continue
                s0 = float(s0v.iloc[-1]); s1p = float(s1v.iloc[-1])
                if s0 <= 0:
                    continue
                iv_i = infer_strike_interval(s0)
                k_long = np.floor(s0 / iv_i) * iv_i      # long put ~ATM
                # short put otm_pct below — otm_pct sets the spread WIDTH
                # (wider = more reward + more defined risk) so it is a real
                # C axis (otm is meaningless for a bear position otherwise).
                k_short = round_put_strike(s0, otm_pct)
                if k_short >= k_long:
                    k_short = k_long - iv_i
                pl0 = bs_price(s0, k_long, t, RISK_FREE, iv, False)
                ps0 = bs_price(s0, k_short, t, RISK_FREE, iv, False)
                debit = pl0 - ps0                         # net debit (risk)
                if debit <= 0:
                    continue
                units = per / debit                       # scale to risk budget
                pl1 = max(k_long - s1p, 0.0)               # at-expiry intrinsic
                ps1 = max(k_short - s1p, 0.0)
                val1 = pl1 - ps1
                gross = (val1 - debit) * units
                gross -= 0.02 * debit * units             # round-trip cost
                pnl[dt1] = pnl.get(dt1, 0.0) + gross
                legs.append(dict(date=str(dt1), variant="C", symbol=sym,
                                 s0=round(s0, 2), s1=round(s1p, 2),
                                 k_long=k_long, k_short=k_short,
                                 debit=round(debit, 2),
                                 nav_pnl=round(gross, 6)))

    sp = pd.Series(pnl).sort_index()
    if not sp.empty:
        sp.index = pd.to_datetime(sp.index)
    return sp, pd.DataFrame(legs)


def main():
    print("Sleeve-3 — loading Sleeve-1 timeline (this replays the base) ...",
          flush=True)
    close, tv = rs2.load()
    import sqlite3
    con = sqlite3.connect(rs2.DB)
    vdf = pd.read_sql("SELECT symbol,date,volume FROM market_data_unified "
                      "WHERE timeframe='day' AND date>='2012-06-01' "
                      "ORDER BY symbol,date", con, parse_dates=["date"])
    con.close()
    vol = vdf.pivot_table(index="date", columns="symbol",
                          values="volume").sort_index()
    nav, timeline = s1.run_sleeve1(close, tv, vol)
    base_final = nav["nav"].iloc[-1]
    print(f"  Sleeve-1 timeline: {len(timeline)} months, "
          f"final NAV={base_final:.2f} (book unit, start 1.0)", flush=True)

    for v in ("A", "B", "C"):
        for otm in (3.0, 5.0, 7.0):
            sp, lg = run_sleeve3(timeline, close, variant=v, otm_pct=otm,
                                 iv=0.25, short_risk_pct=0.05)
            if sp.empty:
                print(f"  {v} otm={otm:.0f}%: no legs"); continue
            tot = sp.sum()
            pct_of_book = tot / base_final * 100
            print(f"  {v} otm={otm:.0f}%: legs={len(lg):4d} "
                  f"sum NAV P&L={tot:+.3f} "
                  f"(~{pct_of_book:+.1f}% of final book) "
                  f"months_active={sp.ne(0).sum()}", flush=True)
            if otm == 5.0:
                lg.to_csv(RES / f"sleeve3_{v}_otm5_legs.csv", index=False)

    print("\n  Note: P&L is in Sleeve-1 NAV units (composes additively in "
          "04). A/B cap upside on held names; C is independent short alpha. "
          "Caveats C1 (flat-IV BS) + C4 (assignment = expiry-intrinsic cap).")


if __name__ == "__main__":
    main()
