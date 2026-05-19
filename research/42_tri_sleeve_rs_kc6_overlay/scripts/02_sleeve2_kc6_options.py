"""Sleeve 2 — KC6 mean-reversion monetised as defined-risk option spreads.

Replays the KC6 signal (services/kc6_scanner logic, verbatim rules) on its
NATIVE Nifty500 ∩ F&O universe, bar-by-bar 2014→2026, and expresses every
KC6 entry as BOTH:

  * CREDIT  — bull-put spread: SELL put ~just-OTM, BUY put one strike-
              interval lower. Profit = bounce + theta. Max loss =
              (width − net_credit) × lot.
  * DEBIT   — bull-call spread: BUY call ~ATM, SELL call at the KC6-mid
              target. Profit = directional bounce. Max loss = net_debit × lot.

Per name we can pick the better expression by realised post-cost
expectancy + an option-liquidity proxy (user decision: "backtest both,
pick per name liquidity"). Option legs priced by the SHARED flat-IV
Black-Scholes pricer in services/collar_engine (repo convention) — see
STATUS-MD caveat C1 (no historical option chains; indicative not
tradeable-accurate; mandatory IV sensitivity {0.20,0.25,0.30}).

Capital model (STATUS-MD §2, user-corrected 2026-05-17): Sleeve-2 funds
= margin against the whole book, CONSTANT and REGIME-INDEPENDENT. The
only size constraint is a swept max-defined-risk-% of book, enforced by
the combined engine (04). This module exposes a clean
`run_sleeve2_kc6(...)` the combined engine calls; standalone `main()` is
a smoke test that does NOT depend on the live collar DB.

Smoke-test (laptop snapshot; real sweep on VPS per canonical-host rule):
    python research/42_tri_sleeve_rs_kc6_overlay/scripts/02_sleeve2_kc6_options.py
"""
from __future__ import annotations
import importlib.util
import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

from config import KC6_DEFAULTS, COLLAR_DEFAULTS  # noqa: E402
from services.kc6_scanner import compute_indicators  # noqa: E402
from services.collar_engine import (  # noqa: E402  shared pure helpers
    bs_price, infer_strike_interval, pick_expiry,
)

DB = ROOT / "backtest_data" / "market_data.db"
START = "2014-01-01"
RISK_FREE = COLLAR_DEFAULTS.get("risk_free_rate", 0.065)

# KC6 rules (from KC6_DEFAULTS) — replayed verbatim.
SL_PCT = KC6_DEFAULTS["sl_pct"] / 100.0          # 5%
TP_PCT = KC6_DEFAULTS["tp_pct"] / 100.0          # 15%
MAX_HOLD = KC6_DEFAULTS["max_hold_days"]          # 15 calendar days
ATR_THRESH = KC6_DEFAULTS["atr_ratio_threshold"]  # 1.3 crash filter
MAX_POS = KC6_DEFAULTS["max_positions"]           # 5 concurrent
EXPIRY_EXIT_DAYS = COLLAR_DEFAULTS.get("expiry_exit_days", 3)
MIN_EXPIRY_DAYS = COLLAR_DEFAULTS.get("min_expiry_days", 7)


# --------------------------------------------------------------------------
# Universe + data
# --------------------------------------------------------------------------

def fno_universe():
    from services.data_manager import FNO_LOT_SIZES
    con = sqlite3.connect(DB)
    have = {r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='day'"
    ).fetchall()}
    con.close()
    uni = {s: int(L) for s, L in FNO_LOT_SIZES.items() if s in have and L}
    return uni


def load_panel(symbols):
    """symbol -> OHLCV DataFrame with KC6 indicators, indexed by date."""
    con = sqlite3.connect(DB)
    out = {}
    for s in symbols:
        df = pd.read_sql_query(
            "SELECT date,open,high,low,close,volume FROM market_data_unified "
            "WHERE symbol=? AND timeframe='day' ORDER BY date",
            con, params=[s], parse_dates=["date"])
        if len(df) < 260:
            continue
        df = df.set_index("date").astype(float)
        compute_indicators(df)             # adds kc6_*, sma200, atr_ratio
        out[s] = df
    con.close()
    return out


def universe_atr_ratio_series(panel):
    """Per-date median atr_ratio across the universe (the KC6 crash filter)."""
    ar = pd.DataFrame({s: d["atr_ratio"] for s, d in panel.items()})
    return ar.median(axis=1)


# --------------------------------------------------------------------------
# Spread construction (flat-IV BS, shared pricer)
# --------------------------------------------------------------------------

def _spread_legs(spot, kc6_mid, expression):
    """Return (k_long, k_short, is_call) for the chosen spread expression.
    CREDIT bull-put: short put just-OTM (below spot), long put 1 interval
                     lower. (k_long < k_short, is_call=False)
    DEBIT bull-call: long call ~ATM, short call at/above KC6-mid target.
                     (k_long < k_short, is_call=True)
    """
    iv = infer_strike_interval(spot)
    if expression == "credit":
        k_short = np.floor(spot / iv) * iv          # just-OTM put (sold)
        k_long = k_short - iv                        # protection (bought)
        return float(k_long), float(k_short), False
    else:  # debit bull-call
        k_long = np.floor(spot / iv) * iv            # ~ATM call (bought)
        tgt = max(kc6_mid, spot * 1.02)              # short at KC6-mid target
        k_short = np.ceil(tgt / iv) * iv
        if k_short <= k_long:
            k_short = k_long + iv
        return float(k_long), float(k_short), True


def _spread_value(spot, k_long, k_short, is_call, expiry, today, iv):
    """Net spread value (per share) at (spot,today). For credit it is the
    cost to BUY BACK (short − long); for debit it is the spread's worth
    (short − long for a call debit = long − short value... use signed)."""
    dte = max((expiry - today).days, 0)
    t = dte / 365.0
    p_long = bs_price(spot, k_long, t, RISK_FREE, iv, is_call)
    p_short = bs_price(spot, k_short, t, RISK_FREE, iv, is_call)
    return p_long, p_short


def run_sleeve2_kc6(panel, lots, uar, expression="credit", iv=0.25,
                    start=START, max_pos=MAX_POS):
    """Replay KC6 → option spread. Returns (daily_pnl, trades_df).

    daily_pnl : pd.Series indexed by date — REALISED spread P&L (rupees)
                booked on the exit date (the combined engine adds this to
                the book and applies the margin/risk-% cap separately).
    trades_df : one row per closed spread (entry/exit, legs, pnl, max_risk).

    Sizing: 1 lot per signal, ≤ max_pos concurrent (KC6 convention). The
    book-level max-risk-% cap is applied by the combined engine, not here.
    """
    all_dates = sorted({d for df in panel.values() for d in df.index
                        if d >= pd.Timestamp(start)})
    open_pos = []          # list of dicts
    trades = []
    pnl = {}

    for dt in all_dates:
        today = dt.date()
        crash_on = bool(uar.get(dt, np.nan) >= ATR_THRESH) if dt in uar.index \
            else False

        # ---- exits first (free slots) ----
        still = []
        for p in open_pos:
            df = panel[p["symbol"]]
            if dt not in df.index:
                still.append(p); continue
            row = df.loc[dt]
            hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])
            kc6_mid = row.get("kc6_mid", np.nan)
            hold_days = (today - p["entry_date"]).days
            dte = (p["expiry"] - today).days
            spot_entry = p["spot_entry"]
            reason = None
            if dte <= EXPIRY_EXIT_DAYS:
                reason = "EXPIRY_NEAR"
            elif lo <= spot_entry * (1 - SL_PCT):
                reason = "STOP_LOSS"
            elif hi >= spot_entry * (1 + TP_PCT):
                reason = "TAKE_PROFIT"
            elif hold_days >= MAX_HOLD:
                reason = "MAX_HOLD"
            elif pd.notna(kc6_mid) and hi > float(kc6_mid):
                reason = "SIGNAL_KC6_MID"
            if reason is None:
                still.append(p); continue

            pl_now, ps_now = _spread_value(cl, p["k_long"], p["k_short"],
                                           p["is_call"], p["expiry"], today, iv)
            if expression == "credit" or p["expression"] == "credit":
                # opened for net credit = ps0 - pl0 ; close cost = ps_now - pl_now
                close_val = ps_now - pl_now
                gross = (p["open_net"] - close_val) * p["lot"]
            else:
                # debit bull-call: opened net debit = pl0 - ps0 ; value now = pl_now - ps_now
                val_now = pl_now - ps_now
                gross = (val_now - p["open_net"]) * p["lot"]
            # round-trip option cost proxy: 1% of |notional touched|
            cost = 0.01 * (abs(p["open_net"]) + abs(close_val if
                     (p["expression"] == "credit") else val_now)) * p["lot"]
            net = gross - cost
            pnl[dt] = pnl.get(dt, 0.0) + net
            trades.append(dict(
                symbol=p["symbol"], expression=p["expression"],
                entry_date=str(p["entry_date"]), exit_date=str(today),
                spot_entry=round(spot_entry, 2), spot_exit=round(cl, 2),
                k_long=p["k_long"], k_short=p["k_short"],
                lot=p["lot"], open_net=round(p["open_net"], 2),
                reason=reason, gross=round(gross, 0), net=round(net, 0),
                max_risk=round(p["max_risk"], 0),
                hold_days=hold_days))
        open_pos = still

        # ---- entries (blocked by crash filter) ----
        if crash_on or len(open_pos) >= max_pos:
            continue
        held = {p["symbol"] for p in open_pos}
        for s, df in panel.items():
            if len(open_pos) >= max_pos:
                break
            if s in held or dt not in df.index:
                continue
            row = df.loc[dt]
            kl, sma2 = row.get("kc6_lower", np.nan), row.get("sma200", np.nan)
            cl = float(row["close"])
            if not (pd.notna(kl) and pd.notna(sma2)
                    and cl < kl and cl > sma2):
                continue
            kc6_mid = float(row.get("kc6_mid", cl))
            expr = expression
            k_long, k_short, is_call = _spread_legs(cl, kc6_mid, expr)
            expiry = pick_expiry(today, MIN_EXPIRY_DAYS)
            pl0, ps0 = _spread_value(cl, k_long, k_short, is_call,
                                     expiry, today, iv)
            width = abs(k_short - k_long)
            if expr == "credit":
                open_net = ps0 - pl0                 # net credit received (>0)
                max_risk = (width - open_net)
            else:
                open_net = pl0 - ps0                 # net debit paid (>0)
                max_risk = open_net
            if max_risk <= 0:
                continue
            open_pos.append(dict(
                symbol=s, expression=expr, entry_date=today,
                spot_entry=cl, k_long=k_long, k_short=k_short,
                is_call=is_call, expiry=expiry, lot=lots.get(s, 1),
                open_net=open_net, max_risk=max_risk * lots.get(s, 1)))

    s_pnl = pd.Series(pnl).sort_index()
    s_pnl.index = pd.to_datetime(s_pnl.index)
    return s_pnl, pd.DataFrame(trades)


# --------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------

def main():
    print("Sleeve-2 KC6-options — building F&O universe ...", flush=True)
    uni = fno_universe()
    print(f"  {len(uni)} F&O names with daily data", flush=True)
    panel = load_panel(list(uni))
    print(f"  loaded {len(panel)} with >=260 bars + KC6 indicators",
          flush=True)
    uar = universe_atr_ratio_series(panel)

    for expr in ("credit", "debit"):
        for iv in (0.20, 0.25, 0.30):
            pnl, tr = run_sleeve2_kc6(panel, uni, uar, expression=expr, iv=iv)
            if tr.empty:
                print(f"  {expr:6} IV={iv:.2f}: NO trades"); continue
            wins = (tr["net"] > 0).mean() * 100
            tot = tr["net"].sum()
            avg_risk = tr["max_risk"].mean()
            print(f"  {expr:6} IV={iv:.2f}: trades={len(tr):4d} "
                  f"win={wins:4.1f}% totP&L=Rs{tot:,.0f} "
                  f"avgMaxRisk=Rs{avg_risk:,.0f} "
                  f"avg/trade=Rs{tot/len(tr):,.0f}", flush=True)
            if iv == 0.25:
                tr.to_csv(RES / f"sleeve2_{expr}_iv25_trades.csv", index=False)

    print("\n  Note: standalone P&L is RUPEES, pre book-level risk-% cap "
          "(applied in 04). Per-name 'best' pick happens in the combined "
          "engine. Caveat C1: flat-IV BS, indicative not tradeable.")


if __name__ == "__main__":
    main()
