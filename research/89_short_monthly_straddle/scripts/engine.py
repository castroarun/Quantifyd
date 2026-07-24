#!/usr/bin/env python3
"""
research/89 — Short Monthly Straddle in Predicted Calm Regimes.

Core engine: DB loader, stdlib Black-Scholes pricer, low-vol regime detector,
synthetic-IV model (Tier C stocks) / real INDIAVIX IV (Tier A index), and a
daily-mark straddle / iron-fly simulator with an exit-policy grid.

Runs on the VPS (canonical DB host). DB is read-only; nothing is written to it.
"""
import os, math, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- DB ----------
def db_path():
    # script at .../research/89_.../scripts/engine.py ; DB at <quantifyd>/backtest_data/
    here = Path(__file__).resolve()
    for up in [here.parents[3] / "backtest_data" / "market_data.db",
               Path("/home/arun/quantifyd/backtest_data/market_data.db")]:
        if up.exists():
            return str(up)
    raise FileNotFoundError("market_data.db not found (expected on VPS)")

def load_daily(symbol, conn):
    q = ("SELECT date, open, high, low, close, volume FROM market_data_unified "
         "WHERE symbol=? AND timeframe='day' ORDER BY date")
    df = pd.read_sql_query(q, conn, params=(symbol,))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"].astype(str).str.slice(0, 10), format="%Y-%m-%d")
    df = df.drop_duplicates("date").set_index("date")
    df = df[df["close"] > 0]
    return df

# ---------------------------------------------------- Black-Scholes (stdlib) ---
R = 0.065  # risk-free (annualized)

def _N(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs(S, K, T, sigma, kind):
    """European option price. T in years, sigma annualized. kind 'C'/'P'."""
    if T <= 1e-9 or sigma <= 1e-9:
        return max(0.0, (S - K) if kind == "C" else (K - S))
    srt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * sigma * sigma) * T) / srt
    d2 = d1 - srt
    if kind == "C":
        return S * _N(d1) - K * math.exp(-R * T) * _N(d2)
    return K * math.exp(-R * T) * _N(-d2) - S * _N(-d1)

def straddle_value(S, K, T, sigma):
    return bs(S, K, T, sigma, "C") + bs(S, K, T, sigma, "P")

# ------------------------------------------------- realized vol & features ----
TRADING_DAYS = 252

def annualized_rv(close, n=20):
    """Trailing n-day annualized realized vol from log returns (causal)."""
    lr = np.log(close / close.shift(1))
    return lr.rolling(n).std() * math.sqrt(TRADING_DAYS)

def add_features(df, rv_n=20, rank_win=252):
    """Attach causal detector features. Winsorize corporate-action-suspect gaps."""
    out = df.copy()
    lr = np.log(out["close"] / out["close"].shift(1))
    # winsorize |overnight gap| > 25% (unadjusted splits / circuits) for RV only
    lr_clean = lr.clip(lower=math.log(0.75), upper=math.log(1.25))
    out["rv20"] = lr_clean.rolling(rv_n).std() * math.sqrt(TRADING_DAYS)
    # causal percentile rank of rv20 within trailing rank_win days (0=calmest)
    out["rv_rank"] = out["rv20"].rolling(rank_win).apply(
        lambda w: (w[:-1] < w[-1]).mean() if np.isfinite(w[-1]) else np.nan, raw=True)
    # supporting features
    sma20 = out["close"].rolling(20).mean()
    out["trend_flat"] = (out["close"] / sma20 - 1.0).abs()
    tr = pd.concat([(out["high"] - out["low"]),
                    (out["high"] - out["close"].shift(1)).abs(),
                    (out["low"] - out["close"].shift(1)).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14).mean() / out["close"]
    bb = out["close"].rolling(20).std()
    out["bbw"] = (4 * bb) / sma20
    out["bbw_rank"] = out["bbw"].rolling(rank_win).apply(
        lambda w: (w[:-1] < w[-1]).mean() if np.isfinite(w[-1]) else np.nan, raw=True)
    out["gap"] = lr  # raw, to flag suspect entries
    return out

# ------------------------------------------------------------ IV model --------
def stock_iv_series(rv20, vrp_mult=1.15, floor=0.08):
    """Modeled IV for a stock: realized vol x volatility-risk-premium multiplier."""
    return (rv20 * vrp_mult).clip(lower=floor)

# ------------------------------------------------- one straddle trade ----------
def simulate_trade(sub, iv_path, H=20, structure="naked", wing_mult=1.0,
                   cost_frac=0.0017, exit_policy=("expiry",), r=R):
    """
    Simulate ONE short straddle entered at sub.index[0], held up to H trading days.
    sub      : DataFrame rows [entry .. entry+H] with 'close'.
    iv_path  : aligned Series of IV per day over the same window.
    Returns dict with pnl (in underlying points, per 1 lot notional=1 unit), diag.
    exit_policy: tuple describing exit, e.g.
        ("expiry",) | ("dte", d) | ("target", frac) | ("stop_prem", mult)
        | ("stop_move", mult) | ("volstop",)  -- combinable via check order.
    """
    if len(sub) < 2:
        return None
    S0 = float(sub["close"].iloc[0])
    K = S0  # ATM (continuous strike; real strike-rounding handled at rupee stage)
    T0 = H / TRADING_DAYS
    iv0 = float(iv_path.iloc[0])
    if not np.isfinite(iv0) or iv0 <= 0:
        return None
    prem0 = straddle_value(S0, K, T0, iv0)          # credit received (short)
    if prem0 <= 0:
        return None
    wing = wing_mult * prem0
    # wings bought (long) for iron fly — priced at entry, held to same exit
    wing_cost0 = 0.0
    if structure == "ironfly":
        Kc, Kp = K + wing, K - wing
        wing_cost0 = bs(S0, Kc, T0, iv0, "C") + bs(S0, Kp, T0, iv0, "P")
    breakeven = prem0  # move beyond +/- prem0 => straddle loses at expiry
    n = len(sub) - 1
    exit_i = n
    exit_reason = "expiry"
    entry_rank = None
    for i in range(1, len(sub)):
        S = float(sub["close"].iloc[i])
        Trem = max((H - i) / TRADING_DAYS, 0.0)
        iv = float(iv_path.iloc[i])
        if not np.isfinite(iv) or iv <= 0:
            iv = iv0
        cur = straddle_value(S, K, Trem, iv)
        move = abs(S - S0)
        # short straddle MTM pnl (before wings/costs): credit - cost-to-close
        mtm = prem0 - cur
        kind = exit_policy[0]
        if kind == "dte" and (H - i) <= exit_policy[1]:
            exit_i, exit_reason = i, "dte"; break
        if kind == "target" and mtm >= exit_policy[1] * prem0:
            exit_i, exit_reason = i, "target"; break
        if kind == "stop_prem" and (-mtm) >= exit_policy[1] * prem0:
            exit_i, exit_reason = i, "stop_prem"; break
        if kind == "stop_move" and move >= exit_policy[1] * breakeven:
            exit_i, exit_reason = i, "stop_move"; break
    # settle at exit_i
    Sx = float(sub["close"].iloc[exit_i])
    Tx = max((H - exit_i) / TRADING_DAYS, 0.0)
    ivx = float(iv_path.iloc[exit_i])
    if not np.isfinite(ivx) or ivx <= 0:
        ivx = iv0
    close_cost = straddle_value(Sx, K, Tx, ivx)
    pnl_short = prem0 - close_cost                    # short straddle leg
    pnl = pnl_short
    wing_val = 0.0
    if structure == "ironfly":
        Kc, Kp = K + wing, K - wing
        wing_val = bs(Sx, Kc, Tx, ivx, "C") + bs(Sx, Kp, Tx, ivx, "P")
        pnl = pnl_short + (wing_val - wing_cost0)     # long wings pnl (buy low, sell/settle)
    # transaction cost: charged on premium turned over (entry + exit), both legs
    turnover = prem0 + close_cost + wing_cost0 + wing_val
    cost = cost_frac * turnover
    pnl_net = pnl - cost
    fwd_move_pct = (Sx - S0) / S0
    return dict(
        S0=S0, prem0=prem0, iv0=iv0, wing_cost0=wing_cost0,
        pnl_gross=pnl, pnl_net=pnl_net, cost=cost,
        pnl_gross_pct=pnl / S0, pnl_net_pct=pnl_net / S0,
        prem0_pct=prem0 / S0, held_days=exit_i, exit_reason=exit_reason,
        fwd_move_pct=fwd_move_pct, win=1 if pnl_net > 0 else 0,
    )

# ------------------------------------------------- forward realized outcome ----
def forward_rv(close, i, H):
    """Realized annualized vol over the FORWARD window [i, i+H] (for persistence)."""
    seg = close.iloc[i:i + H + 1]
    if len(seg) < 3:
        return np.nan
    lr = np.log(seg / seg.shift(1)).dropna()
    return lr.std() * math.sqrt(TRADING_DAYS)


DEFAULT_STOCKS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "SBIN", "ITC",
    "HINDUNILVR", "KOTAKBANK", "BHARTIARTL", "LT", "AXISBANK", "MARUTI",
    "BAJFINANCE", "HCLTECH", "ASIANPAINT", "TITAN", "SUNPHARMA", "ULTRACEMCO",
    "TATAMOTORS", "TATASTEEL", "M&M", "NTPC", "POWERGRID", "WIPRO", "NESTLEIND",
]
TIER_A = ["NIFTY50", "BANKNIFTY"]
