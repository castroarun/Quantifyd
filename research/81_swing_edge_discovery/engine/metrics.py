"""Canonical metrics for research/81 — every number every report shows comes
from here (brief sections 4 and 6). Fixed seed for reproducibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 81
TRADING_DAYS = 252


# ---------------------------------------------------------------- trade level

def trade_stats(trades: pd.DataFrame, ret_col: str = "net_ret") -> dict:
    """Per-trade stats. Expectancy is mean % return per trade (and R if present)."""
    if trades is None or len(trades) == 0:
        return {"n": 0}
    r = trades[ret_col].to_numpy(float)
    wins, losses = r[r > 0], r[r <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 and r.std() > 0 else np.nan
    return {
        "n": len(r),
        "win_rate": len(wins) / len(r),
        "profit_factor": pf,
        "expectancy_pct": r.mean() * 100,
        "avg_win_pct": wins.mean() * 100 if len(wins) else 0.0,
        "avg_loss_pct": losses.mean() * 100 if len(losses) else 0.0,
        "avg_R": trades["r_multiple"].mean() if "r_multiple" in trades else np.nan,
        "avg_hold_sessions": trades["hold_sessions"].mean() if "hold_sessions" in trades else np.nan,
        "t_stat": t,
        "long_n": int((trades["direction"] == 1).sum()) if "direction" in trades else None,
        "short_n": int((trades["direction"] == -1).sum()) if "direction" in trades else None,
    }


# ---------------------------------------------------------------- curve level

def equity_from_daily_returns(daily_ret: pd.Series, start: float = 1.0) -> pd.Series:
    return start * (1 + daily_ret.fillna(0)).cumprod()


def curve_stats(equity: pd.Series) -> dict:
    """CAGR / Sharpe / Sortino / MaxDD(depth+duration) / Calmar from a daily NAV."""
    if equity is None or len(equity) < 3:
        return {}
    eq = equity.dropna()
    ret = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    sharpe = ret.mean() / ret.std() * np.sqrt(TRADING_DAYS) if ret.std() > 0 else np.nan
    dn = ret[ret < 0]
    sortino = (ret.mean() / dn.std() * np.sqrt(TRADING_DAYS)) if len(dn) and dn.std() > 0 else np.nan
    peak = eq.cummax()
    dd = eq / peak - 1
    maxdd = dd.min()
    # longest peak-to-recovery duration in days
    under = dd < 0
    longest = cur = 0
    prev_t = None
    for t, u in under.items():
        if u:
            cur += (t - prev_t).days if prev_t is not None and cur > 0 else 1
            longest = max(longest, cur)
        else:
            cur = 0
        prev_t = t
    return {
        "cagr": cagr, "sharpe": sharpe, "sortino": sortino,
        "max_dd": maxdd, "calmar": (cagr / abs(maxdd)) if maxdd else np.nan,
        "dd_duration_days": longest,
        "years": yrs,
    }


def per_year_table(daily_ret: pd.Series) -> pd.DataFrame:
    g = (1 + daily_ret.fillna(0)).groupby(daily_ret.index.year).prod() - 1
    return g.to_frame("return")


# ---------------------------------------------------------------- monte carlo

def bootstrap_mc(trade_rets: np.ndarray, trades_per_year: float,
                 n_iter: int = 2000, horizon_years: float = 3.0,
                 seed: int = SEED) -> dict:
    """Trade-order bootstrap → distribution of CAGR and max drawdown.
    Returns the 95th-pct drawdown the brief gates on (<= 30%)."""
    if len(trade_rets) < 20:
        return {"insufficient": True}
    rng = np.random.default_rng(seed)
    n_tr = max(int(trades_per_year * horizon_years), 10)
    cagrs, maxdds = np.empty(n_iter), np.empty(n_iter)
    for k in range(n_iter):
        sample = rng.choice(trade_rets, size=n_tr, replace=True)
        eq = np.cumprod(1 + sample)
        peak = np.maximum.accumulate(eq)
        maxdds[k] = (eq / peak - 1).min()
        cagrs[k] = eq[-1] ** (1 / horizon_years) - 1
    return {
        "cagr_median": float(np.median(cagrs)),
        "cagr_p5": float(np.percentile(cagrs, 5)),
        "maxdd_median": float(np.median(maxdds)),
        "maxdd_p95": float(np.percentile(maxdds, 5)),   # 5th pct value = worst-tail dd
        "p_loss_3y": float((cagrs < 0).mean()),
    }


def edge_vs_random_p(trade_rets: np.ndarray, null_rets: np.ndarray,
                     n_iter: int = 5000, seed: int = SEED) -> float:
    """Bootstrap p-value: P(mean of null sample >= observed mean).
    null_rets = returns of random entries with the SAME exit policy."""
    if len(trade_rets) == 0 or len(null_rets) == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    obs = trade_rets.mean()
    n = len(trade_rets)
    means = np.array([rng.choice(null_rets, size=n, replace=True).mean()
                      for _ in range(n_iter)])
    return float((means >= obs).mean())


def wf_efficiency(oos_metric: float, is_metric: float) -> float:
    if is_metric is None or is_metric <= 0 or oos_metric is None:
        return np.nan
    return oos_metric / is_metric
