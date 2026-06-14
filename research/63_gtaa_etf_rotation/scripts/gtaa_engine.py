"""
GTAA monthly ETF-rotation backtest engine (research/63).

Faithfully implements the Upstox "Strategy 1" archetype and its improvements:
  - daily ETF closes -> month-end resample
  - relative momentum: ROC(L) score (L months)
  - absolute momentum / trend filter: close > MA(M) (M months) AND ROC>0 (optional)
  - select top-N by score; unfilled slots -> cash (LIQUIDBEES) when cash_leg=True
  - equal-weight within the held sleeve, monthly rebalance
  - explicit per-side cost in bps on traded notional; gross & net both produced

Look-ahead discipline: signals use data <= month-end t; the resulting weights are
applied to returns over t -> t+1 (next month). No future leak.

Pure-stdlib + pandas/numpy. No project deps so it runs anywhere the DB is present.
"""
from __future__ import annotations
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "backtest_data" / "market_data.db"


# ---------------------------------------------------------------- data loading
def load_monthly_closes(symbols, db_path=DB) -> pd.DataFrame:
    """Return a month-end close panel (index=month-end Timestamp, cols=symbols).

    Month-end = last available trading close in each calendar month per symbol.
    """
    con = sqlite3.connect(str(db_path))
    frames = {}
    for s in symbols:
        df = pd.read_sql_query(
            "SELECT date, close FROM market_data_unified "
            "WHERE symbol=? AND timeframe='day' ORDER BY date", con, params=(s,))
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")["close"].astype(float)
        # last close of each month
        m = df.resample("ME").last()
        frames[s] = m
    con.close()
    if not frames:
        raise ValueError("No data loaded for any symbol")
    panel = pd.DataFrame(frames).sort_index()
    return panel


# ---------------------------------------------------------------- config
@dataclass
class GTAAConfig:
    risk_assets: list           # e.g. ["NIFTYBEES","GOLDBEES","MON100"]
    cash_asset: str = "LIQUIDBEES"
    top_n: int = 1
    roc_months: tuple = (12,)   # blended if >1, e.g. (3,6,12)
    ma_months: int = 6          # trend filter window (0 = no trend filter)
    cash_leg: bool = False      # True: unfilled/untrending slots -> cash
    require_pos_roc: bool = True # with cash_leg: also require ROC>0 to hold an asset
    cost_bps: float = 20.0      # per-side cost on traded notional
    trade_next_open: bool = False  # robustness: weights applied with 1-month lag already
    label: str = ""


# ---------------------------------------------------------------- engine
@dataclass
class GTAAResult:
    label: str
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    vol: float
    turnover_annual: float
    n_months: int
    start: str
    end: str
    final_mult: float
    pct_in_cash: float
    yearly: dict = field(default_factory=dict)
    equity: pd.Series = None


def _metrics(net_monthly: pd.Series) -> dict:
    """net_monthly: monthly portfolio returns (decimal)."""
    eq = (1 + net_monthly).cumprod()
    n = len(net_monthly)
    years = n / 12.0
    final = float(eq.iloc[-1])
    cagr = final ** (1 / years) - 1 if years > 0 else np.nan
    mu, sd = net_monthly.mean(), net_monthly.std(ddof=1)
    sharpe = (mu / sd) * np.sqrt(12) if sd > 0 else np.nan
    downside = net_monthly[net_monthly < 0].std(ddof=1)
    sortino = (mu / downside) * np.sqrt(12) if downside and downside > 0 else np.nan
    roll_max = eq.cummax()
    dd = eq / roll_max - 1
    maxdd = float(dd.min())
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    vol = sd * np.sqrt(12)
    return dict(cagr=cagr, sharpe=sharpe, sortino=sortino, max_drawdown=maxdd,
                calmar=calmar, vol=vol, final_mult=final, equity=eq)


def run_gtaa(panel: pd.DataFrame, cfg: GTAAConfig) -> GTAAResult:
    """panel = month-end close panel including risk assets (+cash asset if used)."""
    px = panel.copy()
    # Trim to the COMMON window where every risk asset has data (avoid the
    # single-asset pre-listing artifact that holds NIFTYBEES through 2008).
    common = px[cfg.risk_assets].dropna()
    if common.empty:
        raise ValueError("No common-history window across risk assets")
    px = px.loc[common.index.min():]

    # monthly simple returns for each asset (t-1 -> t)
    rets = px.pct_change()

    # ---- signals at each month-end t (uses closes <= t only) ----
    # ROC over L months = px_t / px_{t-L} - 1; blended = mean across windows
    roc_parts = []
    for L in cfg.roc_months:
        roc_parts.append(px[cfg.risk_assets] / px[cfg.risk_assets].shift(L) - 1)
    roc = sum(roc_parts) / len(roc_parts)

    # MA(M) trend filter on monthly close
    if cfg.ma_months and cfg.ma_months > 0:
        ma = px[cfg.risk_assets].rolling(cfg.ma_months).mean()
        bullish = px[cfg.risk_assets] > ma
    else:
        bullish = pd.DataFrame(True, index=px.index, columns=cfg.risk_assets)

    # warmup: need max(roc lookback, ma window) months of history
    warm = max(max(cfg.roc_months), cfg.ma_months or 0)
    dates = px.index[warm:]

    # cash monthly return series (0 if no cash asset present)
    if cfg.cash_leg and cfg.cash_asset in panel.columns:
        cash_ret = panel[cfg.cash_asset].pct_change()
    else:
        cash_ret = pd.Series(0.0, index=px.index)

    prev_w = pd.Series(0.0, index=cfg.risk_assets + [cfg.cash_asset])
    port_rets = []
    turnovers = []
    cash_frac = []
    idx_used = []

    for i in range(len(dates) - 0):
        t = dates[i]
        # next-month return realized over t -> t+1
        # find position of t in full index
        pos = px.index.get_loc(t)
        if pos + 1 >= len(px.index):
            break  # no forward month to realize
        t1 = px.index[pos + 1]

        score = roc.loc[t].copy()
        elig = score.dropna()
        # apply trend / positive-momentum gate when cash_leg on
        if cfg.cash_leg:
            ok = bullish.loc[t]
            if cfg.require_pos_roc:
                ok = ok & (score > 0)
            elig = elig[ok.reindex(elig.index).fillna(False)]

        chosen = list(elig.sort_values(ascending=False).index[:cfg.top_n])

        # weights: equal among chosen risk assets, remainder to cash
        w = pd.Series(0.0, index=prev_w.index)
        if chosen:
            wgt = 1.0 / cfg.top_n  # baseline divides by N even if fewer chosen -> rest cash
            for c in chosen:
                w[c] = wgt
        filled = w[cfg.risk_assets].sum()
        w[cfg.cash_asset] = max(0.0, 1.0 - filled)
        if not cfg.cash_leg:
            # faithful: no cash sleeve -> if fewer than N eligible, normalize to chosen
            if chosen:
                w[:] = 0.0
                for c in chosen:
                    w[c] = 1.0 / len(chosen)
                w[cfg.cash_asset] = 0.0
            else:
                w[cfg.cash_asset] = 1.0  # nothing to hold -> park (rare, warmup edge)

        # realized return over t->t+1
        r_assets = rets.loc[t1, cfg.risk_assets].fillna(0.0)
        r = float((w[cfg.risk_assets] * r_assets).sum() + w[cfg.cash_asset] * (cash_ret.loc[t1] if not np.isnan(cash_ret.loc[t1]) else 0.0))

        # turnover & cost: |w_t - w_{t-1}| summed /2 *2 sides... use per-side on traded notional
        turn = float((w - prev_w).abs().sum()) / 2.0  # one-way turnover
        cost = turn * 2 * (cfg.cost_bps / 1e4)  # round-trip = buy+sell legs both sides
        r_net = r - cost

        port_rets.append((t1, r_net))
        turnovers.append(turn)
        cash_frac.append(float(w[cfg.cash_asset]))
        idx_used.append(t1)
        prev_w = w

    net = pd.Series(dict(port_rets)).sort_index()
    m = _metrics(net)
    eq = m["equity"]

    # yearly returns
    yearly = {}
    for yr, grp in net.groupby(net.index.year):
        yearly[int(yr)] = float((1 + grp).prod() - 1)

    avg_turn = float(np.mean(turnovers)) if turnovers else 0.0
    return GTAAResult(
        label=cfg.label or "+".join(cfg.risk_assets),
        cagr=m["cagr"], sharpe=m["sharpe"], sortino=m["sortino"],
        max_drawdown=m["max_drawdown"], calmar=m["calmar"], vol=m["vol"],
        turnover_annual=avg_turn * 12, n_months=len(net),
        start=str(net.index[0].date()), end=str(net.index[-1].date()),
        final_mult=m["final_mult"], pct_in_cash=float(np.mean(cash_frac)) if cash_frac else 0.0,
        yearly=yearly, equity=eq,
    )


if __name__ == "__main__":
    # quick self-test once data exists
    core = ["NIFTYBEES", "GOLDBEES", "MON100"]
    panel = load_monthly_closes(core + ["LIQUIDBEES"])
    print("panel range:", panel.index.min(), "->", panel.index.max(), "cols:", list(panel.columns))
    cfg = GTAAConfig(risk_assets=core, top_n=1, roc_months=(12,), ma_months=6,
                     cash_leg=False, cost_bps=20, label="FAITHFUL_top1_roc12_ma6")
    r = run_gtaa(panel, cfg)
    print(f"{r.label}: CAGR={r.cagr*100:.2f}% MaxDD={r.max_drawdown*100:.2f}% "
          f"Calmar={r.calmar:.2f} Sharpe={r.sharpe:.2f} months={r.n_months} "
          f"[{r.start}->{r.end}]")
