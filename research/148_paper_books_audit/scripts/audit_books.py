"""research/148 — honest audit of the N500M and I75WR green paper books (read-only)."""
import sqlite3, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/arun/quantifyd")
RES = Path(__file__).resolve().parents[1] / "results"; RES.mkdir(exist_ok=True)


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


# ── N500M ──
c = sqlite3.connect(str(ROOT / "backtest_data/n500m_trading.db"))
df = pd.read_sql("SELECT * FROM n500m_positions WHERE status='CLOSED' ORDER BY entry_time", c)
df["notional"] = df.entry_price * df.qty
df["ret"] = df.pnl_inr / df.notional * 100          # % per trade, gross of costs
n = len(df)
print(f"N500M: n={n}  pnl=Rs{df.pnl_inr.sum():,.0f}  WR={(df.pnl_inr>0).mean()*100:.1f}%")
for bps in (0, 5, 10, 15):
    r = df.ret - bps / 100.0
    t = r.mean() / (r.std(ddof=1) / math.sqrt(n))
    lo, hi = r.mean() - 1.96 * r.std(ddof=1) / math.sqrt(n), r.mean() + 1.96 * r.std(ddof=1) / math.sqrt(n)
    print(f"  cost {bps:>2}bps RT: mean {r.mean():+.3f}%/tr  t={t:.2f}  95%CI [{lo:+.3f}, {hi:+.3f}]"
          f"  pnl Rs{(r/100*df.notional).sum():,.0f}")
w = (df.pnl_inr > 0).mean()
print(f"  WR 95% Wilson CI: [{wilson(w,n)[0]*100:.0f}%, {wilson(w,n)[1]*100:.0f}%]")
print("  avg win %+.2f%% / avg loss %+.2f%%  max lose streak %d" % (
    df.ret[df.ret > 0].mean(), df.ret[df.ret <= 0].mean(),
    max((len(list(g)) for k, g in __import__('itertools').groupby(df.pnl_inr <= 0) if k), default=0)))
print("  per-symbol pnl:"); print(df.groupby("symbol").pnl_inr.agg(["count", "sum"]).sort_values("sum", ascending=False).to_string())
top = df.groupby("symbol").pnl_inr.sum().max()
print(f"  top symbol share of net: {top/df.pnl_inr.sum()*100:.0f}%")
df["month"] = df.trade_date.str[:7]
print("  per-month:"); print(df.groupby("month").pnl_inr.agg(["count", "sum"]).round(0).to_string())
print("  exit reasons:", df.exit_reason.value_counts().to_dict())

# shrinkage: promotion-time expectation vs live realized (per rule with >=3 trades)
try:
    import sys
    sys.path.insert(0, str(ROOT))
    from services.n500m_configs import load_all_configs
    cfgs = {(cc.symbol, cc.signal): cc for cc in load_all_configs()}
    rows = []
    for (sym, sig), g in df.groupby(["symbol", "signal_type"]):
        cc = cfgs.get((sym, sig))
        if cc is None:
            continue
        rows.append(dict(symbol=sym, signal=sig, n_live=len(g),
                         exp_mean_pct=round(cc.expected_mean_pct, 3),
                         live_mean_pct=round(g.ret.mean(), 3),
                         exp_sharpe=round(cc.expected_sharpe, 2),
                         exp_wr=round(cc.expected_wr, 3), live_wr=round((g.pnl_inr > 0).mean(), 3)))
    sh = pd.DataFrame(rows).sort_values("n_live", ascending=False)
    sh.to_csv(RES / "n500m_audit.csv", index=False)
    print("\n  SHRINKAGE (promotion-time backtest vs live), rules with live trades:")
    print(sh.to_string(index=False))
    m = sh[sh.n_live >= 3]
    if len(m):
        print(f"  weighted: expected {np.average(m.exp_mean_pct, weights=m.n_live):+.3f}%/tr"
              f" -> live {np.average(m.live_mean_pct, weights=m.n_live):+.3f}%/tr")
except Exception as e:
    print("shrinkage table failed:", e)

# ── I75WR ──
c2 = sqlite3.connect(str(ROOT / "backtest_data/intraday_75wr.db"))
d2 = pd.read_sql("SELECT * FROM i75_positions WHERE status='CLOSED' ORDER BY entry_time", c2)
d2["notional"] = d2.entry_price * d2.qty
d2["ret"] = d2.pnl_inr / d2.notional * 100
d2.to_csv(RES / "i75wr_trades.csv", index=False)
print(f"\nI75WR: n={len(d2)}  pnl=Rs{d2.pnl_inr.sum():,.0f}  WR={(d2.pnl_inr>0).mean()*100:.0f}%"
      f"  mean {d2.ret.mean():+.3f}%/tr  by system: {d2.groupby('system_id').pnl_inr.sum().to_dict()}"
      f"  symbols: {d2.instrument.unique().tolist()}")
print("VERDICT INPUTS DONE")
