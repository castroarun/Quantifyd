"""Nifty 500 Intraday Momentum Portfolio — per-stock STOCK_CONFIGS.

Reads research/34/results/{ccrb_leaders.csv, volbo_leaders.csv} at load time
and exposes a normalized list of per-stock signal configs ranked by Sharpe.

Self-updating: when Phase F/G aggregations re-write the leader CSVs, the next
process restart picks up the expanded universe automatically — no code edit
needed.

Each config is one (symbol, signal_type, variant_params, exit_policy) rule.
A symbol may appear in BOTH lists (CCRB + vol-BO) → both rules eval per day,
either fires.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "research" / "34_nifty500_expansion" / "results"
CCRB_CSV = RESULTS_DIR / "ccrb_leaders.csv"
VOLBO_CSV = RESULTS_DIR / "volbo_leaders.csv"

DEFAULT_TOP_N = 15
DEFAULT_MIN_SHARPE = 0.5
DEFAULT_MIN_N = 10


@dataclass
class StockConfig:
    symbol: str
    signal: str          # 'ccrb' or 'volbo'
    timeframe: str       # '5min' / '10min' / '15min'
    direction: str       # 'long' / 'short'
    variant_raw: str     # full variant string from CSV (audit trail)
    params: dict         # parsed numeric params (today_narrow, vm, etc.)
    exit_policy: str     # 'T_NO' / 'T_HARD_SL' / 'T_ATR_SL_*' / 'T_R_TARGET_*' / 'T_STEP_TRAIL' / etc.
    expected_sharpe: float
    expected_wr: float
    expected_n: int
    expected_mean_pct: float
    promote: bool        # passes strict gate (Sharpe>=0.5 + n>=15 + 3+ robust)
    cohort: str          # 'A' (since 2018) or 'B' (since 2024-03-18)

    def to_dict(self):
        return asdict(self)


def _parse_ccrb_variant(v: str) -> dict:
    """Parse 't0.0030_ctxW_OR_N_w0.0080_n0.0090_15min_off_s'."""
    out: dict = {}
    m = re.search(r"t([0-9.]+)", v);            out["today_narrow"] = float(m.group(1)) if m else None
    if "ctxW_OR_N" in v:        out["yesterday_ctx"] = "W_OR_N"
    elif "ctxW_AND_N" in v:     out["yesterday_ctx"] = "W_AND_N"
    elif "ctxW" in v:           out["yesterday_ctx"] = "W"
    elif "ctxN" in v:           out["yesterday_ctx"] = "N"
    else:                       out["yesterday_ctx"] = None
    m = re.search(r"_w([0-9.]+)", v);           out["y_wide"] = float(m.group(1)) if m else None
    m = re.search(r"_n([0-9.]+)_", v);          out["y_narrow"] = float(m.group(1)) if m else None
    if "_vm1.5_" in v:          out["vol_mode"] = "vm1.5"
    elif "_vm2.0_" in v:        out["vol_mode"] = "vm2.0"
    elif "_off_" in v:          out["vol_mode"] = "off"
    else:                       out["vol_mode"] = None
    return out


def _parse_volbo_variant(v: str) -> dict:
    """Parse 's_vm1.5_gap0.005_rsi40_60'."""
    out: dict = {}
    m = re.search(r"vm([0-9.]+)", v);           out["vm"] = float(m.group(1)) if m else None
    if "gap-off" in v:          out["gap_mode"] = "off"
    else:
        m = re.search(r"gap([0-9.]+)", v);      out["gap_pct"] = float(m.group(1)) if m else None
    if "rsi-off" in v:          out["rsi_mode"] = "off"
    else:
        m = re.search(r"rsi(\d+)_(\d+)", v)
        if m:
            out["rsi_short_thresh"] = int(m.group(1))
            out["rsi_long_thresh"] = int(m.group(2))
    return out


def _load_csv(path: Path, signal: str) -> list[StockConfig]:
    if not path.exists():
        return []
    out: list[StockConfig] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                sharpe = float(r["best_sharpe"])
                n = int(float(r["best_n"]))
                wr_key = "best_win_rate" if "best_win_rate" in r else "best_wr"
                wr = float(r[wr_key])
                mean = float(r.get("best_mean_pct", 0))
                # ccrb mean is in % already (e.g., 0.5565); volbo mean is in
                # percent units already too (e.g., 1.4088). Both stored as %.
                variant = r["best_variant"]
                if signal == "ccrb":
                    params = _parse_ccrb_variant(variant)
                else:
                    params = _parse_volbo_variant(variant)
                out.append(StockConfig(
                    symbol=r["symbol"],
                    signal=signal,
                    timeframe=r["best_timeframe"],
                    direction=r["best_direction"],
                    variant_raw=variant,
                    params=params,
                    exit_policy=r["best_exit_policy"],
                    expected_sharpe=sharpe,
                    expected_wr=wr,
                    expected_n=n,
                    expected_mean_pct=mean,
                    promote=str(r.get("promote", "False")).lower() == "true",
                    cohort=r["cohort"],
                ))
            except (KeyError, ValueError) as e:
                # Skip malformed rows silently
                continue
    return out


def load_all_configs(
    top_n: Optional[int] = DEFAULT_TOP_N,
    min_sharpe: float = DEFAULT_MIN_SHARPE,
    min_n: int = DEFAULT_MIN_N,
    promote_only: bool = False,
) -> list[StockConfig]:
    """Load both CCRB + vol-BO leaders, filter, sort by Sharpe.

    top_n: if set, take only the top N from EACH signal (so up to 2*top_n
           total rules, with overlapping symbols counted twice).
    min_sharpe: cell must clear this Sharpe gate.
    min_n: cell must have this many historical signals.
    promote_only: if True, restrict to strict-gate passers only.
    """
    ccrb = _load_csv(CCRB_CSV, "ccrb")
    volbo = _load_csv(VOLBO_CSV, "volbo")

    def _filter(rows: list[StockConfig]) -> list[StockConfig]:
        out = [c for c in rows
               if c.expected_sharpe >= min_sharpe
               and c.expected_n >= min_n
               and c.expected_mean_pct > 0]
        if promote_only:
            out = [c for c in out if c.promote]
        out.sort(key=lambda c: c.expected_sharpe, reverse=True)
        if top_n:
            out = out[:top_n]
        return out

    return _filter(ccrb) + _filter(volbo)


def stocks_to_watch(top_n: Optional[int] = DEFAULT_TOP_N,
                    min_sharpe: float = DEFAULT_MIN_SHARPE,
                    min_n: int = DEFAULT_MIN_N) -> list[str]:
    """Distinct symbol list (a stock with both CCRB + vol-BO rules listed once)."""
    return sorted({c.symbol for c in load_all_configs(top_n, min_sharpe, min_n)})


if __name__ == "__main__":
    cfgs = load_all_configs()
    print(f"Loaded {len(cfgs)} configs across {len({c.symbol for c in cfgs})} unique stocks")
    print()
    print(f"{'Symbol':<14} {'Signal':<6} {'TF':<6} {'Dir':<6} "
          f"{'Sharpe':>7} {'WR%':>6} {'n':>4} {'Promote':>8}  Exit")
    print("-" * 90)
    for c in sorted(cfgs, key=lambda x: x.expected_sharpe, reverse=True):
        print(f"{c.symbol:<14} {c.signal:<6} {c.timeframe:<6} {c.direction:<6} "
              f"{c.expected_sharpe:>7.3f} {c.expected_wr:>6.1f} {c.expected_n:>4} "
              f"{'YES' if c.promote else '-':>8}  {c.exit_policy}")
