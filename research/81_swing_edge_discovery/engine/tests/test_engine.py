"""Unit tests for the canonical engine — hand-computable synthetic cases.
Run: venv/bin/python3 -m engine.tests.test_engine   (from research/81_swing_edge_discovery)
No pytest dependency; plain asserts, exits nonzero on failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.costs import CostConfig                     # noqa: E402
from engine.backtester import BTConfig, run_symbol     # noqa: E402
from engine.metrics import trade_stats, curve_stats, bootstrap_mc  # noqa: E402

PASS = 0


def ok(name, cond):
    global PASS
    assert cond, f"FAIL: {name}"
    PASS += 1
    print(f"  ok: {name}")


def mkbars(rows):
    """rows = list of (ts, o, h, l, c)."""
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])
    return pd.DataFrame({"open": [r[1] for r in rows], "high": [r[2] for r in rows],
                         "low": [r[3] for r in rows], "close": [r[4] for r in rows],
                         "volume": 1000}, index=idx)


NOCOST = CostConfig(product="FUTURES_PROXY", slippage=0.0)


def zero_charge_cfg(time_stop=4):
    cfg = BTConfig(cost=NOCOST, time_stop_sessions=time_stop)
    return cfg


# ---------------------------------------------------------------- cost model

def test_costs():
    print("cost model:")
    c = CostConfig(product="FUTURES_PROXY", slippage=0.0003)
    # hand-computed at price=100, qty=5000 (turnover 5,00,000):
    # buy : brok 20 + exch 8.65 + sebi 0.50 + stamp 10 + gst 5.2470 = 44.397
    # sell: brok 20 + stt 100 + exch 8.65 + sebi 0.50 + gst 5.2470 = 134.397
    ok("futures buy side", abs(c.side_cost(100, 5000, True) - 44.397) < 0.01)
    ok("futures sell side", abs(c.side_cost(100, 5000, False) - 134.397) < 0.01)
    # round trip: 178.794/5,00,000 = 3.576 bps + slippage 2*3 bps = 9.576 bps
    ok("round_trip_bps", abs(c.round_trip_bps(100, 5000) - 9.5759) < 0.01)
    ok("slippage buy up", c.fill_price(100, True) == 100 * 1.0003)
    ok("slippage sell down", c.fill_price(100, False) == 100 * 0.9997)
    c2 = CostConfig(product="FUTURES_PROXY", slippage=0.0003, slippage_mult=2.0)
    ok("2x slippage", c2.fill_price(100, True) == 100 * 1.0006)


# ---------------------------------------------------------------- backtester

def base_bars():
    return mkbars([
        ("2024-01-01 09:15", 100.0, 101.0, 99.5, 100.4),
        ("2024-01-01 09:20", 100.5, 101.5, 100.0, 101.0),   # entry bar (next open)
        ("2024-01-01 09:25", 101.0, 102.0, 100.5, 101.5),
        ("2024-01-02 09:15", 102.0, 103.5, 101.5, 103.0),
        ("2024-01-02 09:20", 103.0, 104.0, 102.5, 103.5),
        ("2024-01-03 09:15", 104.0, 105.0, 103.5, 104.5),
        ("2024-01-03 09:20", 104.5, 105.5, 104.0, 105.0),
        ("2024-01-04 09:15", 105.0, 106.0, 104.5, 105.5),
        ("2024-01-04 09:20", 105.5, 106.5, 105.0, 106.0),
    ])


def sig(ts, d=1, stop=99.0, target=np.nan):
    return pd.DataFrame({"direction": [d], "stop": [stop], "target": [target]},
                        index=pd.DatetimeIndex([pd.Timestamp(ts)]))


def test_entry_next_bar_open():
    print("entry timing:")
    tr = run_symbol(base_bars(), sig("2024-01-01 09:15"), zero_charge_cfg())
    ok("one trade", len(tr) == 1)
    ok("entry at NEXT bar open (100.5)", tr.iloc[0]["entry_px"] == 100.5)
    ok("entry_time is next bar", tr.iloc[0]["entry_time"] == pd.Timestamp("2024-01-01 09:20"))


def test_target_limit_fill():
    print("target fills:")
    tr = run_symbol(base_bars(), sig("2024-01-01 09:15", target=103.2), zero_charge_cfg())
    # first bar with high>=103.2 is 2024-01-02 09:15 (h=103.5, open=102<target)
    ok("target reason", tr.iloc[0]["exit_reason"] == "TARGET")
    ok("limit fill AT target", tr.iloc[0]["exit_px"] == 103.2)
    ok("gross = 103.2/100.5-1", abs(tr.iloc[0]["gross_ret"] - (103.2 / 100.5 - 1)) < 1e-12)
    # gap-through target: next session opens ABOVE the target → fill at OPEN (better)
    gap_bars = mkbars([
        ("2024-01-01 09:15", 100.0, 100.5, 99.5, 100.0),
        ("2024-01-01 09:20", 100.0, 100.4, 99.8, 100.2),   # entry bar, target untouched
        ("2024-01-02 09:15", 102.0, 102.5, 101.5, 102.0),  # gaps over target 101
    ])
    tr2 = run_symbol(gap_bars, sig("2024-01-01 09:15", stop=95.0, target=101.0),
                     zero_charge_cfg(time_stop=2))
    ok("gap target fills at open", tr2.iloc[0]["exit_px"] == 102.0)


def test_stop_and_gap_stop():
    print("stop fills:")
    bars = mkbars([
        ("2024-01-01 09:15", 100.0, 101.0, 99.5, 100.4),
        ("2024-01-01 09:20", 100.5, 101.0, 100.0, 100.2),
        ("2024-01-01 09:25", 100.0, 100.5, 97.8, 98.0),    # low 97.8 < stop 98.5
        ("2024-01-02 09:15", 95.0, 96.0, 94.0, 95.5),      # (gap case below)
    ])
    tr = run_symbol(bars, sig("2024-01-01 09:15", stop=98.5), zero_charge_cfg(time_stop=2))
    ok("stop reason", tr.iloc[0]["exit_reason"] == "STOP")
    ok("intrabar stop fills AT stop", tr.iloc[0]["exit_px"] == 98.5)
    # gap-through: stop 96 not touched intraday on entry day; next open 95 < 96
    tr2 = run_symbol(bars, sig("2024-01-01 09:20", stop=96.0), zero_charge_cfg(time_stop=2))
    ok("gap stop fills at OPEN (worse)", tr2.iloc[0]["exit_px"] == 95.0)


def test_stop_beats_target_same_bar():
    print("same-bar ambiguity:")
    bars = mkbars([
        ("2024-01-01 09:15", 100.0, 101.0, 99.5, 100.4),
        ("2024-01-01 09:20", 100.5, 104.0, 97.0, 100.0),   # touches BOTH
        ("2024-01-01 09:25", 100.0, 100.5, 99.0, 100.0),
    ])
    tr = run_symbol(bars, sig("2024-01-01 09:15", stop=98.0, target=103.0),
                    zero_charge_cfg(time_stop=1))
    ok("stop precedence", tr.iloc[0]["exit_reason"] == "STOP")


def test_time_stop():
    print("time stop:")
    tr = run_symbol(base_bars(), sig("2024-01-01 09:15", stop=90.0),
                    zero_charge_cfg(time_stop=2))
    # entry session = 01-01; last allowed = 01-02; exit at close of last 01-02 bar
    ok("time reason", tr.iloc[0]["exit_reason"] == "TIME")
    ok("exit at close of session+1 last bar", tr.iloc[0]["exit_px"] == 103.5)
    ok("hold_sessions == 2", tr.iloc[0]["hold_sessions"] == 2)
    try:
        BTConfig(cost=NOCOST, time_stop_sessions=5)
        ok("time-stop cap enforced", False)
    except ValueError:
        ok("time-stop cap enforced", True)


def test_short_symmetric():
    print("short side:")
    bars = mkbars([
        ("2024-01-01 09:15", 100.0, 101.0, 99.5, 100.4),
        ("2024-01-01 09:20", 100.5, 101.0, 99.0, 99.5),
        ("2024-01-01 09:25", 99.0, 99.5, 96.9, 97.0),      # low touches target 97
        ("2024-01-02 09:15", 97.0, 98.0, 96.0, 97.5),
    ])
    tr = run_symbol(bars, sig("2024-01-01 09:15", d=-1, stop=103.0, target=97.0),
                    zero_charge_cfg(time_stop=2))
    ok("short target fill", tr.iloc[0]["exit_reason"] == "TARGET"
       and tr.iloc[0]["exit_px"] == 97.0)
    ok("short gross positive", tr.iloc[0]["gross_ret"] > 0.03)


def test_busy_drops_overlap():
    print("concurrency:")
    ent = pd.concat([sig("2024-01-01 09:15", stop=90.0),
                     sig("2024-01-01 09:25", stop=90.0)])
    tr = run_symbol(base_bars(), ent, zero_charge_cfg())
    ok("overlapping signal dropped", len(tr) == 1)


def test_net_costs_flow():
    print("net vs gross:")
    c = CostConfig(product="FUTURES_PROXY", slippage=0.0003)
    tr = run_symbol(base_bars(), sig("2024-01-01 09:15", target=103.2),
                    BTConfig(cost=c, time_stop_sessions=4))
    row = tr.iloc[0]
    ok("entry slipped up", abs(row["entry_px"] - 100.5 * 1.0003) < 1e-9)
    ok("exit slipped down", abs(row["exit_px"] - 103.2 * 0.9997) < 1e-9)
    ok("net < gross", row["net_ret"] < row["gross_ret"])


def test_metrics():
    print("metrics:")
    tr = pd.DataFrame({"net_ret": [0.02, -0.01, 0.03, -0.01, 0.01],
                       "direction": [1, 1, -1, 1, -1],
                       "r_multiple": [2, -1, 3, -1, 1],
                       "hold_sessions": [2, 3, 1, 4, 2]})
    s = trade_stats(tr)
    ok("win rate 3/5", abs(s["win_rate"] - 0.6) < 1e-12)
    ok("PF = 0.06/0.02 = 3", abs(s["profit_factor"] - 3.0) < 1e-12)
    ok("expectancy 0.8%", abs(s["expectancy_pct"] - 0.8) < 1e-12)
    idx = pd.bdate_range("2020-01-01", periods=504)
    eq = pd.Series(np.linspace(1, 1.44, 504), index=idx)   # ~20%/yr linear
    cs = curve_stats(eq)
    ok("cagr ~20%", 0.18 < cs["cagr"] < 0.22)
    mc = bootstrap_mc(np.array([0.01, -0.005] * 50), trades_per_year=100)
    ok("mc keys", "maxdd_p95" in mc and mc["p_loss_3y"] < 0.5)


if __name__ == "__main__":
    for fn in [test_costs, test_entry_next_bar_open, test_target_limit_fill,
               test_stop_and_gap_stop, test_stop_beats_target_same_bar,
               test_time_stop, test_short_symmetric, test_busy_drops_overlap,
               test_net_costs_flow, test_metrics]:
        fn()
    print(f"\nALL {PASS} ASSERTIONS PASSED")
