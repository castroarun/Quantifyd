"""V2 iron-fly + inside-week breakout — pure signal layer (no Kite, unit-testable).

All features are CAUSAL (known at 09:20 on the decision date): prior completed day /
prior completed week only. The executor injects live spot/expiry and calls the leg
builders. Smoke-test (`python services/v2_breakout_signals.py`) runs the signals
against backtest_data/market_data.db (NIFTY50 daily) for the latest date + a historical
inside-week example.

Findings encoded (research/61):
  - V2 short fly SKIPS entry when prior-day CPR < 0.10% OR last completed week was inside.
  - Inside-week breakout sleeve: UP-break -> call debit spread (runner edge, 78%);
    DOWN-break -> broken-wing iron fly skewed down (no runner edge, premium + capped risk).
"""
from __future__ import annotations
import pandas as pd

CPR_SKIP_PCT = 0.10          # skip V2 entry if prior-day CPR width < this (% of spot)
WING = 500                   # V2 symmetric fly wing (pts)
DEBIT_WIDTH = 500            # breakout call/put debit spread width (pts)
BWF_NEAR = 300               # broken-wing fly near (tight) wing
BWF_FAR = 700                # broken-wing fly far wing
STEP = 50                    # NIFTY strike interval

# default per-side structure for the breakout sleeve (override to 'bwfly'/'debit' as desired)
BREAKOUT_STRUCT = {"up": "debit", "down": "bwfly"}


# ---------- causal features ----------
def _weekly(daily: pd.DataFrame) -> pd.DataFrame:
    w = daily.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    w["inside"] = (w.high < w.high.shift(1)) & (w.low > w.low.shift(1))
    return w


def prior_day_cpr_width_pct(daily: pd.DataFrame, on: pd.Timestamp) -> float | None:
    """CPR width of the day BEFORE `on`, as % of `on`'s open (spot proxy)."""
    on = pd.Timestamp(on)
    if on not in daily.index:
        return None
    pos = daily.index.get_loc(on)
    if pos < 1:
        return None
    p = daily.iloc[pos - 1]
    piv = (p.high + p.low + p.close) / 3
    bc = (p.high + p.low) / 2
    tc = 2 * piv - bc
    return abs(tc - bc) / daily.iloc[pos].open * 100


def last_completed_week(daily: pd.DataFrame, on: pd.Timestamp):
    """Return (row, is_inside) for the last weekly bar strictly before `on`'s week."""
    on = pd.Timestamp(on)
    w = _weekly(daily)
    monday = on - pd.Timedelta(days=on.weekday())
    comp = w[w.index < monday]
    if len(comp) < 2:
        return None, None
    return comp.iloc[-1], bool(comp["inside"].iloc[-1])


def combo_skip(daily: pd.DataFrame, on: pd.Timestamp):
    """V2 entry skip decision. Returns (skip: bool, reasons: list[str])."""
    reasons = []
    cprw = prior_day_cpr_width_pct(daily, on)
    if cprw is not None and cprw < CPR_SKIP_PCT:
        reasons.append(f"narrow_cpr({cprw:.3f}%<{CPR_SKIP_PCT}%)")
    _, inside = last_completed_week(daily, on)
    if inside:
        reasons.append("inside_week")
    return (len(reasons) > 0), reasons


def breakout_state(daily: pd.DataFrame, on: pd.Timestamp):
    """If the last completed week was inside, scan the trade week's daily CLOSES up to
    `on` for the FIRST close beyond that (inside) week's range.
    Returns dict(active, direction in {'up','down',None}, ref_high, ref_low, broke_on)."""
    on = pd.Timestamp(on)
    wk_row, inside = last_completed_week(daily, on)
    if not inside:
        return {"active": False, "direction": None}
    ref_h, ref_l = wk_row.high, wk_row.low
    monday = on - pd.Timedelta(days=on.weekday())
    wkdays = daily[(daily.index >= monday) & (daily.index <= on)]
    direction, broke_on = None, None
    for dt, row in wkdays.iterrows():
        if row.close > ref_h:
            direction, broke_on = "up", dt; break
        if row.close < ref_l:
            direction, broke_on = "down", dt; break
    return {"active": True, "direction": direction, "ref_high": float(ref_h),
            "ref_low": float(ref_l), "broke_on": (str(broke_on.date()) if broke_on is not None else None)}


# ---------- leg builders (executor resolves strike->tradingsymbol & places) ----------
def atm_strike(spot: float, step: int = STEP) -> int:
    return int(round(spot / step) * step)


def _leg(side, itype, strike, role):
    return {"side": side, "instrument_type": itype, "strike": int(strike), "role": role}


def v2_fly_legs(spot: float) -> list[dict]:
    """Symmetric short iron fly (the V2 base book)."""
    k = atm_strike(spot)
    return [
        _leg("SELL", "CE", k, "body"), _leg("SELL", "PE", k, "body"),
        _leg("BUY", "CE", k + WING, "wing"), _leg("BUY", "PE", k - WING, "wing"),
    ]


def breakout_legs(spot: float, direction: str, struct: dict = None) -> list[dict]:
    """Inside-week breakout sleeve legs for the given break direction."""
    struct = struct or BREAKOUT_STRUCT
    k = atm_strike(spot)
    mode = struct.get(direction)
    if direction == "up":
        if mode == "debit":                       # call debit spread — capture the runner
            return [_leg("BUY", "CE", k, "long"), _leg("SELL", "CE", k + DEBIT_WIDTH, "short")]
        # bullish broken-wing fly: near CALL wing (tight up cover), far PUT wing
        return [_leg("SELL", "CE", k, "body"), _leg("SELL", "PE", k, "body"),
                _leg("BUY", "CE", k + BWF_NEAR, "near_wing"), _leg("BUY", "PE", k - BWF_FAR, "far_wing")]
    if direction == "down":
        if mode == "debit":                       # put debit spread
            return [_leg("BUY", "PE", k, "long"), _leg("SELL", "PE", k - DEBIT_WIDTH, "short")]
        # bearish broken-wing fly: near PUT wing (tight down cover), far CALL wing
        return [_leg("SELL", "CE", k, "body"), _leg("SELL", "PE", k, "body"),
                _leg("BUY", "PE", k - BWF_NEAR, "near_wing"), _leg("BUY", "CE", k + BWF_FAR, "far_wing")]
    return []


# ---------- smoke test ----------
if __name__ == "__main__":
    import sqlite3
    from pathlib import Path
    db = Path(__file__).resolve().parent.parent / "backtest_data" / "market_data.db"
    con = sqlite3.connect(str(db))
    rows = con.execute(
        "SELECT date, open, high, low, close FROM market_data_unified "
        "WHERE symbol='NIFTY50' AND timeframe='day' ORDER BY date").fetchall()
    con.close()
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["date"].str.slice(0, 10)); df = df.set_index("date").sort_index()
    print(f"Loaded {len(df)} NIFTY50 daily bars, {df.index[0].date()} -> {df.index[-1].date()}")

    def show(on):
        on = pd.Timestamp(on)
        cprw = prior_day_cpr_width_pct(df, on)
        skip, why = combo_skip(df, on)
        bo = breakout_state(df, on)
        spot = df.loc[on].open if on in df.index else df.iloc[-1].close
        print(f"\n=== {on.date()} (spot~{spot:.0f}) ===")
        print(f"  prior-day CPR width = {cprw:.3f}% " if cprw is not None else "  prior-day CPR n/a")
        print(f"  V2 entry: {'SKIP' if skip else 'TAKE'}  reasons={why or '[]'}")
        if not skip:
            print(f"    -> fly legs: {[(l['side'],l['instrument_type'],l['strike']) for l in v2_fly_legs(spot)]}")
        print(f"  breakout sleeve: active={bo['active']} dir={bo.get('direction')}")
        if bo.get("direction"):
            print(f"    ref H/L={bo.get('ref_high'):.0f}/{bo.get('ref_low'):.0f} broke_on={bo.get('broke_on')}")
            print(f"    -> legs: {[(l['side'],l['instrument_type'],l['strike'],l['role']) for l in breakout_legs(spot, bo['direction'])]}")

    show(df.index[-1])                 # latest date
    # scan recent history for an inside-week example to exercise the breakout path
    found = 0
    for dt in reversed(df.index[-260:]):
        _, ins = last_completed_week(df, dt)
        if ins:
            show(dt); found += 1
        if found >= 2:
            break
    print("\nsmoke test OK")
