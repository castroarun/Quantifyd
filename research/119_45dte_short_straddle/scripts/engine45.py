#!/usr/bin/env python3
"""
research/119 — NIFTY 45-DTE short straddle (replication of "The Long & The Short Ep. 48").

Core engine. Read-only against backtest_data/market_data.db on the VPS.

Ground truth = REAL NSE bhavcopy option prices (nse_options_bhav).
Intraday premium paths are RECONSTRUCTED (Black-76 on real 5-min NIFTY spot, with the
forward and IV backed out of real option closes) because no intraday option history
exists before 2026-04-20. See the STATUS doc, section 3.
"""
import math
import sqlite3
from bisect import bisect_left
from datetime import datetime, timedelta
from pathlib import Path

R = 0.065          # risk-free, annualised
LOT = 65           # NIFTY contract size (confirmed from Kite live instrument master, 2026-08-20)
LOTS = 10          # position size Arun asked for
QTY = LOT * LOTS   # 650 -> 1 point = Rs 650


# ------------------------------------------------------------------ DB --------
def db_path():
    for p in [Path("/home/arun/quantifyd/backtest_data/market_data.db"),
              Path(__file__).resolve().parents[3] / "backtest_data" / "market_data.db"]:
        if p.exists():
            return str(p)
    raise FileNotFoundError("market_data.db not found")


def connect():
    return sqlite3.connect("file:%s?mode=ro" % db_path(), uri=True)


# --------------------------------------------------------- Black-76 ----------
def _N(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def straddle_b76(F, K, T, sigma):
    """Undiscounted-forward straddle value under Black-76 (call + put, same strike)."""
    if T <= 1e-9 or sigma <= 1e-9:
        return abs(F - K) * math.exp(-R * T)
    srt = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / srt
    d2 = d1 - srt
    return math.exp(-R * T) * (F * (2 * _N(d1) - 1.0) + K * (1.0 - 2 * _N(d2)))


def implied_forward(call, put, K, T):
    """Put-call parity: C - P = exp(-rT)(F - K)."""
    return K + (call - put) * math.exp(R * T)


def implied_vol_straddle(price, F, K, T, lo=0.01, hi=3.0):
    """Bisection on the straddle price. Returns None if unbracketed."""
    if T <= 1e-9 or price <= 0:
        return None
    intrinsic = abs(F - K) * math.exp(-R * T)
    if price <= intrinsic:
        return None
    if straddle_b76(F, K, T, hi) < price:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if straddle_b76(F, K, T, mid) < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------- calendars ---------
def trading_days(con, start="2015-01-01"):
    rows = con.execute(
        "SELECT DISTINCT trade_date FROM nse_options_bhav "
        "WHERE symbol='NIFTY' AND trade_date>=? ORDER BY trade_date", (start,)).fetchall()
    return [r[0] for r in rows]


def monthly_expiries(con, days, start="2018-01-01", end="2026-12-31", dte_entry=45):
    """Monthly expiry = the LAST expiry of the calendar month that was ALREADY LISTED
    on its own entry day (expiry - dte_entry, rolled back to a session).

    Naive 'last expiry of the month' is wrong from 2025 on: after NSE shifted the weekly
    expiry day, months appeared where a weekly expires AFTER the monthly (Apr-2025's
    monthly was the 24th, with a weekly on the 30th). Monthlies are listed ~3 months
    ahead, weeklies only ~5 weeks, so requiring the contract to exist 45 days out picks
    the monthly and rejects the trailing weekly — which is exactly what a trader
    standing on the entry day would see.
    """
    rows = con.execute(
        "SELECT expiry_date, MIN(trade_date), SUM(contracts) FROM nse_options_bhav "
        "WHERE symbol='NIFTY' AND expiry_date>=? AND expiry_date<=? "
        "GROUP BY expiry_date ORDER BY expiry_date", (start, end)).fetchall()
    first_td = {e: ft for e, ft, _ in rows}
    by_month = {}
    for e in sorted(first_td):
        ed = prev_session(days, dstr(dparse(e) - timedelta(days=dte_entry)))
        if not ed or first_td[e] > ed:
            continue                      # contract did not exist 45 days out
        by_month[e[:7]] = e               # ascending scan -> keeps the LAST qualifier
    return dict(sorted(by_month.items()))


def prev_session(days, target):
    """Latest trading day <= target."""
    i = bisect_left(days, target)
    if i < len(days) and days[i] == target:
        return target
    return days[i - 1] if i > 0 else None


def next_session(days, target):
    i = bisect_left(days, target)
    return days[i] if i < len(days) else None


def dstr(d):
    return d.strftime("%Y-%m-%d")


def dparse(s):
    return datetime.strptime(s[:10], "%Y-%m-%d")


# --------------------------------------------------------- market data -------
def nifty_daily_close(con):
    rows = con.execute(
        "SELECT date, close FROM market_data_unified "
        "WHERE symbol='NIFTY50' AND timeframe='day' ORDER BY date").fetchall()
    return {r[0][:10]: float(r[1]) for r in rows if r[1]}


def nifty_5min(con, start, end):
    """{date_str: [(datetime, close), ...]} from real 5-min NIFTY spot bars."""
    rows = con.execute(
        "SELECT date, close FROM market_data_unified WHERE symbol='NIFTY50' "
        "AND timeframe='5minute' AND date>=? AND date<=? ORDER BY date",
        (start, end + " 23:59")).fetchall()
    out = {}
    for ds, c in rows:
        if not c:
            continue
        try:
            dt = datetime.strptime(ds[:16], "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        if dt.hour < 9 or dt.hour > 15 or (dt.hour == 15 and dt.minute > 30):
            continue
        out.setdefault(ds[:10], []).append((dt, float(c)))
    return out


def india_vix_daily(con):
    rows = con.execute(
        "SELECT date, close FROM market_data_unified "
        "WHERE symbol='INDIAVIX' AND timeframe='day' ORDER BY date").fetchall()
    return [(r[0][:10], float(r[1])) for r in rows if r[1]]


def vix_rank_series(vix):
    """Percentile rank of today's VIX vs the PREVIOUS 252 sessions (causal)."""
    out = {}
    for i, (d, v) in enumerate(vix):
        if i < 252:
            continue
        window = [x[1] for x in vix[i - 252:i]]
        out[d] = sum(1 for w in window if w < v) / len(window) * 100.0
    return out


# --------------------------------------------------------- option chain ------
def chain_for_expiry(con, expiry, d0, d1):
    """{trade_date: {strike: {'CE': row, 'PE': row}}} for one expiry over a date span."""
    rows = con.execute(
        "SELECT trade_date, strike, option_type, open, high, low, close, settle_price, "
        "       contracts, open_interest "
        "FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date=? "
        "AND trade_date>=? AND trade_date<=? ORDER BY trade_date",
        (expiry, d0, d1)).fetchall()
    out = {}
    for td, k, ot, o, h, lo, c, st, ct, oi in rows:
        if ot not in ("CE", "PE"):
            continue
        out.setdefault(td, {}).setdefault(float(k), {})[ot] = dict(
            open=o or 0.0, high=h or 0.0, low=lo or 0.0, close=c or 0.0,
            settle=st or 0.0, contracts=ct or 0, oi=oi or 0)
    return out


def pick_atm(day_chain, spot, price_field="close"):
    """Nearest strike to spot that has BOTH legs with a positive price and real volume."""
    best, bestd = None, 1e18
    for k, legs in day_chain.items():
        if "CE" not in legs or "PE" not in legs:
            continue
        ce, pe = legs["CE"], legs["PE"]
        if ce[price_field] <= 0 or pe[price_field] <= 0:
            continue
        if ce["contracts"] <= 0 or pe["contracts"] <= 0:   # binding liquidity rule
            continue
        d = abs(k - spot)
        if d < bestd:
            best, bestd = k, d
    return best


# --------------------------------------------------------- costs -------------
def costs_points(entry_prem, exit_prem, slip_pct=0.0025):
    """Round-trip cost in NIFTY points for a 1x straddle.

    slippage on both sides + STT 0.1% of sell premium + exchange txn 0.05% both sides
    + Rs 20/order brokerage over 4 orders (2 legs x 2 sides), GST 18% on brokerage+txn.
    """
    slip = slip_pct * (entry_prem + exit_prem)
    stt = 0.0010 * entry_prem
    txn = 0.0005 * (entry_prem + exit_prem)
    brok_pts = (20.0 * 4) / QTY
    gst = 0.18 * (txn + brok_pts)
    return slip + stt + txn + brok_pts + gst


# --------------------------------------------------------- simulation --------
def build_trade(con, expiry, days, spot_daily, dte_entry=45, dte_exit=21,
                roll="back", price_field="close"):
    """Set up one trade: entry date, ATM strike, credit, and the daily mark path."""
    exp_dt = dparse(expiry)
    t_entry = dstr(exp_dt - timedelta(days=dte_entry))
    t_exit = dstr(exp_dt - timedelta(days=dte_exit))
    ed = prev_session(days, t_entry) if roll == "back" else next_session(days, t_entry)
    xd = prev_session(days, t_exit) if roll == "back" else next_session(days, t_exit)
    if not ed or not xd or ed >= xd:
        return None

    chain = chain_for_expiry(con, expiry, ed, xd)
    if ed not in chain:
        return None
    spot = spot_daily.get(ed)
    if not spot:
        return None
    K = pick_atm(chain[ed], spot, price_field)
    if K is None:
        return None

    path = []
    for td in sorted(chain.keys()):
        legs = chain[td].get(K)
        if not legs or "CE" not in legs or "PE" not in legs:
            continue
        ce, pe = legs["CE"], legs["PE"]
        c = ce[price_field] + pe[price_field]
        if c <= 0:
            continue
        path.append(dict(date=td,
                         ce=ce[price_field], pe=pe[price_field], comb=c,
                         hi=(ce["high"] + pe["high"]) if ce["high"] and pe["high"] else None,
                         lo=(ce["low"] + pe["low"]) if ce["low"] and pe["low"] else None,
                         ce_c=ce["close"], pe_c=pe["close"],
                         vol=ce["contracts"] + pe["contracts"],
                         oi=ce["oi"] + pe["oi"],
                         spot=spot_daily.get(td)))
    if len(path) < 2 or path[0]["date"] != ed:
        return None
    return dict(expiry=expiry, entry_date=ed, time_exit_date=xd, strike=K,
                credit=path[0]["comb"], entry_spot=spot, path=path)


def run_daily(trade, target=0.50, stop=2.00):
    """Exit checked on the DAILY CLOSE only (real traded prices)."""
    credit = trade["credit"]
    for row in trade["path"][1:]:
        if row["comb"] <= target * credit:
            return _close(trade, row, "TARGET", row["comb"])
        if row["comb"] >= stop * credit:
            return _close(trade, row, "STOP", row["comb"])
    last = trade["path"][-1]
    return _close(trade, last, "TIME_21DTE", last["comb"])


def run_touch_bracket(trade, target=0.50, stop=2.00):
    """Absolute bound on ANY intraday scheme, using real bhav daily leg highs/lows.

    Pessimistic: a stop triggers the first day CE.high+PE.high pierces the level.
    Optimistic: a target triggers the first day CE.low+PE.low pierces the level.
    """
    credit = trade["credit"]
    stop_day = target_day = None
    for row in trade["path"][1:]:
        if stop_day is None and row["hi"] and row["hi"] >= stop * credit:
            stop_day = row
        if target_day is None and row["lo"] and row["lo"] <= target * credit:
            target_day = row
        if stop_day or target_day:
            break
    return stop_day, target_day


def _close(trade, row, reason, exit_prem):
    credit = trade["credit"]
    gross = credit - exit_prem
    return dict(expiry=trade["expiry"], entry_date=trade["entry_date"],
                exit_date=row["date"], strike=trade["strike"],
                entry_spot=trade["entry_spot"], exit_spot=row["spot"],
                credit=credit, exit_prem=exit_prem, exit_reason=reason,
                gross_pts=gross, entry_vol=trade["path"][0]["vol"],
                entry_oi=trade["path"][0]["oi"],
                days_held=(dparse(row["date"]) - dparse(trade["entry_date"])).days)


# ------------------------------------------------- intraday reconstruction ---
def run_intraday(trade, bars_by_day, tf_min, target=0.50, stop=2.00, iv_mode="prev"):
    """Exit checked on reconstructed intraday marks at tf_min resolution.

    iv_mode 'prev' : intraday IV = previous session's close IV      (causal, default)
    iv_mode 'same' : intraday IV = the same session's close IV      (anticipatory bracket)

    Forward is scaled by the real spot move: F(t) = F(prev_close) * S(t)/S(prev_close).
    At each daily close the mark snaps back to the REAL option price and is re-checked
    there, so reconstruction error cannot accumulate.
    """
    credit = trade["credit"]
    K = trade["strike"]
    exp_dt = dparse(trade["expiry"]).replace(hour=15, minute=30)
    path = trade["path"]

    # daily IV / forward from real closes
    ivs, fwds = {}, {}
    for row in path:
        T = max((exp_dt - dparse(row["date"]).replace(hour=15, minute=30)).days, 0) / 365.0
        if T <= 0:
            continue
        F = implied_forward(row["ce_c"], row["pe_c"], K, T)
        sig = implied_vol_straddle(row["comb"], F, K, T)
        if sig:
            ivs[row["date"]] = sig
            fwds[row["date"]] = F

    for i in range(1, len(path)):
        row = path[i]
        prev = path[i - 1]
        d = row["date"]
        src = d if iv_mode == "same" else prev["date"]
        sig = ivs.get(src)
        F0, S0 = fwds.get(prev["date"]), prev["spot"]
        bars = bars_by_day.get(d, [])
        if sig and F0 and S0 and bars:
            step = max(1, tf_min // 5)
            for j in range(step - 1, len(bars), step):     # candle CLOSES at tf_min
                dt, s = bars[j]
                if dt.hour == 15 and dt.minute >= 25:      # the 15:30 close is the real mark
                    continue
                T = max((exp_dt - dt).total_seconds(), 0) / (365 * 86400.0)
                if T <= 0:
                    continue
                F = F0 * (s / S0)
                mark = straddle_b76(F, K, T, sig)
                if mark <= target * credit:
                    return _close(trade, row, "TARGET", target * credit), "modelled"
                if mark >= stop * credit:
                    return _close(trade, row, "STOP", stop * credit), "modelled"
        # end-of-day check on the REAL close
        if row["comb"] <= target * credit:
            return _close(trade, row, "TARGET", row["comb"]), "real"
        if row["comb"] >= stop * credit:
            return _close(trade, row, "STOP", row["comb"]), "real"

    last = path[-1]
    return _close(trade, last, "TIME_21DTE", last["comb"]), "real"


# --------------------------------------------------------- stats -------------
def summarise(trades, slip_pct=0.0025):
    """trades: list of _close() dicts. Returns the metric block used in the report."""
    if not trades:
        return {}
    for t in trades:
        t["cost_pts"] = costs_points(t["credit"], t["exit_prem"], slip_pct)
        t["net_pts"] = t["gross_pts"] - t["cost_pts"]
    g = [t["gross_pts"] for t in trades]
    n = [t["net_pts"] for t in trades]
    wins = [x for x in n if x > 0]
    loss = [x for x in n if x <= 0]
    eq, peak, mdd = 0.0, 0.0, 0.0
    for x in n:
        eq += x
        peak = max(peak, eq)
        mdd = min(mdd, eq - peak)
    mean = sum(n) / len(n)
    var = sum((x - mean) ** 2 for x in n) / (len(n) - 1) if len(n) > 1 else 0.0
    sd = math.sqrt(var)
    t_stat = mean / (sd / math.sqrt(len(n))) if sd > 0 else 0.0
    rc = {}
    for t in trades:
        rc[t["exit_reason"]] = rc.get(t["exit_reason"], 0) + 1
    return dict(
        trades=len(trades),
        win_rate=100.0 * len(wins) / len(trades),
        avg_premium=sum(t["credit"] for t in trades) / len(trades),
        total_gross=sum(g), total_net=sum(n),
        avg_gross=sum(g) / len(g), avg_net=mean,
        avg_win=(sum(wins) / len(wins)) if wins else 0.0,
        avg_loss=(sum(loss) / len(loss)) if loss else 0.0,
        best=max(n), worst=min(n), max_dd=mdd, t_stat=t_stat,
        target=rc.get("TARGET", 0), stop=rc.get("STOP", 0), time=rc.get("TIME_21DTE", 0),
        total_net_rs=sum(n) * QTY, avg_net_rs=mean * QTY, max_dd_rs=mdd * QTY,
        avg_cost=sum(t["cost_pts"] for t in trades) / len(trades),
    )


def fmt_row(label, s):
    return ("%-22s n=%3d  win=%5.1f%%  avgPrem=%7.1f  netTot=%9.1f  net/tr=%7.1f  "
            "t=%5.2f  MaxDD=%9.1f  T/S/E=%d/%d/%d" %
            (label, s["trades"], s["win_rate"], s["avg_premium"], s["total_net"],
             s["avg_net"], s["t_stat"], s["max_dd"], s["target"], s["stop"], s["time"]))
