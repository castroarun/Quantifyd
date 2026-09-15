#!/usr/bin/env python3
"""
research/174 — long-tenor short-premium engine (daily close, real NSE bhavcopy).

Design notes that matter:

* DAILY CLOSE ONLY. Expired-contract intraday option data cannot be obtained from Kite,
  and the 1-min recorder starts 2026-04-20 at ~27 DTE. Every exit here is a close-based
  decision on a real traded price. Nothing intraday is claimed.

* BINDING LIQUIDITY RULE (research/89). A leg is fillable on a session only if that
  strike/side has close > 0 AND contracts >= vol_floor on that session. Long-dated
  strikes carry stale settlement prints; filling them manufactures edge. Entry requires
  every leg fillable; an exit that requires a trade requires every leg fillable, and if
  the chain is not fillable on the intended exit day we roll forward to the next
  fillable session (and record the roll).

* Expiry classification is derived from data, never hardcoded: an expiry's LISTING
  HORIZON is (first session it appears) -> expiry. Weeklies are listed ~5-9 weeks out,
  monthlies ~3 months, the quarterly/semi-annual series ~3 years. The monthly weekday
  moved Thursday -> Tuesday, and legacy far-dated Thursday contracts coexist with real
  monthlies, so any "last expiry of the month" rule is unsafe.

* Costs follow research/119 `costs_points`: slippage both sides + STT 0.10% of the sell
  premium + exchange txn 0.05% both sides + Rs 20 per order + GST 18% on brokerage+txn.
  Slippage is a swept axis because a 240-DTE strike is far wider than a 45-DTE one.
"""
import math
import sqlite3
from bisect import bisect_left
from datetime import datetime, timedelta
from pathlib import Path

LOT = 65                    # NIFTY contract size. option_chain.lot_size is WRONG (says 75).


# ----------------------------------------------------------------- db --------
def db_path():
    for p in [Path("/home/arun/quantifyd/backtest_data/market_data.db"),
              Path(__file__).resolve().parents[3] / "backtest_data" / "market_data.db"]:
        if p.exists():
            return str(p)
    raise FileNotFoundError("market_data.db not found")


def connect():
    return sqlite3.connect("file:%s?mode=ro" % db_path(), uri=True)


def dstr(d):
    return d.strftime("%Y-%m-%d")


def dparse(s):
    return datetime.strptime(s[:10], "%Y-%m-%d")


def prev_session(days, target):
    i = bisect_left(days, target)
    if i < len(days) and days[i] == target:
        return target
    return days[i - 1] if i > 0 else None


# ------------------------------------------------------------- calendars -----
def sessions(con, symbol="NIFTY", start="2011-01-01"):
    return [r[0] for r in con.execute(
        "SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol=? AND trade_date>=? "
        "ORDER BY trade_date", (symbol, start))]


def expiry_catalogue(con, symbol="NIFTY"):
    """{expiry: dict(first_seen, horizon_days, cls)} — cls derived, never hardcoded."""
    out = {}
    for e, first in con.execute(
            "SELECT expiry_date, MIN(trade_date) FROM nse_options_bhav WHERE symbol=? "
            "GROUP BY expiry_date", (symbol,)):
        h = (dparse(e) - dparse(first)).days
        cls = "weekly" if h < 80 else ("monthly" if h < 300 else "longdated")
        out[e] = dict(first_seen=first, horizon=h, cls=cls)
    return out


def spot_series(con, underlying="NIFTY50"):
    return {r[0][:10]: float(r[1]) for r in con.execute(
        "SELECT date, close FROM market_data_unified WHERE symbol=? AND timeframe='day'",
        (underlying,)) if r[1]}


def vix_series(con):
    rows = [(r[0][:10], float(r[1])) for r in con.execute(
        "SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' "
        "AND timeframe='day' ORDER BY date") if r[1]]
    lvl = dict(rows)
    rank = {}
    for i, (d, v) in enumerate(rows):
        if i < 252:
            continue
        w = [x[1] for x in rows[i - 252:i]]
        rank[d] = 100.0 * sum(1 for x in w if x < v) / len(w)
    return lvl, rank


# ----------------------------------------------------------------- chain -----
def expiry_chain(con, symbol, expiry):
    """{session: {strike: {'CE': (close,contracts,oi), 'PE': ...}}} for one expiry."""
    out = {}
    for td, k, ot, c, ct, oi in con.execute(
            "SELECT trade_date, strike, option_type, close, contracts, open_interest "
            "FROM nse_options_bhav WHERE symbol=? AND expiry_date=? ORDER BY trade_date",
            (symbol, expiry)):
        if ot not in ("CE", "PE"):
            continue
        out.setdefault(td, {}).setdefault(float(k), {})[ot] = (c or 0.0, ct or 0, oi or 0)
    return out


def fillable(chain_day, strike, side, vol_floor):
    legs = chain_day.get(strike)
    if not legs or side not in legs:
        return False
    c, ct, _ = legs[side]
    return c > 0 and ct >= vol_floor


def pick_strike(chain_day, ref, vol_floor, side_pair=("CE", "PE")):
    """Nearest strike to `ref` where every requested side is fillable."""
    best, bd = None, 1e18
    for k in chain_day:
        if not all(fillable(chain_day, k, s, vol_floor) for s in side_pair):
            continue
        d = abs(k - ref)
        if d < bd:
            best, bd = k, d
    return best


def leg_price(chain_day, strike, side):
    legs = chain_day.get(strike)
    if not legs or side not in legs:
        return None
    c = legs[side][0]
    return c if c > 0 else None


# ----------------------------------------------------------------- costs -----
def costs_points(entry_prem, exit_prem, slip_pct, n_legs=2):
    """Round-trip cost in index points. n_legs counts every leg of the structure."""
    slip = slip_pct * (entry_prem + exit_prem)
    stt = 0.0010 * entry_prem              # STT on the sell side of premium
    txn = 0.0005 * (entry_prem + exit_prem)
    brok_pts = (20.0 * 2 * n_legs) / LOT   # Rs 20 per order, entry+exit, per leg, per LOT
    gst = 0.18 * (txn + brok_pts)
    return slip + stt + txn + brok_pts + gst


# ------------------------------------------------------------- structures ----
def build_position(chain_day, spot, spec, vol_floor):
    """spec = dict(kind='STR'|'STG'|'IC'|'WS', body=frac, wing=frac).

    Returns dict(legs=[(strike, side, sign)], credit=float) where sign -1 = short.
    Every leg must be fillable or None is returned.
    """
    kind = spec["kind"]
    legs = []
    if kind == "STR":
        k = pick_strike(chain_day, spot, vol_floor, ("CE", "PE"))
        if k is None:
            return None
        legs = [(k, "CE", -1), (k, "PE", -1)]
    elif kind in ("STG", "IC"):
        b = spec["body"]
        kc = pick_strike(chain_day, spot * (1 + b), vol_floor, ("CE",))
        kp = pick_strike(chain_day, spot * (1 - b), vol_floor, ("PE",))
        if kc is None or kp is None or kc <= kp:
            return None
        legs = [(kc, "CE", -1), (kp, "PE", -1)]
        if kind == "IC":
            w = spec["wing"]
            wc = pick_strike(chain_day, spot * (1 + b + w), vol_floor, ("CE",))
            wp = pick_strike(chain_day, spot * (1 - b - w), vol_floor, ("PE",))
            if wc is None or wp is None or wc <= kc or wp >= kp:
                return None
            legs += [(wc, "CE", +1), (wp, "PE", +1)]
    elif kind == "WS":
        k = pick_strike(chain_day, spot, vol_floor, ("CE", "PE"))
        if k is None:
            return None
        w = spec["wing"]
        wc = pick_strike(chain_day, spot * (1 + w), vol_floor, ("CE",))
        wp = pick_strike(chain_day, spot * (1 - w), vol_floor, ("PE",))
        if k is None or wc is None or wp is None or wc <= k or wp >= k:
            return None
        legs = [(k, "CE", -1), (k, "PE", -1), (wc, "CE", +1), (wp, "PE", +1)]
    else:
        raise ValueError(kind)

    credit = 0.0
    for k, s, sign in legs:
        p = leg_price(chain_day, k, s)
        if p is None:
            return None
        credit += -sign * p            # short legs add, long legs subtract
    if credit <= 0:
        return None
    return dict(legs=legs, credit=credit)


def mark(chain_day, legs, vol_floor_exit):
    """Net cost to CLOSE the position (positive = we pay). None if not markable."""
    tot = 0.0
    for k, s, sign in legs:
        if not fillable(chain_day, k, s, vol_floor_exit):
            return None
        p = leg_price(chain_day, k, s)
        if p is None:
            return None
        tot += -sign * p
    return tot


def side_mark(chain_day, legs, side, vol_floor_exit):
    """Cost to close just the short legs of one side ('CE' or 'PE')."""
    tot = 0.0
    found = False
    for k, s, sign in legs:
        if s != side:
            continue
        if not fillable(chain_day, k, s, vol_floor_exit):
            return None
        p = leg_price(chain_day, k, s)
        if p is None:
            return None
        tot += -sign * p
        found = True
    return tot if found else None


def costs_points_legs(entry_px, exit_px, slip_pct):
    """Cost in index points for a multi-leg structure, priced leg by leg.

    entry_px / exit_px: list of (price, sign) with sign -1 short, +1 long.

    Why this exists: `costs_points` charges slippage on the NET credit, which is right
    for a straddle (net == gross) and badly wrong for a condor — a condor's net credit
    is small but you still cross four bid/ask spreads. Slippage here is charged on the
    GROSS premium turnover of every leg.

      slippage  : slip_pct x sum(|leg premium|) over entry and exit
      STT       : 0.10% of the SELL-side premium only (entry shorts, exit longs)
      exchange  : 0.05% of total premium turnover
      brokerage : Rs 20 per order, both sides, per leg, converted to points via LOT
      GST       : 18% on brokerage + exchange
    """
    gross = sum(abs(p) for p, _ in entry_px) + sum(abs(p) for p, _ in exit_px)
    sell_prem = (sum(abs(p) for p, s in entry_px if s < 0)
                 + sum(abs(p) for p, s in exit_px if s > 0))
    n_legs = max(len(entry_px), len(exit_px))
    slip = slip_pct * gross
    stt = 0.0010 * sell_prem
    txn = 0.0005 * gross
    brok_pts = (20.0 * 2 * n_legs) / LOT
    gst = 0.18 * (txn + brok_pts)
    return slip + stt + txn + brok_pts + gst
