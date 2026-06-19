"""Parametrized true-replay engine for the 6 ATM NAS systems on the recorded NIFTY
weekly option chain (backtest_data/options_data.db). Derived from research/51's
generate_replay.py with faithfulness fixes + two improvement knobs.

Knobs (the only two levers research/54 left alive + DTE handling):
  stop_mode     : "premium" (per-leg entry*1.3) | "move" (±move_pct underlying, one-and-done)
  strike_offset : 0 = ATM (nearest 50) ; 100 = sell 100pt-OTM strangle
  time_exit     : force squareoff time (live = 15:15)

Faithfulness fixes vs research/51:
  - squeeze ATR uses Wilder/RMA (matches services/nas_scanner.py), start 09:30 (was 09:35)
  - force-exit at 15:15 (live squareoff; research/51 used 14:45)
9:16 entry is exact; squeeze entry is reconstructed from per-min spot (approx).
1-min snapshot cadence => SL/ST resolved to ~1 min; LTP fills, no slippage (optimistic tail).
"""
import sqlite3
from pathlib import Path
from datetime import time as dtime
import numpy as np
import pandas as pd

ROOT = Path("/home/arun/quantifyd")
BASE = Path(__file__).resolve().parents[1]
OPT_DB = ROOT / "backtest_data" / "options_data.db"

LOT = 65; LOTS = 2; QTY = LOT * LOTS      # normalized 2 lots for cross-system comparability
BROK_PER_LEG = 40 * 2                      # sell+buy => Rs 160/strangle
SL_MULT = 1.3
ENTRY_916 = dtime(9, 16); ENTRY_WIN_END = dtime(14, 30)
SQ_START = dtime(9, 30)
DEFAULT_EXIT = dtime(15, 15)               # live squareoff

# The 6 ATM systems going live Monday. (name, entry_mode, mgmt, live_lots)
SYSTEMS_6 = [
    ("Squeeze ATM",  "squeeze", "SL_ST",      2),
    ("Squeeze ATM2", "squeeze", "CASCADE",    2),
    ("Squeeze ATM4", "squeeze", "ROLL_MATCH", 2),
    ("916 ATM",      "t916",    "SL_ST",      1),
    ("916 ATM2",     "t916",    "CASCADE",    1),
    ("916 ATM4",     "t916",    "ROLL_MATCH", 1),
]

oc = sqlite3.connect(str(OPT_DB))
oc.execute("CREATE INDEX IF NOT EXISTS idx_oc_sym_day ON option_chain(symbol, snapshot_time)")
days = [r[0] for r in oc.execute(
    "SELECT DISTINCT substr(snapshot_time,1,10) FROM option_chain WHERE symbol='NIFTY' ORDER BY 1")]


def front_expiry(day_df, day):
    exps = sorted(day_df["expiry_date"].unique())
    fut = [e for e in exps if e >= day]
    return fut[0] if fut else (exps[-1] if exps else None)


def supertrend_up(prem_series, period=7, mult=2.0, tf="5min"):
    """OHLC SuperTrend on a leg's premium; True where trend UP (premium rising
    => exit the short). Matches NasAtm4Executor._compute_supertrend (Wilder ATR)."""
    if prem_series.empty:
        return pd.Series(dtype=bool)
    o = prem_series.resample(tf).agg(["first", "max", "min", "last"]).dropna()
    if len(o) < period + 1:
        return pd.Series(False, index=o.index)
    o.columns = ["open", "high", "low", "close"]
    pc = o["close"].shift(1)
    tr = pd.concat([o["high"] - o["low"], (o["high"] - pc).abs(), (o["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()   # Wilder
    hl2 = (o["high"] + o["low"]) / 2
    up = hl2 - mult * atr; dn = hl2 + mult * atr
    dir_up = pd.Series(index=o.index, dtype=bool)
    fu = fl = np.nan; trend = True
    for i in range(len(o)):
        c = o["close"].iloc[i]
        if np.isnan(atr.iloc[i]):
            dir_up.iloc[i] = True; continue
        fu = up.iloc[i] if (np.isnan(fu) or up.iloc[i] > fu or o["close"].iloc[i - 1] < fu) else fu
        fl = dn.iloc[i] if (np.isnan(fl) or dn.iloc[i] < fl or o["close"].iloc[i - 1] > fl) else fl
        if trend and c < fu:
            trend = False
        elif (not trend) and c > fl:
            trend = True
        dir_up.iloc[i] = trend
    return dir_up


def load_day(day):
    """Load + parse one day's front-expiry chain ONCE (the expensive part).
    Returns a reusable bundle so a sweep can run many systems/configs per day."""
    df = pd.read_sql_query(
        "SELECT snapshot_time, tradingsymbol, strike, instrument_type, ltp, expiry_date, underlying_spot "
        "FROM option_chain WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? AND ltp IS NOT NULL",
        oc, params=[day])
    if df.empty:
        return None
    df["t"] = pd.to_datetime(df["snapshot_time"])
    exp = front_expiry(df, day)
    df = df[df["expiry_date"] == exp]
    if df.empty:
        return None
    dte_day = (pd.to_datetime(exp).date() - pd.to_datetime(day).date()).days
    spot_s = df.groupby("t")["underlying_spot"].first().sort_index()
    chain = {}
    for ts, g in df.groupby("tradingsymbol"):
        g = g.sort_values("t")
        chain[ts] = (g["t"].values, g["ltp"].values, g["strike"].iloc[0], g["instrument_type"].iloc[0])
    return {"day": day, "chain": chain, "spot_s": spot_s, "times": spot_s.index, "dte_day": dte_day}


def sim_day(day, entry_mode, mgmt, **kw):
    b = load_day(day)
    if b is None:
        return []
    return sim_loaded(b, entry_mode, mgmt, **kw)


def sim_loaded(bundle, entry_mode, mgmt, stop_mode="premium", move_pct=0.004,
               strike_offset=0, time_exit=DEFAULT_EXIT, allow_reentry=True,
               sl_mult=SL_MULT, st_period=7, st_mult=2.0, st_tf="5min",
               naked_method="st", pct_trail=0.30,
               profit_target=None, position_stop=None):
    """Replay one preloaded day for one system. Returns list of closed-leg dicts.
    Phase-D knobs: sl_mult (premium SL x), st_* (SuperTrend trail params),
    naked_method in {st, breakeven, pct_trail} (how the naked survivor trails),
    profit_target (close all at X% of credit captured), position_stop (close all
    at X% of credit lost on the whole strangle, instead of per-leg)."""
    day = bundle["day"]; chain = bundle["chain"]; spot_s = bundle["spot_s"]
    times = bundle["times"]; dte_day = bundle["dte_day"]

    def prem(ts, t):
        if ts not in chain:
            return None
        ta, la, _, _ = chain[ts]
        idx = np.searchsorted(ta, np.datetime64(t), side="right") - 1
        if idx < 0:
            return None
        v = la[idx]
        return float(v) if v and v > 0 else None

    def tsym(strike, typ):
        for ts, (_, _, st, ty) in chain.items():
            if int(st) == int(strike) and ty == typ:
                return ts
        return None

    def pick_strangle(spot, offset):
        atm = round(spot / 50) * 50
        ce = tsym(atm + offset, "CE")
        pe = tsym(atm - offset, "PE")
        return ce, pe

    # ---- entry time ----
    if entry_mode == "t916":
        ent_times = [t for t in times if t.time() >= ENTRY_916]
        if not ent_times:
            return []
        t0 = ent_times[0]
    else:  # squeeze: Wilder ATR(14) < SMA(ATR,50) on 5-min spot, first after 09:30
        c5 = spot_s.resample("5min").agg(["first", "max", "min", "last"]).dropna()
        if len(c5) < 55:
            t0 = next((t for t in times if t.time() >= SQ_START), None)
        else:
            c5.columns = ["open", "high", "low", "close"]
            pc = c5["close"].shift(1)
            tr = pd.concat([c5["high"] - c5["low"], (c5["high"] - pc).abs(), (c5["low"] - pc).abs()], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()   # Wilder
            sma = atr.rolling(50).mean()
            sq = (atr < sma) & (c5.index.to_series().dt.time >= SQ_START) & (c5.index.to_series().dt.time <= ENTRY_WIN_END)
            t0 = sq[sq].index[0] if sq.any() else None
        if t0 is None:
            return []
        t0 = min((t for t in times if t >= t0), default=None)
        if t0 is None:
            return []
    spot0 = float(spot_s.loc[t0])

    def open_strangle(t, spot):
        ce, pe = pick_strangle(spot, strike_offset)
        out = []
        for ts in (ce, pe):
            if ts is None:
                continue
            p = prem(ts, t)
            if p is None:
                continue
            out.append({"tsym": ts, "typ": chain[ts][3], "entry": p, "qty": QTY,
                        "sl": p * sl_mult, "open": True, "naked_st": False})
        return out

    legs = open_strangle(t0, spot0)
    if not legs:
        return []
    credit0 = sum(l["entry"] * l["qty"] for l in legs)   # Rs credit received at entry

    realized = []
    reentries = 0
    sim_times = [t for t in times if t >= t0 and t.time() <= time_exit]
    st_cache = {}

    def close_leg(lg, t, reason):
        p = prem(lg["tsym"], t)
        if p is None:
            p = lg["entry"]
        pnl = (lg["entry"] - p) * lg["qty"] - BROK_PER_LEG
        lg["open"] = False
        realized.append({"day": day, "dte": dte_day, "tsym": lg["tsym"], "typ": lg["typ"],
                         "entry": lg["entry"], "exit": p, "qty": lg["qty"],
                         "pnl": pnl, "reason": reason})

    for t in sim_times:
        if not any(l["open"] for l in legs):
            break
        force = t.time() >= time_exit

        # ---- system-level move-stop (one-and-done): close BOTH on underlying move ----
        if stop_mode == "move" and not force:
            sp = float(spot_s.loc[t]) if t in spot_s.index else spot0
            if abs(sp - spot0) / spot0 >= move_pct:
                for lg in legs:
                    if lg["open"]:
                        close_leg(lg, t, "MOVE_STOP")
                break

        # ---- whole-strangle profit-target / position-stop (on open legs vs credit) ----
        if (profit_target or position_stop) and not force:
            openl = [l for l in legs if l["open"]]
            cur_pnl = sum((l["entry"] - (prem(l["tsym"], t) or l["entry"])) * l["qty"] for l in openl)
            if profit_target and cur_pnl >= profit_target * credit0:
                for lg in openl:
                    close_leg(lg, t, "PTARGET")
                break
            if position_stop and cur_pnl <= -position_stop * credit0:
                for lg in openl:
                    close_leg(lg, t, "POS_STOP")
                break

        for lg in legs:
            if not lg["open"]:
                continue
            p = prem(lg["tsym"], t)
            if p is None:
                continue
            if force:
                close_leg(lg, t, "time_exit"); continue
            if lg["naked_st"]:
                if "naked_low" not in lg:
                    lg["naked_low"] = p
                lg["naked_low"] = min(lg["naked_low"], p)
                if naked_method == "breakeven":
                    # lock the survivor's gain: exit if premium climbs back to entry
                    if p >= lg["entry"]:
                        close_leg(lg, t, "BE_EXIT")
                    continue
                if naked_method == "pct_trail":
                    # trailing stop on premium: exit if it rises pct_trail above its post-naked low
                    if p >= lg["naked_low"] * (1 + pct_trail):
                        close_leg(lg, t, "TRAIL_EXIT")
                    continue
                # default: SuperTrend trail
                key = lg["tsym"]
                if key not in st_cache:
                    ta, la, _, _ = chain[lg["tsym"]]
                    ps = pd.Series(la, index=pd.DatetimeIndex(ta)).sort_index()
                    st_cache[key] = supertrend_up(ps, period=st_period, mult=st_mult, tf=st_tf)
                stu = st_cache[key]
                sub = stu[stu.index <= t]
                if len(sub) and bool(sub.iloc[-1]):
                    close_leg(lg, t, "ST_EXIT")
                continue
            # premium-stop management (skipped entirely in move-stop mode)
            if stop_mode == "premium" and p >= lg["sl"]:
                if mgmt == "SL_ST":
                    close_leg(lg, t, "SL_HIT")
                    for o in legs:
                        if o["open"]:
                            o["naked_st"] = True
                elif mgmt == "CASCADE":
                    for o in legs:
                        if o["open"]:
                            close_leg(o, t, "SL_CASCADE")
                    if allow_reentry and reentries < 5 and t.time() <= ENTRY_WIN_END:
                        reentries += 1
                        legs.extend(open_strangle(t, float(spot_s.loc[t]) if t in spot_s.index else spot0))
                    break
                elif mgmt == "ROLL_MATCH":
                    surv = next((o for o in legs if o["open"] and o is not lg), None)
                    close_leg(lg, t, "SL_ROLL")
                    if surv and not lg.get("rolled_once"):
                        sp = prem(surv["tsym"], t) or surv["entry"]
                        nts = None; best = 1e9
                        for ts, (_, _, st, ty) in chain.items():
                            if ty != lg["typ"]:
                                continue
                            pp = prem(ts, t)
                            if pp is None or pp < 5:
                                continue
                            if abs(pp - sp) < best:
                                best = abs(pp - sp); nts = ts
                        if nts:
                            np_ = prem(nts, t)
                            legs.append({"tsym": nts, "typ": lg["typ"], "entry": np_, "qty": QTY,
                                         "sl": np_ * sl_mult, "open": True, "naked_st": False,
                                         "rolled_once": True})
                    else:
                        for o in legs:
                            if o["open"]:
                                o["naked_st"] = True

    last_t = sim_times[-1] if sim_times else t0
    for lg in legs:
        if lg["open"]:
            close_leg(lg, last_t, "eod")
    return realized
