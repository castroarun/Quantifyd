# -*- coding: utf-8 -*-
"""Options Behaviour Study — Phase 1 aggregator (NIFTY ATM straddle).
Pre-aggregates the per-minute NIFTY chain (options_data.db) into a COMPACT JSON the study page reads:
per recorded day, the 09:16-ATM straddle (CE+PE) 5-min premium series + a daily summary. Rebuilds all
days each run (light: ~66 days). Cron after close appends new days automatically."""
import json
import os
import sys
from datetime import datetime, timezone, timedelta, time as dtime

sys.path.insert(0, "/home/arun/quantifyd")
sys.path.insert(0, "/home/arun/quantifyd/research/90_nas_portfolio_bracket/scripts")
import numpy as np
from engine_mtm import load_day, days  # noqa

IST = timezone(timedelta(hours=5, minutes=30))
OUT = "/home/arun/quantifyd/static/app/options_study.json"
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def build_day(day):
    b = load_day(day)
    if b is None:
        return None
    chain = b["chain"]; spot_s = b["spot_s"]; times = b["times"]; dte = b["dte_day"]

    # Convert CALENDAR-days-to-expiry -> TRADING-days-to-expiry. With NIFTY's weekly
    # (Tuesday) expiry, calendar-DTE leaves DTE2/DTE3 on Sat/Sun (never traded) and pushes
    # Wed/Thu to DTE5/DTE6 (off the 0-4 scale). Trading-DTE maps DTE0..4 cleanly to
    # Tue/Mon/Fri/Thu/Wed and keeps DTE0/DTE1 identical (the 0/1-DTE edge is preserved).
    _dd = datetime.strptime(day, "%Y-%m-%d").date()
    _exp = _dd + timedelta(days=int(dte))
    _tdte, _cur = 0, _dd + timedelta(days=1)
    while _cur <= _exp:
        if _cur.weekday() < 5:          # Mon-Fri = trading day (holidays rare, ignored)
            _tdte += 1
        _cur += timedelta(days=1)
    dte = _tdte

    def prem(ts, t):
        ta, la, _, _ = chain[ts]
        i = np.searchsorted(ta, np.datetime64(t), side="right") - 1
        return float(la[i]) if i >= 0 and la[i] and la[i] > 0 else None

    def tsym(strike, typ):
        for ts, (_, _, st, ty) in chain.items():
            if int(st) == int(strike) and ty == typ:
                return ts
        return None

    ent = [t for t in times if t.time() >= dtime(9, 16)]
    if not ent:
        return None
    t0 = ent[0]; spot0 = float(spot_s.loc[t0]); atm = round(spot0 / 50) * 50
    ce, pe = tsym(atm, "CE"), tsym(atm, "PE")
    if not ce or not pe:
        return None
    grid = [t for t in times if dtime(9, 16) <= t.time() <= dtime(15, 30)
            and (t == t0 or t.minute % 5 == 0)]
    series = []
    for t in grid:
        cp, pp = prem(ce, t), prem(pe, t)
        if cp is None or pp is None:
            continue
        sp = float(spot_s.loc[t]) if t in spot_s.index else None
        series.append([t.strftime("%H:%M"), round(cp + pp, 1), round(cp, 1), round(pp, 1),
                       round(sp) if sp else None])
    if len(series) < 2:
        return None
    strad = [s[1] for s in series]
    entry, close, hi, lo = strad[0], strad[-1], max(strad), min(strad)
    last_t = grid[-1]
    spot_close = float(spot_s.loc[last_t]) if last_t in spot_s.index else spot0
    # NIFTY 5-min OHLC candles over the session (from per-tick spot) — for the chart-pattern view
    smask = [dtime(9, 15) <= ix.time() <= dtime(15, 30) for ix in spot_s.index]
    win = spot_s[smask]
    ohlc = []
    if len(win) > 1:
        c5 = win.resample("5min").agg(["first", "max", "min", "last"]).dropna()
        for ts, r in c5.iterrows():
            ohlc.append([ts.strftime("%H:%M"), round(float(r["first"])), round(float(r["max"])),
                         round(float(r["min"])), round(float(r["last"]))])
    day_hi = round(float(win.max())) if len(win) else round(spot0)
    day_lo = round(float(win.min())) if len(win) else round(spot0)
    # TICK-level straddle extremes over the session (~3s recorder) — true Max/Min for
    # Peak+%/Range; the 5-min series stays for charts but understates extremes.
    tick_hi = tick_lo = tick_pt = None
    try:
        ta_c, la_c, _, _ = chain[ce]; ta_p, la_p, _, _ = chain[pe]
        lo_b = np.datetime64(day + "T09:16:00"); hi_b = np.datetime64(day + "T15:30:00")
        allt = np.union1d(ta_c, ta_p)
        allt = allt[(allt >= lo_b) & (allt <= hi_b)]
        if len(allt):
            ic = np.searchsorted(ta_c, allt, side="right") - 1
            ip = np.searchsorted(ta_p, allt, side="right") - 1
            ok = (ic >= 0) & (ip >= 0)
            cv = np.asarray(la_c, dtype=float)[ic[ok]]
            pv = np.asarray(la_p, dtype=float)[ip[ok]]
            tt = allt[ok]
            m = (cv > 0) & (pv > 0)
            if m.any():
                comb = cv[m] + pv[m]; tt = tt[m]
                iH = int(np.argmax(comb)); iL = int(np.argmin(comb))
                tick_hi = round(float(comb[iH]), 1); tick_lo = round(float(comb[iL]), 1)
                tick_pt = str(tt[iH])[11:16]
    except Exception:
        pass
    # OTM strangles at +/- offset points (CE above, PE below) -> combined 5-min series per offset
    otm = {}
    for off in (100, 200, 300):
        cts, pts = tsym(atm + off, "CE"), tsym(atm - off, "PE")
        if not cts or not pts:
            continue
        oser = []
        for t in grid:
            cp, pp = prem(cts, t), prem(pts, t)
            if cp is None or pp is None:
                continue
            oser.append([t.strftime("%H:%M"), round(cp + pp, 1)])
        if len(oser) >= 2:
            otm[str(off)] = oser
    return dict(date=day, weekday=WD[datetime.strptime(day, "%Y-%m-%d").weekday()], dte=dte,
                atm=int(atm), entry=entry, close=close, high=hi, low=lo,
                decay_pct=round((close / entry - 1) * 100, 1) if entry else 0,
                rng=round(hi - lo, 1), spot_open=round(spot0), spot_close=round(spot_close),
                spot_move=round(spot_close - spot0), series=series, otm=otm,
                ohlc=ohlc, day_hlc=[day_hi, day_lo, round(spot_close)],
                tick_high=tick_hi, tick_low=tick_lo, tick_peak_time=tick_pt)


def main():
    out = []
    prev_hlc = None
    for i, day in enumerate(sorted(days)):     # chronological -> CPR uses the prior trading day
        try:
            d = build_day(day)
            if d:
                if prev_hlc:                    # CPR from PRIOR day's High/Low/Close
                    H, L, Cl = prev_hlc
                    piv = (H + L + Cl) / 3.0
                    bc = (H + L) / 2.0
                    tc = 2 * piv - bc
                    lo_, hi_ = sorted([tc, bc])
                    d["cpr"] = dict(tc=round(hi_, 1), pivot=round(piv, 1), bc=round(lo_, 1),
                                    width_pct=round(abs(tc - bc) / piv * 100, 3))
                prev_hlc = d.get("day_hlc")
                d.pop("day_hlc", None)
                out.append(d)
        except Exception as e:
            print("  skip", day, e, flush=True)
        if i % 10 == 0:
            print("  agg %d/%d %s" % (i, len(days), day), flush=True)
    payload = dict(generated_at=datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
                   underlying="NIFTY", n_days=len(out), days=out)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, OUT)
    print("wrote %s (%d days)" % (OUT, len(out)))


if __name__ == "__main__":
    main()
