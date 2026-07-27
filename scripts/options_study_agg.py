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
    return dict(date=day, weekday=WD[datetime.strptime(day, "%Y-%m-%d").weekday()], dte=dte,
                atm=int(atm), entry=entry, close=close, high=hi, low=lo,
                decay_pct=round((close / entry - 1) * 100, 1) if entry else 0,
                rng=round(hi - lo, 1), spot_open=round(spot0), spot_close=round(spot_close),
                spot_move=round(spot_close - spot0), series=series)


def main():
    out = []
    for i, day in enumerate(days):
        try:
            d = build_day(day)
            if d:
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
