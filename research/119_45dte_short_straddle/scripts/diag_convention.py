#!/usr/bin/env python3
"""Diagnostics: why 88 trades and not their 83, and how sensitive the headline is to
the entry-day rolling convention and the entry price field."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine45 import (connect, trading_days, monthly_expiries, nifty_daily_close,
                      chain_for_expiry, build_trade, run_daily, summarise, fmt_row,
                      prev_session, next_session, dparse, dstr)
from datetime import timedelta

WIN_LO, WIN_HI = "2019-01-01", "2026-06-30"


def main():
    con = connect()
    days = trading_days(con, "2018-06-01")
    spot = nifty_daily_close(con)
    exps = monthly_expiries(con, days, "2018-06-01", "2026-08-31")

    print("=== skipped expiries: why? ===")
    for ym, exp in exps.items():
        t = build_trade(con, exp, days, spot)
        if t:
            continue
        exp_dt = dparse(exp)
        ed = prev_session(days, dstr(exp_dt - timedelta(days=45)))
        xd = prev_session(days, dstr(exp_dt - timedelta(days=21)))
        ch = chain_for_expiry(con, exp, ed or "", xd or "") if ed and xd else {}
        n_at_entry = len(ch.get(ed, {})) if ed else 0
        withvol = 0
        for k, legs in ch.get(ed, {}).items():
            if "CE" in legs and "PE" in legs and legs["CE"]["contracts"] > 0 and legs["PE"]["contracts"] > 0:
                withvol += 1
        print("  %s expiry=%s entry=%s exit=%s strikes_at_entry=%d with_both_legs_traded=%d"
              % (ym, exp, ed, xd, n_at_entry, withvol))

    print("\n=== convention sensitivity (window by entry date %s..%s) ===" % (WIN_LO, WIN_HI))
    for roll in ("back", "forward"):
        for field in ("close", "settle"):
            ts = []
            for ym, exp in exps.items():
                t = build_trade(con, exp, days, spot, roll=roll, price_field=field)
                if t and WIN_LO <= t["entry_date"] <= WIN_HI:
                    ts.append(run_daily(t))
            if ts:
                print("  " + fmt_row("roll=%s price=%s" % (roll, field), summarise(ts)))

    print("\n=== DTE-entry sensitivity (roll=back, close) ===")
    for dte in (40, 45, 50, 60):
        ts = []
        for ym, exp in exps.items():
            t = build_trade(con, exp, days, spot, dte_entry=dte)
            if t and WIN_LO <= t["entry_date"] <= WIN_HI:
                ts.append(run_daily(t))
        if ts:
            print("  " + fmt_row("entry %d DTE" % dte, summarise(ts)))

    print("\n=== DTE-exit sensitivity (entry 45 DTE) ===")
    for dte in (0, 7, 14, 21, 28):
        ts = []
        for ym, exp in exps.items():
            t = build_trade(con, exp, days, spot, dte_exit=dte)
            if t and WIN_LO <= t["entry_date"] <= WIN_HI:
                ts.append(run_daily(t))
        if ts:
            print("  " + fmt_row("exit %d DTE" % dte, summarise(ts)))


if __name__ == "__main__":
    main()
