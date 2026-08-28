#!/usr/bin/env python3
"""research/134 Stage C — robustness on the Stage B result, and the two questions
that decide whether any of it is real.

Stage B said a long-only equity sleeve at 30-40% dominates the size-down null by
a wide margin. Two things must be checked before that can be believed:

  1. NIFTY compounded at 17.5%/yr over this window. ANY long-equity sleeve looks
     good in a bull market. Does the result survive an era split -- and what does
     it look like if the equity premium is stripped out entirely?
  2. Stage B's Donchian sleeves were BROKEN (close >= max-of-highs-including-today
     can essentially never fire; they sat in cash for seven years, 1 switch, 0.0%
     MaxDD). Fixed here so the trend family gets a fair hearing.

Also answers the practical question: does trend TIMING add anything over simply
holding the index?
"""
import csv
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(HERE, "results")
MKT = os.path.join(ROOT, "backtest_data", "market_data.db")

CASH_ANN = 0.065
COST_BPS = 10.0 / 10000


def nifty_daily():
    con = sqlite3.connect("file:%s?mode=ro" % MKT, uri=True)
    rows = [(r[0][:10], float(r[1]), float(r[2]), float(r[3])) for r in con.execute(
        "SELECT date, close, high, low FROM market_data_unified "
        "WHERE symbol='NIFTY50' AND timeframe='day' AND close IS NOT NULL ORDER BY date")]
    con.close()
    return rows


def monthly_from_signal(rows, signal, warmup):
    daily_cash = (1 + CASH_ANN) ** (1 / 252.0) - 1
    eq, pos, sw = 1.0, 0, 0
    month_open, cur, out = None, None, {}
    for i in range(warmup, len(rows) - 1):
        m = rows[i + 1][0][:7]
        if cur is None:
            cur, month_open = m, eq
        elif m != cur:
            out[cur] = 100.0 * (eq / month_open - 1)
            cur, month_open = m, eq
        want = 1 if signal(rows, i) else 0
        if want != pos:
            eq *= (1 - COST_BPS)
            sw += 1
            pos = want
        eq *= (1 + ((rows[i + 1][1] / rows[i][1] - 1) if pos else daily_cash))
    if cur is not None:
        out[cur] = 100.0 * (eq / month_open - 1)
    return out, sw


def sig_bh(rows, i):
    return True


def make_ma(n):
    def f(rows, i):
        return i >= n and rows[i][1] > sum(r[1] for r in rows[i - n + 1:i + 1]) / n
    return f


def make_donchian(entry, exit_):
    """FIXED: breakout of the PRIOR window's high, exit on the prior window's low.
    Excluding today's bar is what makes the level a level rather than a tautology."""
    st = {"in": False}

    def f(rows, i):
        if i < max(entry, exit_) + 1:
            return False
        hi = max(r[2] for r in rows[i - entry:i])       # prior `entry` bars, excl today
        lo = min(r[3] for r in rows[i - exit_:i])
        if not st["in"] and rows[i][1] > hi:
            st["in"] = True
        elif st["in"] and rows[i][1] < lo:
            st["in"] = False
        return st["in"]
    return f


def make_tsmom(n):
    def f(rows, i):
        return i >= n and rows[i][1] > rows[i - n][1]
    return f


def perf(series):
    ms = sorted(series)
    if not ms:
        return (float("nan"),) * 4
    eq, peak, mdd = 1.0, 1.0, 0.0
    for m in ms:
        eq *= (1 + series[m] / 100.0)
        peak = max(peak, eq)
        mdd = min(mdd, 100.0 * (eq / peak - 1))
    yrs = len(ms) / 12.0
    cagr = 100.0 * (eq ** (1 / yrs) - 1) if eq > 0 else float("nan")
    return cagr, mdd, (cagr / abs(mdd) if mdd else float("nan")), min(series.values())


def main():
    neutral, run45 = {}, {}
    with open(os.path.join(RES, "stage_a_monthly.csv")) as f:
        for r in csv.DictReader(f):
            neutral[r["month"]] = float(r["combined_pct"])
            run45[r["month"]] = float(r["nifty_run45_pct"])
    months = sorted(neutral)
    rows = nifty_daily()
    W = 300

    sleeves = {
        "NIFTY B&H": make_ma(0) if False else sig_bh,
        "MA200 long/cash": make_ma(200),
        "MA50 long/cash": make_ma(50),
        "Donchian 20/10": make_donchian(20, 10),
        "Donchian 55/20": make_donchian(55, 20),
        "TS-mom 12m": make_tsmom(252),
    }
    built = {}
    print("=" * 92)
    print("1. THE TREND FAMILY, WITH DONCHIAN FIXED  (does timing beat just holding?)")
    print("=" * 92)
    print("%-18s %8s %8s %8s %8s %9s" % ("sleeve", "CAGR%", "MaxDD%", "Calmar", "worst%", "switches"))
    for name, sig in sleeves.items():
        mm, sw = monthly_from_signal(rows, sig, W)
        mm = {m: v for m, v in mm.items() if m in neutral}
        built[name] = mm
        c, d, cal, wo = perf(mm)
        print("%-18s %8.2f %8.1f %8.2f %8.2f %9d" % (name, c, d, cal, wo, sw))

    cash_m = (1 + CASH_ANN) ** (1 / 12.0) - 1

    def sized(s, ms):
        return {m: s * neutral[m] + (1 - s) * 100 * cash_m for m in ms}

    def sizedown_at(worst, ms):
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if perf(sized(mid, ms))[3] < worst:
                hi = mid
            else:
                lo = mid
        return perf(sized(lo, ms))[0]

    # ---- 2. ERA SPLIT ------------------------------------------------------
    eras = [("EARLY 2019-05..2022-12", [m for m in months if m <= "2022-12"]),
            ("LATE  2023-01..2026-07", [m for m in months if m > "2022-12"])]
    print("\n" + "=" * 92)
    print("2. ERA SPLIT  (control C4 -- does the benefit exist in both halves?)")
    print("=" * 92)
    for ename, ms in eras:
        nb = {m: neutral[m] for m in ms}
        c0, d0, cal0, w0 = perf(nb)
        idxc = perf({m: built["NIFTY B&H"][m] for m in ms if m in built["NIFTY B&H"]})[0]
        print("\n%s   (n=%d)   NIFTY CAGR %.1f%%" % (ename, len(ms), idxc))
        print("  neutral alone      CAGR %6.2f%%  MaxDD %6.1f%%  Calmar %5.2f  worst %6.2f%%"
              % (c0, d0, cal0, w0))
        print("  %-18s %8s %8s %8s %8s   %s"
              % ("sleeve @30%", "CAGR%", "MaxDD%", "Calmar", "worst%", "vs size-down"))
        for name, mm in built.items():
            common = [m for m in ms if m in mm]
            if len(common) < 12:
                continue
            comb = {m: 0.7 * neutral[m] + 0.3 * mm[m] for m in common}
            c, d, cal, wo = perf(comb)
            sd = sizedown_at(wo, common)
            print("  %-18s %8.2f %8.1f %8.2f %8.2f   %s"
                  % (name, c, d, cal, wo,
                     ("BEATS +%.2f%%" % (c - sd)) if c > sd else ("loses %.2f%%" % (c - sd))))

    # ---- 3. strip the equity premium --------------------------------------
    print("\n" + "=" * 92)
    print("3. STRIP THE EQUITY PREMIUM  (is it the hedge, or just owning stocks in a bull run?)")
    print("=" * 92)
    print("Re-run with the sleeve's mean monthly return forced to the cash rate --")
    print("same month-to-month SHAPE, zero excess return. Any remaining gain is pure")
    print("diversification; any loss means the benefit was the bull market.\n")
    print("%-18s %8s %8s %8s %8s   %s"
          % ("sleeve @30% (demeaned)", "CAGR%", "MaxDD%", "Calmar", "worst%", "vs size-down"))
    for name, mm in built.items():
        common = [m for m in months if m in mm]
        mu = sum(mm[m] for m in common) / len(common)
        flat = {m: mm[m] - mu + 100 * cash_m for m in common}
        comb = {m: 0.7 * neutral[m] + 0.3 * flat[m] for m in common}
        c, d, cal, wo = perf(comb)
        sd = sizedown_at(wo, common)
        print("%-18s %8.2f %8.1f %8.2f %8.2f   %s"
              % (name, c, d, cal, wo,
                 ("BEATS +%.2f%%" % (c - sd)) if c > sd else ("loses %.2f%%" % (c - sd))))

    # ---- 4. the complementarity, stated plainly ---------------------------
    print("\n" + "=" * 92)
    print("4. THE COMPLEMENTARITY  (why this pairing works at all)")
    print("=" * 92)
    def reg(m):
        r = run45[m]
        return "up" if r >= 5 else ("down" if r <= -5 else "chop")
    bh = built["NIFTY B&H"]
    print("%-22s %14s %14s %14s" % ("", "up-trend", "chop", "down-trend"))
    for lab, ser in (("neutral book", neutral), ("NIFTY B&H sleeve", bh),
                     ("70/30 blend", {m: 0.7 * neutral[m] + 0.3 * bh[m]
                                      for m in months if m in bh})):
        print("%-22s" % lab, end="")
        for k in ("up", "chop", "down"):
            xs = [ser[m] for m in sorted(ser) if reg(m) == k]
            print("%14s" % ("%+.2f%% (%d)" % (sum(xs) / len(xs), len(xs))), end="")
        print()
    print("\nworst month, each: neutral %.2f%%   sleeve %.2f%%   blend %.2f%%"
          % (perf(neutral)[3], perf(bh)[3],
             perf({m: 0.7 * neutral[m] + 0.3 * bh[m] for m in months if m in bh})[3]))


if __name__ == "__main__":
    sys.exit(main())
