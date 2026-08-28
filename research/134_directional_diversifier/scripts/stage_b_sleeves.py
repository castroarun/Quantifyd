#!/usr/bin/env python3
"""research/134 Stage B+D — does a LONG-ONLY TREND sleeve fix what Stage A found?

Stage A: the neutral book bleeds in up-trends (45-day NIFTY run >= +5%: mean
-1.19%, worst -9.27%) and is fine in down-trends. So the sleeve must be long the
upside grind, not long the crash.

A long-only equity trend system is, by construction, invested exactly during
sustained up-runs. This tests whether that mechanical fact survives the cost of
being wrong the rest of the time -- and, crucially, whether it beats the honest
null of simply trading the neutral book smaller (control C1, pre-declared).

Index-level sleeves only. A cross-sectional stock sleeve needs a point-in-time
universe to avoid survivorship; market_data.db carries today's listings, so that
version is deliberately NOT built here -- research/75 already did it properly.

Reads market_data.db + stage_a_monthly.csv. Writes results/ only.
"""
import csv
import os
import sqlite3
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(HERE, "results")
MKT = os.path.join(ROOT, "backtest_data", "market_data.db")

CASH_ANN = 0.065          # idle cash in a liquid fund, matching the live books
COST_BPS = 10.0 / 10000   # 10 bps round trip per switch


def nifty_daily():
    con = sqlite3.connect("file:%s?mode=ro" % MKT, uri=True)
    rows = [(r[0][:10], float(r[1]), float(r[2]), float(r[3])) for r in con.execute(
        "SELECT date, close, high, low FROM market_data_unified "
        "WHERE symbol='NIFTY50' AND timeframe='day' AND close IS NOT NULL ORDER BY date")]
    con.close()
    return rows


def monthly_from_signal(rows, signal, warmup, label):
    """Monthly % returns of a long-only sleeve, no look-ahead, costed."""
    daily_cash = (1 + CASH_ANN) ** (1 / 252.0) - 1
    eq, pos, switches = 1.0, 0, 0
    month_open, cur_month, out = None, None, {}
    for i in range(warmup, len(rows) - 1):
        d_next, c_next = rows[i + 1][0], rows[i + 1][1]
        m = d_next[:7]
        if cur_month is None:
            cur_month, month_open = m, eq
        elif m != cur_month:
            out[cur_month] = 100.0 * (eq / month_open - 1)
            cur_month, month_open = m, eq
        want = 1 if signal(rows, i) else 0
        if want != pos:
            eq *= (1 - COST_BPS)
            switches += 1
            pos = want
        eq *= (1 + ((c_next / rows[i][1] - 1) if pos else daily_cash))
    if cur_month is not None:
        out[cur_month] = 100.0 * (eq / month_open - 1)
    return out, switches


def sig_bh(rows, i):
    return True


def make_ma(n):
    def f(rows, i):
        if i < n:
            return False
        return rows[i][1] > sum(r[1] for r in rows[i - n + 1:i + 1]) / n
    return f


def make_donchian(entry, exit_):
    state = {"in": False}

    def f(rows, i):
        hi = max(r[2] for r in rows[i - entry + 1:i + 1])
        lo = min(r[3] for r in rows[i - exit_ + 1:i + 1])
        if not state["in"] and rows[i][1] >= hi - 1e-9:
            state["in"] = True
        elif state["in"] and rows[i][1] <= lo + 1e-9:
            state["in"] = False
        return state["in"]
    return f


def make_tsmom(n):
    def f(rows, i):
        if i < n:
            return False
        return rows[i][1] > rows[i - n][1]
    return f


def stats(xs):
    n = len(xs)
    mu = sum(xs) / n
    sd = (sum((x - mu) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return mu, sd, n


def corr(a, b):
    common = sorted(set(a) & set(b))
    if len(common) < 12:
        return None
    xa, xb = [a[m] for m in common], [b[m] for m in common]
    ma, sa, _ = stats(xa)
    mb, sb, _ = stats(xb)
    if sa == 0 or sb == 0:
        return None
    return (sum((x - ma) * (y - mb) for x, y in zip(xa, xb)) / (len(common) - 1)) / (sa * sb)


def perf(series):
    """CAGR %, MaxDD %, Calmar, worst month %, over a monthly dict."""
    ms = sorted(series)
    eq, peak, mdd = 1.0, 1.0, 0.0
    for m in ms:
        eq *= (1 + series[m] / 100.0)
        peak = max(peak, eq)
        mdd = min(mdd, 100.0 * (eq / peak - 1))
    yrs = len(ms) / 12.0
    cagr = 100.0 * (eq ** (1 / yrs) - 1) if yrs > 0 and eq > 0 else float("nan")
    worst = min(series[m] for m in ms)
    return cagr, mdd, (cagr / abs(mdd) if mdd else float("nan")), worst


def main():
    # ---- the neutral book, from Stage A -----------------------------------
    neutral, run45 = {}, {}
    with open(os.path.join(RES, "stage_a_monthly.csv")) as f:
        for r in csv.DictReader(f):
            neutral[r["month"]] = float(r["combined_pct"])
            run45[r["month"]] = float(r["nifty_run45_pct"])
    months = sorted(neutral)
    print("neutral book: %d months %s -> %s" % (len(months), months[0], months[-1]))

    rows = nifty_daily()
    warm = 260
    sleeves = {
        "NIFTY B&H": (sig_bh, warm),
        "MA200 long/cash": (make_ma(200), warm),
        "MA50 long/cash": (make_ma(50), warm),
        "Donchian 20/10": (make_donchian(20, 10), warm),
        "Donchian 55/20": (make_donchian(55, 20), warm),
        "TS-mom 12m": (make_tsmom(252), warm),
    }

    built = {}
    print("\n%-18s %8s %8s %8s %8s %7s %8s" % (
        "sleeve", "CAGR%", "MaxDD%", "Calmar", "worst", "corr", "switches"))
    for name, (sig, w) in sleeves.items():
        mm, sw = monthly_from_signal(rows, sig, w, name)
        mm = {m: v for m, v in mm.items() if m in neutral}
        built[name] = mm
        c, d, cal, wo = perf(mm)
        print("%-18s %8.2f %8.1f %8.2f %8.2f %7s %8d" % (
            name, c, d, cal, wo,
            "%+.2f" % corr(neutral, mm) if corr(neutral, mm) is not None else "n/a", sw))

    # ---- does it pay in the months that hurt? ------------------------------
    print("\nSLEEVE RETURN BY REGIME  (the neutral book bleeds in 'up')")
    print("%-18s %18s %18s %18s" % ("sleeve", "up >=+5% (n)", "chop (n)", "down <=-5% (n)"))
    def reg(m):
        r = run45[m]
        return "up" if r >= 5 else ("down" if r <= -5 else "chop")
    print("%-18s" % "NEUTRAL BOOK", end="")
    for k in ("up", "chop", "down"):
        xs = [neutral[m] for m in months if reg(m) == k]
        print("%18s" % ("%+.2f%% (%d)" % (sum(xs) / len(xs), len(xs))), end="")
    print()
    for name, mm in built.items():
        print("%-18s" % name, end="")
        for k in ("up", "chop", "down"):
            xs = [mm[m] for m in sorted(mm) if reg(m) == k]
            print("%18s" % ("%+.2f%% (%d)" % (sum(xs) / len(xs), len(xs)) if xs else "-"), end="")
        print()

    # ---- combination sweep vs the SIZE-DOWN null ---------------------------
    cash_m = (1 + CASH_ANN) ** (1 / 12.0) - 1
    base_c, base_d, base_cal, base_w = perf(neutral)
    print("\nNEUTRAL BOOK ALONE: CAGR %.2f%%  MaxDD %.1f%%  Calmar %.2f  worst %.2f%%"
          % (base_c, base_d, base_cal, base_w))

    def sized(s):
        return {m: s * neutral[m] + (1 - s) * 100 * cash_m for m in months}

    def sizedown_cagr_at(target_worst):
        """CAGR of the pure neutral book shrunk until its worst month == target."""
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if perf(sized(mid))[3] < target_worst:
                hi = mid
            else:
                lo = mid
        return perf(sized(lo))[0], lo

    out_rows = []
    print("\nCOMBINATION SWEEP  -- and the control that can kill it")
    print("%-18s %5s %8s %8s %8s %8s   %s" % (
        "sleeve", "w", "CAGR%", "MaxDD%", "Calmar", "worst%", "vs SIZE-DOWN at same worst"))
    for name, mm in built.items():
        common = [m for m in months if m in mm]
        for w in (0.10, 0.20, 0.30, 0.40, 0.50):
            comb = {m: (1 - w) * neutral[m] + w * mm[m] for m in common}
            c, d, cal, wo = perf(comb)
            sd_c, s = sizedown_cagr_at(wo)
            verdict = "BEATS  +%.2f%%" % (c - sd_c) if c > sd_c else "loses %.2f%%" % (c - sd_c)
            print("%-18s %5.0f%% %8.2f %8.1f %8.2f %8.2f   %s (size %.0f%%, CAGR %.2f%%)"
                  % (name, 100 * w, c, d, cal, wo, verdict, 100 * s, sd_c))
            out_rows.append(dict(sleeve=name, w=w, cagr=round(c, 3), maxdd=round(d, 2),
                                 calmar=round(cal, 3), worst=round(wo, 2),
                                 sizedown_cagr=round(sd_c, 3),
                                 sizedown_scale=round(s, 3),
                                 edge_vs_sizedown=round(c - sd_c, 3)))
    with open(os.path.join(RES, "stage_b_combos.csv"), "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w_.writeheader()
        w_.writerows(out_rows)
    best = max(out_rows, key=lambda r: r["edge_vs_sizedown"])
    print("\nBEST vs the null: %s at %.0f%% -> %+.2f%% CAGR over size-down at the same worst month"
          % (best["sleeve"], 100 * best["w"], best["edge_vs_sizedown"]))
    print("wrote %s" % os.path.join(RES, "stage_b_combos.csv"))


if __name__ == "__main__":
    sys.exit(main())
