"""Assess the two user-supplied afternoon-straddle backtests (12:15 entry, 15:15 exit):
Trades (9).csv = baseline (no stop), Trades (10).csv = baseline + 0.4% underlying SL.
Parses parent rows (net day P&L + VIX), computes total/mean/win/maxDD/tail/Sharpe,
per-year + VIX-regime splits, and a head-to-head baseline-vs-SL comparison."""
import csv, os, math

DL = r"C:\Users\arunc\Downloads"
FILES = {"baseline": os.path.join(DL, "Trades (9).csv"),
         "stop_0p4": os.path.join(DL, "Trades (10).csv")}
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT, exist_ok=True)

def load(path):
    """Return list of dicts: {date, exit_time, vix, pnl} for parent (day) rows only."""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            idx = (r.get("Index") or "").strip()
            if not idx or "." in idx:
                continue  # skip leg rows
            try:
                pnl = float(r["P/L"])
            except (ValueError, KeyError):
                continue
            vix = None
            try:
                vix = float(r["Vix"])
            except (ValueError, KeyError, TypeError):
                pass
            rows.append({"date": r["Entry Date"].strip(),
                         "exit_time": (r["Exit Time"] or "").strip(),
                         "vix": vix, "pnl": pnl})
    return rows

def stats(rows):
    p = [r["pnl"] for r in rows]
    n = len(p); tot = sum(p)
    mean = tot / n
    sd = (sum((x - mean) ** 2 for x in p) / n) ** 0.5
    wins = [x for x in p if x > 0]; losses = [x for x in p if x < 0]
    sp = sorted(p)
    def pct(q):
        i = max(0, min(n - 1, int(q * (n - 1))))
        return sp[i]
    # max drawdown on cumulative equity
    cum = 0.0; peak = 0.0; mdd = 0.0
    for x in p:
        cum += x; peak = max(peak, cum); mdd = min(mdd, cum - peak)
    return {
        "n": n, "total": tot, "mean": mean, "median": pct(0.5), "std": sd,
        "win_rate": 100 * len(wins) / n,
        "avg_win": sum(wins) / len(wins) if wins else 0,
        "avg_loss": sum(losses) / len(losses) if losses else 0,
        "best": max(p), "worst": min(p),
        "p05": pct(0.05), "p95": pct(0.95),
        "sharpe_daily_ann": (mean / sd * (252 ** 0.5)) if sd else 0,
        "max_dd": mdd,
        "expectancy_per_day": mean,
        "sum_wins": sum(wins), "sum_losses": sum(losses),
        "profit_factor": (sum(wins) / -sum(losses)) if losses else float("inf"),
    }

def by_year(rows):
    yrs = {}
    for r in rows:
        y = r["date"][:4]
        yrs.setdefault(y, []).append(r["pnl"])
    return {y: (len(v), sum(v), 100 * sum(1 for x in v if x > 0) / len(v)) for y, v in sorted(yrs.items())}

def vix_split(rows):
    buckets = [("VIX<13", lambda v: v is not None and v < 13),
               ("13-16", lambda v: v is not None and 13 <= v < 16),
               ("16-20", lambda v: v is not None and 16 <= v < 20),
               ("VIX>=20", lambda v: v is not None and v >= 20)]
    out = {}
    for name, f in buckets:
        sel = [r["pnl"] for r in rows if f(r["vix"])]
        if sel:
            out[name] = (len(sel), sum(sel), sum(sel) / len(sel),
                         100 * sum(1 for x in sel if x > 0) / len(sel))
    return out

data = {k: load(v) for k, v in FILES.items()}
lines = []
def P(s=""):
    lines.append(s); print(s)

P("=" * 78)
P("AFTERNOON STRADDLE 12:15->15:15  (NIFTY, 10 lots / QTY 650)  — user backtests")
P("=" * 78)
for k in ("baseline", "stop_0p4"):
    s = stats(data[k])
    P(f"\n### {k.upper()}  ({s['n']} days)")
    P(f"  Total P&L       : Rs {s['total']:>14,.0f}")
    P(f"  Mean / day      : Rs {s['mean']:>14,.0f}    median Rs {s['median']:,.0f}")
    P(f"  Win rate        : {s['win_rate']:.1f}%   (avg win Rs {s['avg_win']:,.0f} | avg loss Rs {s['avg_loss']:,.0f})")
    P(f"  Profit factor   : {s['profit_factor']:.2f}")
    P(f"  Std / day       : Rs {s['std']:,.0f}    Sharpe(ann) {s['sharpe_daily_ann']:.2f}")
    P(f"  Max drawdown    : Rs {s['max_dd']:>14,.0f}")
    P(f"  Best day        : Rs {s['best']:>14,.0f}")
    P(f"  WORST day       : Rs {s['worst']:>14,.0f}")
    P(f"  Tail  P05/P95   : Rs {s['p05']:,.0f}  /  Rs {s['p95']:,.0f}")
    P(f"  Per-year (days, total, win%):")
    for y, (nn, tt, ww) in by_year(data[k]).items():
        P(f"     {y}: {nn:>3}d   Rs {tt:>13,.0f}   {ww:.0f}%")
    P(f"  By VIX regime (days, total, mean/day, win%):")
    for name, (nn, tt, mm, ww) in vix_split(data[k]).items():
        P(f"     {name:>8}: {nn:>3}d   Rs {tt:>12,.0f}   mean {mm:>8,.0f}   {ww:.0f}%")

# head-to-head: per-day delta (same dates)
P("\n" + "=" * 78)
P("HEAD-TO-HEAD: does the 0.4% SL help? (matched by date)")
P("=" * 78)
b = {r["date"]: r["pnl"] for r in data["baseline"]}
s_ = {r["date"]: r["pnl"] for r in data["stop_0p4"]}
common = sorted(set(b) & set(s_))
deltas = [s_[d] - b[d] for d in common]
sl_early = sum(1 for r in data["stop_0p4"] if r["exit_time"] and "3:15" not in r["exit_time"])
helped = sum(1 for x in deltas if x > 0); hurt = sum(1 for x in deltas if x < 0)
P(f"  Matched days        : {len(common)}")
P(f"  Days SL exited early: {sl_early}  ({100*sl_early/len(data['stop_0p4']):.0f}% of days)")
P(f"  SL total - Base total: Rs {sum(s_.values()) - sum(b.values()):,.0f}")
P(f"  Days SL better       : {helped}   Days SL worse: {hurt}")
P(f"  Sum of SL gains on its better days : Rs {sum(x for x in deltas if x>0):,.0f}")
P(f"  Sum of SL losses on its worse days : Rs {sum(x for x in deltas if x<0):,.0f}")
# worst-10 days each
P("\n  Worst 8 days BASELINE:")
for d, v in sorted(b.items(), key=lambda kv: kv[1])[:8]:
    P(f"     {d}: Rs {v:>12,.0f}   (SL that day: Rs {s_.get(d,0):,.0f})")
P("  Worst 8 days +0.4% SL:")
for d, v in sorted(s_.items(), key=lambda kv: kv[1])[:8]:
    P(f"     {d}: Rs {v:>12,.0f}   (Base that day: Rs {b.get(d,0):,.0f})")

with open(os.path.join(OUT, "user_csv_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n-> wrote results/user_csv_summary.txt")
