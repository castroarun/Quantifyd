"""Research 112 - when should FRESH DEPOSITED cash enter the momentum book?

The book has two studied cash policies pointing OPPOSITE ways, and deposits fall under neither:
  stop-out cash          -> MONTHLY (r/108: weekly halves net Calmar; slots are empty BECAUSE those
                            names were falling -> fast refill buys weakness = false-dawn penalty)
  all-cash gate re-entry -> WEEKLY  (r/41 ph-27: the penalty does not apply from an all-cash state)
A deposit carries NO adverse selection - it arrives because Arun had spare cash - so neither verdict
transfers by analogy. Test it.

The SPLIT rule is already settled (equal-rupee, evenly across holdings - live code ~line 1335);
this study fixes only the TIMING.

Fair-comparison device: the `park` arm defines the deposit calendar and rupee amounts; every other
arm receives IDENTICAL cash flows on IDENTICAL dates. Without this the richer arm would deposit more
and the test would be circular. Identical cash in => the arm that ends richer wins.
"""
import importlib.util, csv
from pathlib import Path
import numpy as np, pandas as pd, logging
logging.disable(logging.WARNING)
spec = importlib.util.spec_from_file_location(
    "m75", "/home/arun/quantifyd/research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py")
m75 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m75)
BENCH, EXCLUDE = m75.BENCH, m75.EXCLUDE
RES = Path("/home/arun/quantifyd/research/112_deposit_timing/results"); RES.mkdir(parents=True, exist_ok=True)
START = pd.Timestamp("2011-01-01"); CASH_Y, STCG, RT = 0.065, 0.20, 0.0015
N_HOLD, BUFFER, POOL = 8, 22, 30
DEP_EVERY, DEP_PCT, N_PHASE = 63, 0.05, 12   # quarterly deposits of 5% of reference NAV, 12 calendars
YEARS = 15.6

close, tv = m75.load()
idx = close.index[(close.index >= START) & (close.index <= pd.Timestamp("2026-08-01"))]
ME = set(pd.Timestamp(x) for x in pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max().values)
_iso = idx.isocalendar()
WK = set(pd.Timestamp(x) for x in
         pd.Series(idx, index=idx).groupby([_iso.year.values, _iso.week.values]).last().values)
ROLL_LOW = close.rolling(15, min_periods=15).min().shift(1); NBX = close[BENCH].ffill()

print("precomputing rankings ...", flush=True)
SNAP = {}
for d in sorted(ME | WK):
    h = close.loc[:d].ffill()
    if len(h) <= 253:
        continue
    adv = tv.loc[:d].tail(126).median().dropna()
    adv = adv[[s for s in adv.index if s not in EXCLUDE and s != BENCH]]
    uni = adv.sort_values(ascending=False).index[:200]
    sc = None
    for L, w in ((126, 0.5), (252, 0.5)):
        p0, p1 = h.iloc[-L - 1], h.iloc[-1]
        r = (p1 / p0) / (p1[BENCH] / p0[BENCH]) * w
        sc = r if sc is None else sc.add(r, fill_value=np.nan)
    sub = sc[[s for s in uni if s in sc.index and pd.notna(sc[s])]]
    SNAP[d] = list(sub.sort_values(ascending=False).index[:POOL])
print("  %d ranking dates" % len(SNAP), flush=True)


def run(arm, dep_dates, deps=None):
    """arm: park | weekly | immediate | immediate_topup.
    deps=None -> this run DEFINES the deposit amounts (5%% of its own NAV) and returns them."""
    cash, held, prev, derisked, tax = 1.0, {}, None, False, 0.0
    pending = 0.0            # deposited cash still awaiting deployment under the arm's rule
    made = {}
    contrib = 1.0
    pre, post, idle = [], [], []

    def E():
        return sum(v[0] for v in held.values())

    def sell(s, d):
        nonlocal cash, tax
        v, b, bd = held.pop(s)
        if v > b and (d - bd).days < 365:
            tax += (v - b) * STCG
        cash += v * (1 - RT)

    def buy(s, d, amt):
        nonlocal cash
        amt = min(amt, cash)
        if amt <= 1e-9:
            return 0.0
        if s in held:
            held[s][0] += amt * (1 - RT); held[s][1] += amt   # top-up keeps the cost basis
        else:
            held[s] = [amt * (1 - RT), amt, d]
        cash -= amt
        return amt

    def month_end(d):
        """Rotate-only, exactly like the live book: sell out-of-buffer, buy NEW names from freed
        cash, let kept winners ride (no trim, no top-up of kept names)."""
        nonlocal pending
        etf = SNAP.get(d)
        if not etf:
            return
        buf = set(etf[:BUFFER])
        for s in list(held):
            if s not in buf:
                sell(s, d)
        empty = N_HOLD - len(held)
        if empty > 0 and cash > 0:
            cands = [s for s in etf if s not in held and s in close.columns
                     and pd.notna(close.at[d, s])][:empty]
            per = (E() + cash) / N_HOLD
            for s in cands:
                buy(s, d, min(per, cash))
        pending = max(0.0, min(pending, cash))   # parked money just spent is no longer pending

    def deploy_fresh(d, topup_only):
        """Deploy the pending DEPOSIT: empty slots first (unless top-up-only), then EVEN rupee
        split across held names - the settled split rule."""
        nonlocal pending
        if pending <= 1e-9 or derisked:
            return
        amt = min(pending, cash)
        if amt <= 1e-9:
            return
        dd = max([x for x in SNAP if x <= d], default=None)
        etf = SNAP.get(dd, []) if dd else []
        if not topup_only and len(held) < N_HOLD and etf:
            per = (E() + cash) / N_HOLD
            for s in [x for x in etf if x not in held and x in close.columns
                      and pd.notna(close.at[d, x])][:N_HOLD - len(held)]:
                amt -= buy(s, d, min(per, amt))
                if amt <= 1e-9:
                    break
        live_held = [s for s in held if s in close.columns and pd.notna(close.at[d, s])]
        if amt > 1e-9 and live_held:
            per = amt / len(live_held)          # EVEN rupee split (not pro-rata to position size)
            for s in live_held:
                amt -= buy(s, d, per)
        pending = max(0.0, amt)

    for d in idx:
        if held and prev is not None:
            for s, v in held.items():
                p1 = close.at[d, s] if s in close.columns else np.nan
                p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    v[0] += v[0] * (p1 / p0 - 1.0)
        if cash > 0:
            cash *= (1 + CASH_Y) ** (1 / 252)
        if held:
            lows = ROLL_LOW.loc[d]
            for s in list(held):
                p1 = close.at[d, s] if s in close.columns else np.nan
                ln = lows.get(s, np.nan)
                if pd.notna(p1) and pd.notna(ln) and p1 < ln:
                    sell(s, d)
        # -- the deposit lands --
        if d in dep_dates:
            a = (E() + cash) * DEP_PCT if deps is None else deps.get(d, 0.0)
            if deps is None:
                made[d] = a
            cash += a; pending += a; contrib += a
            if arm in ("immediate", "immediate_topup"):
                deploy_fresh(d, topup_only=(arm == "immediate_topup"))
        if d in WK:
            b = NBX.loc[:d].dropna()
            risk_off = len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()
            if risk_off:
                for s in list(held):
                    sell(s, d)
                derisked = True
            else:
                if derisked:        # gate flipped back ON from all-cash -> r/41 ph-27 weekly re-entry
                    etf = SNAP.get(d)
                    if etf:
                        per = (E() + cash) / N_HOLD
                        for s in etf[:N_HOLD]:
                            buy(s, d, min(per, cash))
                derisked = False
                if arm == "weekly":
                    deploy_fresh(d, topup_only=False)
        if not derisked and d in ME:
            month_end(d)
        nav = E() + cash
        pre.append((d, nav)); post.append((d, nav - tax))
        idle.append(cash / nav if nav > 0 else 0)
        prev = d

    n = pd.DataFrame(post, columns=["d", "v"]).set_index("d")["v"]
    dr = n.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    dd = float(((n - n.cummax()) / n.cummax()).min() * 100)
    term = float(n.iloc[-1])
    gx = term / contrib
    cagr = ((gx ** (1 / YEARS)) - 1) * 100
    return dict(terminal=round(term, 2), contributed=round(contrib, 2), gain_x=round(gx, 3),
                cagr=round(cagr, 2), maxdd=round(dd, 1), sharpe=round(float(sh), 2),
                calmar=round(cagr / abs(dd), 2) if dd < 0 else 0.0,
                idle_pct=round(100 * float(np.mean(idle)), 1)), made


ARMS = ["park", "weekly", "immediate", "immediate_topup"]
rows = []
fh = open(RES / "deposit_timing.csv", "w", newline="")
w = csv.writer(fh)
w.writerow(["arm", "phase", "terminal", "contributed", "gain_x", "cagr", "maxdd", "sharpe",
            "calmar", "idle_pct"])
print("\n%18s %6s %10s %8s %8s %7s" % ("arm", "phase", "terminal", "gain_x", "MaxDD", "idle"), flush=True)
for ph in range(N_PHASE):
    dep_dates = set(idx[i] for i in range(260 + ph * 5, len(idx), DEP_EVERY))
    _, deps = run("park", dep_dates, None)          # park defines the cash-flow calendar
    for arm in ARMS:
        r, _ = run(arm, dep_dates, deps)
        rows.append(dict(arm=arm, phase=ph, **r))
        w.writerow([arm, ph, r["terminal"], r["contributed"], r["gain_x"], r["cagr"], r["maxdd"],
                    r["sharpe"], r["calmar"], r["idle_pct"]])
        fh.flush()
        print("%18s %6d %10.2f %8.3f %7.1f%% %6.1f%%"
              % (arm, ph, r["terminal"], r["gain_x"], r["maxdd"], r["idle_pct"]), flush=True)
fh.close()

df = pd.DataFrame(rows)
base = df[df.arm == "park"].set_index("phase")["gain_x"]
print("\n" + "=" * 88)
print("%18s %11s %9s %10s %9s %10s %12s"
      % ("arm", "med gain_x", "mean", "med CAGR", "med MaxDD", "med idle", "beats park"))
for a in ARMS:
    s = df[df.arm == a].set_index("phase")
    wins = int((s["gain_x"] > base).sum())
    print("%18s %11.3f %9.3f %9.2f%% %9.1f%% %9.1f%% %9d/%d"
          % (a, s["gain_x"].median(), s["gain_x"].mean(), s["cagr"].median(),
             s["maxdd"].median(), s["idle_pct"].median(), wins, N_PHASE))
print("=" * 88)
b = max(ARMS, key=lambda a: df[df.arm == a]["gain_x"].median())
print("BEST by median gain_x: %s" % b)
