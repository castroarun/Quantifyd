"""Research 109 — optimise the breakout book's GATE and CHECK FREQUENCY.

research/71 settled the exit rule (trailing, no target), concurrency (8-10) and that a NIFTY>200DMA
gate is mandatory — but it only ever tested 200DMA *versus no gate*, and always checked entries and
the trail DAILY. Two gaps, both live problems:

  * The book is paralysed right now with NIFTY 0.19% BELOW its 200-DMA — a hair-trigger boundary.
    Does a HYSTERESIS BAND (or a different/slower/faster MA) beat the naked cross?
  * Does checking entries / the trailing exit WEEKLY instead of daily help or hurt?

Trades are generated once per exit-check frequency, then gates x entry-frequency are swept cheaply.
Base exit = Donchian-20 trail + 20% catastrophe stop (research/71 winner). Cost 0.20% RT, capital 10L.
"""
import sqlite3, os, time, json
import numpy as np, pandas as pd

BASE = "/home/arun/quantifyd/research/109_breakout_gate_freq"
RES = os.path.join(BASE, "results"); os.makedirs(RES, exist_ok=True)
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
CAP, COST, TMED, GAP_MAX, MAXBARS, CATA, START = 1_000_000.0, 0.0020, 5.0, 0.15, 120, 0.20, "2006-01-01"
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def macd(c):
    return ema(c, 12) - ema(c, 26)


con = sqlite3.connect(DB)
bn = pd.read_sql("select date,close from market_data_unified where symbol='NIFTYBEES' and timeframe='day' order by date",
                 con, parse_dates=["date"]).set_index("date")
bn = bn[~bn.index.duplicated()]
for n in (50, 100, 150, 200):
    bn[f"ma{n}"] = bn["close"].rolling(n).mean()
bn["ema200"] = ema(bn["close"], 200)
cal = bn.index[bn.index >= START]
FRI = set(d.toordinal() for d in cal if d.weekday() == 4)

# ── gate variants ──────────────────────────────────────────────────────────────
def build_gates():
    g = {}
    c, m200 = bn["close"], bn["ma200"]
    def d2o(ser):
        return {d.toordinal(): bool(v) for d, v in ser.items() if d in set(cal)}
    g["ma200 (current)"] = d2o(c > m200)
    for n in (50, 100, 150):
        g[f"ma{n}"] = d2o(c > bn[f"ma{n}"])
    g["ema200"] = d2o(c > bn["ema200"])
    g["ma200 + rising"] = d2o((c > m200) & (m200 > m200.shift(20)))
    # hysteresis: turn ON above +band, only turn OFF below -band (kills boundary whipsaw)
    for band in (0.01, 0.02, 0.03):
        st, out = False, {}
        for d in bn.index:
            m = bn.at[d, "ma200"]
            if not np.isnan(m):
                if not st and bn.at[d, "close"] > m * (1 + band):
                    st = True
                elif st and bn.at[d, "close"] < m * (1 - band):
                    st = False
            if d in set(cal):
                out[d.toordinal()] = st
        g[f"ma200 hysteresis +/-{int(band*100)}%"] = out
    # gate evaluated only on Friday closes, held all week
    st, out = False, {}
    for d in bn.index:
        m = bn.at[d, "ma200"]
        if d.weekday() == 4 and not np.isnan(m):
            st = bool(bn.at[d, "close"] > m)
        if d in set(cal):
            out[d.toordinal()] = st
    g["ma200 checked weekly"] = out
    return g


GATES = build_gates()
log(f"gates built: {list(GATES)}")
for k, v in GATES.items():
    log(f"  {k:26} risk-on {100*np.mean(list(v.values())):.0f}% of days")

# ── trade generation (Donchian-20 trail), daily vs weekly exit checking ────────
syms = [r[0] for r in con.execute("select distinct symbol from market_data_unified where timeframe='day'")]
symclose, TRADES = {}, {"daily": [], "weekly": []}
log(f"scanning {len(syms)} symbols ...")
for si, sym in enumerate(syms):
    df = pd.read_sql("select date,open,high,low,close,volume from market_data_unified where symbol=? and timeframe='day' order by date",
                     con, params=(sym,), parse_dates=["date"]).set_index("date")
    df = df[df.index.notna()]
    df = df[~df.index.duplicated()]
    if len(df) < 300:
        continue
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    ords = np.array([d.toordinal() for d in df.index])
    symclose[sym] = (ords, c.values)
    md = macd(c)
    mw = macd(c.resample("W-FRI").last()).reindex(df.index, method="ffill")
    mm = macd(c.resample("ME").last()).reindex(df.index, method="ffill")
    hi252 = c.rolling(252, min_periods=200).max()
    v20 = v.rolling(20).mean()
    tmed = (c * v).rolling(20).median() / 1e7          # Rs crore
    run = c / c.shift() - 1
    qual = ((md > 0) & (mw > 0) & (mm > 0) & (c >= 0.98 * hi252) & (v >= 2 * v20) &
            (tmed >= TMED) & (c >= 20) & (run <= 0.15))
    dn20 = l.rolling(20).min().shift()                  # prior 20-day low = the trail
    qpos = np.where(qual.values)[0]
    cv, dnv = c.values, dn20.values
    for i in qpos:
        if i + 1 >= len(cv):
            continue
        ent = cv[i]
        stop_hard = ent * (1 - CATA)
        end = min(i + 1 + MAXBARS, len(cv))
        trail_break = -1
        for j in range(i + 1, end):
            if cv[j] < dnv[j] or cv[j] <= stop_hard:
                trail_break = j
                break
        if trail_break < 0:
            trail_break = end - 1
        # weekly variant: act on the break only at the next Friday close
        wj = trail_break
        while wj < len(cv) - 1 and ords[wj] not in FRI:
            wj += 1
        for key, jx in (("daily", trail_break), ("weekly", wj)):
            TRADES[key].append(dict(sym=sym, entry_ord=int(ords[i]), exit_ord=int(ords[jx]),
                                    entry_px=float(ent), exit_px=float(cv[jx]),
                                    pct_run=float(run.values[i])))
    if si % 300 == 0:
        log(f"  {si}/{len(syms)} symbols, {len(TRADES['daily'])} trades")
log(f"trades: daily={len(TRADES['daily'])} weekly={len(TRADES['weekly'])}")


def close_at(sym, t_ord):
    o, cv = symclose[sym]
    k = np.searchsorted(o, t_ord, "right") - 1
    return cv[k] if k >= 0 else np.nan


def run_book(exit_freq, gate_key, entry_freq, max_open=8):
    by_entry = {}
    for tr in TRADES[exit_freq]:
        by_entry.setdefault(tr["entry_ord"], []).append(tr)
    gate = GATES[gate_key]
    cash, open_pos, curve = CAP, [], []
    for d in cal:
        to = d.toordinal()
        for p in list(open_pos):
            if p["exit_ord"] <= to:
                cash += p["notional"] * (p["exit_px"] / p["entry_px"]) * (1 - COST)
                open_pos.remove(p)
        unreal = sum(p["notional"] * (close_at(p["sym"], to) / p["entry_px"] - 1) for p in open_pos)
        invested = sum(p["notional"] for p in open_pos)
        curve.append((d, cash + invested + unreal))
        take = gate.get(to, False) and len(open_pos) < max_open and to in by_entry
        if entry_freq == "weekly" and to not in FRI:
            take = False
        if take:
            held = {p["sym"] for p in open_pos}
            cands = sorted([x for x in by_entry[to] if x["sym"] not in held], key=lambda x: -x["pct_run"])
            for x in cands:
                if len(open_pos) >= max_open:
                    break
                size = (cash + invested + unreal) / max_open
                if size > cash:
                    break
                cash -= size
                open_pos.append(dict(sym=x["sym"], notional=size * (1 - COST),
                                     entry_px=x["entry_px"], exit_px=x["exit_px"], exit_ord=x["exit_ord"]))
    cur = pd.Series({d: e for d, e in curve})
    yrs = (cur.index[-1] - cur.index[0]).days / 365.25
    cagr = (cur.iloc[-1] / cur.iloc[0]) ** (1 / yrs) - 1
    dr = cur.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    dd = ((cur - cur.cummax()) / cur.cummax()).min()
    return dict(cagr=round(cagr * 100, 1), maxdd=round(dd * 100, 1), sharpe=round(sh, 2),
                calmar=round(cagr / abs(dd), 2) if dd < 0 else 0, mult=round(cur.iloc[-1] / cur.iloc[0], 1))


rows = []
print(f"\n{'gate':>28} {'entry':>7} {'exit chk':>9} {'CAGR':>6} {'MaxDD':>7} {'Sharpe':>7} {'Calmar':>7} {'mult':>7}")
for gk in GATES:
    for ef in ("daily", "weekly"):
        for xf in ("daily", "weekly"):
            if gk != "ma200 (current)" and (ef != "daily" or xf != "daily"):
                continue                                  # full freq grid only on the base gate
            r = run_book(xf, gk, ef)
            rows.append(dict(gate=gk, entry=ef, exit_check=xf, **r))
            print(f"{gk:>28} {ef:>7} {xf:>9} {r['cagr']:>5.1f}% {r['maxdd']:>6.1f}% "
                  f"{r['sharpe']:>7.2f} {r['calmar']:>7.2f} {r['mult']:>6.1f}x", flush=True)
# frequency grid on the best gate too
best = max([r for r in rows if r["entry"] == "daily" and r["exit_check"] == "daily"], key=lambda r: r["calmar"])
if best["gate"] != "ma200 (current)":
    print(f"\n-- frequency grid on the winning gate: {best['gate']} --")
    for ef in ("daily", "weekly"):
        for xf in ("daily", "weekly"):
            if ef == "daily" and xf == "daily":
                continue
            r = run_book(xf, best["gate"], ef)
            rows.append(dict(gate=best["gate"], entry=ef, exit_check=xf, **r))
            print(f"{best['gate']:>28} {ef:>7} {xf:>9} {r['cagr']:>5.1f}% {r['maxdd']:>6.1f}% "
                  f"{r['sharpe']:>7.2f} {r['calmar']:>7.2f} {r['mult']:>6.1f}x", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(RES, "gate_freq_sweep.csv"), index=False)
top = sorted(rows, key=lambda r: -r["calmar"])[:5]
print("\nTOP 5 by Calmar:")
for r in top:
    print(f"  {r['gate']:>28} entry={r['entry']:<7} exit={r['exit_check']:<7} "
          f"CAGR {r['cagr']}%  DD {r['maxdd']}%  Calmar {r['calmar']}")
cur_row = [r for r in rows if r["gate"] == "ma200 (current)" and r["entry"] == "daily" and r["exit_check"] == "daily"][0]
print(f"\nCURRENT LIVE CONFIG: CAGR {cur_row['cagr']}%  DD {cur_row['maxdd']}%  Calmar {cur_row['calmar']}")
log("done")
