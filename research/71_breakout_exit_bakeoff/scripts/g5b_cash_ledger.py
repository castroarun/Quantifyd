"""G5b: liquid-fund settlement realism on the G5 winner (Donchian-20 + gate + 8 slots + 1/day).

Re-runs the winning book through an explicit daily CASH LEDGER instead of the notional model:
  A. published      — G5 as-is: no cash yield, no cash constraint (sanity reproduction)
  B. naive-instant  — one cash pool earning 6.5%/yr, instantly available (current paper-book model)
  C. realistic      — 3 buckets: idle settled BUFFER (1 slot, earns 0) held every day slots are
                      free; liquid fund earns 6.5% (parkings start T+1); redemptions and equity
                      sale proceeds usable T+1; after an EOD buy the buffer is refilled from the
                      fund the same day (usable tomorrow).  <- user's spec
  D. gate-aware     — C, but buffer held only while the NIFTY>200DMA gate is ON (flip-day entries
                      impossible; capability resumes T+1)
"""
import sqlite3, pandas as pd, numpy as np, os, time
from collections import defaultdict

RES = "/home/arun/quantifyd/research/71_breakout_exit_bakeoff/results"
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
CAP = 1_000_000.0; COST = 0.0020; TMED = 5.0; GAP_MAX = 0.15; MAXBARS = 120; CATA = 0.20
START = "2006-01-01"; MAX_OPEN = 8; YIELD = 0.065
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def macd(c): return ema(c, 12) - ema(c, 26)
def fta(b, s):
    sub = b[s:]; w = np.argmax(sub); return s + w if sub.size and sub[w] else -1

con = sqlite3.connect(DB)
bn = pd.read_sql("select date,close from market_data_unified where symbol='NIFTYBEES' and timeframe='day' order by date",
                 con, parse_dates=["date"]).set_index("date")
bn = bn[~bn.index.duplicated()]; bn["ma200"] = bn["close"].rolling(200).mean()
cal = bn.index[bn.index >= START]
regime_ok = {d.toordinal(): (bn.loc[d, "close"] > bn.loc[d, "ma200"]) if not np.isnan(bn.loc[d, "ma200"]) else False for d in cal}
syms = [r[0] for r in con.execute("select distinct symbol from market_data_unified where timeframe='day'")]
symclose = {}; trades = []
for sym in syms:
    df = pd.read_sql("select date,open,high,low,close,volume from market_data_unified where symbol=? and timeframe='day' order by date",
                     con, params=(sym,), parse_dates=["date"]).set_index("date")
    df = df[df.index.notna()]
    if len(df) < 300: continue
    c = df["close"]; o = df["open"].values; hi = df["high"].values; lo = df["low"].values; cl = c.values
    ords = np.array([d.toordinal() for d in df.index])
    md = macd(c); mw = macd(c.resample("W-FRI").last()).reindex(df.index, method="ffill")
    mm = macd(c.resample("ME").last()).reindex(df.index, method="ffill")
    volspike = df["volume"] / df["volume"].rolling(20).mean()
    hi252 = c.rolling(252, min_periods=200).max()
    turn_med = (c * df["volume"]).rolling(20).median() / 1e7
    clean = ((md > 0) & (mw > 0) & (mm > 0) & (volspike >= 2.0) & (c >= 0.98 * hi252) & (c >= 20) & (turn_med >= TMED)).values
    dn = cl < pd.Series(lo, index=df.index).rolling(20).min().shift(1).values
    n = len(df); symclose[sym] = (ords, cl)
    for i in np.where(clean)[0]:
        if i + 1 >= n: continue
        e = i + 1; ep = o[e]
        if ep <= 0 or np.isnan(ep) or (hi[e] - lo[e]) / ep < 0.01 or (ep / cl[i] - 1) > GAP_MAX: continue
        if ords[e] < cal[0].toordinal(): continue
        cs = ep * (1 - CATA); end = min(n - 1, e + MAXBARS)
        tj = fta(dn, e); tj = tj if 0 <= tj <= end else -1
        catj = fta(lo <= cs, e); catj = catj if 0 <= catj <= end else -1
        cands = [x for x in [tj, catj] if x >= 0]
        if not cands: xi, xp = end, cl[end]
        elif catj >= 0 and (tj < 0 or catj <= tj): xi, xp = catj, min(o[catj], cs)
        else: xi = min(tj + 1, n - 1); xp = (o[tj + 1] if tj + 1 < n else cl[tj])
        trades.append(dict(sym=sym, eo=int(ords[e]), ep=float(ep), run=float((cl[i] / cl[i - 1] - 1) * 100),
                           xo=int(ords[xi]), xp=float(xp)))
con.close()
trades.sort(key=lambda x: x["eo"]); by_entry = defaultdict(list)
for tr in trades: by_entry[tr["eo"]].append(tr)
log(f"{len(trades)} candidate trades loaded")

def close_at(sym, to):
    ords, cl = symclose[sym]; k = np.searchsorted(ords, to, side="right") - 1
    return cl[k] if k >= 0 else np.nan

def metrics(curve, extra):
    cur = pd.Series({d: e for d, e in curve})
    yrs = (cur.index[-1] - cur.index[0]).days / 365.25
    cagr = (cur.iloc[-1] / CAP) ** (1 / yrs) - 1
    dr = cur.pct_change().dropna(); sharpe = dr.mean() / (dr.std() + 1e-9) * np.sqrt(252)
    peak = cur.cummax(); maxdd = (cur / peak - 1).min()
    out = dict(cagr=round(cagr * 100, 2), sharpe=round(sharpe, 2), maxdd=round(maxdd * 100, 1),
               calmar=round(cagr / abs(maxdd), 2) if maxdd < 0 else 0, final_L=round(cur.iloc[-1] / 1e5, 1))
    out.update(extra); return out, cur

def sim_A():
    realized = 0.0; open_pos = []; curve = []
    for d in cal:
        to = d.toordinal()
        still = []
        for p in open_pos:
            if to >= p["xo"]: realized += p["notional"] * (p["xp"] / p["ep"] - 1) - p["notional"] * COST
            else: still.append(p)
        open_pos = still
        unreal = sum(p["notional"] * (close_at(p["sym"], to) / p["ep"] - 1) for p in open_pos) if open_pos else 0.0
        eq = CAP + realized + unreal
        if regime_ok.get(to, False) and len(open_pos) < MAX_OPEN and to in by_entry:
            held = {p["sym"] for p in open_pos}
            cands = sorted([x for x in by_entry[to] if x["sym"] not in held], key=lambda x: -x["run"])
            if cands:
                x = cands[0]
                open_pos.append(dict(sym=x["sym"], ep=x["ep"], xo=x["xo"], xp=x["xp"], notional=eq / MAX_OPEN))
        curve.append((d, eq))
    return metrics(curve, dict(variant="A_published", interest_L=0.0, blocked=0))

def sim_B():
    cash = CAP; open_pos = []; curve = []; interest = 0.0; prev = None; blocked = 0
    for d in cal:
        to = d.toordinal()
        if prev is not None:
            g = cash * ((1 + YIELD) ** ((d - prev).days / 365) - 1); cash += g; interest += g
        still = []
        for p in open_pos:
            if to >= p["xo"]:
                proceeds = p["qty"] * p["xp"]; cash += proceeds - proceeds * COST / 2
            else: still.append(p)
        open_pos = still
        eqv = sum(p["qty"] * close_at(p["sym"], to) for p in open_pos) if open_pos else 0.0
        nav = eqv + cash
        if regime_ok.get(to, False) and len(open_pos) < MAX_OPEN and to in by_entry:
            held = {p["sym"] for p in open_pos}
            cands = sorted([x for x in by_entry[to] if x["sym"] not in held], key=lambda x: -x["run"])
            if cands:
                spend = min(cash / (1 + COST / 2), nav / MAX_OPEN)
                if spend >= 5000:
                    x = cands[0]; qty = spend / x["ep"]
                    cash -= spend + spend * COST / 2
                    open_pos.append(dict(sym=x["sym"], ep=x["ep"], qty=qty, xo=x["xo"], xp=x["xp"]))
                else: blocked += 1
        curve.append((d, nav)); prev = d
    return metrics(curve, dict(variant="B_naive_instant", interest_L=round(interest / 1e5, 1), blocked=blocked))

def sim_CD(gate_aware):
    cash_avail = CAP; cash_pending = 0.0; liquid = 0.0; liquid_pending = 0.0
    open_pos = []; curve = []; interest = 0.0; prev = None; blocked = 0
    for d in cal:
        to = d.toordinal()
        if prev is not None:
            g = liquid * ((1 + YIELD) ** ((d - prev).days / 365) - 1); interest += g
            liquid += g + liquid_pending; liquid_pending = 0.0   # parkings earn from T+1
            cash_avail += cash_pending; cash_pending = 0.0
        still = []
        for p in open_pos:
            if to >= p["xo"]:
                proceeds = p["qty"] * p["xp"]; cash_pending += proceeds - proceeds * COST / 2
            else: still.append(p)
        open_pos = still
        eqv = sum(p["qty"] * close_at(p["sym"], to) for p in open_pos) if open_pos else 0.0
        nav = eqv + cash_avail + cash_pending + liquid + liquid_pending
        gate = regime_ok.get(to, False)
        if gate and len(open_pos) < MAX_OPEN and to in by_entry:
            held = {p["sym"] for p in open_pos}
            cands = sorted([x for x in by_entry[to] if x["sym"] not in held], key=lambda x: -x["run"])
            if cands:
                spend = min(cash_avail / (1 + COST / 2), nav / MAX_OPEN)
                if spend >= 5000:
                    x = cands[0]; qty = spend / x["ep"]
                    cash_avail -= spend + spend * COST / 2
                    open_pos.append(dict(sym=x["sym"], ep=x["ep"], qty=qty, xo=x["xo"], xp=x["xp"]))
                else: blocked += 1
        want_buffer = (len(open_pos) < MAX_OPEN) and (gate if gate_aware else True)
        target = (nav / MAX_OPEN) * 1.002 if want_buffer else 0.0
        if cash_avail > target:
            liquid_pending += cash_avail - target; cash_avail = target
        elif cash_avail < target and liquid > 0:
            red = min(liquid, target - cash_avail); liquid -= red; cash_pending += red
        curve.append((d, nav)); prev = d
    name = "D_gate_aware_buffer" if gate_aware else "C_realistic_buffer"
    return metrics(curve, dict(variant=name, interest_L=round(interest / 1e5, 1), blocked=blocked))

rows = []; curves = {}
for fn in [sim_A, sim_B, lambda: sim_CD(False), lambda: sim_CD(True)]:
    r, cur = fn(); rows.append(r); curves[r["variant"]] = cur
    log(f"done {r}")
sm = pd.DataFrame(rows)[["variant", "cagr", "sharpe", "maxdd", "calmar", "final_L", "interest_L", "blocked"]]
sm.to_csv(os.path.join(RES, "g5b_cash_ledger.csv"), index=False)
for v, cur in curves.items():
    cur.to_csv(os.path.join(RES, f"g5b_curve_{v}.csv"))
log("=== G5b cash-ledger variants (winner book: Donchian-20 + gate + 8 slots + 1/day) ===")
print(sm.to_string(index=False), flush=True)
log("DONE")
