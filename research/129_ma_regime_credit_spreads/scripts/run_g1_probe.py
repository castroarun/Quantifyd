#!/usr/bin/env python3
"""research/129 G1 — price-only kill test for MA/EMA-regime credit spreads.
What a credit spread needs: the regime must shift the OPPOSITE-side tail prob
over the 24-session hold. Bull state -> P(fwd < -2.5%) must DROP (bull put
spread survives); bear state -> P(fwd > +2.5%) must DROP (bear call spread
survives). Report vs unconditional + per-year stability."""
import math, sqlite3, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "89_short_monthly_straddle" / "scripts"))
import engine as E

H = 24              # forward sessions ~ the 45->21 calendar window
THR = 0.025         # the +/-2.5% short strike
START = "2016-01-01"

def _rsi(c, n=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / l)

def _stoch_k(d, n=14, sm=3):
    lo = d["low"].rolling(n).min(); hi = d["high"].rolling(n).max()
    return (100 * (d["close"] - lo) / (hi - lo)).rolling(sm).mean()

# each signal takes the daily OHLC frame, returns the BULL-state boolean series
SIGS = {
    "SMA100":   lambda d: d["close"] > d["close"].rolling(100).mean(),
    "SMA150":   lambda d: d["close"] > d["close"].rolling(150).mean(),
    "SMA200":   lambda d: d["close"] > d["close"].rolling(200).mean(),
    "EMA100":   lambda d: d["close"] > d["close"].ewm(span=100, adjust=False).mean(),
    "EMA150":   lambda d: d["close"] > d["close"].ewm(span=150, adjust=False).mean(),
    "EMA200":   lambda d: d["close"] > d["close"].ewm(span=200, adjust=False).mean(),
    "EMA20>50": lambda d: d["close"].ewm(span=20, adjust=False).mean() > d["close"].ewm(span=50, adjust=False).mean(),
    "EMA10>30": lambda d: d["close"].ewm(span=10, adjust=False).mean() > d["close"].ewm(span=30, adjust=False).mean(),
    "EMA50>100": lambda d: d["close"].ewm(span=50, adjust=False).mean() > d["close"].ewm(span=100, adjust=False).mean(),
    "RSI>50":   lambda d: _rsi(d["close"]) > 50,
    "RSI>60":   lambda d: _rsi(d["close"]) > 60,       # BEAR side = RSI<60 (incl. mid)
    "RSI>40":   lambda d: _rsi(d["close"]) > 40,       # BEAR = RSI<40 (oversold)
    "STOCH_K>D": lambda d: _stoch_k(d) > _stoch_k(d).rolling(3).mean(),
    "STOCH>80":  lambda d: _stoch_k(d) > 80,
    "STOCH>20":  lambda d: _stoch_k(d) > 20,           # BEAR = %K<20 (oversold)
}

def main():
    conn = sqlite3.connect(E.db_path())
    syms = [r[0] for r in conn.execute(
        "SELECT symbol, COUNT(*) c FROM nse_options_bhav "
        "WHERE symbol NOT IN ('NIFTY','BANKNIFTY') GROUP BY symbol HAVING c>500 ORDER BY symbol")]
    frames = []
    for s in syms:
        d = E.load_daily(s, conn)
        if len(d) < 300:
            continue
        c = d["close"]
        lr = np.log(c / c.shift())
        c_clean = c[lr.abs().fillna(0) <= 0.25]     # drop split-gap days from fwd calc
        fwd = c.shift(-H) / c - 1
        f = pd.DataFrame({"fwd": fwd, "year": c.index.year})
        for name, fn in SIGS.items():
            f[name] = fn(d)
        f = f[(f.index >= START)].dropna(subset=["fwd"])
        f = f[f["fwd"].abs() < 0.60]                # winsorize split artifacts
        f["symbol"] = s
        frames.append(f)
    df = pd.concat(frames)
    print(f"{len(syms)} symbols, {len(df):,} stock-days {START}->, H={H} sessions, thr ±{THR*100:.1f}%\n")

    u_dn = (df["fwd"] < -THR).mean()
    u_up = (df["fwd"] > THR).mean()
    print(f"UNCONDITIONAL: P(fwd<-2.5%)={u_dn*100:.1f}%   P(fwd>+2.5%)={u_up*100:.1f}%   "
          f"mean fwd={df['fwd'].mean()*100:+.2f}%\n")
    print(f"{'signal':10s} {'state':5s} {'n':>8s} {'meanFwd':>8s} {'P(<-2.5%)':>10s} {'P(>+2.5%)':>10s} "
          f"{'tail vs uncond':>15s} {'t(diff)':>8s}")
    rows = []
    for name in SIGS:
        for state, lab in [(True, "BULL"), (False, "BEAR")]:
            d = df[df[name] == state]
            if len(d) < 500:
                continue
            p_dn = (d["fwd"] < -THR).mean()
            p_up = (d["fwd"] > THR).mean()
            # the tail the spread SELLS: bull sells the downside, bear sells the upside
            rel = (p_dn / u_dn - 1) if state else (p_up / u_up - 1)
            other = df[df[name] != state]["fwd"]
            t = ((d["fwd"].mean() - other.mean()) /
                 math.sqrt(d["fwd"].var()/len(d) + other.var()/len(other))) / math.sqrt(H)
            print(f"{name:10s} {lab:5s} {len(d):8,} {d['fwd'].mean()*100:+7.2f}% "
                  f"{p_dn*100:9.1f}% {p_up*100:9.1f}% {rel*100:+14.1f}% {t:+8.2f}")
            rows.append((name, lab, rel, d))
    print("\n(negative 'tail vs uncond' = the sold tail SHRINKS in-state = good for the spread;")
    print(" t(diff) is Newey-West-downscaled by sqrt(H) for overlapping windows)\n")

    print("--- per-year stability of the BEST tail-shrinkers (need same sign every year) ---")
    best = sorted([r for r in rows if r[2] < 0], key=lambda r: r[2])[:4]
    for name, lab, rel, d in best:
        tail = "fwd < -2.5%" if lab == "BULL" else "fwd > +2.5%"
        yr = []
        for y in range(2016, 2027):
            dy = d[d["year"] == y]
            uy = df[df["year"] == y]
            if len(dy) < 300 or len(uy) < 300:
                continue
            py = (dy["fwd"] < -THR).mean() if lab == "BULL" else (dy["fwd"] > THR).mean()
            uu = (uy["fwd"] < -THR).mean() if lab == "BULL" else (uy["fwd"] > THR).mean()
            yr.append((y, (py/uu - 1) * 100 if uu > 0 else float("nan")))
        shrinks = sum(1 for _, v in yr if v < 0)
        print(f"{name} {lab} (sold tail {tail}): overall {rel*100:+.1f}%  per-year "
              + " ".join(f"{y}:{v:+.0f}%" for y, v in yr)
              + f"   -> shrinks in {shrinks}/{len(yr)} years")

if __name__ == "__main__":
    import logging; logging.disable(logging.WARNING)
    t0 = time.time(); main(); print(f"\n({time.time()-t0:.0f}s)")
