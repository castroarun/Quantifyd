#!/usr/bin/env python3
"""
CPR-ST Morning Scalp — Phase 1: underlying signal-favourability (no options).

Classifies each NIFTY morning by the 1st 5-min candle vs the daily CPR, then
for SYS-A (directional) days measures whether price reaches the R1/S1 target
BEFORE the Supertrend flips against the trade. Also runs the narrow-CPR
volatility check that justifies the skip rule, and gap stats. Grouped by DTE.

Pure stdlib (sqlite3 + math + datetime). Reads market_data.db read-only.
Designed to run on the VPS per the vps-canonical rule.
"""
import sqlite3, math, csv, sys
from datetime import datetime, date
from collections import defaultdict

DB = "/home/arun/quantifyd/backtest_data/market_data.db"
SYM = "NIFTY50"

# ---- Phase-1 base params (Phase 2 will sweep these) ----
ST_PERIOD   = 10
ST_MULT     = 3.0
THR_NARROW  = 0.0010   # CPR width < 0.10% of spot -> skip (volatile day)
THR_FAR     = 0.0050   # |close-band|/spot > 0.50% -> too far -> SYS-B (gap/strangle)
EXIT_HHMM   = (15, 15) # intraday time-stop
EXPIRY_WD   = 3        # weekly expiry weekday: Thursday (Mon=0). Caveat: NSE shifted over 2024-25.

# ---------------------------------------------------------------------------
def load_5min(con):
    rows = con.execute(
        "SELECT date,open,high,low,close FROM market_data_unified "
        "WHERE symbol=? AND timeframe='5minute' ORDER BY date", (SYM,)).fetchall()
    out = []
    for d, o, h, l, c in rows:
        dt = datetime.strptime(d, "%Y-%m-%d %H:%M:%S")
        out.append((dt, float(o), float(h), float(l), float(c)))
    return out

def load_daily_cpr(con):
    """Return {trade_date: cpr_levels} where levels come from the PRIOR day."""
    rows = con.execute(
        "SELECT date,high,low,close FROM market_data_unified "
        "WHERE symbol=? AND timeframe='day' ORDER BY date", (SYM,)).fetchall()
    days = []
    for d, h, l, c in rows:
        dd = datetime.strptime(d.split()[0], "%Y-%m-%d").date()
        days.append((dd, float(h), float(l), float(c)))
    cpr = {}
    for i in range(1, len(days)):
        _, ph, pl, pc = days[i-1]
        P  = (ph + pl + pc) / 3.0
        BC = (ph + pl) / 2.0
        TC = 2*P - BC
        lo, hi = min(BC, TC), max(BC, TC)
        cpr[days[i][0]] = dict(P=P, BC=lo, TC=hi,
                               R1=2*P - pl, S1=2*P - ph,
                               width=abs(TC - BC))
    return cpr

# ---- Supertrend (continuous across the full 5-min series) ----
def supertrend(bars, period, mult):
    n = len(bars)
    tr = [0.0]*n
    for i in range(n):
        _, o, h, l, c = bars[i]
        if i == 0:
            tr[i] = h - l
        else:
            pc = bars[i-1][4]
            tr[i] = max(h - l, abs(h - pc), abs(l - pc))
    # Wilder ATR
    atr = [0.0]*n
    atr[period-1] = sum(tr[:period]) / period
    for i in range(period, n):
        atr[i] = (atr[i-1]*(period-1) + tr[i]) / period
    direction = [0]*n          # +1 up, -1 down
    st = [0.0]*n
    fub = flb = 0.0
    for i in range(n):
        _, o, h, l, c = bars[i]
        mid = (h + l) / 2.0
        if i < period:
            direction[i] = 1
            continue
        bub = mid + mult*atr[i]
        blb = mid - mult*atr[i]
        fub = bub if (bub < fub or bars[i-1][4] > fub or fub == 0) else fub
        flb = blb if (blb > flb or bars[i-1][4] < flb or flb == 0) else flb
        prev = direction[i-1] if direction[i-1] != 0 else 1
        if prev == 1:
            direction[i] = -1 if c < flb else 1
        else:
            direction[i] = 1 if c > fub else -1
        st[i] = flb if direction[i] == 1 else fub
    return direction

def dte_to_expiry(d):
    return (EXPIRY_WD - d.weekday()) % 7

def dte_bucket(n):
    return "0DTE" if n == 0 else "1" if n == 1 else "2" if n == 2 else "3+"

# ---------------------------------------------------------------------------
def main():
    con = sqlite3.connect(DB)
    bars = load_5min(con)
    cpr  = load_daily_cpr(con)
    direction = supertrend(bars, ST_PERIOD, ST_MULT)
    con.close()

    # group bar indices by trading date
    byday = defaultdict(list)
    for i, b in enumerate(bars):
        byday[b[0].date()].append(i)

    buckets = defaultdict(int)
    sysA = []            # per-trade records
    vol_by_cprtype = defaultdict(list)   # day range% by narrow/normal/wide
    gaps = []

    prev_close = None
    for d in sorted(byday):
        idx = byday[d]
        day_open  = bars[idx[0]][1]
        day_hi    = max(bars[i][2] for i in idx)
        day_lo    = min(bars[i][3] for i in idx)
        rng_pct   = (day_hi - day_lo) / day_open
        if prev_close is not None:
            gaps.append((day_open - prev_close) / prev_close)
        prev_close = bars[idx[-1]][4]

        lv = cpr.get(d)
        if lv is None:
            continue
        spot = day_open
        wpct = lv["width"] / spot
        cprtype = "narrow" if wpct < THR_NARROW else ("wide" if wpct > 0.005 else "normal")
        vol_by_cprtype[cprtype].append(rng_pct)

        # 1st 5-min candle = the 09:15 bar
        first = idx[0]
        fclose = bars[first][4]
        n_dte  = dte_to_expiry(d)
        bkt    = dte_bucket(n_dte)

        # ---- classify ----
        if wpct < THR_NARROW:
            buckets["skip_narrow"] += 1;  continue
        if lv["BC"] <= fclose <= lv["TC"]:
            buckets["skip_inside"] += 1;  continue

        if fclose > lv["TC"]:
            dist = (fclose - lv["TC"]) / spot
            side = "bull"
        else:
            dist = (lv["BC"] - fclose) / spot
            side = "bear"
        if dist > THR_FAR:
            buckets["sysB_gap_%s" % side] += 1;  continue

        buckets["sysA_%s" % side] += 1
        # ---- simulate SYS-A from entry bar (next bar after 1st candle) ----
        if len(idx) < 2:
            continue
        entry_i = idx[1]
        entry   = bars[entry_i][1]                # open of 09:20 bar
        target  = lv["R1"] if side == "bull" else lv["S1"]
        exit_reason, exit_px = "time", bars[idx[-1]][4]
        mfe = mae = 0.0
        for i in idx[1:]:
            _, o, h, l, c = bars[i]
            up = h - entry; dn = l - entry
            if side == "bull":
                mfe = max(mfe, up); mae = min(mae, dn)
            else:
                mfe = max(mfe, -dn); mae = min(mae, -up)
            t = bars[i][0].time()
            # target?
            if side == "bull" and h >= target:
                exit_reason, exit_px = "target", target; break
            if side == "bear" and l <= target:
                exit_reason, exit_px = "target", target; break
            # supertrend flip against trade?
            want = 1 if side == "bull" else -1
            if direction[i] != want and direction[i] != 0:
                exit_reason, exit_px = "st_flip", c; break
            if (t.hour, t.minute) >= EXIT_HHMM:
                exit_reason, exit_px = "time", c; break
        captured = (exit_px - entry) if side == "bull" else (entry - exit_px)
        sysA.append(dict(date=d, side=side, dte=n_dte, bkt=bkt,
                         entry=entry, target=target, exit_reason=exit_reason,
                         captured=captured, mfe=mfe, mae=mae,
                         dist_pct=dist*100, width_pct=wpct*100))
    report(buckets, sysA, vol_by_cprtype, gaps)

def pct(x): return f"{100*x:5.1f}%"

def report(buckets, sysA, vol_by_cprtype, gaps):
    L = []
    P = L.append
    total = sum(buckets.values())
    P("="*72); P("PHASE 1 — UNDERLYING SIGNAL-FAVOURABILITY"); P("="*72)
    P(f"\nClassified mornings: {total}\n")
    P("Day-type frequency:")
    for k in sorted(buckets):
        P(f"  {k:20s} {buckets[k]:4d}  ({100*buckets[k]/total:4.1f}%)")

    # narrow-CPR volatility check
    P("\nNarrow-CPR volatility check (avg intraday range as % of open):")
    for t in ("narrow", "normal", "wide"):
        v = vol_by_cprtype.get(t, [])
        if v:
            P(f"  {t:7s} n={len(v):4d}  avg_range={100*sum(v)/len(v):4.2f}%  "
              f"max={100*max(v):4.2f}%")
    P("  -> rule is justified if 'narrow' shows higher avg range than 'normal'.")

    # gap stats
    if gaps:
        ag = sorted(abs(g) for g in gaps)
        P(f"\nGap (|open-prevclose|/prevclose): median={100*ag[len(ag)//2]:.2f}%  "
          f"p90={100*ag[int(len(ag)*0.9)]:.2f}%  "
          f">0.5%: {100*sum(1 for g in ag if g>0.005)/len(ag):.0f}% of days")

    # SYS-A edge
    if sysA:
        P("\n" + "-"*72)
        P(f"SYS-A directional trades: {len(sysA)}")
        def stats(rows, label):
            if not rows: return
            n = len(rows)
            tgt = sum(1 for r in rows if r["exit_reason"]=="target")
            flip= sum(1 for r in rows if r["exit_reason"]=="st_flip")
            tim = sum(1 for r in rows if r["exit_reason"]=="time")
            avgc= sum(r["captured"] for r in rows)/n
            wins= sum(1 for r in rows if r["captured"]>0)
            avgmfe = sum(r["mfe"] for r in rows)/n
            avgmae = sum(r["mae"] for r in rows)/n
            P(f"\n  [{label}]  n={n}")
            P(f"    target-before-flip : {tgt:3d} ({100*tgt/n:4.1f}%)   "
              f"st_flip: {flip} ({100*flip/n:4.1f}%)   time: {tim} ({100*tim/n:4.1f}%)")
            P(f"    win-rate (pts>0)   : {100*wins/n:4.1f}%   "
              f"avg captured: {avgc:+6.1f} pts")
            P(f"    avg MFE: {avgmfe:+6.1f}   avg MAE: {avgmae:+6.1f}  "
              f"(favourable if |MFE|>>|MAE|)")
        stats(sysA, "ALL")
        stats([r for r in sysA if r["side"]=="bull"], "BULL (short PE)")
        stats([r for r in sysA if r["side"]=="bear"], "BEAR (short CE)")
        P("\n  By DTE:")
        for b in ("0DTE","1","2","3+"):
            stats([r for r in sysA if r["bkt"]==b], "DTE "+b)

        # dump CSV
        with open("sysA_trades.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sysA[0].keys()))
            w.writeheader(); w.writerows(sysA)
        P("\n  -> sysA_trades.csv written")
    print("\n".join(L))

if __name__ == "__main__":
    main()
