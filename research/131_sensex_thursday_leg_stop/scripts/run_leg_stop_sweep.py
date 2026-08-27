#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/131 - SENSEX DTE0 WIDE per-leg stop sweep on the real 1-minute chain.

The question research/114 never asked: a 30% leg stop is a noise-harvester on expiry day,
but is there a level WIDE enough to cap a genuine breakaway leg while leaving the decay
edge intact? And when a leg does stop, what happens to the survivor?

Grid (pre-registered in SENSEX_DTE0_WIDE_LEG_STOP_1MIN_SWEEP_STATUS.md):
    level    : HOLD | LEG30/40/50/60/75/100 (% of leg entry) | RUP1500/2500/4000/6000/8000 (Rs/lot)
    survivor : SBOTH (close both) | SHOLD (hold survivor to EOD) | STRAIL (live ST(7,3) ceiling)
    outer    : STANDALONE | VENUE (deployed DTE0 book stop -3000/lot, TP +4000/lot)
    entryset : A0920 (all 17 DTE0 days, uniform 09:20 entry)
               B0916 (the 12 days that actually have a 09:16 print - live-exact, r/114-comparable)

READ-ONLY on the DB. Writes results/leg_stop_detail.csv + results/run.log.
"""
import sqlite3, csv, os, sys
from datetime import date

Q = "/home/arun/quantifyd/"
CHAIN = Q + "backtest_data/options_data.db"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DETAIL = os.path.join(RES, "leg_stop_detail.csv")
LOG = os.path.join(RES, "run.log")

LOT = 20            # SENSEX lot. option_chain.lot_size is WRONG (memory: r/119 data traps).
STEP = 100
EOD_M = 15 * 60 + 15

# ---------------------------------------------------------------------------
# MEASURED, outcome-aware cost model - research/122 stage_a_alldays.cost_per_lot,
# decomposed to PER LEG so legs exiting for different reasons are charged differently.
# 443 real live leg-sides (Kite fill vs chain LTP, same minute), 2026-08-25.
SLIP_ENTRY = 0.0
SLIP_TIME = 0.178          # time / EOD exit
SLIP_STOP = 6.548          # forced / stop exit  <- the whole asymmetry lives here
NLOTS_REF = 10
FORCED = {"LEG_SL", "SL_PAIR", "BOOK_STOP", "BOOK_TP", "TRAIL"}

# retired r/114 heavy assumption, kept only for the reconciliation cell
R114_SLIP = 1.0            # pts per leg-side
R114_CHG = 30.0            # Rs per leg-side per lot


def leg_cost(entry_p, exit_p, reason):
    """Rs per lot, one leg, full round trip."""
    sell = entry_p * LOT
    buy = exit_p * LOT
    tot = sell + buy
    brok = 40.0 / NLOTS_REF          # Rs20 x 2 orders for this leg, over the study lot count
    stt = 0.001 * sell               # STT on the sell side of an option
    txn = 0.0003503 * tot
    ipft = 0.0000050 * tot
    sebi = 0.0000010 * tot
    stamp = 0.00003 * buy
    gst = 0.18 * (brok + txn + ipft + sebi)
    slip = SLIP_ENTRY + (SLIP_STOP if reason in FORCED else SLIP_TIME)
    return brok + stt + txn + ipft + sebi + stamp + gst + slip * LOT


def leg_cost_r114(entry_p, exit_p, reason):
    """r/114's model, per leg: 1.0 pt slip per leg-side + Rs30 per leg-side per lot."""
    return 2 * R114_SLIP * LOT + 2 * R114_CHG


# ---------------------------------------------------------------------------
# Verbatim port of NasAtm4Executor.compute_short_trailing_stop (services/nas_atm4_executor.py
# lines 101-149). Ported rather than imported so this study cannot touch a live service module.
def compute_short_trailing_stop(candles, period=7, multiplier=3.0):
    n = len(candles)
    if n < period + 1:
        return None, None
    high = [c["high"] for c in candles]
    low = [c["low"] for c in candles]
    close = [c["close"] for c in candles]
    tr = [0.0] * n
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = [0.0] * n
    atr[period - 1] = sum(tr[:period]) / period
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    hl2 = [(high[i] + low[i]) / 2.0 for i in range(n)]
    upper = [hl2[i] + multiplier * atr[i] for i in range(n)]
    stop = [0.0] * n
    stop[period - 1] = upper[period - 1]
    for i in range(period, n):
        cand = min(upper[i], stop[i - 1])       # ratchet DOWN only
        stop[i] = upper[i] if close[i] > cand else cand
    return round(float(stop[-1]), 2), False


ATR_PERIOD, MULT, CONFIRM = 7, 3.0, 2   # CONFIRM=2 x 1-min prints ~= the live 6 polls x 10s


def log(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
    print(m, flush=True)


def hm2m(h):
    return int(h[:2]) * 60 + int(h[3:5])


def m2hm(m):
    return "%02d:%02d" % (m // 60, m % 60)


# ---------------------------------------------------------------------------
def dte0_days(c):
    """Every recorded SENSEX day whose FRONT expiry is the day itself. DTE-era safe:
    SENSEX expiry moved Fri -> Tue -> Thu, so weekday selection (r/114) mislabels."""
    days = [r[0] for r in c.execute(
        "SELECT DISTINCT substr(snapshot_time,1,10) d FROM download_log WHERE symbol='SENSEX' "
        "ORDER BY d")]
    out = []
    for d in days:
        if date.fromisoformat(d).weekday() >= 5:
            continue
        r = c.execute("SELECT MIN(expiry_date) FROM option_chain WHERE symbol='SENSEX' "
                      "AND snapshot_time>=? AND snapshot_time<? AND ltp IS NOT NULL "
                      "AND expiry_date>=?", (d, d + "z", d)).fetchone()
        if r and r[0] == d:
            out.append(d)
    return out


def load_day(c, day):
    rows = c.execute(
        "SELECT snapshot_time, expiry_date, strike, instrument_type, ltp, underlying_spot "
        "FROM option_chain WHERE symbol='SENSEX' AND snapshot_time>=? AND snapshot_time<? "
        "AND ltp IS NOT NULL", (day, day + "z")).fetchall()
    if not rows:
        return None, "no rows"
    if max(r[0] for r in rows)[11:16] < "15:15":
        return None, "partial session"
    fexp = min(e for (_, e, _, _, _, _) in rows if e and e >= day)
    spot, chain = {}, {}
    for st, e, k, it, ltp, sp in rows:
        mi = hm2m(st[11:16])
        if sp and mi not in spot:
            spot[mi] = sp
        if e != fexp:
            continue
        chain.setdefault(mi, {}).setdefault(k, {})[it] = ltp
    if len(set(spot.values())) < 50:
        return None, "frozen chain (holiday guard)"
    ch = {}
    for mi, ks in chain.items():
        d = {k: (v["CE"], v["PE"]) for k, v in ks.items() if "CE" in v and "PE" in v}
        if d:
            ch[mi] = d
    if len(ch) < 200:
        return None, "thin (%d mins)" % len(ch)
    return (fexp, spot, ch), None


def build_series(spot, ch, entry_hm):
    """(K, ce0, pe0, [(minute, ce, pe)]) from entry_hm to 15:15 on the ATM strike."""
    e_target = hm2m(entry_hm)
    cands = sorted(m for m in ch if m >= e_target)
    if not cands:
        return None, "no minute >= %s" % entry_hm
    em = cands[0]
    if em > e_target + 2:
        return None, "no print near %s (first %s)" % (entry_hm, m2hm(em))
    sp = [spot[m] for m in sorted(spot) if m <= em]
    if not sp:
        return None, "no spot at/before %s" % m2hm(em)
    K = round(sp[-1] / STEP) * STEP
    if K not in ch[em]:
        return None, "ATM %d not recorded at %s" % (K, m2hm(em))
    ce0, pe0 = ch[em][K]
    ser = [(m, ch[m][K][0], ch[m][K][1]) for m in sorted(ch) if em <= m <= EOD_M and K in ch[m]]
    if len(ser) < 200:
        return None, "series too short (%d)" % len(ser)
    return (K, ce0, pe0, ser), None


def ceiling_series(ser, idx):
    """Per-minute ST(7,3) ceiling for one leg, exactly as the live trail computes it:
    5-min premium candles, recomputed on BAR CLOSE, ratcheting upper band, seeded from entry."""
    bars, order = {}, []
    for (m, ce, pe) in ser:
        p = ce if idx == 0 else pe
        b = (m // 5) * 5
        if b not in bars:
            bars[b] = {"open": p, "high": p, "low": p, "close": p}
            order.append(b)
        else:
            d = bars[b]
            d["high"] = max(d["high"], p)
            d["low"] = min(d["low"], p)
            d["close"] = p
    # ceil_before[i] = ceiling available while bar i is still forming (uses bars 0..i-1)
    ceil_before = {}
    for i, b in enumerate(order):
        ceil_before[b] = compute_short_trailing_stop([bars[x] for x in order[:i]],
                                                     ATR_PERIOD, MULT)[0] if i >= ATR_PERIOD + 1 else None
    return {m: ceil_before[(m // 5) * 5] for (m, _, _) in ser}


# ---------------------------------------------------------------------------
LEVELS = [("HOLD", None, None),
          ("LEG30", 0.30, None), ("LEG40", 0.40, None), ("LEG50", 0.50, None),
          ("LEG60", 0.60, None), ("LEG75", 0.75, None), ("LEG100", 1.00, None),
          ("RUP1500", None, 1500.0), ("RUP2500", None, 2500.0), ("RUP4000", None, 4000.0),
          ("RUP6000", None, 6000.0), ("RUP8000", None, 8000.0)]
SURVS = ["SBOTH", "SHOLD", "STRAIL"]
OUTERS = ["STANDALONE", "VENUE"]
BOOK_STOP = -3000.0      # services/nas_portfolio_stop.py, SENSEX DTE0
BOOK_TP = 4000.0


def replay(ser, ce0, pe0, ceils, pct, rup, surv, outer):
    ent = {"CE": ce0, "PE": pe0}
    st = {L: {"open": True, "px": None, "reason": None, "m": None} for L in ("CE", "PE")}
    breach = 0
    armed = None

    def close(L, m, p, reason):
        st[L].update(open=False, px=p, reason=reason, m=m)

    for (m, ce, pe) in ser:
        px = {"CE": ce, "PE": pe}
        openL = [L for L in ("CE", "PE") if st[L]["open"]]
        if not openL:
            break
        tot = sum((ent[L] - (px[L] if st[L]["open"] else st[L]["px"])) * LOT for L in ("CE", "PE"))
        if outer == "VENUE":
            if tot <= BOOK_STOP:
                for L in openL:
                    close(L, m, px[L], "BOOK_STOP")
                break
            if tot >= BOOK_TP:
                for L in openL:
                    close(L, m, px[L], "BOOK_TP")
                break
        if len(openL) == 2 and (pct is not None or rup is not None):
            hit = []
            for L in ("CE", "PE"):
                if pct is not None and px[L] >= ent[L] * (1 + pct):
                    hit.append((px[L] / ent[L], L))
                elif rup is not None and (px[L] - ent[L]) * LOT >= rup:
                    hit.append((px[L] / ent[L], L))
            if hit:
                hit.sort(reverse=True)
                B = hit[0][1]
                S = "PE" if B == "CE" else "CE"
                close(B, m, px[B], "LEG_SL")
                if surv == "SBOTH":
                    close(S, m, px[S], "SL_PAIR")
                    break
                armed = S if surv == "STRAIL" else None
                breach = 0
                continue
        if armed and st[armed]["open"]:
            c = ceils[armed].get(m)
            if c is not None:
                c = min(c, ent[armed])          # clamp to breakeven (live behaviour)
                if px[armed] > c:
                    breach += 1
                    if breach >= CONFIRM:
                        close(armed, m, px[armed], "TRAIL")
                        break
                else:
                    breach = 0
    lm, lce, lpe = ser[-1]
    for L, p in (("CE", lce), ("PE", lpe)):
        if st[L]["open"]:
            close(L, lm, p, "EOD")
    return ent, st


def row_of(day, K, ent, st, cost_fn):
    gross = sum((ent[L] - st[L]["px"]) * LOT for L in ("CE", "PE"))
    cost = sum(cost_fn(ent[L], st[L]["px"], st[L]["reason"]) for L in ("CE", "PE"))
    fired = any(st[L]["reason"] == "LEG_SL" for L in ("CE", "PE"))
    fire_m = min([st[L]["m"] for L in ("CE", "PE") if st[L]["reason"] == "LEG_SL"], default=None)
    return dict(day=day, strike=K, credit=round(ent["CE"] + ent["PE"], 2),
                ce_exit=round(st["CE"]["px"], 2), pe_exit=round(st["PE"]["px"], 2),
                ce_reason=st["CE"]["reason"], pe_reason=st["PE"]["reason"],
                ce_exit_hm=m2hm(st["CE"]["m"]), pe_exit_hm=m2hm(st["PE"]["m"]),
                gross=round(gross, 1), cost=round(cost, 1), net=round(gross - cost, 1),
                leg_fired=int(fired), fire_hm=m2hm(fire_m) if fire_m else "")


FIELDS = ["entryset", "level", "surv", "outer", "costmodel", "day", "strike", "credit",
          "ce_exit", "pe_exit", "ce_reason", "pe_reason", "ce_exit_hm", "pe_exit_hm",
          "gross", "cost", "net", "leg_fired", "fire_hm"]


def main():
    os.makedirs(RES, exist_ok=True)
    open(LOG, "w").close()
    c = sqlite3.connect("file:%s?mode=ro" % CHAIN, uri=True)
    days = dte0_days(c)
    log("SENSEX DTE0 sessions recorded: %d (%s .. %s)" % (len(days), days[0], days[-1]))

    loaded = {}
    for d in days:
        got, why = load_day(c, d)
        if not got:
            log("  %s REJECT: %s" % (d, why))
            continue
        loaded[d] = got
    log("usable after guards: %d" % len(loaded))

    out = open(DETAIL, "w", newline="")
    w = csv.DictWriter(out, fieldnames=FIELDS)
    w.writeheader()
    nrows = 0
    for eset, ehm in (("A0920", "09:20"), ("B0916", "09:16")):
        used = []
        for d in sorted(loaded):
            fexp, spot, ch = loaded[d]
            got, why = build_series(spot, ch, ehm)
            if not got:
                log("  [%s] %s skip: %s" % (eset, d, why))
                continue
            K, ce0, pe0, ser = got
            used.append(d)
            ceils = {"CE": ceiling_series(ser, 0), "PE": ceiling_series(ser, 1)}
            for lname, pct, rup in LEVELS:
                for surv in (["SBOTH"] if lname == "HOLD" else SURVS):
                    for outer in OUTERS:
                        ent, stt = replay(ser, ce0, pe0, ceils, pct, rup, surv, outer)
                        r = row_of(d, K, ent, stt, leg_cost)
                        r.update(entryset=eset, level=lname,
                                 surv="-" if lname == "HOLD" else surv,
                                 outer=outer, costmodel="MEASURED")
                        w.writerow(r)
                        nrows += 1
                        # r/114 cost model, reconciliation family (STANDALONE only)
                        if outer == "STANDALONE" and lname in ("HOLD", "LEG30"):
                            r2 = row_of(d, K, ent, stt, leg_cost_r114)
                            r2.update(entryset=eset, level=lname,
                                      surv="-" if lname == "HOLD" else surv,
                                      outer=outer, costmodel="R114COST")
                            w.writerow(r2)
                            nrows += 1
        log("[%s] sessions used: %d (%s)" % (eset, len(used), ", ".join(used)))
    out.close()
    log("wrote %d rows -> %s" % (nrows, DETAIL))


if __name__ == "__main__":
    main()
