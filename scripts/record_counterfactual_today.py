# -*- coding: utf-8 -*-
"""One-shot: record the 2026-08-03 kill-flag MISS as counterfactual paper trades. The whole 09:16 book
(NAS 916 ATM family + SENSEX) was blocked by a stale kill flag, so today has no record. This computes
the HOLD-to-15:15 straddle P&L (accurate on a calm day — today NIFTY moved ~+0.1%, no 30% SL / 0.4%
move) and inserts one tagged trade per system so the day isn't a blind spot.
Run AFTER close (>=15:15). Date-guarded to 2026-08-03 only. If the day was NOT calm, it still records
hold-to-EOD but tags HOLDAPPROX (real management would differ) and logs a warning."""
import sqlite3, sys, datetime
sys.path.insert(0, "/home/arun/quantifyd")

TARGET = "2026-08-03"
now = datetime.datetime.now()
if now.strftime("%Y-%m-%d") != TARGET:
    print("[%s] not target date %s — no-op." % (now, TARGET)); sys.exit(0)
if now.strftime("%H:%M") < "15:15":
    print("[%s] before 15:15 — need the full day; skipping (cron reruns)." % now); sys.exit(0)

DB = "backtest_data/options_data.db"
oc = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)

INDEXES = [
    dict(sym="NIFTY", step=50, qty=130, brok=160, dbs=["nas_916_atm", "nas_916_atm2", "nas_916_atm4"]),
    dict(sym="SENSEX", step=100, qty=40, brok=160, dbs=["sensex_atm", "sensex_atm2", "sensex_atm4"]),
]


def spot_series(sym):
    d = {}
    for st, sp in oc.execute("SELECT snapshot_time,spot_price FROM underlying_spot WHERE symbol=? AND "
                             "substr(snapshot_time,1,10)=? AND spot_price>0", (sym, TARGET)):
        d[st[11:16]] = sp
    return d


def val_at(series, hhmm, last=False):
    ks = sorted(k for k in series if series[k] and series[k] > 0)
    if not ks:
        return None
    if last:
        cand = [k for k in ks if k <= hhmm]
        return series[cand[-1]] if cand else None
    cand = [k for k in ks if k >= hhmm]
    return series[cand[0]] if cand else None


for ix in INDEXES:
    sym = ix["sym"]
    sp = spot_series(sym)
    s0 = val_at(sp, "09:16")
    if not s0:
        print("[%s] no %s spot today — skip." % (now, sym)); continue
    atm = round(s0 / ix["step"]) * ix["step"]
    exps = sorted({r[0][:10] for r in oc.execute("SELECT DISTINCT expiry_date FROM option_chain WHERE "
                   "symbol=? AND substr(snapshot_time,1,10)=?", (sym, TARGET))})
    exp = [e for e in exps if e >= TARGET][0]

    def leg(typ):
        d = {}
        for st, ltp in oc.execute("SELECT snapshot_time,ltp FROM option_chain WHERE symbol=? AND expiry_date LIKE ? "
                                  "AND strike=? AND instrument_type=? AND substr(snapshot_time,1,10)=?",
                                  (sym, exp + "%", atm, typ, TARGET)):
            d[st[11:16]] = ltp or 0
        return d
    ce, pe = leg("CE"), leg("PE")
    ce_e, pe_e = val_at(ce, "09:16"), val_at(pe, "09:16")
    ce_x, pe_x = val_at(ce, "15:15", last=True), val_at(pe, "15:15", last=True)
    if not all([ce_e, pe_e, ce_x, pe_x]):
        print("[%s] %s missing premiums (ce_e=%s pe_e=%s ce_x=%s pe_x=%s) — skip." % (now, sym, ce_e, pe_e, ce_x, pe_x)); continue

    ce_max = max([v for v in ce.values() if v > 0] or [0]); pe_max = max([v for v in pe.values() if v > 0] or [0])
    smax = max(sp.values()); smin = min(sp.values())
    calm = (ce_max < 1.3 * ce_e) and (pe_max < 1.3 * pe_e) and (abs(smax - s0) / s0 < 0.004) and (abs(smin - s0) / s0 < 0.004)
    tag = "COUNTERFACTUAL_KILLFLAG" if calm else "COUNTERFACTUAL_KILLFLAG_HOLDAPPROX"
    if not calm:
        print("[%s] %s NOT calm (ce_max %.1f/%.1f pe_max %.1f/%.1f move %.2f%%) — hold-to-EOD is approximate." %
              (now, sym, ce_max, 1.3 * ce_e, pe_max, 1.3 * pe_e, max(abs(smax - s0), abs(smin - s0)) / s0 * 100))

    collected = round(ce_e + pe_e, 2); paid = round(ce_x + pe_x, 2)
    qty = ix["qty"]
    gross = round((collected - paid) * qty, 2); net = round(gross - ix["brok"], 2)
    print("[%s] %s hold-to-EOD: ATM %d exp %s  CE %.1f->%.1f  PE %.1f->%.1f  credit %.1f  gross %+d net %+d  [%s]" %
          (now, sym, atm, exp, ce_e, ce_x, pe_e, pe_x, collected, gross, net, "CALM" if calm else "APPROX"))

    for db in ix["dbs"]:
        con = sqlite3.connect("backtest_data/%s_trading.db" % db)
        # skip if already recorded (idempotent rerun)
        ex = con.execute("SELECT COUNT(*) FROM nas_atm_trades WHERE trade_date=? AND exit_reason LIKE 'COUNTERFACTUAL_KILLFLAG%'", (TARGET,)).fetchone()[0]
        if ex:
            print("     %s already has counterfactual row — skip." % db); con.close(); continue
        con.execute(
            "INSERT INTO nas_atm_trades (strangle_id, trade_date, entry_time, exit_time, spot_at_entry, spot_at_exit, "
            "call_strike, put_strike, call_entry_premium, put_entry_premium, call_exit_premium, put_exit_premium, "
            "total_premium_collected, total_premium_paid, gross_pnl, net_pnl, lots, adjustments, exit_reason, expiry_date, notes, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (99999, TARGET, TARGET + "T09:16:00", TARGET + "T15:15:00", s0, val_at(sp, "15:15", last=True),
             atm, atm, ce_e, pe_e, ce_x, pe_x, collected, paid, gross, net, 2, 0, tag, exp,
             "kill-flag miss 2026-08-03; no live/paper entry occurred; hold-to-EOD counterfactual", now.isoformat()))
        con.commit(); con.close()
        print("     recorded counterfactual into %s (net %+d)" % (db, net))
print("[%s] done." % now)
