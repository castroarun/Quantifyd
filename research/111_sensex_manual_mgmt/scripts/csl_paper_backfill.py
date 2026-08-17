"""research/111 - CSL PAPER BOOKS BACKFILL: replay all 5 paper-book configs over every
recorded day in options_data.db (~3-sec chain snapshots) with the SAME mechanic as the
live executor (combined-SL, 2-snap dwell -> exit at next snap; 'none' -> 50% backstop),
emitting per-day records with source=BACKTEST and ~1-min intraday P&L series so the
/app/straddles day-curves grid has history before live paper recording starts 14-AUG.
Days already present in csl_paper_state.json (live PAPER/REAL records) are SKIPPED —
live data always wins; this file only fills the gaps. Safe to re-run daily (15:40 cron)."""
import json, os, sqlite3
from bisect import bisect_right
from datetime import datetime, timedelta

Q = "/home/arun/quantifyd"
DB = Q + "/backtest_data/options_data.db"
CONFIG = Q + "/backtest_data/csl_paper_config.json"
STATE = Q + "/backtest_data/csl_paper_state.json"
OUTS = [Q + "/static/app/csl_paper_backfill.json", Q + "/frontend/public/csl_paper_backfill.json"]
COST = 160
BACKSTOP = 50  # % for SL 'none'

NIFTY = {"sym": "NIFTY", "step": 50}
SENSEX = {"sym": "SENSEX", "step": 100}
BOOKS = {
    "CSL_TIMEB_NIFTY": {**NIFTY, "lots": 8, "qty": 520},
    "CSL_TIMEB_SENSEX": {**SENSEX, "lots": 6, "qty": 120},
    "NAS_COMB20": {**NIFTY, "lots": 2, "qty": 130},
    "CSL30F_NIFTY": {**NIFTY, "lots": 2, "qty": 130},
    "CSL30F_SENSEX": {**SENSEX, "lots": 3, "qty": 60},
}

cfg = json.load(open(CONFIG))
try:
    live = {(r["day"], r.get("book") or r.get("sym")) for r in json.load(open(STATE)).get("records", [])}
except Exception:
    live = set()

oc = sqlite3.connect(DB)

def tdte(E, day):
    dd = datetime.strptime(day, "%Y-%m-%d").date(); ed = datetime.strptime(E, "%Y-%m-%d").date()
    n, cur = 0, dd + timedelta(days=1)
    while cur <= ed:
        if cur.weekday() < 5: n += 1
        cur += timedelta(days=1)
    return n

records = []
for SYM in ("NIFTY", "SENSEX"):
    step = 50 if SYM == "NIFTY" else 100
    days = [r[0] for r in oc.execute(
        "SELECT DISTINCT substr(snapshot_time,1,10) FROM underlying_spot WHERE symbol=? AND spot_price>0 ORDER BY 1", (SYM,))]
    books = [bk for bk, B in BOOKS.items() if B["sym"] == SYM]
    print("START %s days=%d books=%s" % (SYM, len(days), books), flush=True)
    for i, day in enumerate(days):
        exps = sorted({r[0] for r in oc.execute(
            "SELECT DISTINCT expiry_date FROM option_chain WHERE symbol=? AND substr(snapshot_time,1,10)=? AND expiry_date>=?",
            (SYM, day, day))})
        if not exps: continue
        E = exps[0]; k = tdte(E, day)
        sp = [(r[0][11:19], float(r[1])) for r in oc.execute(
            "SELECT snapshot_time,spot_price FROM underlying_spot WHERE symbol=? AND substr(snapshot_time,1,10)=? AND spot_price>0 ORDER BY snapshot_time", (SYM, day))]
        if not sp: continue
        st = [a for a, _ in sp]
        legcache = {}
        def leg(strike, ty):
            key = (strike, ty)
            if key not in legcache:
                legcache[key] = [(r[0][11:19], float(r[1])) for r in oc.execute(
                    "SELECT snapshot_time,ltp FROM option_chain WHERE symbol=? AND expiry_date=? AND strike=? AND instrument_type=? AND substr(snapshot_time,1,10)=? AND ltp>0 ORDER BY snapshot_time",
                    (SYM, E, strike, ty, day))]
            return legcache[key]
        for bk in books:
            if (day, bk) in live: continue          # live paper/real record exists -> never overwrite
            B = BOOKS[bk]
            c = (cfg["books"].get(bk) or {}).get(str(k))
            if not c: continue                       # book doesn't trade this DTE
            ent_t, ex_t = c["entry"], c["exit"]
            sl = BACKSTOP if c["sl"] == "none" else float(c["sl"])
            j = bisect_right(st, ent_t + ":59")
            if not j: continue
            if st[j - 1] < (datetime.strptime(ent_t, "%H:%M") - timedelta(minutes=5)).strftime("%H:%M"):
                continue                             # no spot near entry minute -> book skips the day
            K = round(sp[j - 1][1] / step) * step
            ce, pe = leg(K, "CE"), leg(K, "PE")
            if not (ce and pe): continue
            pk_ = [a for a, _ in pe]; pv = [v for _, v in pe]
            path = []
            for t, cv in ce:
                if t < ent_t + ":00" or t > "15:25:59": continue
                jp = bisect_right(pk_, t)
                if jp: path.append((t, cv + pv[jp - 1]))
            if len(path) < 10: continue              # sparse coverage -> skip (resolution-ladder rule)
            ent = path[0][1]
            thr = ent * (1 + sl / 100.0)
            streak = 0; reason = None; exit_i = None
            for m in range(1, len(path)):
                t, comb = path[m]
                if t > ex_t + ":59":
                    reason, exit_i = "TIME_EXIT", m; break
                if comb >= thr:
                    streak += 1
                    if streak >= 2:
                        reason, exit_i = "SL_DWELL", min(m + 1, len(path) - 1); break
                else:
                    streak = 0
            if exit_i is None:
                reason, exit_i = "EOD_FORCE" if ex_t >= "15:20" else "TIME_EXIT", len(path) - 1
            exit_t, exit_comb = path[exit_i]
            pnl = round((ent - exit_comb) * B["qty"] - COST)
            series, last_min = [], None
            for t, comb in path[:exit_i + 1]:
                if t[:5] != last_min:
                    series.append([t[:5], round((ent - comb) * B["qty"])]); last_min = t[:5]
            series.append([exit_t[:5], round((ent - exit_comb) * B["qty"])])
            records.append({"day": day, "book": bk, "sym": SYM, "dte": k,
                            "cfg": "%s->%s SL%s" % (ent_t, ex_t, c["sl"]), "strike": K, "expiry": E,
                            "credit": round(ent, 2), "entry_ts": path[0][0], "exit_ts": exit_t,
                            "exit_comb": round(exit_comb, 2), "reason": reason, "pnl": pnl,
                            "series": series, "lots": B["lots"], "qty": B["qty"], "source": "BACKTEST"})
        if i % 15 == 0: print("%s %d/%d recs=%d" % (SYM, i, len(days), len(records)), flush=True)

records.sort(key=lambda r: (r["day"], r["book"]))
meta = {}
for bk in BOOKS:
    rs = [r for r in records if r["book"] == bk]
    if rs:
        meta[bk] = {"n": len(rs), "from": rs[0]["day"], "to": rs[-1]["day"],
                    "net": sum(r["pnl"] for r in rs), "lots": BOOKS[bk]["lots"]}
out = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "source": "BACKTEST",
       "note": "recorded-chain replay of the 5 frozen paper-book configs; live records always take precedence",
       "meta": meta, "records": records}
for p in OUTS:
    try: json.dump(out, open(p, "w"))
    except Exception as e: print("write %s: %s" % (p, e))
print("\nDONE records=%d" % len(records))
for bk, m in meta.items():
    print("  %-14s n=%3d %s->%s net %+9d (%d lots)" % (bk, m["n"], m["from"], m["to"], m["net"], m["lots"]))
