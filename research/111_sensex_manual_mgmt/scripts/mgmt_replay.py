"""research/111 — replay the 12-Aug manual management state-machine over ALL recorded
SENSEX days (options_data.db chain, 2026-04-20..now), vs two baselines on the SAME days:
  STATIC : sell ATM straddle 09:20, hold to 15:20.
  HARD30 : same entry; buy back a leg once it trades >= +30% over its entry; no re-entry.
  MANAGED: codified from the manual day — on a 30% breach: close leg, EXPAND (re-sell same
           side at ATM+/-300 from current spot) and TRAIL the untested leg to ATM-/+100;
           every new leg gets its own 30% stop; max 4 rolls/side; square 15:20.
qty 100 (5 lots x 20). Costs Rs80 per leg round-trip (entry pair 160; each roll adds 160).
Close-in tightening ("move spent") NOT in v0 — noted as refinement. Groups by trading-DTE.
Writes results/mgmt_replay.json + prints tables."""
import json, os, sqlite3
from bisect import bisect_right
from datetime import datetime, timedelta

ROOT = "/home/arun/quantifyd"
DB = os.path.join(ROOT, "backtest_data", "options_data.db")
OUT = os.path.join(ROOT, "research/111_sensex_manual_mgmt/results/mgmt_replay.json")
SYM = "SENSEX"; STEP = 100; QTY = 100; LEGCOST = 80
SL = 0.30; EXPAND = 300; TRAIL = 100; MAXROLL = 4

oc = sqlite3.connect(DB)
DAYS = [r[0] for r in oc.execute(
    "SELECT DISTINCT substr(snapshot_time,1,10) d FROM underlying_spot WHERE symbol=? AND spot_price>0 ORDER BY d", (SYM,))]
GRID = []
t = datetime.strptime("09:20", "%H:%M")
while t <= datetime.strptime("15:20", "%H:%M"):
    GRID.append(t.strftime("%H:%M")); t += timedelta(minutes=5)

def tdte(E, day):
    dd = datetime.strptime(day, "%Y-%m-%d").date(); ed = datetime.strptime(E, "%Y-%m-%d").date()
    n, cur = 0, dd + timedelta(days=1)
    while cur <= ed:
        if cur.weekday() < 5: n += 1
        cur += timedelta(days=1)
    return n

def load_day(day):
    sp = [(r[0][11:16], float(r[1])) for r in oc.execute(
        "SELECT snapshot_time, spot_price FROM underlying_spot WHERE symbol=? AND substr(snapshot_time,1,10)=? AND spot_price>0 ORDER BY snapshot_time", (SYM, day))]
    if not sp: return None
    exps = sorted({r[0] for r in oc.execute(
        "SELECT DISTINCT expiry_date FROM option_chain WHERE symbol=? AND substr(snapshot_time,1,10)=? AND expiry_date>=?", (SYM, day, day))})
    if not exps: return None
    E = exps[0]
    ch = {}
    for st, k, ty, lp in oc.execute(
        "SELECT snapshot_time, strike, instrument_type, ltp FROM option_chain WHERE symbol=? AND expiry_date=? AND substr(snapshot_time,1,10)=? AND ltp>0", (SYM, E, day)):
        ch.setdefault((int(k), ty), []).append((st[11:16], float(lp)))
    for v in ch.values(): v.sort()
    return dict(spot=sp, exp=E, chain=ch)

def mk_at(series):
    ts = [t for t, _ in series]
    def at(h):
        i = bisect_right(ts, h)
        return series[i-1][1] if i else None
    return at

def run_day(day, mode):
    b = LD.get(day)
    if not b: return None
    spot_at = mk_at(b["spot"]) ; ch = b["chain"]
    def ltp(k, ty, h):
        s = ch.get((k, ty))
        if not s: return None
        i = bisect_right([t for t, _ in s], h)
        return s[i-1][1] if i else None
    s0 = spot_at("09:20")
    if not s0: return None
    K = round(s0 / STEP) * STEP
    ce0, pe0 = ltp(K, "CE", "09:20"), ltp(K, "PE", "09:20")
    if not (ce0 and pe0): return None
    # legs: dict side -> {k, entry, alive} ; realized pnl accumulates
    legs = {"CE": {"k": K, "e": ce0, "alive": True}, "PE": {"k": K, "e": pe0, "alive": True}}
    pnl = -2 * LEGCOST; rolls = 0; breached = False
    for h in GRID[1:]:
        sp = spot_at(h)
        if sp is None: continue
        for side in ("CE", "PE"):
            L = legs[side]
            if not L["alive"]: continue
            cur = ltp(L["k"], side, h)
            if cur is None: continue
            if cur >= (1 + SL) * L["e"]:                    # breach
                breached = True
                pnl += (L["e"] - cur) * QTY - LEGCOST       # buy back at loss
                if mode == "HARD30":
                    L["alive"] = False
                elif mode == "MANAGED" and rolls < MAXROLL:
                    atm = round(sp / STEP) * STEP
                    nk = atm + EXPAND if side == "CE" else atm - EXPAND
                    np_ = ltp(nk, side, h)
                    if np_:
                        L.update(k=nk, e=np_); pnl -= LEGCOST  # re-sell rolled leg
                    else:
                        L["alive"] = False
                    # trail the untested side to ATM -/+ TRAIL
                    o = "PE" if side == "CE" else "CE"
                    OL = legs[o]
                    if OL["alive"]:
                        oc_ = ltp(OL["k"], o, h)
                        tk = atm - TRAIL if o == "PE" else atm + TRAIL
                        tp_ = ltp(tk, o, h)
                        if oc_ is not None and tp_ is not None and tk != OL["k"]:
                            pnl += (OL["e"] - oc_) * QTY - LEGCOST
                            OL.update(k=tk, e=tp_); pnl -= LEGCOST
                    rolls += 1
                else:
                    L["alive"] = False
    for side in ("CE", "PE"):                                # square 15:20
        L = legs[side]
        if L["alive"]:
            cur = ltp(L["k"], side, "15:20")
            if cur is not None:
                pnl += (L["e"] - cur) * QTY - LEGCOST
    return dict(pnl=round(pnl), rolls=rolls, breached=breached, dte=tdte(b["exp"], day))

LD = {}
res = {m: [] for m in ("STATIC", "HARD30", "MANAGED")}
for i, day in enumerate(DAYS):
    LD = {day: load_day(day)}
    for m in res:
        r = run_day(day, m)
        if r: res[m].append({"day": day, **r})
    if i % 10 == 0: print("  %d/%d %s" % (i, len(DAYS), day), flush=True)

def agg(rows):
    f = [r["pnl"] for r in rows]
    if not f: return None
    c = pk = dd = 0
    for v in f:
        c += v; pk = max(pk, c); dd = min(dd, c - pk)
    return dict(n=len(f), total=sum(f), mean=round(sum(f)/len(f)),
                win=round(100*sum(1 for v in f if v > 0)/len(f)),
                worst=min(f), maxdd=round(dd),
                breach_pct=round(100*sum(1 for r in rows if r["breached"])/len(rows)))

out = {"generated": datetime.now().isoformat()[:16], "qty": QTY, "days": len(DAYS), "modes": {}}
for m, rows in res.items():
    out["modes"][m] = {"all": agg(rows), "by_dte": {}, "rows": rows}
    for k in range(5):
        a = agg([r for r in rows if r["dte"] == k])
        if a: out["modes"][m]["by_dte"][k] = a
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"))
print("\n=== ALL DAYS (qty %d = 5 lots) ===" % QTY)
for m in res:
    a = out["modes"][m]["all"]
    print("%-8s n=%2d total %+9d mean %+7d win %d%% worst %+8d maxDD %+8d breach %d%%" % (
        m, a["n"], a["total"], a["mean"], a["win"], a["worst"], a["maxdd"], a["breach_pct"]))
print("\n=== BY TRADING-DTE (total | mean | win | maxDD) ===")
for k in range(5):
    line = "DTE%d: " % k
    for m in res:
        a = out["modes"][m]["by_dte"].get(k)
        line += ("%s[%+d|%+d|%d%%|%+d]  " % (m[0], a["total"], a["mean"], a["win"], a["maxdd"])) if a else "%s[--]  " % m[0]
    print(line)
