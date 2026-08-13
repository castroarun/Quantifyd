"""research/111 sec 15 - NAS SUITE vs CSL-REPLACEMENT vs HYBRID (NIFTY, 3 lots/system).

The live suite = 3 ATM straddles at 09:16 differing ONLY in post-SL management:
  ATM  : per-leg SL30 -> close hit leg, trail survivor TO COST, re-enter new ATM (<=5/day)
  ATM2 : per-leg SL30 -> close BOTH (one-and-done) + Rs2,500/lot rupee MTM stop
  ATM4 : per-leg SL30 -> close both + ROLL to new ATM once
Arms replayed per system (identical days, raw 3-sec chain, 2-snap dwell, Rs160/cycle):
  BASE : the live per-leg mechanics above (modeled; live actuals reported separately)
  COMB : per-leg logic fully REPLACED by combined-premium SL X% (one-and-done)  [X swept]
  HYB  : CSL X% as the trigger; on hit apply the system's own management
         (ATM: loser out + trail winner to cost + re-enter; ATM4: roll once;
          ATM2: both out = identical to COMB by construction)
Suites: SUITE_BASE / SUITE_COMB (=3x COMB - NOTE: replacement collapses diversification)
        / SUITE_HYB (COMB + HYB_ATM + HYB_ATM4) + correlations.
Caveat: live portfolio stop (-1300/lot/venue) not modeled in replay arms (live actuals have it).
Writes results/nas_suite_csl_replay.json; prints tables."""
import json, sqlite3
from bisect import bisect_right
from datetime import datetime, timedelta

DB = "/home/arun/quantifyd/backtest_data/options_data.db"
OUT = "/home/arun/quantifyd/research/111_sensex_manual_mgmt/results/nas_suite_csl_replay.json"
SYM, STEP, QTY, LOTS = "NIFTY", 50, 195, 3
ENT_T, EX_T, REENTER_CUT = "09:16", "15:15", "14:50"
LEG_SL = 0.30
RUPEE_STOP = -2500 * LOTS      # ATM2 MTM stop (rupees, per system)
SLS = (20, 25, 30, 40)         # CSL sweep for COMB/HYB
MAX_CYCLES = 5                 # ATM re-enter cap (live max_reentries)
COST = 160

oc = sqlite3.connect(DB)

def tdte(E, day):
    dd = datetime.strptime(day, "%Y-%m-%d").date(); ed = datetime.strptime(E, "%Y-%m-%d").date()
    n, cur = 0, dd + timedelta(days=1)
    while cur <= ed:
        if cur.weekday() < 5: n += 1
        cur += timedelta(days=1)
    return n

def day_chain(day, E):
    m = {}
    for t, k, ty, v in oc.execute(
        "SELECT snapshot_time,strike,instrument_type,ltp FROM option_chain INDEXED BY idx_oc_symbol_time "
        "WHERE symbol=? AND snapshot_time BETWEEN ? AND ? AND expiry_date=? AND ltp>0 ORDER BY snapshot_time",
            (SYM, day + "T09:00:00", day + "T15:26:00", E)):
        m.setdefault((k, ty), []).append((t[11:19], float(v)))
    return m

days = [r[0] for r in oc.execute(
    "SELECT DISTINCT substr(snapshot_time,1,10) FROM underlying_spot WHERE symbol=? AND spot_price>0 ORDER BY 1", (SYM,))]

ARMS = ["BASE_ATM", "BASE_ATM2", "BASE_ATM4"] + \
       ["COMB%d" % x for x in SLS] + \
       ["HYB_ATM%d" % x for x in SLS] + ["HYB_ATM4_%d" % x for x in SLS]
res = {a: [] for a in ARMS}
dte_of = {}

for di, day in enumerate(days):
    exps = sorted({r[0] for r in oc.execute(
        "SELECT DISTINCT expiry_date FROM option_chain INDEXED BY idx_oc_symbol_time WHERE symbol=? AND snapshot_time BETWEEN ? AND ? AND expiry_date>=?",
        (SYM, day + "T09:00:00", day + "T15:26:00", day))})
    if not exps: continue
    E = exps[0]; k_dte = tdte(E, day)
    sp = [(r[0][11:19], float(r[1])) for r in oc.execute(
        "SELECT snapshot_time,spot_price FROM underlying_spot WHERE symbol=? AND snapshot_time BETWEEN ? AND ? AND spot_price>0 ORDER BY snapshot_time",
        (SYM, day + "T09:00:00", day + "T15:26:00"))]
    if not sp: continue
    spt = [a for a, _ in sp]
    j0 = bisect_right(spt, ENT_T + ":59")
    if not j0 or spt[j0 - 1] < "09:11": continue
    chain = day_chain(day, E)

    def atm_at(t):
        jj = bisect_right(spt, t)
        return round(sp[jj - 1][1] / STEP) * STEP if jj else None

    def legs_path(K, t0):
        """[(t, ce, pe)] from t0 to EX_T, on CE snap clock with PE sampled."""
        ce, pe = chain.get((K, "CE"), []), chain.get((K, "PE"), [])
        if not (ce and pe): return None
        pk = [a for a, _ in pe]; pv = [v for _, v in pe]
        out = []
        for t, cv in ce:
            if t < t0 or t > EX_T + ":59": continue
            jp = bisect_right(pk, t)
            if jp: out.append((t, cv, pv[jp - 1]))
        return out if len(out) >= 10 else None

    def run_cycle(K, t0, mode, X=None):
        """One straddle cycle. Returns (points_pnl, exit_time, kind) where kind in
        HOLD / LEG_CE / LEG_PE / CSL / RUPEE. mode: base_atm|base_atm2|base_atm4|comb|hyb_atm|hyb_atm4"""
        P = legs_path(K, t0)
        if not P: return None
        ce_e, pe_e = P[0][1], P[0][2]
        ent = ce_e + pe_e
        per_leg = mode.startswith("base")
        thr_comb = ent * (1 + (X or 0) / 100.0)
        s_ce = s_pe = s_cb = s_ru = 0
        for m in range(1, len(P)):
            t, cv, pv2 = P[m]
            nx = P[min(m + 1, len(P) - 1)]
            if per_leg:
                if cv >= ce_e * (1 + LEG_SL): s_ce += 1
                else: s_ce = 0
                if pv2 >= pe_e * (1 + LEG_SL): s_pe += 1
                else: s_pe = 0
                if mode == "base_atm2":
                    if ((ce_e - cv) + (pe_e - pv2)) * QTY <= RUPEE_STOP: s_ru += 1
                    else: s_ru = 0
                    if s_ru >= 2:
                        return ((ent - nx[1] - nx[2]), nx[0], "RUPEE")
                if s_ce >= 2 or s_pe >= 2:
                    kind = "LEG_CE" if s_ce >= 2 else "LEG_PE"
                    if mode == "base_atm2":       # both out
                        return ((ent - nx[1] - nx[2]), nx[0], kind)
                    if mode == "base_atm4":       # both out (roll handled by caller)
                        return ((ent - nx[1] - nx[2]), nx[0], kind)
                    # base_atm: close hit leg, trail survivor TO COST
                    if kind == "LEG_CE":
                        realized = (ce_e - nx[1]); win_e = pe_e; wi = 2
                    else:
                        realized = (pe_e - nx[2]); win_e = ce_e; wi = 1
                    s_w = 0
                    for m2 in range(min(m + 1, len(P) - 1), len(P)):
                        wv = P[m2][wi]
                        if wv >= win_e: s_w += 1
                        else: s_w = 0
                        if s_w >= 2:
                            nx2 = P[min(m2 + 1, len(P) - 1)]
                            return (realized + (win_e - nx2[wi]), nx2[0], kind)
                    return (realized + (win_e - P[-1][wi]), P[-1][0], kind)
            else:
                if cv + pv2 >= thr_comb: s_cb += 1
                else: s_cb = 0
                if s_cb >= 2:
                    if mode == "hyb_atm":         # loser out, trail winner to cost
                        if (nx[1] / max(ce_e, 0.05)) >= (nx[2] / max(pe_e, 0.05)):
                            realized = (ce_e - nx[1]); win_e = pe_e; wi = 2
                        else:
                            realized = (pe_e - nx[2]); win_e = ce_e; wi = 1
                        s_w = 0
                        for m2 in range(min(m + 1, len(P) - 1), len(P)):
                            wv = P[m2][wi]
                            if wv >= win_e: s_w += 1
                            else: s_w = 0
                            if s_w >= 2:
                                nx2 = P[min(m2 + 1, len(P) - 1)]
                                return (realized + (win_e - nx2[wi]), nx2[0], "CSL")
                        return (realized + (win_e - P[-1][wi]), P[-1][0], "CSL")
                    return ((ent - nx[1] - nx[2]), nx[0], "CSL")  # comb / hyb_atm4: both out
        return ((ent - P[-1][1] - P[-1][2]), P[-1][0], "HOLD")

    def run_day(mode, X=None, reenter=0):
        """Full day: cycles with re-entry/roll semantics. Returns rupees."""
        total = 0.0; t_cur = ENT_T + ":00"; cycles = 0
        while True:
            K = atm_at(t_cur)
            if K is None: break
            r = run_cycle(K, t_cur, mode, X)
            if r is None: break
            pts, t_x, kind = r
            total += pts * QTY - COST
            cycles += 1
            if kind == "HOLD" or cycles > reenter or t_x > REENTER_CUT + ":59":
                break
            t_cur = t_x
        return round(total)

    ok = True
    for arm, mode, X, re_n in (
            [("BASE_ATM", "base_atm", None, MAX_CYCLES), ("BASE_ATM2", "base_atm2", None, 0),
             ("BASE_ATM4", "base_atm4", None, 1)] +
            [("COMB%d" % x, "comb", x, 0) for x in SLS] +
            [("HYB_ATM%d" % x, "hyb_atm", x, MAX_CYCLES) for x in SLS] +
            [("HYB_ATM4_%d" % x, "hyb_atm4", x, 1) for x in SLS]):
        v = run_day(mode, X, re_n)
        if v is None: ok = False; break
        res[arm].append((day, v))
    if ok: dte_of[day] = k_dte
    if di % 10 == 0: print("%d/%d %s" % (di, len(days), day), flush=True)

def agg(fp):
    f = [v for _, v in fp]
    if not f: return None
    c = pk = dd = 0
    for v in f: c += v; pk = max(pk, c); dd = min(dd, c - pk)
    n = len(f)
    return dict(total=round(sum(f)), mean=round(sum(f) / n), win=round(100 * sum(1 for v in f if v > 0) / n),
                maxdd=round(dd), n=n, ratio=(round(sum(f) / abs(dd), 1) if dd < 0 else 99.0))

def corr(a, b):
    da = dict(a); db = dict(b); ks = sorted(set(da) & set(db))
    if len(ks) < 5: return None
    xa = [da[k] for k in ks]; xb = [db[k] for k in ks]
    ma = sum(xa) / len(xa); mb = sum(xb) / len(xb)
    num = sum((p - ma) * (q - mb) for p, q in zip(xa, xb))
    va = sum((p - ma) ** 2 for p in xa) ** 0.5; vb = sum((q - mb) ** 2 for q in xb) ** 0.5
    return round(num / (va * vb), 2) if va > 0 and vb > 0 else None

# suites (per X)
suites = {}
suites["SUITE_BASE"] = [(d, res["BASE_ATM"][i][1] + res["BASE_ATM2"][i][1] + res["BASE_ATM4"][i][1])
                        for i, (d, _) in enumerate(res["BASE_ATM"])]
for x in SLS:
    suites["SUITE_COMB%d" % x] = [(d, 3 * v) for d, v in res["COMB%d" % x]]
    suites["SUITE_HYB%d" % x] = [(d, res["COMB%d" % x][i][1] + res["HYB_ATM%d" % x][i][1] + res["HYB_ATM4_%d" % x][i][1])
                                 for i, (d, _) in enumerate(res["COMB%d" % x])]

out = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
       "cfg": {"sym": SYM, "lots_per_system": LOTS, "qty": QTY, "leg_sl": 30, "rupee_stop_per_lot": 2500,
               "entry": ENT_T, "exit": EX_T, "sls_swept": list(SLS), "cost_per_cycle": COST,
               "caveat": "portfolio stop -1300/lot/venue NOT modeled; live actuals include it"},
       "dte_of": dte_of,
       "arms": {k: {"stats": agg(v), "series": v} for k, v in res.items()},
       "suites": {k: {"stats": agg(v), "series": v} for k, v in suites.items()},
       "corr": {"BASE_ATM_vs_ATM2": corr(res["BASE_ATM"], res["BASE_ATM2"]),
                "BASE_ATM_vs_ATM4": corr(res["BASE_ATM"], res["BASE_ATM4"]),
                "BASE_ATM2_vs_ATM4": corr(res["BASE_ATM2"], res["BASE_ATM4"]),
                "HYB_ATM30_vs_COMB30": corr(res["HYB_ATM30"], res["COMB30"]),
                "HYB_ATM30_vs_HYB_ATM4_30": corr(res["HYB_ATM30"], res["HYB_ATM4_30"]),
                "COMB30_vs_HYB_ATM4_30": corr(res["COMB30"], res["HYB_ATM4_30"]),
                "SUITE_BASE_vs_SUITE_HYB30": corr(suites["SUITE_BASE"], suites["SUITE_HYB30"]),
                "SUITE_BASE_vs_SUITE_COMB30": corr(suites["SUITE_BASE"], suites["SUITE_COMB30"])}}
json.dump(out, open(OUT, "w"))

n0 = agg(res["BASE_ATM"])["n"] if res["BASE_ATM"] else 0
print("\nNAS SUITE vs CSL arms - NIFTY, %d lots/system, n=%d days (%s -> %s)" % (LOTS, n0, days[0], days[-1]))
print("%-14s %10s %8s %5s %10s %7s" % ("ARM", "TOTAL", "MEAN/d", "WIN%", "MAXDD", "RATIO"))
for a in ARMS:
    st = agg(res[a])
    if st: print("%-14s %+10d %+8d %4d%% %+10d %7.1f" % (a, st["total"], st["mean"], st["win"], st["maxdd"], st["ratio"]))
print("\nSUITES (3 systems x 3 lots = 9 lots):")
for k in sorted(suites):
    st = agg(suites[k])
    if st: print("%-14s %+10d %+8d %4d%% %+10d %7.1f" % (k, st["total"], st["mean"], st["win"], st["maxdd"], st["ratio"]))
print("\nCORRELATIONS:", json.dumps(out["corr"], indent=1))
print("DONE")
