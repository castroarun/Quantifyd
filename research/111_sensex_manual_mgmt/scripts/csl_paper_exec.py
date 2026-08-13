"""CSL PAPER BOOKS (research/111) — NIFTY @12 lots (2u) + SENSEX @6 lots (1u).
Trades the FROZEN best-config schedule (backtest_data/csl_paper_config.json — snapshotted
once from the Lab JSON; weekly Lab regen does NOT move this book's rules) as live PAPER:
  - per index: today's trading-DTE (weekday map) -> config (entry, exit, SL)
  - at entry: sell ATM straddle (strike from live spot), record entry premiums
  - poll ~5s: combined-SL with 2-consecutive-poll dwell -> exit at next poll (market model)
  - SL 'none' books carry a 50% DISASTER BACKSTOP (never truly stopless live)
  - at exit time (or 15:25 hard stop): close at market ltp
State: backtest_data/csl_paper_state.json (append per day) + copy to static/app/csl_paper.json.
Launched by cron 09:12 Mon-Fri; standalone (no Flask/gunicorn involvement)."""
import json, os, sys, time
from datetime import datetime, date

Q = "/home/arun/quantifyd"
sys.path.insert(0, Q)
CONFIG = Q + "/backtest_data/csl_paper_config.json"
STATE = Q + "/backtest_data/csl_paper_state.json"
PUB = Q + "/static/app/csl_paper.json"
PUBLIVE = Q + "/static/app/csl_paper_live.json"
NIFTY_MKT = {"sym": "NIFTY", "step": 50, "lot": 65, "spot_key": "NSE:NIFTY 50", "seg": "NFO",
             "wd2dte": {0: 1, 1: 0, 2: 4, 3: 3, 4: 2}}
SENSEX_MKT = {"sym": "SENSEX", "step": 100, "lot": 20, "spot_key": "BSE:SENSEX", "seg": "BFO",
              "wd2dte": {0: 3, 1: 2, 2: 1, 3: 0, 4: 4}}
BOOKS = {
    "CSL_TIMEB_NIFTY": {**NIFTY_MKT, "lots": 12, "qty": 780, "cfg_from": "lab"},
    "CSL_TIMEB_SENSEX": {**SENSEX_MKT, "lots": 6, "qty": 120, "cfg_from": "lab"},
    # A/B twin of the live nas_916_atm mechanic question: same venue/entry, COMBINED-20% stop
    "NAS_COMB20": {**NIFTY_MKT, "lots": 3, "qty": 195, "cfg_from": "fixed",
                   "fixed_cfg": {"entry": "09:16", "exit": "15:20", "sl": 20}},
    # FIXED-CSL30 books (the flat rule, un-windowed) — the variable-vs-fixed live A/B
    "CSL30F_NIFTY": {**NIFTY_MKT, "lots": 3, "qty": 195, "cfg_from": "fixed",
                     "fixed_cfg": {"entry": "09:16", "exit": "15:20", "sl": 30}},
    "CSL30F_SENSEX": {**SENSEX_MKT, "lots": 3, "qty": 60, "cfg_from": "fixed",
                      "fixed_cfg": {"entry": "09:16", "exit": "15:20", "sl": 30}},
}
BACKSTOP = 0.50   # for SL 'none' configs
POLL = 5          # seconds
SAMPLE_EVERY = 12  # record MTM every ~60s for day curves

def log(m): print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), m), flush=True)

def freeze_config():
    if os.path.exists(CONFIG): return json.load(open(CONFIG))
    lab = json.load(open(Q + "/static/app/straddles/csl_best_configs.json"))
    cfg = {"frozen_at": datetime.now().isoformat()[:16], "source_generated_at": lab.get("generated_at"),
           "note": "FROZEN for out-of-sample paper validation; re-freeze consciously, never silently.",
           "books": {}}
    for bk, B in BOOKS.items():
        if B["cfg_from"] == "lab":
            cfg["books"][bk] = {k: {kk: b[kk] for kk in ("entry", "exit", "sl")}
                                for k, b in lab["best"][B["sym"]].items()}
        else:
            cfg["books"][bk] = {str(k): dict(B["fixed_cfg"]) for k in range(5)}
    json.dump(cfg, open(CONFIG, "w"), indent=1)
    log("config FROZEN from Lab (%s)" % cfg["source_generated_at"])
    return cfg

def kite():
    from kiteconnect import KiteConnect
    tok = json.load(open(Q + "/backtest_data/access_token.json"))
    try:
        from config import KITE_API_KEY as AK
    except Exception:
        AK = tok.get("api_key")
    k = KiteConnect(api_key=AK); k.set_access_token(tok["access_token"])
    return k

def load_state():
    try: return json.load(open(STATE))
    except Exception: return {"records": [], "cum": {}}

def save_state(st):
    json.dump(st, open(STATE, "w"))
    try: json.dump(st, open(PUB, "w"))
    except Exception: pass

def push_event(st, book, etype, msg):
    """Alert feed consumed by the Windows watcher (scripts/csl_alert_watcher.pyw)."""
    st.setdefault("events", []).append({
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "book": book,
        "type": etype, "source": "PAPER", "msg": msg})
    st["events"] = st["events"][-200:]

def write_live(plans, today):
    try:
        json.dump({"day": today, "at": datetime.now().strftime("%H:%M:%S"),
                   "books": {bk: {"state": P["state"], "credit": P.get("credit"),
                                   "series": P.get("series", [])} for bk, P in plans.items()}},
                  open(PUBLIVE, "w"))
    except Exception:
        pass

def resolve_legs(k, B, strike):
    sym = B["sym"]; seg = B["seg"]
    ins = resolve_legs.cache.get(seg)
    if ins is None:
        ins = [i for i in k.instruments(seg) if i["name"] == sym and i["instrument_type"] in ("CE", "PE")]
        resolve_legs.cache[seg] = ins
    today = date.today()
    exps = sorted({i["expiry"] for i in ins if i["expiry"] >= today})
    if not exps: return None, None, None
    E = exps[0]
    ce = next((i for i in ins if i["expiry"] == E and i["strike"] == strike and i["instrument_type"] == "CE"), None)
    pe = next((i for i in ins if i["expiry"] == E and i["strike"] == strike and i["instrument_type"] == "PE"), None)
    return ce, pe, E
resolve_legs.cache = {}

def main():
    probe = "--probe" in sys.argv
    cfg = freeze_config()
    k = kite()
    st = load_state()
    today = date.today().isoformat()
    wd = date.today().weekday()
    if wd > 4: return
    plans = {}
    for bk, B in BOOKS.items():
        dte = B["wd2dte"].get(wd)
        c = (cfg["books"].get(bk) or {}).get(str(dte))
        if not c:
            log("%s: no config for DTE%s — skip today" % (bk, dte)); continue
        if any(r["day"] == today and (r.get("book") == bk or (not r.get("book") and r.get("sym") == B["sym"])) for r in st["records"]):
            log("%s: already recorded today — skip" % bk); continue
        plans[bk] = {"dte": dte, **c, "state": "WAIT_ENTRY", "K": None, "legs": None,
                     "ce0": None, "pe0": None, "credit": None, "streak": 0, "last_comb": None,
                     "series": [], "tick": 0}
        log("%s plan: DTE%d %s->%s SL%s qty %d (%d lots)" % (bk, dte, c["entry"], c["exit"], c["sl"], B["qty"], B["lots"]))
    if probe:
        for bk in plans:
            B = BOOKS[bk]
            sp = k.ltp([B["spot_key"]])[B["spot_key"]]["last_price"]
            K = round(sp / B["step"]) * B["step"]
            ce, pe, E = resolve_legs(k, B, K)
            log("PROBE %s spot %.0f ATM %d exp %s CE %s PE %s" % (
                bk, sp, K, E, ce and ce["tradingsymbol"], pe and pe["tradingsymbol"]))
        return
    while plans:
        now = datetime.now().strftime("%H:%M")
        if now >= "15:26": now_force = True
        else: now_force = False
        for sym in list(plans):     # sym here = book key
            P = plans[sym]; B = BOOKS[sym]
            try:
                if P["state"] == "WAIT_ENTRY":
                    em = int(P["entry"][:2]) * 60 + int(P["entry"][3:5])
                    nm = int(now[:2]) * 60 + int(now[3:5])
                    if nm > em + 15:
                        log("%s: entry window stale (now %s > %s+15m) — skip day" % (sym, now, P["entry"]))
                        del plans[sym]; continue
                    if now >= P["entry"]:
                        sp = k.ltp([B["spot_key"]])[B["spot_key"]]["last_price"]
                        K = round(sp / B["step"]) * B["step"]
                        ce, pe, E = resolve_legs(k, B, K)
                        if not (ce and pe):
                            log("%s: legs unresolved — abort today" % sym); del plans[sym]; continue
                        q = k.ltp(["%s:%s" % (B["seg"], ce["tradingsymbol"]), "%s:%s" % (B["seg"], pe["tradingsymbol"])])
                        P["ce0"] = q["%s:%s" % (B["seg"], ce["tradingsymbol"])]["last_price"]
                        P["pe0"] = q["%s:%s" % (B["seg"], pe["tradingsymbol"])]["last_price"]
                        P.update(K=K, legs=(ce["tradingsymbol"], pe["tradingsymbol"]), state="OPEN",
                                 credit=P["ce0"] + P["pe0"], entry_ts=datetime.now().strftime("%H:%M:%S"), expiry=str(E))
                        log("%s ENTER K=%d credit %.2f (%s+%s)" % (sym, K, P["credit"], P["ce0"], P["pe0"]))
                        push_event(st, sym, "ENTRY", "SOLD %d straddle @ %.2f credit (%d lots, DTE%d, %s->%s SL%s)" % (
                            K, P["credit"], B["lots"], P["dte"], P["entry"], P["exit"], P["sl"]))
                        save_state(st)
                elif P["state"] == "OPEN":
                    ce_s, pe_s = P["legs"]
                    q = k.ltp(["%s:%s" % (B["seg"], ce_s), "%s:%s" % (B["seg"], pe_s)])
                    comb = q["%s:%s" % (B["seg"], ce_s)]["last_price"] + q["%s:%s" % (B["seg"], pe_s)]["last_price"]
                    P["tick"] += 1
                    if P["tick"] % SAMPLE_EVERY == 1:
                        P["series"].append([now, round((P["credit"] - comb) * B["qty"])])
                        write_live(plans, today)
                    sl = BACKSTOP if P["sl"] == "none" else P["sl"] / 100.0
                    thr = (1 + sl) * P["credit"]
                    reason = None
                    if P["last_comb"] is not None and P["streak"] >= 2:
                        reason = "SL_DWELL"          # dwell confirmed on prior polls -> exit THIS poll
                    if comb >= thr: P["streak"] += 1
                    else: P["streak"] = 0
                    P["last_comb"] = comb
                    if now >= P["exit"]: reason = reason or "TIME_EXIT"
                    if now_force: reason = reason or "EOD_FORCE"
                    if reason:
                        pnl = round((P["credit"] - comb) * B["qty"] - 160)
                        rec = {"day": today, "book": sym, "sym": B["sym"], "dte": P["dte"], "cfg": "%s->%s SL%s" % (P["entry"], P["exit"], P["sl"]),
                               "strike": P["K"], "expiry": P["expiry"], "credit": round(P["credit"], 2),
                               "entry_ts": P["entry_ts"], "exit_ts": datetime.now().strftime("%H:%M:%S"),
                               "exit_comb": round(comb, 2), "reason": reason, "pnl": pnl, "series": P.get("series", []),
                               "lots": B["lots"], "qty": B["qty"], "source": "PAPER"}
                        st["records"].append(rec)
                        push_event(st, sym, "EXIT", "%s: closed %d straddle @ %.2f -> P&L %+d (%d lots, cum %+d)" % (
                            reason, P["K"], comb, pnl, B["lots"], st["cum"].get(sym, 0) + pnl))
                        st["cum"][sym] = st["cum"].get(sym, 0) + pnl
                        save_state(st)
                        log("%s EXIT %s pnl %+d (cum %+d)" % (sym, reason, pnl, st["cum"][sym]))
                        del plans[sym]
            except Exception as ex:
                log("%s poll err: %s" % (sym, str(ex)[:80]))
        time.sleep(POLL)
        if datetime.now().strftime("%H:%M") >= "15:30" and not any(p["state"] == "OPEN" for p in plans.values()):
            break
    log("day done")

if __name__ == "__main__":
    main()
