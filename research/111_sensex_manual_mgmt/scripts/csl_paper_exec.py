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
    "CSL_TIMEB_NIFTY": {**NIFTY_MKT, "lots": 8, "qty": 520, "cfg_from": "lab", "mode": "live"},  # 2026-08-17: 6->8 (sec-18b step-2, user)
    "CSL_TIMEB_SENSEX": {**SENSEX_MKT, "lots": 8, "qty": 160, "cfg_from": "lab", "mode": "live"},  # 2026-08-18: 6L->8L REAL (notional parity w/ NIFTY TB@8L); suite Wed->paper same night
    # 2026-08-18: SECOND time-slots evidence book (sweep sec: Mon+Tue have a 2nd earnings pocket)
    "CSL_TIMEB2_NIFTY": {**NIFTY_MKT, "lots": 2, "qty": 130, "cfg_from": "fixed",
                         "fixed_cfg": {"entry": "13:00", "exit": "14:00", "sl": 25}},
    # research/125 expiry-Tuesday afternoon window - REAL (user 2026-08-25). One-shot
    # precedent traded 25-Aug (-2,990); standing book from 26-Aug. Tuesday (DTE0) only -
    # config json carries the single cell, so fixed_cfg is bootstrap-only.
    "CSL_TIMEB2_LIVE": {**NIFTY_MKT, "lots": 8, "qty": 520, "cfg_from": "fixed", "mode": "live",
                        "fixed_cfg": {"entry": "13:15", "exit": "14:30", "sl": 30}},
    # A/B twin of the live nas_916_atm mechanic question: same venue/entry, COMBINED-20% stop
    "NAS_COMB20": {**NIFTY_MKT, "lots": 2, "qty": 130, "cfg_from": "fixed", "mode": "live",
                   "fixed_cfg": {"entry": "09:16", "exit": "15:20", "sl": 20}},
    # FIXED-CSL30 books (the flat rule, un-windowed) — the variable-vs-fixed live A/B
    "CSL_TIMEB_NIFTY_THU": {**NIFTY_MKT, "lots": 3, "qty": 195, "cfg_from": "fixed", "mode": "live",
                            "fixed_cfg": {"entry": "09:25", "exit": "15:20", "sl": 20}},  # 2026-08-19: Thursday-only TB-N at reduced size (Option B) - DTE3 is the 2nd-best NIFTY cell; 3L is the max that clears every margin gate at current capital. Config json trimmed to DTE3 only.
    "CSL30F_SENSEX_WED": {**SENSEX_MKT, "lots": 3, "qty": 60, "cfg_from": "fixed", "mode": "live",
                          "fixed_cfg": {"entry": "09:16", "exit": "15:20", "sl": 30}},  # 2026-08-20 USER OVERRIDE vs study: Wed full-day cell is -571/day 64% (n=11) and verdict Q4 said windows-only - Arun chose live anyway after seeing the table. Config json trimmed to DTE1 only. Review after 4 live Wednesdays (Ops). Paper control book unchanged.
"CSL_TIMEB_NIFTY_MON_AM": {**NIFTY_MKT, "lots": 8, "qty": 520, "cfg_from": "fixed", "mode": "live",
                               "fixed_cfg": {"entry": "09:16", "exit": "11:16", "sl": "rs1000"}},  # 2026-08-25 USER OVERRIDE vs study: r/124 re-run makes this the best Monday cell (median +6,920@8L, win 88.9%, R:R@p95 1:1.0, stop-invariant) BUT it FAILS the label-shuffle null (p=0.376, n=18) - indistinguishable from mined noise. Arun chose live anyway at 8L with the Rs1,000/lot rupee stop, which caps the worst day best (-15,752 vs -20,464 nostop / -28,496 SLP20). Enters 09:16 alongside the 6-lot suite + 2-lot COMB on the SAME strike - the r/126 Arm C concentration caveat applies. Review after 4 live Mondays (Ops).
    "NAS_COMB20_THU": {**NIFTY_MKT, "lots": 5, "qty": 325, "cfg_from": "fixed",
                       "fixed_cfg": {"entry": "09:16", "exit": "15:20", "sl": 20}},  # 2026-08-27: NIFTY is OFF on Thursdays live (Arun). Reverses the 19-20 Aug Option-B merge that put NIFTY-Thu into NAS_COMB20 as a 5L DTE3 cell - Thursday is SENSEX expiry and NIFTY was competing for the same margin on SENSEX best day. PAPER twin keeps the DTE3 evidence accumulating (grid mean ~16,956 at 91% is what motivated Option B) so this is revisited on data. Config seeded DTE3 only.
        "CSL_TIMEB_NIFTY_MON": {**NIFTY_MKT, "lots": 8, "qty": 520, "cfg_from": "fixed",
                            "fixed_cfg": {"entry": "13:00", "exit": "14:00", "sl": 20}},  # 2026-08-23 (rev same day): only MONDAY dropped from the live TimeB book (condemned by r/120+121+122; Arun first dropped Fri too, then kept it on its KEEP verdict). This PAPER twin keeps the Monday cell trading for the 2026-11 re-run. Config seeded with DTE1 only. Thu SX bump to 10L declined same day - stays 8L.
    "CSL30F_NIFTY": {**NIFTY_MKT, "lots": 2, "qty": 130, "cfg_from": "fixed",
                     "fixed_cfg": {"entry": "09:16", "exit": "15:20", "sl": 30}},
    "CSL30F_SENSEX": {**SENSEX_MKT, "lots": 3, "qty": 60, "cfg_from": "fixed",
                      "fixed_cfg": {"entry": "09:16", "exit": "15:20", "sl": 30}},
    # post-CSL management A/Bs (research/111 sec 14): on CSL hit, manage instead of quit
    "NAS_C20_TRAIL": {**NIFTY_MKT, "lots": 2, "qty": 130, "cfg_from": "fixed", "mgmt": "trail",
                      "fixed_cfg": {"entry": "09:16", "exit": "15:20", "sl": 20}},
    "NAS_C20_SHIFT": {**NIFTY_MKT, "lots": 2, "qty": 130, "cfg_from": "fixed", "mgmt": "shift",
                      "fixed_cfg": {"entry": "09:16", "exit": "15:20", "sl": 20}},
}
TRAIL_BOUNCE = 1.30   # trail arm: exit winner on >=30% bounce off post-trigger low
MAX_SHIFTS = 3        # shift arm: max re-centers per day
SHIFT_CUTOFF = "14:30"
BACKSTOP = 0.50   # for SL 'none' configs
POLL = 5          # seconds
SAMPLE_EVERY = 12  # record MTM every ~60s for day curves

def log(m): print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), m), flush=True)

def freeze_config():
    if os.path.exists(CONFIG):
        cfg = json.load(open(CONFIG))
        added = [bk for bk, B in BOOKS.items() if bk not in cfg["books"] and B["cfg_from"] == "fixed"]
        for bk in added:
            cfg["books"][bk] = {str(k): dict(BOOKS[bk]["fixed_cfg"]) for k in range(5)}
        if added:
            json.dump(cfg, open(CONFIG, "w"), indent=1)
            log("config: added new fixed books %s" % added)
        return cfg
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
    # Shared wrapped client: services.kite_service.get_kite monkey-wraps place_order to
    # auto-inject market_protection (Kite rejects bare MARKET on options, 2026-08-14) and
    # applies the suite's proven guards. Raw client only as last-resort fallback.
    try:
        from services.kite_service import get_kite
        return get_kite()
    except Exception as _e:
        from kiteconnect import KiteConnect
        tok = json.load(open(Q + "/backtest_data/access_token.json"))
        try:
            from config import KITE_API_KEY as AK
        except Exception:
            AK = tok.get("api_key")
        k = KiteConnect(api_key=AK); k.set_access_token(tok["access_token"])
        return k

KILL_FLAG = Q + "/backtest_data/nas_kill.flag"
FREEZE_FLAG = Q + "/backtest_data/nas_manual_freeze.flag"
MASTER = Q + "/backtest_data/nas_master_mode.json"
MARGIN_PER_LOT = 165000.0   # est short-straddle margin per NIFTY lot
MARGIN_HEADROOM = 1.3

def live_allowed():
    """Global gates shared with the NAS suite: one panic lever stops the whole stack."""
    if os.path.exists(KILL_FLAG): return False, "nas_kill.flag present"
    if os.path.exists(FREEZE_FLAG): return False, "manual freeze"
    try:
        if json.load(open(MASTER)).get("mode") != "live": return False, "master mode != live"
    except Exception:
        return False, "master mode unreadable"
    return True, ""

def is_live_book(B):
    # mgmt (trail/shift) books have no live order path yet -> always paper
    return B.get("mode") == "live" and not B.get("mgmt")

def _tick(px):
    return max(0.05, round(round(px * 20) / 20.0, 2))

def place_market(k, B, ts, side, qty, tag, ref_px=None):
    """Marketable-LIMIT MIS order (Kite API rejects plain MARKET on options: 2026-08-14
    'Market orders without market protection are not allowed'). Aggressive limit at
    ref +/-3% fills like market but passes the API check. Timeout-verify per the
    2026-08-06 lesson: a timed-out request may have gone through - check the orderbook."""
    try:
        # MARKET via the shared wrapped client (market_protection auto-injected upstream).
        return k.place_order(variety="regular", exchange=B["seg"], tradingsymbol=ts,
                             transaction_type=side, quantity=int(qty), product="MIS",
                             order_type="MARKET", tag=tag[:20])
    except Exception as ex:
        log("ORDER %s %s EXC: %s - verifying orderbook" % (side, ts, str(ex)[:70]))
        time.sleep(2)
        try:
            for o in k.orders():
                if (o.get("tag") or "") == tag[:20] and o["tradingsymbol"] == ts                         and o["transaction_type"] == side and o["status"] not in ("REJECTED", "CANCELLED"):
                    log("ORDER %s %s found in book after exc (%s)" % (side, ts, o["status"]))
                    return o["order_id"]
        except Exception:
            pass
        return None

def order_fill(k, oid, wait_s=20):
    """Wait for COMPLETE; return average fill price, else None (rejected/timeout)."""
    t0 = time.time()
    while time.time() - t0 < wait_s:
        try:
            hist = k.order_history(oid)
            stt = hist[-1]["status"]
            if stt == "COMPLETE":
                ap = float(hist[-1].get("average_price") or 0)
                return ap if ap > 0 else None
            if stt in ("REJECTED", "CANCELLED"):
                log("ORDER %s %s: %s" % (oid, stt, str(hist[-1].get("status_message"))[:70]))
                return None
        except Exception as ex:
            log("order_fill err: %s" % str(ex)[:60])
        time.sleep(2)
    return None

def margin_ok(k, lots):
    """Use total net margin (cash + pledged collateral - utilised) - option selling
    is collateral-eligible; live_balance alone ignores pledges."""
    try:
        eq = k.margins()["equity"]
        avail = float(eq.get("net") or 0)
        if avail <= 0:
            av = eq.get("available", {})
            avail = float(av.get("live_balance") or 0) + float(av.get("collateral") or 0)
        need = MARGIN_PER_LOT * lots * MARGIN_HEADROOM
        return avail >= need, avail, need
    except Exception:
        return False, 0.0, 0.0

def broker_short_qty(k, tradingsymbol):
    """Broker-truth guard (ported from the 916 executors, 2026-08-18 after the 08-17
    COMB desync): positive = MIS qty actually SHORT at the broker for this leg;
    0 = leg no longer held (manual close / covered at account level) -> caller must
    RECONCILE, not buy; -1 = positions API failed (UNKNOWN) -> caller falls back to
    placing the exit, because protecting a real short outranks avoiding a phantom buy."""
    try:
        for _p in k.positions().get("net", []):
            if _p.get("tradingsymbol") == tradingsymbol and _p.get("product") == "MIS":
                return max(0, -int(_p.get("quantity") or 0))
        return 0
    except Exception as ex:
        log("broker_short_qty err %s: %s" % (tradingsymbol, str(ex)[:60]))
        return -1


def load_state():
    try: return json.load(open(STATE))
    except Exception: return {"records": [], "cum": {}}

def save_state(st):
    json.dump(st, open(STATE, "w"))
    try: json.dump(st, open(PUB, "w"))
    except Exception: pass

def push_event(st, book, etype, msg, source="PAPER"):
    """Alert feed consumed by the Windows watcher (scripts/csl_alert_watcher.pyw)."""
    st.setdefault("events", []).append({
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "book": book,
        "type": etype, "source": source, "msg": msg})
    st["events"] = st["events"][-200:]

def write_live(plans, today):
    try:
        def _bd(P, bk=None):
            _B = BOOKS.get(bk, {}) if bk else {}
            d = {"state": P["state"], "credit": P.get("credit"), "series": P.get("series", []),
                 "dte": P.get("dte"), "entry": P.get("entry"), "exit": P.get("exit"), "sl": P.get("sl"),
                 # real deployed size incl. any per-DTE override (the app showed a hardcoded
                 # 130 on Thursday while the account held 325 -- user, 2026-08-20)
                 "lots": P.get("lots", _B.get("lots")), "qty": P.get("qty", _B.get("qty"))}
            lg = P.get("legs")
            if lg:
                d.update(K=P.get("K"), ce_sym=lg[0], pe_sym=lg[1], ce0=P.get("ce0"), pe0=P.get("pe0"),
                         ce_last=P.get("ce_last"), pe_last=P.get("pe_last"), sl=P.get("sl"),
                         live=bool(P.get("live")), entry_ts=P.get("entry_ts"), dte=P.get("dte"))
            return d
        json.dump({"day": today, "at": datetime.now().strftime("%H:%M:%S"),
                   "books": {bk: _bd(P, bk) for bk, P in plans.items()}}, open(PUBLIVE, "w"))
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
        effB = {**B, "lots": c.get("lots", B["lots"]), "qty": c.get("qty", B["qty"])}
        plans[bk] = {"dte": dte, **c, "state": "WAIT_ENTRY", "K": None, "legs": None,
                     "ce0": None, "pe0": None, "credit": None, "streak": 0, "last_comb": None,
                     "series": [], "tick": 0, "realized_rs": 0.0, "cost": 160, "shifts": 0,
                     "B": effB}
        log("%s plan%s: DTE%d %s->%s SL%s qty %d (%d lots)" % (bk, " [LIVE]" if is_live_book(B) else "", dte, c["entry"], c["exit"], c["sl"], effB["qty"], effB["lots"]))
    if probe:
        for bk in plans:
            B = BOOKS[bk]
            sp = k.ltp([B["spot_key"]])[B["spot_key"]]["last_price"]
            # spot-ATM by design. The 9:16 suite instead re-snaps to the synthetic
            # forward, so the two families choose different strikes on ~31% of NIFTY and
            # ~48% of SENSEX mornings (a cost-of-carry basis that widens with DTE).
            # research/119 tested porting the snap here: -65/lot/day, and there is no
            # directional tilt to fix - these books are short GAMMA, not delta. Keeping
            # spot-ATM also keeps r/111 per-DTE windows and SLs on their validated credit.
            K = round(sp / B["step"]) * B["step"]
            ce, pe, E = resolve_legs(k, B, K)
            ok, why = live_allowed()
            mok, avail, need = margin_ok(k, B["lots"])
            log("PROBE %s spot %.0f ATM %d exp %s CE %s PE %s | mode=%s gates_ok=%s%s margin avail=%.0f need=%.0f ok=%s" % (
                bk, sp, K, E, ce and ce["tradingsymbol"], pe and pe["tradingsymbol"],
                "LIVE" if is_live_book(B) else "paper", ok, (" (%s)" % why) if why else "", avail, need, mok))
        return
    while plans:
        now = datetime.now().strftime("%H:%M")
        if now >= "15:26": now_force = True
        else: now_force = False
        # BATCH LTP: one quote call for all OPEN/TRAIL legs (Kite 3 req/s shared limit;
        # un-batched polling was starving the live suite SL/portfolio monitors, 2026-08-14).
        QB = {}; _need = set()
        for _s in list(plans):
            _P = plans[_s]; _B = BOOKS[_s]
            if _P.get("state") == "OPEN" and _P.get("legs"):
                _need.add("%s:%s" % (_B["seg"], _P["legs"][0]))
                _need.add("%s:%s" % (_B["seg"], _P["legs"][1]))
            elif _P.get("state") == "TRAIL" and _P.get("win_sym"):
                _need.add("%s:%s" % (_B["seg"], _P["win_sym"]))
        if _need:
            try: QB = k.ltp(list(_need))
            except Exception as _ex: log("batch ltp err: %s" % str(_ex)[:60])
        for sym in list(plans):     # sym here = book key
            P = plans[sym]; B = P.get("B") or BOOKS[sym]
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
                        # FORWARD SNAP (2026-08-27, ported from nas_atm_executor ~641 / research/119b).
                        # The index LTP is not what the options price off - on SENSEX the gap ran ~66
                        # points today - so round(spot/step) lands a strike or two below the true ATM and
                        # the "straddle" becomes a directional bet. The synthetic forward K + (CE - PE)
                        # is self-calibrating: put-call parity gives the level the chain itself is using.
                        # Fail-safe: no usable quote at the forward strike -> keep the spot strike.
                        try:
                            _fwd = K + (P["ce0"] - P["pe0"])
                            _fk = int(round(_fwd / B["step"]) * B["step"])
                            if _fk != K:
                                _c2, _p2, _E2 = resolve_legs(k, B, _fk)
                                if _c2 and _p2:
                                    _q2 = k.ltp(["%s:%s" % (B["seg"], _c2["tradingsymbol"]),
                                                 "%s:%s" % (B["seg"], _p2["tradingsymbol"])])
                                    _ce2 = _q2["%s:%s" % (B["seg"], _c2["tradingsymbol"])]["last_price"]
                                    _pe2 = _q2["%s:%s" % (B["seg"], _p2["tradingsymbol"])]["last_price"]
                                    if _ce2 and _pe2 and _ce2 > 0 and _pe2 > 0:
                                        log("%s FWD-SNAP: spot %.1f spot-K %d (CE %.2f/PE %.2f gap %.1f)"
                                            " -> fwd %.1f fwd-K %d (CE %.2f/PE %.2f gap %.1f)"
                                            % (sym, sp, K, P["ce0"], P["pe0"], P["ce0"] - P["pe0"],
                                               _fwd, _fk, _ce2, _pe2, _ce2 - _pe2))
                                        K, ce, pe, E = _fk, _c2, _p2, _E2
                                        P["ce0"], P["pe0"] = _ce2, _pe2
                                    else:
                                        log("%s FWD-SNAP skipped - no quote at %d, keeping spot-K %d" % (sym, _fk, K))
                                else:
                                    log("%s FWD-SNAP skipped - %d unresolved, keeping spot-K %d" % (sym, _fk, K))
                        except Exception as _fe:
                            log("%s FWD-SNAP error (%s) - keeping spot-K %d" % (sym, str(_fe)[:60], K))
                        P["live"] = False
                        if is_live_book(B):
                            ok, why = live_allowed()
                            mok, avail, need = margin_ok(k, B["lots"]) if ok else (False, 0.0, 0.0)
                            if not ok or not mok:
                                why = why or ("margin %.0f < need %.0f" % (avail, need))
                                log("%s: LIVE BLOCKED (%s) - paper fallback today" % (sym, why))
                                push_event(st, sym, "WARN", "LIVE blocked (%s) - paper fallback today" % why, "REAL")
                            else:
                                tag = ("CSL_" + sym)[:20]
                                oid_ce = place_market(k, B, ce["tradingsymbol"], "SELL", B["qty"], tag, P["ce0"])
                                f_ce = order_fill(k, oid_ce) if oid_ce else None
                                if f_ce is None:
                                    log("%s: CE entry order failed - paper fallback today" % sym)
                                    push_event(st, sym, "WARN", "CE entry order failed - paper fallback today", "REAL")
                                else:
                                    oid_pe = place_market(k, B, pe["tradingsymbol"], "SELL", B["qty"], tag, P["pe0"])
                                    f_pe = order_fill(k, oid_pe) if oid_pe else None
                                    if f_pe is None:
                                        log("%s: PE entry failed - UNWINDING CE (no naked leg)" % sym)
                                        push_event(st, sym, "WARN", "PE entry failed - unwinding CE leg, book skipped today. CHECK KITE.", "REAL")
                                        save_state(st)
                                        for _ in range(5):
                                            oid_u = place_market(k, B, ce["tradingsymbol"], "BUY", B["qty"], tag)
                                            if oid_u and order_fill(k, oid_u) is not None:
                                                log("%s: CE unwound clean" % sym); break
                                            time.sleep(3)
                                        del plans[sym]; continue
                                    P["ce0"], P["pe0"] = f_ce, f_pe
                                    P["live"] = True
                        P.update(K=K, legs=(ce["tradingsymbol"], pe["tradingsymbol"]), state="OPEN",
                                 credit=P["ce0"] + P["pe0"], entry_ts=datetime.now().strftime("%H:%M:%S"), expiry=str(E))
                        log("%s ENTER%s K=%d credit %.2f (%s+%s)" % (sym, " [LIVE]" if P["live"] else "", K, P["credit"], P["ce0"], P["pe0"]))
                        push_event(st, sym, "ENTRY", "SOLD %d straddle @ %.2f credit (%d lots, DTE%d, %s->%s SL%s)%s" % (
                            K, P["credit"], B["lots"], P["dte"], P["entry"], P["exit"], P["sl"],
                            " [REAL MONEY]" if P["live"] else ""), "REAL" if P["live"] else "PAPER")
                        save_state(st)
                elif P["state"] == "OPEN":
                    ce_s, pe_s = P["legs"]
                    _ck = "%s:%s" % (B["seg"], ce_s); _pk = "%s:%s" % (B["seg"], pe_s)
                    q = QB if (_ck in QB and _pk in QB) else k.ltp([_ck, _pk])
                    P["ce_last"] = q["%s:%s" % (B["seg"], ce_s)]["last_price"]
                    P["pe_last"] = q["%s:%s" % (B["seg"], pe_s)]["last_price"]
                    comb = P["ce_last"] + P["pe_last"]
                    P["tick"] += 1
                    if P["tick"] % SAMPLE_EVERY == 1:
                        P["series"].append([now, round(P["realized_rs"] + (P["credit"] - comb) * B["qty"])])
                        write_live(plans, today)
                    # Stop threshold. Three shapes:
                    #   "none"   -> 50% disaster backstop (never truly stopless live)
                    #   <number> -> percent of credit (scales with the credit collected)
                    #   "rsN"    -> Rs N per LOT, converted to points via the lot size. DTE-agnostic
                    #               (research/96 shape): it does NOT scale with credit, so a thin
                    #               credit day gets the same rupee risk as a fat one.
                    _slcfg = P["sl"]
                    if isinstance(_slcfg, str) and _slcfg.startswith("rs"):
                        thr = P["credit"] + float(_slcfg[2:]) / float(B["lot"])
                    else:
                        sl = BACKSTOP if _slcfg == "none" else _slcfg / 100.0
                        thr = (1 + sl) * P["credit"]
                    reason = None
                    if P["last_comb"] is not None and P["streak"] >= 2:
                        reason = "SL_DWELL"          # dwell confirmed on prior polls -> exit THIS poll
                    if comb >= thr: P["streak"] += 1
                    else: P["streak"] = 0
                    P["last_comb"] = comb
                    if now >= P["exit"]: reason = reason or "TIME_EXIT"
                    if now_force: reason = reason or "EOD_FORCE"
                    if "x_ce" in P or "x_pe" in P: reason = reason or "EXIT_RETRY"
                    if reason == "SL_DWELL" and B.get("mgmt") == "trail":
                        ce_l = q["%s:%s" % (B["seg"], ce_s)]["last_price"]
                        pe_l = q["%s:%s" % (B["seg"], pe_s)]["last_price"]
                        if (ce_l / max(P["ce0"], 0.05)) >= (pe_l / max(P["pe0"], 0.05)):
                            lose_e, lose_x, win_sym, win_e, win_l = P["ce0"], ce_l, pe_s, P["pe0"], pe_l
                        else:
                            lose_e, lose_x, win_sym, win_e, win_l = P["pe0"], pe_l, ce_s, P["ce0"], ce_l
                        P["realized_rs"] += (lose_e - lose_x) * B["qty"]
                        P.update(state="TRAIL", win_sym=win_sym, win_e=win_e, win_lo=win_l, streak=0)
                        push_event(st, sym, "ADJUST", "CSL hit: closed loser leg @ %.2f, TRAILING winner %s from %.2f (30%% bounce stop)" % (lose_x, win_sym, win_l))
                        save_state(st)
                        log("%s CSL->TRAIL: loser closed %.2f, winner %s @ %.2f" % (sym, lose_x, win_sym, win_l))
                        continue
                    if reason == "SL_DWELL" and B.get("mgmt") == "shift" and P["shifts"] < MAX_SHIFTS and now <= SHIFT_CUTOFF:
                        P["realized_rs"] += (P["credit"] - comb) * B["qty"]
                        sp = k.ltp([B["spot_key"]])[B["spot_key"]]["last_price"]
                        K = round(sp / B["step"]) * B["step"]
                        ce, pe, E = resolve_legs(k, B, K)
                        if ce and pe:
                            q2 = k.ltp(["%s:%s" % (B["seg"], ce["tradingsymbol"]), "%s:%s" % (B["seg"], pe["tradingsymbol"])])
                            P["ce0"] = q2["%s:%s" % (B["seg"], ce["tradingsymbol"])]["last_price"]
                            P["pe0"] = q2["%s:%s" % (B["seg"], pe["tradingsymbol"])]["last_price"]
                            P.update(K=K, legs=(ce["tradingsymbol"], pe["tradingsymbol"]), credit=P["ce0"] + P["pe0"],
                                     streak=0, last_comb=None, shifts=P["shifts"] + 1, expiry=str(E))
                            P["cost"] += 160
                            push_event(st, sym, "ADJUST", "CSL hit: SHIFTED to new ATM %d straddle @ %.2f credit (shift %d/%d)" % (K, P["credit"], P["shifts"], MAX_SHIFTS))
                            save_state(st)
                            log("%s CSL->SHIFT %d: new K=%d credit %.2f" % (sym, P["shifts"], K, P["credit"]))
                            continue
                    if reason:
                        exit_comb = comb
                        if P.get("live"):
                            tag = ("CSL_" + sym)[:20]
                            if "x_ce" not in P:
                                held_ce = broker_short_qty(k, ce_s)
                                if held_ce == 0:
                                    try: P["x_ce"] = q["%s:%s" % (B["seg"], ce_s)]["last_price"]
                                    except Exception: P["x_ce"] = P.get("ce_last") or P["ce0"]
                                    push_event(st, sym, "WARN", "MANUAL/EXTERNAL close detected on %s - no exit order placed, reconciled @ %.2f" % (ce_s, P["x_ce"]), "REAL")
                                    log("%s RECONCILE %s: broker holds no short - manual/external close, no BUY placed" % (sym, ce_s))
                                else:
                                    q_ce = B["qty"] if held_ce < 0 else min(B["qty"], held_ce)
                                    if q_ce != B["qty"]:
                                        log("%s PARTIAL RECONCILE %s: buying back %d of %d (broker-held)" % (sym, ce_s, q_ce, B["qty"]))
                                    oid_c = place_market(k, B, ce_s, "BUY", q_ce, tag)
                                    f = order_fill(k, oid_c) if oid_c else None
                                    if f is not None: P["x_ce"] = f
                            if "x_pe" not in P:
                                held_pe = broker_short_qty(k, pe_s)
                                if held_pe == 0:
                                    try: P["x_pe"] = q["%s:%s" % (B["seg"], pe_s)]["last_price"]
                                    except Exception: P["x_pe"] = P.get("pe_last") or P["pe0"]
                                    push_event(st, sym, "WARN", "MANUAL/EXTERNAL close detected on %s - no exit order placed, reconciled @ %.2f" % (pe_s, P["x_pe"]), "REAL")
                                    log("%s RECONCILE %s: broker holds no short - manual/external close, no BUY placed" % (sym, pe_s))
                                else:
                                    q_pe = B["qty"] if held_pe < 0 else min(B["qty"], held_pe)
                                    if q_pe != B["qty"]:
                                        log("%s PARTIAL RECONCILE %s: buying back %d of %d (broker-held)" % (sym, pe_s, q_pe, B["qty"]))
                                    oid_p = place_market(k, B, pe_s, "BUY", q_pe, tag)
                                    f = order_fill(k, oid_p) if oid_p else None
                                    if f is not None: P["x_pe"] = f
                            if "x_ce" not in P or "x_pe" not in P:
                                P["exit_fail"] = P.get("exit_fail", 0) + 1
                                log("%s LIVE EXIT INCOMPLETE (ce=%s pe=%s) attempt %d - retrying" % (
                                    sym, P.get("x_ce"), P.get("x_pe"), P["exit_fail"]))
                                if P["exit_fail"] in (1, 3, 6, 12):
                                    push_event(st, sym, "WARN", "LIVE exit incomplete (attempt %d) - auto-retrying. CHECK KITE." % P["exit_fail"], "REAL")
                                    save_state(st)
                                time.sleep(2)
                                continue
                            exit_comb = P["x_ce"] + P["x_pe"]
                        pnl = round(P["realized_rs"] + (P["credit"] - exit_comb) * B["qty"] - P["cost"])
                        rec = {"day": today, "book": sym, "sym": B["sym"], "dte": P["dte"], "cfg": "%s->%s SL%s" % (P["entry"], P["exit"], P["sl"]),
                               "strike": P["K"], "expiry": P["expiry"], "credit": round(P["credit"], 2),
                               "entry_ts": P["entry_ts"], "exit_ts": datetime.now().strftime("%H:%M:%S"),
                               "exit_comb": round(exit_comb, 2), "reason": reason, "pnl": pnl, "series": P.get("series", []),
                               "lots": B["lots"], "qty": B["qty"], "source": "REAL" if P.get("live") else "PAPER",
                               # per-leg detail so a booked trade can still be shown leg by leg
                               # (the live book is nulled on exit and used to take these with it)
                               "ce_sym": (P.get("legs") or [None, None])[0],
                               "pe_sym": (P.get("legs") or [None, None])[1],
                               "ce0": P.get("ce0"), "pe0": P.get("pe0"),
                               "ce_exit": P.get("x_ce", P.get("ce_last")),
                               "pe_exit": P.get("x_pe", P.get("pe_last"))}
                        st["records"].append(rec)
                        push_event(st, sym, "EXIT", "%s: closed %d straddle @ %.2f -> P&L %+d (%d lots, cum %+d)%s" % (
                            reason, P["K"], exit_comb, pnl, B["lots"], st["cum"].get(sym, 0) + pnl,
                            " [REAL MONEY]" if P.get("live") else ""), "REAL" if P.get("live") else "PAPER")
                        st["cum"][sym] = st["cum"].get(sym, 0) + pnl
                        save_state(st)
                        log("%s EXIT %s pnl %+d (cum %+d)" % (sym, reason, pnl, st["cum"][sym]))
                        del plans[sym]
                elif P["state"] == "TRAIL":
                    wkey = "%s:%s" % (B["seg"], P["win_sym"])
                    wl = (QB.get(wkey) or k.ltp([wkey])[wkey])["last_price"]
                    P["tick"] += 1
                    P["win_lo"] = min(P["win_lo"], wl)
                    if P["tick"] % SAMPLE_EVERY == 1:
                        P["series"].append([now, round(P["realized_rs"] + (P["win_e"] - wl) * B["qty"])])
                        write_live(plans, today)
                    reason = None
                    if P["streak"] >= 2: reason = "TRAIL_EXIT"
                    if wl >= P["win_lo"] * TRAIL_BOUNCE: P["streak"] += 1
                    else: P["streak"] = 0
                    if now >= P["exit"]: reason = reason or "TIME_EXIT"
                    if now_force: reason = reason or "EOD_FORCE"
                    if reason:
                        pnl = round(P["realized_rs"] + (P["win_e"] - wl) * B["qty"] - P["cost"])
                        rec = {"day": today, "book": sym, "sym": B["sym"], "dte": P["dte"],
                               "cfg": "%s->%s SL%s+trail" % (P["entry"], P["exit"], P["sl"]),
                               "strike": P["K"], "expiry": P.get("expiry"), "credit": round(P["credit"], 2),
                               "entry_ts": P.get("entry_ts"), "exit_ts": datetime.now().strftime("%H:%M:%S"),
                               "exit_comb": round(wl, 2), "reason": reason, "pnl": pnl, "series": P.get("series", []),
                               "lots": B["lots"], "qty": B["qty"], "source": "PAPER"}
                        st["records"].append(rec)
                        push_event(st, sym, "EXIT", "%s: winner leg closed @ %.2f -> day P&L %+d (%d lots)" % (reason, wl, pnl, B["lots"]))
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
