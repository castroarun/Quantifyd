"""TIMEB2 one-shot LIVE - NIFTY expiry-Tuesday afternoon straddle, research/125 slot.
13:15 -> 14:30, combined-SL 30% (2-poll dwell, 5s polls), 8 lots (qty 520), REAL.
User-directed 2026-08-25 ("lets implement this for 8 lots today, call it TimeB2").
Standalone one-shot because the daemon's plans froze at 09:12; imports the proven
order layer from csl_paper_exec (marketable-LIMIT + verify, margin/kill gates,
naked-leg unwind, broker-qty check before exit BUY)."""
import sys, time, json, os
from datetime import datetime
sys.path.insert(0, "/home/arun/quantifyd/research/111_sensex_manual_mgmt/scripts")
sys.path.insert(0, "/home/arun/quantifyd")
import csl_paper_exec as X

B = {**X.NIFTY_MKT, "lots": 8, "qty": 520, "mode": "live"}
ENTRY, EXIT_T, SL = "13:15", "14:30", 0.30
TAG = "TIMEB2_NIFTY"
OUTJ = "/home/arun/quantifyd/research/125_expiry_afternoon_straddle/results/timeb2_live_days.json"

def log(m): print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), m), flush=True)

def rec(**kw):
    rows = []
    if os.path.exists(OUTJ):
        rows = json.load(open(OUTJ))
    rows.append({"day": datetime.now().strftime("%Y-%m-%d"), **kw})
    json.dump(rows, open(OUTJ, "w"), indent=1)

log("TIMEB2 armed: %s->%s combined-SL%d%% qty %d (8 lots) REAL - research/125 slot" %
    (ENTRY, EXIT_T, SL * 100, B["qty"]))

while datetime.now().strftime("%H:%M") < ENTRY:
    time.sleep(2)
if datetime.now().strftime("%H:%M") > "13:25":
    log("ABORT: past entry window"); sys.exit(1)

k = X.kite()
ok, why = X.live_allowed()
if not ok:
    log("ABORT: gates - %s" % why); rec(status="ABORT_GATES", why=why); sys.exit(1)
mok, avail, need = X.margin_ok(k, B["lots"])
if not mok:
    log("ABORT: margin %.0f < need %.0f" % (avail, need)); rec(status="ABORT_MARGIN"); sys.exit(1)

spot = k.ltp([B["spot_key"]])[B["spot_key"]]["last_price"]
K = round(spot / B["step"]) * B["step"]
ce, pe, E = X.resolve_legs(k, B, K)
if not (ce and pe):
    log("ABORT: could not resolve legs"); rec(status="ABORT_LEGS"); sys.exit(1)
kc, kp = "%s:%s" % (B["seg"], ce["tradingsymbol"]), "%s:%s" % (B["seg"], pe["tradingsymbol"])
log("spot %.1f ATM %d expiry %s legs %s / %s" % (spot, K, E, ce["tradingsymbol"], pe["tradingsymbol"]))

q = k.ltp([kc, kp])
oid_ce = X.place_market(k, B, ce["tradingsymbol"], "SELL", B["qty"], TAG, q[kc]["last_price"])
f_ce = X.order_fill(k, oid_ce) if oid_ce else None
if not f_ce:
    log("ABORT: CE sell failed - nothing open"); rec(status="ABORT_CE_SELL"); sys.exit(1)
oid_pe = X.place_market(k, B, pe["tradingsymbol"], "SELL", B["qty"], TAG, q[kp]["last_price"])
f_pe = X.order_fill(k, oid_pe) if oid_pe else None
if not f_pe:
    log("PE sell failed -> UNWINDING naked CE")
    oid_u = X.place_market(k, B, ce["tradingsymbol"], "BUY", B["qty"], TAG)
    X.order_fill(k, oid_u, 30)
    rec(status="ABORT_PE_SELL_UNWOUND"); sys.exit(1)

credit = f_ce + f_pe
thr = credit * (1 + SL)
log("OPEN [LIVE] credit %.2f (CE %.2f + PE %.2f) | SL trigger %.2f | time exit %s" %
    (credit, f_ce, f_pe, thr, EXIT_T))

streak, reason, last_comb = 0, None, None
while True:
    if datetime.now().strftime("%H:%M") >= EXIT_T:
        reason = "TIME_EXIT"; break
    try:
        q = k.ltp([kc, kp])
        last_comb = q[kc]["last_price"] + q[kp]["last_price"]
        if last_comb >= thr:
            streak += 1
            if streak >= 2:
                reason = "SL_HIT"; break
        else:
            streak = 0
    except Exception as ex:
        log("poll err: %s" % str(ex)[:60])
    time.sleep(5)
log("EXIT trigger: %s (last comb %s)" % (reason, last_comb))

fills = {}
for leg, f0 in ((ce, f_ce), (pe, f_pe)):
    ts = leg["tradingsymbol"]
    held = X.broker_short_qty(k, ts)
    if held is None or held <= 0:
        log("RECONCILE %s: broker holds no short - external close, no BUY placed" % ts)
        fills[ts] = None
        continue
    oid = X.place_market(k, B, ts, "BUY", int(min(held, B["qty"])), TAG)
    fx = X.order_fill(k, oid, 30)
    if fx is None:
        log("EXIT %s FILL UNCONFIRMED - CHECK ORDERBOOK MANUALLY" % ts)
    fills[ts] = fx
    log("EXIT %s buy fill %s" % (ts, fx))

if all(v is not None for v in fills.values()):
    debit = sum(fills.values())
    pnl = round((credit - debit) * B["qty"])
    log("DONE [LIVE] %s credit %.2f -> debit %.2f | P&L %+d on 8 lots" % (reason, credit, debit, pnl))
    rec(status="DONE", reason=reason, strike=K, expiry=str(E), credit=round(credit, 2),
        debit=round(debit, 2), pnl=pnl, qty=B["qty"], lots=B["lots"],
        entry_fills={"CE": f_ce, "PE": f_pe}, exit_fills=fills)
else:
    log("DONE with UNVERIFIED exit fills - reconcile against orderbook")
    rec(status="DONE_UNVERIFIED", reason=reason, strike=K, expiry=str(E),
        credit=round(credit, 2), exit_fills=fills, qty=B["qty"], lots=B["lots"])
