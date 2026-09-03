#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wed->Fri iron-condor PAPER book (research/80). Standalone; places NO orders.

The strategy research/80 landed on for the days NAS-OPT leaves idle:
  ENTER  Wednesday ~close: SELL a ~0.8%-OTM NIFTY strangle, BUY wings 1.0% beyond each short,
         front-of-next weekly (Tuesday) expiry -> DTE 6. 10 lots/leg = 650 qty (was 2 lots to 2026-09-03).
  MARK   live off the options recorder every run.
  EXIT   Friday ~close. Never held over the weekend, never into Mon/Tue -> never competes with
         NAS-OPT for margin. The wings ARE the stop (max loss known at entry).

Runs from system cron (every minute, 09:15-15:30, Mon-Fri), like the straddle paper book -- so it
needs NO Flask restart. Reads options_data.db, keeps state in backtest_data/condor_paper_state.json,
and writes static/app/condor_paper.json for the /app/straddles page.

Verdict is SIGNAL, not STRATEGY (see /app/backtest/fardte-rescue) -- this book accumulates honest,
zero-risk forward evidence while the open questions (portfolio correlation, real-chain validation)
are settled.
"""
import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta

ROOT = "/home/arun/quantifyd"
OPT = os.path.join(ROOT, "backtest_data", "options_data.db")
STATE = os.path.join(ROOT, "backtest_data", "condor_paper_state.json")
OUT = os.path.join(ROOT, "static", "app", "condor_paper.json")

LOT, LOTS = 65, 10
QTY = LOT * LOTS            # 650
SHORT_PCT = 0.008          # shorts ~0.8% OTM
WING_PCT = 0.010           # wings 1.0% beyond each short
BROK = 20                  # per leg, round trip
ENTRY_AFTER = "15:10"      # enter at the Wednesday close
EXIT_AFTER = "15:10"       # exit at the Friday close

oc = sqlite3.connect(f"file:{OPT}?mode=ro", uri=True)


def now_ist():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


DAY = sys.argv[1] if len(sys.argv) > 1 else now_ist().strftime("%Y-%m-%d")
HHMM = now_ist().strftime("%H:%M") if (len(sys.argv) <= 1) else "15:29"


def latest_spot(day, upto="23:59"):
    r = oc.execute("SELECT spot_price FROM underlying_spot WHERE symbol='NIFTY' "
                   "AND substr(snapshot_time,1,10)=? AND snapshot_time<=? AND spot_price>0 "
                   "ORDER BY snapshot_time DESC LIMIT 1", (day, f"{day}T{upto}:59")).fetchone()
    return float(r[0]) if r and r[0] else None


def ltp(day, expiry, strike, ot, upto="23:59"):
    r = oc.execute("SELECT ltp FROM option_chain WHERE symbol='NIFTY' AND expiry_date=? AND strike=? "
                   "AND instrument_type=? AND substr(snapshot_time,1,10)=? AND snapshot_time<=? "
                   "AND ltp>0 ORDER BY snapshot_time DESC LIMIT 1",
                   (expiry, strike, ot, day, f"{day}T{upto}:59")).fetchone()
    return float(r[0]) if r and r[0] else None


def next_tue_expiry(day):
    exps = sorted({r[0] for r in oc.execute(
        "SELECT DISTINCT expiry_date FROM option_chain WHERE symbol='NIFTY' "
        "AND substr(snapshot_time,1,10)=? AND expiry_date>=?", (day, day))})
    return exps[0] if exps else None


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"open": None, "history": []}


def save_state(st):
    json.dump(st, open(STATE, "w"), indent=2)


def structure_value(day, pos, upto="23:59"):
    """Cost to buy the whole condor back right now (points). None if a leg is unquotable."""
    E = pos["expiry"]
    vals = {}
    for lg in pos["legs"]:
        v = ltp(day, E, lg["strike"], lg["type"], upto)
        if v is None:
            v = lg.get("ltp") or lg["entry"]
        vals[(lg["strike"], lg["type"], lg["side"])] = v
    short_val = sum(v for (k, ot, sd), v in vals.items() if sd == "SELL")
    long_val = sum(v for (k, ot, sd), v in vals.items() if sd == "BUY")
    return short_val - long_val, vals


def mark_legs(pos, vals):
    out = []
    for lg in pos["legs"]:
        now = vals.get((lg["strike"], lg["type"], lg["side"]), lg.get("ltp") or lg["entry"])
        sign = 1 if lg["side"] == "SELL" else -1        # short profits as premium falls
        pnl = round(sign * (lg["entry"] - now) * QTY)
        out.append({**lg, "ltp": round(now, 2), "pnl": pnl})
    return out


def enter(day):
    spot = latest_spot(day, HHMM)
    E = next_tue_expiry(day)
    if not spot or not E:
        return None
    sc = round(spot * (1 + SHORT_PCT) / 50) * 50
    sp = round(spot * (1 - SHORT_PCT) / 50) * 50
    wc = round((sc + spot * WING_PCT) / 50) * 50
    wp = round((sp - spot * WING_PCT) / 50) * 50
    legs_def = [(sc, "CE", "SELL"), (sp, "PE", "SELL"), (wc, "CE", "BUY"), (wp, "PE", "BUY")]
    legs = []
    for k, ot, sd in legs_def:
        e = ltp(day, E, k, ot, HHMM)
        if e is None:
            return None
        legs.append({"strike": k, "type": ot, "side": sd, "qty": QTY,
                     "entry": round(e, 2), "ltp": round(e, 2), "pnl": 0,
                     "entry_time": HHMM})
    credit = ((legs[0]["entry"] + legs[1]["entry"]) - (legs[2]["entry"] + legs[3]["entry"]))
    dte = (datetime.strptime(E, "%Y-%m-%d").date() - datetime.strptime(day, "%Y-%m-%d").date()).days
    maxloss = round((abs(wc - sc) - credit) * QTY)      # wings cap the loss
    return {"entry_day": day, "expiry": E, "dte_at_entry": dte, "spot_at_entry": round(spot, 2),
            "credit": round(credit, 2), "credit_inr": round(credit * QTY),
            "max_loss_inr": maxloss, "legs": legs, "entry_time": HHMM}


def close(day, pos, vals):
    val, _ = structure_value(day, pos, HHMM)
    pnl = round((pos["credit"] - val) * QTY - 4 * BROK)
    return {"entry_day": pos["entry_day"], "exit_day": day, "expiry": pos["expiry"],
            "spot_at_entry": pos["spot_at_entry"], "spot_at_exit": round(latest_spot(day, HHMM) or 0, 2),
            "credit": pos["credit"], "exit_value": round(val, 2),
            "pnl": pnl, "legs": mark_legs(pos, vals)}


def main():
    st = load_state()
    wd = datetime.strptime(DAY, "%Y-%m-%d").weekday()   # Mon=0 .. Fri=4
    pos = st.get("open")

    if pos:
        val, vals = structure_value(DAY, pos, HHMM)
        pos["legs"] = mark_legs(pos, vals)
        pos["cur_value"] = round(val, 2)
        pos["day_pnl"] = round((pos["credit"] - val) * QTY - 4 * BROK)
        # EXIT: Friday at/after the close, or a fail-safe if it survived to Mon/Tue unclosed
        exit_now = (wd == 4 and HHMM >= EXIT_AFTER) or (wd in (0, 1))
        if exit_now:
            rec = close(DAY, pos, vals)
            st["history"].insert(0, rec)
            st["open"] = None
            pos = None
    elif wd == 2 and HHMM >= ENTRY_AFTER:               # ENTER only on Wednesday, at the close
        # one cycle per week
        if not any(h["entry_day"] == DAY for h in st["history"]):
            pos = enter(DAY)
            if pos:
                st["open"] = pos

    save_state(st)

    closed = st["history"]
    tot = sum(h["pnl"] for h in closed)
    wins = sum(1 for h in closed if h["pnl"] > 0)
    payload = {
        "name": "Wed→Fri Iron Condor",
        "subtitle": "~0.8% shorts / 1.0% wings · enter Wed close, exit Fri close · uses NAS-OPT's idle days",
        "study": "/app/backtest/fardte-rescue",
        "mode": "paper", "lots": LOTS, "qty": QTY,
        "open": st.get("open"),
        "closed_total_pnl": round(tot), "closed_trades": len(closed),
        "win_rate": round(100 * wins / len(closed)) if closed else None,
        "history": closed[:30],
        "updated": now_ist().strftime("%Y-%m-%d %H:%M:%S"),
    }
    json.dump(payload, open(OUT, "w"), indent=2)
    if st.get("open"):
        p = st["open"]
        print(f"[condor] OPEN {p['entry_day']} exp {p['expiry']} credit {p['credit']} "
              f"cur {p.get('cur_value')} day_pnl {p.get('day_pnl')}")
    else:
        print(f"[condor] FLAT. closed={len(closed)} total={round(tot)}")


if __name__ == "__main__":
    main()
