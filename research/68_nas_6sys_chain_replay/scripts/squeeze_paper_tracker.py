"""Paper-forward squeeze tracker (research/68 strategy question): does the ATR-squeeze ATM
straddle make money in PAPER, managed with the v3 +-0.4% move-stop + re-center? Runs every
5 min via cron during market hours. Pure paper — NO real orders, independent of the live
scanner (whose OTM-premium filter is buggy on expiry day).

Logic each run:
  - Compute squeeze: ATR(14) Wilder < SMA(ATR,50) on 5-min NIFTY (the scanner's method).
  - First fresh squeeze of the day (count>=MIN_BARS, in 09:30-14:30) -> PAPER-enter an ATM
    straddle (record entry spot, strikes, real CE/PE premiums).
  - Manage the open straddle: if |spot - entry_spot|/entry_spot >= 0.4% -> MOVE_STOP close
    both + re-center to the new ATM (if strike changed, still in window). 15:15 -> EOD close.
  - Append every closed straddle to squeeze_paper_log.csv (date, entry/exit, P&L, recenters).
State: backtest_data/squeeze_paper_state.json. Log: backtest_data/squeeze_paper_log.csv."""
import sys, json, csv, datetime as dt
from pathlib import Path
import pandas as pd

ROOT = Path("/home/arun/quantifyd"); sys.path.insert(0, str(ROOT))
import config as C
from services.kite_service import get_kite
from services.nas_scanner import NasScanner, get_current_week_expiry

STATE = ROOT / "backtest_data" / "squeeze_paper_state.json"
LOG = ROOT / "backtest_data" / "squeeze_paper_log.csv"
MOVE_PCT, MIN_BARS, STEP, QTY = 0.004, 1, 50, 65
WIN_S, WIN_E, EOD = "09:30", "14:30", "15:15"
COLS = ["date", "entry_time", "entry_spot", "strike", "ce_sym", "pe_sym", "ce_entry", "pe_entry",
        "exit_time", "exit_reason", "ce_exit", "pe_exit", "pnl", "recenters"]

k = get_kite()
sc = NasScanner(C.NAS_ATM_DEFAULTS)
now = dt.datetime.now(); hhmm = now.strftime("%H:%M"); today = now.date().isoformat()


def log(m): print("[%s] %s" % (now.strftime("%H:%M:%S"), m), flush=True)
def spot(): return k.ltp(["NSE:NIFTY 50"])["NSE:NIFTY 50"]["last_price"]


def ltp(sym):
    try: return k.ltp(["NFO:" + sym])["NFO:" + sym]["last_price"]
    except Exception: return None


def atm_legs(sp):
    atm = round(sp / STEP) * STEP
    exp = get_current_week_expiry(now.date())
    return atm, sc._build_tradingsymbol("CE", atm, exp), sc._build_tradingsymbol("PE", atm, exp)


def squeeze():
    df = pd.DataFrame(k.historical_data(256265, now - dt.timedelta(days=4), now, "5minute"))
    if df.empty: return False, 0
    df = df[df["date"].astype(str).str[11:16] >= "09:15"].reset_index(drop=True)
    h, l, c = df["high"], df["low"], df["close"]; pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean(); sma = atr.rolling(50).mean()
    sq = (atr < sma).dropna()
    if not len(sq): return False, 0
    cnt = 0
    for v in reversed(sq.tolist()):
        if v: cnt += 1
        else: break
    return bool(sq.iloc[-1]), cnt


def load_state():
    if STATE.exists():
        try: return json.load(open(STATE))
        except Exception: return {}
    return {}


def save_state(s): json.dump(s, open(STATE, "w"), indent=2)


def traded_today():
    if not LOG.exists(): return False
    return any(r["date"] == today for r in csv.DictReader(open(LOG)))


def close_pos(st, sp, reason):
    ce_x = ltp(st["ce_sym"]); pe_x = ltp(st["pe_sym"])
    if ce_x is None: ce_x = max(0.0, sp - st["strike"])      # expiry settlement fallback
    if pe_x is None: pe_x = max(0.0, st["strike"] - sp)
    pnl = round((st["ce_entry"] - ce_x) * QTY + (st["pe_entry"] - pe_x) * QTY)
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new: w.writeheader()
        w.writerow({"date": st["date"], "entry_time": st["entry_time"], "entry_spot": round(st["entry_spot"], 1),
                    "strike": st["strike"], "ce_sym": st["ce_sym"], "pe_sym": st["pe_sym"],
                    "ce_entry": st["ce_entry"], "pe_entry": st["pe_entry"], "exit_time": hhmm,
                    "exit_reason": reason, "ce_exit": round(ce_x, 2), "pe_exit": round(pe_x, 2),
                    "pnl": pnl, "recenters": st.get("recenters", 0)})
    log("CLOSED %s straddle: CE %.1f->%.1f PE %.1f->%.1f pnl=%+d (%s)"
        % (st["strike"], st["ce_entry"], ce_x, st["pe_entry"], pe_x, pnl, reason))
    return pnl


def open_pos(sp, recenters=0):
    atm, ce, pe = atm_legs(sp)
    ce_e, pe_e = ltp(ce), ltp(pe)
    if not ce_e or not pe_e:
        log("entry skipped — no premium for %s/%s" % (ce, pe)); return None
    log("PAPER ENTRY %s straddle: CE %.1f + PE %.1f (credit %.1f) spot %.1f%s"
        % (atm, ce_e, pe_e, ce_e + pe_e, sp, " [re-center]" if recenters else ""))
    return {"date": today, "entry_time": hhmm, "entry_spot": sp, "strike": atm, "ce_sym": ce,
            "pe_sym": pe, "ce_entry": ce_e, "pe_entry": pe_e, "recenters": recenters}


def main():
    try: sp = spot()
    except Exception as e: log("kite spot failed: %s" % e); return
    st = load_state()
    if st.get("date") and st["date"] != today:        # stale prior-day state
        save_state({}); st = {}
    is_sq, cnt = squeeze()
    log("NIFTY %.1f | squeezing=%s count=%d | open_paper=%s" % (sp, is_sq, cnt, bool(st.get("ce_sym"))))

    # manage open paper straddle
    if st.get("ce_sym"):
        if hhmm >= EOD:
            close_pos(st, sp, "eod_squareoff"); save_state({}); return
        if abs(sp - st["entry_spot"]) / st["entry_spot"] >= MOVE_PCT:
            close_pos(st, sp, "MOVE_STOP")
            new_atm = round(sp / STEP) * STEP
            if new_atm != st["strike"] and WIN_S <= hhmm <= WIN_E:
                save_state(open_pos(sp, st.get("recenters", 0) + 1) or {})
            else:
                save_state({})
        return

    # fresh entry — first squeeze of the day in the window
    if is_sq and cnt >= MIN_BARS and WIN_S <= hhmm <= WIN_E and not traded_today():
        ns = open_pos(sp)
        if ns: save_state(ns)


if __name__ == "__main__":
    main()
