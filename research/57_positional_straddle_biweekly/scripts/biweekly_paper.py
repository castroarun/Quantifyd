"""Forward PAPER logger v2 — Positional Bi-Weekly Short Straddle (research/57).

Standalone (cron, NOT in gunicorn -> no service restart). PAPER ONLY, 1 lot, NO Kite orders.
Reads the live options recorder (options_data.db latest snapshot) -> live == backtest.
DB: research/57.../results/paper_straddle.db.  Log: /tmp/biweekly_paper.log

RECIPE v2 (locked SIGNAL, G0-G6):
  - 09:20: if flat, SELL ATM straddle in the 2nd-NEAREST weekly expiry (bi-weekly ~8-12 DTE).
  - every ~5 min: CRASH stop -> if |spot-entry|>=2.0% exit immediately (-> re-enter next 09:20).
  - 15:20 (EOD): exit if |spot-entry|>=1.5% (move-stop) OR profit>=40% of credit (PT)
                 OR expiry DTE<=1 (roll). On an EOD exit, RE-ENTER IMMEDIATELY at 15:20 (stay short
                 overnight -> captures overnight theta). Else just keep holding.
  - OVERNIGHT WINGS: whenever holding a straddle past 15:20, BUY far-OTM wings (+/-500pt ~2%) at
    15:20; SELL them next 09:20. Caps the overnight gap that staying-short exposes. Tracked SEPARATELY.
Run by cron every 5 min, Mon-Fri 09:15-15:30 IST. Idempotent per tick.
"""
import sqlite3, logging
from pathlib import Path
from datetime import datetime, date

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("bw_paper")
ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("/home/arun/quantifyd/research/57_positional_straddle_biweekly")
OPT_DB = Path("/home/arun/quantifyd/backtest_data/options_data.db")
DB = ROOT / "results" / "paper_straddle.db"
LOT = 65; BROK = 80
MOVE_STOP = 1.5; PT = 40; CRASH = 2.0; ROLL_DTE = 1; WING_PTS = 500
ENTRY_LO, ENTRY_HI = "09:20", "09:40"
MORN_LO, MORN_HI = "09:18", "09:45"     # wing-settle window
EOD_LO, EOD_HI = "15:18", "15:30"


def conn():
    c = sqlite3.connect(str(DB)); c.row_factory = sqlite3.Row; return c

def init_db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = conn()
    c.executescript("""
      CREATE TABLE IF NOT EXISTS positions(
        id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT DEFAULT 'OPEN',
        entry_day TEXT, entry_time TEXT, expiry TEXT, dte_entry INTEGER,
        strike INTEGER, ce_sym TEXT, pe_sym TEXT, ce_entry REAL, pe_entry REAL, credit REAL, entry_spot REAL,
        exit_time TEXT, exit_reason TEXT, exit_spot REAL, ce_exit REAL, pe_exit REAL, pnl REAL, mode TEXT DEFAULT 'paper');
      CREATE TABLE IF NOT EXISTS wings(
        id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT DEFAULT 'HELD',
        bought_time TEXT, expiry TEXT, ce_strike INTEGER, pe_strike INTEGER,
        ce_sym TEXT, pe_sym TEXT, ce_buy REAL, pe_buy REAL, debit REAL,
        sold_time TEXT, ce_sell REAL, pe_sell REAL, pnl REAL, straddle_id INTEGER);
      CREATE TABLE IF NOT EXISTS actions(ts TEXT, action TEXT, detail TEXT);
    """)
    c.commit(); c.close()

def act(action, detail=""):
    c = conn(); c.execute("INSERT INTO actions(ts,action,detail) VALUES(?,?,?)",
                          (datetime.now().isoformat()[:19], action, detail)); c.commit(); c.close()
    log.info("%s | %s", action, detail)

def snapshot():
    if not OPT_DB.exists(): return None
    oc = sqlite3.connect(str(OPT_DB)); today = date.today().isoformat()
    row = oc.execute("SELECT MAX(snapshot_time) FROM option_chain WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=?", (today,)).fetchone()
    if not row or not row[0]: oc.close(); return None
    snap = row[0]
    rows = oc.execute("SELECT tradingsymbol,strike,instrument_type,ltp,expiry_date,underlying_spot FROM option_chain "
                      "WHERE symbol='NIFTY' AND snapshot_time=? AND ltp IS NOT NULL", (snap,)).fetchall()
    oc.close()
    if not rows: return None
    exps = sorted({r[4] for r in rows if r[4] >= today})
    spot = next((r[5] for r in rows if r[5]), None)
    chain = {(int(r[1]), r[2], r[4]): (r[0], float(r[3])) for r in rows}
    return dict(snap=snap, spot=spot, exps=exps, chain=chain)

def open_pos():
    c = conn(); r = c.execute("SELECT * FROM positions WHERE status='OPEN' ORDER BY id DESC LIMIT 1").fetchone(); c.close()
    return dict(r) if r else None

def open_wing():
    c = conn(); r = c.execute("SELECT * FROM wings WHERE status='HELD' ORDER BY id DESC LIMIT 1").fetchone(); c.close()
    return dict(r) if r else None

def dte(expiry):
    return (datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days

def enter(s):
    spot = s["spot"]
    if not spot or len(s["exps"]) < 2: act("ENTER-SKIP", "no spot or <2 expiries"); return None
    E = s["exps"][1]; K = round(spot / 50) * 50
    ce = s["chain"].get((K, "CE", E)); pe = s["chain"].get((K, "PE", E))
    if not ce or not pe: act("ENTER-SKIP", f"ATM {K} not in chain {E}"); return None
    credit = ce[1] + pe[1]
    c = conn()
    cur = c.execute("INSERT INTO positions(status,entry_day,entry_time,expiry,dte_entry,strike,ce_sym,pe_sym,ce_entry,pe_entry,credit,entry_spot)"
                    " VALUES('OPEN',?,?,?,?,?,?,?,?,?,?,?)",
                    (date.today().isoformat(), s["snap"], E, dte(E), K, ce[0], pe[0], ce[1], pe[1], credit, spot))
    c.commit(); pid = cur.lastrowid; c.close()
    act("ENTER", f"SELL {K} straddle {E} | CE {ce[1]} + PE {pe[1]} = credit {credit:.1f} | spot {spot:.0f} dte {dte(E)}")
    return pid

def close(pos, s, reason):
    E, K = pos["expiry"], pos["strike"]
    ce = s["chain"].get((K, "CE", E)); pe = s["chain"].get((K, "PE", E))
    ce_x = ce[1] if ce else pos["ce_entry"]; pe_x = pe[1] if pe else pos["pe_entry"]
    pnl = ((pos["ce_entry"] - ce_x) + (pos["pe_entry"] - pe_x)) * LOT - 2 * BROK
    c = conn()
    c.execute("UPDATE positions SET status='CLOSED',exit_time=?,exit_reason=?,exit_spot=?,ce_exit=?,pe_exit=?,pnl=? WHERE id=?",
              (s["snap"], reason, s["spot"], ce_x, pe_x, round(pnl), pos["id"]))
    c.commit(); c.close()
    act("EXIT", f"{reason} | buyback CE {ce_x} + PE {pe_x} | pnl {pnl:+.0f} | spot {s['spot']:.0f}")

def buy_wings(s, pos):
    E, K = pos["expiry"], pos["strike"]
    ck = K + WING_PTS; pk = K - WING_PTS
    ce = s["chain"].get((ck, "CE", E)); pe = s["chain"].get((pk, "PE", E))
    if not ce or not pe: act("WING-SKIP", f"wings {ck}CE/{pk}PE not in chain"); return
    debit = ce[1] + pe[1]
    c = conn()
    c.execute("INSERT INTO wings(status,bought_time,expiry,ce_strike,pe_strike,ce_sym,pe_sym,ce_buy,pe_buy,debit,straddle_id)"
              " VALUES('HELD',?,?,?,?,?,?,?,?,?,?)",
              (s["snap"], E, ck, pk, ce[0], pe[0], ce[1], pe[1], debit, pos["id"]))
    c.commit(); c.close()
    act("WING-BUY", f"BUY {ck}CE {ce[1]} + {pk}PE {pe[1]} = debit {debit:.1f} (overnight gap cap)")

def sell_wings(w, s):
    ce = s["chain"].get((w["ce_strike"], "CE", w["expiry"])); pe = s["chain"].get((w["pe_strike"], "PE", w["expiry"]))
    ce_s = ce[1] if ce else 0.0; pe_s = pe[1] if pe else 0.0
    pnl = ((ce_s - w["ce_buy"]) + (pe_s - w["pe_buy"])) * LOT - 2 * BROK
    c = conn()
    c.execute("UPDATE wings SET status='SOLD',sold_time=?,ce_sell=?,pe_sell=?,pnl=? WHERE id=?",
              (s["snap"], ce_s, pe_s, round(pnl), w["id"]))
    c.commit(); c.close()
    act("WING-SELL", f"SELL wings CE {ce_s} + PE {pe_s} | wing pnl {pnl:+.0f}")

def tick():
    init_db()
    now = datetime.now(); hm = now.strftime("%H:%M")
    if now.weekday() >= 5: return
    s = snapshot()
    if not s: act("NO-SNAP", "no recorder snapshot yet"); return
    # MORNING: settle overnight wings
    if MORN_LO <= hm <= MORN_HI:
        w = open_wing()
        if w: sell_wings(w, s)
    pos = open_pos()
    # ENTRY (only when flat: first start, or morning after a crash-stop)
    if pos is None:
        if ENTRY_LO <= hm <= ENTRY_HI:
            enter(s)
        return
    # MANAGE
    K, E = pos["strike"], pos["expiry"]
    ce = s["chain"].get((K, "CE", E)); pe = s["chain"].get((K, "PE", E))
    if not ce or not pe: return
    movepct = abs(s["spot"] - pos["entry_spot"]) / pos["entry_spot"] * 100
    mtm = ((pos["ce_entry"] - ce[1]) + (pos["pe_entry"] - pe[1])) * LOT
    if movepct >= CRASH:
        close(pos, s, "crash-stop 2.0%%")   # flat -> re-enter next 09:20 (don't chase the crash)
        return
    if EOD_LO <= hm <= EOD_HI:
        exited = False
        if movepct >= MOVE_STOP:
            close(pos, s, "move-stop 1.5%%"); exited = True
        elif mtm >= PT / 100.0 * pos["credit"] * LOT:
            close(pos, s, "profit-target 40%%"); exited = True
        elif dte(E) <= ROLL_DTE:
            close(pos, s, "roll (DTE<=1)"); exited = True
        if exited:
            pos = open_pos()
            if pos is None:
                pid = enter(s)   # IMMEDIATE re-entry at 15:20 (stay short overnight)
                pos = open_pos() if pid else None
        # OVERNIGHT WINGS: holding past 15:20 + no wing yet -> buy wings
        if pos is not None and open_wing() is None:
            buy_wings(s, pos)
        if not exited:
            act("HOLD", f"overnight | spot {s['spot']:.0f} move {movepct:.2f}% mtm {mtm:+.0f}")

if __name__ == "__main__":
    tick()
