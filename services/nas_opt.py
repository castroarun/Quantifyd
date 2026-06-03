"""services/nas_opt.py — NAS-OPT paper-trading variant (research/54 refined system).

System: on each 0/1-DTE day, at 09:20 sell ~100pt-OTM NIFTY strangle (2 strikes OTM each side),
±0.4% underlying-move stop (full exit, one-and-done), else time-exit 14:45. PAPER ONLY (no Kite
orders). Marks entry/exit from the live options recorder (options_data.db) — the same data the
research/54 backtest used, so live == backtest by construction.

Scheduler (wired in app.py): entry 09:20, monitor every 1 min 09:21-14:44, exit 14:45 — Mon-Fri.
DB: backtest_data/nas_opt_trading.db. API: app.py /api/nas-opt/*.
"""
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, date

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
OPT_DB = ROOT / "backtest_data" / "options_data.db"
DB = ROOT / "backtest_data" / "nas_opt_trading.db"
LOT = 65
QTY = LOT * 2          # lots_per_leg = 2
BROK = 80              # per-leg round-trip brokerage
OTM_STRIKES = 2        # ~100 pts OTM each side (2 x 50)
MOVE_PCT = 0.4         # ±0.4% underlying-move stop
STRIKE_STEP = 50


def _conn():
    c = sqlite3.connect(str(DB)); c.row_factory = sqlite3.Row
    return c


def init_db():
    c = _conn()
    c.executescript(
        """
        CREATE TABLE IF NOT EXISTS nas_opt_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            day TEXT, dte INTEGER, weekday TEXT,
            entry_time TEXT, entry_spot REAL,
            ce_sym TEXT, pe_sym TEXT, ce_strike INTEGER, pe_strike INTEGER,
            ce_entry REAL, pe_entry REAL, credit REAL,
            exit_time TEXT, exit_reason TEXT, exit_spot REAL, ce_exit REAL, pe_exit REAL,
            pnl REAL, status TEXT DEFAULT 'OPEN', mode TEXT DEFAULT 'paper',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_nasopt_day ON nas_opt_positions(day);
        """
    )
    c.commit(); c.close()


# ---- live snapshot from the options recorder ----
def _snapshot(at_latest=True):
    """Return (snapshot_time, spot, dte, chain) from the most recent NIFTY chain snapshot today.
    chain: {(strike, type): (tradingsymbol, ltp)} for the nearest (current-week) expiry."""
    if not OPT_DB.exists():
        return None
    oc = sqlite3.connect(str(OPT_DB))
    today = date.today().isoformat()
    row = oc.execute("SELECT MAX(snapshot_time) FROM option_chain WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=?",
                     (today,)).fetchone()
    if not row or not row[0]:
        oc.close(); return None
    snap = row[0]
    rows = oc.execute(
        "SELECT tradingsymbol,strike,instrument_type,ltp,expiry_date,underlying_spot "
        "FROM option_chain WHERE symbol='NIFTY' AND snapshot_time=? AND ltp IS NOT NULL", (snap,)).fetchall()
    oc.close()
    if not rows:
        return None
    exps = sorted({r[4] for r in rows}); fut = [e for e in exps if e >= today]
    exp = fut[0] if fut else exps[-1]
    spot = next((r[5] for r in rows if r[5]), None)
    dte = (datetime.strptime(exp, "%Y-%m-%d").date() - date.today()).days
    chain = {(int(r[1]), r[2]): (r[0], float(r[3])) for r in rows if r[4] == exp and r[3] and r[3] > 0}
    return snap, spot, dte, chain


def _open_today():
    c = _conn(); r = c.execute("SELECT * FROM nas_opt_positions WHERE day=? ORDER BY id DESC LIMIT 1",
                               (date.today().isoformat(),)).fetchone(); c.close()
    return dict(r) if r else None


# ---- scheduler entry points ----
def entry_job():
    """09:20 — open a paper strangle if today is 0/1-DTE and we have not traded today."""
    init_db()
    snap = _snapshot()
    if not snap:
        logger.info("[NAS-OPT] entry: no chain snapshot yet"); return
    snap_t, spot, dte, chain = snap
    if dte > 1:
        logger.info(f"[NAS-OPT] entry skipped: DTE {dte} > 1 (trades 0/1-DTE only)"); return
    ex = _open_today()
    if ex:
        logger.info("[NAS-OPT] entry skipped: already have today's position"); return
    if not spot:
        logger.info("[NAS-OPT] entry: no spot"); return
    atm = round(spot / STRIKE_STEP) * STRIKE_STEP
    ce_k = atm + OTM_STRIKES * STRIKE_STEP; pe_k = atm - OTM_STRIKES * STRIKE_STEP
    ce = chain.get((ce_k, "CE")); pe = chain.get((pe_k, "PE"))
    if not ce or not pe:
        logger.warning(f"[NAS-OPT] entry: strikes {ce_k}CE/{pe_k}PE not in snapshot"); return
    c = _conn()
    c.execute("INSERT INTO nas_opt_positions (day,dte,weekday,entry_time,entry_spot,ce_sym,pe_sym,"
              "ce_strike,pe_strike,ce_entry,pe_entry,credit,status,mode) "
              "VALUES (?,?,?,?,?,?,?,?,?,?,?,?, 'OPEN','paper')",
              (date.today().isoformat(), dte, datetime.now().strftime("%a"), snap_t, spot,
               ce[0], pe[0], ce_k, pe_k, ce[1], pe[1], ce[1] + pe[1]))
    c.commit(); c.close()
    logger.info(f"[NAS-OPT] PAPER ENTRY {pe_k}PE@{pe[1]} + {ce_k}CE@{ce[1]} credit={ce[1]+pe[1]:.1f} spot={spot:.0f} dte={dte}")


def _close(pos, reason):
    snap = _snapshot()
    if not snap:
        return
    snap_t, spot, dte, chain = snap
    ce = chain.get((pos["ce_strike"], "CE")); pe = chain.get((pos["pe_strike"], "PE"))
    ce_x = ce[1] if ce else pos["ce_entry"]; pe_x = pe[1] if pe else pos["pe_entry"]
    pnl = ((pos["ce_entry"] - ce_x) + (pos["pe_entry"] - pe_x)) * QTY - 2 * BROK
    c = _conn()
    c.execute("UPDATE nas_opt_positions SET exit_time=?,exit_reason=?,exit_spot=?,ce_exit=?,pe_exit=?,pnl=?,status='CLOSED' WHERE id=?",
              (snap_t, reason, spot, ce_x, pe_x, round(pnl), pos["id"]))
    c.commit(); c.close()
    logger.info(f"[NAS-OPT] PAPER EXIT ({reason}) pnl={pnl:.0f} spot={spot:.0f}")


def monitor_job():
    """Every 1 min 09:21-14:44 — exit the open paper strangle on a ±0.4% underlying move."""
    pos = _open_today()
    if not pos or pos["status"] != "OPEN":
        return
    snap = _snapshot()
    if not snap:
        return
    _, spot, _, _ = snap
    if spot and pos["entry_spot"] and abs(spot - pos["entry_spot"]) / pos["entry_spot"] * 100 >= MOVE_PCT:
        _close(pos, "move0.4%")


def exit_job():
    """14:45 — time-exit any open paper strangle."""
    pos = _open_today()
    if pos and pos["status"] == "OPEN":
        _close(pos, "time1445")


# ---- API getters ----
def get_state():
    init_db()
    pos = _open_today()
    c = _conn()
    tot = c.execute("SELECT COALESCE(SUM(pnl),0), COUNT(*) FROM nas_opt_positions WHERE status='CLOSED'").fetchone()
    c.close()
    return {"name": "NAS-OPT", "mode": "paper",
            "system": "0/1-DTE ~100pt-OTM strangle + ±0.4% move-stop, 09:20 entry, 14:45 exit",
            "today": pos, "closed_total_pnl": round(tot[0] or 0), "closed_trades": tot[1]}


def get_trades(limit=200):
    init_db(); c = _conn()
    rows = [dict(r) for r in c.execute("SELECT * FROM nas_opt_positions ORDER BY id DESC LIMIT ?", (limit,))]
    c.close(); return rows


def get_equity():
    init_db(); c = _conn()
    rows = c.execute("SELECT day, pnl FROM nas_opt_positions WHERE status='CLOSED' ORDER BY day").fetchall()
    c.close()
    cum = 0.0; out = []
    for r in rows:
        cum += r[1] or 0; out.append({"day": r[0], "pnl": r[1], "cum": round(cum)})
    return out

def register(app, scheduler):
    """Wire NAS-OPT into the Flask app + APScheduler (called once from app.py)."""
    from flask import jsonify
    init_db()
    app.add_url_rule('/api/nas-opt/state', 'nas_opt_state', lambda: jsonify(get_state()))
    app.add_url_rule('/api/nas-opt/trades', 'nas_opt_trades', lambda: jsonify(get_trades()))
    app.add_url_rule('/api/nas-opt/equity', 'nas_opt_equity', lambda: jsonify(get_equity()))
    scheduler.add_job(entry_job, 'cron', day_of_week='mon-fri', hour=9, minute=20, id='nas_opt_entry', replace_existing=True)
    scheduler.add_job(monitor_job, 'cron', day_of_week='mon-fri', hour='9-14', minute='*', id='nas_opt_monitor', replace_existing=True)
    scheduler.add_job(exit_job, 'cron', day_of_week='mon-fri', hour=14, minute=45, id='nas_opt_exit', replace_existing=True)
    logger.info('[NAS-OPT] registered: 3 API routes + entry(09:20)/monitor(1min)/exit(14:45) paper jobs')
