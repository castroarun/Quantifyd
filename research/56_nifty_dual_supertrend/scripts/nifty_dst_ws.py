#!/usr/bin/env python3
"""
NIFTY Dual-Supertrend LIVE websocket ticker (sub-second, like NAS).

Standalone process (run with the venv python). Opens its OWN Kite websocket
(Kite allows multiple connections -> isolated from the NAS in-process ticker, no
gunicorn restart, no reactor sharing). Subscribes the DST open-leg instrument
tokens + the NIFTY 50 index, and on every tick overlays LIVE per-leg prices/P&L
onto the snapshot in paper_dst.db (throttled ~1s). The /app/dst page (polling
~2s) then shows near-real-time P&L.

Does NOT trade or manage — V4 open/close stays on the 15-min management cron;
this only refreshes live prices. Self-exits after market close.

Modes:
  (default)  run the websocket loop until ~15:35 IST / token death.
  once       resolve tokens, pull one kite.ltp snapshot, write the live overlay,
             print it, exit. (Verifiable after hours.)

Run:  /home/arun/quantifyd/venv/bin/python3 scripts/nifty_dst_ws.py [once]
"""
import os, json, sqlite3, time, threading, sys
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent            # /home/arun/quantifyd
RESDIR = HERE.parent / "results"
PAPER_DB = str(RESDIR / "paper_dst.db")
SRC_DB = str(ROOT / "backtest_data" / "options_data.db")
TOKEN_JSON = str(ROOT / "backtest_data" / "access_token.json")
SYM = "NIFTY"; LOT = 75
NIFTY_INDEX_TOKEN = 256265                   # NIFTY 50 spot

try:
    from dotenv import load_dotenv
    load_dotenv(str(ROOT / ".env")); load_dotenv()
except Exception:
    pass
KITE_API_KEY = os.getenv("KITE_API_KEY", "")


def log(m): print(f"[{datetime.utcnow()+timedelta(hours=5,minutes=30):%H:%M:%S} IST] {m}", flush=True)
def ist_now(): return datetime.utcnow() + timedelta(hours=5, minutes=30)

# ---------- paper db helpers ----------
def _con():
    c = sqlite3.connect(PAPER_DB); c.row_factory = sqlite3.Row; return c
def sget(c, k, d=None):
    r = c.execute("SELECT v FROM state WHERE k=?", (k,)).fetchone()
    return json.loads(r["v"]) if r else d
def sset(c, k, v):
    c.execute("INSERT INTO state(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=?",
              (k, json.dumps(v), json.dumps(v)))

def load_book_and_snap():
    c = _con()
    book = sget(c, "book", []) or []
    snap = sget(c, "snapshot", {}) or {}
    c.close()
    return book, snap

def leg_symbol(strike, it, expiry):
    """Resolve a leg's Kite tradingsymbol from the recorder's option_chain."""
    s = sqlite3.connect(f"file:{SRC_DB}?mode=ro", uri=True)
    r = s.execute("SELECT tradingsymbol FROM option_chain WHERE symbol=? AND "
                  "expiry_date=? AND strike=? AND instrument_type=? "
                  "ORDER BY id DESC LIMIT 1", (SYM, expiry, float(strike), it)).fetchone()
    s.close()
    return r[0] if r else None

def build_legmap(book):
    """[(spread_i, leg_i, tradingsymbol, qty, entry)] for every open leg."""
    out = []
    for si, sp in enumerate(book):
        exp = sp["ed"][:10] if len(sp["ed"]) >= 10 else sp["ed"]
        # expiry stored may be 'YYYY-MM-DD' or with time; option_chain expiry_date is date
        for li, (q, k, it, e) in enumerate(sp["legs"]):
            ts = leg_symbol(k, it, exp)
            out.append((si, li, ts, q, e, k, it))
    return out

# ---------- overlay snapshot with live prices ----------
def write_overlay(book, base_snap, ltp_by_sym, spot, legmap, source="ws"):
    obook = []; open_mtm = 0.0
    sym_of = {(si, li): ts for si, li, ts, q, e, k, it in legmap}
    for si, sp in enumerate(book):
        legs = []; smtm = 0.0
        credit = sum(-q * e for q, k, it, e in sp["legs"]) * LOT
        for li, (q, k, it, e) in enumerate(sp["legs"]):
            ts = sym_of.get((si, li))
            ltp = ltp_by_sym.get(ts)
            if ltp is None:
                cur = e; lp = 0.0          # no live price yet -> flat at entry
            else:
                cur = ltp; lp = q * (ltp - e) * LOT
            smtm += lp
            legs.append(dict(side=("SELL" if q < 0 else "BUY"), strike=k, type=it,
                             qty=(-q if q < 0 else q) * LOT, entry=round(e, 2),
                             cur=round(cur, 2), pnl=round(lp), entry_ts=sp["open_ts"]))
        obook.append(dict(role=sp["role"], dir=sp["dir"], expiry=sp["ed"][:10],
                          open_ts=sp["open_ts"], credit=round(credit),
                          mtm=round(smtm), legs=legs))
        open_mtm += smtm
    snap = dict(base_snap)
    snap.update(open_book=obook, open_mtm=round(open_mtm), n_open=len(book),
                equity_net=round(base_snap.get("realized_net", 0) + open_mtm),
                live_ts=f"{ist_now():%Y-%m-%dT%H:%M:%S}", live_source=source)
    if spot: snap["spot"] = round(spot, 2)
    c = _con(); sset(c, "snapshot", snap); c.commit(); c.close()
    return open_mtm

# ---------- kite ----------
def get_kite():
    from kiteconnect import KiteConnect
    at = json.load(open(TOKEN_JSON))["access_token"]
    k = KiteConnect(api_key=KITE_API_KEY); k.set_access_token(at)
    return k, at

def resolve_tokens(kite, legmap):
    """tradingsymbol -> token via kite.ltp; returns (sym->token, token->sym)."""
    syms = sorted({m[2] for m in legmap if m[2]})
    keys = [f"NFO:{s}" for s in syms]
    s2t = {}; t2s = {}
    if keys:
        q = kite.ltp(keys)
        for full, d in q.items():
            sym = full.split(":", 1)[1]; tok = d["instrument_token"]
            s2t[sym] = tok; t2s[tok] = sym
    return s2t, t2s

# ---------- once (after-hours testable) ----------
def run_once():
    book, snap = load_book_and_snap()
    if not book:
        log("no open spreads — nothing to price"); return
    legmap = build_legmap(book)
    miss = [m for m in legmap if not m[2]]
    if miss: log(f"WARN unresolved tradingsymbols: {[(m[5],m[6]) for m in miss]}")
    kite, _ = get_kite()
    s2t, t2s = resolve_tokens(kite, legmap)
    log(f"resolved {len(s2t)} tokens: {s2t}")
    # pull last_price via ltp for each leg symbol + index spot
    keys = [f"NFO:{m[2]}" for m in legmap if m[2]]
    q = kite.ltp(keys + ["NSE:NIFTY 50"])
    ltp_by_sym = {full.split(':',1)[1]: d["last_price"] for full, d in q.items()}
    spot = ltp_by_sym.get("NIFTY 50")
    mtm = write_overlay(book, snap, ltp_by_sym, spot, legmap, source="ltp-once")
    log(f"overlay written: spot={spot} open_mtm={mtm:.0f}")
    for m in legmap:
        log(f"  leg {m[6]} {m[5]} sym={m[2]} ltp={ltp_by_sym.get(m[2])} entry={m[4]}")

# ---------- live websocket loop ----------
class WS:
    def __init__(self):
        self.ltp_by_tok = {}; self.t2s = {}; self.s2t = {}
        self.legmap = []; self.book = []; self.snap = {}
        self.book_json = None; self.last_write = 0; self.kws = None; self.spot = None

    def reload_book(self):
        book, snap = load_book_and_snap()
        bj = json.dumps(book)
        self.snap = snap
        if bj == self.book_json:
            return False
        self.book_json = bj; self.book = book
        self.legmap = build_legmap(book)
        return True

    def resubscribe(self, kite):
        self.s2t, self.t2s = resolve_tokens(kite, self.legmap)
        toks = list(self.t2s.keys()) + [NIFTY_INDEX_TOKEN]
        self.t2s[NIFTY_INDEX_TOKEN] = "NIFTY 50"
        if self.kws and toks:
            self.kws.subscribe(toks); self.kws.set_mode(self.kws.MODE_LTP, toks)
            log(f"subscribed {len(toks)} tokens")

    def on_ticks(self, ws, ticks):
        for t in ticks:
            self.ltp_by_tok[t["instrument_token"]] = t.get("last_price")
        now = time.time()
        if now - self.last_write < 1.0:   # throttle snapshot writes to ~1/s
            return
        self.last_write = now
        ltp_by_sym = {self.t2s.get(tok): lp for tok, lp in self.ltp_by_tok.items()
                      if self.t2s.get(tok)}
        self.spot = ltp_by_sym.get("NIFTY 50", self.spot)
        try:
            write_overlay(self.book, self.snap, ltp_by_sym, self.spot, self.legmap)
        except Exception as e:
            log(f"overlay error: {e}")

    def on_connect(self, ws, resp):
        log("ws connected");
        toks = list(self.t2s.keys()) + [NIFTY_INDEX_TOKEN]
        if toks: ws.subscribe(toks); ws.set_mode(ws.MODE_LTP, toks)

    def on_close(self, ws, code, reason): log(f"ws closed {code} {reason}")
    def on_error(self, ws, code, reason): log(f"ws error {code} {reason}")

    def watchdog(self):
        """Exit cleanly after market close or token death (cron relaunches 09:15)."""
        while True:
            time.sleep(30)
            if ist_now().strftime("%H:%M") >= "15:35":
                log("after 15:35 IST — stopping ws");
                try: self.kws.close()
                except Exception: pass
                os._exit(0)
            # periodic book reload + resubscribe
            try:
                if self.reload_book():
                    from kiteconnect import KiteConnect
                    k, _ = get_kite(); self.resubscribe(k)
            except Exception as e:
                log(f"reload error: {e}")

    def run(self):
        from kiteconnect import KiteTicker
        self.reload_book()      # may be empty (flat) — still run all session
        kite, at = get_kite()
        self.s2t, self.t2s = resolve_tokens(kite, self.legmap)
        self.t2s[NIFTY_INDEX_TOKEN] = "NIFTY 50"
        log(f"resolved tokens: {self.s2t} (book has {len(self.book)} spreads)")
        self.kws = KiteTicker(KITE_API_KEY, at)
        self.kws.on_ticks = self.on_ticks
        self.kws.on_connect = self.on_connect
        self.kws.on_close = self.on_close
        self.kws.on_error = self.on_error
        threading.Thread(target=self.watchdog, daemon=True).start()
        log("connecting websocket…")
        self.kws.connect(threaded=False)   # blocks (Twisted reactor, one per process)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        run_once()
    else:
        WS().run()
