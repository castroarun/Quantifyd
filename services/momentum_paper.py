"""services/momentum_paper.py — Momentum-30 Sub-Selection PAPER book (research/62 winner).

Live-reflective ₹20L paper deployment of the validated large-cap momentum system. FULLY
AUTOMATED: monthly rebalance + weekly NIFTYBEES-100DMA gate + daily 15-day Donchian stop, all
run by APScheduler. Net of ~0.3% round-trip cost; 20% STCG tracked & shown SEPARATELY (not baked
into NAV). This module NEVER places a real order — it only records paper fills, so there is no
live-trading risk.

System rules (the winner — research/62 STATUS-MD):
  Universe : top-200 NSE stocks by trailing-6mo median traded value (close×volume), rebuilt monthly
  Score    : 6m & 12m relative strength vs NIFTYBEES (rsblend); top-30 = the "ETF"
  Hold     : top-8 of the 30, equal-weight, 100% invested when risk-on
  Buffer   : keep a holding while it stays in the top-22 of the 30; else rotate to best un-owned
  Gate     : weekly — NIFTYBEES < 100-day SMA → liquidate to cash; back above → redeploy next rebal
  Stop     : daily — holding closes below its prior-15-day low → exit that stock to cash

DB: backtest_data/momentum_paper.db   API: /api/momentum-paper/*   Page: /app/momentum-paper
"""
from __future__ import annotations
import sqlite3, json, logging
from pathlib import Path
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "backtest_data" / "momentum_paper.db"
MARKET_DB = ROOT / "backtest_data" / "market_data.db"
BENCH = "NIFTYBEES"
EXCLUDE = {"NIFTYBEES", "NIFTY50", "BANKNIFTY", "INDIAVIX", "NIFTYJR", "NIFTYMID",
           "NIFTYIT", "FINNIFTY", "MIDCPNIFTY", "JUNIORBEES",
           # commodity / index / debt ETFs that rank high by traded value but aren't equities
           "GOLDBEES", "SILVERBEES", "LIQUIDBEES", "BANKBEES", "ICICIB22", "CPSEETF",
           "MON100", "MAFANG", "SETFNIF50", "SETFGOLD", "SETFNIFBK", "GOLDCASE",
           "KOTAKGOLD", "AXISGOLD", "HDFCGOLD", "GOLDSHARE", "GOLD1", "SILVER1"}


def _is_etf(sym):
    return sym.endswith("BEES") or sym.endswith("ETF") or sym.endswith("GOLD") \
        or sym.endswith("SILVER") or sym in EXCLUDE


N200_CACHE = ROOT / "backtest_data" / "nifty200_official.csv"
N200_URL = "https://niftyindices.com/IndexConstituent/ind_nifty200list.csv"


def _official_nifty200(force=False):
    """The OFFICIAL NSE Nifty-200 constituents (market-cap defined), from niftyindices.com.
    Cached locally; refreshed if older than ~20 days (index reconstitutes semi-annually).
    This is the real index list — not a traded-value proxy. Falls back to traded-value
    only if the list can't be loaded."""
    import csv as _csv
    import urllib.request
    try:
        stale = (not N200_CACHE.exists() or
                 (date.today() - date.fromtimestamp(N200_CACHE.stat().st_mtime)).days > 20)
        if force or stale:
            req = urllib.request.Request(N200_URL, headers={"User-Agent": "Mozilla/5.0"})
            N200_CACHE.write_bytes(urllib.request.urlopen(req, timeout=20).read())
    except Exception as e:
        logger.warning(f"[MP] Nifty200 list refresh failed (using cache): {e}")
    try:
        with open(N200_CACHE, newline="", encoding="utf-8-sig") as f:
            syms = [(r.get("Symbol") or "").strip() for r in _csv.DictReader(f)]
        syms = [s for s in syms if s and not _is_etf(s)]
        return syms if len(syms) >= 150 else None
    except Exception:
        return None


N50_CACHE = ROOT / "backtest_data" / "nifty50_official.csv"
NEXT50_CACHE = ROOT / "backtest_data" / "niftynext50_official.csv"
_TIER = {}


def _mcap_tier(sym):
    """Market-cap tier from official index membership (Nifty 200 = Nifty 100 + Midcap 100;
    Nifty 100 = Nifty 50 + Next 50). These indices ARE free-float-mcap defined, so the tier
    is a real size bucket: Nifty 50 ≈ mcap rank 1–50, Next 50 ≈ 51–100, Midcap ≈ 101–200."""
    import csv as _csv
    if not _TIER:
        for key, path in (("n50", N50_CACHE), ("nxt", NEXT50_CACHE)):
            try:
                with open(path, newline="", encoding="utf-8-sig") as f:
                    _TIER[key] = {(r.get("Symbol") or "").strip() for r in _csv.DictReader(f)}
            except Exception:
                _TIER[key] = set()
    if sym in _TIER.get("n50", set()):
        return "Nifty 50"
    if sym in _TIER.get("nxt", set()):
        return "Next 50"
    return "Midcap"

CFG = dict(
    capital=300_000,          # Rs3L allocated inside the shared account (NAS + manual trades
                              # + 30-odd personal holdings share the rest of the balance)        # ₹20 lakh
    n_hold=8,
    buffer=22,
    etf_size=30,
    universe_size=200,
    gate_sma=100,             # NIFTYBEES 100-day SMA
    donchian=15,              # 15-day low stop
    cost_rt=0.003,            # 0.3% round-trip (mostly STT; Zerodha delivery brokerage ≈ 0)
    stcg_pct=0.20,            # short-term capital-gains tax, shown separately
    cash_yield=0.065,         # idle/risk-off cash parked in a liquid fund (LIQUIDBEES) @6.5% p.a.
    liq_lookback=320,         # ~trailing 6mo (in trading days *1.6) for traded-value median
    refresh_candidates=380,   # how many top-liquidity names to refresh from Kite at rebalance
    # ── LIVE-execution guardrails (only consulted when live_mode is ON) ──
    shared_account=True,      # the account also runs NAS + 30-odd personal holdings + manual trades
    order_product="CNC",      # start unlevered. At Rs3L, 1.0x sizes all 8 slots cleanly and pays
                              # no MTF interest. Switch to "MTF" when stepping up to 1.3x.      # Arun's choice: fund the book via the MTF facility, not full cash
    clash_alert=True,         # email when the book buys a name already held personally (merged line)
    live_max_order_value=1_500_000,  # per-order sanity cap (₹) — refuse any single live order above this
    live_fill_timeout=90,     # seconds to poll a MARKET order for a COMPLETE fill before giving up
    live_slippage_alert=0.01,  # log a SLIPPAGE alert if |fill − expected| exceeds 1% of expected
    live_rebalance_trim=False,
    # ── PUT HEDGE (research/105: bi-weekly 2x is the best tenor/ratio) ──
    hedge_enabled=False,      # NOT VIABLE yet — one NIFTY lot (75 x spot) is ~Rs18L of notional,
                              # so a 2x-of-equity hedge needs ~Rs9.1L of equity (~Rs12.4L capital
                              # at ~74% invested) before it lands on a whole lot. Below that the
                              # only choice is a large over-hedge. Live maths on the /app page.
                              # (was: NOT VIABLE at Rs3L: one NIFTY lot is Rs16L notional = 5.3x this
                              # book. Needs ~Rs8L equity to size to one lot. Until then the book
                              # runs the cash-exit gate, which beat the hedge over the full cycle.       # at a risk-off gate: hold the stocks + buy NIFTY puts instead of selling
    hedge_ratio=2.0,          # put notional = 2.0 x equity value
    hedge_dte_target=14,      # bi-weekly — the next week's expiry (best tenor tested)
    hedge_dte_min=8, hedge_dte_max=20,
    hedge_gate_daily=True,    # research/108b: check the 100-SMA EVERY EOD for the put decision
                              # (netCal 1.32 -> 1.43). The STOCK book stays weekly/monthly.
    hedge_resize_drift=0.25,  # re-size when notional drifts >25% from target (stocks stopping out)
    hedge_moneyness=0.0,      # ATM
    hedge_max_premium_pct=0.06,  # refuse to spend more than 6% of NAV on one hedge (sanity cap)
    sweep_symbol="CASHIETF",  # ICICI Pru BSE Liquid Rate ETF - GROWTH. Deliberately NOT
                              # LIQUIDCASE: Arun holds 17,276 of those PLEDGED in the same
                              # account, so the line would be ambiguous AND pledged units
                              # hide in collateral_quantity where _broker_qty() cannot see
                              # them. CASHIETF is unheld => every unit here is the book's.
                              # (growth-style; cleaner than
                                # LIQUIDBEES which pays daily fractional-unit dividends)
    sweep_min=25_000,         # don't bother sweeping less than this
    sweep_yield_actual=0.052,  # MEASURED 2026-08-17: CASHIETF 5.18% p.a. since listing, LIQUIDCASE
                              # 5.16%. The 6.5% in cash_yield is a backtest assumption that
                              # overstates real liquid-ETF returns by ~130bps — do not show it live.
    live_cash_sweep=True,     # ON 2026-08-16. Idle cash is stop-out cash awaiting the month-end
                              # rebalance (research/108 says do NOT redeploy it early), so it should
                              # at least EARN. Buy/sell orders are implemented; unsweep() caps the
                              # sell at the system's own sweep_units so personal units are safe.
                              # While False, LIVE mode accrues NO cash yield (idle cash in the Zerodha
                              # ledger genuinely earns 0) so live NAV is never overstated. PAPER keeps
                              # the 6.5% accrual so it stays comparable with the backtest.
    cash_reserve_pct=0.03,    # NEVER deploy the last 3% of NAV — keeps the hedge premium always fundable
                              # (a 2x bi-weekly ATM put costs ~1.2-1.5% of NAV; 3% gives ~2x headroom)  # v1 monthly rebalance policy: never TRIM kept winners (top-up only) →
                                # avoids partial-lot STCG churn; new names sized to equal weight
)

# Each rule = (name, frequency of check/action, what happens)
RULES = [
    ("Universe", "fixed (index reconstitutes semi-annually)",
     "The Nifty 200 — the 200 largest NSE stocks by (free-float) market cap. The entire candidate pool."),
    ("Selection", "at each monthly rebalance",
     "Rank all 200 by momentum (6-month + 12-month relative strength vs NIFTYBEES); hold the top 8, equal-weight."),
    ("Capital", "—",
     "₹20,00,000 · 100% invested when risk-on · equal-weight across the 8 names (~₹2.5L each)."),
    ("Rebalance + buffer", "MONTHLY — last trading day, ~14:45 IST (early, ~45-min runway)",
     "Re-rank the 200 (the heavy step, run early to leave time to catch issues). Keep a holding while it stays in the top 22; if it drops out, sell it and buy the best-ranked name not already owned."),
    ("Macro gate", "WEEKLY — last trading day of week, ~15:15 IST (pre-close)",
     "If NIFTYBEES is below its 100-day SMA → liquidate ALL 8 to cash. Redeploys at the next month-end once it reclaims the 100-DMA."),
    ("Hedge gate", "DAILY — ~15:15 IST",
     "The put decision is checked every EOD against the NIFTYBEES 100-day SMA: buy on a breach, sell "
     "when it is reclaimed (research/108b: net Calmar 1.32 -> 1.43 vs checking only weekly)."),
    ("Donchian stop", "DAILY — ~15:05 IST (before the 15:15-15:20 closing-auction window, when NSE rejects new orders)",
     "If any holding is below its own prior-15-day low → exit just that one stock to cash."),
    ("Hedge cash reserve", "continuous",
     "Never deploy the last 3% of NAV — guarantees the bi-weekly put premium can always be funded, "
     "so the book is never left invested and unhedged through a risk-off gate."),
    ("Idle cash", "continuous",
     "Idle cash is stop-out cash waiting for the month-end rebalance (research/108: do NOT "
     "redeploy it early), so it is swept into CASHIETF (ICICI Pru liquid-rate ETF, ~6.5% p.a.) "
     "at the daily 15:05 job and released before the rebalance buys. The last 3% of NAV is "
     "never swept so a hedge premium stays payable in cash."),
    ("Costs", "per trade",
     "Net of ~0.3% round-trip (mostly STT; Zerodha delivery brokerage ≈ 0)."),
    ("Tax", "on booked gains",
     "20% STCG on gains held < 1 year — tracked & shown separately, not baked into NAV."),
]


# ───────────────────────── DB ─────────────────────────
def _conn():
    c = sqlite3.connect(str(DB)); c.row_factory = sqlite3.Row; return c


def init_db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = _conn()
    c.executescript("""
      CREATE TABLE IF NOT EXISTS mp_positions(
        symbol TEXT PRIMARY KEY, qty REAL, entry_date TEXT, entry_price REAL,
        invested REAL, peak_price REAL);
      CREATE TABLE IF NOT EXISTS mp_closed(
        id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, entry_date TEXT, entry_price REAL,
        exit_date TEXT, exit_price REAL, qty REAL, gross_pnl REAL, gross_pct REAL,
        cost REAL, net_pnl REAL, reason TEXT, holding_days INTEGER, stcg_tax REAL);
      CREATE TABLE IF NOT EXISTS mp_fills(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, symbol TEXT, side TEXT,
        price REAL, qty REAL, value REAL, cost REAL, reason TEXT);
      CREATE TABLE IF NOT EXISTS mp_nav(
        d TEXT PRIMARY KEY, equity REAL, cash REAL, nav REAL, invested_pct REAL,
        gate TEXT, bench_close REAL, unrealized REAL);
      CREATE TABLE IF NOT EXISTS mp_state(key TEXT PRIMARY KEY, val TEXT);
      CREATE TABLE IF NOT EXISTS mp_hedge(
        id INTEGER PRIMARY KEY CHECK(id=1), tsym TEXT, token INTEGER, strike REAL, expiry TEXT,
        lot INTEGER, qty REAL, entry_price REAL, entry_date TEXT, entry_spot REAL, cost REAL);
      CREATE TABLE IF NOT EXISTS mp_hedge_closed(
        id INTEGER PRIMARY KEY AUTOINCREMENT, tsym TEXT, strike REAL, expiry TEXT, qty REAL,
        entry_date TEXT, entry_price REAL, exit_date TEXT, exit_price REAL,
        cost REAL, proceeds REAL, pnl REAL, reason TEXT);
    """)
    c.commit(); c.close()


def _get(key, default=None):
    c = _conn(); r = c.execute("SELECT val FROM mp_state WHERE key=?", (key,)).fetchone(); c.close()
    return json.loads(r["val"]) if r else default


def _set(key, val):
    c = _conn(); c.execute("INSERT OR REPLACE INTO mp_state(key,val) VALUES(?,?)",
                           (key, json.dumps(val))); c.commit(); c.close()


# ───────────────────── kite / data ─────────────────────
def _kite():
    from services.kite_service import get_kite_with_refresh
    return get_kite_with_refresh()


def _dm():
    from services.data_manager import get_data_manager
    return get_data_manager(_kite())


def _live_prices(symbols):
    """Latest traded price per symbol via Kite (after close = the day's close)."""
    if not symbols:
        return {}
    out = {}
    try:
        k = _kite()
        keys = [f"NSE:{s}" for s in symbols]
        for i in range(0, len(keys), 200):
            q = k.quote(keys[i:i + 200])
            for key, v in q.items():
                out[key.split(":", 1)[1]] = float(v.get("last_price") or
                                                   (v.get("ohlc") or {}).get("close") or 0)
    except Exception as e:
        logger.warning(f"[MP] live price fetch failed: {e}")
    return {s: p for s, p in out.items() if p > 0}


# ───────────────────── LIVE execution (real Kite CNC orders) ─────────────────────
# This module is PAPER by default. Real orders are placed ONLY when the persisted
# `live_mode` setting is "1". Flip it via POST /api/momentum-paper/toggle-mode; kill via
# /api/momentum-paper/kill-switch. All order flow funnels through _buy/_sell, so turning the
# flag on is the single switch that makes the whole book trade real money.
def _is_live():
    return str(_get("live_mode", "0")) in ("1", "true", "True", "on")


def _market_open_now():
    from datetime import time as _t
    now = datetime.now()
    if now.weekday() >= 5:                     # Sat/Sun
        return False
    return _t(9, 15) <= now.time() <= _t(15, 30)


def _slippage_check(symbol, side, fill, expected):
    if expected and abs(fill - expected) / expected > CFG["live_slippage_alert"]:
        logger.warning(f"[MP-LIVE] SLIPPAGE {side} {symbol}: fill {fill:.2f} vs expected "
                       f"{expected:.2f} ({(fill / expected - 1) * 100:+.2f}%)")


def _alert(title, msg, priority="normal"):
    """Fire an alert email; never raises (dispatch must never break trading)."""
    try:
        from services.momentum_eod_report import send_alert
        send_alert(title, msg, priority)
    except Exception as _e:
        logger.error(f"[MP] alert dispatch failed: {_e}")


def _order_product(k):
    """CNC by default; MTF when configured (shared account funds the book via margin trading)."""
    p = (CFG.get("order_product") or "CNC").upper()
    if p == "MTF":
        return getattr(k, "PRODUCT_MTF", "MTF")
    if p == "NRML":
        return k.PRODUCT_NRML
    return k.PRODUCT_CNC


def _broker_qty():
    """Total quantity per symbol at the broker (holdings + carry-forward positions), across products.
    Includes Arun's personal shares — this is the ACCOUNT view, not the system's."""
    out = {}
    try:
        k = _kite()
        for h in k.holdings():
            # quantity = settled; t1_quantity = bought yesterday, not yet settled but still OURS and
            # sellable. On T+1 quantity is 0 and the whole position sits in t1_quantity.
            owned = int(h.get("quantity") or 0) + int(h.get("t1_quantity") or 0)
            out[h["tradingsymbol"]] = out.get(h["tradingsymbol"], 0) + owned
        for p in k.positions().get("net", []):
            if p.get("product") in ("CNC", "MTF") and int(p.get("quantity") or 0):
                out[p["tradingsymbol"]] = out.get(p["tradingsymbol"], 0) + int(p["quantity"])
    except Exception as e:
        logger.error(f"[MP-SHARED] broker view failed: {e}")
    return out


def _system_deployed(live=None):
    """Rupee value the SYSTEM currently has deployed (its own ledger, not the account)."""
    pos = _positions()
    if not pos:
        return 0.0
    live = live or _live_prices(list(pos))
    return sum(p["qty"] * (live.get(s) or p["entry_price"]) for s, p in pos.items())


def _capital_fence_ok(add_rupees, live=None):
    """The book may never exceed its allocated capital — the rest of the account belongs to NAS,
    manual positions and personal holdings."""
    cap = float(_get("capital", CFG["capital"]) or CFG["capital"])
    used = _system_deployed(live)
    if used + add_rupees <= cap * 1.02:
        return True
    logger.warning(f"[MP-SHARED] capital fence: deployed Rs{used:,.0f} + Rs{add_rupees:,.0f} "
                   f"would exceed the allocated Rs{cap:,.0f} — order refused")
    _alert("CAPITAL FENCE HIT",
           f"Momentum tried to deploy Rs{add_rupees:,.0f} on top of Rs{used:,.0f}, beyond its "
           f"allocated Rs{cap:,.0f}. Order refused. Raise the allocation if this is intended.", "normal")
    return False


def _flag_clash(symbol, sys_qty):
    """Warn when the book buys into a name already held personally — the broker line is now merged."""
    if not (CFG["shared_account"] and CFG["clash_alert"]):
        return
    try:
        existing = _broker_qty().get(symbol, 0)
    except Exception:
        return
    if existing > 0:
        logger.warning(f"[MP-SHARED] CLASH {symbol}: you already hold {existing}; system adding {sys_qty}")
        _alert("MERGED POSITION",
               f"Momentum bought {sys_qty} {symbol}, which you ALREADY hold ({existing} shares).\n"
               f"The broker now shows one merged line of {existing + sys_qty}.\n"
               f"The system will only ever sell its own {sys_qty} — your {existing} are protected by "
               f"the sell guard — but do not manually sell this line without adjusting the book.",
               "normal")


def _sellable_qty(symbol, want_qty):
    """Never sell more than the system owns, and never more than the broker actually has."""
    want = int(round(want_qty))
    if not CFG["shared_account"] or not _is_live():
        return want
    have = _broker_qty().get(symbol, 0)
    if have < want:
        logger.error(f"[MP-SHARED] {symbol}: system wants to sell {want} but broker shows {have}")
        _alert("SELL BLOCKED — LEDGER MISMATCH",
               f"The book believes it owns {want} {symbol} but the broker shows only {have}. "
               f"Selling only {max(0, have)}. Someone may have sold this line manually — reconcile "
               f"the book before it trades again.", "high")
        return max(0, have)
    return want


def _place_cnc_market(symbol, side, qty):
    """Place a real NSE CNC MARKET order and BLOCK until it fills. Returns (avg_price, filled_qty).
    Raises on rejection/timeout. Reached ONLY when live_mode is on — this spends real money."""
    import time as _time
    if str(_get("live_armed", "0")).lower() not in ("1", "true", "on", "yes"):
        _alert("LIVE ORDER BLOCKED", f"{side} {symbol} x{int(qty)} blocked — two-key safety: live_armed not set.", "high")
        raise RuntimeError("two-key safety: live_armed not set — order blocked")
    k = _kite()
    oid = k.place_order(
        variety=k.VARIETY_REGULAR, exchange=k.EXCHANGE_NSE, tradingsymbol=symbol,
        transaction_type=(k.TRANSACTION_TYPE_BUY if side == "BUY" else k.TRANSACTION_TYPE_SELL),
        quantity=int(qty), product=_order_product(k), order_type=k.ORDER_TYPE_MARKET,
        validity=k.VALIDITY_DAY, tag="MOMENTUM")
    logger.warning(f"[MP-LIVE] {side} {symbol} x{int(qty)} → order {oid} placed")
    deadline = _time.time() + CFG["live_fill_timeout"]
    while _time.time() < deadline:
        _time.sleep(1.5)
        try:
            hist = k.order_history(oid)
        except Exception as e:
            logger.warning(f"[MP-LIVE] order_history({oid}) error: {e}")
            continue
        last = hist[-1] if hist else {}
        status = (last.get("status") or "").upper()
        if status == "COMPLETE":
            avg = float(last.get("average_price") or 0)
            fq = int(last.get("filled_quantity") or qty)
            logger.warning(f"[MP-LIVE] {side} {symbol} FILLED {fq}@{avg:.2f} (order {oid})")
            return avg, fq
        if status in ("REJECTED", "CANCELLED"):
            _alert(f"LIVE ORDER {status}", f"{side} {symbol} x{int(qty)} — order {oid} {status}: {last.get('status_message')}", "high")
            raise RuntimeError(f"order {oid} {status}: {last.get('status_message')}")
    _alert("LIVE ORDER TIMEOUT", f"{side} {symbol} x{int(qty)} — order {oid} not filled within {CFG['live_fill_timeout']}s", "high")
    raise TimeoutError(f"order {oid} not COMPLETE within {CFG['live_fill_timeout']}s")


def reconcile_holdings():
    """Compare our book (mp_positions) vs actual Kite holdings; alert-only (no auto-correct)."""
    if not _is_live():
        return {"live": False, "note": "paper mode"}
    try:
        # holdings() alone misses same-day CNC buys (equity settles T+1); _broker_qty() adds
        # positions() so a freshly bought line is not mistaken for a missing one.
        broker = _broker_qty()
    except Exception as e:
        return {"live": True, "error": str(e)}
    ours = {s: int(round(p["qty"])) for s, p in _positions().items()}
    if CFG["shared_account"]:
        # The account holds NAS + 30-odd personal names, so broker > book is EXPECTED and not an
        # error. Only the dangerous direction is a real mismatch: the book claiming more than exists.
        diffs = [{"symbol": s, "book": ours[s], "broker": broker.get(s, 0), "issue": "book > broker"}
                 for s in ours if broker.get(s, 0) < ours[s]]
        merged = [{"symbol": s, "book": ours[s], "broker": broker.get(s, 0)}
                  for s in ours if broker.get(s, 0) > ours[s]]
        if merged:
            logger.info(f"[MP-SHARED] {len(merged)} merged lines (system + personal): "
                        + ", ".join(f"{m['symbol']} {m['book']}/{m['broker']}" for m in merged))
    else:
        diffs = [{"symbol": s, "book": ours.get(s, 0), "broker": broker.get(s, 0)}
                 for s in sorted(set(ours) | set(broker)) if ours.get(s, 0) != broker.get(s, 0)]
        merged = []
    # LIQUIDCASE is not an mp_position, so it escaped reconciliation entirely. Check the one
    # dangerous direction: the book claiming more swept units than the broker holds. Broker > book
    # is EXPECTED here — Arun holds liquid funds personally in the same account.
    _su = _sweep_units()
    if _su > 0:
        _bl = broker.get(CFG["sweep_symbol"], 0)
        if _bl < int(_su):
            diffs.append({"symbol": CFG["sweep_symbol"], "book": int(_su), "broker": _bl,
                          "issue": "book > broker (swept units)"})
    if diffs:
        logger.warning(f"[MP-LIVE] HOLDINGS MISMATCH (book vs broker): {diffs}")
        _alert("HOLDINGS MISMATCH", "Book vs broker holdings differ:\n" +
               "\n".join(f"  {d['symbol']}: book {d['book']} vs broker {d['broker']}" for d in diffs), "high")
    return {"live": True, "match": not diffs, "diffs": diffs,
            "merged": merged, "shared_account": CFG["shared_account"]}


def _toggle_mode(body):
    """Flip PAPER↔LIVE and optionally set live capital. LIVE means the next scheduled
    rebalance/exit places REAL Kite CNC orders. Body: {"live": true/false, "capital": <rupees>}."""
    want = str(body.get("live", "")).lower() in ("1", "true", "on", "yes")
    if body.get("arm") is not None:
        _set("live_armed", "1" if str(body.get("arm")).lower() in ("1", "true", "on", "yes") else "0")
    cap = body.get("capital")
    if cap is not None:
        _set("capital", float(cap))
    _set("live_mode", "1" if want else "0")
    logger.warning(f"[MP] *** MODE → {'LIVE (real orders)' if want else 'PAPER'} ***"
                   + (f"  capital ₹{float(cap):,.0f}" if cap is not None else ""))
    return {"live_mode": want, "mode": "LIVE" if want else "PAPER", "capital": _get("capital")}


def _kill_switch():
    """Emergency: force back to PAPER so no further real orders are placed. Existing broker
    positions are LEFT UNTOUCHED (square off manually or let the next risk-off gate exit them)."""
    _set("live_mode", "0")
    _set("live_armed", "0")
    logger.warning("[MP] *** KILL SWITCH → live_mode + live_armed OFF. Open positions untouched. ***")
    return {"live_mode": False, "killed": True,
            "note": "back to PAPER; broker positions unchanged — square off manually if needed"}


_PANEL_CACHE = {}

def _panel(start="2022-06-01"):
    # Memoize by (start, DB mtime): the daily panel is a ~1000x1600 pivot that costs
    # ~7s to build. get_state() is on the request path, so without this the page hangs.
    # Any write to market_data.db (e.g. the EOD refresh) bumps mtime and invalidates.
    import os
    try:
        key = (start, os.path.getmtime(str(MARKET_DB)))
    except OSError:
        key = None
    if key is not None and _PANEL_CACHE.get("key") == key:
        return _PANEL_CACHE["val"]
    con = sqlite3.connect(str(MARKET_DB))
    df = pd.read_sql("SELECT symbol,date,close,volume FROM market_data_unified "
                     "WHERE timeframe='day' AND close IS NOT NULL AND date>=? ORDER BY symbol,date",
                     con, params=(start,), parse_dates=["date"])
    con.close()
    df["tv"] = df["close"] * df["volume"].fillna(0)
    close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
    tv = df.pivot_table(index="date", columns="symbol", values="tv").sort_index()
    if key is not None:
        _PANEL_CACHE["key"] = key
        _PANEL_CACHE["val"] = (close, tv)
    return close, tv


def refresh_universe(full=False):
    """Pull fresh daily bars from Kite for the working universe (held + top-liquidity pool)."""
    try:
        official = _official_nifty200(force=full)
        if official:
            pool = official                        # refresh exactly the official Nifty 200
        else:
            close, tv = _panel()
            asof = close.index[-1]
            med = tv.loc[:asof].tail(CFG["liq_lookback"]).median().sort_values(ascending=False)
            pool = [s for s in med.index if not _is_etf(s)][:CFG["refresh_candidates"]]
        held = [r["symbol"] for r in _conn().execute("SELECT symbol FROM mp_positions")]
        syms = sorted(set(pool + held + [BENCH]))
        frm = (date.today() - timedelta(days=20 if not full else 400))
        ok, fail, _ = _dm().download_data(syms, "day", datetime.combine(frm, datetime.min.time()),
                                          datetime.now())
        logger.info(f"[MP] refresh: {ok} ok / {fail} fail ({len(syms)} syms)")
        return ok
    except Exception as e:
        logger.error(f"[MP] refresh failed: {e}")
        return 0


# ───────────────────── signals ─────────────────────
def _universe(close, tv, asof):
    official = _official_nifty200()
    if official:                                   # the REAL Nifty 200 (market-cap defined)
        return [s for s in official if s in close.columns]
    # fallback only: traded-value proxy (the backtest method) if the official list is unavailable
    w = tv.loc[:asof].tail(CFG["liq_lookback"])
    cnt = w.notna().sum(); med = w.median()
    elig = med[(cnt >= 75) & (med > 0)].sort_values(ascending=False)
    return [s for s in elig.index if not _is_etf(s)][:CFG["universe_size"]]


def _rs_basket(close, tv, asof):
    """Return ranked top-30 'ETF' by 6m/12m relative strength within the top-200 universe."""
    uni = set(_universe(close, tv, asof))
    h = close.loc[:asof].ffill()          # ffill: last panel date can miss the benchmark/some names
    if BENCH not in h.columns or len(h) <= 252 or pd.isna(h[BENCH].iloc[-1]):
        return None
    out = {}
    for L, wt in ((126, 0.5), (252, 0.5)):
        p0 = h.iloc[-L - 1]; p1 = h.iloc[-1]
        nf = p1[BENCH] / p0[BENCH]
        r = (p1 / p0) / nf
        for s, v in r.items():
            if s in uni and s not in EXCLUDE and pd.notna(v):
                out[s] = out.get(s, 0) + wt * v
    if not out:
        return None
    return list(pd.Series(out).sort_values(ascending=False).index[:CFG["etf_size"]])


def _gate_risk_off(close, asof):
    b = close[BENCH].loc[:asof].dropna()
    return bool(len(b) >= CFG["gate_sma"] and b.iloc[-1] < b.tail(CFG["gate_sma"]).mean())


def _donchian_low(close, sym, asof):
    cs = close[sym].loc[:asof].dropna() if sym in close.columns else pd.Series(dtype=float)
    n = CFG["donchian"]
    return float(cs.iloc[-n - 1:-1].min()) if len(cs) > n else None




# ───────────────────── LIQUIDCASE sweep ─────────────────────
def _sweep_units():
    return float(_get("sweep_units", 0.0) or 0.0)


def _sweep_price():
    try:
        q = _kite().quote([f"NSE:{CFG['sweep_symbol']}"])
        return float(q[f"NSE:{CFG['sweep_symbol']}"]["last_price"])
    except Exception:
        return float(_get("sweep_last_px", 0.0) or 0.0)


def _sweep_value():
    u = _sweep_units()
    return u * _sweep_price() if u else 0.0


def sweep_idle_cash():
    """Park cash ABOVE the hedge reserve into LIQUIDCASE. Never touches the reserve, so a put premium
    is always payable from real cash on the same day."""
    if not CFG["live_cash_sweep"]:
        return None
    nav = _equity_value() + _cash() + _sweep_value()
    reserve = nav * CFG["cash_reserve_pct"]
    spare = _cash() - reserve
    if spare < CFG["sweep_min"]:
        return None
    px = _sweep_price()
    if px <= 0:
        return None
    qty = int(spare // px)
    if qty <= 0:
        return None
    if _is_live():
        try:
            px, qty = _place_cnc_market(CFG["sweep_symbol"], "BUY", qty)
        except Exception as e:
            logger.error(f"[MP-SWEEP] buy failed: {e}")
            _alert("SWEEP BUY FAILED", str(e), "normal"); return None
    cost = px * qty
    _set("cash", _cash() - cost)
    _set("sweep_units", _sweep_units() + qty)
    _set("sweep_cost", float(_get("sweep_cost", 0.0) or 0.0) + cost)   # cost basis -> real gain
    if not _get("sweep_since"):        # opened a fresh position -> start the holding clock today
        _set("sweep_since", _dt.date.today().isoformat())
    _set("sweep_last_px", px)
    logger.warning(f"[MP-SWEEP] parked Rs{cost:,.0f} in {CFG['sweep_symbol']} ({qty} @ {px:.2f})")
    return dict(qty=qty, price=px, value=cost)


def unsweep(amount=None):
    """Sell LIQUIDCASE back to cash. amount=None sells everything (used before the monthly rebalance).
    Proceeds are available for stock purchases the same day."""
    u = _sweep_units()
    if u <= 0:
        return None
    px = _sweep_price()
    if px <= 0:
        return None
    qty = u if amount is None else min(u, int(amount // px) + 1)
    qty = int(min(u, qty))
    if qty <= 0:
        return None
    if _is_live():
        try:
            px, qty = _place_cnc_market(CFG["sweep_symbol"], "SELL", qty)
        except Exception as e:
            logger.error(f"[MP-SWEEP] sell failed: {e}")
            _alert("SWEEP SELL FAILED — CASH MAY BE SHORT", str(e), "high"); return None
    proceeds = px * qty
    _set("cash", _cash() + proceeds)
    _u_before = _sweep_units()
    _set("sweep_units", max(0.0, _u_before - qty))
    # retire the cost basis proportionally, so the remaining units keep an honest gain
    _c = float(_get("sweep_cost", 0.0) or 0.0)
    _set("sweep_cost", max(0.0, _c * (max(0.0, _u_before - qty) / _u_before)) if _u_before else 0.0)
    if _sweep_units() <= 0:            # flat again -> next buy restarts the holding clock
        _set("sweep_since", "")
    _set("sweep_last_px", px)
    logger.warning(f"[MP-SWEEP] released Rs{proceeds:,.0f} from {CFG['sweep_symbol']}")
    return dict(qty=qty, price=px, value=proceeds)


def _days_to_rebalance():
    """Calendar days until the month-end rebalance (when idle cash gets redeployed)."""
    from datetime import date as _d, timedelta as _td
    t = _d.today()
    nxt = (t.replace(day=28) + _td(days=4)).replace(day=1) - _td(days=1)
    return max(0, (nxt - t).days)


# ───────────────────── benchmark series (for the live P&L chart) ─────────────────────
# Real NSE index closes from market_data.db — NOT ETF proxies. research/86 burned us once by
# reasoning about smallcaps off a proxy that agreed with the real index only 68% of the time.
BENCH_SERIES = [
    ("NIFTY50",        "Nifty 50"),
    ("NIFTY500",       "Nifty 500"),
    ("NIFTYMIDCAP150", "Midcap 150"),
    ("NIFTYSMLCAP250", "Smallcap 250"),
]


def book_curve():
    """The book's cumulative return with DEPOSITS REMOVED (time-weighted).

    Raw NAV cannot be compared to an index: capital went Rs3L -> Rs7.69L in nine days, so plotting
    NAV growth would have shown +156% against Nifty 50's -2.15% when the book had in truth done
    almost nothing. Each day's return is measured against the PREVIOUS day's NAV after backing out
    that day's deposit or withdrawal, then chained — so adding cash moves the line by exactly zero,
    which is the entire point.
    """
    rows = list(_conn().execute(
        "SELECT d, nav, COALESCE(capital,0) FROM mp_nav WHERE nav > 0 ORDER BY d"))
    if not rows:
        return []
    out, cum = [], 1.0
    prev_nav, prev_cap = None, None
    for d, nav, cap in rows:
        if prev_nav:
            flow = (cap - prev_cap) if (cap and prev_cap) else 0.0
            r = ((nav - flow) / prev_nav) - 1.0 if prev_nav else 0.0
            cum *= (1.0 + r)
        out.append({"d": d[:10], "r": round((cum - 1.0) * 100, 4), "nav": round(nav)})
        prev_nav, prev_cap = nav, (cap or prev_cap)
    return out


def benchmark_series(start=None):
    """Daily closes for each comparison index from the book's inception.

    Cached for an hour: this feeds a chart that re-fetches on every page load, and the underlying
    daily bars only change once a day after the market closes."""
    import time as _t
    start = start or (_get("inception") or date.today().isoformat())[:10]
    ck = f"{start}"
    try:
        if _get("bench_cache_key") == ck and (_t.time() - float(_get("bench_cache_ts", 0) or 0)) < 3600:
            return json.loads(_get("bench_cache") or "{}")
    except Exception:
        pass
    out = {}
    try:
        con = sqlite3.connect(str(ROOT / "backtest_data" / "market_data.db"))
        for sym, label in BENCH_SERIES:
            rows = con.execute(
                "SELECT date, close FROM market_data_unified WHERE symbol=? AND timeframe='day' "
                "AND date >= ? AND close > 0 ORDER BY date", (sym, start)).fetchall()
            if rows:
                out[sym] = dict(label=label, points=[{"d": d[:10], "c": float(c)} for d, c in rows])
        con.close()
    except Exception as e:
        logger.error(f"[MP] benchmark_series failed: {e}")
        return {}
    try:
        _set("bench_cache", json.dumps(out)); _set("bench_cache_key", ck); _set("bench_cache_ts", _t.time())
    except Exception:
        pass
    return out


# ───────────────────── PUT HEDGE (bi-weekly 2x) ─────────────────────
def _hedge_get():
    r = _conn().execute("SELECT * FROM mp_hedge WHERE id=1").fetchone()
    return dict(r) if r else None


def _hedge_save(d):
    c = _conn()
    c.execute("INSERT OR REPLACE INTO mp_hedge(id,tsym,token,strike,expiry,lot,qty,entry_price,"
              "entry_date,entry_spot,cost) VALUES(1,?,?,?,?,?,?,?,?,?,?)",
              (d["tsym"], d["token"], d["strike"], d["expiry"], d["lot"], d["qty"],
               d["entry_price"], d["entry_date"], d["entry_spot"], d["cost"]))
    c.commit()


def _hedge_clear():
    c = _conn(); c.execute("DELETE FROM mp_hedge WHERE id=1"); c.commit()


def _nifty_spot():
    q = _kite().quote(["NSE:NIFTY 50"])
    return float(q["NSE:NIFTY 50"]["last_price"])


def _nifty_spot_cached(max_age_s=900):
    """Spot for DISPLAY maths only. get_state() runs on every page load and poll, so an uncached
    Kite call there would add latency and fail outright on a stale token. Never use this for order
    sizing — call _nifty_spot() directly for that."""
    import time as _t
    try:
        ts = float(_get("nifty_spot_ts", 0.0) or 0.0)
        px = float(_get("nifty_spot_px", 0.0) or 0.0)
    except Exception:
        ts, px = 0.0, 0.0
    if px > 0 and (_t.time() - ts) < max_age_s:
        return px
    try:
        px = _nifty_spot()
        _set("nifty_spot_px", px); _set("nifty_spot_ts", _t.time())
    except Exception:
        pass                      # keep the stale value rather than showing nothing
    return px


NIFTY_LOT = 75                    # NSE NIFTY F&O lot size


def _hedge_viability(equity, nav):
    """Can a 2x-of-equity NIFTY put hedge be sized to a WHOLE lot at today's book size?

    Below one lot the only options are no hedge, or one lot that massively over-hedges — which is
    the research/105 failure mode: as stocks stop out the fixed put position becomes a net-short
    directional bet rather than a hedge."""
    spot = _nifty_spot_cached()
    if not spot or not equity:
        return None
    ratio = CFG["hedge_ratio"]
    lot_notional = spot * NIFTY_LOT
    lots = (equity * ratio) / lot_notional
    equity_needed = lot_notional / ratio
    invested_frac = (equity / nav) if nav else 0.0
    capital_needed = (equity_needed / invested_frac) if invested_frac else None
    return dict(
        enabled=CFG["hedge_enabled"], ratio=ratio, lot=NIFTY_LOT,
        spot=round(spot, 1), lot_notional=round(lot_notional),
        equity=round(equity), target_notional=round(equity * ratio),
        lots_needed=round(lots, 2), viable=lots >= 1.0,
        over_hedge_x=round(lot_notional / equity, 1) if equity else None,
        equity_for_one_lot=round(equity_needed),
        capital_needed=(round(capital_needed) if capital_needed else None),
        capital_now=round(float(_get("capital", CFG["capital"]))),
        shortfall=(round(max(0.0, capital_needed - float(_get("capital", CFG["capital"]))))
                   if capital_needed else None),
        dte_target=CFG["hedge_dte_target"],
    )


def _opt_ltp(tsym):
    q = _kite().quote([f"NFO:{tsym}"])
    return float(q[f"NFO:{tsym}"]["last_price"])


def _hedge_pick():
    """Pick the ~14-DTE (bi-weekly) NIFTY PE nearest ATM. Returns contract dict or None."""
    from datetime import date as _d
    k = _kite()
    inst = [i for i in k.instruments("NFO")
            if i.get("name") == "NIFTY" and i.get("instrument_type") == "PE"]
    if not inst:
        return None
    today = _d.today()

    def dte(e):
        e = e.date() if hasattr(e, "date") else e
        return (e - today).days
    cand = sorted({i["expiry"] for i in inst
                   if CFG["hedge_dte_min"] <= dte(i["expiry"]) <= CFG["hedge_dte_max"]},
                  key=lambda e: abs(dte(e) - CFG["hedge_dte_target"]))
    if not cand:
        logger.warning("[MP-HEDGE] no expiry in the %d-%d DTE window",
                       CFG["hedge_dte_min"], CFG["hedge_dte_max"])
        return None
    exp = cand[0]
    spot = _nifty_spot()
    target = spot * (1 + CFG["hedge_moneyness"])
    pool = [i for i in inst if i["expiry"] == exp]
    best = min(pool, key=lambda i: abs(float(i["strike"]) - target))
    e = best["expiry"]
    return dict(tsym=best["tradingsymbol"], token=int(best["instrument_token"]),
                strike=float(best["strike"]),
                expiry=(e.date() if hasattr(e, "date") else e).isoformat(),
                lot=int(best["lot_size"]), spot=spot, dte=dte(best["expiry"]))


def _hedge_target_qty(equity, spot, lot):
    """Whole lots closest to ratio x equity of notional."""
    if spot <= 0 or lot <= 0:
        return 0
    lots = round(CFG["hedge_ratio"] * equity / (spot * lot))
    return int(max(0, lots) * lot)


def _equity_value(live=None):
    pos = _positions()
    if not pos:
        return 0.0
    live = live or _live_prices(list(pos))
    return sum(p["qty"] * (live.get(s) or p["entry_price"]) for s, p in pos.items())


def _place_opt_market(tsym, side, qty):
    """Real NFO option MARKET order (NRML), blocking until filled. Mirrors _place_cnc_market,
    including the two-key safety. Reached ONLY when live_mode is on."""
    import time as _time
    if str(_get("live_armed", "0")).lower() not in ("1", "true", "on", "yes"):
        _alert("LIVE ORDER BLOCKED", f"{side} {tsym} x{int(qty)} blocked — live_armed not set.", "high")
        raise RuntimeError("two-key safety: live_armed not set — option order blocked")
    k = _kite()
    oid = k.place_order(
        variety=k.VARIETY_REGULAR, exchange=k.EXCHANGE_NFO, tradingsymbol=tsym,
        transaction_type=(k.TRANSACTION_TYPE_BUY if side == "BUY" else k.TRANSACTION_TYPE_SELL),
        quantity=int(qty), product=k.PRODUCT_NRML, order_type=k.ORDER_TYPE_MARKET,
        validity=k.VALIDITY_DAY, tag="MOM-HEDGE")
    logger.warning(f"[MP-HEDGE-LIVE] {side} {tsym} x{int(qty)} → order {oid} placed")
    deadline = _time.time() + CFG["live_fill_timeout"]
    while _time.time() < deadline:
        _time.sleep(1.5)
        try:
            hist = k.order_history(oid)
        except Exception as e:
            logger.warning(f"[MP-HEDGE-LIVE] order_history({oid}) error: {e}")
            continue
        last = hist[-1] if hist else {}
        status = (last.get("status") or "").upper()
        if status == "COMPLETE":
            avg = float(last.get("average_price") or 0)
            fq = int(last.get("filled_quantity") or qty)
            logger.warning(f"[MP-HEDGE-LIVE] {side} {tsym} FILLED {fq}@{avg:.2f}")
            return avg, fq
        if status in ("REJECTED", "CANCELLED"):
            _alert(f"HEDGE ORDER {status}", f"{side} {tsym} x{int(qty)}: {last.get('status_message')}", "high")
            raise RuntimeError(f"option order {oid} {status}: {last.get('status_message')}")
    _alert("HEDGE ORDER TIMEOUT", f"{side} {tsym} x{int(qty)} not filled in {CFG['live_fill_timeout']}s", "high")
    raise TimeoutError(f"option order {oid} not COMPLETE within {CFG['live_fill_timeout']}s")


def _hedge_fund(amount, d):
    """Premium is paid from idle cash only. The book normally carries enough (Donchian stop-outs and
    the debt sleeve leave cash idle); if it does not, we SKIP the hedge rather than force-sell stock —
    trimming winners to buy insurance is exactly the churn this book was fixed to avoid."""
    if _cash() >= amount:
        return True
    logger.warning("[MP-HEDGE] insufficient idle cash (Rs%.0f) for premium Rs%.0f — skipping hedge",
                   _cash(), amount)
    _alert("HEDGE SKIPPED — NO CASH",
           f"Needed Rs{amount:,.0f} premium but only Rs{_cash():,.0f} idle cash. Book is UNHEDGED "
           f"and still fully invested through a risk-off gate.", "high")
    return False


def hedge_open(reason="GATE_RISK_OFF"):
    """Buy the bi-weekly ATM put sized to hedge_ratio x equity."""
    from datetime import date as _d
    if not CFG["hedge_enabled"] or _hedge_get():
        return None
    eq = _equity_value()
    if eq <= 0:
        return None
    try:
        c = _hedge_pick()
    except Exception as e:
        logger.error(f"[MP-HEDGE] contract pick failed: {e}"); _alert("HEDGE PICK FAILED", str(e), "high")
        return None
    if not c:
        return None
    qty = _hedge_target_qty(eq, c["spot"], c["lot"])
    if qty <= 0:
        logger.warning("[MP-HEDGE] equity too small for one lot — no hedge")
        return None
    try:
        ltp = _opt_ltp(c["tsym"])
    except Exception as e:
        logger.error(f"[MP-HEDGE] ltp failed: {e}"); return None
    cost = ltp * qty
    nav = eq + _cash() + _sweep_value()
    if cost > nav * CFG["hedge_max_premium_pct"]:
        logger.warning("[MP-HEDGE] premium %.0f > %.0f%% of NAV — skipping hedge",
                       cost, CFG["hedge_max_premium_pct"] * 100)
        _alert("HEDGE SKIPPED", f"Premium Rs{cost:,.0f} exceeds "
               f"{CFG['hedge_max_premium_pct']*100:.0f}% of NAV Rs{nav:,.0f}", "normal")
        return None
    d = _d.today().isoformat()
    if not _hedge_fund(cost, d):
        logger.warning("[MP-HEDGE] could not fund premium"); return None
    if _is_live():
        try:
            ltp, qty = _place_opt_market(c["tsym"], "BUY", qty)
            cost = ltp * qty
        except Exception as e:
            logger.error(f"[MP-HEDGE] live buy failed: {e}")
            _alert("HEDGE BUY FAILED", f"{c['tsym']} x{qty}: {e}", "high"); return None
    _set("cash", _cash() - cost)
    _hedge_save(dict(tsym=c["tsym"], token=c["token"], strike=c["strike"], expiry=c["expiry"],
                     lot=c["lot"], qty=qty, entry_price=ltp, entry_date=d, entry_spot=c["spot"],
                     cost=cost))
    logger.warning(f"[MP-HEDGE] BOUGHT {qty} x {c['tsym']} @ {ltp:.2f} (Rs{cost:,.0f}, "
                   f"{c['dte']}DTE, spot {c['spot']:.0f}) — {reason}")
    return dict(tsym=c["tsym"], qty=qty, price=ltp, cost=cost)


def hedge_close(reason="GATE_RISK_ON"):
    from datetime import date as _d
    h = _hedge_get()
    if not h:
        return None
    try:
        ltp = _opt_ltp(h["tsym"])
    except Exception as e:
        logger.error(f"[MP-HEDGE] ltp on close failed: {e}"); return None
    qty = h["qty"]
    if _is_live():
        try:
            ltp, qty = _place_opt_market(h["tsym"], "SELL", qty)
        except Exception as e:
            logger.error(f"[MP-HEDGE] live sell failed: {e}")
            _alert("HEDGE SELL FAILED", f"{h['tsym']} x{qty}: {e}", "high"); return None
    proceeds = ltp * qty
    d = _d.today().isoformat()
    _set("cash", _cash() + proceeds)
    c = _conn()
    c.execute("INSERT INTO mp_hedge_closed(tsym,strike,expiry,qty,entry_date,entry_price,exit_date,"
              "exit_price,cost,proceeds,pnl,reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
              (h["tsym"], h["strike"], h["expiry"], qty, h["entry_date"], h["entry_price"], d,
               ltp, h["cost"], proceeds, proceeds - h["cost"], reason))
    c.commit(); _hedge_clear()
    logger.warning(f"[MP-HEDGE] CLOSED {h['tsym']} @ {ltp:.2f} — P&L Rs{proceeds-h['cost']:,.0f} ({reason})")
    return dict(pnl=proceeds - h["cost"], reason=reason)


def hedge_maintain():
    """Daily: roll at expiry while still risk-off, and re-size if equity has drifted."""
    from datetime import date as _d
    h = _hedge_get()
    if not h:
        return
    today = _d.today()
    if CFG["hedge_gate_daily"]:
        try:
            close, _tv = _panel()
            risk_off = _gate_risk_off(close, close.index[-1])
        except Exception as _e:
            logger.error(f"[MP-HEDGE] daily gate check failed, falling back to weekly state: {_e}")
            risk_off = _get("gate", "ON") == "OFF"
    else:
        risk_off = _get("gate", "ON") == "OFF"
    # roll at (or just before) expiry
    if date.fromisoformat(h["expiry"]) <= today:
        hedge_close("EXPIRY_ROLL")
        if risk_off:
            hedge_open("EXPIRY_ROLL")
        return
    if not risk_off:
        hedge_close("GATE_RISK_ON"); return
    # re-size as stocks stop out (the notional must track CURRENT equity)
    eq = _equity_value()
    if eq <= 0:
        hedge_close("NO_EQUITY_LEFT"); return
    try:
        spot = _nifty_spot()
    except Exception:
        return
    tgt = _hedge_target_qty(eq, spot, h["lot"])
    if tgt <= 0:
        hedge_close("EQUITY_BELOW_ONE_LOT"); return
    if abs(h["qty"] - tgt) / max(1, tgt) > CFG["hedge_resize_drift"]:
        logger.warning(f"[MP-HEDGE] re-sizing {h['qty']} -> {tgt} (equity Rs{eq:,.0f})")
        hedge_close("RESIZE"); hedge_open("RESIZE")


# ───────────────────── book ops ─────────────────────
def _positions():
    return {r["symbol"]: dict(r) for r in _conn().execute("SELECT * FROM mp_positions")}


def _cash():
    return float(_get("cash", 0.0))


def _deployable_cash(nav=None):
    """Cash available for STOCK purchases — always holds back cash_reserve_pct of NAV so a hedge
    premium can be funded at any risk-off gate. Never returns a negative number."""
    reserve = (nav if nav is not None else (_equity_value() + _cash())) * CFG["cash_reserve_pct"]
    return max(0.0, _cash() - reserve)


def _record_fill(symbol, side, price, qty, reason):
    value = price * qty; cost = value * (CFG["cost_rt"] / 2)
    c = _conn()
    c.execute("INSERT INTO mp_fills(ts,symbol,side,price,qty,value,cost,reason) VALUES(?,?,?,?,?,?,?,?)",
              (datetime.now().isoformat(timespec="seconds"), symbol, side, price, qty, value, cost, reason))
    c.commit(); c.close()
    return cost


def _buy(symbol, price, rupees, d, reason):
    if _is_live():
        qty = int(rupees // price)                       # whole shares for CNC delivery
        if qty < 1:
            logger.warning(f"[MP-LIVE] skip BUY {symbol}: budget ₹{rupees:.0f} < 1 share @ {price:.1f}")
            return
        if not _market_open_now():
            logger.error(f"[MP-LIVE] REFUSE BUY {symbol}: market closed"); return
        if qty * price > CFG["live_max_order_value"]:
            logger.error(f"[MP-LIVE] REFUSE BUY {symbol}: value ₹{qty * price:.0f} > cap"); return
        try:
            fill, fq = _place_cnc_market(symbol, "BUY", qty)
        except Exception as e:
            logger.error(f"[MP-LIVE] BUY {symbol} FAILED — not recording: {e}"); return
        _slippage_check(symbol, "BUY", fill, price)
        price, qty = fill, fq
    else:
        qty = rupees / price                             # paper allows fractional shares
    cost = _record_fill(symbol, "BUY", price, qty, reason)
    c = _conn()
    row = c.execute("SELECT qty,entry_date,entry_price,invested,peak_price FROM mp_positions "
                    "WHERE symbol=?", (symbol,)).fetchone()
    if row and (row["qty"] or 0) > 0:
        # TOP-UP of a name already held. INSERT OR REPLACE alone would overwrite the row and discard
        # the existing quantity — _sell() sells the RECORDED qty, so the next Donchian stop would
        # sell only the top-up and orphan the rest at the broker with no stop on it (2026-08-14).
        prev_qty = float(row["qty"])
        prev_inv = float(row["invested"] or prev_qty * float(row["entry_price"] or price))
        n_qty = prev_qty + qty
        n_inv = prev_inv + qty * price
        n_entry = row["entry_date"]                       # keep the ORIGINAL date — STCG clock
        n_peak = max(float(row["peak_price"] or 0.0), price)
        n_price = n_inv / n_qty                           # weighted-average cost basis
    else:
        n_qty, n_inv, n_entry, n_peak, n_price = qty, qty * price, d, price, price
    c.execute("INSERT OR REPLACE INTO mp_positions(symbol,qty,entry_date,entry_price,invested,peak_price) "
              "VALUES(?,?,?,?,?,?)", (symbol, n_qty, n_entry, n_price, n_inv, n_peak))
    c.commit(); c.close()
    _set("cash", _cash() - qty * price - cost)


def _sell(symbol, price, d, reason, qty=None):
    pos = _positions().get(symbol)
    if not pos:
        return
    full = qty is None or qty >= pos["qty"]
    qty = pos["qty"] if full else qty
    if _is_live():
        sell_qty = int(round(qty))
        if sell_qty < 1:
            return
        if not _market_open_now():
            logger.error(f"[MP-LIVE] REFUSE SELL {symbol}: market closed — POSITION STILL HELD"); return
        try:
            fill, fq = _place_cnc_market(symbol, "SELL", sell_qty)
        except Exception as e:
            logger.error(f"[MP-LIVE] SELL {symbol} FAILED — POSITION STILL HELD: {e}"); return
        _slippage_check(symbol, "SELL", fill, price)
        price = fill; qty = fq; full = fq >= pos["qty"]
    cost = _record_fill(symbol, "SELL", price, qty, reason)
    gross = (price - pos["entry_price"]) * qty
    gpct = (price / pos["entry_price"] - 1) * 100
    hold = (datetime.fromisoformat(d).date() - datetime.fromisoformat(pos["entry_date"]).date()).days \
        if "T" not in d else (date.fromisoformat(d) - date.fromisoformat(pos["entry_date"])).days
    entry_half = pos["invested"] * (qty / pos["qty"]) * (CFG["cost_rt"] / 2)  # proportional on partials
    rt_cost = entry_half + cost                               # entry + exit halves
    stcg = gross * CFG["stcg_pct"] if (gross > 0 and hold < 365) else 0.0
    c = _conn()
    c.execute("INSERT INTO mp_closed(symbol,entry_date,entry_price,exit_date,exit_price,qty,"
              "gross_pnl,gross_pct,cost,net_pnl,reason,holding_days,stcg_tax) "
              "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
              (symbol, pos["entry_date"], pos["entry_price"], d, price, qty, gross, gpct,
               rt_cost, gross - rt_cost, reason, hold, stcg))
    if full:
        c.execute("DELETE FROM mp_positions WHERE symbol=?", (symbol,))
    else:                                                    # partial trim — keep the remainder
        rem = pos["qty"] - qty
        c.execute("UPDATE mp_positions SET qty=?, invested=? WHERE symbol=?",
                  (rem, pos["invested"] * (rem / pos["qty"]), symbol))
    c.commit(); c.close()
    _set("cash", _cash() + qty * price - cost)


def _mark_nav(close, asof_iso, live=None):
    pos = _positions()
    syms = list(pos)
    px = live or (_live_prices(syms) if syms else {})
    equity = 0.0; unreal = 0.0
    c = _conn()
    for s, p in pos.items():
        pr = px.get(s) or p["entry_price"]
        equity += p["qty"] * pr
        unreal += (pr - p["entry_price"]) * p["qty"]
        peak = max(p["peak_price"], pr)
        if peak > p["peak_price"]:
            c.execute("UPDATE mp_positions SET peak_price=? WHERE symbol=?", (peak, s))
    c.commit(); c.close()
    # Swept cash sits in CASHIETF units, not the cash ledger. It is still OUR money: if it is
    # left out of NAV the equity curve shows a phantom crash the day the sweep first runs.
    cash = _cash() + _sweep_value(); nav = equity + cash
    bench = None
    try:
        bench = float(close[BENCH].loc[:asof_iso].dropna().iloc[-1])
    except Exception:
        pass
    gate = _get("gate", "ON")
    c = _conn()
    c.execute("INSERT OR REPLACE INTO mp_nav(d,equity,cash,nav,invested_pct,gate,bench_close,unrealized,capital) "
              "VALUES(?,?,?,?,?,?,?,?,?)",
              (asof_iso[:10], equity, cash, nav, (equity / nav * 100) if nav else 0,
               gate, bench, unreal, float(_get("capital", CFG["capital"]))))
    c.commit(); c.close()
    return nav


# ───────────────────── jobs ─────────────────────
def seed(force=False):
    """One-time: deploy ₹20L into today's top-8 momentum picks at live prices."""
    init_db()
    if _get("seeded") and not force:
        return {"ok": False, "msg": "already seeded"}
    refresh_universe(full=True)
    close, tv = _panel()
    asof = close.index[-1]
    etf = _rs_basket(close, tv, asof)
    if not etf:
        return {"ok": False, "msg": "could not compute basket"}
    top8 = etf[:CFG["n_hold"]]
    risk_off = _gate_risk_off(close, asof)
    _set("capital", CFG["capital"]); _set("cash", CFG["capital"])
    _set("inception", date.today().isoformat()); _set("gate", "OFF" if risk_off else "ON")
    d = date.today().isoformat()
    if not risk_off:
        live = _live_prices(top8)
        per = (CFG["capital"] * (1 - CFG["cash_reserve_pct"])) / len(top8)
        for s in top8:
            p = live.get(s)
            if p:
                _buy(s, p, per, d, "SEED")
    _mark_nav(close, asof.isoformat())
    _set("seeded", True); _set("last_monthly", d)
    return {"ok": True, "held": list(_positions()), "gate": _get("gate"),
            "basket": etf, "risk_off": risk_off}


def daily_job(panel=None):
    """Accrue liquid-fund yield on cash + mark P&L + 15-day Donchian stops.
    `panel` = pre-loaded (close, tv) so the single EOD run refreshes only once."""
    if not _get("seeded"):
        return
    # one trading-day of liquid-fund yield on idle/risk-off cash (LIQUIDCASE @6.5% p.a.)
    # Modelled liquid-fund yield. In LIVE this is only real if the cash is actually swept into
    # LIQUIDCASE — until that is implemented, accrue nothing so live NAV stays honest.
    # PAPER only. In LIVE the idle cash is really in CASHIETF and its return arrives as the ETF
    # price rising inside _sweep_value() — accruing a modelled yield on top would invent rupees
    # that do not exist at the broker and drift NAV upward on fictional money.
    _yield = 0.0 if _is_live() else CFG["cash_yield"]
    if _cash() > 0 and _yield > 0:
        new_cash = _cash() * (1 + _yield) ** (1 / 252)
        _set("interest_earned", round(_get("interest_earned", 0.0) + (new_cash - _cash()), 2))
        _set("cash", new_cash)
    if panel is None:
        refresh_universe(full=False); close, tv = _panel()
    else:
        close, tv = panel
    asof = close.index[-1]; d = date.today().isoformat()
    live = _live_prices(list(_positions()))
    for s in list(_positions()):
        low = _donchian_low(close, s, asof)
        pr = live.get(s)
        if low is not None and pr is not None and pr < low:
            _sell(s, pr, d, "DONCHIAN")
            logger.info(f"[MP] Donchian exit {s} @ {pr:.1f} (<15d low {low:.1f})")
    if CFG["live_cash_sweep"]:
        try:
            sweep_idle_cash()
        except Exception as _e:
            logger.error(f"[MP-SWEEP] error: {_e}")
    if CFG["hedge_enabled"]:
        try:
            if CFG["hedge_gate_daily"] and not _hedge_get() and _positions():
                # gate may break mid-week — buy the put the same EOD rather than waiting for Friday
                try:
                    _c, _t = _panel()
                    if _gate_risk_off(_c, _c.index[-1]):
                        hedge_open("GATE_RISK_OFF_DAILY")
                except Exception as _ge:
                    logger.error(f"[MP-HEDGE] daily gate open check failed: {_ge}")
            hedge_maintain()
        except Exception as _e:
            logger.error(f"[MP-HEDGE] maintain error: {_e}")
    _mark_nav(close, asof.isoformat(), live=live)
    _set("last_daily", d)


def weekly_job(panel=None):
    """NIFTYBEES 100-DMA gate (last trading day of week)."""
    if not _get("seeded"):
        return
    if panel is None:
        refresh_universe(full=False); close, tv = _panel()
    else:
        close, tv = panel
    asof = close.index[-1]; d = date.today().isoformat()
    risk_off = _gate_risk_off(close, asof)
    if risk_off and _positions():
        if CFG["hedge_enabled"]:
            # HOLD the stocks and buy a bi-weekly 2x NIFTY put instead of liquidating (research/105)
            _set("gate", "OFF")
            if not _hedge_get():
                hedge_open("GATE_RISK_OFF")
        else:
            live = _live_prices(list(_positions()))
            for s in list(_positions()):
                _sell(s, live.get(s, close[s].loc[:asof].dropna().iloc[-1]), d, "GATE_RISK_OFF")
        _set("gate", "OFF")
        logger.info("[MP] GATE risk-off → liquidated to cash")
    elif not risk_off:
        _set("gate", "ON")
        if not _positions():
            # r/41 phase-27: when FULLY in cash and the gate flips back on,
            # re-enter at the weekly check instead of waiting for month-end
            # (allcash+weekly re-entry won, Cal 1.72; the false-dawn penalty
            # applies to the partially-held state, not the all-cash state).
            etf = _rs_basket(close, tv, asof)
            if etf:
                top8 = etf[:CFG["n_hold"]]
                live = _live_prices(top8)
                per = _deployable_cash() / len(top8)
                for s in top8:
                    px = live.get(s, close[s].loc[:asof].dropna().iloc[-1])
                    _buy(s, px, per, d, "GATE_REENTRY")
                logger.info(f"[MP] GATE weekly re-entry (all-cash): {top8}")
    _mark_nav(close, asof.isoformat())
    _set("last_weekly", d)


def monthly_job(panel=None):
    """Rebalance: top-8 with top-22 buffer (last trading day of month)."""
    if not _get("seeded"):
        return
    if panel is None:
        refresh_universe(full=True); close, tv = _panel()
    else:
        close, tv = panel
    asof = close.index[-1]; d = date.today().isoformat()
    if _gate_risk_off(close, asof):
        _set("gate", "OFF"); _mark_nav(close, asof.isoformat()); _set("last_monthly", d)
        logger.info("[MP] monthly: risk-off, staying in cash")
        return
    _set("gate", "ON")
    if CFG["live_cash_sweep"] and _sweep_units() > 0:
        unsweep()                      # release parked cash BEFORE buying stocks (same-day settled)
        # NOTE: this ordering is load-bearing — the `nav` computed below is equity + cash and
        # does NOT add _sweep_value(). It is only correct because sweep_units is 0 by now.
    etf = _rs_basket(close, tv, asof)
    if not etf:
        return
    top8 = etf[:CFG["n_hold"]]; buf = set(etf[:CFG["buffer"]])
    held = _positions()
    live = _live_prices(sorted(set(list(held) + top8)))
    # 1) sell holds that fell out of the top-22 buffer
    for s in list(held):
        if s not in buf:
            _sell(s, live.get(s, close[s].loc[:asof].dropna().iloc[-1]), d, "BUFFER_ROTATE")
    # 2) target = kept (still in buffer) + new top-8; equal-weight whole book
    kept = [s for s in _positions() if s in buf]
    target = (kept + [s for s in top8 if s not in kept])[:CFG["n_hold"]]
    nav = sum(_positions()[s]["qty"] * live.get(s, _positions()[s]["entry_price"])
              for s in _positions()) + _cash()
    per = (nav * (1 - CFG["cash_reserve_pct"])) / len(target)
    # Rotate-only in BOTH modes: sell names that fell out of the top-22 buffer, buy new names from
    # freed cash, and let winners RIDE (no equal-weight trim). Backtest-confirmed return-neutral vs
    # trimming (34.3% vs 34.6% CAGR), with ~13% less turnover cost and ~20% less realized STCG. The
    # old PAPER path liquidated + rebuilt the whole book every month, churning winners (e.g.
    # LAURUSLABS was sold + rebought at the same price, needlessly realizing STCG for nothing).
    _rebalance_live_delta(target, per, live, close, asof, d)
    _mark_nav(close, asof.isoformat(), live=live)
    _set("last_monthly", d)
    logger.info(f"[MP] monthly rebalance → {target}")


def _rebalance_live_delta(target, per, live, close, asof, d):
    """LIVE monthly rebalance — rotate-only, do NOT churn the whole book (that would pay
    needless brokerage + 20% STCG on winners every month). Policy v1:
      • exit any held name NOT in the new target (full sell)
      • buy brand-new target names, cash-aware & equal-weight (never overspend into negative cash)
      • kept names ride as-is (no top-up/trim → let winners run, zero avoidable tax)
    Weight top-up/trim of kept names is deferred (needs cost-basis averaging); flag it later
    via CFG['live_rebalance_trim'] if the equal-weight drift ever matters."""
    tset = set(target)
    for s in list(_positions()):
        if s not in tset:
            px = live.get(s)
            if px is None and close is not None:
                px = float(close[s].loc[:asof].dropna().iloc[-1])
            if not px:
                logger.warning(f"[MP-LIVE] rebalance: no price to exit {s} — holding, will retry next cycle")
                continue
            _sell(s, px, d, "REBALANCE")
    new_names = [s for s in target if s not in _positions() and live.get(s)]
    if not new_names:
        return
    budget_each = _deployable_cash() / len(new_names)   # keeps the hedge reserve intact
    for s in new_names:
        _buy(s, live[s], min(per, budget_each), d, "REBALANCE")


# ───────────────────── API getters ─────────────────────
def _hedge_state():
    """Current hedge marked to market (None when unhedged)."""
    h = _hedge_get()
    if not h:
        return None
    ltp = None
    try:
        ltp = _opt_ltp(h["tsym"])
    except Exception:
        pass
    val = (ltp or h["entry_price"]) * h["qty"]
    return dict(tsym=h["tsym"], strike=h["strike"], expiry=h["expiry"], qty=h["qty"],
                lots=int(h["qty"] / h["lot"]) if h["lot"] else None,
                entry_date=h["entry_date"], entry_price=round(h["entry_price"], 2),
                price=round(ltp, 2) if ltp else None, cost=round(h["cost"]),
                value=round(val), pnl=round(val - h["cost"]),
                pnl_pct=round((val / h["cost"] - 1) * 100, 1) if h["cost"] else 0,
                dte=(date.fromisoformat(h["expiry"]) - date.today()).days)


def get_state():
    init_db()
    pos = _positions()
    syms = list(pos)
    live = _live_prices(syms) if syms else {}
    cap = _get("capital", CFG["capital"]); cash = _cash()
    close = tvp = None
    try:
        close, tvp = _panel()
    except Exception:
        pass
    holdings = []
    equity = 0.0
    for s, p in pos.items():
        pr = live.get(s) or p["entry_price"]
        val = p["qty"] * pr; equity += val
        low = _donchian_low(close, s, close.index[-1]) if close is not None else None
        hold = (date.today() - date.fromisoformat(p["entry_date"][:10])).days
        holdings.append(dict(
            symbol=s, qty=round(p["qty"], 1), entry_date=p["entry_date"][:10],
            entry_price=round(p["entry_price"], 1), price=round(pr, 1),
            value=round(val), weight=0, pnl=round(val - p["invested"]),
            pnl_pct=round((pr / p["entry_price"] - 1) * 100, 1), days=hold,
            stop=round(low, 1) if low else None,
            stop_dist_pct=round((pr / low - 1) * 100, 1) if low else None))
    # fold swept ETF value back into cash — it is cash-equivalent, redeemable same-day
    cash = cash + _sweep_value()
    nav = equity + cash
    n_stocks = len(holdings)
    for h in holdings:
        h["weight"] = round(h["value"] / nav * 100, 1) if nav else 0
    holdings.sort(key=lambda x: -x["value"])
    # idle/risk-off cash shown AS A HOLDING — parked in LIQUIDCASE (liquid fund @6.5%)
    if cash > 1000:
        # Date the ETF was actually BOUGHT — not the book's inception. Showing inception made a
        # position opened today read as 7 days old. When nothing is swept this is plain cash, which
        # has no entry date or holding period at all.
        _u = _sweep_units()
        _since = (_get("sweep_since") or "") if _u else ""
        _gain = (_sweep_value() - float(_get("sweep_cost", 0.0) or 0.0)) if _u else 0.0
        holdings.insert(0, dict(
            symbol=CFG["sweep_symbol"], qty=(round(_u, 2) if _u else None),
            entry_date=(_since[:10] if _since else None), entry_price=None,
            price=None, value=round(cash), weight=round(cash / nav * 100, 1) if nav else 0,
            pnl=round(_gain if _is_live() else _get("interest_earned", 0.0)),
            pnl_pct=round((CFG["sweep_yield_actual"] if _is_live() else CFG["cash_yield"]) * 100, 1),
            is_cash=True,
            days=((date.today() - date.fromisoformat(_since[:10])).days if _since else 0),
            stop=None, stop_dist_pct=None))
    navcurve = [dict(d=r["d"], nav=round(r["nav"]), bench=r["bench_close"], gate=r["gate"])
                for r in _conn().execute("SELECT * FROM mp_nav ORDER BY d")]
    closed = [dict(r) for r in _conn().execute(
        "SELECT * FROM mp_closed ORDER BY exit_date DESC, id DESC LIMIT 200")]
    stcg_open = sum(max(0.0, (live.get(s, p["entry_price"]) - p["entry_price"]) * p["qty"]) * CFG["stcg_pct"]
                    for s, p in pos.items())
    realized = sum(r["net_pnl"] for r in closed)
    stcg_booked = sum(r["stcg_tax"] for r in closed)
    incep = _get("inception")
    # target basket (what the book holds, or WOULD hold at next risk-on) + gate detail
    target, gate_last, gate_sma, gate_gap = [], None, None, None
    if close is not None:
        try:
            asof = close.index[-1]
            etf = _rs_basket(close, tvp, asof)
            if etf:
                target = [{"symbol": s, "rank": i + 1, "tier": _mcap_tier(s)}
                          for i, s in enumerate(etf[:CFG["n_hold"]])]
            b = close[BENCH].dropna()
            gate_last = round(float(b.iloc[-1]), 2)
            gate_sma = round(float(b.tail(CFG["gate_sma"]).mean()), 2)
            gate_gap = round((gate_last / gate_sma - 1) * 100, 2)
        except Exception:
            pass
    return dict(
        seeded=bool(_get("seeded")), gate=_get("gate", "ON"), inception=incep,
        mode=("LIVE" if _is_live() else "PAPER"), live_mode=_is_live(),
        # the second key of the two-key safety — the page must be able to show whether the
        # book is actually armed to place orders, not just that it is in live MODE
        live_armed=str(_get("live_armed", "0")).lower() in ("1", "true", "on", "yes"),
        target_basket=target, gate_last=gate_last, gate_sma=gate_sma, gate_gap_pct=gate_gap,
        capital=cap, nav=round(nav), cash=round(cash), equity=round(equity),
        invested_pct=round(equity / nav * 100, 1) if nav else 0,
        total_return_pct=round((nav / cap - 1) * 100, 2) if cap else 0,
        unrealized=round(equity - sum(p["invested"] for p in pos.values())),
        realized_net=round(realized), n_holdings=n_stocks,
        # in LIVE the "interest" IS the ETF's gain; the modelled accrual is paper-only and always 0
        interest_earned=(round(_sweep_value() - float(_get("sweep_cost", 0.0) or 0.0))
                         if (_is_live() and _sweep_units()) else round(_get("interest_earned", 0.0))),
        cash_yield_pct=(CFG["sweep_yield_actual"] * 100 if _is_live() else CFG["cash_yield"] * 100),
        stcg_unbooked=round(stcg_open), stcg_booked=round(stcg_booked),
        last_daily=_get("last_daily"), last_weekly=_get("last_weekly"),
        last_monthly=_get("last_monthly"),
        data_asof=(close.index[-1].date().isoformat() if close is not None else None),
        idle_cash=round(cash), idle_pct=round(cash / nav * 100, 1) if nav else 0,
        days_to_rebalance=_days_to_rebalance(),
        # ledger cash vs money parked in the ETF — the KPI strip showed one blended "CASH"
        # figure, so it looked like 26% was sitting idle when almost all of it was in CASHIETF
        hedge_viability=_hedge_viability(equity, nav),
        ledger_cash=round(_cash()), swept_value=round(_sweep_value()),
        sweep_gain=round(_sweep_value() - float(_get("sweep_cost", 0.0) or 0.0)) if _sweep_units() else 0,
        sweep=dict(enabled=CFG["live_cash_sweep"], symbol=CFG["sweep_symbol"],
                   units=round(_sweep_units(), 2), value=round(_sweep_value())),
        cash_reserve=round(nav * CFG["cash_reserve_pct"]),
        hedge=_hedge_state(), hedge_closed=[dict(r) for r in _conn().execute(
            "SELECT * FROM mp_hedge_closed ORDER BY id DESC LIMIT 50")],
        holdings=holdings, navcurve=navcurve, closed=closed, rules=RULES)


# ───────────────────── register ─────────────────────
def _held_weak_first():
    """Held symbols ordered WEAKEST momentum first, with their blended relative-strength score."""
    close, tvp = _panel()
    asof = close.index[-1]
    h = close.loc[:asof].ffill()
    score = {}
    for L, wt in ((126, 0.5), (252, 0.5)):
        p0 = h.iloc[-L - 1]; p1 = h.iloc[-1]; nf = p1[BENCH] / p0[BENCH]; r = (p1 / p0) / nf
        for s, v in r.items():
            if pd.notna(v):
                score[s] = score.get(s, 0) + wt * v
    return sorted(list(_positions()), key=lambda s: score.get(s, -9)), score


def cash_deposit(amount, mode="immediate", dry_run=True):
    # default = immediate even top-up of names already HELD (research/112: beat parking on
    # 12/12 deposit calendars). Falls back to park by itself when the gate is risk-OFF.
    """Add funds. mode='park' (default) -> liquid, deploys at the next rebalance. mode='immediate' ->
    equal-rupee top-up across current holdings IF the gate is risk-on (else parks). Returns the plan;
    executes only when dry_run is False. Works in paper and (when live) via real orders."""
    from datetime import date as _date
    try:
        amount = float(amount)
    except Exception:
        return {"ok": False, "error": "invalid amount"}
    if amount <= 0:
        return {"ok": False, "error": "amount must be > 0"}
    d = _date.today().isoformat()
    pos = _positions()
    close, tvp = _panel()
    gate_on = not _gate_risk_off(close, close.index[-1])
    plan = []; live = {}
    use_immediate = (mode == "immediate" and pos and gate_on)
    if use_immediate:
        live = _live_prices(list(pos)); per = amount / len(pos)
        for s in pos:
            pr = live.get(s) or pos[s]["entry_price"]
            qty = int(per / pr) if pr else 0
            if qty > 0:
                plan.append({"symbol": s, "action": "BUY", "qty": qty, "value": round(qty * pr)})
        note = f"Immediate equal-rupee top-up across {len(plan)} holdings (gate risk-ON)."
    else:
        mode = "park"
        note = ("Park in liquid (6.5%); deploys into the top-8 at the next month-end rebalance."
                if (gate_on or not pos) else "Gate is risk-OFF — parked in liquid until it turns risk-on.")
    res = {"ok": True, "action": "deposit", "mode": mode, "amount": round(amount),
           "plan": plan, "note": note, "dry_run": dry_run}
    if not dry_run:
        # Credit the deposit FIRST, then let _buy() deduct as it spends. The old code let _buy()
        # deduct AND then subtracted `spent` again, so cash came out at old + amount - 2*spent
        # (the 2026-08-14 Rs1L top-up left cash Rs78,113 short).
        _set("cash", _cash() + amount)
        if use_immediate:
            for p in plan:
                _buy(p["symbol"], live.get(p["symbol"]) or pos[p["symbol"]]["entry_price"], p["value"], d, "DEPOSIT_TOPUP")
        _set("capital", float(_get("capital", CFG["capital"])) + amount)
        logger.warning(f"[MP] DEPOSIT Rs{amount:,.0f} ({mode})")
        # Park the leftover in the liquid ETF right away rather than letting it idle until the next
        # 15:05 job — the whole amount in `park` mode, or the whole-share rounding remainder in
        # `immediate` mode. sweep_idle_cash() already respects the cash reserve and the min size,
        # and no-ops when the sweep is off or the market is shut.
        if CFG["live_cash_sweep"]:
            try:
                sw = sweep_idle_cash()
                if sw:
                    res["swept"] = sw
                    logger.warning(f"[MP] DEPOSIT → swept Rs{sw['value']:,.0f} into "
                                   f"{CFG['sweep_symbol']}")
            except Exception as _e:
                logger.error(f"[MP] deposit sweep failed (cash stays idle, next daily job "
                             f"will retry): {_e}")
        res["executed"] = True
    return res


def cash_withdraw(amount, dry_run=True):
    """Withdraw funds: use idle cash first, then SELL from the WEAKEST momentum rank upward
    (keeps winners; tax-efficient). Fully liquidates the weakest name, then the next. Returns the
    plan; executes only when dry_run is False."""
    from datetime import date as _date
    try:
        amount = float(amount)
    except Exception:
        return {"ok": False, "error": "invalid amount"}
    if amount <= 0:
        return {"ok": False, "error": "amount must be > 0"}
    d = _date.today().isoformat()
    # Release parked ETF cash FIRST. _cash() is ledger cash only, so with the sweep on it reads far
    # lower than the money actually available — without this a withdrawal would start SELLING STOCK
    # while idle cash sat in CASHIETF.
    if not dry_run and CFG["live_cash_sweep"] and _sweep_units() > 0 and amount > _cash():
        try:
            unsweep(amount - _cash())
        except Exception as _e:
            logger.error(f"[MP] withdraw unsweep failed — may sell stock unnecessarily: {_e}")
    pos = _positions(); cash = _cash()
    plan = []; raised = 0.0
    from_cash = min(cash, amount)
    if from_cash > 0:
        plan.append({"source": "idle cash", "value": round(from_cash)}); raised += from_cash
    need = amount - raised
    weak, score = _held_weak_first()
    live = _live_prices(weak) if weak else {}
    for s in weak:
        if need <= 1:
            break
        pr = live.get(s) or pos[s]["entry_price"]
        held_val = pos[s]["qty"] * pr
        qty = int(min(held_val, need) / pr) if pr else 0
        if qty <= 0:
            continue
        if held_val - qty * pr < pr:                     # selling nearly all -> take the whole lot
            qty = int(round(pos[s]["qty"]))
        val = qty * pr
        plan.append({"source": s, "action": "SELL", "qty": qty, "value": round(val),
                     "score": round(score.get(s, 0), 2)})
        raised += val; need -= val
    res = {"ok": True, "action": "withdraw", "amount": round(amount), "raised": round(raised),
           "shortfall": round(max(0.0, amount - raised)), "plan": plan, "dry_run": dry_run}
    if not dry_run:
        for p in plan:
            if p.get("action") == "SELL":
                _sell(p["source"], live.get(p["source"]) or pos[p["source"]]["entry_price"], d, "WITHDRAWAL")
        _set("cash", max(0.0, _cash() - amount))
        _set("capital", max(0.0, float(_get("capital", CFG["capital"])) - amount))
        logger.warning(f"[MP] WITHDRAW Rs{amount:,.0f}")
        res["executed"] = True
    return res


def register(app, scheduler):
    from flask import jsonify, request
    init_db()
    app.add_url_rule("/api/momentum-paper/state", "mp_state", lambda: jsonify(get_state()))
    app.add_url_rule("/api/momentum-paper/seed", "mp_seed",
                     lambda: jsonify(seed(bool((request.get_json(silent=True) or {}).get("force")))),
                     methods=["POST"])
    app.add_url_rule("/api/momentum-paper/run-daily", "mp_run_daily",
                     lambda: (daily_job() or jsonify({"ok": True})), methods=["POST"])
    app.add_url_rule("/api/momentum-paper/run-rebalance", "mp_run_rebal",
                     lambda: (monthly_job() or jsonify({"ok": True})), methods=["POST"])
    # ── LIVE controls ──
    app.add_url_rule("/api/momentum-paper/toggle-mode", "mp_toggle",
                     lambda: jsonify(_toggle_mode(request.get_json(silent=True) or {})), methods=["POST"])
    app.add_url_rule("/api/momentum-paper/kill-switch", "mp_kill",
                     lambda: jsonify(_kill_switch()), methods=["POST"])
    app.add_url_rule("/api/momentum-paper/reconcile", "mp_reconcile",
                     lambda: jsonify(reconcile_holdings()))
    app.add_url_rule("/api/momentum-paper/hedge", "mp_hedge",
                     lambda: jsonify({"hedge": _hedge_state(), "enabled": CFG["hedge_enabled"],
                                      "ratio": CFG["hedge_ratio"], "dte_target": CFG["hedge_dte_target"]}))
    app.add_url_rule("/api/momentum-paper/hedge/close", "mp_hedge_close", methods=["POST"],
                     view_func=lambda: jsonify(hedge_close("MANUAL") or {"ok": False, "note": "no hedge open"}))
    app.add_url_rule("/api/momentum-paper/deposit", "mp_deposit", methods=["POST"],
                     view_func=lambda: jsonify(cash_deposit(
                         (request.get_json(silent=True) or {}).get("amount"),
                         (request.get_json(silent=True) or {}).get("mode", "immediate"),  # research/112 winner
                         bool((request.get_json(silent=True) or {}).get("dry_run", True)))))
    app.add_url_rule("/api/momentum-paper/benchmarks", "mp_benchmarks", methods=["GET"],
                     view_func=lambda: jsonify(dict(
                         inception=(_get("inception") or "")[:10],
                         book=book_curve(), series=benchmark_series())))
    app.add_url_rule("/api/momentum-paper/withdraw", "mp_withdraw", methods=["POST"],
                     view_func=lambda: jsonify(cash_withdraw(
                         (request.get_json(silent=True) or {}).get("amount"),
                         bool((request.get_json(silent=True) or {}).get("dry_run", True)))))
    # Monthly re-rank runs EARLY (~14:45) for runway; light Donchian+gate near close (~15:15).
    scheduler.add_job(rebalance_job, "cron", day_of_week="mon-fri", hour=14, minute=45,
                      id="mp_rebalance", replace_existing=True)
    # 15:05, NOT 15:15 — 15:15-15:20 is the NSE Closing Auction Session transition window where new
    # orders are rejected. A Donchian stop that fires then simply cannot fill (seen live 2026-08-10).
    scheduler.add_job(eod_job, "cron", day_of_week="mon-fri", hour=15, minute=5,
                      id="mp_eod", replace_existing=True,
                      misfire_grace_time=900, coalesce=True)

    def _mp_eod_catchup(source):
        """Run today's EOD if its scheduled firing was LOST.

        2026-08-19: the service was restarted 7 times during the session, twice within a minute of
        15:05 — the EOD firing fell into the gap and was gone. APScheduler's default memory job
        store recomputes next_run_time forward on startup, so a run missed across a process restart
        is never replayed and nothing complains. That day no Donchian stop was breached, so nothing
        was lost, but the book spent a full session with its only exit rule unapplied.

        Donchian stops must SELL, and _sell() refuses once the market is shut, so a catch-up is only
        useful before ~15:29. Past that the honest move is a loud alert, not a silent skip."""
        try:
            from services.trading_calendar import get_default_calendar
            from datetime import date as _d, datetime as _dtm
            today = _d.today()
            if not get_default_calendar().is_trading_day(today):
                return
            if not _get("seeded"):
                return
            if _get("last_daily") == today.isoformat():
                return                                    # already ran — nothing to do
            mins = _dtm.now().hour * 60 + _dtm.now().minute
            if mins < 15 * 60 + 5:
                return                                    # not due yet today
            if mins >= 15 * 60 + 29:
                if _get("eod_miss_alerted") != today.isoformat():
                    _set("eod_miss_alerted", today.isoformat())
                    logger.error(f"[MP] EOD NEVER RAN TODAY ({source}) — stops unapplied, market shut")
                    _alert("MOMENTUM EOD DID NOT RUN",
                           f"last_daily={_get('last_daily')} but today is {today}. The Donchian stop "
                           f"and macro gate were NOT applied and the market is now closed, so a stop "
                           f"cannot be filled today. Check positions against their 15-day lows before "
                           f"the next open.", "high")
                return
            logger.warning(f"[MP] EOD CATCH-UP ({source}) — last_daily={_get('last_daily')}, running now")
            eod_job()
            _alert("MOMENTUM EOD CATCH-UP",
                   f"The 15:05 EOD run was missed (likely a restart landed on it). It has been "
                   f"re-run at {_dtm.now().strftime('%H:%M')} and stops/gate are now applied.", "normal")
        except Exception as _e:
            logger.error(f"[MP] eod catch-up error: {_e}")

    # (a) on every boot — covers a restart that swallowed the 15:05 firing
    from datetime import datetime as _dtm0, timedelta as _td0
    scheduler.add_job(lambda: _mp_eod_catchup("startup"), "date",
                      run_date=_dtm0.now() + _td0(seconds=45),
                      id="mp_eod_catchup_boot", replace_existing=True)
    # (b) a standing backstop while the market is still open, for a miss with no restart at all
    scheduler.add_job(lambda: _mp_eod_catchup("15:25 backstop"), "cron", day_of_week="mon-fri",
                      hour=15, minute=25, id="mp_eod_backstop", replace_existing=True,
                      misfire_grace_time=600, coalesce=True)

    def _mp_eod_report_job():                              # EOD email report ~15:35 (after the EOD job)
        try:
            from services.trading_calendar import get_default_calendar
            from datetime import date as _d
            if not get_default_calendar().is_trading_day(_d.today()):
                return
            from services.momentum_eod_report import send_eod_report
            send_eod_report()
        except Exception as _e:
            logger.error(f"[MP] eod-report job error: {_e}")
    scheduler.add_job(_mp_eod_report_job, "cron", day_of_week="mon-fri", hour=15, minute=35,
                      id="mp_eod_report", replace_existing=True)

    def _mp_reconcile_job():                              # morning DB<->broker reconcile (live-only; alerts on mismatch)
        try:
            if _is_live():
                reconcile_holdings()
        except Exception as _e:
            logger.error(f"[MP] reconcile job error: {_e}")
    scheduler.add_job(_mp_reconcile_job, "cron", day_of_week="mon-fri", hour=9, minute=20,
                      id="mp_reconcile", replace_existing=True)

    def _mp_monthly_report_job():                         # month-end summary email (last trading day, ~15:40)
        try:
            if _is_last_trading_day():
                from services.momentum_eod_report import send_monthly_report
                send_monthly_report()
        except Exception as _e:
            logger.error(f"[MP] monthly-report job error: {_e}")
    scheduler.add_job(_mp_monthly_report_job, "cron", day_of_week="mon-fri", hour=15, minute=40,
                      id="mp_monthly_report", replace_existing=True)
    for jid in ("mp_daily", "mp_weekly", "mp_monthly"):     # drop legacy split jobs
        try:
            scheduler.remove_job(jid)
        except Exception:
            pass
    logger.info("[MP] registered /api/momentum-paper/* + rebalance 14:45 (month-end) + "
                "EOD 15:15 (Donchian daily · gate last-day-of-week) — MODE=%s"
                % ("LIVE" if _is_live() else "PAPER"))
    # Pre-warm the panel/state cache off the request path so the first page load after
    # a restart is instant instead of paying the ~15-35s cold pivot+RS build.
    def _prewarm():
        try:
            get_state()
            logger.info("[MP] state cache pre-warmed")
        except Exception as _e:
            logger.warning(f"[MP] pre-warm failed: {_e}")
    import threading
    threading.Thread(target=_prewarm, name="mp-prewarm", daemon=True).start()


def rebalance_job():
    """MONTHLY re-rank, run EARLY (~14:45 IST) so there's ~45 min of runway before the
    15:30 close to catch/fix any issue (it refreshes 200 names + re-ranks + rotates 8
    positions — the heaviest, most consequential step). Acts only on the month's last
    trading day; a 14:45 price is a fine EOD proxy for monthly momentum signals."""
    if not _get("seeded") or not _is_last_trading_day():
        return
    refresh_universe(full=True)
    monthly_job(_panel())


def eod_job():
    """Light pre-close EOD run (~15:15 IST): daily Donchian stop + weekly macro gate, at
    near-close prices (executable while open). The heavy monthly re-rank runs earlier
    (rebalance_job, ~14:45)."""
    if not _get("seeded"):
        return
    refresh_universe(full=False)
    panel = _panel()
    daily_job(panel)                                   # interest + mark + Donchian (every day)
    if _is_last_trading_day_of_week():
        weekly_job(panel)                              # macro gate


def _is_last_trading_day_of_week():
    today = date.today()
    nxt = today + timedelta(days=1)
    while nxt.weekday() >= 5:                           # skip Sat/Sun
        nxt += timedelta(days=1)
    return nxt.isocalendar()[1] != today.isocalendar()[1]


def _is_last_trading_day():
    today = date.today()
    nxt = today + timedelta(days=1)
    while nxt.weekday() >= 5:                      # skip weekend
        nxt += timedelta(days=1)
    return nxt.month != today.month
