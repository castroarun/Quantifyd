"""Dad's holdings digest — a lean, read-only version of the primary holdings digest.

Builds the same HoldingsDigest shape the Charts UI consumes, straight from Dad's
Kite holdings (no meta DB, no corporate-actions feed). Charts get their candles from
static/dad_holdings_ohlc.json; week52/ATH/sparkline are left null and the frontend
computes what it needs from the OHLC bars. movers_today is computed from day change.
"""
import logging

logger = logging.getLogger(__name__)


def _row(h: dict) -> dict:
    qty = h.get("quantity", 0) or 0
    avg = h.get("average_price", 0) or 0
    ltp = h.get("last_price", 0) or 0
    prev = h.get("close_price", 0) or 0    # Kite field is close_price, not close
    invested = qty * avg
    current = qty * ltp
    day_move = h.get("day_change")          # Kite gives per-unit day change directly
    if day_move is None:
        day_move = (ltp - prev)
    day_pct = h.get("day_change_percentage")
    if day_pct is None:
        day_pct = ((ltp / prev - 1) * 100) if prev else 0
    return {
        "tradingsymbol": h.get("tradingsymbol", ""),
        "qty": qty,
        "avg_price": round(avg, 2),
        "ltp": round(ltp, 2),
        "prev_close": round(prev, 2),
        "day_pct": round(day_pct, 2),
        "day_pnl_inr": round(qty * day_move),
        "invested": round(invested),
        "current": round(current),
        "total_pnl_inr": round(current - invested),
        "total_pnl_pct": round((current - invested) / invested * 100, 2) if invested else 0,
        # meta the lean digest doesn't compute — frontend derives from the OHLC file:
        "week52_high": None, "week52_low": None, "all_time_high": None,
        "pct_from_ath": None, "change_5d_pct": None, "sparkline": None,
    }


def _empty(configured: bool, error: str = None) -> dict:
    return {
        "configured": configured, "error": error,
        "summary": {"count": 0, "invested": 0, "current": 0, "day_pnl": 0,
                    "day_pct": 0, "total_pnl": 0, "total_pct": 0},
        "holdings": [], "movers_today": {"gainers": [], "losers": []},
        "movers_weekly": {"gainers": [], "losers": []},
        "extremes": {"high": [], "low": []}, "events": [], "next_event": None,
    }


def get_dad_digest() -> dict:
    from services.dad_kite import get_dad_kite, is_configured
    if not is_configured():
        return _empty(False, "Dad's account is not configured (DAD_KITE_API_KEY missing).")
    try:
        kite = get_dad_kite()
        raw = kite.holdings() or []
    except Exception as e:  # noqa: BLE001
        logger.error(f"[DAD] holdings fetch failed: {e}")
        return _empty(True, f"Dad's session not ready (login pending?): {e}")
    rows = [_row(h) for h in raw if (h.get("quantity") or 0) > 0]
    if not rows:
        return _empty(True, "No holdings in Dad's account.")

    invested = sum(r["invested"] for r in rows)
    current = sum(r["current"] for r in rows)
    day_pnl = sum(r["day_pnl_inr"] for r in rows)
    total_pnl = current - invested
    prev_val = current - day_pnl

    gainers = sorted([r for r in rows if r["day_pct"] > 0], key=lambda r: r["day_pct"], reverse=True)[:5]
    losers = sorted([r for r in rows if r["day_pct"] < 0], key=lambda r: r["day_pct"])[:5]

    return {
        "configured": True,
        "summary": {
            "count": len(rows),
            "invested": invested, "current": current,
            "day_pnl": day_pnl, "day_pct": round(day_pnl / prev_val * 100, 2) if prev_val else 0,
            "total_pnl": total_pnl, "total_pct": round(total_pnl / invested * 100, 2) if invested else 0,
        },
        "holdings": rows,
        "movers_today": {"gainers": gainers, "losers": losers},
        "movers_weekly": {"gainers": [], "losers": []},
        "extremes": {"high": [], "low": []},
        "events": [], "next_event": None,
    }
