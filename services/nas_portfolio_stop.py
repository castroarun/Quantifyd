"""NAS daily PORTFOLIO stop — proportional, per-venue.

Research/90 (64-day, 1-min faithful replay) showed the combined 916 book's single best risk lever is
a WIDE daily stop (~-1,300 per lot) with NO take-profit / trail. This enforces it live: every 10s,
sum the combined LIVE (MIS) day-P&L across a venue's 3 systems and, if it falls to
    -STOP_PER_LOT x (total live lots deployed today),
flatten every open live leg in that venue for the rest of the day.

Proportional by construction: threshold scales with the lots actually deployed, so changing lots or
adding/dropping a system needs no re-tuning. Per-venue (NIFTY 916 on NFO, SENSEX on BFO) because the
two trade on different weekdays and carry different lot notionals.

Re-entry after a stop is impossible without any gating: flattening closes every active leg, so the
move-stop/roll paths (which act on active legs) find nothing, and fresh entry only happens once at
09:16. A date-stamped flag just prevents re-firing and records that it tripped.

SENSEX threshold is provisional — the -1,300/lot is NIFTY-calibrated (SENSEX has ~2 live days); a
SENSEX-chain calibration can refine it later.

FAIL-SAFE: if any live leg cannot be priced this cycle, the whole check aborts (never flatten on
incomplete data). Only runs 09:16-15:15 (before entry there is nothing; after 15:15 EOD handles it).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

STOP_PER_LOT = 1300.0
FLAG_DIR = Path(__file__).resolve().parents[1] / "backtest_data"

VENUES = {
    "nifty": {
        "exchange": "NFO",
        "tp_per_lot": None,          # research/90: a FIXED take-profit HURTS NIFTY (grinds up).
        "trail_arm_per_lot": 2000.0,      # research/90 trail sweep: instead of a fixed TP, arm a
        "trail_giveback_per_lot": 350.0,  # TRAILING profit-lock -- once the combined book is up
                                          # Rs2000/lot, exit if it retraces Rs350/lot from its peak.
                                          # Beats stop-only on total (+~8%), only engages on big days
                                          # so it never caps the grind days. Rolled out 2026-07-27.
        "systems": [
            ("NAS-916-ATM", "Nas916AtmExecutor", "NAS_916_ATM_DEFAULTS"),
            ("NAS-916-ATM2", "Nas916Atm2Executor", "NAS_916_ATM2_DEFAULTS"),
            ("NAS-916-ATM4", "Nas916Atm4Executor", "NAS_916_ATM4_DEFAULTS"),
        ],
    },
    "sensex": {
        "exchange": "BFO",
        "tp_per_lot": 1667.0,        # research/91: SENSEX fades its intraday gains into the close
                                     # (median EOD -396) -> a ~Rs1667/lot (Rs10k on 6 lots) daily
                                     # take-profit HELPS (best total, both halves positive). Venue-
                                     # specific: the opposite holds for NIFTY.
        "trail_arm_per_lot": None,   # SENSEX uses the fixed TP, not a trail (trail untested on SENSEX).
        "trail_giveback_per_lot": None,
        "systems": [
            ("SENSEX-ATM", "SensexAtmExecutor", "SENSEX_ATM_DEFAULTS"),
            ("SENSEX-ATM2", "SensexAtm2Executor", "SENSEX_ATM2_DEFAULTS"),
            ("SENSEX-ATM4", "SensexAtm4Executor", "SENSEX_ATM4_DEFAULTS"),
        ],
    },
}


def _flag_path(venue: str) -> Path:
    return FLAG_DIR / f"portfolio_stop_{venue}_{date.today().isoformat()}.flag"


def already_fired(venue: str) -> bool:
    return _flag_path(venue).exists()


def _write_flag(venue: str, combined: float, threshold: float, lots: int) -> None:
    try:
        _flag_path(venue).write_text(json.dumps({
            "fired_at": datetime.now().isoformat(timespec="seconds"),
            "venue": venue, "combined_pnl": round(combined),
            "threshold": round(threshold), "lots": lots,
        }), encoding="utf-8")
    except Exception as e:
        logger.error("[PORT-STOP %s] flag write failed: %s", venue, e)


def _peak_path(venue: str) -> Path:
    return FLAG_DIR / f"trail_peak_{venue}_{date.today().isoformat()}.json"


def _read_peak(venue: str) -> float:
    try:
        return float(json.loads(_peak_path(venue).read_text())["peak"])
    except Exception:
        return -1e18


def _update_peak(venue: str, combined: float) -> float:
    """Track the day's high-water mark of the combined book P&L (date-stamped, resets daily). The 10s
    monitor calls this every cycle, so the trailing lock always sees the running peak."""
    p = max(_read_peak(venue), combined)
    try:
        _peak_path(venue).write_text(json.dumps({"peak": p}), encoding="utf-8")
    except Exception as e:
        logger.error("[PORT-STOP %s] peak write failed: %s", venue, e)
    return p


def _instantiate(venue: str, cls_name: str, cfg: dict):
    if venue == "nifty":
        import services.nas_916_executors as m
        return getattr(m, cls_name)(config=cfg)
    import services.sensex_executors as m
    return getattr(m, cls_name)()


def _cfg(cfg_name: str) -> dict:
    import config
    return getattr(config, cfg_name)


def _today_live_leg_count(db_path: str) -> int:
    """Legs that actually FILLED today in live mode (status ACTIVE or CLOSED) — proves the system
    deployed a real live position today. Excludes order-rejected / failed / pending: a rejected
    order never held a position, so it must not inflate the deployed-lots the stop scales with."""
    try:
        con = sqlite3.connect(db_path)
        n = con.execute(
            "SELECT COUNT(*) FROM nas_atm_positions "
            "WHERE substr(entry_time,1,10)=? AND mode='live' AND status IN ('ACTIVE','CLOSED')",
            (date.today().isoformat(),)
        ).fetchone()[0]
        con.close()
        return int(n or 0)
    except Exception as e:
        logger.error("[PORT-STOP] today-live query failed (%s): %s", db_path, e)
        return 0


def compute_venue(venue: str, kite):
    """Return (combined_pnl, deployed_lots, to_flatten[(label,exec)], details, incomplete).

    combined_pnl = sum over the venue's LIVE-deployed systems of realized (state.daily_pnl) +
    unrealized (open live legs marked at live LTP). incomplete=True if any open live leg is
    unpriced (caller must NOT act)."""
    v = VENUES[venue]
    exch = v["exchange"]
    deployed_lots = 0
    sys_objs = []          # (label, executor, active_live_legs, realized)
    all_syms = set()

    for label, cls_name, cfg_name in v["systems"]:
        cfg = _cfg(cfg_name)
        try:
            ex = _instantiate(venue, cls_name, cfg)
        except Exception as e:
            logger.error("[PORT-STOP %s] cannot instantiate %s: %s", venue, cls_name, e)
            continue
        if _today_live_leg_count(ex.db.db_path) == 0:
            continue                      # system not live-deployed today
        deployed_lots += int(cfg.get("lots_per_leg") or 0)
        active_live = [p for p in (ex.db.get_active_positions() or [])
                       if (p.get("mode") or "paper") == "live"]
        realized = float((ex.db.get_state() or {}).get("daily_pnl") or 0)
        sys_objs.append((label, ex, active_live, realized))
        for p in active_live:
            if p.get("tradingsymbol"):
                all_syms.add(p["tradingsymbol"])

    ltp_map = {}
    if all_syms:
        try:
            q = kite.ltp([f"{exch}:{s}" for s in all_syms]) or {}
            for s in all_syms:
                val = q.get(f"{exch}:{s}")
                if val and val.get("last_price") is not None:
                    ltp_map[s] = val["last_price"]
        except Exception as e:
            logger.warning("[PORT-STOP %s] ltp fetch failed: %s", venue, e)
            return 0.0, deployed_lots, [], [], True

    combined = 0.0
    to_flatten = []
    details = []
    incomplete = False
    for label, ex, active_live, realized in sys_objs:
        unreal = 0.0
        for p in active_live:
            ltp = ltp_map.get(p.get("tradingsymbol"))
            if ltp is None:
                incomplete = True
                break
            unreal += (p["entry_price"] - ltp) * p["qty"]
        if incomplete:
            break
        combined += realized + unreal
        if active_live:
            to_flatten.append((label, ex))
        details.append((label, round(realized), round(unreal), len(active_live)))
    return combined, deployed_lots, to_flatten, details, incomplete


def check_and_apply(venue: str, dry_run: bool = False):
    """The 10s tick. Returns a dict when the stop fires (or would, if dry_run), else None."""
    if already_fired(venue) and not dry_run:
        return None
    # 2026-08-06: a manual freeze halts the venue portfolio stop too (Arun is managing by hand).
    from services.nas_kill_switch import is_frozen as _pf_frozen
    if _pf_frozen() and not dry_run:
        return None
    now = datetime.now()
    if not (now.weekday() < 5 and (9, 16) <= (now.hour, now.minute) <= (15, 15)) and not dry_run:
        return None
    try:
        from services.kite_service import get_kite
        kite = get_kite()
    except Exception as e:
        logger.warning("[PORT-STOP %s] kite unavailable: %s", venue, e)
        return None

    combined, lots, to_flatten, details, incomplete = compute_venue(venue, kite)
    if incomplete or lots <= 0:
        return None

    stop_threshold = -STOP_PER_LOT * lots
    tp_per_lot = VENUES[venue].get("tp_per_lot")
    tp_threshold = (tp_per_lot * lots) if tp_per_lot else None
    arm_pl = VENUES[venue].get("trail_arm_per_lot")
    give_pl = VENUES[venue].get("trail_giveback_per_lot")

    # trailing profit-lock: track the day's peak combined P&L; once it has reached the arm
    # (arm_per_lot x lots), exit if it retraces the give-back (give_per_lot x lots) from that peak.
    peak = None
    trail_floor = None
    if arm_pl:
        peak = max(_read_peak(venue), combined) if dry_run else _update_peak(venue, combined)
        if peak >= arm_pl * lots:
            trail_floor = peak - give_pl * lots

    kind, thr = None, None
    if combined <= stop_threshold:
        kind, thr = "STOP", stop_threshold
    elif tp_threshold is not None and combined >= tp_threshold:
        kind, thr = "TP", tp_threshold
    elif trail_floor is not None and combined <= trail_floor:
        kind, thr = "TRAIL", trail_floor

    info = dict(venue=venue, combined=round(combined), stop_threshold=round(stop_threshold),
                tp_threshold=(round(tp_threshold) if tp_threshold else None),
                trail_arm=(round(arm_pl * lots) if arm_pl else None),
                trail_peak=(round(peak) if peak is not None else None),
                trail_floor=(round(trail_floor) if trail_floor is not None else None),
                lots=lots, details=details, hit=kind)
    if kind is None:
        return info if dry_run else None

    reason = {"STOP": "PORTFOLIO_STOP", "TP": "PORTFOLIO_TP", "TRAIL": "PORTFOLIO_TRAIL"}[kind]
    logger.critical("[PORT-%s %s] %s combined=%.0f vs %.0f (lots=%d) details=%s",
                    kind, venue, reason, combined, thr, lots, details)
    if dry_run:
        info["would_flatten"] = [lbl for lbl, _ in to_flatten]
        return info

    flattened = []
    for label, ex in to_flatten:
        try:
            exits = ex.exit_all_positions(reason)
            flattened.append((label, len(exits or [])))
            logger.critical("[PORT-%s %s] %s flattened %d legs", kind, venue, label, len(exits or []))
        except Exception as e:
            logger.error("[PORT-%s %s] %s flatten FAILED: %s", kind, venue, label, e, exc_info=True)
    _write_flag(venue, combined, thr, lots)
    info["flattened"] = flattened
    return info
