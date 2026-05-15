"""NAS Pipeline Watchdog & Heartbeat
===================================

The MST incident (2026-05-07 to 05-15) proved the NasTicker tick->candle
pipeline can silently freeze for days with zero alerting. NAS shares that
same singleton. This watchdog is the mitigation:

  * Runs every few minutes during market hours (trading days only).
  * Reads the freshest 5-min candle the NiftyCandleAggregator has produced.
  * If, during market hours, no fresh candle has appeared within
    STALE_THRESHOLD_SEC, the pipeline is considered FROZEN -> email alert
    (de-duplicated so it doesn't spam).
  * Every check writes a heartbeat record. The EOD report reads the day's
    heartbeats to show a pipeline-health timeline + any gaps/delays.

Email goes via the same SMTP the premarket brief uses (ORB_DEFAULTS).
This is an explicitly sanctioned alert channel (execution-issue surfacing),
distinct from the strategy alerts that were globally disabled.
"""
from __future__ import annotations

import json
import logging
import smtplib
from datetime import datetime, timedelta, time as dtime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "backtest_data"
STATE_PATH = DATA_DIR / "nas_watchdog_state.json"

# A 5-min candle should appear at least every 5 min; allow 3 min slack for
# tick lulls / callback thread scheduling before declaring a freeze.
STALE_THRESHOLD_SEC = 8 * 60
# Don't re-send the freeze alert more than once per this interval.
ALERT_DEDUP_SEC = 20 * 60
# Market hours (IST). Start the check a bit after open so the first live
# 5-min candle (09:20 close) has had a chance to form.
MARKET_CHECK_START = dtime(9, 22)
MARKET_CHECK_END = dtime(15, 31)


def _now_ist() -> datetime:
    # Server runs in IST (Asia/Kolkata) per VPS config; datetime.now() is IST.
    return datetime.now()


def _is_trading_day(d) -> bool:
    try:
        from services.trading_calendar import get_default_calendar
        return get_default_calendar().is_trading_day(d)
    except Exception:
        # Fail safe: treat weekdays as trading days if calendar unavailable
        return d.weekday() < 5


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"date": None, "heartbeats": [], "last_alert_ts": None,
            "stale_episode_open": False}


def _save_state(st: dict) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(st, indent=2, default=str))
    except Exception as e:
        logger.error(f"[NAS-WD] state save failed: {e}")


def _latest_candle_dt() -> Optional[datetime]:
    """Most recent candle the aggregator has (completed or in-progress)."""
    try:
        from services.nas_ticker import get_nas_ticker
        ticker = get_nas_ticker()
        agg = getattr(ticker, "aggregator", None)
        if agg is None:
            return None
        latest = None
        # in-progress candle is the freshest signal of "ticks are flowing"
        cur = getattr(agg, "current_candle", None)
        if cur and cur.get("date"):
            latest = cur["date"]
        comp = list(getattr(agg, "completed_candles", []))
        if comp:
            c = comp[-1].get("date")
            if c and (latest is None or c > latest):
                latest = c
        if isinstance(latest, str):
            latest = datetime.fromisoformat(latest)
        return latest
    except Exception as e:
        logger.error(f"[NAS-WD] latest candle lookup failed: {e}")
        return None


def _ticker_connected() -> Optional[bool]:
    try:
        from services.nas_ticker import get_nas_ticker
        return bool(getattr(get_nas_ticker(), "is_connected", False))
    except Exception:
        return None


def _send_alert(subject: str, body_html: str) -> bool:
    try:
        from config import ORB_DEFAULTS as cfg
        sender = cfg.get("email_from", "")
        pw = cfg.get("email_app_password", "")
        rcpt = cfg.get("email_to", "")
        if not (sender and pw and rcpt):
            logger.warning("[NAS-WD] email config incomplete; alert not sent")
            return False
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"Quantifyd Watchdog <{sender}>"
        msg["To"] = rcpt
        msg.attach(MIMEText(body_html, "html"))
        with smtplib.SMTP(cfg.get("smtp_host", "smtp.gmail.com"),
                          cfg.get("smtp_port", 587), timeout=20) as s:
            s.starttls()
            s.login(sender, pw)
            s.send_message(msg)
        logger.warning(f"[NAS-WD] ALERT EMAIL SENT: {subject}")
        return True
    except Exception as e:
        logger.error(f"[NAS-WD] alert email failed: {e}")
        return False


def check_pipeline() -> dict:
    """Scheduler entry point. Returns the heartbeat record it wrote."""
    now = _now_ist()
    today_iso = now.date().isoformat()

    in_window = (MARKET_CHECK_START <= now.time() <= MARKET_CHECK_END)
    trading = _is_trading_day(now.date())
    market_active = in_window and trading

    latest = _latest_candle_dt()
    lag_sec = None
    if latest is not None:
        lag_sec = (now - latest).total_seconds()
    connected = _ticker_connected()

    if not market_active:
        status = "outside_market"
    elif latest is None:
        status = "no_candle_data"
    elif lag_sec is not None and lag_sec > STALE_THRESHOLD_SEC:
        status = "FROZEN"
    else:
        status = "ok"

    st = _load_state()
    if st.get("date") != today_iso:
        # New day → reset the heartbeat trail + episode flags
        st = {"date": today_iso, "heartbeats": [], "last_alert_ts": None,
              "stale_episode_open": False}

    hb = {
        "ts": now.isoformat(timespec="seconds"),
        "market_active": market_active,
        "latest_candle": latest.isoformat() if latest else None,
        "lag_sec": round(lag_sec) if lag_sec is not None else None,
        "connected": connected,
        "status": status,
    }
    st["heartbeats"].append(hb)
    # Keep heartbeat list bounded (a check every ~3 min => ~130/day)
    st["heartbeats"] = st["heartbeats"][-400:]

    # Alert logic: only during market hours, only on FROZEN, de-duplicated.
    if status == "FROZEN":
        last_alert = st.get("last_alert_ts")
        send = True
        if last_alert:
            try:
                if (now - datetime.fromisoformat(last_alert)).total_seconds() < ALERT_DEDUP_SEC:
                    send = False
            except Exception:
                pass
        if send:
            mins = int(lag_sec // 60) if lag_sec else "?"
            html = f"""
            <div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;
                        max-width:560px;border:1px solid #e0241b;border-radius:8px;
                        padding:18px;background:#fff5f5;color:#1a1a1a">
              <div style="font-size:18px;font-weight:700;color:#c0241b">
                🔴 NAS PIPELINE FROZEN</div>
              <p style="font-size:14px;line-height:1.6">
                The NasTicker candle pipeline has produced <b>no fresh 5-min
                candle for ~{mins} minutes</b> during market hours.
                NAS signal generation is currently <b>BLIND</b>. This is the
                same failure mode as the MST incident.
              </p>
              <table style="font-size:13px;border-collapse:collapse">
                <tr><td style="padding:3px 12px 3px 0;color:#666">Checked at</td>
                    <td><b>{now.strftime('%d-%b-%Y %H:%M:%S IST')}</b></td></tr>
                <tr><td style="padding:3px 12px 3px 0;color:#666">Latest candle</td>
                    <td><b>{hb['latest_candle'] or '—'}</b></td></tr>
                <tr><td style="padding:3px 12px 3px 0;color:#666">Lag</td>
                    <td><b>{hb['lag_sec']} sec</b></td></tr>
                <tr><td style="padding:3px 12px 3px 0;color:#666">Ticker connected</td>
                    <td><b>{connected}</b></td></tr>
              </table>
              <p style="font-size:13px;color:#666;margin-top:14px">
                Action: check Kite token / WebSocket on the VPS. While frozen,
                no NAS system can fire — paper or live.
              </p>
            </div>"""
            if _send_alert(
                f"🔴 NAS PIPELINE FROZEN — no candle {int(lag_sec//60) if lag_sec else '?'}min "
                f"({now.strftime('%d-%b %H:%M')})", html):
                st["last_alert_ts"] = now.isoformat()
                st["stale_episode_open"] = True
    elif status == "ok" and st.get("stale_episode_open"):
        # Recovered — note it (EOD report will show the gap)
        st["stale_episode_open"] = False
        logger.info("[NAS-WD] pipeline recovered (was frozen)")

    _save_state(st)
    logger.info(f"[NAS-WD] {status} lag={hb['lag_sec']} connected={connected} "
                f"latest={hb['latest_candle']}")
    return hb


def get_today_heartbeats() -> dict:
    """For the EOD report: today's heartbeat trail + summary."""
    st = _load_state()
    hbs = st.get("heartbeats", [])
    market_hbs = [h for h in hbs if h.get("market_active")]
    frozen = [h for h in market_hbs if h.get("status") == "FROZEN"]
    no_data = [h for h in market_hbs if h.get("status") == "no_candle_data"]
    lags = [h["lag_sec"] for h in market_hbs if h.get("lag_sec") is not None]
    return {
        "date": st.get("date"),
        "checks_total": len(hbs),
        "checks_market": len(market_hbs),
        "frozen_count": len(frozen),
        "no_data_count": len(no_data),
        "max_lag_sec": max(lags) if lags else None,
        "ever_frozen": bool(frozen or no_data),
        "first_frozen": frozen[0]["ts"] if frozen else None,
        "last_alert_ts": st.get("last_alert_ts"),
        "heartbeats": market_hbs,
    }
