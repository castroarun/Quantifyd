"""NAS End-of-Day Trade Report (paper shadow)
=============================================

Daily EOD email for the Mon/Tue (and ongoing) paper shadow of the 8 NAS
systems. Designed to answer, at a glance:

  1. Did the pipeline stay alive all day? (watchdog heartbeat summary)
  2. Did each of the 8 systems fire? entries / exits / signals
  3. Today's paper P&L per system + combined
  4. Any execution issues — rejected orders, freeze episodes, lot-size
     divergence, missed-fire flags

Sent via the same SMTP the premarket brief uses (ORB_DEFAULTS). This is an
explicitly sanctioned channel; strategy alert emails remain globally off.

One-shot: build + send is idempotent. Safe to re-run.
"""
from __future__ import annotations

import logging
import smtplib
from datetime import datetime, date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

# (display name, group, dotted db getter) — group used for OTM vs ATM rollup
SYSTEMS = [
    ("NAS Squeeze OTM",  "OTM", "services.nas_db:get_nas_db"),
    ("NAS Squeeze ATM",  "ATM", "services.nas_atm_db:get_nas_atm_db"),
    ("NAS Squeeze ATM2", "ATM", "services.nas_atm2_db:get_nas_atm2_db"),
    ("NAS Squeeze ATM4", "ATM", "services.nas_atm4_db:get_nas_atm4_db"),
    ("NAS 9:16 OTM",     "OTM", "services.nas_916_db:get_nas_916_otm_db"),
    ("NAS 9:16 ATM",     "ATM", "services.nas_916_db:get_nas_916_atm_db"),
    ("NAS 9:16 ATM2",    "ATM", "services.nas_916_db:get_nas_916_atm2_db"),
    ("NAS 9:16 ATM4",    "ATM", "services.nas_916_db:get_nas_916_atm4_db"),
]


def _resolve(getter: str):
    mod, fn = getter.split(":")
    import importlib
    return getattr(importlib.import_module(mod), fn)()


def _system_snapshot(name: str, group: str, getter: str) -> dict:
    """Pull today's activity for one system. Never raises."""
    snap = {"name": name, "group": group, "ok": True, "error": None,
            "signals": 0, "closed": 0, "open": 0, "day_pnl": 0.0,
            "orders": 0, "win_rate": None, "fired": False, "lots": 0,
            "curve": []}
    try:
        db = _resolve(getter)
        today = date.today().isoformat()

        # closed today
        try:
            closed = db.get_today_closed_positions() or []
        except Exception:
            closed = []
        snap["closed"] = len(closed)

        # Day P&L. Authoritative source is the completed-trades table
        # (nas_trades / nas_atm_trades) where the strategy books realized
        # net_pnl per closed strangle — NOT the position legs, which carry
        # only entry_price/exit_price and no pnl column (the old code read
        # non-existent pnl/pnl_inr/realized_pnl keys → always ₹0).
        pnl = 0.0
        booked = False
        try:
            trades = db.get_recent_trades(limit=200) or []
            for t in trades:
                td = str(t.get("trade_date") or t.get("created_at") or "")[:10]
                if td == today:
                    v = t.get("net_pnl")
                    if v is None:
                        v = t.get("gross_pnl") or 0
                    try:
                        pnl += float(v)
                        booked = True
                    except Exception:
                        pass
        except Exception:
            pass
        # Fallback: no completed-trade row yet today (e.g. strangle still
        # mid-roll) — approximate realized P&L from closed legs. Short
        # option: (entry-exit)*qty; long: (exit-entry)*qty.
        if not booked:
            for c in closed:
                try:
                    ep = c.get("entry_price")
                    xp = c.get("exit_price")
                    q = c.get("qty") or 0
                    if ep is None or xp is None or not q:
                        continue
                    side = str(c.get("transaction_type") or "SELL").upper()
                    leg_pnl = (ep - xp) * q if side == "SELL" else (xp - ep) * q
                    pnl += float(leg_pnl)
                except Exception:
                    pass
        snap["day_pnl"] = round(pnl, 2)

        # lots traded today = max leg qty / NIFTY lot size (65)
        try:
            active = db.get_active_positions() or []
            qtys = [int(p.get("qty") or 0)
                    for p in (list(closed) + list(active)) if p.get("qty")]
            if qtys:
                snap["lots"] = round(max(qtys) / 65)
            snap["open"] = len(active)
        except Exception:
            pass

        # today order count
        try:
            snap["orders"] = int(db.get_today_order_count() or 0)
        except Exception:
            pass

        # today's signals (filter recent to today)
        try:
            sigs = db.get_recent_signals(limit=200) or []
            tcount = 0
            for s in sigs:
                ts = str(s.get("created_at") or s.get("signal_time") or "")
                if ts.startswith(today):
                    tcount += 1
            snap["signals"] = tcount
        except Exception:
            pass

        # all-time win rate for context
        try:
            st = db.get_stats() or {}
            snap["win_rate"] = st.get("win_rate")
        except Exception:
            pass

        # intraday MTM curve (realized+unrealized over the session)
        try:
            from services.nas_mtm import get_today_curve
            snap["curve"] = get_today_curve(db)
        except Exception:
            snap["curve"] = []

        snap["fired"] = (snap["signals"] > 0 or snap["closed"] > 0
                         or snap["open"] > 0 or snap["orders"] > 0)
    except Exception as e:
        snap["ok"] = False
        snap["error"] = str(e)
        logger.error(f"[NAS-EOD] snapshot failed for {name}: {e}")
    return snap


def _pipeline_section() -> dict:
    try:
        from services.nas_watchdog import get_today_heartbeats
        return get_today_heartbeats()
    except Exception as e:
        logger.error(f"[NAS-EOD] watchdog read failed: {e}")
        return {"ever_frozen": None, "checks_market": 0, "frozen_count": 0,
                "no_data_count": 0, "max_lag_sec": None, "heartbeats": []}


def _lot_size_check() -> dict:
    try:
        from services.nas_scanner import LOT_SIZE, kite_nifty_lot_size
        live = kite_nifty_lot_size(default=LOT_SIZE)
        return {"configured": LOT_SIZE, "exchange": live,
                "mismatch": (live != LOT_SIZE)}
    except Exception as e:
        return {"configured": None, "exchange": None, "mismatch": None,
                "error": str(e)}


def build_report() -> dict:
    snaps = [_system_snapshot(n, g, gt) for n, g, gt in SYSTEMS]
    pipe = _pipeline_section()
    lot = _lot_size_check()

    combined_pnl = round(sum(s["day_pnl"] for s in snaps), 2)
    fired = [s for s in snaps if s["fired"]]
    silent = [s for s in snaps if not s["fired"] and s["ok"]]
    errored = [s for s in snaps if not s["ok"]]

    issues = []
    if pipe.get("ever_frozen"):
        issues.append(
            f"Pipeline FROZE {pipe.get('frozen_count', 0)} check(s) "
            f"(max lag {pipe.get('max_lag_sec')}s) — signals were blind during that window")
    if pipe.get("no_data_count"):
        issues.append(f"{pipe['no_data_count']} check(s) had NO candle data during market hours")
    if pipe.get("checks_market", 0) == 0:
        issues.append("Watchdog ran 0 in-market checks today — was it a trading day / was the scheduler alive?")
    if lot.get("mismatch"):
        issues.append(
            f"LOT SIZE MISMATCH: code={lot['configured']} but Kite reports "
            f"{lot['exchange']} — order quantities are wrong, fix before live")
    for s in errored:
        issues.append(f"{s['name']}: data error — {s['error']}")
    if not fired and pipe.get("checks_market", 0) > 0 and not pipe.get("ever_frozen"):
        issues.append("NO system fired all day despite a live pipeline — verify entry conditions / signal logic")

    curve_png = _render_curve_png(snaps)
    return {
        "date": date.today().isoformat(),
        "generated_at": datetime.now().strftime("%d-%b-%Y %H:%M:%S IST"),
        "snaps": snaps, "pipeline": pipe, "lot": lot,
        "combined_pnl": combined_pnl,
        "n_fired": len(fired), "n_silent": len(silent), "n_error": len(errored),
        "issues": issues,
        "curve_png": curve_png, "has_curve": curve_png is not None,
    }


def _render_curve_png(snaps: list):
    """Intraday day-P&L curve per system → PNG bytes (or None)."""
    series = [(s["name"], s.get("curve") or []) for s in snaps
              if s.get("curve")]
    if not series:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from io import BytesIO

        fig, ax = plt.subplots(figsize=(9, 4.2), dpi=110)
        for name, curve in series:
            xs, ys = [], []
            for ts, v in curve:
                try:
                    xs.append(datetime.fromisoformat(ts)); ys.append(float(v))
                except Exception:
                    pass
            if xs:
                ax.plot(xs, ys, linewidth=1.4, label=name)
        ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
        ax.set_title("NAS intraday day-P&L (realized + open MTM)",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("₹"); ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.legend(fontsize=7, ncol=4, loc="upper left", framealpha=0.4)
        fig.autofmt_xdate(rotation=0)
        fig.tight_layout()
        buf = BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"[NAS-EOD] curve render failed: {e}")
        return None


def _pnl_color(v) -> str:
    try:
        v = float(v)
    except Exception:
        return "#666"
    return "#1f9d55" if v > 0 else ("#c0241b" if v < 0 else "#666")


def render_html(r: dict) -> str:
    rows = ""
    for s in r["snaps"]:
        fired_badge = (
            '<span style="color:#1f9d55;font-weight:700">●</span>'
            if s["fired"] else
            ('<span style="color:#c0241b;font-weight:700">○ silent</span>'
             if s["ok"] else '<span style="color:#c0241b">⚠ error</span>'))
        rows += f"""
        <tr style="border-bottom:1px solid #eee">
          <td style="padding:7px 8px;font-size:13px">{s['name']}
              <span style="color:#999;font-size:11px">[{s['group']}]</span></td>
          <td style="padding:7px 8px;text-align:center">{fired_badge}</td>
          <td style="padding:7px 8px;text-align:right;font-size:13px">{s['signals']}</td>
          <td style="padding:7px 8px;text-align:right;font-size:13px">{s['closed']}</td>
          <td style="padding:7px 8px;text-align:right;font-size:13px">{s['open']}</td>
          <td style="padding:7px 8px;text-align:right;font-size:13px">{s['orders']}</td>
          <td style="padding:7px 8px;text-align:right;font-size:13px">{s['lots']}</td>
          <td style="padding:7px 8px;text-align:right;font-size:13px;font-weight:700;
                     color:{_pnl_color(s['day_pnl'])}">₹{s['day_pnl']:,.0f}</td>
        </tr>"""

    pipe = r["pipeline"]
    pstatus_ok = (pipe.get("checks_market", 0) > 0 and not pipe.get("ever_frozen"))
    pcolor = "#1f9d55" if pstatus_ok else "#c0241b"
    pverdict = ("HEALTHY — pipeline stayed alive all session" if pstatus_ok
                else ("NEVER RAN / NO MARKET CHECKS" if pipe.get("checks_market", 0) == 0
                      else "DEGRADED — freeze/no-data episodes occurred"))

    issues_html = ""
    if r["issues"]:
        items = "".join(f'<li style="margin:4px 0">{i}</li>' for i in r["issues"])
        issues_html = f"""
        <div style="margin-top:18px;border:1px solid #e0241b;border-radius:8px;
                    background:#fff5f5;padding:14px 16px">
          <div style="font-weight:700;color:#c0241b;font-size:14px;margin-bottom:6px">
            ⚠ Execution issues &amp; delays ({len(r['issues'])})</div>
          <ul style="margin:6px 0 0 18px;padding:0;font-size:13px;color:#1a1a1a">{items}</ul>
        </div>"""
    else:
        issues_html = """
        <div style="margin-top:18px;border:1px solid #1f9d55;border-radius:8px;
                    background:#f3fbf6;padding:12px 16px;font-size:13px;color:#1f7a44">
          ✓ No execution issues detected — pipeline healthy, lot size correct,
          no order rejections.</div>"""

    combined_color = _pnl_color(r["combined_pnl"])
    lot = r["lot"]
    lot_line = (f"code {lot.get('configured')} = exchange {lot.get('exchange')} ✓"
                if lot.get("mismatch") is False
                else f"code {lot.get('configured')} vs exchange {lot.get('exchange')} ⚠")

    return f"""
    <div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:680px;
                margin:0 auto;color:#1a1a1a">
      <div style="border:1px solid #e3e3e3;border-radius:10px;overflow:hidden">
        <div style="background:#111;color:#fff;padding:16px 20px">
          <div style="font-size:18px;font-weight:700">NAS EOD Report
            <span style="font-size:12px;font-weight:500;color:#f5a623;
                         border:1px solid #f5a623;border-radius:4px;padding:1px 6px;
                         margin-left:8px">PAPER SHADOW</span></div>
          <div style="font-size:12px;color:#bbb;margin-top:3px">
            {r['date']} · generated {r['generated_at']}</div>
        </div>

        <div style="padding:18px 20px">
          <table style="width:100%;border-collapse:collapse;margin-bottom:6px">
            <tr>
              <td style="font-size:13px;color:#666">Combined paper P&amp;L</td>
              <td style="text-align:right;font-size:20px;font-weight:800;
                         color:{combined_color}">₹{r['combined_pnl']:,.0f}</td>
            </tr>
            <tr>
              <td style="font-size:13px;color:#666;padding-top:4px">Systems fired</td>
              <td style="text-align:right;font-size:13px;padding-top:4px">
                <b>{r['n_fired']}/8</b> fired · {r['n_silent']} silent
                {f"· {r['n_error']} error" if r['n_error'] else ""}</td>
            </tr>
          </table>

          <div style="margin:14px 0;padding:12px 14px;border-radius:8px;
                      background:{'#f3fbf6' if pstatus_ok else '#fff5f5'};
                      border:1px solid {pcolor}">
            <div style="font-weight:700;color:{pcolor};font-size:13px">
              Pipeline: {pverdict}</div>
            <div style="font-size:12px;color:#555;margin-top:4px">
              {pipe.get('checks_market', 0)} in-market checks ·
              {pipe.get('frozen_count', 0)} frozen ·
              {pipe.get('no_data_count', 0)} no-data ·
              max lag {pipe.get('max_lag_sec') if pipe.get('max_lag_sec') is not None else '—'}s ·
              lot size {lot_line}</div>
          </div>

          <table style="width:100%;border-collapse:collapse;margin-top:10px">
            <thead>
              <tr style="background:#f6f6f6;text-align:left">
                <th style="padding:8px;font-size:11px;color:#666;text-transform:uppercase">System</th>
                <th style="padding:8px;font-size:11px;color:#666;text-align:center">Fired</th>
                <th style="padding:8px;font-size:11px;color:#666;text-align:right">Sig</th>
                <th style="padding:8px;font-size:11px;color:#666;text-align:right">Closed</th>
                <th style="padding:8px;font-size:11px;color:#666;text-align:right">Open</th>
                <th style="padding:8px;font-size:11px;color:#666;text-align:right">Ord</th>
                <th style="padding:8px;font-size:11px;color:#666;text-align:right">Lots</th>
                <th style="padding:8px;font-size:11px;color:#666;text-align:right">Day P&amp;L</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>

          {('<div style="margin-top:16px"><div style="font-size:11px;'
            'color:#666;text-transform:uppercase;margin-bottom:4px">'
            'Intraday P&amp;L curve</div>'
            '<img src="cid:nascurve" alt="intraday P&L curve" '
            'style="width:100%;max-width:660px;border:1px solid #eee;'
            'border-radius:6px"/></div>') if r.get("has_curve") else ''}

          {issues_html}

          <div style="margin-top:16px;font-size:11px;color:#999;line-height:1.5">
            Paper-shadow mode — no real orders placed. Real-money go-live is
            gated on 2 consecutive clean sessions (pipeline healthy + all
            systems firing as expected). This report auto-sends each trading
            day at ~15:35 IST.
          </div>
        </div>
      </div>
    </div>"""


def send_eod_report() -> dict:
    """Scheduler entry point. Builds + emails the report. Never raises."""
    try:
        r = build_report()
        html = render_html(r)
        from config import ORB_DEFAULTS as cfg
        sender = cfg.get("email_from", "")
        pw = cfg.get("email_app_password", "")
        rcpt = cfg.get("email_to", "")
        if not (sender and pw and rcpt):
            logger.warning("[NAS-EOD] email config incomplete; not sent")
            return {"sent": False, "reason": "email config incomplete", "report": r}
        verdict = (f"+₹{r['combined_pnl']:,.0f}" if r["combined_pnl"] >= 0
                   else f"-₹{abs(r['combined_pnl']):,.0f}")
        flag = " ⚠" if r["issues"] else ""
        subject = (f"[NAS EOD · PAPER] {verdict} · {r['n_fired']}/8 fired · "
                   f"{r['date']}{flag}")
        root = MIMEMultipart("mixed")
        root["Subject"] = subject
        root["From"] = f"Quantifyd <{sender}>"
        root["To"] = rcpt
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(html, "html"))
        root.attach(alt)
        png = r.get("curve_png")
        if png:
            try:
                from email.mime.image import MIMEImage
                img = MIMEImage(png, _subtype="png")
                img.add_header("Content-ID", "<nascurve>")
                img.add_header("Content-Disposition", "inline",
                               filename=f"nas_pnl_curve_{r['date']}.png")
                root.attach(img)
            except Exception as _ie:
                logger.warning(f"[NAS-EOD] curve attach failed: {_ie}")
        with smtplib.SMTP(cfg.get("smtp_host", "smtp.gmail.com"),
                          cfg.get("smtp_port", 587), timeout=25) as s:
            s.starttls()
            s.login(sender, pw)
            s.send_message(root)
        logger.info(f"[NAS-EOD] report sent: {subject}")
        return {"sent": True, "subject": subject, "report": r}
    except Exception as e:
        logger.error(f"[NAS-EOD] send failed: {e}", exc_info=True)
        return {"sent": False, "reason": str(e)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        rep = build_report()
        out = "backtest_data/nas_eod_preview.html"
        open(out, "w", encoding="utf-8").write(render_html(rep))
        print(f"preview written to {out}")
        print({k: v for k, v in rep.items() if k not in ("snaps", "pipeline")})
    else:
        print(send_eod_report())
