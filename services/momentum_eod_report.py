"""services/momentum_eod_report.py — EOD email for the Momentum-30 book (mirrors nas_eod_report.py).

Sends ONCE per trading day (~15:35 IST, after the 15:15 EOD job) summarising: today's entries/exits,
current holdings, NAV / P&L, and the macro gate. Reuses config.ORB_DEFAULTS Gmail SMTP settings. Never
raises. Preview:  python3 -m services.momentum_eod_report preview  ·  Send:  python3 -m services.momentum_eod_report
"""
from __future__ import annotations
import logging
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)
REASON = {"DONCHIAN": "Donchian stop", "GATE_RISK_OFF": "Macro gate risk-off",
          "BUFFER_ROTATE": "Buffer rotation", "REBALANCE": "Monthly rebalance", "SEED": "Seed"}


def build_report() -> dict:
    from services import momentum_paper as mp
    st = mp.get_state()
    today = date.today().isoformat()
    fills = []
    try:
        for row in mp._conn().execute(
                "SELECT ts,symbol,side,price,qty,value,reason FROM mp_fills WHERE ts LIKE ? ORDER BY ts",
                (today + "%",)):
            fills.append(dict(row))
    except Exception as e:
        logger.warning(f"[MOM-EOD] fills read failed: {e}")
    entries = [f for f in fills if f["side"] == "BUY"]
    exits = [f for f in fills if f["side"] == "SELL"]
    closed_today = [c for c in st.get("closed", []) if str(c.get("exit_date", "")).startswith(today)]
    pnl_today = sum(c.get("net_pnl", 0) for c in closed_today)
    holdings = [h for h in st.get("holdings", []) if not h.get("is_cash")]
    issues = []
    if st.get("last_daily") != today:
        issues.append(f"EOD job may not have run today (last_daily={st.get('last_daily')}) — Donchian stops/gate not applied")
    if st.get("mode") == "LIVE":
        # surface any live-execution concern for the alert flag
        if st.get("n_holdings", 0) == 0 and st.get("gate") == "ON":
            issues.append("Gate ON but 0 holdings — check reconcile / fills")
    return dict(
        date=today, mode=st.get("mode", "PAPER"), gate=st.get("gate", "ON"),
        nav=st.get("nav", 0), cash=st.get("cash", 0), equity=st.get("equity", 0),
        invested_pct=st.get("invested_pct", 0), total_return=st.get("total_return_pct", 0),
        unrealized=st.get("unrealized", 0), realized_net=st.get("realized_net", 0),
        n_holdings=st.get("n_holdings", 0), holdings=holdings,
        entries=entries, exits=exits, closed_today=closed_today, pnl_today=pnl_today,
        gate_last=st.get("gate_last"), gate_sma=st.get("gate_sma"), gate_gap=st.get("gate_gap_pct"),
        stcg_booked=st.get("stcg_booked", 0), last_daily=st.get("last_daily"),
        last_monthly=st.get("last_monthly"), issues=issues)


def _inr(n):
    try:
        return "₹{:,.0f}".format(round(n))
    except Exception:
        return "—"


def render_html(r) -> str:
    on = r["gate"] == "ON"
    live = r["mode"] == "LIVE"
    modecol = "#c0392b" if live else "#6b7581"
    gatecol = "#1f9d55" if on else "#d97706"
    pos = r["pnl_today"] >= 0

    def row_cells(cells):
        return "".join(f'<td style="padding:6px 10px;border-bottom:1px solid #eee">{c}</td>' for c in cells)

    # today's activity
    act = ""
    if r["entries"] or r["exits"]:
        act += '<table style="border-collapse:collapse;width:100%;font-size:13px;margin:6px 0 4px">'
        act += '<tr style="color:#888;text-align:left"><th style="padding:6px 10px">Action</th><th style="padding:6px 10px">Stock</th><th style="padding:6px 10px">Price</th><th style="padding:6px 10px">Qty</th><th style="padding:6px 10px">Why</th></tr>'
        for f in r["exits"]:
            act += f'<tr>{row_cells(["<b style=color:#c0392b>SELL</b>", f["symbol"], _inr(f["price"]), int(f["qty"]), REASON.get(f["reason"], f["reason"])])}</tr>'
        for f in r["entries"]:
            act += f'<tr>{row_cells(["<b style=color:#1f9d55>BUY</b>", f["symbol"], _inr(f["price"]), int(f["qty"]), REASON.get(f["reason"], f["reason"])])}</tr>'
        act += "</table>"
    else:
        act = '<div style="color:#888;font-size:13px;padding:6px 0">No trades today — holding the book.</div>'

    # holdings
    hold = '<table style="border-collapse:collapse;width:100%;font-size:13px;margin-top:6px">'
    hold += '<tr style="color:#888;text-align:left"><th style="padding:6px 10px">Stock</th><th style="padding:6px 10px">Wt</th><th style="padding:6px 10px">Now ₹</th><th style="padding:6px 10px">P&L%</th><th style="padding:6px 10px">Days</th><th style="padding:6px 10px">Donchian stop</th></tr>'
    for h in r["holdings"]:
        pc = h.get("pnl_pct")
        pcs = ("+" if (pc or 0) >= 0 else "") + (f"{pc}%" if pc is not None else "—")
        col = "#1f9d55" if (pc or 0) >= 0 else "#c0392b"
        sym = h.get("symbol", "")
        cells = [f"<b>{sym}</b>", f'{h.get("weight", 0)}%', h.get("price", "—"),
                 f'<span style="color:{col}">{pcs}</span>', h.get("days", "—"), h.get("stop", "—")]
        hold += f"<tr>{row_cells(cells)}</tr>"
    hold += "</table>"

    flag = " ⚠️" if r["issues"] else ""
    issues_html = ""
    if r["issues"]:
        issues_html = '<div style="background:#fff3cd;border:1px solid #ffe08a;border-radius:6px;padding:10px;margin:10px 0;font-size:13px;color:#8a6d00">⚠️ ' + "<br>".join(r["issues"]) + "</div>"
    return f"""<div style="font-family:Segoe UI,Arial,sans-serif;max-width:640px;margin:0 auto;color:#222">
      <div style="padding:14px 18px;background:#0d1117;border-radius:10px 10px 0 0;color:#e6edf3">
        <div style="font-size:12px;letter-spacing:.05em;color:#9aa4af">MOMENTUM-30 · EOD REPORT{flag}</div>
        <div style="font-size:20px;font-weight:700;margin-top:2px">NAV {_inr(r['nav'])}
          <span style="font-size:12px;color:#9aa4af">· total {('+' if r['total_return']>=0 else '')}{r['total_return']}%</span></div>
        <div style="margin-top:6px">
          <span style="background:{modecol};padding:2px 9px;border-radius:5px;font-size:11px;font-weight:700">{r['mode']}</span>
          <span style="background:{gatecol};padding:2px 9px;border-radius:5px;font-size:11px;font-weight:700;margin-left:6px">GATE {'RISK-ON' if on else 'RISK-OFF'}</span>
          <span style="color:#9aa4af;font-size:12px;margin-left:8px">{r['date']}</span>
        </div>
      </div>
      <div style="border:1px solid #e6e6e6;border-top:0;border-radius:0 0 10px 10px;padding:16px 18px">
        {issues_html}
        <table style="width:100%;font-size:13px;margin-bottom:8px"><tr>
          <td>Today's P&L <b style="color:{'#1f9d55' if pos else '#c0392b'}">{('+' if pos else '')}{_inr(r['pnl_today'])}</b></td>
          <td>Invested <b>{r['invested_pct']}%</b></td>
          <td>Cash <b>{_inr(r['cash'])}</b></td>
          <td>Holdings <b>{r['n_holdings']}</b></td>
        </tr></table>
        <div style="font-weight:600;margin:12px 0 2px">Today's activity</div>
        {act}
        <div style="font-weight:600;margin:14px 0 2px">Holdings</div>
        {hold}
        <div style="font-size:12px;color:#666;margin-top:10px">
          Gate: NIFTYBEES {r['gate_last']} vs 100-DMA {r['gate_sma']} ({('+' if (r['gate_gap'] or 0)>=0 else '')}{r['gate_gap']}%) ·
          Realized net {_inr(r['realized_net'])} · Booked STCG {_inr(r['stcg_booked'])}
        </div>
        <div style="font-size:11px;color:#999;margin-top:12px;line-height:1.5">
          {'PAPER — no real orders placed.' if not live else 'LIVE — real orders placed via Kite CNC.'}
          Auto-sends each trading day ~15:35 IST after the EOD job (Donchian stops + weekly gate; monthly rebalance on the last trading day).
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
            logger.warning("[MOM-EOD] email config incomplete; not sent")
            return {"sent": False, "reason": "email config incomplete", "report": r}
        verdict = (f"+{_inr(r['pnl_today'])}" if r["pnl_today"] >= 0 else f"-{_inr(abs(r['pnl_today']))}")
        act = f"{len(r['entries'])}B/{len(r['exits'])}S" if (r["entries"] or r["exits"]) else "hold"
        flag = " ⚠️" if r["issues"] else ""
        subject = f"[Momentum EOD · {r['mode']}] NAV {_inr(r['nav'])} · {r['n_holdings']} held · {act} · today {verdict} · {r['date']}{flag}"
        root = MIMEMultipart("alternative")
        root["Subject"] = subject
        root["From"] = f"Quantifyd <{sender}>"
        root["To"] = rcpt
        root.attach(MIMEText(html, "html"))
        with smtplib.SMTP(cfg.get("smtp_host", "smtp.gmail.com"), cfg.get("smtp_port", 587), timeout=25) as s:
            s.starttls()
            s.login(sender, pw)
            s.send_message(root)
        logger.info(f"[MOM-EOD] report sent: {subject}")
        return {"sent": True, "subject": subject}
    except Exception as e:
        logger.error(f"[MOM-EOD] send failed: {e}", exc_info=True)
        return {"sent": False, "reason": str(e)}


def send_alert(title: str, message: str, priority: str = "normal") -> dict:
    """Immediate alert email — order rejection/timeout, reconcile mismatch, EOD-job miss. Never raises."""
    try:
        from config import ORB_DEFAULTS as cfg
        sender = cfg.get("email_from", ""); pw = cfg.get("email_app_password", ""); rcpt = cfg.get("email_to", "")
        if not (sender and pw and rcpt):
            logger.warning("[MOM-ALERT] email config incomplete; not sent")
            return {"sent": False, "reason": "email config incomplete"}
        icon = "🔴" if priority == "high" else "⚠️"
        subject = f"[Momentum ALERT] {icon} {title}"
        html = (f'<div style="font-family:Segoe UI,Arial,sans-serif;max-width:560px;margin:0 auto;color:#222">'
                f'<div style="padding:12px 16px;background:{"#c0392b" if priority=="high" else "#d97706"};color:#fff;'
                f'border-radius:8px 8px 0 0;font-weight:700">{icon} {title}</div>'
                f'<div style="border:1px solid #eee;border-top:0;border-radius:0 0 8px 8px;padding:14px 16px;'
                f'font-size:14px;line-height:1.6;white-space:pre-wrap">{message}</div></div>')
        root = MIMEMultipart("alternative"); root["Subject"] = subject
        root["From"] = f"Quantifyd <{sender}>"; root["To"] = rcpt
        root.attach(MIMEText(html, "html"))
        with smtplib.SMTP(cfg.get("smtp_host", "smtp.gmail.com"), cfg.get("smtp_port", 587), timeout=25) as s:
            s.starttls(); s.login(sender, pw); s.send_message(root)
        logger.info(f"[MOM-ALERT] sent: {title}")
        return {"sent": True}
    except Exception as e:
        logger.error(f"[MOM-ALERT] failed: {e}")
        return {"sent": False, "reason": str(e)}


def build_monthly() -> dict:
    from services import momentum_paper as mp
    st = mp.get_state()
    today = date.today(); ym = today.strftime("%Y-%m")
    conn = mp._conn()
    navrows = [dict(r) for r in conn.execute("SELECT d,nav,bench_close FROM mp_nav ORDER BY d")]
    month = [r for r in navrows if str(r["d"]).startswith(ym)]
    prior = [r for r in navrows if str(r["d"]) < ym + "-01"]
    nav_start = prior[-1]["nav"] if prior else (month[0]["nav"] if month else st.get("nav", 0))
    nav_end = month[-1]["nav"] if month else st.get("nav", 0)
    mret = (nav_end / nav_start - 1) * 100 if nav_start else 0
    b0 = (prior[-1]["bench_close"] if prior else (month[0]["bench_close"] if month else None))
    b1 = month[-1]["bench_close"] if month else None
    bret = (b1 / b0 - 1) * 100 if (b0 and b1) else None
    closed = [dict(r) for r in conn.execute("SELECT * FROM mp_closed WHERE exit_date LIKE ? ORDER BY exit_date DESC", (ym + "%",))]
    return dict(ym=ym, monthname=today.strftime("%B %Y"), mode=st.get("mode", "PAPER"),
                nav_start=nav_start, nav_end=nav_end, mret=mret, bret=bret,
                closed=closed, realized=sum(c["net_pnl"] for c in closed),
                stcg=sum(c["stcg_tax"] for c in closed),
                holdings=[h for h in st.get("holdings", []) if not h.get("is_cash")],
                n_holdings=st.get("n_holdings", 0), total_return=st.get("total_return_pct", 0))


def render_monthly_html(r) -> str:
    beat = (r["bret"] is not None and r["mret"] >= r["bret"])
    rows = ""
    for c in r["closed"][:40]:
        col = "#1f9d55" if c["net_pnl"] >= 0 else "#c0392b"
        rows += (f'<tr><td style="padding:5px 9px;border-bottom:1px solid #eee">{c["symbol"]}</td>'
                 f'<td style="padding:5px 9px;border-bottom:1px solid #eee;color:#888">{c["entry_date"]}→{c["exit_date"]}</td>'
                 f'<td style="padding:5px 9px;border-bottom:1px solid #eee;color:{col}">{_inr(c["net_pnl"])}</td>'
                 f'<td style="padding:5px 9px;border-bottom:1px solid #eee;color:#888">{REASON.get(c["reason"], c["reason"])}</td></tr>')
    hold = ", ".join(h.get("symbol", "") for h in r["holdings"]) or "—"
    bcmp = (f'{("+" if r["bret"]>=0 else "")}{r["bret"]:.1f}%' if r["bret"] is not None else "—")
    return f"""<div style="font-family:Segoe UI,Arial,sans-serif;max-width:640px;margin:0 auto;color:#222">
      <div style="padding:16px 18px;background:#0d1117;color:#e6edf3;border-radius:10px 10px 0 0">
        <div style="font-size:12px;letter-spacing:.05em;color:#9aa4af">MOMENTUM-30 · MONTHLY REPORT · {r['mode']}</div>
        <div style="font-size:22px;font-weight:700;margin-top:2px">{r['monthname']}</div>
        <div style="font-size:14px;margin-top:6px">Month {('+' if r['mret']>=0 else '')}{r['mret']:.1f}%
          <span style="color:#9aa4af">vs NIFTYBEES {bcmp} · {'✓ beat' if beat else 'lagged'} the index</span></div>
      </div>
      <div style="border:1px solid #e6e6e6;border-top:0;border-radius:0 0 10px 10px;padding:16px 18px">
        <table style="width:100%;font-size:13px;margin-bottom:10px"><tr>
          <td>NAV {_inr(r['nav_start'])} → <b>{_inr(r['nav_end'])}</b></td>
          <td>Realized <b style="color:{'#1f9d55' if r['realized']>=0 else '#c0392b'}">{_inr(r['realized'])}</b></td>
          <td>STCG booked <b>{_inr(r['stcg'])}</b></td>
          <td>Total since start <b>{('+' if r['total_return']>=0 else '')}{r['total_return']}%</b></td>
        </tr></table>
        <div style="font-weight:600;margin:12px 0 4px">Closed this month ({len(r['closed'])})</div>
        <table style="border-collapse:collapse;width:100%;font-size:13px">{rows or '<tr><td style=color:#888;padding:6px>No exits this month.</td></tr>'}</table>
        <div style="font-size:13px;color:#444;margin-top:12px"><b>Now holding ({r['n_holdings']}):</b> {hold}</div>
        <div style="font-size:11px;color:#999;margin-top:12px">Auto-sends on the last trading day of each month. {'PAPER — no real orders.' if r['mode']!='LIVE' else 'LIVE.'}</div>
      </div>
    </div>"""


def send_monthly_report() -> dict:
    try:
        r = build_monthly()
        html = render_monthly_html(r)
        from config import ORB_DEFAULTS as cfg
        sender = cfg.get("email_from", ""); pw = cfg.get("email_app_password", ""); rcpt = cfg.get("email_to", "")
        if not (sender and pw and rcpt):
            return {"sent": False, "reason": "email config incomplete"}
        subject = f"[Momentum MONTHLY · {r['mode']}] {r['monthname']} · {('+' if r['mret']>=0 else '')}{r['mret']:.1f}% · NAV {_inr(r['nav_end'])}"
        root = MIMEMultipart("alternative"); root["Subject"] = subject
        root["From"] = f"Quantifyd <{sender}>"; root["To"] = rcpt
        root.attach(MIMEText(html, "html"))
        with smtplib.SMTP(cfg.get("smtp_host", "smtp.gmail.com"), cfg.get("smtp_port", 587), timeout=25) as s:
            s.starttls(); s.login(sender, pw); s.send_message(root)
        logger.info(f"[MOM-MONTHLY] sent: {subject}")
        return {"sent": True, "subject": subject}
    except Exception as e:
        logger.error(f"[MOM-MONTHLY] send failed: {e}", exc_info=True)
        return {"sent": False, "reason": str(e)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "monthly":
        print(send_monthly_report())
    elif len(sys.argv) > 1 and sys.argv[1] == "monthly-preview":
        rep = build_monthly()
        open("backtest_data/momentum_monthly_preview.html", "w", encoding="utf-8").write(render_monthly_html(rep))
        print("preview written"); print({k: v for k, v in rep.items() if k not in ("closed", "holdings")})
    elif len(sys.argv) > 1 and sys.argv[1] == "preview":
        rep = build_report()
        out = "backtest_data/momentum_eod_preview.html"
        open(out, "w", encoding="utf-8").write(render_html(rep))
        print(f"preview written to {out}")
        print({k: v for k, v in rep.items() if k not in ("holdings", "entries", "exits", "closed_today")})
    else:
        print(send_eod_report())
