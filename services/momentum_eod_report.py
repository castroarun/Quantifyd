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


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        rep = build_report()
        out = "backtest_data/momentum_eod_preview.html"
        open(out, "w", encoding="utf-8").write(render_html(rep))
        print(f"preview written to {out}")
        print({k: v for k, v in rep.items() if k not in ("holdings", "entries", "exits", "closed_today")})
    else:
        print(send_eod_report())
