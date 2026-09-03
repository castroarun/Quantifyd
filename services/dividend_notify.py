"""
Dividend declaration notices — email (registrar-style), WhatsApp (Twilio), desktop.

Channels are configured via .env and stay DORMANT until credentials exist:
  EMAIL_SMTP_HOST / EMAIL_SMTP_PORT / EMAIL_SMTP_USER / EMAIL_SMTP_PASS / EMAIL_TO
  TWILIO_SID / TWILIO_TOKEN / TWILIO_WHATSAPP_FROM / WHATSAPP_TO / WHATSAPP_ENABLED=1

The email is styled after Indian registrar/depository intimations (formal serif
header, particulars table, record-date language) but branded QUANTIFYD FUND
SERVICES — our own fund desk, never an imitation of NSDL/NSE/CDSL.

Usage:
    from services.dividend_notify import notify_declaration
    notify_declaration(dict(book='Open Alpha', quarter='2026-Q3',
                            record_date='30-Sep-2026', payment_date='03-Oct-2026',
                            nav=7611120, new_profit=820679, dividend=295444,
                            source='profit', reserve=73300, hwm=7282848,
                            console_amount=295444))
"""
import json
import os
import smtplib
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALERT_FEED = Path('/tmp/quantifyd_dividend_alerts.log')

EMAIL_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#f2f2ee;font-family:Georgia,'Times New Roman',serif;color:#1a1a1a;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f2f2ee;padding:24px 0;"><tr><td align="center">
<table width="640" cellpadding="0" cellspacing="0" style="background:#ffffff;border:1px solid #d8d5cc;">
  <tr><td style="background:#0b2e59;padding:18px 28px;">
    <div style="color:#ffffff;font-size:19px;letter-spacing:0.06em;font-weight:bold;">QUANTIFYD FUND SERVICES</div>
    <div style="color:#b9c7db;font-size:11px;letter-spacing:0.14em;margin-top:3px;">DIVIDEND / INCOME DISTRIBUTION INTIMATION</div>
  </td></tr>
  <tr><td style="padding:24px 28px 8px;">
    <div style="font-size:13px;line-height:1.7;">
      Dear Investor,<br><br>
      This is to inform you that the under-mentioned scheme has <b>declared an income
      distribution</b> for the period stated below, credited to your designated
      distribution pool. Particulars are as follows:
    </div>
  </td></tr>
  <tr><td style="padding:14px 28px;">
    <table width="100%" cellpadding="7" cellspacing="0" style="border:1px solid #c9c5ba;font-size:12.5px;font-family:Arial,Helvetica,sans-serif;">
      <tr style="background:#eef1f6;">
        <td style="border-bottom:1px solid #c9c5ba;width:46%;color:#40404a;">Scheme / Book</td>
        <td style="border-bottom:1px solid #c9c5ba;"><b>{book}</b></td></tr>
      <tr><td style="border-bottom:1px solid #dedbd2;color:#40404a;">Distribution period</td>
        <td style="border-bottom:1px solid #dedbd2;">{quarter}</td></tr>
      <tr style="background:#fafaf7;"><td style="border-bottom:1px solid #dedbd2;color:#40404a;">Record date</td>
        <td style="border-bottom:1px solid #dedbd2;">{record_date}</td></tr>
      <tr><td style="border-bottom:1px solid #dedbd2;color:#40404a;">Credit / payment date</td>
        <td style="border-bottom:1px solid #dedbd2;">{payment_date}</td></tr>
      <tr style="background:#fafaf7;"><td style="border-bottom:1px solid #dedbd2;color:#40404a;">Book NAV on record date</td>
        <td style="border-bottom:1px solid #dedbd2;">&#8377; {nav:,}</td></tr>
      <tr><td style="border-bottom:1px solid #dedbd2;color:#40404a;">New profit above high-water mark</td>
        <td style="border-bottom:1px solid #dedbd2;">&#8377; {new_profit:,}</td></tr>
      <tr style="background:#eef7ef;"><td style="border-bottom:1px solid #dedbd2;color:#40404a;"><b>Distribution amount</b></td>
        <td style="border-bottom:1px solid #dedbd2;font-size:15px;"><b>&#8377; {dividend:,}</b></td></tr>
      <tr><td style="border-bottom:1px solid #dedbd2;color:#40404a;">Source</td>
        <td style="border-bottom:1px solid #dedbd2;">{source_label}</td></tr>
      <tr style="background:#fafaf7;"><td style="border-bottom:1px solid #dedbd2;color:#40404a;">Equalization reserve after distribution</td>
        <td style="border-bottom:1px solid #dedbd2;">&#8377; {reserve:,}</td></tr>
      <tr><td style="color:#40404a;">High-water mark (reset)</td><td>&#8377; {hwm:,}</td></tr>
    </table>
  </td></tr>
  <tr><td style="padding:6px 28px 18px;">
    <div style="font-size:12.5px;line-height:1.7;background:#fff8e6;border:1px solid #e6d9a8;padding:10px 14px;font-family:Arial,Helvetica,sans-serif;">
      <b>Bank transfer (manual step):</b> to move this distribution to your bank account,
      place a funds withdrawal of <b>&#8377; {console_amount:,}</b> in Zerodha Console
      (Funds &rarr; Withdraw). Broker APIs do not permit automated payouts.
    </div>
  </td></tr>
  <tr><td style="padding:0 28px 22px;">
    <div style="font-size:10.5px;color:#7a776e;line-height:1.6;font-family:Arial,Helvetica,sans-serif;border-top:1px solid #dedbd2;padding-top:12px;">
      This intimation is generated by your own Quantifyd fund desk under the quarterly
      high-water-mark distribution policy (40% of new profit; 10% of each distribution
      retained in the equalization reserve; reserve sustains up to half the trailing
      average through profitless quarters; capital is never distributed). Distributions
      are internal transfers within your own account and are not income from a third
      party. This is a system-generated notice &mdash; declarations are also recorded in
      the Ops &amp; Review registry.
    </div>
  </td></tr>
</table>
</td></tr></table></body></html>"""


def render_email(d):
    src = {'profit': 'Current-quarter profit above the high-water mark',
           'reserve': 'Dividend-equalization reserve (no new profit this quarter)',
           }.get(d.get('source', 'profit'), d.get('source'))
    return EMAIL_TEMPLATE.format(source_label=src, **{k: v for k, v in d.items() if k != 'source'})


def send_email(subject, html):
    host = os.getenv('EMAIL_SMTP_HOST')
    user = os.getenv('EMAIL_SMTP_USER')
    pw = os.getenv('EMAIL_SMTP_PASS')
    to = os.getenv('EMAIL_TO')
    if not all([host, user, pw, to]):
        return 'email DORMANT (set EMAIL_SMTP_HOST/USER/PASS and EMAIL_TO in .env)'
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = user
    msg['To'] = to
    msg.attach(MIMEText(html, 'html'))
    port = int(os.getenv('EMAIL_SMTP_PORT', '465'))
    with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context()) as s:
        s.login(user, pw)
        s.sendmail(user, [to], msg.as_string())
    return f'email sent to {to}'


def send_whatsapp(text):
    if os.getenv('WHATSAPP_ENABLED') != '1':
        return 'whatsapp DORMANT (set WHATSAPP_ENABLED=1 + TWILIO_* + WHATSAPP_TO in .env)'
    sid = os.getenv('TWILIO_SID')
    tok = os.getenv('TWILIO_TOKEN')
    frm = os.getenv('TWILIO_WHATSAPP_FROM')   # e.g. whatsapp:+14155238886
    to = os.getenv('WHATSAPP_TO')             # e.g. whatsapp:+91XXXXXXXXXX
    if not all([sid, tok, frm, to]):
        return 'whatsapp DORMANT (missing TWILIO_SID/TOKEN/TWILIO_WHATSAPP_FROM/WHATSAPP_TO)'
    import urllib.parse
    import urllib.request
    import base64
    url = f'https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json'
    data = urllib.parse.urlencode({'From': frm, 'To': to, 'Body': text}).encode()
    req = urllib.request.Request(url, data=data)
    auth = base64.b64encode(f'{sid}:{tok}'.encode()).decode()
    req.add_header('Authorization', f'Basic {auth}')
    with urllib.request.urlopen(req, timeout=20) as r:
        return f'whatsapp sent (HTTP {r.status})'


def desktop_alert(title, body):
    with open(ALERT_FEED, 'a') as f:
        f.write(json.dumps(dict(ts=str(datetime.now()), title=title, body=body)) + '\n')
    return f'alert appended to {ALERT_FEED}'


def notify_declaration(d):
    """Fire all channels for one declaration dict; returns per-channel status."""
    subject = (f"Income distribution — {d['book']} {d['quarter']}: "
               f"Rs {d['dividend']:,} ({d.get('source', 'profit')})")
    wa = (f"Quantifyd {d['book']} {d['quarter']}: dividend Rs {d['dividend']:,} "
          f"declared ({d.get('source', 'profit')}). Console withdrawal for bank: "
          f"Rs {d['console_amount']:,}. Reserve: Rs {d['reserve']:,}.")
    out = {}
    try:
        out['email'] = send_email(subject, render_email(d))
    except Exception as e:
        out['email'] = f'email FAILED: {e}'
    try:
        out['whatsapp'] = send_whatsapp(wa)
    except Exception as e:
        out['whatsapp'] = f'whatsapp FAILED: {e}'
    out['desktop'] = desktop_alert(subject, wa)
    return out


if __name__ == '__main__':
    sample = dict(book='Open Alpha (paper)', quarter='2026-Q3 (SAMPLE from backtest sim)',
                  record_date='30-Sep-2026', payment_date='03-Oct-2026',
                  nav=7_611_120, new_profit=820_679, dividend=295_444,
                  source='profit', reserve=73_300, hwm=7_282_848,
                  console_amount=295_444)
    html = render_email(sample)
    out = ROOT / 'static' / 'app' / 'dividend_notice_sample.html'
    out.write_text(html, encoding='utf-8')
    print('sample written:', out)
    print(notify_declaration(sample))
