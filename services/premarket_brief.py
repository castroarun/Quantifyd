"""
Pre-Market Brief
================
Daily 08:00 IST email digest for the operator. Pulls live market data from
yfinance, holdings calendar from holdings_events.db, F&O ban list from NSE.

Phase 1 (this commit): KPI tiles, Market Context, Your Book Today,
rule-based bias verdict for the One-Liner. F&O ban intersection.

Phase 2 (follow-up): news headlines (RSS) + Claude Code routine for
sentiment synthesis + earnings calendar + FII/DII flow.

Public entry points:
    run_premarket_brief() -> dict   # build raw + send email
    get_latest_brief()    -> dict   # most recent persisted brief

API endpoints (in app.py):
    GET  /api/premarket/brief/latest  -> dict
    GET  /api/premarket/brief/raw     -> dict (un-styled)
    POST /api/premarket/brief/run     -> trigger now
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import smtplib
import sqlite3
from datetime import datetime, date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / 'backtest_data'
LATEST_PATH = DATA_DIR / 'premarket_brief_latest.json'

# yfinance symbols. GIFT Nifty (NSE-IX) trades 21h/day; Yahoo proxies it
# with `^NSEI` for spot close, but the futures continuation is at NSEIY=F.
# Fallbacks try multiple variants because Yahoo occasionally drops symbols.
YF_SYMBOLS = {
    'GIFT_NIFTY': ['NSEIY=F', '^NSEI'],
    'SP500':      ['^GSPC'],
    'NASDAQ':     ['^IXIC'],
    'DOW':        ['^DJI'],
    'US_VIX':     ['^VIX'],
    'USDINR':     ['INR=X', 'USDINR=X'],
    'INDIA_VIX':  ['^INDIAVIX'],
    'BRENT':      ['BZ=F'],
    'GOLD':       ['GC=F'],
    'UST10Y':     ['^TNX'],
    'DXY':        ['DX-Y.NYB', 'DX=F'],
    'NIKKEI':     ['^N225'],
    'HSI':        ['^HSI'],
    'KOSPI':      ['^KS11'],
}

# Round and format helpers
def _fmt_num(x, decimals=2):
    if x is None:
        return '—'
    try:
        return f'{float(x):,.{decimals}f}'
    except Exception:
        return str(x)


def _fmt_pct(x, decimals=2):
    if x is None:
        return '—'
    try:
        v = float(x)
        sign = '+' if v >= 0 else ''
        return f'{sign}{v:.{decimals}f}%'
    except Exception:
        return str(x)


def _fmt_signed(x, decimals=2):
    if x is None:
        return '—'
    try:
        v = float(x)
        sign = '+' if v >= 0 else ''
        return f'{sign}{v:.{decimals}f}'
    except Exception:
        return str(x)


def _color(v):
    """Return color hex for a numeric move."""
    if v is None:
        return '#6b7280'
    try:
        return '#16a34a' if float(v) >= 0 else '#dc2626'
    except Exception:
        return '#6b7280'


# ---------------------------------------------------------------------------
# Market data via yfinance
# ---------------------------------------------------------------------------

def _yf_quote(symbols: list[str]) -> Optional[dict]:
    """Fetch most-recent close + day move for first symbol that resolves."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error('[brief] yfinance not installed')
        return None

    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            # `history(period='5d')` is more reliable than `info` for liquid symbols
            hist = t.history(period='5d', auto_adjust=False)
            if hist is None or hist.empty or 'Close' not in hist:
                continue
            closes = hist['Close'].dropna()
            if len(closes) < 2:
                continue
            last = float(closes.iloc[-1])
            prev = float(closes.iloc[-2])
            pct = ((last - prev) / prev * 100) if prev else 0.0
            return {
                'symbol': sym,
                'last': last,
                'prev': prev,
                'change': last - prev,
                'pct': pct,
            }
        except Exception as e:
            logger.debug(f'[brief] yf {sym} failed: {e}')
            continue
    return None


def fetch_market_data() -> dict:
    """Pull all market data points in one shot. Returns dict keyed by name.
    Each value is {symbol, last, prev, change, pct} or None on failure."""
    out = {}
    for key, syms in YF_SYMBOLS.items():
        out[key] = _yf_quote(syms)
    return out


# ---------------------------------------------------------------------------
# Holdings events from holdings_events.db
# ---------------------------------------------------------------------------

def fetch_holdings_events_today() -> list[dict]:
    """Today's events on user's holdings (results, ex-div, splits, bonus)."""
    db_path = DATA_DIR / 'holdings_events.db'
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            today = date.today().isoformat()
            rows = conn.execute(
                "SELECT * FROM holdings_events WHERE event_date = ? "
                "ORDER BY tradingsymbol",
                (today,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f'[brief] holdings_events read failed: {e}')
        return []


def fetch_holdings_events_upcoming(days_ahead: int = 7) -> list[dict]:
    """Events in the next N days for context (excluding today)."""
    db_path = DATA_DIR / 'holdings_events.db'
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            today = date.today().isoformat()
            from datetime import timedelta
            end = (date.today() + timedelta(days=days_ahead)).isoformat()
            rows = conn.execute(
                "SELECT * FROM holdings_events WHERE event_date > ? "
                "AND event_date <= ? ORDER BY event_date, tradingsymbol",
                (today, end),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f'[brief] holdings_events upcoming read failed: {e}')
        return []


# ---------------------------------------------------------------------------
# F&O ban list from NSE archive CSV
# ---------------------------------------------------------------------------

def fetch_fno_ban_list() -> list[str]:
    """Download today's F&O securities-in-ban-period list from NSE archive."""
    try:
        import requests
        url = 'https://archives.nseindia.com/content/fo/fo_secban.csv'
        r = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Quantifyd PreMarketBrief)',
        })
        if r.status_code != 200:
            return []
        # CSV format: Sr.No.,Security
        reader = csv.reader(io.StringIO(r.text))
        out = []
        for i, row in enumerate(reader):
            if not row or i == 0:
                continue
            sec = row[-1].strip().upper() if row[-1].strip() else ''
            if sec and sec not in ('SECURITY', 'NIL'):
                out.append(sec)
        return out
    except Exception as e:
        logger.warning(f'[brief] fno ban list fetch failed: {e}')
        return []


# ---------------------------------------------------------------------------
# Holdings symbols (for ban-list intersection)
# ---------------------------------------------------------------------------

def fetch_holdings_symbols() -> list[str]:
    """Pull current holdings tradingsymbols from Kite session if available."""
    try:
        from services.kite_service import get_kite
        kite = get_kite()
        if kite is None:
            return []
        h = kite.holdings()
        return [str(x.get('tradingsymbol', '')).upper() for x in h
                if x.get('tradingsymbol')]
    except Exception as e:
        logger.debug(f'[brief] kite holdings unavailable: {e}')
        return []


# ---------------------------------------------------------------------------
# News headlines from RSS feeds
# ---------------------------------------------------------------------------

NEWS_FEEDS = [
    ('Moneycontrol Markets', 'https://www.moneycontrol.com/rss/marketreports.xml'),
    ('Mint Markets',         'https://www.livemint.com/rss/markets'),
    ('Reuters India Markets', 'https://www.reuters.com/markets/asia/rss'),
]


def fetch_news_headlines(max_per_feed: int = 8, total_cap: int = 25) -> list[dict]:
    """Pull recent headlines from each RSS feed; dedupe + cap.

    Returns list of {title, link, source, published, summary} dicts ordered
    newest first. The cloud routine receives this and picks the top 5 with
    sentiment tags.
    """
    try:
        import feedparser
    except ImportError:
        logger.warning('[brief] feedparser missing; news headlines skipped')
        return []

    seen_titles = set()
    out = []
    for src_name, url in NEWS_FEEDS:
        try:
            d = feedparser.parse(url)
            for entry in d.entries[:max_per_feed]:
                title = (entry.get('title') or '').strip()
                if not title or title.lower() in seen_titles:
                    continue
                seen_titles.add(title.lower())
                out.append({
                    'title': title,
                    'link': entry.get('link', ''),
                    'source': src_name,
                    'published': entry.get('published') or entry.get('updated') or '',
                    'summary': (entry.get('summary') or '')[:300],
                })
                if len(out) >= total_cap:
                    return out
        except Exception as e:
            logger.warning(f'[brief] RSS {src_name} failed: {e}')
    return out


# ---------------------------------------------------------------------------
# Bias verdict (rule-based, until LLM synthesis is wired in)
# ---------------------------------------------------------------------------

def compute_bias(market: dict) -> dict:
    """Rule-based open-bias verdict.

    Logic:
      - GIFT Nifty pct is the primary signal (most predictive)
      - S&P 500 pct from prev close confirms direction
      - US VIX direction: rising VIX = caution
      - India VIX: rising = caution
      - Brent: spikes >1.5% are mildly negative for India (CAD)

    Returns: {label, color, summary}
    """
    gift = (market.get('GIFT_NIFTY') or {}).get('pct')
    sp = (market.get('SP500') or {}).get('pct')
    vix = (market.get('US_VIX') or {}).get('pct')
    invix = (market.get('INDIA_VIX') or {}).get('pct')
    brent = (market.get('BRENT') or {}).get('pct')

    score = 0.0
    if gift is not None:
        score += gift * 1.5
    if sp is not None:
        score += sp * 1.0
    if vix is not None:
        score -= max(0, vix) * 0.3  # only rising VIX hurts
    if invix is not None:
        score -= max(0, invix) * 0.3
    if brent is not None and brent > 1.5:
        score -= 0.3

    if score >= 0.4:
        label = 'POSITIVE'
        color = '#16a34a'
    elif score <= -0.4:
        label = 'CAUTIOUS'
        color = '#dc2626'
    else:
        label = 'MIXED'
        color = '#6b7280'

    # One-liner narrative — concise factual summary
    parts = []
    if gift is not None:
        gift_pts = gift / 100 * (market.get('GIFT_NIFTY', {}).get('prev') or 24500)
        parts.append(
            f"NIFTY indicated <b style='color:{_color(gift)}'>{_fmt_signed(gift_pts, 0)} pts</b> on "
            f"<b>{_fmt_pct(gift, 2)}</b> GIFT premium"
        )
    if sp is not None:
        if sp >= 0:
            parts.append(f"US strength carries in (S&amp;P {_fmt_pct(sp, 2)})")
        else:
            parts.append(f"US weakness drags (S&amp;P {_fmt_pct(sp, 2)})")
    summary = '. '.join(parts) + '.' if parts else 'Awaiting overnight data.'

    return {'label': label, 'color': color, 'summary': summary, 'score': round(score, 2)}


# ---------------------------------------------------------------------------
# Build the raw brief dict
# ---------------------------------------------------------------------------

def build_brief() -> dict:
    """Assemble the full raw brief dict. Idempotent; safe to call any time."""
    logger.info('[brief] building...')
    market = fetch_market_data()
    holdings_today = fetch_holdings_events_today()
    holdings_upcoming = fetch_holdings_events_upcoming(7)
    fno_ban = fetch_fno_ban_list()
    holdings_syms = fetch_holdings_symbols()
    ban_in_holdings = sorted(set(fno_ban) & set(holdings_syms))
    bias = compute_bias(market)

    return {
        'report_type': 'premarket_brief',
        'generated_at': datetime.now().isoformat(),
        'date': date.today().isoformat(),
        'market': market,
        'bias': bias,
        'holdings': {
            'today': holdings_today,
            'upcoming': holdings_upcoming,
            'symbols': holdings_syms,
        },
        'fno_ban': {
            'all': fno_ban,
            'in_holdings': ban_in_holdings,
            'count': len(fno_ban),
        },
        # Phase 2: raw news for cloud routine to synthesize
        'headlines_raw': fetch_news_headlines(),
        # Phase 2 placeholders (filled by cloud routine)
        'headlines_synthesized': None,  # [{'tag': POS|NEG|NEU, 'text': '...', 'source': '...'}, ...]
        'narrative_summary': None,       # 2-3 paragraph synthesis
        # Future Phase 2c
        'earnings_today': None,
        'flow': None,
        'strategy_outlook': _strategy_outlook(market, holdings_today, ban_in_holdings),
    }


def _strategy_outlook(market: dict, holdings_today: list[dict],
                      ban_in_holdings: list[str]) -> dict:
    """Deterministic per-system outlook based on the day's data."""
    invix = (market.get('INDIA_VIX') or {}).get('last')
    invix_pct = (market.get('INDIA_VIX') or {}).get('pct')
    # Stocks with results today (any holding with event_type='results')
    result_syms = [h.get('tradingsymbol', '') for h in holdings_today
                   if (h.get('event_type') or '').lower() in ('results', 'result')]

    orb_text = (
        f"ORB cash universe — {len(result_syms)} of your watchlist exclude due to "
        f"results today: {', '.join(result_syms[:6])}." if result_syms
        else "ORB cash watchlist clear — no overlapping result events today."
    )
    nas_text = (
        f"NAS strangle — India VIX {_fmt_num(invix, 2)} ({_fmt_pct(invix_pct, 2)}); "
        + ("vol elevated, consider wider SL." if (invix_pct or 0) > 5
           else "vol stable, baseline lots OK.")
    )
    orbi_text = "ORB index (paper) — strangle master tick every 60s, EOD squareoff 15:25 IST."

    return {
        'orb_cash': orb_text,
        'nas_strangle': nas_text,
        'orb_index': orbi_text,
    }


# ---------------------------------------------------------------------------
# HTML renderer (Mock C v5 template, populated with live data)
# ---------------------------------------------------------------------------

def render_html(brief: dict) -> str:
    """Render the brief dict as the locked Mock C v5 HTML."""
    m = brief.get('market', {})
    bias = brief.get('bias', {})
    h = brief.get('holdings', {})
    ban = brief.get('fno_ban', {})

    # Pull individual quotes safely
    def q(key, attr):
        v = m.get(key) or {}
        return v.get(attr)

    # Date display: 28-APR-2026 · MON · 08:00 IST
    now = datetime.now()
    date_disp = now.strftime('%d-%b-%Y · %a · %H:%M IST').upper().replace(
        now.strftime('%a').upper(), now.strftime('%a').upper()
    )

    # KPI tile values
    gift_last = q('GIFT_NIFTY', 'last')
    gift_pct = q('GIFT_NIFTY', 'pct')
    gift_pts_open = (gift_pct / 100 * gift_last) if (gift_pct and gift_last) else None

    sp_last = q('SP500', 'last')
    sp_pct = q('SP500', 'pct')
    vix_last = q('US_VIX', 'last')
    vix_pct = q('US_VIX', 'pct')

    usdinr = q('USDINR', 'last')
    usdinr_pct = q('USDINR', 'pct')

    invix_last = q('INDIA_VIX', 'last')
    invix_pct = q('INDIA_VIX', 'pct')

    # Market context
    brent_last = q('BRENT', 'last')
    brent_chg = q('BRENT', 'change')
    brent_pct = q('BRENT', 'pct')

    gold_last = q('GOLD', 'last')
    gold_chg = q('GOLD', 'change')
    gold_pct = q('GOLD', 'pct')

    ust_last = q('UST10Y', 'last')
    ust_chg = q('UST10Y', 'change')

    dxy_last = q('DXY', 'last')
    dxy_chg = q('DXY', 'change')
    dxy_pct = q('DXY', 'pct')

    nikkei_pct = q('NIKKEI', 'pct')
    hsi_pct = q('HSI', 'pct')
    kospi_pct = q('KOSPI', 'pct')

    # Holdings rows
    def event_pill(ev_type: str) -> str:
        et = (ev_type or '').lower()
        if et in ('results', 'result'):
            return f"<span style='color:#a16207;font-size:10px;font-weight:700'>RESULT</span>"
        if et in ('dividend', 'ex-div'):
            return f"<span style='color:#16a34a;font-size:10px;font-weight:700'>EX-DIV</span>"
        if et == 'split':
            return f"<span style='color:#1e40af;font-size:10px;font-weight:700'>SPLIT</span>"
        if et == 'bonus':
            return f"<span style='color:#1e40af;font-size:10px;font-weight:700'>BONUS</span>"
        return f"<span style='color:#6b7280;font-size:10px;font-weight:700'>{(ev_type or '').upper()}</span>"

    holdings_today_rows = ''
    for ev in h.get('today', [])[:8]:
        sym = ev.get('tradingsymbol', '')
        purpose = ev.get('purpose') or ev.get('details') or ''
        holdings_today_rows += (
            f"<tr style='border-bottom:1px solid #f1f5f9'>"
            f"<td style='padding:5px 20px;width:110px;font-weight:700;color:#1e40af;font-family:\"SF Mono\",Consolas,Monaco,monospace'>{sym}</td>"
            f"<td style='padding:5px 0;width:60px'>{event_pill(ev.get('event_type'))}</td>"
            f"<td style='padding:5px 20px'>{purpose}</td>"
            f"</tr>"
        )
    if not holdings_today_rows:
        holdings_today_rows = (
            "<tr><td colspan='3' style='padding:10px 20px;color:#6b7280;font-size:12px'>"
            "No corporate-action events on your holdings today.</td></tr>"
        )

    # Upcoming holdings (next 7 days)
    upcoming_rows = ''
    for ev in h.get('upcoming', [])[:6]:
        sym = ev.get('tradingsymbol', '')
        purpose = ev.get('purpose') or ev.get('details') or ''
        ed = ev.get('event_date', '')
        upcoming_rows += (
            f"<tr style='border-bottom:1px solid #f1f5f9'>"
            f"<td style='padding:5px 20px;width:110px;font-weight:700;color:#1e40af;font-family:\"SF Mono\",Consolas,Monaco,monospace'>{sym}</td>"
            f"<td style='padding:5px 0;width:80px;font-family:\"SF Mono\",Consolas,Monaco,monospace;color:#374151'>{ed}</td>"
            f"<td style='padding:5px 0;width:60px'>{event_pill(ev.get('event_type'))}</td>"
            f"<td style='padding:5px 20px'>{purpose}</td>"
            f"</tr>"
        )

    # F&O ban summary
    if ban.get('in_holdings'):
        ban_text = (
            f"<span style='color:#dc2626;font-weight:600'>"
            f"{', '.join(ban['in_holdings'])} in your holdings</span> · "
            f"{ban.get('count', 0)} total in ban today"
        )
    else:
        ban_text = (
            f"{ban.get('count', 0)} in ban today · "
            f"<span style='color:#16a34a'>none in your holdings ✓</span>"
        )

    # Strategy outlook
    so = brief.get('strategy_outlook', {})

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Pre-Market Brief</title></head>
<body style='margin:0;font-family:"Amazon Ember","AmazonEmber","Helvetica Neue",Helvetica,Arial,sans-serif;background:#fafafa;color:#1a1d1f;padding:20px'>
<div style='max-width:640px;margin:0 auto;background:white;border:1px solid #e5e7eb;border-radius:4px'>

<!-- HEADER -->
<div style='padding:18px 24px;background:#ffffff;border-radius:4px 4px 0 0;border-bottom:2px solid #1e40af'>
  <table style='width:100%;border-collapse:collapse'><tr>
    <td style='vertical-align:middle'>
      <div style='font-size:11px;letter-spacing:1.5px;color:#1e40af;font-weight:600'>QUANTIFYD · PRE-MARKET</div>
      <div style='font-size:13px;font-weight:400;margin-top:4px;font-family:"SF Mono",Consolas,Monaco,monospace;color:#374151'>{date_disp}</div>
    </td>
    <td style='vertical-align:middle;text-align:right;width:130px'>
      <div style='display:inline-block;color:{bias.get("color", "#6b7280")};font-size:13px;font-weight:700;letter-spacing:1.5px;border:1.5px solid {bias.get("color", "#6b7280")};padding:8px 18px;border-radius:4px'>{bias.get("label", "—")}</div>
    </td>
  </tr></table>
</div>

<!-- ONE-LINER -->
<div style='padding:18px 24px 18px 27px;background:#ffffff;border-bottom:1px solid #e5e7eb;border-left:4px solid #1e40af'>
  <div style='font-size:11px;color:#6b7280;letter-spacing:1.5px;text-transform:uppercase;font-weight:700;margin-bottom:8px'>Today's One-Liner</div>
  <div style='font-size:13px;line-height:1.55;color:#374151'>
    {bias.get('summary', '—')}
  </div>
</div>

<!-- KPI TILES -->
<div style='padding:18px 14px;background:#fafafa;border-bottom:1px solid #e5e7eb'>
  <table style='width:100%;border-collapse:separate;border-spacing:6px 0'><tr>
    <td style='width:25%;background:#ffffff;border:1px solid #e5e7eb;border-radius:4px;padding:14px;vertical-align:top'>
      <div style='font-size:10px;color:#6b7280;letter-spacing:1px;font-weight:600;text-transform:uppercase'>GIFT Nifty</div>
      <div style='font-size:20px;font-weight:700;margin-top:6px;font-family:"SF Mono",Consolas,Monaco,monospace;color:#111827'>{_fmt_num(gift_last, 0)}</div>
      <div style='font-size:13px;font-weight:700;color:{_color(gift_pct)};margin-top:2px'>{_fmt_pct(gift_pct, 2)}</div>
      <div style='font-size:10px;color:#6b7280;margin-top:4px'>≈ {_fmt_signed(gift_pts_open, 0)} NIFTY open</div>
    </td>
    <td style='width:25%;background:#ffffff;border:1px solid #e5e7eb;border-radius:4px;padding:14px;vertical-align:top'>
      <div style='font-size:10px;color:#6b7280;letter-spacing:1px;font-weight:600;text-transform:uppercase'>S&amp;P 500</div>
      <div style='font-size:20px;font-weight:700;margin-top:6px;font-family:"SF Mono",Consolas,Monaco,monospace;color:#111827'>{_fmt_num(sp_last, 0)}</div>
      <div style='font-size:13px;font-weight:700;color:{_color(sp_pct)};margin-top:2px'>{_fmt_pct(sp_pct, 2)}</div>
      <div style='font-size:10px;color:#6b7280;margin-top:4px'>US VIX {_fmt_num(vix_last, 1)} <span style='color:{_color(vix_pct)}'>{_fmt_pct(vix_pct, 1)}</span></div>
    </td>
    <td style='width:25%;background:#ffffff;border:1px solid #e5e7eb;border-radius:4px;padding:14px;vertical-align:top'>
      <div style='font-size:10px;color:#6b7280;letter-spacing:1px;font-weight:600;text-transform:uppercase'>USDINR</div>
      <div style='font-size:20px;font-weight:700;margin-top:6px;font-family:"SF Mono",Consolas,Monaco,monospace;color:#111827'>{_fmt_num(usdinr, 2)}</div>
      <div style='font-size:13px;font-weight:700;color:{_color(usdinr_pct)};margin-top:2px'>{_fmt_pct(usdinr_pct, 2)}</div>
      <div style='font-size:10px;color:#6b7280;margin-top:4px'>{"INR firmer" if (usdinr_pct or 0) <= 0 else "INR weaker"}</div>
    </td>
    <td style='width:25%;background:#ffffff;border:1px solid #e5e7eb;border-radius:4px;padding:14px;vertical-align:top'>
      <div style='font-size:10px;color:#6b7280;letter-spacing:1px;font-weight:600;text-transform:uppercase'>India VIX</div>
      <div style='font-size:20px;font-weight:700;margin-top:6px;font-family:"SF Mono",Consolas,Monaco,monospace;color:#111827'>{_fmt_num(invix_last, 2)}</div>
      <div style='font-size:13px;font-weight:700;color:{_color(invix_pct)};margin-top:2px'>{_fmt_pct(invix_pct, 2)}</div>
      <div style='font-size:10px;color:#6b7280;margin-top:4px'>{"vol cooling" if (invix_pct or 0) <= 0 else "vol rising"}</div>
    </td>
  </tr></table>
</div>

<!-- MARKET CONTEXT -->
<div style='padding:0'>
  <table style='width:100%;border-collapse:collapse;font-family:"SF Mono",Consolas,Monaco,monospace;font-size:12px'>
    <tr style='background:#ffffff'>
      <td colspan='4' style='padding:10px 20px 6px 20px;font-family:-apple-system,Segoe UI,sans-serif;font-size:11px;letter-spacing:1.5px;color:#1e40af;font-weight:700;border-bottom:1px solid #e5e7eb'>MARKET CONTEXT</td>
    </tr>
    <tr style='border-bottom:1px solid #f1f5f9'>
      <td style='padding:6px 20px;width:35%;color:#6b7280'>Brent crude</td>
      <td style='padding:6px 0;font-weight:600'>${_fmt_num(brent_last, 2)}</td>
      <td style='padding:6px 0;color:{_color(brent_chg)}'>{_fmt_signed(brent_chg, 2)}</td>
      <td style='padding:6px 20px;color:{_color(brent_pct)}'>{_fmt_pct(brent_pct, 2)}</td>
    </tr>
    <tr style='border-bottom:1px solid #f1f5f9'>
      <td style='padding:6px 20px;color:#6b7280'>Gold (spot)</td>
      <td style='padding:6px 0;font-weight:600'>${_fmt_num(gold_last, 0)}</td>
      <td style='padding:6px 0;color:{_color(gold_chg)}'>{_fmt_signed(gold_chg, 2)}</td>
      <td style='padding:6px 20px;color:{_color(gold_pct)}'>{_fmt_pct(gold_pct, 2)}</td>
    </tr>
    <tr style='border-bottom:1px solid #f1f5f9'>
      <td style='padding:6px 20px;color:#6b7280'>10Y UST yield</td>
      <td style='padding:6px 0;font-weight:600'>{_fmt_num(ust_last, 2)}%</td>
      <td style='padding:6px 0;color:{_color(ust_chg)}'>{_fmt_signed((ust_chg or 0) * 100 / 10, 0)} bps</td>
      <td style='padding:6px 20px;color:#6b7280'>{"watch Powell" if abs(ust_chg or 0) > 0.02 else "stable"}</td>
    </tr>
    <tr style='border-bottom:1px solid #f1f5f9'>
      <td style='padding:6px 20px;color:#6b7280'>Dollar Index <span style='color:#9ca3af;font-size:10px'>· DXY</span></td>
      <td style='padding:6px 0;font-weight:600'>{_fmt_num(dxy_last, 2)}</td>
      <td style='padding:6px 0;color:{_color(dxy_chg)}'>{_fmt_signed(dxy_chg, 2)}</td>
      <td style='padding:6px 20px;color:{_color(dxy_pct)}'>{_fmt_pct(dxy_pct, 2)}</td>
    </tr>
    <tr style='background:#fafafa'>
      <td colspan='4' style='padding:8px 20px;font-family:-apple-system,Segoe UI,sans-serif;font-size:11px;color:#6b7280'>
        <b style='color:#374151'>Asia 07:30:</b>
        Nikkei <span style='color:{_color(nikkei_pct)}'>{_fmt_pct(nikkei_pct, 2)}</span> ·
        Hang Seng <span style='color:{_color(hsi_pct)}'>{_fmt_pct(hsi_pct, 2)}</span> ·
        Kospi <span style='color:{_color(kospi_pct)}'>{_fmt_pct(kospi_pct, 2)}</span>
      </td>
    </tr>
  </table>
</div>

<!-- YOUR BOOK -->
<div style='padding:0;border-top:1px solid #e5e7eb'>
  <div style='padding:8px 20px;background:#ffffff;color:#1e40af;font-size:11px;font-weight:700;letter-spacing:1.5px;border-bottom:1px solid #e5e7eb'>► YOUR BOOK · TODAY</div>
  <table style='width:100%;border-collapse:collapse;font-size:12px'>
    <tr style='background:#fafafa'>
      <td colspan='3' style='padding:6px 20px;font-size:10px;letter-spacing:1px;color:#374151;font-weight:700'>EVENTS ON HOLDINGS</td>
    </tr>
    {holdings_today_rows}
    {("<tr style='background:#fafafa'><td colspan='3' style='padding:6px 20px;font-size:10px;letter-spacing:1px;color:#374151;font-weight:700;border-top:1px solid #f1f5f9'>NEXT 7 DAYS</td></tr>" + upcoming_rows) if upcoming_rows else ""}
    <tr style='background:#fafafa'>
      <td colspan='3' style='padding:6px 20px;font-size:10px;letter-spacing:1px;color:#374151;font-weight:700;border-top:1px solid #f1f5f9'>F&amp;O BAN LIST</td>
    </tr>
    <tr><td colspan='3' style='padding:8px 20px;font-size:12px;color:#374151'>{ban_text}</td></tr>
  </table>
</div>

{_render_headlines_section(brief)}

<!-- STRATEGY OUTLOOK -->
<div style='padding:0;border-top:1px solid #e5e7eb'>
  <div style='padding:8px 20px;background:#ffffff;color:#1e40af;font-size:11px;font-weight:700;letter-spacing:1.5px;border-bottom:1px solid #e5e7eb'>► STRATEGY OUTLOOK</div>
  <table style='width:100%;border-collapse:collapse;font-size:12px;line-height:1.5'>
    <tr style='border-bottom:1px solid #f1f5f9'>
      <td style='padding:8px 20px;width:110px;font-weight:700;color:#1e40af;vertical-align:top'>ORB cash</td>
      <td style='padding:8px 20px 8px 0;color:#374151'>{so.get('orb_cash', '—')}</td>
    </tr>
    <tr style='border-bottom:1px solid #f1f5f9'>
      <td style='padding:8px 20px;font-weight:700;color:#1e40af;vertical-align:top'>NAS x8</td>
      <td style='padding:8px 20px 8px 0;color:#374151'>{so.get('nas_strangle', '—')}</td>
    </tr>
    <tr>
      <td style='padding:8px 20px;font-weight:700;color:#1e40af;vertical-align:top'>ORB index</td>
      <td style='padding:8px 20px 8px 0;color:#374151'>{so.get('orb_index', '—')}</td>
    </tr>
  </table>
</div>

<!-- FOOTER -->
<div style='padding:14px 20px;text-align:center;color:#9ca3af;font-size:10px;line-height:1.6;border-top:1px solid #e5e7eb;border-radius:0 0 4px 4px'>
  Sources: yfinance · holdings_events.db · NSE F&amp;O archive<br>
  Phase 1 — Phase 2 will add news headlines, earnings calendar, and FII/DII flow.
</div>

</div>
</body>
</html>"""


def _render_headlines_section(brief: dict) -> str:
    """Render headlines: synthesized (with tags) preferred, else raw list."""
    synth = brief.get('headlines_synthesized')
    raw = brief.get('headlines_raw') or []

    rows = ''
    if synth and isinstance(synth, list):
        # Synthesized: each item has 'tag' (POS/NEG/NEU) + 'text' + optional 'source'
        for h in synth[:5]:
            tag = (h.get('tag') or 'NEU').upper()
            text = h.get('text') or ''
            source = h.get('source') or ''
            tag_color = {'POS': '#16a34a', 'NEG': '#dc2626', 'NEU': '#6b7280'}.get(tag, '#6b7280')
            rows += (
                f"<tr style='border-bottom:1px solid #f1f5f9'>"
                f"<td style='padding:8px 0 8px 20px;width:50px;vertical-align:top'>"
                f"<span style='color:{tag_color};font-size:10px;font-weight:700'>{tag}</span></td>"
                f"<td style='padding:8px 20px 8px 0;color:#374151;line-height:1.5;font-size:12px'>"
                f"{text}{(' <i style=\"color:#9ca3af\">' + source + '</i>') if source else ''}"
                f"</td></tr>"
            )
        sub = ('Synthesized by Claude routine · '
               + str(len(brief.get('headlines_raw') or [])) + ' raw inputs')
    elif raw:
        # Raw fallback: just titles + sources
        for h in raw[:5]:
            rows += (
                f"<tr style='border-bottom:1px solid #f1f5f9'>"
                f"<td style='padding:8px 20px;color:#374151;line-height:1.5;font-size:12px'>"
                f"{h.get('title', '')} <i style='color:#9ca3af'>· {h.get('source', '')}</i>"
                f"</td></tr>"
            )
        sub = 'Raw RSS — sentiment synthesis pending'
    else:
        return ''  # nothing to show

    return f"""<!-- OVERNIGHT HEADLINES -->
<div style='padding:0;border-top:1px solid #e5e7eb'>
  <div style='padding:8px 20px;background:#ffffff;color:#1e40af;font-size:11px;font-weight:700;letter-spacing:1.5px;border-bottom:1px solid #e5e7eb'>► OVERNIGHT · 5 STORIES <span style='color:#9ca3af;font-weight:400;letter-spacing:0;text-transform:none;font-size:10px'>· {sub}</span></div>
  <table style='width:100%;border-collapse:collapse'>{rows}</table>
</div>"""


# ---------------------------------------------------------------------------
# WhatsApp send (Twilio) — concise plain-text summary of the brief
# ---------------------------------------------------------------------------

def _whatsapp_text(brief: dict) -> str:
    """Compact plain-text summary suitable for WhatsApp (≤ 1500 chars)."""
    bias = brief.get('bias', {})
    m = brief.get('market', {})

    def _q(k, attr):
        v = m.get(k) or {}
        return v.get(attr)

    gift = _q('GIFT_NIFTY', 'pct')
    sp = _q('SP500', 'pct')
    invix = _q('INDIA_VIX', 'last')
    invix_pct = _q('INDIA_VIX', 'pct')
    brent = _q('BRENT', 'pct')

    lines = [
        f"*The Morning Note · {brief.get('date', '')}*",
        f"Bias: *{bias.get('label', '—')}*",
        '',
        f"GIFT Nifty: {_fmt_pct(gift, 2)}",
        f"S&P 500 (Fri): {_fmt_pct(sp, 2)}",
        f"India VIX: {_fmt_num(invix, 2)} ({_fmt_pct(invix_pct, 1)})",
        f"Brent: {_fmt_pct(brent, 2)}",
    ]

    h_today = (brief.get('holdings') or {}).get('today') or []
    if h_today:
        lines.append('')
        lines.append('*Holdings today:*')
        for ev in h_today[:5]:
            lines.append(
                f"• {ev.get('tradingsymbol', '')} — {(ev.get('event_type') or '').upper()} · "
                f"{(ev.get('purpose') or ev.get('details') or '')[:60]}"
            )

    ban = (brief.get('fno_ban') or {}).get('in_holdings') or []
    if ban:
        lines.append('')
        lines.append(f"⚠ F&O ban (your holdings): {', '.join(ban)}")

    # Synthesized headlines (top 3)
    synth = brief.get('headlines_synthesized') or []
    if synth:
        lines.append('')
        lines.append('*Top headlines:*')
        for h in synth[:3]:
            tag = h.get('tag', 'NEU')
            txt = h.get('text', '')
            lines.append(f"[{tag}] {txt[:120]}")

    lines.append('')
    lines.append('— full brief in your inbox')
    return '\n'.join(lines)


def _send_whatsapp(brief: dict) -> bool:
    """Dispatch the brief summary via Twilio WhatsApp using the existing config."""
    try:
        from config import ORB_DEFAULTS
        cfg = ORB_DEFAULTS
        if not cfg.get('whatsapp_enabled', False):
            logger.info('[brief] WhatsApp disabled by config')
            return False
        sid = cfg.get('twilio_account_sid', '')
        tok = cfg.get('twilio_auth_token', '')
        frm = cfg.get('twilio_whatsapp_from', '')
        to = cfg.get('twilio_whatsapp_to', '')
        if not all([sid, tok, frm, to]):
            logger.warning('[brief] WhatsApp config incomplete; skipping')
            return False

        from twilio.rest import Client
        client = Client(sid, tok)
        client.messages.create(body=_whatsapp_text(brief), from_=frm, to=to)
        logger.info(f"[brief] WhatsApp sent for {brief.get('date')}")
        return True
    except Exception as e:
        logger.error(f'[brief] WhatsApp send failed: {e}', exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Email send
# ---------------------------------------------------------------------------

def _send_email(brief: dict, html: str) -> bool:
    try:
        from config import ORB_DEFAULTS
        cfg = ORB_DEFAULTS
        if not cfg.get('email_enabled', True):
            logger.info('[brief] email disabled')
            return False
        sender = cfg.get('email_from', '')
        password = cfg.get('email_app_password', '')
        recipient = cfg.get('email_to', '')
        if not sender or not password or not recipient:
            logger.warning('[brief] email config incomplete; skipping')
            return False
        verdict = brief.get('bias', {}).get('label', '—')
        # Format date as DD-MON-YYYY for the subject (per user preference)
        try:
            from datetime import datetime as _dt
            disp_date = _dt.strptime(brief['date'], '%Y-%m-%d').strftime('%d-%b-%Y')
        except Exception:
            disp_date = brief.get('date', '')
        subject = f"[The Morning Note] {verdict} · {disp_date}"
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = f"Quantifyd <{sender}>"
        msg['To'] = recipient
        msg.attach(MIMEText(html, 'html'))
        with smtplib.SMTP(cfg.get('smtp_host', 'smtp.gmail.com'),
                          cfg.get('smtp_port', 587), timeout=20) as srv:
            srv.starttls()
            srv.login(sender, password)
            srv.send_message(msg)
        logger.info(f'[brief] email sent: {subject}')
        return True
    except Exception as e:
        logger.error(f'[brief] email send failed: {e}', exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Persistence + entry point
# ---------------------------------------------------------------------------

def _persist(brief: dict):
    try:
        DATA_DIR.mkdir(exist_ok=True)
        LATEST_PATH.write_text(json.dumps(brief, indent=2, default=str))
    except Exception as e:
        logger.warning(f'[brief] persist failed: {e}')


def get_latest_brief() -> Optional[dict]:
    if not LATEST_PATH.exists():
        return None
    try:
        return json.loads(LATEST_PATH.read_text())
    except Exception:
        return None


def run_premarket_brief_build_only() -> dict:
    """08:00 IST scheduler step 1: build raw data, persist. Do NOT email yet.

    The cloud Claude routine fires at 08:02 IST, fetches the raw JSON,
    synthesizes headlines/narrative, and POSTs back to /api/premarket/
    brief/synthesized which then renders + sends. If that doesn't happen
    by 08:08 IST, the fallback step sends the un-synthesized version.
    """
    logger.info('[brief] build (no email yet) starting')
    brief = build_brief()
    brief['_built_at'] = datetime.now().isoformat()
    brief['_email_sent'] = False
    _persist(brief)
    logger.info(
        f'[brief] built bias={brief.get("bias", {}).get("label")} '
        f'headlines_raw={len(brief.get("headlines_raw") or [])}'
    )
    return brief


def receive_synthesis_and_send(synthesis: dict) -> dict:
    """Called by /api/premarket/brief/synthesized when cloud routine POSTs.

    synthesis dict shape:
      {
        'narrative_summary': str,        # optional, replaces One-Liner
        'headlines_synthesized': [       # required
          {'tag': 'POS'|'NEG'|'NEU', 'text': str, 'source': str},
          ... up to 5
        ]
      }
    """
    brief = get_latest_brief() or {}
    if not brief:
        # No build today yet — build now
        brief = build_brief()
        brief['_built_at'] = datetime.now().isoformat()
        brief['_email_sent'] = False

    # Merge synthesis into brief
    if isinstance(synthesis, dict):
        if synthesis.get('headlines_synthesized'):
            brief['headlines_synthesized'] = synthesis['headlines_synthesized']
        if synthesis.get('narrative_summary'):
            brief['narrative_summary'] = synthesis['narrative_summary']
            # Replace the One-Liner with the narrative if provided
            if brief.get('bias'):
                brief['bias']['summary'] = synthesis['narrative_summary']

    brief['_synthesized_at'] = datetime.now().isoformat()
    brief['_email_sent'] = True
    _persist(brief)
    html = render_html(brief)
    sent = _send_email(brief, html)
    _send_whatsapp(brief)
    return {'sent': sent, 'brief_date': brief.get('date')}


def run_premarket_brief_fallback() -> dict:
    """08:08 IST scheduler: if no synthesized email went out yet, send the
    un-synthesized version so the operator always gets a brief."""
    brief = get_latest_brief() or {}
    if brief.get('_email_sent'):
        logger.info('[brief] fallback skipped — already sent (synthesized)')
        return {'skipped': True, 'reason': 'already sent'}
    if not brief or brief.get('date') != date.today().isoformat():
        # No build today yet — build fresh
        logger.warning('[brief] fallback: no build today, building now')
        brief = build_brief()
        brief['_built_at'] = datetime.now().isoformat()
    logger.info('[brief] fallback sending un-synthesized brief')
    brief['_email_sent'] = True
    brief['_fallback'] = True
    _persist(brief)
    html = render_html(brief)
    sent = _send_email(brief, html)
    _send_whatsapp(brief)
    return {'sent': sent, 'fallback': True}


def run_premarket_brief() -> dict:
    """Original full-cycle: build + persist + email immediately.
    Used by /api/premarket/brief/run for manual triggers + when no cloud
    routine is configured."""
    logger.info('[brief] full run (build + send) starting')
    brief = build_brief()
    brief['_built_at'] = datetime.now().isoformat()
    brief['_email_sent'] = True
    _persist(brief)
    html = render_html(brief)
    _send_email(brief, html)
    _send_whatsapp(brief)
    logger.info(
        f'[brief] done bias={brief.get("bias", {}).get("label")} '
        f'holdings_today={len(brief.get("holdings", {}).get("today", []))}'
    )
    return brief


# ---------------------------------------------------------------------------
# CLI for local testing
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'build'
    if cmd == 'build':
        b = build_brief()
        print(json.dumps({
            'bias': b['bias'],
            'gift': b['market'].get('GIFT_NIFTY'),
            'sp500': b['market'].get('SP500'),
            'india_vix': b['market'].get('INDIA_VIX'),
            'holdings_today': len(b['holdings']['today']),
            'holdings_upcoming': len(b['holdings']['upcoming']),
            'fno_ban_count': b['fno_ban']['count'],
            'fno_in_holdings': b['fno_ban']['in_holdings'],
        }, indent=2, default=str))
    elif cmd == 'send':
        run_premarket_brief()
    elif cmd == 'preview':
        b = build_brief()
        html = render_html(b)
        out = DATA_DIR / 'premarket_brief_preview.html'
        out.write_text(html, encoding='utf-8')
        print(f'preview: {out}')
    else:
        print('usage: python -m services.premarket_brief [build|send|preview]')
        sys.exit(1)
