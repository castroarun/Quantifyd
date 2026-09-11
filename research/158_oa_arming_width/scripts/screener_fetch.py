# -*- coding: utf-8 -*-
"""Annual financial history from Screener, for the 638 OA breakout candidates.

Yahoo returns four fiscal years, which makes a three-year growth test possible only from
mid-2026. Screener's company page carries roughly a decade, so FY2021 onward is available for
a January-2025 decision - which is what Arun's criteria actually need.

Per symbol it stores, per fiscal year, the rows the criteria need:

    sales, net_profit            -> three-year growth in both
    operating_profit            -> kept for context
    borrowings, equity          -> debt / equity, with equity = Equity Capital + Reserves
    total_assets                -> context
    roce_pct                    -> taken from Screener directly, NOT computed: its balance
                                   sheet lumps liabilities and never splits out current
                                   liabilities, so EBIT/(assets - current liabilities) off
                                   this page would be invention
    roe_pct                     -> computed as net_profit / equity, which this page does
                                   support

CONSOLIDATED vs STANDALONE. Screener serves consolidated figures where a company reports
them. Which page answered is recorded per symbol, because silently mixing the two makes the
debt and equity series inconsistent between names.

TWO HONEST RESIDUALS, recorded rather than smoothed over:
  * these are figures as they stand today, not as first reported. Annual restatements are
    usually small, but this is not a true as-reported vintage and a filter built on it
    carries that much look-ahead.
  * companies delisted since are absent from Screener, so the fundamental leg has a
    survivorship edge the price leg does not. The price database deliberately keeps dead
    names; Screener does not.

Polite by construction: one request per company, ~2.5s apart with jitter, a real
User-Agent, backoff on 429, and resume from cache so an interrupted run costs nothing.
"""
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
RES = ROOT / 'research/158_oa_arming_width/results'
CACHE = RES / 'screener_cache'
CACHE.mkdir(parents=True, exist_ok=True)
SYMS = json.load(open(RES / 'oa_candidates_2y.json'))

UA = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
      'Chrome/124.0 Safari/537.36')
WANT = {
    'sales': ['sales', 'revenue'],
    'net_profit': ['net profit'],
    'operating_profit': ['operating profit'],
    'borrowings': ['borrowings'],
    'equity_capital': ['equity capital'],
    'reserves': ['reserves'],
    'total_assets': ['total assets'],
    'roce_pct': ['roce %'],
}


def get(url, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={'User-Agent': UA, 'Accept-Language': 'en-US,en;q=0.9'})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.status, r.read().decode('utf-8', 'replace')
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 404, ''
            if e.code in (429, 503) and k < tries - 1:
                time.sleep(20 * (k + 1))     # back off rather than hammer
                continue
            return e.code, ''
        except Exception:
            if k < tries - 1:
                time.sleep(5)
                continue
            return 0, ''
    return 0, ''


def clean(x):
    """'1,234' / '17%' / '-45' / '' -> float or None."""
    s = str(x).replace(',', '').replace('%', '').strip()
    if s in ('', 'nan', '-'):
        return None
    try:
        return float(s)
    except ValueError:
        return None


class _Tables(HTMLParser):
    """Every <table> on the page as a list of rows of plain cell strings.

    html.parser rather than lxml: lxml is absent from the venv that runs live trading, and
    installing into it to read a web page is not a trade worth making.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self._t, self._r, self._c = [], None, None, None

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self._t = []
        elif tag == 'tr' and self._t is not None:
            self._r = []
        elif tag in ('td', 'th') and self._r is not None:
            self._c = []

    def handle_data(self, d):
        if self._c is not None:
            self._c.append(d)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self._c is not None:
            self._r.append(re.sub(r'\s+', ' ', ''.join(self._c)).strip())
            self._c = None
        elif tag == 'tr' and self._r is not None:
            if self._t is not None:
                self._t.append(self._r)
            self._r = None
        elif tag == 'table' and self._t is not None:
            self.tables.append(self._t)
            self._t = None


def parse(html):
    """-> {fy_end_iso: {field: value}} from every annual table on the page."""
    pr = _Tables()
    try:
        pr.feed(html)
    except Exception:
        return {}
    out = {}
    for tb in pr.tables:
        if len(tb) < 2 or len(tb[0]) < 3:
            continue
        head = tb[0]
        cols = {}
        for j, c in enumerate(head[1:], start=1):
            m = re.match(r'^([A-Z][a-z]{2})\s+(\d{4})$', c.strip())
            # Screener labels a fiscal year by the month it ends in: Mar 2024. Only
            # March year-ends are taken; a TTM or quarterly column is skipped.
            if m and m.group(1) == 'Mar':
                cols[j] = '%s-03-31' % m.group(2)
        if not cols:
            continue                       # quarterly table, or shareholding
        for row in tb[1:]:
            if not row:
                continue
            label = re.sub(r'[+\s]+$', '', row[0]).strip().lower()
            field = next((f for f, names in WANT.items()
                          if any(label == n for n in names)), None)
            if not field:
                continue
            for j, fy in cols.items():
                if j >= len(row):
                    continue
                v = clean(row[j])
                if v is not None:
                    out.setdefault(fy, {})[field] = v
    # equity = equity capital + reserves; ROE from it
    for fy, d in out.items():
        ec, rv = d.get('equity_capital'), d.get('reserves')
        if ec is not None and rv is not None:
            d['equity'] = ec + rv
            if d.get('net_profit') is not None and d['equity']:
                d['roe_pct'] = 100.0 * d['net_profit'] / d['equity']
        if d.get('borrowings') is not None and d.get('equity'):
            d['de'] = d['borrowings'] / d['equity']
    return out


def main():
    todo = [s for s in SYMS if not (CACHE / ('%s.json' % s)).exists()]
    print('%d symbols, %d already cached, %d to fetch'
          % (len(SYMS), len(SYMS) - len(todo), len(todo)), flush=True)
    ok = thin = miss = 0
    for i, s in enumerate(todo, 1):
        rec = dict(symbol=s, source=None, fy={}, note='')
        # Try BOTH and keep whichever carries more fiscal years. Stopping at the first
        # page that returns anything is what left CUPID with 2021 then 2024-2026: a company
        # that only recently began consolidated reporting has a thin consolidated page and a
        # long standalone one.
        best = None
        for path in ('consolidated/', ''):
            code, html = get('https://www.screener.in/company/%s/%s' % (s, path))
            if code == 200:
                fy = parse(html)
                if fy and (best is None or len(fy) > len(best[1])):
                    best = ('consolidated' if path else 'standalone', fy)
            elif code == 404:
                rec['note'] = 'not on screener'
            else:
                rec['note'] = 'http %s' % code
            time.sleep(1.5)
        if best:
            rec['source'], rec['fy'] = best
        n = len(rec['fy'])
        if n >= 5:
            ok += 1
        elif n:
            thin += 1
        else:
            miss += 1
        rec['n_fy'] = n
        json.dump(rec, open(CACHE / ('%s.json' % s), 'w'), indent=1)
        if i % 20 == 0:
            print('  %d/%d   %d with 5+ years, %d thin, %d missing'
                  % (i, len(todo), ok, thin, miss), flush=True)
        time.sleep(2.5 + random.random())
    print('DONE screener: %d with 5+ years, %d thin, %d missing' % (ok, thin, miss),
          flush=True)


if __name__ == '__main__':
    main()
