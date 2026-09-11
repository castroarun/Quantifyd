# -*- coding: utf-8 -*-
"""research/160 step 3 - full fundamental history from Screener, for the whole universe.

r/158 fetched 638 names and kept annual rows only. Arun's screen needs more than that: OPM%
across the years (his "margins steady or rising" step), face value (the only route to a
point-in-time share count, and therefore to a point-in-time market cap), and the quarterly
table (so "rising margins" can also be read on the recent quarters rather than only on
year-ends). So this re-fetches from scratch with a wider parser. Nothing under research/158
is touched.

PARSED BY data-date-key, NOT BY THE COLUMN CAPTION. Screener stamps every period column with
an exact end date, e.g. data-date-key="2023-06-30". r/158's parser instead matched the
caption /^Mar (\\d{4})$/ on every table, which the QUARTERLY table also satisfies for its
March quarter - a March-quarter Sales figure could be read as a full fiscal year wherever the
annual table had not yet published that year. Reading the stamped date and the section id
removes the whole class of confusion.

Sections: #quarters is quarterly; #profit-loss, #balance-sheet and #ratios are annual. A
column with no date key (the TTM column) is skipped.

LENDERS. Banks and NBFCs label their rows differently - Revenue rather than Sales, Financing
Profit / Financing Margin % rather than Operating Profit / OPM % - and Screener publishes no
ROCE for them at all, because capital employed is not a meaningful denominator. The aliases
are read; the missing ROCE is recorded as absent, never as a failure.

TOP RATIOS are today's values. They are stored ONLY for sanity checks - face value for the
share count, and market cap / current price to verify that share count reconciles. They can
never be used point-in-time and the panel builder does not read them as features.

Polite by construction: ONE worker, ~2.5 s apart with jitter, a real User-Agent, backoff on
429/503, consolidated page first and standalone only when consolidated is missing or thin
(which roughly halves the request count against r/158's fetch-both). Resumable: an
interrupted run costs nothing but the page in flight.
"""
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
RES = ROOT / 'research/160_quality_growth_near_ath/results'
CACHE = RES / 'screener_cache'
CACHE.mkdir(parents=True, exist_ok=True)

UA = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
      'Chrome/124.0 Safari/537.36')

# label (lowercased, nbsp and trailing '+' stripped) -> field.  Lists are aliases.
ANNUAL_ROWS = {
    'sales': ['sales', 'revenue'],
    'expenses': ['expenses'],
    'operating_profit': ['operating profit', 'financing profit'],
    'opm_pct': ['opm %', 'financing margin %'],
    'other_income': ['other income'],
    'interest': ['interest'],
    'depreciation': ['depreciation'],
    'profit_before_tax': ['profit before tax'],
    'net_profit': ['net profit'],
    'eps': ['eps in rs', 'eps'],
    'equity_capital': ['equity capital'],
    'reserves': ['reserves'],
    'borrowings': ['borrowings'],
    'other_liabilities': ['other liabilities'],
    'total_assets': ['total assets'],
    'roce_pct': ['roce %'],
    'dividend_payout_pct': ['dividend payout %'],
}
QUARTER_ROWS = {
    'sales': ['sales', 'revenue'],
    'operating_profit': ['operating profit', 'financing profit'],
    'opm_pct': ['opm %', 'financing margin %'],
    'net_profit': ['net profit'],
}
ANNUAL_SECTIONS = {'profit-loss', 'balance-sheet', 'ratios'}


def clean(x):
    """'1,234' / '17%' / '-45' / '' -> float or None."""
    s = str(x).replace(',', '').replace('%', '').replace('\xa0', '').strip()
    if s in ('', 'nan', '-', ''):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def norm_label(s):
    s = s.replace('\xa0', ' ')
    s = re.sub(r'[+\s]+$', '', s)
    return re.sub(r'\s+', ' ', s).strip().lower()


class Page(HTMLParser):
    """Pull every data table out, tagged with the section it lives in and with each
    column's stamped period-end date."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sec = None
        self.tables = []          # (section_id, {col_index: iso_date}, [ [cell,..], .. ])
        self._cols = None
        self._rows = None
        self._row = None
        self._cell = None
        self._ci = 0
        self._in_head = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'section' and a.get('id'):
            self.sec = a['id']
        elif tag == 'table':
            self._cols, self._rows = {}, []
        elif tag == 'thead':
            self._in_head = True
        elif tag == 'tr' and self._rows is not None:
            self._row, self._ci = [], 0
        elif tag in ('td', 'th') and self._row is not None:
            self._cell = []
            dk = a.get('data-date-key')
            # 'TTM' is stamped like a period but is not one; only real ISO dates are kept
            if tag == 'th' and self._in_head and dk and re.match(r'^\d{4}-\d{2}-\d{2}$', dk):
                self._cols[self._ci] = dk

    def handle_data(self, d):
        if self._cell is not None:
            self._cell.append(d)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self._cell is not None:
            self._row.append(re.sub(r'\s+', ' ', ''.join(self._cell)).strip())
            self._cell = None
            self._ci += 1
        elif tag == 'tr' and self._row is not None:
            if self._row:
                self._rows.append(self._row)
            self._row = None
        elif tag == 'thead':
            self._in_head = False
        elif tag == 'table' and self._rows is not None:
            if self._cols:
                self.tables.append((self.sec, self._cols, self._rows))
            self._cols = self._rows = None


TOP_RE = re.compile(
    r'<span class="name">\s*(?P<name>[^<]+?)\s*</span>.{0,200}?'
    r'<span class="nowrap value">(?P<val>.{0,400}?)</li>', re.S)
NUM_RE = re.compile(r'<span class="number">([^<]+)</span>')


def parse_top(html):
    """The #top-ratios list - TODAY's values, for sanity checks only."""
    i = html.find('id="top-ratios"')
    if i < 0:
        return {}
    seg = html[i:i + 6000]
    out = {}
    for m in TOP_RE.finditer(seg):
        nums = NUM_RE.findall(m.group('val'))
        if not nums:
            continue
        key = norm_label(m.group('name')).replace(' ', '_').replace('/', '_')
        v = clean(nums[0])
        if v is not None:
            out[key] = v
    return out


def parse(html):
    p = Page()
    try:
        p.feed(html)
    except Exception:
        return {}, {}
    annual, quarterly = {}, {}
    for sec, cols, rows in p.tables:
        if sec == 'quarters':
            want, sink = QUARTER_ROWS, quarterly
        elif sec in ANNUAL_SECTIONS:
            want, sink = ANNUAL_ROWS, annual
        else:
            continue                      # shareholding, peers, cash-flow: not needed
        for row in rows:
            if not row:
                continue
            label = norm_label(row[0])
            field = next((f for f, names in want.items() if label in names), None)
            if field is None:
                continue
            for j, d in cols.items():
                if j >= len(row):
                    continue
                v = clean(row[j])
                if v is not None:
                    sink.setdefault(d, {})[field] = v
    return annual, quarterly


def get(url, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={'User-Agent': UA, 'Accept-Language': 'en-US,en;q=0.9'})
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.status, r.read().decode('utf-8', 'replace')
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 404, ''
            if e.code in (429, 503) and k < tries - 1:
                time.sleep(30 * (k + 1))
                continue
            return e.code, ''
        except Exception:
            if k < tries - 1:
                time.sleep(6)
                continue
            return 0, ''
    return 0, ''


THIN = 6          # fewer annual years than this on consolidated -> also try standalone


def fetch_one(t):
    rec = dict(ticker=t, source=None, http={}, fetched_at=datetime.now().isoformat(
        timespec='seconds'), annual={}, quarterly={}, top={}, note='')
    code, html = get('https://www.screener.in/company/%s/consolidated/' % t)
    rec['http']['consolidated'] = code
    best = None
    if code == 200:
        a, q = parse(html)
        if a:
            best = ('consolidated', a, q, parse_top(html))
    if best is None or len(best[1]) < THIN:
        time.sleep(2.0 + random.random())
        code2, html2 = get('https://www.screener.in/company/%s/' % t)
        rec['http']['standalone'] = code2
        if code2 == 200:
            a2, q2 = parse(html2)
            if a2 and (best is None or len(a2) > len(best[1])):
                best = ('standalone', a2, q2, parse_top(html2))
        elif code2 == 404 and best is None:
            rec['note'] = 'not on screener'
    if best:
        rec['source'], rec['annual'], rec['quarterly'], rec['top'] = best
    elif not rec['note']:
        rec['note'] = 'http %s' % rec['http']
    rec['n_fy'] = len(rec['annual'])
    rec['n_q'] = len(rec['quarterly'])
    return rec


def main():
    import csv
    with open(RES / 'universe.csv') as f:
        uni = list(csv.DictReader(f))
    # dedupe by screener ticker (a symbol and its -BE twin are the same company); keep the
    # liquidity ordering so the names that could actually be traded are cached first
    seen, todo = set(), []
    for r in uni:
        t = r['screener_ticker']
        if t in seen:
            continue
        seen.add(t)
        todo.append(t)
    pending = [t for t in todo if not (CACHE / ('%s.json' % t)).exists()]
    print('%d universe symbols -> %d distinct tickers, %d cached, %d to fetch'
          % (len(uni), len(todo), len(todo) - len(pending), len(pending)), flush=True)
    t0 = time.time()
    ok = thin = miss = 0
    for i, t in enumerate(pending, 1):
        try:
            rec = fetch_one(t)
        except Exception as e:                      # never let one page kill the run
            rec = dict(ticker=t, source=None, http={}, note='exc %s' % e,
                       annual={}, quarterly={}, top={}, n_fy=0, n_q=0,
                       fetched_at=datetime.now().isoformat(timespec='seconds'))
        json.dump(rec, open(CACHE / ('%s.json' % t), 'w'), indent=1)
        if rec['n_fy'] >= 5:
            ok += 1
        elif rec['n_fy']:
            thin += 1
        else:
            miss += 1
        if i % 25 == 0 or i == len(pending):
            el = time.time() - t0
            eta = el / i * (len(pending) - i) / 60.0
            print('  %d/%d  %.0f%%  %d full, %d thin, %d missing  |  %.1f s/name, ETA %.0f min'
                  % (i, len(pending), 100.0 * i / len(pending), ok, thin, miss,
                     el / i, eta), flush=True)
        time.sleep(2.5 + random.random())
    print('DONE screener fetch: %d full, %d thin, %d missing in %.0f min'
          % (ok, thin, miss, (time.time() - t0) / 60.0), flush=True)


if __name__ == '__main__':
    sys.exit(main())
