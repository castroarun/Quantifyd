"""
Quarterly dividend engine — True North (momentum) + Open Alpha (bluesky).

ADOPTED POLICY (Arun, 03-Sep-2026, research/142 dividend_sim_v2 variant E):
  - Declaration on the first run on/after each calendar quarter end
  - new profit  = NAV above the flow-adjusted high-water mark (HWM)
  - entitlement = 25% of new profit  (leaves the book; 75% keeps compounding)
  - payout CAP  = last paid dividend x 1.075 (the income line only steps up
    ~7.5%/qtr; it never spikes). First-ever payout seeds the line.
  - surplus entitlement above the cap -> equalization RESERVE (conceptually a
    liquid-ETF pocket OUTSIDE book NAV; accrues ~6% p.a., credited quarterly)
  - dry / weak quarters: the reserve tops the payout up toward the cap line
  - if the reserve empties mid-drought the payout falls to what is available
    and the line re-bases from there (an honest dividend cut)
  - capital is NEVER invaded: outflow is capped at the book's liquid cash
    (cash + CASHIETF); positions are never force-sold. Any liquidity-clipped
    entitlement simply stays in the book (it is NOT owed later).

MECHANICS — why this never touches trading logic:
  Between record dates the engines run untouched and reinvest 100% of booked
  profits (position sizing keys off full NAV, exactly as coded). On the record
  date this module removes the entitlement from book cash the same way a user
  withdrawal does; from the next cycle the engine sizes off the smaller NAV
  automatically. To the trading loop a dividend is indistinguishable from a
  withdrawal.

  Deposits/withdrawals between quarters do NOT create/destroy "profit": the
  HWM is adjusted by net external flows (from each book's fund-flow ledger)
  before comparing to NAV.

State:
  Open Alpha -> "dividend" key inside backtest_data/bluesky_paper_state.json
               (edited under the engine's own lockfile, atomic replace)
  True North -> "dividend" row in mp_state (backtest_data/momentum_paper.db)

Each declaration fires services.dividend_notify (email/WhatsApp when armed,
desktop alert always) with the Console withdrawal amount.
"""
import json
import os
import sqlite3
import time
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

POLICY = dict(baseline=0.25, cap_growth_q=0.075, reserve_rate_pa=0.06)
DECLARE_WINDOW_DAYS = 12   # declare on first run within N days after quarter end


# ───────────────────────── book adapters ─────────────────────────
class OpenAlphaBook:
    name = 'Open Alpha'
    slug = 'openalpha'
    STATE = ROOT / 'backtest_data' / 'bluesky_paper_state.json'
    LOCK = ROOT / 'backtest_data' / 'bluesky_paper_state.lock'
    UI = ROOT / 'static' / 'app' / 'bluesky_paper.json'

    def _lock(self):
        for _ in range(15):
            try:
                fd = os.open(self.LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, b'dividend_engine'); os.close(fd)
                return True
            except FileExistsError:
                time.sleep(2)
        return False

    def _unlock(self):
        if self.LOCK.exists():
            self.LOCK.unlink()

    def _load(self):
        return json.load(open(self.STATE))

    def _save(self, st):
        tmp = self.STATE.with_suffix('.json.tmp')
        json.dump(st, open(tmp, 'w'), indent=1, default=str)
        os.replace(tmp, self.STATE)

    def nav(self):
        try:
            ui = json.load(open(self.UI))
            if ui.get('nav'):
                return float(ui['nav'])
        except Exception:
            pass
        st = self._load()   # fallback: cash + sweep + positions at entry
        raw = st.get('positions') or []
        plist = raw.values() if isinstance(raw, dict) else raw
        pos = sum(float(p.get('qty', 0)) * float(p.get('entry', p.get('entry_price', 0)) or 0)
                  for p in plist)
        return float(st.get('cash', 0)) + float((st.get('sweep') or {}).get('cost', 0)) + pos

    def liquid(self, st):
        return float(st.get('cash', 0)) + float((st.get('sweep') or {}).get('cost', 0))

    def get_div(self):
        st = self._load()
        return st.get('dividend'), st

    def init_div(self, st):
        # HWM seeds at adoption: max(current NAV, contributed capital). The
        # Open Alpha 'capital' field is the 2020 rupee-rebased BACKFILL seed
        # (not contributed capital), so NAV-at-adoption is the anchor —
        # backfilled gains are capital, never distributable.
        return dict(hwm=max(self.nav(), float(st.get('capital', 0.0))), cap=None,
                    reserve=0.0, ledger=[], last_flow_ts=str(datetime.now()),
                    adopted=str(date.today()))

    def save_div(self, div):
        st = self._load()
        st['dividend'] = div
        self._save(st)

    def net_flows_since(self, st, ts):
        net = 0.0
        for f in st.get('fund_flows') or []:
            if str(f.get('ts', '')) > ts:
                net += f['amount'] if f['kind'] == 'deposit' else -f['amount']
        return net

    def apply(self, st, div, outflow, note):
        """Remove `outflow` from cash+sweep, persist dividend block. Called under lock."""
        take_cash = min(float(st.get('cash', 0)), outflow)
        st['cash'] = float(st.get('cash', 0)) - take_cash
        rem = outflow - take_cash
        if rem > 1e-6:
            sw = st['sweep']
            frac = rem / sw['cost'] if sw.get('cost') else 0
            sw['units'] = round(sw['units'] * (1 - frac), 3)
            sw['cost'] = round(sw['cost'] - rem, 2)
        st.setdefault('fund_flows', []).append(dict(
            ts=str(datetime.now()), kind='dividend', amount=outflow,
            via='dividend engine', positions_touched=False, note=note))
        st['dividend'] = div
        self._save(st)

    def declare_ctx(self):
        """Yields (div_state_or_None, raw_state); caller must run inside with_lock."""
        return self.get_div()

    def with_lock(self, fn):
        if not self._lock():
            raise RuntimeError('open-alpha state is locked (nightly run?) — try later')
        try:
            return fn()
        finally:
            self._unlock()


class TrueNorthBook:
    name = 'True North'
    slug = 'truenorth'
    DB = ROOT / 'backtest_data' / 'momentum_paper.db'

    def _conn(self):
        c = sqlite3.connect(str(self.DB)); c.row_factory = sqlite3.Row
        return c

    def _get(self, key, default=None):
        c = self._conn()
        r = c.execute('SELECT val FROM mp_state WHERE key=?', (key,)).fetchone()
        c.close()
        return json.loads(r['val']) if r else default

    def _set(self, key, val, conn=None):
        c = conn or self._conn()
        c.execute('INSERT OR REPLACE INTO mp_state(key,val) VALUES(?,?)',
                  (key, json.dumps(val)))
        if conn is None:
            c.commit(); c.close()

    def nav(self):
        c = self._conn()
        r = c.execute('SELECT nav FROM mp_nav ORDER BY d DESC LIMIT 1').fetchone()
        c.close()
        return float(r['nav']) if r else float(self._get('cash', 0.0))

    def liquid(self, st=None):
        return float(self._get('cash', 0.0))

    def get_div(self):
        return self._get('dividend'), None

    def init_div(self, st=None):
        return dict(hwm=max(self.nav(), float(self._get('capital', 0.0))), cap=None,
                    reserve=0.0, ledger=[], last_flow_ts=str(datetime.now()),
                    adopted=str(date.today()))

    def save_div(self, div):
        self._set('dividend', div)

    def net_flows_since(self, st, ts):
        net = 0.0
        for f in self._get('fund_flows', []) or []:
            if str(f.get('ts', '')) > ts:
                net += f['amount'] if f['kind'] == 'deposit' else -f['amount']
        return net

    def apply(self, st, div, outflow, note):
        c = self._conn()
        cash = float(self._get('cash', 0.0))
        self._set('cash', cash - outflow, conn=c)
        flows = self._get('fund_flows', []) or []
        flows.append(dict(ts=str(datetime.now()), kind='dividend', amount=outflow,
                          via='dividend engine', positions_touched=False, note=note))
        self._set('fund_flows', flows, conn=c)
        self._set('dividend', div, conn=c)
        c.commit(); c.close()

    def with_lock(self, fn):
        return fn()   # sqlite transactionality suffices for this single writer


BOOKS = {'openalpha': OpenAlphaBook, 'truenorth': TrueNorthBook}


# ───────────────────────── quarter helpers ─────────────────────────
def latest_quarter_end(today: date) -> date:
    qm = ((today.month - 1) // 3) * 3          # 0,3,6,9
    if qm == 0:
        return date(today.year - 1, 12, 31)
    return date(today.year, qm, [31, 30, 30][qm // 3 - 1])


def qtag(qe: date) -> str:
    return f'{qe.year}-Q{(qe.month - 1) // 3 + 1}'


# ───────────────────────── declaration ─────────────────────────
def declare(book_slug, today=None, dry_run=False, notify=True):
    """Run one declaration if due. Returns a result dict (always safe to call)."""
    today = today or date.today()
    qe = latest_quarter_end(today)
    tag = qtag(qe)
    if (today - qe).days > DECLARE_WINDOW_DAYS:
        return dict(book=book_slug, skipped=f'outside declare window for {tag}')
    book = BOOKS[book_slug]()

    def _run():
        div, st = book.get_div()
        if div is None:
            div = book.init_div(st)
        if any(r['quarter'] == tag for r in div['ledger']):
            return dict(book=book_slug, skipped=f'{tag} already declared')

        nav = book.nav()
        flows = book.net_flows_since(st, div.get('last_flow_ts', '1970-01-01'))
        hwm = div['hwm'] + flows               # deposits are not profit
        reserve = div['reserve'] * (1 + POLICY['reserve_rate_pa'] / 4)

        new_profit = max(0.0, nav - hwm)
        entitlement = POLICY['baseline'] * new_profit
        target = entitlement if div['cap'] is None else div['cap'] * (1 + POLICY['cap_growth_q'])

        liq = book.liquid(st)
        clipped = entitlement > liq
        if clipped:                            # never force-sell; excess stays in book
            entitlement = max(0.0, liq)

        from_profit = min(entitlement, target)
        from_reserve = min(reserve, max(0.0, target - from_profit))
        paid = from_profit + from_reserve
        surplus = entitlement - from_profit
        reserve += surplus - from_reserve
        if new_profit > 0 and entitlement > 0:
            hwm = nav - entitlement
        elif new_profit > 0:
            hwm = nav                          # profit but zero entitlement (fully clipped)
        cap = paid if paid > 0 else div['cap']

        row = dict(quarter=tag, ts=str(datetime.now()), nav=round(nav),
                   flow_adjusted_hwm_before=round(div['hwm'] + flows),
                   new_profit=round(new_profit), entitlement=round(entitlement),
                   paid=round(paid), from_reserve=round(from_reserve),
                   to_reserve=round(max(0.0, surplus - from_reserve)),
                   reserve_after=round(reserve), hwm_after=round(hwm),
                   source=('profit' if from_profit >= from_reserve else 'reserve'),
                   liquidity_clipped=clipped, dry_run=dry_run)
        if dry_run:
            return dict(book=book_slug, declaration=row)

        div.update(hwm=hwm, cap=cap, reserve=reserve,
                   last_flow_ts=str(datetime.now()))
        div['ledger'].append(row)
        book.apply(st, div, entitlement, f'{tag} declaration: paid {paid:,.0f}, '
                                         f'to reserve {max(0.0, surplus - from_reserve):,.0f}')
        if notify and paid > 0:
            try:
                from services.dividend_notify import notify_declaration
                notify_declaration(dict(
                    book=f'{book.name} (paper)', quarter=tag,
                    record_date=qe.strftime('%d-%b-%Y'),
                    payment_date=today.strftime('%d-%b-%Y'),
                    nav=round(nav), new_profit=round(new_profit),
                    dividend=round(paid), source=row['source'],
                    reserve=round(reserve), hwm=round(hwm),
                    console_amount=round(paid)))
            except Exception as e:
                row['notify_error'] = str(e)
        return dict(book=book_slug, declaration=row)

    return book.with_lock(_run)


def status(book_slug):
    book = BOOKS[book_slug]()
    div, _ = book.get_div()
    if div is None:
        return dict(book=book_slug, initialized=False,
                    note='initializes at first declaration (HWM = contributed capital)')
    return dict(book=book_slug, initialized=True, hwm=round(div['hwm']),
                cap=round(div['cap']) if div.get('cap') else None,
                reserve=round(div['reserve']), ledger=div['ledger'][-8:])


def init(book_slug):
    """Persist the dividend block now (adoption day) so the HWM anchors today."""
    book = BOOKS[book_slug]()

    def _run():
        div, st = book.get_div()
        if div is not None:
            return dict(book=book_slug, skipped='already initialized', hwm=round(div['hwm']))
        div = book.init_div(st)
        book.save_div(div)
        return dict(book=book_slug, initialized=True, hwm=round(div['hwm']))

    return book.with_lock(_run)


if __name__ == '__main__':
    import sys
    if '--init' in sys.argv:
        for slug in ('truenorth', 'openalpha'):
            print(json.dumps(init(slug), default=str))
    else:
        dry = '--dry' in sys.argv
        for slug in ('truenorth', 'openalpha'):
            print(json.dumps(declare(slug, dry_run=dry), indent=1, default=str))
