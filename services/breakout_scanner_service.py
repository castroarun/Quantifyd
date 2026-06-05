"""
Breakout Scanner Service
========================
Live intraday breakout scanner across selectable universes (F&O / Nifty500 /
Smallcap / All-liquid), two rules (compression-breakout, Donchian-20+volume),
with user-configurable filters. Reuses the F&O scanner_service helpers for
market-hours + weekly-CPR, the central daily DB for context, and kite.quote
for live LTP / cumulative volume.

A "breakout" (LONG) = daily uptrend + price breaks the reference high + volume
surge (+ optional narrow-CPR / consolidation / RSI). The 15-min poll job emails
NEW breakouts (deduped per symbol+rule+day) when email alerts are enabled.

Settings + alert-dedup persist in backtest_data/breakout_scanner.db so they
survive restarts. No live-order code — read-only discovery/alert tool.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from services.data_manager import FNO_LOT_SIZES, get_data_manager
from services.scanner_service import _is_market_open, _weekly_cpr_width_pct

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / 'backtest_data' / 'market_data.db'
STATE_DB = Path(__file__).parent.parent / 'backtest_data' / 'breakout_scanner.db'

UNIVERSE_KEYS = ['fno', 'nifty500', 'smallcap', 'all_liquid']

DEFAULT_SETTINGS = {
    'universes': ['fno'],                 # multi-select
    'rules': ['compression', 'donchian'],  # which rules to flag
    'trend': 'sma50',                     # sma50 | sma200 | both | off
    'cpr_narrow_pct': 0.6,                # compression: today's weekly-CPR width% <=
    'contract_pct': 0.12,                 # compression: 10-day range / price <=
    'donchian_lb': 20,                    # donchian breakout lookback (days)
    'vol_mult': 1.5,                      # volume-surge multiple (intraday-projected)
    'rsi_min': 0,                         # 0 = off; else require RSI14 >=
    'min_price': 30.0,
    'min_turn_cr': 3.0,                   # 20d median turnover (Rs cr)
    'email_enabled': False,
    'email_to': 'arun.castromin@gmail.com',
}


# --------------------------------------------------------------------------
# State DB (settings + alert dedup)
# --------------------------------------------------------------------------
def _init_state_db():
    con = sqlite3.connect(str(STATE_DB))
    con.execute("CREATE TABLE IF NOT EXISTS settings (k TEXT PRIMARY KEY, v TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS alerts "
                "(day TEXT, symbol TEXT, rule TEXT, ts TEXT, "
                "PRIMARY KEY (day, symbol, rule))")
    con.commit()
    con.close()


def load_settings() -> dict:
    _init_state_db()
    con = sqlite3.connect(str(STATE_DB))
    row = con.execute("SELECT v FROM settings WHERE k='main'").fetchone()
    con.close()
    s = dict(DEFAULT_SETTINGS)
    if row:
        try:
            s.update(json.loads(row[0]))
        except Exception:
            pass
    return s


def save_settings(patch: dict) -> dict:
    s = load_settings()
    for k, v in (patch or {}).items():
        if k in DEFAULT_SETTINGS:
            s[k] = v
    con = sqlite3.connect(str(STATE_DB))
    con.execute("INSERT INTO settings (k, v) VALUES ('main', ?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (json.dumps(s),))
    con.commit()
    con.close()
    return s


def _already_alerted(day: str, symbol: str, rule: str) -> bool:
    con = sqlite3.connect(str(STATE_DB))
    r = con.execute("SELECT 1 FROM alerts WHERE day=? AND symbol=? AND rule=?",
                    (day, symbol, rule)).fetchone()
    con.close()
    return r is not None


def _mark_alerted(day: str, symbol: str, rule: str):
    con = sqlite3.connect(str(STATE_DB))
    con.execute("INSERT OR IGNORE INTO alerts (day, symbol, rule, ts) VALUES (?,?,?,?)",
                (day, symbol, rule, datetime.now().isoformat()))
    con.commit()
    con.close()


# --------------------------------------------------------------------------
# Universe builders
# --------------------------------------------------------------------------
def _read_csv_symbols(path: Path, col_candidates=('symbol', 'Symbol', 'SYMBOL')) -> List[str]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
        for c in col_candidates:
            if c in df.columns:
                return [str(x).strip().upper() for x in df[c].dropna().tolist()]
        return [str(x).strip().upper() for x in df.iloc[:, 0].dropna().tolist()]
    except Exception:
        return []


class BreakoutScannerService:
    def __init__(self):
        self._dm = get_data_manager()
        self._lock = threading.Lock()
        self._daily_cache_date: Optional[str] = None
        self._daily_ctx: Dict[str, dict] = {}
        self._tokmap = None  # NSE tradingsymbol -> instrument_token (cached once)
        self._building = False  # background daily-cache build in progress
        self._uni_cache: Dict[str, List[str]] = {}
        self._db_symbols: Optional[set] = None

    # ---- universes ----
    def _all_daily_symbols(self) -> set:
        if self._db_symbols is None:
            con = sqlite3.connect(str(DB_PATH))
            self._db_symbols = set(
                r[0] for r in con.execute(
                    "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='day'"))
            con.close()
        return self._db_symbols

    def universe(self, key: str) -> List[str]:
        if key in self._uni_cache:
            return self._uni_cache[key]
        have = self._all_daily_symbols()
        syms: List[str] = []
        if key == 'fno':
            syms = sorted(set(FNO_LOT_SIZES.keys()) & have)
        elif key == 'nifty500':
            lst = _read_csv_symbols(DB_PATH.parent.parent / 'data' / 'nifty500_list.csv')
            syms = sorted(set(lst) & have) or sorted(have)
        elif key == 'smallcap':
            tc = DB_PATH.parent.parent / 'research' / '34_nifty500_expansion' / 'results' / 'top_by_turnover.csv'
            sel = set()
            if tc.exists():
                try:
                    for _, r in pd.read_csv(tc).iterrows():
                        if float(r.get('avg_turnover_cr', 0)) < 50.0:
                            sel.add(str(r['symbol']).strip().upper())
                except Exception:
                    pass
            syms = sorted(sel & have)
        elif key == 'all_liquid':
            syms = sorted(have)
        self._uni_cache[key] = syms
        return syms

    def _selected_symbols(self, universes: List[str]) -> List[str]:
        out: set = set()
        for k in universes:
            if k in UNIVERSE_KEYS:
                out.update(self.universe(k))
        return sorted(out)

    # ---- daily context (cached per day) ----
    def _get_token(self, sym: str):
        """NSE tradingsymbol -> instrument_token via a once-cached instruments dump."""
        if self._tokmap is None:
            try:
                from services.kite_service import get_kite
                kite = get_kite()
                insts = kite.instruments('NSE') if kite else []
                self._tokmap = {i['tradingsymbol']: i['instrument_token'] for i in insts}
                logger.info('[BreakoutScanner] cached %d NSE instrument tokens', len(self._tokmap))
            except Exception as e:
                logger.warning('[BreakoutScanner] instruments() failed: %s', e)
                self._tokmap = {}
        return self._tokmap.get(sym)

    def _fetch_daily_live(self, sym: str, from_date: datetime, to_date: datetime):
        """Daily bars LIVE from Kite, in-memory (NOT persisted to DB), so the scanner
        reflects current prices and never depends on a stale local DB."""
        import time as _t
        try:
            from services.kite_service import get_kite
            tok = self._get_token(sym)
            kite = get_kite()
            if not tok or not kite:
                return None
            data = kite.historical_data(tok, from_date, to_date, 'day')
            _t.sleep(0.34)  # Kite 3 req/sec
            if not data:
                return None
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            df = df.sort_values('date').drop_duplicates(subset=['date']).set_index('date')
            return df[['open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            logger.warning('[BreakoutScanner] live Kite fetch failed %s: %s', sym, e)
            return None

    def _daily_context(self, sym: str, today: datetime, lb_donchian: int) -> Optional[dict]:
        try:
            df = self._fetch_daily_live(sym, today - timedelta(days=420), today)
        except Exception:
            return None
        if df is None or len(df) < 60:
            return None
        df = df.sort_index()
        c = df['close']
        last = float(c.iloc[-1])
        sma50 = float(c.tail(50).mean()) if len(c) >= 50 else None
        sma200 = float(c.tail(200).mean()) if len(c) >= 200 else None
        vol = df['volume']
        avg20 = float(vol.iloc[-21:-1].mean()) if len(vol) >= 21 else float(vol.mean())
        last_bar_vol = float(vol.iloc[-1])
        prev_day_high = float(df['high'].iloc[-2]) if len(df) >= 2 else None
        # donchian high = max high over prior lb bars EXCLUDING today's bar
        donch = float(df['high'].iloc[-(lb_donchian + 1):-1].max()) if len(df) > lb_donchian else None
        # 10-day contraction range %
        rng10 = float(df['high'].tail(10).max() - df['low'].tail(10).min())
        range10_pct = (rng10 / last) if last else None
        # NR7
        rng = (df['high'] - df['low'])
        nr7 = bool(rng.iloc[-1] <= rng.tail(7).min()) if len(rng) >= 7 else False
        cpr_w = _weekly_cpr_width_pct(df, today)
        # RSI14 (Wilder)
        d = c.diff()
        up = d.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        dn = (-d).clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rsi = None
        if len(c) >= 15 and dn.iloc[-1] and not np.isnan(dn.iloc[-1]) and dn.iloc[-1] > 0:
            rsi = float(100 - 100 / (1 + up.iloc[-1] / dn.iloc[-1]))
        # 20d median turnover (cr)
        turn_cr = float((c * vol).tail(20).median() / 1e7) if len(c) >= 20 else None
        return {
            'last_close': last, 'sma50': sma50, 'sma200': sma200, 'avg20_vol': avg20,
            'last_bar_vol': last_bar_vol,
            'prev_day_high': prev_day_high, 'donchian_high': donch,
            'range10_pct': range10_pct, 'nr7': nr7, 'cpr_width_pct': cpr_w,
            'rsi': rsi, 'turn_cr': turn_cr,
        }

    def _ensure_daily_cache(self, symbols: List[str], today: datetime, lb_donchian: int):
        import threading
        key = today.strftime('%Y-%m-%d') + f':{lb_donchian}:' + str(hash(tuple(symbols)) & 0xffff)
        with self._lock:
            if self._daily_cache_date == key and self._daily_ctx:
                return
            if self._building:
                return  # build in progress — serve current cache; next poll picks up fresh data
            self._building = True
        def _build():
            try:
                ctx = {}
                for s in symbols:
                    f = self._daily_context(s, today, lb_donchian)
                    if f:
                        ctx[s] = f
                with self._lock:
                    self._daily_ctx = ctx
                    self._daily_cache_date = key
                logger.info("[BreakoutScanner] daily ctx built LIVE %s — %d/%d", key, len(ctx), len(symbols))
            finally:
                with self._lock:
                    self._building = False
        threading.Thread(target=_build, daemon=True, name='bo-ctx-build').start()

    # ---- live quotes (batched) ----
    def _live_quotes(self, symbols: List[str]) -> Dict[str, dict]:
        try:
            from services.kite_service import get_kite
            kite = get_kite()
        except Exception as e:
            logger.debug("[BreakoutScanner] no kite: %s", e)
            return {}
        out: Dict[str, dict] = {}
        for i in range(0, len(symbols), 200):
            chunk = symbols[i:i + 200]
            try:
                q = kite.quote(['NSE:' + s for s in chunk]) or {}
                for s in chunk:
                    d = q.get('NSE:' + s)
                    if d:
                        out[s] = {'ltp': d.get('last_price'),
                                  'volume': d.get('volume'),
                                  'ohlc': d.get('ohlc') or {}}
            except Exception as e:
                logger.debug("[BreakoutScanner] quote chunk failed: %s", e)
        return out

    @staticmethod
    def _elapsed_frac(now: datetime) -> float:
        start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        mins = (now - start).total_seconds() / 60.0
        return max(0.05, min(1.0, mins / 375.0))

    # ---- the scan ----
    def scan(self, settings: Optional[dict] = None) -> dict:
        s = dict(DEFAULT_SETTINGS)
        s.update(settings or load_settings())
        now = datetime.now()
        market_open = _is_market_open(now)
        symbols = self._selected_symbols(s['universes']) or self.universe('fno')
        self._ensure_daily_cache(symbols, now, int(s['donchian_lb']))
        quotes = self._live_quotes(symbols) if market_open else {}
        frac = self._elapsed_frac(now)
        rules = set(s.get('rules') or ['compression', 'donchian'])
        matches = []
        for sym in symbols:
            f = self._daily_ctx.get(sym)
            if not f:
                continue
            if f['last_close'] < s['min_price']:
                continue
            if f.get('turn_cr') is not None and f['turn_cr'] < s['min_turn_cr']:
                continue
            q = quotes.get(sym, {})
            ltp = q.get('ltp')
            if ltp is None:
                ltp = f['last_close']          # off-hours / no quote -> last close
            prev_close = (q.get('ohlc') or {}).get('close') or f['last_close']
            chg = round((ltp - prev_close) / prev_close * 100, 2) if prev_close else None

            # trend gate
            sma50, sma200 = f['sma50'], f['sma200']
            trend_ok = True
            mode = s.get('trend', 'sma50')
            if mode == 'sma50':
                trend_ok = sma50 is not None and ltp > sma50
            elif mode == 'sma200':
                trend_ok = sma200 is not None and ltp > sma200
            elif mode == 'both':
                trend_ok = sma50 and sma200 and ltp > sma50 and sma50 >= sma200
            if not trend_ok:
                continue

            # volume surge — intraday projected (cum vol vs avg-pace), else daily
            tv = q.get('volume')
            if tv is not None and f['avg20_vol']:
                vol_surge = round(tv / (f['avg20_vol'] * frac), 2)   # intraday-projected pace
            elif f['avg20_vol']:
                vol_surge = round(f['last_bar_vol'] / f['avg20_vol'], 2)  # off-hours: last bar vs avg20
            else:
                vol_surge = None
            vol_ok = (vol_surge is not None and vol_surge >= s['vol_mult'])

            # RSI gate (optional)
            rsi_min = float(s.get('rsi_min') or 0)
            rsi_ok = (rsi_min <= 0) or (f.get('rsi') is not None and f['rsi'] >= rsi_min)

            for rule in rules:
                if rule == 'compression':
                    pdh = f['prev_day_high']
                    broke = pdh is not None and ltp > pdh
                    coil = (f.get('range10_pct') is not None and f['range10_pct'] < s['contract_pct']) \
                        and (f.get('cpr_width_pct') is None or f['cpr_width_pct'] <= s['cpr_narrow_pct'] or f['nr7'])
                    if broke and coil and vol_ok and rsi_ok:
                        matches.append(self._row(sym, ltp, chg, 'compression', pdh, f, vol_surge, now, market_open))
                elif rule == 'donchian':
                    dh = f['donchian_high']
                    broke = dh is not None and ltp > dh
                    if broke and vol_ok and rsi_ok:
                        matches.append(self._row(sym, ltp, chg, 'donchian', dh, f, vol_surge, now, market_open))

        matches.sort(key=lambda r: (r['vol_surge'] or 0), reverse=True)
        return {
            'generated_at': now.isoformat(),
            'market_open': market_open,
            'universes': s['universes'],
            'rules': sorted(rules),
            'scanned': len(symbols),
            'count': len(matches),
            'settings': s,
            'matches': matches,
            'note': ('Live LTP during market hours; off-hours uses last daily close '
                     '(breakouts off-hours are indicative, vol-surge ~stale).'),
        }

    @staticmethod
    def _row(sym, ltp, chg, rule, level, f, vol_surge, now, market_open):
        return {
            'symbol': sym, 'ltp': round(ltp, 2), 'day_change_pct': chg,
            'rule': rule, 'breakout_level': round(level, 2) if level else None,
            'trend_sma50': round(f['sma50'], 2) if f['sma50'] else None,
            'cpr_width_pct': round(f['cpr_width_pct'], 3) if f['cpr_width_pct'] is not None else None,
            'range10_pct': round(f['range10_pct'] * 100, 2) if f['range10_pct'] is not None else None,
            'nr7': f['nr7'], 'vol_surge': vol_surge,
            'rsi': round(f['rsi'], 1) if f.get('rsi') is not None else None,
            'turn_cr': round(f['turn_cr'], 1) if f.get('turn_cr') is not None else None,
            'as_of': now.isoformat() if market_open else None,
        }

    # ---- poll + email (called by scheduler every 15 min) ----
    def poll_and_alert(self) -> dict:
        if not _is_market_open():
            return {'skipped': 'market_closed'}
        s = load_settings()
        res = self.scan(s)
        if not s.get('email_enabled'):
            return {'scanned': res['scanned'], 'matches': res['count'], 'emailed': 0}
        day = datetime.now().strftime('%Y-%m-%d')
        fresh = [m for m in res['matches'] if not _already_alerted(day, m['symbol'], m['rule'])]
        if not fresh:
            return {'scanned': res['scanned'], 'matches': res['count'], 'emailed': 0}
        try:
            self._email(fresh, s)
            for m in fresh:
                _mark_alerted(day, m['symbol'], m['rule'])
            return {'scanned': res['scanned'], 'matches': res['count'], 'emailed': len(fresh)}
        except Exception as e:
            logger.warning("[BreakoutScanner] email failed: %s", e)
            return {'scanned': res['scanned'], 'matches': res['count'], 'emailed': 0, 'error': str(e)}

    @staticmethod
    def _email(matches: List[dict], s: dict):
        from services.notifications import get_notification_service
        ns = get_notification_service()
        lines = [f"{m['symbol']:<14} {m['rule']:<11} LTP {m['ltp']}  "
                 f"({'+' if (m['day_change_pct'] or 0) >= 0 else ''}{m['day_change_pct']}%)  "
                 f"brk>{m['breakout_level']}  vol×{m['vol_surge']}  RSI {m['rsi']}"
                 for m in matches]
        body = ("New breakouts detected by the Quantifyd Breakout Scanner:\n\n"
                + "\n".join(lines)
                + f"\n\nUniverses: {', '.join(s['universes'])} | Rules: {', '.join(s['rules'])}"
                + f" | vol×≥{s['vol_mult']}\n"
                + "View: http://94.136.185.54:5000/app/breakout-scanner\n")
        ns.send_alert('breakout', f"📈 {len(matches)} new breakout(s)", body,
                      data={'matches': matches}, priority='normal')


_instance: Optional[BreakoutScannerService] = None
_instance_lock = threading.Lock()


def get_breakout_scanner_service() -> BreakoutScannerService:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = BreakoutScannerService()
    return _instance
