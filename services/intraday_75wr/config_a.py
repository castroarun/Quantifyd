"""Config A engine — 3-System Original (TP 0.5% / SL 1.5%).

Composes A1 (Diamond Short, 09:45 scan), A2 (Long-MR, 11:15-13:15 continuous),
A3 (Long-TC, 09:15-10:30 continuous). Subclass of IntradayEngineBase that
wires each sub-signal to a scan_*() method called from cron.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date

import pandas as pd

from services.intraday_75wr.engine_base import IntradayEngineBase
from services.intraday_75wr import nifty_regime
from services.intraday_75wr.signal_lib import (
    diamond_short, long_mr, long_tc,
)

logger = logging.getLogger(__name__)


class ConfigAEngine(IntradayEngineBase):
    """Config A — Diamond Short + Long-MR + Long-TC, all on TP 0.5/SL 1.5.

    Sub-signal IDs persisted to DB.system_id:
        'A1' Diamond Short, 'A2' Long-MR, 'A3' Long-TC.
    """

    @property
    def cohort_short(self) -> list[str]:
        return self.load_cohort(self.cfg['cohort_short_path'])

    @property
    def cohort_long_mr(self) -> list[str]:
        return self.load_cohort(self.cfg['cohort_long_mr_path'])

    @property
    def cohort_long_tc(self) -> list[str]:
        return self.load_cohort(self.cfg['cohort_long_tc_path'])

    # =========================================================================
    # Scan A1 — Diamond Short, single 09:45 IST scan
    # =========================================================================

    def scan_a1(self, fetcher=None) -> list[dict]:
        if not self.cfg.get('a1_enabled', True):
            return []
        return self._scan_generic(
            system_id=self._sys('1'),
            cohort=self.cohort_short,
            evaluator=diamond_short.evaluate,
            fetcher=fetcher,
        )

    # =========================================================================
    # Scan A2 — Long-MR continuous 11:15-13:15
    # =========================================================================

    def scan_a2(self, fetcher=None) -> list[dict]:
        if not self.cfg.get('a2_enabled', True):
            return []
        return self._scan_generic(
            system_id=self._sys('2'),
            cohort=self.cohort_long_mr,
            evaluator=long_mr.evaluate,
            fetcher=fetcher,
        )

    # =========================================================================
    # Scan A3 — Long-TC continuous 09:15-10:30
    # =========================================================================

    def scan_a3(self, fetcher=None) -> list[dict]:
        if not self.cfg.get('a3_enabled', True):
            return []
        return self._scan_generic(
            system_id=self._sys('3'),
            cohort=self.cohort_long_tc,
            evaluator=long_tc.evaluate,
            fetcher=fetcher,
        )

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _sys(self, suffix: str) -> str:
        """Build sub-system id like 'A1', 'A2', 'A3' or 'B1', 'B2', 'B3'."""
        return f'{self.config_id}{suffix}'

    def _scan_generic(self, *, system_id, cohort, evaluator, fetcher) -> list[dict]:
        """Scan the cohort with the given signal evaluator and place orders
        for every fired signal, gated by can_enter_now().

        `fetcher` is a callable (symbol) -> 5-min DataFrame. Defaults to
        the live data_manager / Kite fetcher when None. Tests inject a
        mocked fetcher.
        """
        if not self.is_enabled():
            return []

        results: list[dict] = []
        today = pd.Timestamp(date.today())

        # Compute NIFTY regime once per scan tick
        try:
            nifty_ctx = nifty_regime.compute_regime(today=today.date())
        except Exception as e:
            logger.error(f'[I75-{self.config_id}] nifty regime err: {e}')
            nifty_ctx = {}

        # Default fetcher uses bridge/data_manager
        if fetcher is None:
            fetcher = self._default_fetcher

        for sym in cohort:
            try:
                # Pre-flight gate
                allowed, reason = self.can_enter_now(system_id, sym)
                if not allowed:
                    self._log_skip(system_id, sym, reason)
                    continue

                df = fetcher(sym)
                if df is None or len(df) == 0:
                    continue

                sig = evaluator(
                    df, instrument=sym, today=today,
                    cfg=self.cfg, nifty_ctx=nifty_ctx,
                )
                if not sig:
                    continue

                # Compute qty
                qty_info = self.compute_qty(sig['entry_price'], sig['sl_price'])
                if qty_info['qty'] <= 0:
                    self._log_skip(
                        system_id, sym,
                        f'qty=0 ({qty_info.get("reason", "")})'
                    )
                    continue

                # Persist signal log
                try:
                    self.db.log_signal(
                        system_id=system_id, instrument=sym,
                        signal_time=datetime.now().isoformat(),
                        direction=sig['direction'],
                        entry_price=sig['entry_price'],
                        sl_price=sig['sl_price'],
                        target_price=sig['target_price'],
                        rsi=sig['meta'].get('rsi'),
                        vwap=sig['meta'].get('vwap'),
                        gap_pct=sig['meta'].get('gap_pct'),
                        nifty_filter_value=str(nifty_ctx.get('b3_change_pct')),
                        nifty_filter_pass=1,
                        signal_meta=json.dumps(sig['meta']),
                        action_taken='ENTERED',
                        paper_mode=1 if self.is_paper() else 0,
                    )
                except Exception as e:
                    logger.warning(f'[I75-{self.config_id}] log_signal err: {e}')

                # Place order (paper or live)
                out = self.place_order_live_or_paper(
                    system_id=system_id,
                    instrument=sym,
                    direction=sig['direction'],
                    qty=qty_info['qty'],
                    entry_price=sig['entry_price'],
                    sl_price=sig['sl_price'],
                    target_price=sig['target_price'],
                    signal_meta=sig['meta'],
                )
                if out:
                    results.append({**out, 'instrument': sym, 'system_id': system_id})
                    logger.info(
                        f'[I75-{system_id}] ENTRY {sig["direction"]} {sym} '
                        f'qty={qty_info["qty"]} entry={sig["entry_price"]} '
                        f'SL={sig["sl_price"]} TGT={sig["target_price"]} '
                        f'paper={self.is_paper()}'
                    )
            except Exception as e:
                logger.error(
                    f'[I75-{self.config_id}/{sym}] scan err: {e}', exc_info=True,
                )
        return results

    def _log_skip(self, system_id: str, sym: str, reason: str):
        try:
            self.db.log_signal(
                system_id=system_id, instrument=sym,
                signal_time=datetime.now().isoformat(),
                direction='-',
                entry_price=None, sl_price=None, target_price=None,
                action_taken='BLOCKED',
                block_reason=reason,
                paper_mode=1 if self.is_paper() else 0,
            )
        except Exception:
            pass

    def _default_fetcher(self, symbol: str) -> pd.DataFrame:
        """Live-mode default: pull 5-min OHLCV for symbol from market_data.db.
        Includes today's bars plus enough history for RSI/VWAP seeding."""
        from datetime import timedelta
        try:
            import sqlite3
            from config import MARKET_DATA_DB
            end = datetime.now()
            start = end - timedelta(days=5)
            conn = sqlite3.connect(str(MARKET_DATA_DB))
            try:
                df = pd.read_sql_query(
                    'SELECT date, open, high, low, close, volume '
                    'FROM market_data_unified '
                    'WHERE symbol=? AND timeframe=? AND date BETWEEN ? AND ? '
                    'ORDER BY date',
                    conn,
                    params=(symbol, '5minute', start.isoformat(), end.isoformat()),
                )
            finally:
                conn.close()
            if df.empty:
                return df
            df['date'] = pd.to_datetime(df['date'])
            return df.set_index('date')
        except Exception as e:
            logger.warning(
                f'[I75-{self.config_id}] fetch err for {symbol}: {e}',
            )
            return pd.DataFrame()
