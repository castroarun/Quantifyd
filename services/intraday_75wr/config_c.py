"""Config C engine — Multi-Bar SHORT Bounce (TP 1.5% / SL 1.0%).

Single sub-system 'C', continuous scan during session. Cohort = same 25
short-diamonds as A1/B1.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date, timedelta

import pandas as pd

from services.intraday_75wr.engine_base import IntradayEngineBase
from services.intraday_75wr import nifty_regime
from services.intraday_75wr.signal_lib import multi_bar_bounce

logger = logging.getLogger(__name__)


class ConfigCEngine(IntradayEngineBase):
    """Multi-bar SHORT bounce scanner. system_id stored in DB = 'C'."""

    @property
    def cohort(self) -> list[str]:
        return self.load_cohort(self.cfg['cohort_short_path'])

    def scan(self, fetcher=None) -> list[dict]:
        """Continuous scan tick. Called every 5 min between 09:30-15:00."""
        if not self.is_enabled():
            return []

        results: list[dict] = []
        today = pd.Timestamp(date.today())
        system_id = self.config_id  # 'C'

        try:
            nifty_ctx = nifty_regime.compute_regime(today=today.date())
        except Exception as e:
            logger.error(f'[I75-{self.config_id}] nifty regime err: {e}')
            nifty_ctx = {}

        if fetcher is None:
            fetcher = self._default_fetcher

        for sym in self.cohort:
            try:
                allowed, reason = self.can_enter_now(system_id, sym)
                if not allowed:
                    self._log_skip(system_id, sym, reason)
                    continue

                df = fetcher(sym)
                if df is None or len(df) == 0:
                    continue

                sig = multi_bar_bounce.evaluate(
                    df, instrument=sym, today=today,
                    cfg=self.cfg, nifty_ctx=nifty_ctx,
                )
                if not sig:
                    continue

                qty_info = self.compute_qty(sig['entry_price'], sig['sl_price'])
                if qty_info['qty'] <= 0:
                    self._log_skip(
                        system_id, sym,
                        f'qty=0 ({qty_info.get("reason", "")})',
                    )
                    continue

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
                        signal_meta=json.dumps(sig['meta']),
                        action_taken='ENTERED',
                        nifty_filter_pass=1,
                        paper_mode=1 if self.is_paper() else 0,
                    )
                except Exception as e:
                    logger.warning(f'[I75-{self.config_id}] log_signal err: {e}')

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
                    f'[I75-{self.config_id}/{sym}] scan err: {e}',
                    exc_info=True,
                )
        return results

    def _log_skip(self, system_id, sym, reason):
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
