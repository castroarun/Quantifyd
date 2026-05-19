"""Smoke test — Config A (3-System Original, TP 0.5/SL 1.5).

Verifies:
  1. Engine defaults to PAPER MODE (paper_trading_mode=True,
     live_trading_enabled=False)
  2. With a known synthetic 5-min bar pattern matching the A1 Diamond Short
     signal, scan_a1() FIRES the signal and writes a position to DB.
  3. In paper mode, NO real Kite order is placed — the order_id is a
     PAPER-* string, and the wrapped Kite client (mocked) is never called.
  4. Flipping to live mode (paper=False AND live_trading_enabled=True),
     a Kite mock IS invoked.

Run: python -m pytest tests/test_intraday75wr_a.py -v
"""

from __future__ import annotations

import os
import sys
import sqlite3
import tempfile
from datetime import datetime, date, time, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root on path so 'config', 'services' import cleanly
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Force the I75 DB into a temp dir so tests don't touch real data."""
    fake_data_dir = tmp_path / 'data'
    fake_data_dir.mkdir(parents=True)
    monkeypatch.setattr(
        'services.intraday_75wr.db.DB_PATH',
        str(fake_data_dir / 'intraday_75wr_test.db'),
    )
    # Force singleton re-init
    import services.intraday_75wr.db as _dbmod
    _dbmod._instance = None
    yield
    _dbmod._instance = None


@pytest.fixture
def reset_engines():
    from services.intraday_75wr import reset_engines_for_test
    reset_engines_for_test()
    yield
    reset_engines_for_test()


def _build_diamond_short_df(today: date) -> pd.DataFrame:
    """Synthetic 5-min OHLCV that yields a Diamond-Short signal at bar 6.

    Stock RSI(14) sustained < 40, close < VWAP. Use a steady downtrend that
    starts above and falls below the VWAP."""
    bars = 7
    times = [
        datetime.combine(today, time(9, 15)) + timedelta(minutes=5 * i)
        for i in range(bars)
    ]
    # Falling close: 100 -> 99.4
    close = np.linspace(100.0, 99.4, bars)
    open_ = close + 0.05
    high = open_ + 0.10
    low = close - 0.10
    volume = [10_000] * bars
    df = pd.DataFrame({
        'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': volume,
    }, index=pd.DatetimeIndex(times))
    # Pre-roll 30 bars yesterday so RSI seeds below 40
    yesterday = today - timedelta(days=1)
    pre_times = [
        datetime.combine(yesterday, time(9, 15)) + timedelta(minutes=5 * i)
        for i in range(30)
    ]
    pre_close = np.linspace(110.0, 100.5, 30)
    pre = pd.DataFrame({
        'open': pre_close + 0.1, 'high': pre_close + 0.2,
        'low': pre_close - 0.2, 'close': pre_close,
        'volume': [10_000] * 30,
    }, index=pd.DatetimeIndex(pre_times))
    return pd.concat([pre, df])


def _bearish_nifty_ctx() -> dict:
    """NIFTY context that satisfies a1_nifty_filter='b3_change_neg'."""
    return {
        'today': str(date.today()),
        'gap_pct': -0.2,
        'day_open': 25_000.0,
        'prev_close': 25_050.0,
        'b3_close': 24_950.0,
        'b3_vwap': 25_010.0,
        'b3_rsi': 42.0,
        'b3_change_pct': -0.2,
        'below_vwap_b3': True,
        'b6_close': 24_940.0,
        'b6_vwap': 25_005.0,
        'below_vwap_b6': True,
        'first_bearish': True,
        'first_bullish': False,
        'current_close': 24_940.0,
        'current_vwap': 25_005.0,
    }


def test_config_a_defaults_to_paper_mode(isolated_db, reset_engines):
    """Critical: Config A must default to PAPER, with live blocked."""
    from services.intraday_75wr import get_engine
    eng = get_engine('A')
    assert eng.is_enabled() is True, 'Config A should be enabled'
    assert eng.is_paper() is True, 'Config A must default to paper mode'
    assert eng.is_live() is False, (
        'Config A must NOT be live until BOTH flags are flipped'
    )


def test_config_a_a1_signal_fires_in_paper_no_kite_call(isolated_db, reset_engines):
    """A1 scan with a known signal: enters position, NO real Kite order."""
    from services.intraday_75wr import get_engine
    eng = get_engine('A')

    # Fake Kite — will FAIL the test if called in paper mode
    mock_kite = MagicMock()
    mock_kite.place_order.side_effect = AssertionError(
        'Kite.place_order MUST NOT be called in paper mode'
    )
    today = date.today()

    with patch.object(eng, '_get_kite', return_value=mock_kite), \
         patch(
             'services.intraday_75wr.nifty_regime.compute_regime',
             return_value=_bearish_nifty_ctx(),
         ):
        # Override cohort to 1 stock so we don't hit a thousand fakes
        eng.cfg = {**eng.cfg, 'cohort_short_path': eng.cfg['cohort_short_path']}
        # Inject a fetcher that returns the synthetic df for a known stock
        df = _build_diamond_short_df(today)
        fetcher = lambda sym: df if sym == 'ZEEL' else pd.DataFrame()

        # Monkey-patch cohort load to return only ZEEL
        eng.load_cohort = staticmethod(lambda path: ['ZEEL'])

        results = eng.scan_a1(fetcher=fetcher)

    # Signal should have fired and a paper position recorded
    assert len(results) == 1, f'expected 1 entry, got {len(results)}: {results}'
    r = results[0]
    assert r['paper'] is True, 'paper flag must be True in paper mode'
    assert str(r['order_id']).startswith('PAPER-'), (
        f'order_id should be PAPER-*, got {r["order_id"]}'
    )
    # Kite was never called
    mock_kite.place_order.assert_not_called()


def test_config_a_only_real_kite_when_both_flags_flipped(isolated_db, reset_engines):
    """Live mode requires BOTH paper=False AND live=True. Belt-and-suspenders."""
    from services.intraday_75wr import get_engine
    eng = get_engine('A')

    today = date.today()
    df = _build_diamond_short_df(today)
    eng.load_cohort = staticmethod(lambda path: ['ZEEL'])

    mock_kite = MagicMock()
    mock_kite.place_order.return_value = 'KITE-ORDER-12345'

    # Case 1: paper=False but live=False -> still blocked (treat as paper)
    eng.cfg = {**eng.cfg,
               'paper_trading_mode': False,
               'live_trading_enabled': False,
               'enabled': True}
    assert eng.is_live() is False, 'one-flag-only must NOT authorise live'
    with patch.object(eng, '_get_kite', return_value=mock_kite), \
         patch(
             'services.intraday_75wr.nifty_regime.compute_regime',
             return_value=_bearish_nifty_ctx(),
         ):
        results_blocked = eng.scan_a1(fetcher=lambda sym: df)
    # In this state, place_order_live_or_paper sees is_live()=False so
    # it should still take the paper path
    assert len(results_blocked) == 1
    assert results_blocked[0]['paper'] is True
    mock_kite.place_order.assert_not_called()

    # Reset DB state for next case (clear positions)
    from services.intraday_75wr.db import get_i75_db
    db = get_i75_db()
    with db.db_lock:
        conn = db._get_conn()
        try:
            conn.execute('DELETE FROM i75_positions')
            conn.commit()
        finally:
            conn.close()

    # Case 2: BOTH flags flipped -> live, Kite invoked
    eng.cfg = {**eng.cfg,
               'paper_trading_mode': False,
               'live_trading_enabled': True,
               'enabled': True}
    assert eng.is_live() is True, 'both flags should authorise live'
    with patch.object(eng, '_get_kite', return_value=mock_kite), \
         patch(
             'services.intraday_75wr.nifty_regime.compute_regime',
             return_value=_bearish_nifty_ctx(),
         ):
        results_live = eng.scan_a1(fetcher=lambda sym: df)
    assert len(results_live) == 1
    assert results_live[0]['paper'] is False, 'live mode means paper=False'
    mock_kite.place_order.assert_called_once()


def test_config_a_concurrency_cap_blocks_extra_entries(isolated_db, reset_engines):
    """Combined cap = 5 across A+B+C. Block when >=5 already open."""
    from services.intraday_75wr import get_engine
    from services.intraday_75wr.db import get_i75_db
    eng = get_engine('A')
    db = get_i75_db()

    today = date.today().isoformat()
    # Pre-fill 5 dummy open positions
    for i in range(5):
        db.add_position(
            system_id='A1', instrument=f'DUMMY{i}',
            trade_date=today, direction='SHORT', qty=1,
            entry_price=100.0, entry_time=datetime.now().isoformat(),
            sl_price=101.5, target_price=99.5, status='OPEN',
            kite_entry_order_id=f'PAPER-{i}',
            paper_mode=1,
        )

    conc = eng.concurrency_check()
    assert conc['open_count'] >= 5
    assert conc['allowed'] is False
