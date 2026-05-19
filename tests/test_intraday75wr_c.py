"""Smoke test — Config C (Multi-Bar SHORT Bounce, TP 1.5/SL 1.0).

Verifies:
  1. Config C engine defaults to PAPER MODE.
  2. With a synthetic 5-min pattern containing 4 consecutive bearish bars
     with lower highs, RSI <= 55, close < own VWAP, and NIFTY < its own
     VWAP — the multi_bar_bounce signal FIRES, recording a position
     with system_id='C', TP 1.5% below entry, SL 1.0% above entry.
  3. In paper mode no real Kite order is placed.
  4. Removing the 'NIFTY < own VWAP' precondition causes the signal to
     NOT fire (broad-market filter is binding).

Run: python -m pytest tests/test_intraday75wr_c.py -v
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, date, time, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    fake_data_dir = tmp_path / 'data'
    fake_data_dir.mkdir(parents=True)
    monkeypatch.setattr(
        'services.intraday_75wr.db.DB_PATH',
        str(fake_data_dir / 'intraday_75wr_test.db'),
    )
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


def _build_multi_bar_bounce_df(today: date) -> pd.DataFrame:
    """Synthetic: yesterday flat at 110, then today's session opens at 110
    and falls steadily so that bars 5-8 are 4 consecutive bearish bars with
    lower highs. Latest bar (bar 8) is at the lower price, sustained
    downtrend means RSI < 55, close < session VWAP."""
    # Yesterday — 30 bars at 110
    yesterday = today - timedelta(days=1)
    pre_times = [
        datetime.combine(yesterday, time(9, 15)) + timedelta(minutes=5 * i)
        for i in range(30)
    ]
    pre = pd.DataFrame({
        'open': [110.0] * 30, 'high': [110.5] * 30,
        'low': [109.5] * 30, 'close': [110.0] * 30,
        'volume': [50_000] * 30,
    }, index=pd.DatetimeIndex(pre_times))

    # Today — 9 bars (bar 0..8), with bars 5..8 each strictly bearish + lower highs
    bars = 9
    times = [
        datetime.combine(today, time(9, 15)) + timedelta(minutes=5 * i)
        for i in range(bars)
    ]
    # Bars 0..4 — drift down a touch but bullish bodies (let VWAP settle)
    open0 = [109.5, 109.4, 109.3, 109.2, 109.1]
    close0 = [109.4, 109.3, 109.2, 109.1, 109.0]
    high0 = [109.7, 109.6, 109.5, 109.4, 109.3]
    low0 = [109.3, 109.2, 109.1, 109.0, 108.9]
    # Bars 5..8 — 4 consecutive bearish bars, each with lower highs
    open1 = [109.0, 108.6, 108.2, 107.8]
    close1 = [108.6, 108.2, 107.8, 107.4]      # close < open => bearish
    high1 = [109.1, 108.7, 108.3, 107.9]       # strictly decreasing
    low1 = [108.4, 108.0, 107.6, 107.2]
    df_today = pd.DataFrame({
        'open': open0 + open1,
        'high': high0 + high1,
        'low': low0 + low1,
        'close': close0 + close1,
        'volume': [50_000] * bars,
    }, index=pd.DatetimeIndex(times))
    return pd.concat([pre, df_today])


def _nifty_below_own_vwap_ctx() -> dict:
    return {
        'today': str(date.today()),
        'current_close': 24_900.0,
        'current_vwap': 25_000.0,        # close < vwap, signal allowed
    }


def _nifty_above_own_vwap_ctx() -> dict:
    return {
        'today': str(date.today()),
        'current_close': 25_050.0,
        'current_vwap': 25_000.0,        # close > vwap, signal BLOCKED
    }


def test_config_c_defaults_to_paper_mode(isolated_db, reset_engines):
    from services.intraday_75wr import get_engine
    eng = get_engine('C')
    assert eng.is_enabled() is True
    assert eng.is_paper() is True
    assert eng.is_live() is False
    assert eng.cfg['tp_pct'] == 1.5
    assert eng.cfg['sl_pct'] == 1.0


def test_config_c_signal_fires_on_multi_bar_pattern(isolated_db, reset_engines):
    """4 consecutive bearish bars + RSI<=55 + close<VWAP + NIFTY<VWAP."""
    from services.intraday_75wr import get_engine
    from services.intraday_75wr.db import get_i75_db
    eng = get_engine('C')

    mock_kite = MagicMock()
    mock_kite.place_order.side_effect = AssertionError(
        'Kite must NOT be called in paper mode'
    )
    today = date.today()
    df = _build_multi_bar_bounce_df(today)
    eng.load_cohort = staticmethod(lambda path: ['ZEEL'])

    with patch.object(eng, '_get_kite', return_value=mock_kite), \
         patch(
             'services.intraday_75wr.nifty_regime.compute_regime',
             return_value=_nifty_below_own_vwap_ctx(),
         ):
        results = eng.scan(fetcher=lambda sym: df)

    assert len(results) == 1, f'expected 1 entry, got {len(results)}: {results}'
    r = results[0]
    assert r['paper'] is True
    assert str(r['order_id']).startswith('PAPER-')
    mock_kite.place_order.assert_not_called()

    # Verify persisted position has correct system_id and TP/SL distances
    db = get_i75_db()
    open_pos = db.get_open_positions()
    assert len(open_pos) == 1
    pos = open_pos[0]
    assert pos['system_id'] == 'C', f'expected C, got {pos["system_id"]}'
    entry = pos['entry_price']
    target = pos['target_price']
    sl = pos['sl_price']
    tp_pct_actual = (entry - target) / entry * 100  # SHORT: TP < entry
    sl_pct_actual = (sl - entry) / entry * 100      # SHORT: SL > entry
    assert abs(tp_pct_actual - 1.5) < 0.05, (
        f'Config C TP must be 1.5%, got {tp_pct_actual:.3f}%'
    )
    assert abs(sl_pct_actual - 1.0) < 0.05, (
        f'Config C SL must be 1.0%, got {sl_pct_actual:.3f}%'
    )


def test_config_c_blocked_when_nifty_above_own_vwap(isolated_db, reset_engines):
    """If broad-market filter fails (NIFTY close >= own VWAP), no entry."""
    from services.intraday_75wr import get_engine
    eng = get_engine('C')

    mock_kite = MagicMock()
    today = date.today()
    df = _build_multi_bar_bounce_df(today)
    eng.load_cohort = staticmethod(lambda path: ['ZEEL'])

    with patch.object(eng, '_get_kite', return_value=mock_kite), \
         patch(
             'services.intraday_75wr.nifty_regime.compute_regime',
             return_value=_nifty_above_own_vwap_ctx(),
         ):
        results = eng.scan(fetcher=lambda sym: df)
    assert len(results) == 0, (
        'NIFTY > own VWAP must block Config C entry — got results: '
        f'{results}'
    )
    mock_kite.place_order.assert_not_called()
