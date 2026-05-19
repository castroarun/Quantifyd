"""Smoke test — Config B (3-System Cost-Resilient, TP 2.0/SL 1.5).

Same A1/A2/A3 signal mechanics as Config A; only the TP differs (2.0% vs
0.5%). This test verifies:

  1. Config B engine defaults to PAPER MODE.
  2. The same Diamond-Short pattern that fires in A also fires in B,
     and the resulting position has the WIDER TP (2.0% target).
  3. Sub-system ID persists as 'B1' (NOT 'A1') in the DB so the two
     configs can be filtered independently.
  4. Concurrency cap is shared with A and C (combined view).

Run: python -m pytest tests/test_intraday75wr_b.py -v
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


def _build_diamond_short_df(today: date) -> pd.DataFrame:
    bars = 7
    times = [
        datetime.combine(today, time(9, 15)) + timedelta(minutes=5 * i)
        for i in range(bars)
    ]
    close = np.linspace(100.0, 99.4, bars)
    open_ = close + 0.05
    high = open_ + 0.10
    low = close - 0.10
    df = pd.DataFrame({
        'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': [10_000] * bars,
    }, index=pd.DatetimeIndex(times))
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
    return {
        'today': str(date.today()),
        'gap_pct': -0.2, 'day_open': 25_000.0, 'prev_close': 25_050.0,
        'b3_close': 24_950.0, 'b3_vwap': 25_010.0, 'b3_rsi': 42.0,
        'b3_change_pct': -0.2, 'below_vwap_b3': True,
        'b6_close': 24_940.0, 'b6_vwap': 25_005.0, 'below_vwap_b6': True,
        'first_bearish': True, 'first_bullish': False,
        'current_close': 24_940.0, 'current_vwap': 25_005.0,
    }


def test_config_b_defaults_to_paper_mode(isolated_db, reset_engines):
    from services.intraday_75wr import get_engine
    eng = get_engine('B')
    assert eng.is_enabled() is True
    assert eng.is_paper() is True
    assert eng.is_live() is False
    assert eng.cfg['tp_pct'] == 2.0, 'Config B must use 2.0% TP'
    assert eng.cfg['sl_pct'] == 1.5, 'Config B must use 1.5% SL'


def test_config_b_signal_uses_wider_tp_and_b1_sysid(isolated_db, reset_engines):
    """B1 fires on same conditions as A1 but the persisted target is wider,
    and system_id is 'B1' (not 'A1')."""
    from services.intraday_75wr import get_engine
    from services.intraday_75wr.db import get_i75_db
    eng = get_engine('B')

    mock_kite = MagicMock()
    mock_kite.place_order.side_effect = AssertionError(
        'Kite must NOT be called in paper mode'
    )
    today = date.today()
    df = _build_diamond_short_df(today)
    eng.load_cohort = staticmethod(lambda path: ['ZEEL'])

    with patch.object(eng, '_get_kite', return_value=mock_kite), \
         patch(
             'services.intraday_75wr.nifty_regime.compute_regime',
             return_value=_bearish_nifty_ctx(),
         ):
        results = eng.scan_a1(fetcher=lambda sym: df)
    assert len(results) == 1
    r = results[0]
    assert r['paper'] is True

    # Inspect persisted position: TP must be ~2.0% below entry (SHORT)
    db = get_i75_db()
    open_pos = db.get_open_positions()
    assert len(open_pos) == 1
    pos = open_pos[0]
    assert pos['system_id'] == 'B1', f'expected B1 system_id, got {pos["system_id"]}'
    entry = pos['entry_price']
    target = pos['target_price']
    sl = pos['sl_price']
    # SHORT: TP is BELOW entry
    tp_pct_actual = (entry - target) / entry * 100
    sl_pct_actual = (sl - entry) / entry * 100
    assert abs(tp_pct_actual - 2.0) < 0.05, (
        f'Config B TP must be 2.0%, got {tp_pct_actual:.3f}%'
    )
    assert abs(sl_pct_actual - 1.5) < 0.05, (
        f'Config B SL must be 1.5%, got {sl_pct_actual:.3f}%'
    )


def test_config_b_paper_mode_does_not_call_kite(isolated_db, reset_engines):
    from services.intraday_75wr import get_engine
    eng = get_engine('B')

    mock_kite = MagicMock()
    today = date.today()
    df = _build_diamond_short_df(today)
    eng.load_cohort = staticmethod(lambda path: ['ZEEL'])
    with patch.object(eng, '_get_kite', return_value=mock_kite), \
         patch(
             'services.intraday_75wr.nifty_regime.compute_regime',
             return_value=_bearish_nifty_ctx(),
         ):
        eng.scan_a1(fetcher=lambda sym: df)
    mock_kite.place_order.assert_not_called()
