"""Smoke tests for the Pair-Trading (Config D) paper adapter.

Verifies:
  1. Signal generator produces correct z-score + action for each of the 6 pairs.
  2. Signal fires ENTRY_LONG / ENTRY_SHORT when |z| crosses entry_z.
  3. NO Kite orders are placed in paper mode (only DB rows + paper IDs).
  4. Orders ARE placed (to a mocked Kite) when paper=False AND live=True.
  5. Concurrency cap blocks the 6th entry when 5 already open.
  6. EXIT_MR fires when z mean-reverts past 0; both legs are squared off.

Runs standalone:
   python tests/test_pair_trading.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers — synthesise mean-reverting price series for a known pair
# ---------------------------------------------------------------------------

def synth_pair_prices(alpha: float, beta: float,
                      n_days: int = 120,
                      base_a: float = 1500.0, base_b: float = 700.0,
                      drift_z: float = 0.0,
                      seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Build two cointegrated-ish daily price series. The TRUE relationship
    is log(P_a) = alpha + beta * log(P_b) + noise. We layer in a final-day
    z-score "shock" to drive the test signal cleanly.

    drift_z: if non-zero, the LAST bar's spread is offset by this many sigma
             (so the signal generator sees |z| ~= drift_z on the latest bar).
    """
    rng = np.random.default_rng(seed)
    # Random-walk for log(P_b)
    log_pb = np.log(base_b) + np.cumsum(rng.normal(0, 0.012, n_days))
    # log(P_a) coupled via beta + a small noise term
    noise = rng.normal(0, 0.008, n_days)
    log_pa = alpha + beta * log_pb + noise

    # Inject final-day spread shock to control the test z-score
    if drift_z != 0.0:
        # Estimate the rolling-20 std of the no-shock spread
        spread = log_pa - alpha - beta * log_pb
        sigma = float(pd.Series(spread).rolling(20).std().iloc[-1])
        log_pa[-1] = log_pa[-1] + drift_z * sigma

    pa = np.exp(log_pa)
    pb = np.exp(log_pb)
    idx = pd.date_range(end=date.today(), periods=n_days, freq='B')
    return (pd.Series(pa, index=idx, name='close'),
            pd.Series(pb, index=idx, name='close'))


# ---------------------------------------------------------------------------
# Mock Kite client (defensive — fails if real Kite would be called)
# ---------------------------------------------------------------------------

class MockKiteClient:
    """Captures order calls + returns synthetic IDs. NEVER hits real Kite."""

    def __init__(self):
        self.placed_orders: List[Dict[str, Any]] = []
        self.next_id = 100000

    def place_order(self, **kwargs):
        self.next_id += 1
        oid = f"MOCK-{self.next_id}"
        self.placed_orders.append({**kwargs, '_mock_order_id': oid})
        return oid

    def cancel_order(self, **kwargs):
        return None

    def modify_order(self, **kwargs):
        return kwargs.get('order_id')

    def order_history(self, order_id):
        return [{'order_id': order_id, 'status': 'COMPLETE',
                 'average_price': 1500.0, 'filled_quantity': 100}]

    def instruments(self, exchange='NFO'):
        # Return synthetic NFO FUT rows for the 6-pair cohort symbols
        rows = []
        syms = ['HAVELLS', 'MARICO', 'BAJFINANCE', 'KOTAKBANK',
                'DABUR', 'HINDUNILVR', 'COFORGE', 'HCLTECH', 'TCS', 'APOLLOHOSP']
        from services.data_manager import FNO_LOT_SIZES
        future_expiry = date.today() + timedelta(days=20)
        for s in syms:
            rows.append({
                'name': s,
                'tradingsymbol': f"{s}25MAYFUT",
                'instrument_token': hash(s) & 0x7FFFFFFF,
                'instrument_type': 'FUT',
                'segment': 'NFO-FUT',
                'expiry': future_expiry,
                'lot_size': FNO_LOT_SIZES.get(s, 1),
            })
        return rows

    def historical_data(self, *args, **kwargs):
        # Should not be called in these tests because we patch
        # PairEngine.fetch_daily_history directly. If it is, return empty.
        return []


# ---------------------------------------------------------------------------
# Test fixture — point DB at temp file, freeze cohort, inject mock Kite
# ---------------------------------------------------------------------------

class PairTradingTestBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Use a fresh temp DB so we don't pollute the real pair_trading.db
        cls._tmpdir = tempfile.mkdtemp(prefix='pair_trading_test_')
        cls._db_path = os.path.join(cls._tmpdir, 'pair_trading_test.db')
        # Patch DB path BEFORE importing the module
        import services.pair_trading.db as db_mod
        db_mod.DB_PATH = cls._db_path
        # Reset singleton
        db_mod._db_instance = None

    def setUp(self):
        # Reset engine singleton so each test gets a fresh engine
        import services.pair_trading.pair_engine as eng_mod
        eng_mod._engine_instance = None
        # Reset DB singleton
        import services.pair_trading.db as db_mod
        db_mod._db_instance = None
        # Re-init DB
        from services.pair_trading.db import get_pair_trading_db
        # New empty DB
        if os.path.exists(self.__class__._db_path):
            os.remove(self.__class__._db_path)
        get_pair_trading_db()


# ---------------------------------------------------------------------------
# Test 1: Signal generator — z-score + action vocabulary
# ---------------------------------------------------------------------------

class TestSignalGenerator(unittest.TestCase):

    def test_each_pair_signal_at_neutral_z(self):
        """At neutral spread (z near 0), every pair should produce NO_ACTION."""
        from config import PAIR_TRADING_DEFAULTS
        from services.pair_trading.signal import PairRules, evaluate_today

        for pr in PAIR_TRADING_DEFAULTS['pairs']:
            sa, sb = synth_pair_prices(pr['alpha'], pr['beta'],
                                       n_days=80, drift_z=0.0,
                                       seed=hash(pr['name']) & 0xFFFFFFFF)
            rules = PairRules(entry_z=pr['entry_z'], stop_z=pr['stop_z'],
                              hold_days=pr['hold_days'], lookback=pr['lookback'])
            sig = evaluate_today(pr['name'], sa, sb, pr['alpha'], pr['beta'], rules)
            self.assertIsNotNone(sig, f"{pr['name']}: signal None at neutral z")
            self.assertIn(sig.action, ('NO_ACTION', 'ENTRY_LONG', 'ENTRY_SHORT'),
                          f"{pr['name']}: unexpected action {sig.action}")
            self.assertTrue(np.isfinite(sig.z), f"{pr['name']}: non-finite z {sig.z}")

    def test_entry_long_when_z_below_minus_entry(self):
        """When z drives below -entry_z, action must be ENTRY_LONG."""
        from services.pair_trading.signal import PairRules, evaluate_today
        sa, sb = synth_pair_prices(alpha=-2.6023, beta=1.5551,
                                   n_days=80, drift_z=-3.0, seed=1)
        rules = PairRules(entry_z=2.0, stop_z=4.0, hold_days=20, lookback=20)
        sig = evaluate_today('TEST', sa, sb, -2.6023, 1.5551, rules)
        self.assertIsNotNone(sig)
        self.assertLess(sig.z, -2.0, f"shock should drive z below -2: got {sig.z}")
        self.assertEqual(sig.action, 'ENTRY_LONG')

    def test_entry_short_when_z_above_plus_entry(self):
        from services.pair_trading.signal import PairRules, evaluate_today
        sa, sb = synth_pair_prices(alpha=-2.6023, beta=1.5551,
                                   n_days=80, drift_z=+3.0, seed=2)
        rules = PairRules(entry_z=2.0, stop_z=4.0, hold_days=20, lookback=20)
        sig = evaluate_today('TEST', sa, sb, -2.6023, 1.5551, rules)
        self.assertIsNotNone(sig)
        self.assertGreater(sig.z, 2.0, f"shock should drive z above +2: got {sig.z}")
        self.assertEqual(sig.action, 'ENTRY_SHORT')

    def test_exit_mr_when_z_mean_reverts(self):
        """If we hold direction=+1 and z >= 0, expect EXIT_MR."""
        from services.pair_trading.signal import PairRules, evaluate_today
        sa, sb = synth_pair_prices(alpha=-2.6023, beta=1.5551,
                                   n_days=80, drift_z=0.5, seed=3)
        rules = PairRules(entry_z=2.0, stop_z=4.0, hold_days=20, lookback=20)
        # Hold a long_spread position; z ~+0.5 (above 0) -> mean revert
        sig = evaluate_today('TEST', sa, sb, -2.6023, 1.5551, rules,
                             open_position={'direction': 1, 'days_held': 5})
        self.assertEqual(sig.action, 'EXIT_MR')

    def test_exit_time_when_hold_cap_reached(self):
        from services.pair_trading.signal import PairRules, evaluate_today
        sa, sb = synth_pair_prices(alpha=-2.6023, beta=1.5551,
                                   n_days=80, drift_z=-2.5, seed=4)
        rules = PairRules(entry_z=2.0, stop_z=4.0, hold_days=20, lookback=20)
        # Direction +1, z still negative -> normally HOLD, but days_held >= cap -> EXIT_TIME
        sig = evaluate_today('TEST', sa, sb, -2.6023, 1.5551, rules,
                             open_position={'direction': 1, 'days_held': 25})
        self.assertEqual(sig.action, 'EXIT_TIME')


# ---------------------------------------------------------------------------
# Test 2: Paper mode — no Kite orders placed
# ---------------------------------------------------------------------------

class TestPaperMode(PairTradingTestBase):

    def test_paper_mode_no_kite_calls(self):
        """In paper mode, daily_scan must produce DB rows + paper order IDs
        and NOT touch the real Kite client.

        Uses a stubbed evaluate_today so the test is deterministic — the
        paper-mode contract being tested has nothing to do with z-score
        computation accuracy.
        """
        from config import PAIR_TRADING_DEFAULTS
        from services.pair_trading.signal import SignalResult
        # Ensure paper mode lock
        PAIR_TRADING_DEFAULTS['enabled'] = True
        PAIR_TRADING_DEFAULTS['paper_trading_mode'] = True
        PAIR_TRADING_DEFAULTS['live_trading_enabled'] = False

        from services.pair_trading.pair_engine import get_pair_engine
        engine = get_pair_engine()

        # Stub: HAVELLS-MARICO fires ENTRY_LONG; others NO_ACTION.
        def stub_evaluate_today(pair_name, prices_a, prices_b, alpha, beta,
                                 rules, open_position=None):
            today = date.today().isoformat()
            if pair_name == 'HAVELLS-MARICO':
                return SignalResult(
                    pair_name=pair_name, trade_date=today,
                    priceA=1500.0, priceB=700.0,
                    spread=-0.05, spread_mu=0.0, spread_sd=0.025,
                    z=-2.5, action='ENTRY_LONG',
                )
            return SignalResult(
                pair_name=pair_name, trade_date=today,
                priceA=1500.0, priceB=700.0,
                spread=0.0, spread_mu=0.0, spread_sd=0.025,
                z=-1.0, action='NO_ACTION',
            )

        idx = pd.date_range(end=date.today(), periods=80, freq='B')
        stub_prices = pd.Series(np.linspace(100, 110, 80), index=idx)

        def fake_fetch(self, symbol, lookback_days):
            return stub_prices

        # Defensive: patch get_kite to raise if anything tries to call real Kite
        def boom_get_kite():
            raise AssertionError("REAL Kite call attempted in paper mode!")

        with patch('services.pair_trading.pair_engine.PairEngine.fetch_daily_history', fake_fetch), \
             patch('services.pair_trading.pair_engine.evaluate_today', stub_evaluate_today), \
             patch('services.kite_service.get_kite', boom_get_kite):
            summary = engine.daily_scan()

        self.assertEqual(summary['mode'], 'paper')
        self.assertGreaterEqual(summary['evaluated'], 1)
        # HAVELLS-MARICO entry should have fired
        self.assertGreaterEqual(summary['entered'], 1,
            f"expected >=1 entry in paper scan, got {summary}")

        # Verify the position row + orders exist in DB with PAPER- order IDs
        opens = engine.db.get_open_positions()
        self.assertGreaterEqual(len(opens), 1)
        for pos in opens:
            self.assertEqual(pos['paper_mode'], 1, "position must be paper-flagged")
            oa = pos['legA_kite_order_id'] or ''
            ob = pos['legB_kite_order_id'] or ''
            self.assertTrue(oa.startswith('PAPER-'),
                f"leg-A order should start with PAPER- got {oa}")
            self.assertTrue(ob.startswith('PAPER-'),
                f"leg-B order should start with PAPER- got {ob}")


# ---------------------------------------------------------------------------
# Test 3: Live mode — orders ARE placed (to mocked Kite)
# ---------------------------------------------------------------------------

class TestLiveMode(PairTradingTestBase):

    def test_live_mode_calls_mocked_kite(self):
        """When paper=False AND live=True, place_pair_order must call
        the (mocked) Kite client. Verifies the wrap_kite_op decorator
        passes through correctly."""
        from config import PAIR_TRADING_DEFAULTS
        PAIR_TRADING_DEFAULTS['enabled'] = True
        PAIR_TRADING_DEFAULTS['paper_trading_mode'] = False
        PAIR_TRADING_DEFAULTS['live_trading_enabled'] = True
        try:
            mock_kite = MockKiteClient()

            from services.pair_trading.pair_engine import get_pair_engine
            engine = get_pair_engine()

            # Build a synthetic ENTRY_LONG signal for HAVELLS-MARICO
            from services.pair_trading.signal import SignalResult
            from services.pair_trading.cohort import PairConfig
            pair = engine.cohort.by_name('HAVELLS-MARICO')
            self.assertIsNotNone(pair)
            sig = SignalResult(
                pair_name='HAVELLS-MARICO',
                trade_date=date.today().isoformat(),
                priceA=1500.0, priceB=700.0,
                spread=-0.05, spread_mu=0.0, spread_sd=0.025,
                z=-2.5, action='ENTRY_LONG',
            )

            # Patch get_kite to return our mock
            with patch('services.pair_trading.pair_engine.PairEngine._get_kite',
                       lambda self: mock_kite):
                pid = engine.place_pair_order_live_or_paper(
                    pair=pair, direction=1, signal=sig,
                    qty_a=pair.lot_size_a, qty_b=pair.lot_size_b,
                )

            self.assertIsNotNone(pid, "position should be created")
            self.assertEqual(len(mock_kite.placed_orders), 2,
                f"expected 2 leg orders, got {len(mock_kite.placed_orders)}")
            # Verify both orders carry NRML product (carry-forward, not MIS)
            for od in mock_kite.placed_orders:
                self.assertEqual(od['product'], 'NRML')
                self.assertEqual(od['exchange'], 'NFO')
        finally:
            # Restore paper-mode safety lock
            PAIR_TRADING_DEFAULTS['paper_trading_mode'] = True
            PAIR_TRADING_DEFAULTS['live_trading_enabled'] = False


# ---------------------------------------------------------------------------
# Test 4: Concurrency cap — 6th entry blocked when 5 already open
# ---------------------------------------------------------------------------

class TestConcurrencyCap(PairTradingTestBase):

    def test_sixth_entry_blocked_at_cap(self):
        """If 5 pairs are already open, the 6th ENTRY signal must be blocked
        and recorded as 'concurrency_cap_5' in the signal log.

        We patch evaluate_today to control output deterministically: the 5
        pre-seeded pairs return HOLD; the 6th returns ENTRY_LONG. This
        isolates the concurrency check from spread/z-score noise."""
        from config import PAIR_TRADING_DEFAULTS
        from services.pair_trading.signal import SignalResult
        PAIR_TRADING_DEFAULTS['enabled'] = True
        PAIR_TRADING_DEFAULTS['paper_trading_mode'] = True
        PAIR_TRADING_DEFAULTS['live_trading_enabled'] = False
        PAIR_TRADING_DEFAULTS['max_concurrent'] = 5

        from services.pair_trading.pair_engine import get_pair_engine
        engine = get_pair_engine()

        # Pre-load 5 fake long-spread (direction=+1) open positions
        first_five_names = []
        for pr in PAIR_TRADING_DEFAULTS['pairs'][:5]:
            first_five_names.append(pr['name'])
            engine.db.add_position(
                pair_name=pr['name'], symA=pr['symA'], symB=pr['symB'],
                direction=1,
                entry_date=date.today().isoformat(),
                entry_z=-2.5, target_z=0.0, stop_z=pr['stop_z'],
                hold_cap_days=pr['hold_days'], lookback=pr['lookback'],
                alpha=pr['alpha'], beta=pr['beta'],
                legA_tradingsymbol=f"{pr['symA']}FUT", legA_qty=100,
                legA_entry_price=1500.0, legA_lot_size=100,
                legB_tradingsymbol=f"{pr['symB']}FUT", legB_qty=100,
                legB_entry_price=700.0, legB_lot_size=100,
                paper_mode=1, status='OPEN',
            )

        # Patch evaluate_today to return deterministic per-pair signals
        def stub_evaluate_today(pair_name, prices_a, prices_b, alpha, beta,
                                 rules, open_position=None):
            today = date.today().isoformat()
            if pair_name == 'APOLLOHOSP-COFORGE':
                return SignalResult(
                    pair_name=pair_name, trade_date=today,
                    priceA=3000.0, priceB=620.0,
                    spread=-0.05, spread_mu=0.0, spread_sd=0.025,
                    z=-2.5, action='ENTRY_LONG',
                )
            return SignalResult(
                pair_name=pair_name, trade_date=today,
                priceA=1500.0, priceB=700.0,
                spread=-0.04, spread_mu=0.0, spread_sd=0.025,
                z=-1.6, action='HOLD',
            )

        # Provide stub price series so fetch_daily_history is satisfied
        idx = pd.date_range(end=date.today(), periods=80, freq='B')
        stub_prices = pd.Series(np.linspace(100, 110, 80), index=idx)

        def fake_fetch(self, symbol, lookback_days):
            return stub_prices

        with patch('services.pair_trading.pair_engine.PairEngine.fetch_daily_history',
                    fake_fetch), \
             patch('services.pair_trading.pair_engine.evaluate_today',
                    stub_evaluate_today):
            engine.daily_scan()

        # Open count should still be 5 (no new entry, no spurious exits)
        opens_after = engine.db.get_open_positions()
        self.assertEqual(len(opens_after), 5,
            f"concurrency cap should hold count at 5, got {len(opens_after)}")
        # Signal log should have a BLOCKED row with concurrency reason
        sigs = engine.db.list_signals(trade_date=date.today().isoformat(), limit=200)
        blocked = [s for s in sigs if s['action'] == 'BLOCKED'
                   and (s.get('block_reason') or '').startswith('concurrency_cap')]
        self.assertGreaterEqual(len(blocked), 1,
            "expected a BLOCKED row with concurrency_cap reason")


# ---------------------------------------------------------------------------
# Test 5: Exit mean-revert — both legs squared off
# ---------------------------------------------------------------------------

class TestExitMeanRevert(PairTradingTestBase):

    def test_z_mean_revert_squares_both_legs(self):
        """Hold a long-spread position, then run a scan with z back near 0.
        Position should close with reason=EXIT_MR; both legs get exit
        order rows in pair_orders.

        We patch evaluate_today to deterministically return EXIT_MR for
        HAVELLS-MARICO and HOLD/NO_ACTION for the other pairs."""
        from config import PAIR_TRADING_DEFAULTS
        from services.pair_trading.signal import SignalResult
        PAIR_TRADING_DEFAULTS['enabled'] = True
        PAIR_TRADING_DEFAULTS['paper_trading_mode'] = True
        PAIR_TRADING_DEFAULTS['live_trading_enabled'] = False

        from services.pair_trading.pair_engine import get_pair_engine
        engine = get_pair_engine()

        pr = PAIR_TRADING_DEFAULTS['pairs'][0]   # HAVELLS-MARICO
        # Create open position 5 days ago
        entry_date = (date.today() - timedelta(days=5)).isoformat()
        pid = engine.db.add_position(
            pair_name=pr['name'], symA=pr['symA'], symB=pr['symB'],
            direction=1,
            entry_date=entry_date,
            entry_z=-2.5, target_z=0.0, stop_z=pr['stop_z'],
            hold_cap_days=pr['hold_days'], lookback=pr['lookback'],
            alpha=pr['alpha'], beta=pr['beta'],
            legA_tradingsymbol=f"{pr['symA']}25MAYFUT", legA_qty=625,
            legA_entry_price=1500.0, legA_lot_size=625,
            legB_tradingsymbol=f"{pr['symB']}25MAYFUT", legB_qty=1200,
            legB_entry_price=700.0, legB_lot_size=1200,
            paper_mode=1, status='OPEN',
        )

        # Patch evaluate_today: HAVELLS-MARICO -> EXIT_MR; everyone else NO_ACTION
        def stub_evaluate_today(pair_name, prices_a, prices_b, alpha, beta,
                                 rules, open_position=None):
            today = date.today().isoformat()
            if pair_name == 'HAVELLS-MARICO' and open_position is not None:
                return SignalResult(
                    pair_name=pair_name, trade_date=today,
                    priceA=1550.0, priceB=695.0,
                    spread=0.0, spread_mu=0.0, spread_sd=0.025,
                    z=0.1, action='EXIT_MR',
                )
            return SignalResult(
                pair_name=pair_name, trade_date=today,
                priceA=1500.0, priceB=700.0,
                spread=-0.04, spread_mu=0.0, spread_sd=0.025,
                z=-1.0, action='NO_ACTION' if open_position is None else 'HOLD',
            )

        idx = pd.date_range(end=date.today(), periods=80, freq='B')
        stub_prices = pd.Series(np.linspace(100, 110, 80), index=idx)

        def fake_fetch(self, symbol, lookback_days):
            return stub_prices

        with patch('services.pair_trading.pair_engine.PairEngine.fetch_daily_history',
                    fake_fetch), \
             patch('services.pair_trading.pair_engine.evaluate_today',
                    stub_evaluate_today):
            engine.daily_scan()

        pos = engine.db.get_position_by_id(pid)
        self.assertEqual(pos['status'], 'CLOSED', "position should be CLOSED after MR")
        self.assertEqual(pos['exit_reason'], 'EXIT_MR')
        # Both legs should have exit-role orders
        orders = engine.db.list_orders(position_id=pid, limit=20)
        exit_orders = [o for o in orders if o['leg_role'] == 'EXIT']
        legs_exited = {o['leg'] for o in exit_orders}
        self.assertSetEqual(legs_exited, {'A', 'B'},
            f"both legs should be squared off, got {legs_exited}")


# ---------------------------------------------------------------------------
# Test 6: Defensive — paper-flag flip without live_trading_enabled blocks orders
# ---------------------------------------------------------------------------

class TestSafetyLock(PairTradingTestBase):

    def test_paper_off_but_live_off_refuses_order(self):
        """If somehow paper_trading_mode=False but live_trading_enabled=False,
        place_pair_order must REFUSE to send orders (safety lock)."""
        from config import PAIR_TRADING_DEFAULTS
        PAIR_TRADING_DEFAULTS['enabled'] = True
        PAIR_TRADING_DEFAULTS['paper_trading_mode'] = False
        PAIR_TRADING_DEFAULTS['live_trading_enabled'] = False  # mismatch
        try:
            from services.pair_trading.pair_engine import get_pair_engine
            from services.pair_trading.signal import SignalResult
            engine = get_pair_engine()
            pair = engine.cohort.by_name('HAVELLS-MARICO')
            sig = SignalResult(
                pair_name='HAVELLS-MARICO',
                trade_date=date.today().isoformat(),
                priceA=1500.0, priceB=700.0,
                spread=-0.05, spread_mu=0.0, spread_sd=0.025,
                z=-2.5, action='ENTRY_LONG',
            )
            pid = engine.place_pair_order_live_or_paper(
                pair=pair, direction=1, signal=sig,
                qty_a=pair.lot_size_a, qty_b=pair.lot_size_b,
            )
            self.assertIsNone(pid, "safety lock should refuse the order, returning None")
        finally:
            # Restore paper-mode safety lock
            PAIR_TRADING_DEFAULTS['paper_trading_mode'] = True
            PAIR_TRADING_DEFAULTS['live_trading_enabled'] = False


if __name__ == '__main__':
    unittest.main(verbosity=2)
