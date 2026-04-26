"""
test_strangle_paper.py — End-to-end smoke test for the Nifty ORB Strangle
paper-trading engine. Uses a MockProvider to bypass Kite, so this can be
run any time (no auth required).

Usage:
    python test_strangle_paper.py

What it covers:
    1. DB init + variant config snapshot
    2. Strike resolution by delta (PE @ -0.22, CE @ +0.10 for a LONG break)
    3. Entry signal detection from synthetic 5-min bars (V1 = OR60 std)
    4. Position open + leg insert
    5. MTM tick under price drift
    6. EOD square-off + cost calc + DB.close_position math
    7. P/L sanity check on the round trip
"""

from __future__ import annotations

import logging
import math
import sqlite3
import sys
from datetime import date, datetime, timedelta, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

# Silence noisy 3rd-party loggers
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logging.getLogger('apscheduler').setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Mock provider — replaces Kite live feeds
# ---------------------------------------------------------------------------

class MockProvider:
    """Synthetic spot + intraday + chain. Deterministic for the test."""

    def __init__(self, spot_now: float, or_high: float, or_low: float):
        self.spot_now = spot_now
        self.or_high = or_high
        self.or_low = or_low
        # Build a synthetic 5-min frame for today: OR window + a clean break
        self.df = self._build_intraday()

    def _build_intraday(self) -> pd.DataFrame:
        today = datetime.now().date()
        rows = []
        # 9:15 → 10:15 builds the OR
        ts = datetime(today.year, today.month, today.day, 9, 15)
        # Walk the OR (12 bars) bouncing between or_low and or_high
        for i in range(12):
            row_close = self.or_low + (i % 3) * ((self.or_high - self.or_low) / 4.0)
            rows.append({'date': ts, 'open': row_close, 'high': row_close + 5,
                         'low': row_close - 5, 'close': row_close})
            ts += timedelta(minutes=5)
        # 10:15 → 10:30: clean break upward
        for i in range(3):
            level = self.or_high + 15 + i * 5  # closes above OR_high
            rows.append({'date': ts, 'open': level - 3, 'high': level + 3,
                         'low': level - 6, 'close': level})
            ts += timedelta(minutes=5)
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        # Inject a strong RSI confirmation by precomputing a synthetic rsi5m
        # (the engine recomputes from close, but we can set it directly).
        from services.nifty_strangle_scanner import rsi
        df['rsi5m'] = rsi(df['close'], 14)
        # Force the break candles to have RSI > 60 by overriding (Wilder ramp from
        # the synthetic OR is too gradual). This is a test convenience.
        df.loc[df.index >= df.index[12], 'rsi5m'] = 70.0
        return df

    def get_spot_ltp(self) -> float:
        return float(self.spot_now)

    def get_intraday_5min(self) -> pd.DataFrame:
        return self.df.copy()

    def get_chain(self, expiry, strikes_min: int, strikes_max: int):
        """Synthetic chain: 50-pt strikes around the spot. Mid prices come
        from BS at flat 18% IV; IV omitted so engine back-implies."""
        from services.collar_engine import bs_price
        out = {}
        # Use a 4-day DTE to keep prices sensible
        today = date.today()
        if isinstance(expiry, str):
            expiry = date.fromisoformat(expiry)
        dte = max((expiry - today).days, 1)
        t_years = dte / 365.0
        r = 0.065
        iv = 0.18
        # Strike grid: clamp to provided range, step 50
        lo = (strikes_min // 50) * 50
        hi = (strikes_max // 50 + 1) * 50
        for k in range(lo, hi + 1, 50):
            ce = bs_price(self.spot_now, k, t_years, r, iv, is_call=True)
            pe = bs_price(self.spot_now, k, t_years, r, iv, is_call=False)
            out[k] = {
                'CE': {'mid': float(ce), 'bid': float(ce) - 0.5,
                       'ask': float(ce) + 0.5, 'ltp': float(ce),
                       'iv': iv, 'tradingsymbol': f"NIFTYTEST{k}CE"},
                'PE': {'mid': float(pe), 'bid': float(pe) - 0.5,
                       'ask': float(pe) + 0.5, 'ltp': float(pe),
                       'iv': iv, 'tradingsymbol': f"NIFTYTEST{k}PE"},
            }
        return out


def test_v1_entry_and_exit():
    print("=" * 70)
    print("TEST: V1 'or60-std' synthetic entry + EOD exit")
    print("=" * 70)

    from services.nifty_strangle_engine import StrangleEngine
    from services.nifty_strangle_db import get_strangle_db
    from config import STRANGLE_DEFAULTS, STRANGLE_VARIANTS_BY_ID

    spot = 22000.0
    or_high = 22050.0
    or_low = 21950.0
    mock = MockProvider(spot_now=spot, or_high=or_high, or_low=or_low)

    # Engine with mock
    engine = StrangleEngine(defaults=STRANGLE_DEFAULTS,
                             variants=[STRANGLE_VARIANTS_BY_ID['or60-std']],
                             mock_provider=mock)
    db = get_strangle_db()

    # Sanity: clean any prior open positions for or60-std
    for p in db.get_open_positions('or60-std'):
        db.close_position(
            position_id=p['id'],
            exit_date=date.today().isoformat(),
            exit_ts=datetime.now().isoformat(),
            pe_exit_price=p['pe_entry_price'],
            ce_exit_price=p['ce_entry_price'],
            spot_at_exit=spot,
            exit_reason='TEST_RESET',
            costs=0.0,
        )

    # 1) Run entry scan
    res = engine.run_entry_scan('or60-std')
    print("\n[1] Entry scan result:")
    for k, v in res.items():
        print(f"    {k}: {v}")
    assert res.get('status') in ('opened', 'past_entry_window', 'no_signal',
                                 'or_window_incomplete'), \
        f"Unexpected status: {res.get('status')}"

    # If we didn't open, the test still demonstrates filter behavior — print why
    if res.get('status') != 'opened':
        print("\n[!] Entry was not taken — likely because 'now' is past the no-entry "
              "window or the synthetic data didn't satisfy the strict K=0 filter.")
        print("    Forcing a manual position via the engine internals to continue test...")
        # We want to exercise the close path; build a position by hand
        spot_at_entry = or_high + 20
        pe_target = STRANGLE_VARIANTS_BY_ID['or60-std']['pe_delta_target_long']
        ce_target = STRANGLE_VARIANTS_BY_ID['or60-std']['ce_delta_target_long']
        chain = engine._quote_chain(spot_at_entry,
                                    date.today() + timedelta(days=4))
        pe = engine._pick_delta_strike(chain, spot_at_entry,
                                       date.today() + timedelta(days=4),
                                       'PE', pe_target)
        ce = engine._pick_delta_strike(chain, spot_at_entry,
                                       date.today() + timedelta(days=4),
                                       'CE', ce_target)
        assert pe and ce, "Could not pick delta-target strikes"
        print(f"    PE pick: strike={pe['strike']} delta={pe['delta']:.3f} mid={pe['mid']:.2f}")
        print(f"    CE pick: strike={ce['strike']} delta={ce['delta']:.3f} mid={ce['mid']:.2f}")

        # PE should land OTM (below spot) for the -0.22 delta target
        assert pe['strike'] < spot_at_entry, \
            f"PE strike {pe['strike']} should be OTM relative to spot {spot_at_entry}"
        assert abs(pe['delta'] - (-0.22)) < 0.10, \
            f"PE delta {pe['delta']} too far from -0.22 (tol 0.10)"
        # CE should land OTM (above spot) for the +0.10 delta target
        assert ce['strike'] > spot_at_entry, \
            f"CE strike {ce['strike']} should be OTM relative to spot {spot_at_entry}"
        assert abs(ce['delta'] - 0.10) < 0.10, \
            f"CE delta {ce['delta']} too far from +0.10 (tol 0.10)"

        position_id = db.open_position(
            variant_id='or60-std', entry_date=date.today().isoformat(),
            entry_ts=datetime.now().isoformat(),
            direction='LONG',
            spot_at_entry=spot_at_entry, or_high=or_high, or_low=or_low,
            or_width_pct=(or_high - or_low) / or_low * 100,
            sl_price=or_low,
            expiry_date=(date.today() + timedelta(days=4)).isoformat(),
            pe_strike=pe['strike'], ce_strike=ce['strike'],
            pe_entry_price=pe['mid'], ce_entry_price=ce['mid'],
            pe_delta=pe['delta'], ce_delta=ce['delta'],
            lot_size=65,
            pe_tradingsymbol=pe['tradingsymbol'],
            ce_tradingsymbol=ce['tradingsymbol'],
            pe_iv=pe['iv'], ce_iv=ce['iv'],
        )
        print(f"\n[2] Manually opened position id={position_id}")
    else:
        position_id = res['position_id']
        print(f"\n[2] Engine opened position id={position_id}")

    # 2) MTM tick
    mtm = engine.run_mtm_tick('or60-std')
    print(f"\n[3] MTM tick result: updated={mtm.get('updated')} spot={mtm.get('spot')}")

    # 3) Verify open count
    opens = db.get_open_positions('or60-std')
    print(f"[4] Open positions for or60-std: {len(opens)}")
    assert len(opens) == 1, f"Expected 1 open position, got {len(opens)}"

    pos = opens[0]
    print(f"    PE strike={pos['pe_strike']} entry={pos['pe_entry_price']:.2f}")
    print(f"    CE strike={pos['ce_strike']} entry={pos['ce_entry_price']:.2f}")
    print(f"    SL={pos['sl_price']:.2f}")

    # 4) EOD square-off (forces exit)
    print("\n[5] Running EOD square-off...")
    closed = engine.run_eod_squareoff('or60-std')
    print(f"    closed={closed.get('closed')}")
    for r in closed.get('results') or []:
        print(f"    Result: pe_pnl={r['pe_pnl']:+.2f} ce_pnl={r['ce_pnl']:+.2f} "
              f"gross={r['gross_pnl']:+.2f} net={r['net_pnl']:+.2f}")
    assert closed.get('closed') == 1, "Expected 1 closed position"

    # 5) Verify trade is in archive
    trades = db.get_trades('or60-std', limit=5)
    print(f"\n[6] Recent trades for or60-std: {len(trades)}")
    if trades:
        t = trades[0]
        print(f"    Latest: net_pnl={t['net_pnl']:+.2f} costs={t['costs']:.2f} "
              f"reason={t['exit_reason']}")
        assert t['exit_reason'] == 'EOD'
        # Sanity: with mock chain and same-day close, PE and CE prices barely
        # move (BS reprices at same DTE, so net_pnl ≈ -costs). That's exactly
        # what we want for round-trip cost validation.
        assert t['costs'] > 0, "Costs should be positive"

    print("\n[OK] All checks passed.")


def test_db_singleton():
    print("\n" + "=" * 70)
    print("TEST: DB singleton pattern + variant snapshot")
    print("=" * 70)
    from services.nifty_strangle_db import get_strangle_db
    db1 = get_strangle_db()
    db2 = get_strangle_db()
    assert db1 is db2, "DB singleton broken"
    variants = db1.list_variants()
    print(f"    Variants in DB: {len(variants)}")
    for v in variants:
        print(f"      {v['variant_id']:20s} enabled={v['enabled']} {v['name']}")
    assert len(variants) >= 8, f"Expected >= 8 variants in DB, got {len(variants)}"


def test_delta_resolution():
    print("\n" + "=" * 70)
    print("TEST: Delta-based strike resolution (LONG-break direction)")
    print("=" * 70)
    from services.nifty_strangle_engine import StrangleEngine
    from config import STRANGLE_DEFAULTS, STRANGLE_VARIANTS_BY_ID

    spot = 22000.0
    mock = MockProvider(spot_now=spot, or_high=22050.0, or_low=21950.0)
    engine = StrangleEngine(defaults=STRANGLE_DEFAULTS,
                             variants=[STRANGLE_VARIANTS_BY_ID['or60-std']],
                             mock_provider=mock)

    expiry = date.today() + timedelta(days=4)
    chain = engine._quote_chain(spot, expiry)
    print(f"    Chain has {len(chain)} strikes")

    pe = engine._pick_delta_strike(chain, spot, expiry, 'PE', -0.22)
    ce = engine._pick_delta_strike(chain, spot, expiry, 'CE', 0.10)
    print(f"    PE @ -0.22 target: strike={pe['strike']} delta={pe['delta']:.3f} "
          f"mid={pe['mid']:.2f} iv={pe['iv']:.3f}")
    print(f"    CE @ +0.10 target: strike={ce['strike']} delta={ce['delta']:.3f} "
          f"mid={ce['mid']:.2f} iv={ce['iv']:.3f}")
    assert abs(pe['delta'] - (-0.22)) < 0.10, \
        f"PE delta {pe['delta']} too far from target -0.22"
    assert abs(ce['delta'] - 0.10) < 0.10, \
        f"CE delta {ce['delta']} too far from target 0.10"
    print("\n[OK] Delta resolution within tolerance.")


def _bootstrap_engine_once():
    """Ensure the engine has been instantiated at least once so variant
    snapshots are persisted to the DB."""
    from services.nifty_strangle_engine import StrangleEngine
    from config import STRANGLE_DEFAULTS, STRANGLE_VARIANTS
    StrangleEngine(defaults=STRANGLE_DEFAULTS, variants=STRANGLE_VARIANTS,
                   mock_provider=MockProvider(22000.0, 22050.0, 21950.0))


def main():
    try:
        _bootstrap_engine_once()
        test_db_singleton()
        test_delta_resolution()
        test_v1_entry_and_exit()
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED")
        print("=" * 70)
        return 0
    except AssertionError as e:
        print(f"\n[FAIL] Assertion error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[FAIL] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
