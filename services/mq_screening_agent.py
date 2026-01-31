"""
MQ Screening Agent
===================

Monthly full-universe screening of Nifty 500 stocks.
Runs on the 1st trading day of each month + on-demand for replacements.

Pipeline:
1. Market regime check (Nifty 500 vs 200 DMA + India VIX)
2. Momentum scan (ATH proximity filter)
3. Fundamental filter (quality scoring)
4. Composite ranking
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np

try:
    import yfinance as yf
except ImportError:
    yf = None

from .mq_agent_db import get_agent_db, MQAgentDB
from .mq_agent_reports import MarketRegime, ScreeningReport
from .nifty500_universe import get_nifty500
from .momentum_filter import screen_momentum_fast
from .fundamental_data_service import (
    assess_quality_batch, rank_quality_stocks, FundamentalCriteria
)

logger = logging.getLogger(__name__)

# Config
from config import MQ_DEFAULTS


class ScreeningAgent:
    """Monthly full-universe screening agent."""

    def __init__(self, db: MQAgentDB = None):
        self.db = db or get_agent_db()

    def run(self, as_of_date: datetime = None) -> ScreeningReport:
        """Run the full screening pipeline."""
        start_time = time.time()
        run_id = self.db.start_agent_run('screening')

        try:
            # Step 1: Market regime
            regime = self._check_market_regime()
            logger.info(f"Market regime: {regime.regime} (200DMA: {'above' if regime.above_200dma else 'below'})")

            # Step 2: Load universe
            universe = get_nifty500()
            total_scanned = len(universe.symbols)
            logger.info(f"Scanning {total_scanned} stocks...")

            # Step 3: Momentum scan
            momentum_result = screen_momentum_fast(
                symbols=universe.symbols,
                as_of_date=as_of_date,
                threshold=MQ_DEFAULTS['ath_proximity_threshold'],
            )
            momentum_symbols = [r.symbol for r in momentum_result.passed]
            logger.info(f"Momentum filter: {len(momentum_symbols)}/{total_scanned} passed")

            # Step 4: Fundamental filter
            quality_scores = []
            if momentum_symbols:
                quality_scores = assess_quality_batch(
                    symbols=momentum_symbols,
                    is_financial_fn=lambda s: universe.is_financial(s),
                    use_cache=True,
                )

            # Step 5: Rank and select top candidates
            ranked = rank_quality_stocks(quality_scores, top_n=50)
            quality_passed = len(ranked)
            logger.info(f"Quality filter: {quality_passed} passed")

            # Build top-ranked list with metadata
            momentum_lookup = {r.symbol: r for r in momentum_result.passed}
            top_ranked = []
            for i, score in enumerate(ranked):
                mom = momentum_lookup.get(score.symbol)
                stock = universe.get(score.symbol)
                top_ranked.append({
                    'rank': i + 1,
                    'symbol': score.symbol,
                    'sector': stock.sector if stock else 'Unknown',
                    'is_financial': stock.is_financial if stock else False,
                    'distance_from_ath': round(mom.distance_from_ath * 100, 1) if mom else None,
                    'current_price': mom.current_price if mom else 0,
                    'composite_score': round(score.composite_score, 3),
                    'passes_all': score.passes_all,
                    'fail_reasons': score.fail_reasons,
                })

            # Build funnel data
            funnel = {
                'universe': total_scanned,
                'momentum': len(momentum_symbols),
                'quality': quality_passed,
                'top_30': min(30, quality_passed),
            }

            report = ScreeningReport(
                run_date=as_of_date or datetime.now(),
                regime=regime,
                total_scanned=total_scanned,
                momentum_passed=len(momentum_symbols),
                quality_passed=quality_passed,
                top_ranked=top_ranked[:30],
                pipeline_candidates=top_ranked[:50],
                funnel=funnel,
            )

            duration = time.time() - start_time
            self.db.complete_agent_run(
                run_id,
                signals_count=0,
                report_type='monthly_screening',
                summary=f"Scanned {total_scanned}: {len(momentum_symbols)} momentum, {quality_passed} quality. Regime: {regime.regime}",
                duration_seconds=round(duration, 1),
            )

            logger.info(f"Screening complete in {duration:.1f}s")
            return report

        except Exception as e:
            self.db.fail_agent_run(run_id, str(e))
            logger.error(f"Screening failed: {e}")
            raise

    def _check_market_regime(self) -> MarketRegime:
        """Check market regime via Nifty 500 vs 200 DMA + India VIX."""
        regime = MarketRegime()

        if not yf:
            logger.warning("yfinance not available, defaulting to BULLISH")
            return regime

        try:
            # Nifty 500 index
            nifty500 = yf.Ticker('^CRSLDX')
            hist = nifty500.history(period='1y')

            if not hist.empty:
                regime.index_close = float(hist['Close'].iloc[-1])
                regime.index_200dma = float(hist['Close'].tail(200).mean()) if len(hist) >= 200 else float(hist['Close'].mean())
                regime.above_200dma = regime.index_close > regime.index_200dma

            # India VIX
            try:
                vix = yf.Ticker('^INDIAVIX')
                vix_hist = vix.history(period='5d')
                if not vix_hist.empty:
                    regime.vix = float(vix_hist['Close'].iloc[-1])
                    regime.vix_ok = regime.vix < 25  # VIX < 25 is normal
            except Exception:
                pass  # VIX data not always available

            # Determine regime
            if not regime.above_200dma:
                regime.regime = 'BEARISH'
                regime.allow_entries = False
            elif regime.vix and regime.vix >= 25:
                regime.regime = 'HIGH_VOLATILITY'
                regime.allow_entries = True  # Allow but with caution
            else:
                regime.regime = 'BULLISH'
                regime.allow_entries = True

        except Exception as e:
            logger.warning(f"Could not check market regime: {e}. Defaulting to BULLISH.")

        return regime
