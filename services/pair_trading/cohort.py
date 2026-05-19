"""Pair-Trading (Config D) — Cohort manager.

Loads + manages the 6-pair cohort + alpha/beta values + per-pair config
from PAIR_TRADING_DEFAULTS in config.py.

Each pair carries its own:
  - symA / symB underlyings (F&O futures, lot size from FNO_LOT_SIZES)
  - alpha + beta (TRAIN-FIT regression coefficients)
  - entry_z / stop_z / hold_days / lookback (per-pair tuned parameters)

The cohort is intended to be REPLACED quarterly (cohort_refresh_quarterly=True).
The refresh process re-runs Engle-Granger cointegration on the F&O universe
and rotates decayed pairs out — see regime.py for the (stub) implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PairConfig:
    """Single-pair configuration (entry/exit rules + train-fit alpha/beta)."""
    name: str
    symA: str
    symB: str
    alpha: float
    beta: float
    half_life: float
    entry_z: float
    stop_z: float
    hold_days: int
    lookback: int
    te_wr: float = 0.0
    te_pf: float = 0.0
    # F&O lot sizes — populated from FNO_LOT_SIZES at cohort load
    lot_size_a: int = 0
    lot_size_b: int = 0


@dataclass
class Cohort:
    """6-pair cohort wrapper."""
    pairs: List[PairConfig] = field(default_factory=list)

    def by_name(self, name: str) -> Optional[PairConfig]:
        for p in self.pairs:
            if p.name == name:
                return p
        return None

    def all_symbols(self) -> List[str]:
        seen = set()
        out: List[str] = []
        for p in self.pairs:
            for s in (p.symA, p.symB):
                if s not in seen:
                    seen.add(s)
                    out.append(s)
        return out

    def to_dicts(self) -> List[Dict]:
        out: List[Dict] = []
        for p in self.pairs:
            out.append({
                'name': p.name,
                'symA': p.symA,
                'symB': p.symB,
                'alpha': p.alpha,
                'beta': p.beta,
                'half_life': p.half_life,
                'entry_z': p.entry_z,
                'stop_z': p.stop_z,
                'hold_days': p.hold_days,
                'lookback': p.lookback,
                'te_wr': p.te_wr,
                'te_pf': p.te_pf,
                'lot_size_a': p.lot_size_a,
                'lot_size_b': p.lot_size_b,
            })
        return out


def _load_lot_sizes() -> Dict[str, int]:
    """Pull FNO_LOT_SIZES from data_manager (deferred import to avoid Kite calls)."""
    try:
        from services.data_manager import FNO_LOT_SIZES
        return dict(FNO_LOT_SIZES)
    except Exception as e:
        logger.warning(f"[PairTrading.cohort] FNO_LOT_SIZES unavailable: {e}")
        return {}


def load_cohort(config: Dict) -> Cohort:
    """Build a Cohort from PAIR_TRADING_DEFAULTS-style dict."""
    lot_sizes = _load_lot_sizes()
    cohort = Cohort()
    raw_pairs = config.get('pairs', []) or []
    for pr in raw_pairs:
        pc = PairConfig(
            name=pr['name'],
            symA=pr['symA'], symB=pr['symB'],
            alpha=float(pr['alpha']), beta=float(pr['beta']),
            half_life=float(pr.get('half_life', 0.0)),
            entry_z=float(pr['entry_z']),
            stop_z=float(pr['stop_z']),
            hold_days=int(pr['hold_days']),
            lookback=int(pr['lookback']),
            te_wr=float(pr.get('te_wr', 0.0)),
            te_pf=float(pr.get('te_pf', 0.0)),
        )
        pc.lot_size_a = int(lot_sizes.get(pc.symA, 0))
        pc.lot_size_b = int(lot_sizes.get(pc.symB, 0))
        if pc.lot_size_a == 0 or pc.lot_size_b == 0:
            logger.warning(
                f"[PairTrading.cohort] {pc.name}: missing lot size — "
                f"{pc.symA}={pc.lot_size_a}, {pc.symB}={pc.lot_size_b}"
            )
        cohort.pairs.append(pc)
    logger.info(
        f"[PairTrading.cohort] Loaded {len(cohort.pairs)} pairs across "
        f"{len(cohort.all_symbols())} unique symbols"
    )
    return cohort
