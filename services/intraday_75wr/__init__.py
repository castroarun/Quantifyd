"""Intraday 75% WR — three-config live trading engine.

THREE configs (locked 2026-05-06, source research/37 + research/38):

  Config A — 3-System Original (TP 0.5%/SL 1.5%): A1 Diamond Short +
             A2 Long-MR + A3 Long-TC, OOS 78.1% WR, PF 1.28
  Config B — 3-System Cost-Resilient (TP 2.0%/SL 1.5%): same A1/A2/A3
             signals, wider TP, cost-resilient at 0.10%/side
  Config C — Multi-Bar SHORT Bounce (TP 1.5%/SL 1.0%): research/38 winner

ALL THREE default to PAPER MODE — no real Kite orders until both
paper_trading_mode=False AND live_trading_enabled=True.

Sub-system ids in DB.system_id: A1, A2, A3, B1, B2, B3, C.

Concurrency cap = 5 OPEN positions across A+B+C combined.
"""

import threading

_lock = threading.Lock()
_engines: dict = {}


def get_engine(config_id: str):
    """Return the process-wide singleton engine instance for a config_id.

    config_id ∈ {'A', 'B', 'C'}.
    """
    cid = (config_id or '').upper().strip()
    with _lock:
        if cid in _engines:
            return _engines[cid]
        from config import (
            INTRADAY_CONFIG_A_DEFAULTS,
            INTRADAY_CONFIG_B_DEFAULTS,
            INTRADAY_CONFIG_C_DEFAULTS,
        )
        if cid == 'A':
            from services.intraday_75wr.config_a import ConfigAEngine
            _engines['A'] = ConfigAEngine(INTRADAY_CONFIG_A_DEFAULTS)
        elif cid == 'B':
            from services.intraday_75wr.config_b import ConfigBEngine
            _engines['B'] = ConfigBEngine(INTRADAY_CONFIG_B_DEFAULTS)
        elif cid == 'C':
            from services.intraday_75wr.config_c import ConfigCEngine
            _engines['C'] = ConfigCEngine(INTRADAY_CONFIG_C_DEFAULTS)
        else:
            raise ValueError(f'Unknown config_id: {config_id!r}')
        return _engines[cid]


def get_all_engines() -> dict:
    """Return {'A': ConfigAEngine, 'B': ConfigBEngine, 'C': ConfigCEngine}."""
    return {cid: get_engine(cid) for cid in ('A', 'B', 'C')}


def reset_engines_for_test():
    """Test helper — drop the singleton cache."""
    with _lock:
        _engines.clear()


__all__ = ['get_engine', 'get_all_engines', 'reset_engines_for_test']
