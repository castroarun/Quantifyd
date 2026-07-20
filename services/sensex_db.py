"""SENSEX 9:16 — database layer for the 3 variants.

Same pattern as nas_916_db: thin wrappers over NasAtmDB pointing at separate DB files, so the
SENSEX book is kept completely apart from the NIFTY books (separate P&L, separate reconciliation,
separate history) while reusing the identical schema and access code.
"""
import threading

from services.nas_db import DATA_DIR
from services.nas_atm_db import NasAtmDB


class _SensexDB(NasAtmDB):
    FILE = None

    def __init__(self):
        self.db_path = str(DATA_DIR / self.FILE)
        self.db_lock = threading.Lock()
        self._init_database()


class SensexAtmDB(_SensexDB):
    FILE = "sensex_atm_trading.db"


class SensexAtm2DB(_SensexDB):
    FILE = "sensex_atm2_trading.db"


class SensexAtm4DB(_SensexDB):
    FILE = "sensex_atm4_trading.db"


_inst, _lock = {}, threading.Lock()


def _get(cls):
    key = cls.__name__
    if key not in _inst:
        with _lock:
            if key not in _inst:
                _inst[key] = cls()
    return _inst[key]


def get_sensex_atm_db():
    return _get(SensexAtmDB)


def get_sensex_atm2_db():
    return _get(SensexAtm2DB)


def get_sensex_atm4_db():
    return _get(SensexAtm4DB)
