"""
NAS ATM4 — Database Layer (thin wrapper over NasAtmDB)
======================================================
Uses nas_atm4_trading.db with identical schema and methods.
"""

import threading
from services.nas_atm_db import NasAtmDB, DATA_DIR


DB4_PATH = str(DATA_DIR / "nas_atm4_trading.db")


class NasAtm4DB(NasAtmDB):
    def __init__(self):
        self.db_path = DB4_PATH
        self.db_lock = threading.Lock()
        self._init_database()


# --- Singleton ---------------------------------------------------------

_instance = None
_instance_lock = threading.Lock()


def get_nas_atm4_db():
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = NasAtm4DB()
    return _instance
