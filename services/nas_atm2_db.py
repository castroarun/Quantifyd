"""
NAS ATM2 — Database Layer (thin wrapper over NasAtmDB)
======================================================
Uses nas_atm2_trading.db with identical schema and methods.
"""

import threading
from services.nas_atm_db import NasAtmDB, DATA_DIR


DB2_PATH = str(DATA_DIR / "nas_atm2_trading.db")


class NasAtm2DB(NasAtmDB):
    def __init__(self):
        self.db_path = DB2_PATH
        self.db_lock = threading.Lock()
        self._init_database()


# --- Singleton ---------------------------------------------------------

_instance = None
_instance_lock = threading.Lock()


def get_nas_atm2_db():
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = NasAtm2DB()
    return _instance
