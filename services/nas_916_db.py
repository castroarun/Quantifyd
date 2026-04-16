"""
NAS 916 — Database Layer for all 4 x 9:16 Entry variants
=========================================================
Thin wrappers over NasDB / NasAtmDB pointing to separate DB files.
"""

import threading
from services.nas_db import NasDB, DATA_DIR
from services.nas_atm_db import NasAtmDB


# --- OTM 916 DB ---

class Nas916OtmDB(NasDB):
    def __init__(self):
        self.db_path = str(DATA_DIR / "nas_916_otm_trading.db")
        self.db_lock = threading.Lock()
        self._init_database()


_otm_instance = None
_otm_lock = threading.Lock()


def get_nas_916_otm_db():
    global _otm_instance
    if _otm_instance is None:
        with _otm_lock:
            if _otm_instance is None:
                _otm_instance = Nas916OtmDB()
    return _otm_instance


# --- ATM 916 DB ---

class Nas916AtmDB(NasAtmDB):
    def __init__(self):
        self.db_path = str(DATA_DIR / "nas_916_atm_trading.db")
        self.db_lock = threading.Lock()
        self._init_database()


_atm_instance = None
_atm_lock = threading.Lock()


def get_nas_916_atm_db():
    global _atm_instance
    if _atm_instance is None:
        with _atm_lock:
            if _atm_instance is None:
                _atm_instance = Nas916AtmDB()
    return _atm_instance


# --- ATM2 916 DB ---

class Nas916Atm2DB(NasAtmDB):
    def __init__(self):
        self.db_path = str(DATA_DIR / "nas_916_atm2_trading.db")
        self.db_lock = threading.Lock()
        self._init_database()


_atm2_instance = None
_atm2_lock = threading.Lock()


def get_nas_916_atm2_db():
    global _atm2_instance
    if _atm2_instance is None:
        with _atm2_lock:
            if _atm2_instance is None:
                _atm2_instance = Nas916Atm2DB()
    return _atm2_instance


# --- ATM4 916 DB ---

class Nas916Atm4DB(NasAtmDB):
    def __init__(self):
        self.db_path = str(DATA_DIR / "nas_916_atm4_trading.db")
        self.db_lock = threading.Lock()
        self._init_database()


_atm4_instance = None
_atm4_lock = threading.Lock()


def get_nas_916_atm4_db():
    global _atm4_instance
    if _atm4_instance is None:
        with _atm4_lock:
            if _atm4_instance is None:
                _atm4_instance = Nas916Atm4DB()
    return _atm4_instance
