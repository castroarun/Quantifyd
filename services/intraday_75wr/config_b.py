"""Config B engine — 3-System Cost-Resilient (TP 2.0% / SL 1.5%).

Same A1/A2/A3 signal mechanics as Config A; only the TP is wider.
Subclass of ConfigAEngine with the config_id flipped to 'B'. Sub-signal IDs
become B1, B2, B3.
"""

from __future__ import annotations

from services.intraday_75wr.config_a import ConfigAEngine


class ConfigBEngine(ConfigAEngine):
    """Config B re-uses the A1/A2/A3 signal logic and order machinery.
    Only the cfg dict differs (TP 2.0% vs 0.5%, config_id='B'). Sub-system
    ids inherit from self.config_id, so persisted system_id's automatically
    become 'B1', 'B2', 'B3' (see ConfigAEngine._sys()).
    """
    pass
