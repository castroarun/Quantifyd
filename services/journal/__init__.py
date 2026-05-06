"""
Trading Journal package.

Layer above strategy DBs that projects executed trades into a unified
`journal.db` and adds journal-only enrichment (tags, notes, grades,
daily reviews). See docs/Design/TRADING-JOURNAL-DESIGN.md.
"""

from .journal_db import JournalDB, get_journal_db
from .sync import sync_all

__all__ = ['JournalDB', 'get_journal_db', 'sync_all']
