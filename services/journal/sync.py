"""
Sync orchestrator. Reads from each source and projects rows into journal.db.
Idempotent: UNIQUE(source_db, source_table, source_id) prevents dupes.
"""

from __future__ import annotations
import logging
from typing import Dict

from .journal_db import get_journal_db
from .sources import orb_source, kc6_source, nas_source, strangle_source

logger = logging.getLogger(__name__)


def sync_all(verbose: bool = False) -> Dict[str, int]:
    """Pull from every source and upsert into journal_trades.
    Returns counts per source.
    """
    db = get_journal_db()
    counts: Dict[str, int] = {}
    for name, fetcher in (
        ('orb', orb_source.fetch_closed_trades),
        ('kc6', kc6_source.fetch_closed_trades),
        ('nas', nas_source.fetch_closed_trades),
        ('strangle', strangle_source.fetch_closed_trades),
    ):
        try:
            rows = fetcher()
        except Exception as e:
            logger.warning('sync source %s failed: %s', name, e)
            counts[name] = 0
            continue
        n = 0
        for r in rows:
            try:
                db.upsert_trade(r)
                n += 1
            except Exception as e:
                logger.warning('upsert failed for %s row %s: %s', name, r.get('source_id'), e)
        counts[name] = n
        if verbose:
            print(f'[journal.sync] {name}: {n} rows', flush=True)
    return counts
