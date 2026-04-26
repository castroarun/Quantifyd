"""
DB Integrity Watchdog
=====================
Periodically verifies trading-DB schema is intact. Triggers a critical
alert if any expected table is missing or the DB is corrupted.

Born out of the 2026-04-24 incident where all 8 NAS DB tables silently
vanished mid-session and trading state was lost for 2+ days before
anyone noticed. The backup pipeline was also silently broken (script
had lost its +x bit), so backups stopped 5 days before the wipe.

Usage from app.py scheduler:
    from services.db_watchdog import run_watchdog_check
    scheduler.add_job(run_watchdog_check, 'cron',
                      day_of_week='mon-fri', hour='9-15', minute='*/5',
                      id='db_watchdog', replace_existing=True)
"""
import logging
import os
import sqlite3
from typing import List, Tuple

logger = logging.getLogger(__name__)

# (db_path_relative, [expected_table_names]) — paths relative to repo root.
# If any expected table is missing from a present DB file, watchdog alerts.
WATCHED_DBS: List[Tuple[str, List[str]]] = [
    ('backtest_data/nas_trading.db',
     ['nas_state', 'nas_positions', 'nas_trades', 'nas_orders',
      'nas_signals', 'nas_daily_state']),
    ('backtest_data/nas_atm_trading.db',
     ['nas_atm_state', 'nas_atm_positions', 'nas_atm_trades',
      'nas_atm_orders', 'nas_atm_signals', 'nas_atm_daily_state']),
    ('backtest_data/nas_atm2_trading.db',
     ['nas_atm_state', 'nas_atm_positions', 'nas_atm_trades',
      'nas_atm_orders', 'nas_atm_signals', 'nas_atm_daily_state']),
    ('backtest_data/nas_atm4_trading.db',
     ['nas_atm_state', 'nas_atm_positions', 'nas_atm_trades',
      'nas_atm_orders', 'nas_atm_signals', 'nas_atm_daily_state']),
    ('backtest_data/nas_916_otm_trading.db',
     ['nas_state', 'nas_positions', 'nas_trades', 'nas_orders',
      'nas_signals', 'nas_daily_state']),
    ('backtest_data/nas_916_atm_trading.db',
     ['nas_atm_state', 'nas_atm_positions', 'nas_atm_trades',
      'nas_atm_orders', 'nas_atm_signals', 'nas_atm_daily_state']),
    ('backtest_data/nas_916_atm2_trading.db',
     ['nas_atm_state', 'nas_atm_positions', 'nas_atm_trades',
      'nas_atm_orders', 'nas_atm_signals', 'nas_atm_daily_state']),
    ('backtest_data/nas_916_atm4_trading.db',
     ['nas_atm_state', 'nas_atm_positions', 'nas_atm_trades',
      'nas_atm_orders', 'nas_atm_signals', 'nas_atm_daily_state']),
    ('backtest_data/orb_trading.db',
     ['orb_daily_state', 'orb_positions', 'orb_signals', 'orb_orders']),
]


def check_integrity() -> List[dict]:
    """Inspect every watched DB. Returns list of issue dicts (empty if all OK).

    Each issue has: db, kind ('missing-file'|'missing-tables'|'corrupted'|'open-error'),
    and detail-specific fields.
    """
    issues: List[dict] = []
    for db_path, expected in WATCHED_DBS:
        if not os.path.exists(db_path):
            # A missing file is OK if the strategy hasn't been initialized yet
            # — only alert if it was present before. Track via marker file.
            marker = db_path + '.seen'
            if os.path.exists(marker):
                issues.append({'db': db_path, 'kind': 'missing-file'})
            continue
        # Mark as seen so any future disappearance is flagged
        try:
            with open(db_path + '.seen', 'w') as f:
                f.write('1')
        except Exception:
            pass
        try:
            conn = sqlite3.connect(db_path, timeout=2.0)
        except Exception as e:
            issues.append({'db': db_path, 'kind': 'open-error', 'error': str(e)})
            continue
        try:
            try:
                tables = {r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()}
            except Exception as e:
                issues.append({'db': db_path, 'kind': 'open-error', 'error': str(e)})
                continue
            missing = [t for t in expected if t not in tables]
            if missing:
                issues.append({
                    'db': db_path,
                    'kind': 'missing-tables',
                    'missing': missing,
                    'present': sorted(tables),
                })
                continue
            # Quick integrity probe — only on healthy schema (avoid false positives
            # when tables are missing). PRAGMA quick_check is faster than integrity_check.
            try:
                row = conn.execute('PRAGMA quick_check').fetchone()
                if row and row[0] != 'ok':
                    issues.append({
                        'db': db_path,
                        'kind': 'corrupted',
                        'pragma_result': row[0],
                    })
            except Exception as e:
                issues.append({'db': db_path, 'kind': 'open-error', 'error': str(e)})
        finally:
            conn.close()
    return issues


def run_watchdog_check():
    """Scheduler entry point. Logs healthy state once, alerts on any issue."""
    try:
        issues = check_integrity()
    except Exception as e:
        logger.error(f"[DB-WATCHDOG] check_integrity raised: {e}", exc_info=True)
        return

    if not issues:
        logger.debug("[DB-WATCHDOG] all DBs healthy")
        return

    # Build a critical alert message
    lines = ["DB integrity watchdog detected issues:"]
    for i in issues:
        if i['kind'] == 'missing-file':
            lines.append(f"  - {i['db']}: FILE DISAPPEARED")
        elif i['kind'] == 'missing-tables':
            lines.append(f"  - {i['db']}: missing tables {i['missing']}")
        elif i['kind'] == 'corrupted':
            lines.append(f"  - {i['db']}: CORRUPTED ({i['pragma_result']})")
        elif i['kind'] == 'open-error':
            lines.append(f"  - {i['db']}: open-error {i['error']}")

    msg = '\n'.join(lines)
    logger.error(f"[DB-WATCHDOG] {msg}")

    # Fire a critical notification (in-app + email)
    try:
        from services.notifications import get_notification_service
        svc = get_notification_service()
        svc.send_alert(
            alert_type='system_alert',
            title=f'DB integrity FAIL ({len(issues)} db)',
            message=msg,
            priority='critical',
            data={'issues': issues},
        )
    except Exception as e:
        logger.error(f"[DB-WATCHDOG] failed to dispatch alert: {e}")


if __name__ == '__main__':
    # CLI: python -m services.db_watchdog
    logging.basicConfig(level=logging.INFO)
    issues = check_integrity()
    if issues:
        for i in issues:
            print(i)
    else:
        print("all DBs healthy")
