"""
System Validator
================
Pre-market readiness check (08:50 IST) and EOD validation (15:40 IST) for the
three priority systems:

  1. ORB cash (live)         — backtest_data/orb_trading.db
  2. NAS x8   (paper)        — backtest_data/nas_*.db
  3. ORB index / Strangle    — backtest_data/strangle_trading.db

Each report is a structured dict (sections + checks with PASS/WARN/FAIL),
rolled up to an overall verdict with a "user actions" panel at the top.
Emailed via Gmail SMTP (config from ORB_DEFAULTS) and stored in the in-app
notifications DB.

Born out of the 2026-04-24 NAS DB-wipe incident — the operator must know
*before* market open that everything is green, and *immediately after close*
whether the day's session matched expectations.

Usage from app.py scheduler:
    from services.system_validator import run_premarket_check, run_eod_check
    scheduler.add_job(run_premarket_check, 'cron',
                      day_of_week='mon-fri', hour=8, minute=50,
                      id='system_validator_premarket', replace_existing=True)
    scheduler.add_job(run_eod_check, 'cron',
                      day_of_week='mon-fri', hour=15, minute=40,
                      id='system_validator_eod', replace_existing=True)

Manual invocation:
    python -m services.system_validator premarket
    python -m services.system_validator eod
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / 'backtest_data'

# (system_label, db_path, table_prefix) — used by both reports
PRIORITY_SYSTEMS = [
    ('ORB cash',     'backtest_data/orb_trading.db',         'orb_'),
    ('NAS OTM',      'backtest_data/nas_trading.db',         'nas_'),
    ('NAS ATM',      'backtest_data/nas_atm_trading.db',     'nas_atm_'),
    ('NAS ATM2',     'backtest_data/nas_atm2_trading.db',    'nas_atm_'),
    ('NAS ATM4',     'backtest_data/nas_atm4_trading.db',    'nas_atm_'),
    ('NAS 916-OTM',  'backtest_data/nas_916_otm_trading.db', 'nas_'),
    ('NAS 916-ATM',  'backtest_data/nas_916_atm_trading.db', 'nas_atm_'),
    ('NAS 916-ATM2', 'backtest_data/nas_916_atm2_trading.db', 'nas_atm_'),
    ('NAS 916-ATM4', 'backtest_data/nas_916_atm4_trading.db', 'nas_atm_'),
    ('ORB index',    'backtest_data/strangle_trading.db',    'strangle_'),
]

# Scheduler job IDs that MUST be registered for the priority systems to work.
# Pre-market check fails if any are missing or have no next-run.
EXPECTED_JOBS = {
    # ORB cash
    'orb_init_day':                 'ORB cash daily init',
    'orb_update_or':                'ORB cash OR update',
    'orb_eval_signals':             'ORB cash signal evaluator',
    'orb_monitor_pos':              'ORB cash position monitor',
    'orb_activate_trail':           'ORB cash V9t_lock50 trail',
    'orb_eod_squareoff':            'ORB cash EOD squareoff',
    'orb_eod_report':               'ORB cash EOD report',
    'orb_daily_backtest':           'ORB cash daily backtest',
    # NAS shared
    'nas_ticker_autostart':         'NAS WebSocket ticker autostart',
    'nas_eod_squareoff':            'NAS OTM EOD squareoff',
    'nas_market_close':             'NAS forced candle close',
    'nas_daily_summary':            'NAS OTM daily summary',
    # NAS ATM / ATM2 / ATM4
    'nas_atm_eod_squareoff':        'NAS ATM EOD squareoff',
    'nas_atm_daily_summary':        'NAS ATM daily summary',
    'nas_atm2_eod_squareoff':       'NAS ATM2 EOD squareoff',
    'nas_atm2_daily_summary':       'NAS ATM2 daily summary',
    'nas_atm4_eod_squareoff':       'NAS ATM4 EOD squareoff',
    'nas_atm4_daily_summary':       'NAS ATM4 daily summary',
    # NAS 9:16 variants
    'nas_916_sl_monitor':           'NAS 9:16 SL monitor',
    'nas_916_auto_entry':           'NAS 9:16 auto-entry trigger',
    'nas_916_eod_squareoff':        'NAS 9:16 EOD squareoff',
    # ORB index / Strangle
    'strangle_master_tick':         'ORB index master tick',
    'strangle_eod_squareoff':       'ORB index EOD squareoff',
    'strangle_daily_summary':       'ORB index daily summary',
    # Resilience layer
    'db_integrity_watchdog':        'DB integrity watchdog',
    # Self
    'system_validator_premarket':   'Pre-market validator',
    'system_validator_eod':         'EOD validator',
}

PASS = 'PASS'
WARN = 'WARN'
FAIL = 'FAIL'


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _check(name: str, status: str, detail: str = '',
           remediation: str = '') -> dict:
    """Build a single check result."""
    return {'name': name, 'status': status, 'detail': detail,
            'remediation': remediation}


def _safe(fn: Callable[[], dict], name: str) -> dict:
    """Call a check function; if it raises, return a FAIL with the exception."""
    try:
        return fn()
    except Exception as e:
        logger.exception(f'[validator] {name} check raised')
        return _check(name, FAIL, f'check itself crashed: {e}',
                      remediation='inspect logs at journalctl -u quantifyd')


def _rollup(checks: list[dict]) -> str:
    if any(c['status'] == FAIL for c in checks):
        return FAIL
    if any(c['status'] == WARN for c in checks):
        return WARN
    return PASS


def _today_iso() -> str:
    return date.today().isoformat()


def _journalctl(args: list[str], timeout: int = 15) -> str:
    """Read journal for the quantifyd unit. Empty string on failure."""
    try:
        out = subprocess.run(
            ['journalctl', '-u', 'quantifyd', '--no-pager'] + args,
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        return out.stdout
    except Exception as e:
        logger.warning(f'[validator] journalctl failed: {e}')
        return ''


def _connect(db_rel: str) -> Optional[sqlite3.Connection]:
    p = REPO_ROOT / db_rel
    if not p.exists():
        return None
    try:
        c = sqlite3.connect(str(p), timeout=2.0)
        c.row_factory = sqlite3.Row
        return c
    except Exception:
        return None


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


# ----------------------------------------------------------------------------
# Pre-market checks
# ----------------------------------------------------------------------------

def _check_disk() -> dict:
    usage = shutil.disk_usage(str(REPO_ROOT))
    free_gb = usage.free / (1024 ** 3)
    detail = f'{free_gb:.1f} GB free of {usage.total / (1024 ** 3):.0f} GB'
    if free_gb < 1.0:
        return _check('Disk space', FAIL, detail,
                      remediation='free up space on VPS or expand volume')
    if free_gb < 2.0:
        return _check('Disk space', WARN, detail,
                      remediation='under 2 GB — clean old logs/tarballs')
    return _check('Disk space', PASS, detail)


def _check_db_integrity() -> dict:
    try:
        from services.db_watchdog import check_integrity
        issues = check_integrity()
    except Exception as e:
        return _check('DB integrity', FAIL, f'watchdog import failed: {e}')
    if not issues:
        return _check('DB integrity', PASS, 'all 9 watched DBs healthy')
    summary = '; '.join(
        f"{i['db'].split('/')[-1]}: {i['kind']}" for i in issues
    )
    return _check('DB integrity', FAIL, summary,
                  remediation='check journalctl for table-vanish events')


def _check_backup_age() -> dict:
    """Check the latest backup tarball age via local backup.log.

    Backups run Mon-Fri 16:00 IST. After a weekend the gap is naturally
    ~72h, so backup staleness is never a trading-day blocker — surface as
    WARN at most. Operational follow-up belongs in the EOD report, not
    pre-market.
    """
    log_path = REPO_ROOT / 'logs' / 'backup.log'
    if not log_path.exists():
        return _check('Backup pipeline', WARN, 'no logs/backup.log yet')
    try:
        mtime = datetime.fromtimestamp(log_path.stat().st_mtime)
    except Exception as e:
        return _check('Backup pipeline', WARN, f'cannot stat log: {e}')
    age_h = (datetime.now() - mtime).total_seconds() / 3600
    today = datetime.now()
    is_monday = today.weekday() == 0
    # Mon morning gap of 60-80h is the normal weekend gap (Fri 16:00 → Mon 08:50)
    if is_monday and age_h <= 80:
        return _check('Backup pipeline', PASS,
                      f'last activity {age_h:.0f}h ago (normal weekend gap)')
    if age_h > 80:
        return _check(
            'Backup pipeline', WARN,
            f'last backup activity {age_h:.0f}h ago — cron may be broken',
            remediation='chmod +x scripts/backup_to_github_release.sh; check cron',
        )
    if age_h > 28:
        return _check(
            'Backup pipeline', WARN,
            f'last backup activity {age_h:.0f}h ago — verify yesterday\'s 16:00 cron',
        )
    return _check('Backup pipeline', PASS, f'last activity {age_h:.0f}h ago')


def _is_pre_login() -> bool:
    """True if current IST time is before the 08:55 auto-login slot.

    Anything before that — including the 08:50 pre-market validator — should
    not FAIL on token issues, since auto-login is literally about to fire.
    """
    now = datetime.now()
    return (now.hour, now.minute) < (8, 55)


def _check_kite_token() -> dict:
    p = DATA_DIR / 'access_token.json'
    if not p.exists():
        sev = WARN if _is_pre_login() else FAIL
        return _check(
            'Kite access token', sev, 'access_token.json missing',
            remediation='run auto-login or visit /login on dashboard',
        )
    age_h = (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime))\
        .total_seconds() / 3600
    if age_h > 23:
        return _check(
            'Kite access token', WARN,
            f'token age {age_h:.0f}h — auto-login at 08:55 will refresh',
            remediation='auto_login.sh runs at 08:55 IST Mon-Fri',
        )
    return _check('Kite access token', PASS, f'token age {age_h:.0f}h')


def _check_kite_profile() -> dict:
    """Try a profile() call.

    Before 08:55 IST: token failures = WARN (auto-login about to fire).
    After: token failures = FAIL (session genuinely broken).
    """
    pre = _is_pre_login()
    sev_on_fail = WARN if pre else FAIL
    rem_on_fail = ('auto-login at 08:55 will refresh — only act if it persists past 09:00'
                   if pre else 'manual TOTP login required at /login')
    try:
        from services.kite_service import get_kite
        kite = get_kite()
        if kite is None or not getattr(kite, 'access_token', None):
            return _check(
                'Kite API connectivity', sev_on_fail,
                'no kite session yet (auto-login at 08:55 will set one up)',
                remediation=rem_on_fail if not pre else '',
            )
        prof = kite.profile()
        name = prof.get('user_name', '?')
        return _check('Kite API connectivity', PASS, f'profile OK ({name})')
    except Exception as e:
        msg = str(e)[:120]
        return _check(
            'Kite API connectivity', sev_on_fail, msg,
            remediation=rem_on_fail,
        )


def _check_kite_quote() -> dict:
    pre = _is_pre_login()
    sev_on_fail = WARN if pre else FAIL
    try:
        from services.kite_service import get_kite
        kite = get_kite()
        if kite is None or not getattr(kite, 'access_token', None):
            return _check(
                'Kite quote (NIFTY 50)', sev_on_fail,
                'no kite session yet',
            )
        q = kite.ltp(['NSE:NIFTY 50'])
        ltp = q.get('NSE:NIFTY 50', {}).get('last_price')
        if ltp:
            return _check('Kite quote (NIFTY 50)', PASS, f'LTP {ltp}')
        return _check('Kite quote (NIFTY 50)', sev_on_fail, 'empty response')
    except Exception as e:
        return _check('Kite quote (NIFTY 50)', sev_on_fail, str(e)[:120])


def _check_scheduled_jobs() -> list[dict]:
    """One check per expected job. Reuses the running scheduler instance."""
    try:
        # Late import to avoid circular dependency
        from app import scheduler
    except Exception as e:
        return [_check('Scheduler import', FAIL, f'{e}',
                       remediation='inspect app.py scheduler initialization')]
    jobs = {j.id: j for j in scheduler.get_jobs()}
    out = []
    for jid, label in EXPECTED_JOBS.items():
        if jid not in jobs:
            out.append(_check(f'Job: {label}', FAIL, f'id={jid} not registered',
                              remediation=f'verify scheduler.add_job for {jid}'))
            continue
        j = jobs[jid]
        nrt = getattr(j, 'next_run_time', None)
        if nrt is None:
            out.append(_check(f'Job: {label}', WARN, f'id={jid} has no next run',
                              remediation='job may be paused or expired'))
            continue
        out.append(_check(f'Job: {label}', PASS,
                          f'next {nrt.strftime("%Y-%m-%d %H:%M %Z")}'))
    return out


def _check_open_positions_carryforward() -> list[dict]:
    """Any OPEN position carrying over before today's session is suspicious."""
    out = []
    today = _today_iso()
    for label, db_rel, prefix in PRIORITY_SYSTEMS:
        c = _connect(db_rel)
        if c is None:
            out.append(_check(f'Open carry: {label}', WARN, 'DB missing'))
            continue
        try:
            tbl = prefix + 'positions'
            if not _table_exists(c, tbl):
                out.append(_check(f'Open carry: {label}', WARN,
                                  f'table {tbl} missing'))
                continue
            # Count OPEN positions with trade_date / entry_time before today
            try:
                row = c.execute(
                    f"SELECT COUNT(*) AS n FROM {tbl} "
                    f"WHERE status='OPEN' AND COALESCE(trade_date, "
                    f"substr(entry_time,1,10)) < ?",
                    (today,),
                ).fetchone()
                n = row['n'] if row else 0
            except sqlite3.OperationalError:
                # schema mismatch — fall back
                row = c.execute(
                    f"SELECT COUNT(*) AS n FROM {tbl} WHERE status='OPEN'"
                ).fetchone()
                n = row['n'] if row else 0
            if n == 0:
                out.append(_check(f'Open carry: {label}', PASS,
                                  '0 OPEN positions from prior days'))
            elif label == 'ORB cash':
                # ORB is intraday — OPEN before today is a clear FAIL
                out.append(_check(
                    f'Open carry: {label}', FAIL,
                    f'{n} OPEN positions from prior days (intraday system!)',
                    remediation='manual square-off then investigate squareoff handler',
                ))
            else:
                out.append(_check(
                    f'Open carry: {label}', WARN,
                    f'{n} OPEN positions from prior days',
                    remediation='verify EOD squareoff fired yesterday',
                ))
        finally:
            c.close()
    return out


def _check_modes() -> list[dict]:
    """Report the live/paper mode of each priority system. Always PASS — informational."""
    out = []
    try:
        from config import (
            ORB_DEFAULTS, NAS_DEFAULTS, NAS_ATM_DEFAULTS,
            NAS_ATM2_DEFAULTS, NAS_ATM4_DEFAULTS,
            NAS_916_OTM_DEFAULTS, NAS_916_ATM_DEFAULTS,
            NAS_916_ATM2_DEFAULTS, NAS_916_ATM4_DEFAULTS,
        )
    except Exception as e:
        return [_check('Mode lookup', FAIL, str(e))]
    pairs = [
        ('ORB cash',     ORB_DEFAULTS,           'orb_paper_mode', False),
        ('NAS OTM',      NAS_DEFAULTS,           'paper_trading_mode', True),
        ('NAS ATM',      NAS_ATM_DEFAULTS,       'paper_trading_mode', True),
        ('NAS ATM2',     NAS_ATM2_DEFAULTS,      'paper_trading_mode', True),
        ('NAS ATM4',     NAS_ATM4_DEFAULTS,      'paper_trading_mode', True),
        ('NAS 916-OTM',  NAS_916_OTM_DEFAULTS,   'paper_trading_mode', True),
        ('NAS 916-ATM',  NAS_916_ATM_DEFAULTS,   'paper_trading_mode', True),
        ('NAS 916-ATM2', NAS_916_ATM2_DEFAULTS,  'paper_trading_mode', True),
        ('NAS 916-ATM4', NAS_916_ATM4_DEFAULTS,  'paper_trading_mode', True),
    ]
    try:
        from config import STRANGLE_DEFAULTS
        pairs.append(('ORB index', STRANGLE_DEFAULTS, 'paper_trading_mode', True))
    except Exception:
        pass

    for label, cfg, key, expected_paper in pairs:
        paper = cfg.get(key, expected_paper)
        actual = 'PAPER' if paper else 'LIVE'
        target = 'PAPER' if expected_paper else 'LIVE'
        if actual == target:
            out.append(_check(f'Mode: {label}', PASS, f'{actual} (expected)'))
        else:
            out.append(_check(
                f'Mode: {label}', WARN,
                f'{actual} — expected {target}',
                remediation=f'toggle via /api/{label.lower().replace(" ","-")}/toggle-mode',
            ))
    return out


def _build_premarket_report() -> dict:
    sections = [
        {'title': 'Infrastructure', 'checks': [
            _safe(_check_disk, 'disk'),
            _safe(_check_db_integrity, 'db_integrity'),
            _safe(_check_backup_age, 'backup_age'),
        ]},
        {'title': 'Kite / API', 'checks': [
            _safe(_check_kite_token, 'kite_token'),
            _safe(_check_kite_profile, 'kite_profile'),
            _safe(_check_kite_quote, 'kite_quote'),
        ]},
        {'title': 'Scheduled jobs', 'checks':
            _safe(lambda: {'_list': _check_scheduled_jobs()}, 'sched').get(
                '_list', _safe(_check_scheduled_jobs, 'sched_jobs') if False
                else _check_scheduled_jobs()
            )
        },
        {'title': 'Carry-forward (open positions)', 'checks':
            _check_open_positions_carryforward()
        },
        {'title': 'Mode confirmation', 'checks': _check_modes()},
    ]

    for s in sections:
        s['status'] = _rollup(s['checks'])

    overall = _rollup([{'status': s['status']} for s in sections])

    user_actions = []
    for s in sections:
        for c in s['checks']:
            if c['status'] in (FAIL, WARN) and c.get('remediation'):
                user_actions.append({
                    'priority': c['status'],
                    'check': c['name'],
                    'detail': c['detail'],
                    'action': c['remediation'],
                })

    return {
        'report_type': 'premarket',
        'timestamp': datetime.now().isoformat(),
        'date': _today_iso(),
        'overall': overall,
        'summary': _premarket_summary(overall, user_actions),
        'user_actions': user_actions,
        'sections': sections,
    }


def _premarket_summary(overall: str, actions: list[dict]) -> str:
    if overall == PASS:
        return 'All systems ready for market open.'
    n_fail = sum(1 for a in actions if a['priority'] == FAIL)
    n_warn = sum(1 for a in actions if a['priority'] == WARN)
    parts = []
    if n_fail:
        parts.append(f'{n_fail} blocker{"s" if n_fail > 1 else ""}')
    if n_warn:
        parts.append(f'{n_warn} warning{"s" if n_warn > 1 else ""}')
    return f'Pre-market: {" + ".join(parts)} — see actions below.'


# ----------------------------------------------------------------------------
# EOD checks
# ----------------------------------------------------------------------------

def _today_strategy_summary(label: str, db_rel: str, prefix: str) -> dict:
    """Per-system day summary: trades, P&L, win rate, entries/exits, residue."""
    today = _today_iso()
    out = {
        'label': label,
        'db': db_rel,
        'trades_today': 0,
        'entries_today': 0,
        'exits_today': 0,
        'open_after_close': 0,
        'pnl_today': 0.0,
        'wins_today': 0,
        'losses_today': 0,
        'win_rate_today': 0.0,
        'sample_entries': [],
        'sample_exits': [],
        'integrity_issues': [],
    }
    c = _connect(db_rel)
    if c is None:
        out['integrity_issues'].append(f'DB file missing: {db_rel}')
        return out
    try:
        tbl = prefix + 'positions'
        if not _table_exists(c, tbl):
            out['integrity_issues'].append(f'table {tbl} missing')
            return out
        # Probe columns — schemas differ across ORB / NAS / Strangle families.
        cols = {r[1] for r in c.execute(f"PRAGMA table_info({tbl})").fetchall()}

        # Pick the best column for each role; None if not present.
        def first(*candidates):
            for k in candidates:
                if k in cols:
                    return k
            return None

        col_entry_time = first('entry_time', 'entry_ts')
        col_exit_time = first('exit_time', 'exit_ts')
        col_trade_date = first('trade_date', 'entry_date')
        col_pnl = first('pnl_inr', 'net_pnl', 'gross_pnl')
        col_sym = first('instrument', 'tradingsymbol', 'symbol')
        col_entry_price = first('entry_price')  # ORB / NAS only — strangle splits CE/PE
        col_exit_price = first('exit_price')
        col_qty = first('qty')

        # WHERE clause to grab today's entries — prefer trade_date, else strip date from entry_time
        where_today = None
        params = (today,)
        if col_trade_date:
            where_today = f"{col_trade_date} = ?"
        elif col_entry_time:
            where_today = f"substr({col_entry_time},1,10) = ?"
        if where_today is None:
            out['integrity_issues'].append('no date column to filter today')
            return out
        entries = c.execute(
            f"SELECT * FROM {tbl} WHERE {where_today}", params
        ).fetchall()

        out['entries_today'] = len(entries)

        # 'CLOSED' rows that look truly closed (have an exit price OR an exit time)
        def _is_closed(e):
            if e['status'] != 'CLOSED':
                return False
            if col_exit_price and e[col_exit_price] is None:
                # ORB / NAS — needs exit_price set
                return False
            return True

        closed = [e for e in entries if _is_closed(e)]
        out['exits_today'] = len(closed)
        out['trades_today'] = len(entries)

        # P&L
        def _row_pnl(e):
            if col_pnl and e[col_pnl] is not None:
                return e[col_pnl]
            if col_entry_price and col_exit_price and col_qty \
                    and e[col_entry_price] is not None \
                    and e[col_exit_price] is not None \
                    and e[col_qty] is not None:
                return ((e[col_entry_price]) - (e[col_exit_price])) * (e[col_qty])
            return 0.0

        pnl_today = sum(_row_pnl(e) for e in closed)
        out['pnl_today'] = round(pnl_today, 2)
        wins = [e for e in closed if _row_pnl(e) > 0]
        out['wins_today'] = len(wins)
        out['losses_today'] = max(0, len(closed) - len(wins))
        out['win_rate_today'] = (round(len(wins) / len(closed) * 100, 1)
                                 if closed else 0.0)

        # Sample entries/exits — capped at 5
        def _sym(e):
            if col_sym:
                return e[col_sym]
            # strangle has pe_strike/ce_strike but no single symbol — synthesize
            if 'pe_strike' in cols and 'ce_strike' in cols:
                return f"PE {e['pe_strike']}/CE {e['ce_strike']}"
            return '?'

        for e in entries[:5]:
            t = ''
            if col_entry_time and e[col_entry_time]:
                t = str(e[col_entry_time])[11:16]
            entry_px = (e[col_entry_price] if col_entry_price else None)
            if entry_px is None and 'pe_entry_price' in cols and 'ce_entry_price' in cols:
                entry_px = (e['pe_entry_price'] or 0) + (e['ce_entry_price'] or 0)
            out['sample_entries'].append({
                'time': t,
                'symbol': _sym(e),
                'qty': (e[col_qty] if col_qty else 0),
                'entry_price': entry_px,
            })

        for e in closed[:5]:
            t = ''
            if col_exit_time and e[col_exit_time]:
                t = str(e[col_exit_time])[11:16]
            out['sample_exits'].append({
                'time': t,
                'symbol': _sym(e),
                'reason': e['exit_reason'] if 'exit_reason' in cols else '',
                'pnl': round(_row_pnl(e), 2),
            })

        # Residual OPEN positions at EOD
        open_now = c.execute(
            f"SELECT COUNT(*) AS n FROM {tbl} WHERE status='OPEN'"
        ).fetchone()['n']
        out['open_after_close'] = open_now

        # Integrity sweep on today's rows only (cap to 10 issues to avoid noise)
        invalid = []
        for e in entries:
            if e['status'] == 'CLOSED' and col_exit_price \
                    and e[col_exit_price] is None:
                invalid.append(f"id={e['id']} CLOSED but no {col_exit_price}")
            if col_qty and e[col_qty] is not None and e[col_qty] <= 0:
                invalid.append(f"id={e['id']} invalid qty={e[col_qty]}")
            if col_entry_time and col_exit_time \
                    and e[col_exit_time] and e[col_entry_time] \
                    and str(e[col_exit_time]) < str(e[col_entry_time]):
                invalid.append(f"id={e['id']} exit before entry")
        out['integrity_issues'].extend(invalid[:10])
    finally:
        c.close()
    return out


def _check_eod_strategies() -> list[dict]:
    """One check per priority system."""
    out = []
    for label, db_rel, prefix in PRIORITY_SYSTEMS:
        s = _today_strategy_summary(label, db_rel, prefix)
        bits = []
        bits.append(f"{s['trades_today']} trades")
        bits.append(f"{s['exits_today']} closed")
        bits.append(f"P&L Rs {s['pnl_today']:,.0f}")
        bits.append(f"WR {s['win_rate_today']}%")
        detail = ' · '.join(bits)
        # Status logic
        if s['integrity_issues']:
            status = FAIL
            detail += f' · ISSUES: {"; ".join(s["integrity_issues"][:3])}'
            rem = 'inspect DB; see EOD section below for full list'
        elif s['open_after_close'] > 0:
            status = FAIL
            detail += f' · {s["open_after_close"]} OPEN AFTER 15:30'
            rem = 'verify EOD squareoff handler ran; manual square-off may be needed'
        elif s['trades_today'] == 0:
            # Zero trades is informational on a slow day. WARN to draw attention.
            status = WARN
            detail += ' · no trades today'
            rem = ('verify ticker connected, signal logic firing; check 9:16 entry log'
                   if 'NAS' in label else 'verify entry/scan handlers fired today')
        else:
            status = PASS
            rem = ''
        chk = _check(f'Day summary: {label}', status, detail, remediation=rem)
        chk['_summary'] = s  # rich data, used by email template
        out.append(chk)
    return out


def _check_eod_jobs_executed() -> dict:
    """Inspect today's journalctl for any 'raised an exception' job entries."""
    log = _journalctl(['--since', f'{_today_iso()} 00:00'])
    if not log:
        return _check('Scheduler audit', WARN,
                      'journalctl unavailable', remediation='not blocking')
    err_lines = [ln for ln in log.splitlines()
                 if 'raised an exception' in ln
                 or 'Job ' in ln and ' missed' in ln]
    if err_lines:
        sample = ' | '.join(ln[-140:] for ln in err_lines[:3])
        return _check('Scheduler audit', FAIL,
                      f'{len(err_lines)} job errors today: {sample}',
                      remediation='journalctl -u quantifyd --since today | grep exception')
    # Count successful executions for spot-check
    ok = sum(1 for ln in log.splitlines() if 'executed successfully' in ln)
    return _check('Scheduler audit', PASS, f'{ok} jobs executed cleanly today')


def _check_eod_anomalies() -> list[dict]:
    """Watchdog firings, ERROR-level lines, WebSocket reconnects."""
    out = []
    log = _journalctl(['--since', f'{_today_iso()} 00:00'])
    if not log:
        out.append(_check('Anomaly scan', WARN, 'journalctl unavailable'))
        return out
    lines = log.splitlines()
    watchdog = [l for l in lines if 'DB-WATCHDOG' in l and 'detected' in l]
    if watchdog:
        out.append(_check('Watchdog firings', FAIL,
                          f'{len(watchdog)} alert(s) today',
                          remediation='inspect journalctl for table-vanish details'))
    else:
        out.append(_check('Watchdog firings', PASS, 'none today'))
    errs = [l for l in lines if ' ERROR:' in l or ' CRITICAL:' in l]
    # Filter out the noisy benign cases (kite quote 304s, pickling warnings, etc.)
    errs = [l for l in errs if 'kite' not in l.lower() or 'token' in l.lower()]
    if len(errs) > 50:
        sample = ' | '.join(l[-120:] for l in errs[-3:])
        out.append(_check('ERROR log volume', WARN,
                          f'{len(errs)} ERROR/CRITICAL lines today (high). last: {sample}',
                          remediation='spot-check via journalctl, look for repeats'))
    elif errs:
        out.append(_check('ERROR log volume', PASS,
                          f'{len(errs)} ERROR/CRITICAL lines (within tolerance)'))
    else:
        out.append(_check('ERROR log volume', PASS, 'no errors today'))
    reconnects = [l for l in lines if 'KiteTicker' in l
                  and ('reconnect' in l.lower() or 'disconnect' in l.lower())]
    if reconnects:
        out.append(_check('WebSocket stability', WARN,
                          f'{len(reconnects)} reconnect/disconnect events',
                          remediation='check NAS ticker logs around those times'))
    else:
        out.append(_check('WebSocket stability', PASS,
                          'no reconnect/disconnect events'))
    return out


def _build_eod_report() -> dict:
    sections = [
        {'title': 'Per-strategy day summary', 'checks':
            _safe(lambda: {'_list': _check_eod_strategies()}, 'eod_strats').get(
                '_list', _check_eod_strategies()
            )
        },
        {'title': 'Scheduler audit', 'checks':
            [_safe(_check_eod_jobs_executed, 'sched_audit')]
        },
        {'title': 'Anomaly scan', 'checks': _check_eod_anomalies()},
        {'title': 'DB integrity (EOD)', 'checks':
            [_safe(_check_db_integrity, 'db_integrity_eod')]
        },
    ]
    for s in sections:
        s['status'] = _rollup(s['checks'])
    overall = _rollup([{'status': s['status']} for s in sections])

    # Build "actions" list (FAILs only — WARN is informational at EOD)
    user_actions = []
    for s in sections:
        for c in s['checks']:
            if c['status'] == FAIL:
                user_actions.append({
                    'priority': FAIL,
                    'check': c['name'],
                    'detail': c['detail'],
                    'action': c.get('remediation', ''),
                })

    # Aggregate day P&L across the 3 priority systems
    total_pnl = 0.0
    total_trades = 0
    for s in sections[0]['checks']:
        summ = s.get('_summary')
        if summ:
            total_pnl += summ['pnl_today']
            total_trades += summ['trades_today']

    return {
        'report_type': 'eod',
        'timestamp': datetime.now().isoformat(),
        'date': _today_iso(),
        'overall': overall,
        'summary': (
            f'EOD: {total_trades} trades · net P&L Rs {total_pnl:,.0f} · '
            f'overall {overall}'
        ),
        'total_pnl_today': round(total_pnl, 2),
        'total_trades_today': total_trades,
        'user_actions': user_actions,
        'sections': sections,
    }


# ----------------------------------------------------------------------------
# Email rendering
# ----------------------------------------------------------------------------

_STATUS_COLORS = {PASS: '#3fb950', WARN: '#f59e0b', FAIL: '#f85149'}
_STATUS_ICONS = {PASS: '✅', WARN: '⚠️', FAIL: '\U0001f534'}


def _render_html(report: dict) -> str:
    rt = report['report_type']
    title = ('Pre-market Validation' if rt == 'premarket'
             else 'EOD Validation Report')
    overall = report['overall']
    color = _STATUS_COLORS[overall]
    icon = _STATUS_ICONS[overall]
    actions = report.get('user_actions', [])

    def _section_html(s: dict) -> str:
        rows = []
        for c in s['checks']:
            cc = _STATUS_COLORS[c['status']]
            ci = _STATUS_ICONS[c['status']]
            rem = (f"<div style='color:#8b949e;font-size:12px;margin-top:2px'>"
                   f"→ {c['remediation']}</div>"
                   if c.get('remediation') and c['status'] != PASS else '')
            rows.append(
                f"<tr>"
                f"<td style='padding:6px 10px;width:24px;'>{ci}</td>"
                f"<td style='padding:6px 10px;'>"
                f"<div style='font-weight:500'>{c['name']}</div>"
                f"<div style='color:#8b949e;font-size:13px'>{c['detail']}</div>"
                f"{rem}"
                f"</td>"
                f"<td style='padding:6px 10px;text-align:right;color:{cc};"
                f"font-weight:600;font-size:12px'>{c['status']}</td>"
                f"</tr>"
            )
        sc = _STATUS_COLORS[s['status']]
        return (
            f"<div style='margin:18px 0;border:1px solid #30363d;"
            f"border-radius:8px;background:#0d1117'>"
            f"<div style='padding:10px 14px;border-bottom:1px solid #30363d;"
            f"display:flex;justify-content:space-between'>"
            f"<span style='font-weight:600'>{s['title']}</span>"
            f"<span style='color:{sc};font-weight:600;font-size:12px'>"
            f"{s['status']}</span></div>"
            f"<table style='width:100%;border-collapse:collapse'>"
            + ''.join(rows) + "</table></div>"
        )

    actions_html = ''
    if actions:
        rows = []
        for a in actions:
            ac = _STATUS_COLORS[a['priority']]
            rows.append(
                f"<tr>"
                f"<td style='padding:6px 10px;width:60px;color:{ac};"
                f"font-weight:600'>{a['priority']}</td>"
                f"<td style='padding:6px 10px;'>"
                f"<div style='font-weight:500'>{a['check']}</div>"
                f"<div style='color:#8b949e;font-size:13px'>{a['detail']}</div>"
                f"<div style='color:#58a6ff;font-size:13px;margin-top:2px'>"
                f"→ {a['action']}</div>"
                f"</td>"
                f"</tr>"
            )
        actions_html = (
            f"<div style='margin:18px 0;border:2px solid #f85149;"
            f"border-radius:8px;background:#0d1117'>"
            f"<div style='padding:10px 14px;border-bottom:1px solid #30363d;"
            f"font-weight:600;color:#f85149'>"
            f"\U0001f6a8 Action items ({len(actions)})</div>"
            f"<table style='width:100%;border-collapse:collapse'>"
            + ''.join(rows) + "</table></div>"
        )

    sections_html = ''.join(_section_html(s) for s in report['sections'])

    # Top P&L band for EOD
    pnl_band = ''
    if rt == 'eod':
        pnl = report.get('total_pnl_today', 0)
        n = report.get('total_trades_today', 0)
        pnl_color = '#3fb950' if pnl >= 0 else '#f85149'
        pnl_band = (
            f"<div style='padding:14px;background:#161b22;"
            f"border:1px solid #30363d;border-radius:8px;margin:14px 0'>"
            f"<div style='font-size:12px;color:#8b949e'>Net day P&L "
            f"({n} trades across 3 systems)</div>"
            f"<div style='font-size:24px;font-weight:600;color:{pnl_color}'>"
            f"Rs {pnl:,.0f}</div></div>"
        )

    return f"""<!DOCTYPE html>
<html><body style='margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;
background:#010409;color:#c9d1d9;padding:20px'>
<div style='max-width:760px;margin:0 auto'>
<div style='padding:20px;background:#161b22;border:1px solid #30363d;
border-radius:8px;border-left:6px solid {color}'>
<div style='font-size:12px;color:#8b949e'>Quantifyd · {report['date']}</div>
<div style='font-size:22px;font-weight:600;margin-top:4px'>
{icon} {title}</div>
<div style='margin-top:6px;color:#c9d1d9'>{report['summary']}</div>
</div>
{pnl_band}
{actions_html}
{sections_html}
<div style='margin-top:24px;padding:10px;color:#6e7681;font-size:11px;
text-align:center'>
Generated {report['timestamp']} · system_validator
</div>
</div></body></html>"""


def _send_email(report: dict) -> bool:
    """Send via Gmail SMTP using ORB_DEFAULTS config."""
    try:
        from config import ORB_DEFAULTS
        cfg = ORB_DEFAULTS
        if not cfg.get('email_enabled', True):
            logger.info('[validator] email disabled by config')
            return False
        host = cfg.get('smtp_host', 'smtp.gmail.com')
        port = cfg.get('smtp_port', 587)
        sender = cfg.get('email_from', '')
        password = cfg.get('email_app_password', '')
        recipient = cfg.get('email_to', '')
        if not sender or not password or not recipient:
            logger.warning('[validator] email config incomplete; skipping')
            return False

        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        rt = report['report_type']
        subj_prefix = ('[Quantifyd PRE-MARKET]' if rt == 'premarket'
                       else '[Quantifyd EOD]')
        verdict = report['overall']
        subject = f"{subj_prefix} {verdict} · {report['date']}"
        if rt == 'eod':
            subject += f" · P&L Rs {report.get('total_pnl_today', 0):,.0f}"

        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = f'Quantifyd Validator <{sender}>'
        msg['To'] = recipient
        msg.attach(MIMEText(_render_html(report), 'html'))

        with smtplib.SMTP(host, port, timeout=20) as srv:
            srv.starttls()
            srv.login(sender, password)
            srv.send_message(msg)
        logger.info(f'[validator] email sent: {subject}')
        return True
    except Exception as e:
        logger.error(f'[validator] email send failed: {e}', exc_info=True)
        return False


def _store_inapp(report: dict):
    """Persist a copy in the in-app notifications DB."""
    try:
        from services.orb_db import get_orb_db
        db = get_orb_db()
        rt = report['report_type']
        title = ('Pre-market validation' if rt == 'premarket'
                 else 'EOD validation report')
        db.log_notification(
            type='system_alert',
            title=f"{title} ({report['overall']})",
            message=report['summary'],
            data=json.dumps(report)[:8000],
            priority='high' if report['overall'] == FAIL else 'normal',
        )
    except Exception as e:
        logger.warning(f'[validator] in-app store failed: {e}')


# ----------------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------------

def run_premarket_check() -> dict:
    logger.info('[validator] pre-market check starting')
    report = _build_premarket_report()
    _persist_latest(report)
    _store_inapp(report)
    _send_email(report)
    logger.info(
        f"[validator] pre-market done overall={report['overall']} "
        f"actions={len(report.get('user_actions', []))}"
    )
    return report


def run_eod_check() -> dict:
    logger.info('[validator] EOD check starting')
    report = _build_eod_report()
    _persist_latest(report)
    _store_inapp(report)
    _send_email(report)
    logger.info(
        f"[validator] EOD done overall={report['overall']} "
        f"pnl={report.get('total_pnl_today', 0)}"
    )
    return report


def get_latest(report_type: str) -> Optional[dict]:
    """Read most recent report dict from disk; None if not yet generated."""
    p = _latest_path(report_type)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Disk persistence (for /api/validation/<type>/latest endpoint)
# ----------------------------------------------------------------------------

def _latest_path(report_type: str) -> Path:
    return DATA_DIR / f'validator_latest_{report_type}.json'


def _persist_latest(report: dict):
    try:
        DATA_DIR.mkdir(exist_ok=True)
        p = _latest_path(report['report_type'])
        p.write_text(json.dumps(report, indent=2, default=str))
    except Exception as e:
        logger.warning(f'[validator] persist failed: {e}')


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    mode = sys.argv[1] if len(sys.argv) > 1 else 'premarket'
    if mode == 'premarket':
        r = run_premarket_check()
    elif mode == 'eod':
        r = run_eod_check()
    else:
        print('usage: python -m services.system_validator [premarket|eod]')
        sys.exit(1)
    print(json.dumps({
        'overall': r['overall'],
        'summary': r['summary'],
        'sections': [{'title': s['title'], 'status': s['status']}
                     for s in r['sections']],
        'actions': len(r.get('user_actions', [])),
    }, indent=2))
