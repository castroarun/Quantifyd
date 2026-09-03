"""Nightly cron: declare quarterly dividends when due (idempotent).

Runs every weekday evening; services/dividend_engine.py only acts within
DECLARE_WINDOW_DAYS after a calendar quarter end and never declares the same
quarter twice, so this is safe to run daily.

Cron (VPS): 15 19 * * 1-5  cd /home/arun/quantifyd && venv/bin/python scripts/dividend_declare.py >> /tmp/dividend_declare.log 2>&1
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.dividend_engine import declare  # noqa: E402

if __name__ == '__main__':
    print(f'--- dividend_declare {datetime.now()} ---')
    for slug in ('truenorth', 'openalpha'):
        try:
            print(json.dumps(declare(slug), default=str))
        except Exception as e:
            print(f'{slug} FAILED: {e}')
