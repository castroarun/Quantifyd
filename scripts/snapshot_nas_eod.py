#!/usr/bin/env python3
"""Daily 15:32 IST: snapshot today's NAS intraday curves to /static/snapshots/

Reads the cron-maintained static/nas_mtm.json and writes a date-stamped
copy under static/snapshots/. Also regenerates static/snapshots/index.json
so the Performance tab can list historical days without a backend route.

Idempotent: re-running the same day overwrites that day's snapshot.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'static' / 'nas_mtm.json'
DEST_DIR = ROOT / 'static' / 'snapshots'


def _summary(payload: dict) -> dict:
    sys_dict = payload.get('systems') or {}
    combined = payload.get('combined') or {}
    n_fired = sum(1 for v in sys_dict.values()
                  if (v.get('events') or []) or
                     any(p[1] != 0 for p in (v.get('points') or [])))
    return {
        'combined_last': round(float(combined.get('last') or 0.0), 2),
        'n_fired': n_fired,
        'n_systems': len(sys_dict),
        'n_points': sum(len(v.get('points') or []) for v in sys_dict.values()),
    }


def main() -> int:
    if not SRC.exists():
        print(f"[snapshot] source missing: {SRC}", file=sys.stderr)
        return 1
    payload = json.loads(SRC.read_text())
    d = date.today().isoformat()
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    out_file = DEST_DIR / f'nas_mtm_{d}.json'
    out_file.write_text(json.dumps(payload, separators=(',', ':')))
    print(f"[snapshot] wrote {out_file.name}")

    # Rebuild index.json — date + summary per file. Sorted newest first.
    entries = []
    for p in DEST_DIR.glob('nas_mtm_*.json'):
        try:
            stem = p.stem.replace('nas_mtm_', '')
            try:
                datetime.strptime(stem, '%Y-%m-%d')
            except ValueError:
                continue
            data = json.loads(p.read_text())
            entries.append({
                'date': stem,
                'file': p.name,
                **_summary(data),
            })
        except Exception as e:
            print(f"[snapshot] skip {p.name}: {e}", file=sys.stderr)
    entries.sort(key=lambda x: x['date'], reverse=True)
    index = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'days': entries,
    }
    (DEST_DIR / 'index.json').write_text(json.dumps(index, indent=2))
    print(f"[snapshot] index: {len(entries)} day(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
