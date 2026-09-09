# -*- coding: utf-8 -*-
"""Bake True North's whole state to a static file, the way Open Alpha does.

THE PROBLEM. /api/momentum-paper/state takes 11.6s cold and 0.7s warm, because get_state()
does a live Kite quote plus a large pandas pivot on the REQUEST path. The cache that makes
it warm does not survive long - a gunicorn worker recycle or the EOD data refresh is enough
- so in practice the page sits on its first-paint skeleton for ten seconds and then snaps
into place. Open Alpha has never had that problem for one reason: its page reads a file a
cron already wrote, and renders in about 2ms.

THE PAYLOAD IS 11 KB. There was never anything heavy about the answer, only about
computing it. So compute it out of band, on a schedule, and let the page read the answer.

Written atomically (tmp + os.replace) so a page fetching mid-write never sees half a file.
Carries its own `baked` timestamp so the page can tell how old the slow-moving half is,
separately from the per-minute price file.
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / 'static' / 'app' / 'momentum_state.json'


def main():
    from services import momentum_paper as mp
    t0 = time.time()
    try:
        st = mp.get_state()
    except Exception as e:
        # Leave the previous file in place: a stale answer the page can render beats no
        # answer at all, and the timestamp inside it tells the reader how stale.
        print('get_state failed, keeping the previous bake: %s' % e)
        raise SystemExit(1)
    took = time.time() - t0

    st['baked'] = str(datetime.now())
    st['bake_seconds'] = round(took, 2)
    tmp = OUT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(st, default=str), encoding='utf-8')
    os.replace(tmp, OUT)
    print('%s baked %d keys in %.1fs -> %s (%.1f KB)'
          % (datetime.now().strftime('%H:%M:%S'), len(st), took, OUT.name,
             OUT.stat().st_size / 1024))


if __name__ == '__main__':
    main()
