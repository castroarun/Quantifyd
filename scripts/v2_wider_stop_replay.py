"""What a WIDER stop would have done — replayed against the recorded option chain.

The live V2 book exits at 2.0%, so a wider stop cannot be observed from the
executor: by the time the underlying travels 2.5% the position is already closed
and there is nothing left to mark. I initially concluded that made wider stops
unmeasurable. That was wrong: the full-chain recorder has logged every strike
since 2026-04-20, so the same four legs CAN be re-marked forward past the live
exit, right up to the point a wider stop would have fired.

This does that. For every closed V2 position it:

  1. reads the legs and the entry spot from v2_positions
  2. walks the recorded underlying spot forward minute by minute from entry
  3. finds where each candidate stop would first have fired
  4. prices the four legs from option_chain at that timestamp
  5. reports the P&L each stop would have realised against what actually happened

Read-only throughout: options_data.db and the V2 database both open mode=ro, no
order is placed, no live state is touched. This runs on demand, not on the
monitor loop — the archive is 13 GB and has no business inside a per-minute job.

Usage:  venv/bin/python3 scripts/v2_wider_stop_replay.py [--stops 2.5,3.0]
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

logging.disable(logging.WARNING)

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT))

V2_DB = ROOT / 'backtest_data' / 'v2_ironfly.db'
CHAIN = ROOT / 'backtest_data' / 'options_data.db'
OUT = ROOT / 'research' / '60_v2_straddle_optimization' / 'results'


def _find_v2_db() -> Path:
    """The V2 store has moved before; locate it rather than assume."""
    if V2_DB.exists():
        return V2_DB
    for p in (ROOT / 'backtest_data').glob('*.db'):
        try:
            c = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
            names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            c.close()
            if 'v2_positions' in names:
                return p
        except Exception:
            continue
    raise SystemExit('could not find a database containing v2_positions')


def leg_price(chain: sqlite3.Connection, tsym: str, at: str):
    """Nearest recorded quote at or before `at` for one leg. Volume/OI carried so a
    stale print can be told from a real one — research/89's binding rule."""
    r = chain.execute(
        "SELECT ltp, bid, ask, volume, oi, snapshot_time FROM option_chain "
        "WHERE tradingsymbol = ? AND snapshot_time <= ? "
        "ORDER BY snapshot_time DESC LIMIT 1", (tsym, at)).fetchone()
    return dict(zip(('ltp', 'bid', 'ask', 'volume', 'oi', 'at'), r)) if r else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stops', default='2.5,3.0',
                    help='comma-separated stop levels in percent (wider than live)')
    args = ap.parse_args()
    stops = [float(x) / 100.0 for x in args.stops.split(',') if x.strip()]

    v2 = sqlite3.connect(f'file:{_find_v2_db()}?mode=ro', uri=True)
    v2.row_factory = sqlite3.Row
    chain = sqlite3.connect(f'file:{CHAIN}?mode=ro', uri=True)

    rows = [dict(r) for r in v2.execute(
        "SELECT id, system, day AS entry_day, entry_time, entry_spot, net_entry, legs_json, "
        "       exit_day, exit_time, exit_reason, exit_spot, pnl, expiry "
        "FROM v2_positions WHERE status='CLOSED' ORDER BY id")]
    print(f'{len(rows)} closed V2 positions · replaying stops '
          f'{", ".join(f"{s:.1%}" for s in stops)} against the recorded chain\n', flush=True)

    out = []
    for r in rows:
        try:
            legs = json.loads(r['legs_json'] or '[]')
        except Exception:
            legs = []
        es = r['entry_spot']
        if not legs or not es:
            continue

        # the recorded underlying path from entry onward
        path = chain.execute(
            "SELECT snapshot_time, spot_price FROM underlying_spot "
            "WHERE symbol='NIFTY' AND snapshot_time >= ? ORDER BY snapshot_time",
            (f"{r['entry_day']} 00:00:00",)).fetchall()
        if not path:
            print(f"  pos {r['id']}: no recorded spot path — skipped")
            continue

        rec = dict(pos=r['id'], entry_day=r['entry_day'], actual_exit=r['exit_day'],
                   actual_reason=r['exit_reason'], actual_pnl=r['pnl'])

        # A wider stop can only change an outcome the LIVE STOP caused. If the position
        # left on a profit target, a roll or the kill switch, the 2.0% stop never fired —
        # so a 2.5% one did not either, and the trade is identical by construction.
        # Counting those as differences was the flaw in the first run.
        if (r['exit_reason'] or '') not in ('move2%', 'gap_or_break'):
            for s in stops:
                rec[f'stop_{s:.3f}'] = dict(fired=False, identical=True,
                                            note=f"actual exit was {r['exit_reason']} — "
                                                 "the live stop never fired, so a wider one "
                                                 "changes nothing")
            out.append(rec)
            print(f"  pos {r['id']:>3} {r['entry_day']} actual {r['exit_reason']:<12} "
                  f"{r['pnl'] or 0:>+10,.0f}   |  unchanged by any wider stop", flush=True)
            continue

        # Stopped out live. The wider stop would have kept it open — walk forward from the
        # ACTUAL EXIT and bound the search at expiry, since the structure dies there.
        after = r['exit_time'] or f"{r['exit_day']} 00:00:00"
        limit = f"{r['expiry']} 23:59:59" if r.get('expiry') else '9999'
        path = [q for q in path if after <= q[0] <= limit] or path

        for s in stops:
            hit = next((p for p in path if p[1] and abs(p[1] - es) / es >= s), None)
            if not hit:
                rec[f'stop_{s:.3f}'] = dict(fired=False,
                                            note='never travelled this far while recorded')
                continue
            at, spot = hit[0], hit[1]
            marks, missing = [], []
            for lg in legs:
                q = leg_price(chain, lg.get('sym') or lg.get('tradingsymbol'), at)
                if not q:
                    missing.append(lg.get('sym'))
                    continue
                px = q['ltp']
                per = (lg['entry'] - px) if lg['side'] == 'SELL' else (px - lg['entry'])
                marks.append(dict(sym=lg.get('sym'), side=lg['side'], entry=lg['entry'],
                                  mark=px, vol=q['volume'], oi=q['oi'], per=per))
            if missing:
                rec[f'stop_{s:.3f}'] = dict(fired=True, at=at, spot=spot,
                                            note=f'no recorded quote for {missing}')
                continue
            qty = legs[0].get('qty') or 650
            pnl = sum(m['per'] for m in marks) * qty
            rec[f'stop_{s:.3f}'] = dict(fired=True, at=at, spot=spot,
                                        move_pct=round(100 * (spot - es) / es, 2),
                                        pnl=round(pnl, 0), vs_actual=round(pnl - (r['pnl'] or 0), 0),
                                        thin_legs=[m['sym'] for m in marks if not m['vol']])
        out.append(rec)

        line = f"  pos {r['id']:>3} {r['entry_day']} actual {r['exit_reason']:<12} {r['pnl'] or 0:>+10,.0f}"
        for s in stops:
            d = rec.get(f'stop_{s:.3f}', {})
            line += (f"   |  {s:.1%} " +
                     (f"{d['pnl']:>+10,.0f} ({d['vs_actual']:>+9,.0f})" if d.get('pnl') is not None
                      else f"{'not fired' if not d.get('fired') else 'no quote':>22}"))
        print(line, flush=True)

    print()
    for s in stops:
        eligible = [r for r in out if not r.get(f'stop_{s:.3f}', {}).get('identical')]
        eff = [r[f'stop_{s:.3f}']['vs_actual'] for r in eligible
               if r.get(f'stop_{s:.3f}', {}).get('vs_actual') is not None]
        if eff:
            print(f'  {s:.1%} stop: {len(eligible)} of {len(out)} positions could be affected '
                  f'(the live-stop exits) · priced {len(eff)} · '
                  f'total effect {sum(eff):>+12,.0f} · mean {sum(eff)/len(eff):>+10,.0f}')
        else:
            print(f'  {s:.1%} stop: never priced — no overlap between the position and the recorder')

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'wider_stop_replay.json').write_text(json.dumps(out, indent=1, default=str),
                                                encoding='utf-8')
    print(f'\nwrote {OUT / "wider_stop_replay.json"}')
    v2.close(); chain.close()


if __name__ == '__main__':
    main()
