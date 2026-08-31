"""What every stop level would have done on the trades the book actually took.

The live shadow logger only starts recording tomorrow, so the six closed V2
positions have no shadow data — but the full option chain has been recorded since
2026-04-20, so all of them can be backfilled. Both directions work, for different
reasons:

  TIGHTER than live (1.0%, 1.5%) — search from entry to the ACTUAL EXIT. If the
      underlying reached that band first, the tighter stop would have closed the
      position early, at the marks recorded at that minute.

  WIDER than live (2.5%, 3.0%) — only meaningful where the live 2.0% stop is what
      ended the trade. Then the position would have stayed open, so search from the
      actual exit forward, bounded by expiry.

  Anything else — a profit target, a roll, the kill switch — is untouched by a
      wider stop and only touched by a tighter one if it fired first.

Writes static/app/v2_stop_matrix.json for the page, so the card can show the
columns without a backend restart.

Read-only: every database opens mode=ro, nothing is written to any trading store.
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
CHAIN = ROOT / 'backtest_data' / 'options_data.db'
OUT_JSON = ROOT / 'static' / 'app' / 'v2_stop_matrix.json'
LIVE_STOP = 0.02


def find_v2_db() -> Path:
    for p in (ROOT / 'backtest_data').glob('*.db'):
        try:
            c = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
            names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            c.close()
            if 'v2_positions' in names:
                return p
        except Exception:
            continue
    raise SystemExit('no database with v2_positions')


def leg_mark(chain, tsym, at):
    r = chain.execute(
        "SELECT ltp, volume, oi FROM option_chain WHERE tradingsymbol=? AND snapshot_time<=? "
        "ORDER BY snapshot_time DESC LIMIT 1", (tsym, at)).fetchone()
    return dict(zip(('ltp', 'volume', 'oi'), r)) if r else None


def price_at(chain, legs, at):
    """P&L of the whole structure at one timestamp, or None if any leg is unquoted."""
    per, thin = 0.0, 0
    for lg in legs:
        q = leg_mark(chain, lg.get('sym') or lg.get('tradingsymbol'), at)
        if not q or q['ltp'] is None:
            return None, None
        per += (lg['entry'] - q['ltp']) if lg['side'] == 'SELL' else (q['ltp'] - lg['entry'])
        thin += 0 if q['volume'] else 1
    return per * (legs[0].get('qty') or 650), thin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stops', default='1.0,1.5,2.5')
    # v2_positions also holds the inside-week breakout sleeve; the iron-fly card
    # shows only system='v2', and mixing them made my net disagree with the page
    ap.add_argument('--system', default='v2')
    args = ap.parse_args()
    stops = sorted(float(x) / 100.0 for x in args.stops.split(',') if x.strip())

    v2 = sqlite3.connect(f'file:{find_v2_db()}?mode=ro', uri=True)
    v2.row_factory = sqlite3.Row
    chain = sqlite3.connect(f'file:{CHAIN}?mode=ro', uri=True)

    rows = [dict(r) for r in v2.execute(
        "SELECT id, day, entry_time, entry_spot, legs_json, expiry, exit_day, exit_time, "
        "       exit_reason, pnl, system FROM v2_positions "
        "WHERE status='CLOSED' AND system=? ORDER BY id DESC", (args.system,))]

    hdr = ' '.join(f'{s:>13.1%}' for s in stops)
    print(f"{'pos':>4} {'entry':11} {'actual':<12} {'P&L':>10}  {hdr}")
    print('-' * (40 + 14 * len(stops)))

    out, totals = [], {s: [] for s in stops}
    for r in rows:
        legs = json.loads(r['legs_json'] or '[]')
        es = r['entry_spot']
        # exit_time is a bare clock time and the recorder stamps ISO with a 'T'.
        # Comparing those as strings silently matched nothing — every tighter stop
        # came back "same", which cannot be true of a trade that stopped at 2.0%.
        live_exit = (f"{r['exit_day']}T{r['exit_time']}" if r['exit_day'] and r['exit_time']
                     else f"{r['exit_day']}T23:59:59")
        entry_dt = f"{r['day']}T{r['entry_time'] or '00:00:00'}"
        if not legs or not es:
            continue
        stopped_live = (r['exit_reason'] or '') in ('move2%', 'gap_or_break')

        path = chain.execute(
            "SELECT snapshot_time, spot_price FROM underlying_spot WHERE symbol='NIFTY' "
            "AND snapshot_time >= ? AND snapshot_time <= ? ORDER BY snapshot_time",
            (f"{r['day']}T00:00:00", f"{r['expiry']}T23:59:59")).fetchall()

        held = [q for q in path if entry_dt <= q[0] <= live_exit and q[1]]
        after = [q for q in path if q[0] >= live_exit and q[1]]
        mae_held = max((abs(q[1] - es) / es for q in held), default=None)
        mae_after = max((abs(q[1] - es) / es for q in after), default=None)

        rec = dict(pos=r['id'], entry_day=r['day'], exit_day=r['exit_day'],
                   reason=r['exit_reason'], actual_pnl=r['pnl'],
                   # the furthest the underlying got while the trade was open, and the
                   # furthest it got afterwards. A level can only fire if the move reached
                   # it — this is what makes an "unchanged" cell checkable rather than a
                   # claim the reader has to take on trust.
                   max_move_held=round(100 * mae_held, 2) if mae_held is not None else None,
                   max_move_after=round(100 * mae_after, 2) if mae_after is not None else None,
                   stops={})
        cells = []
        for s in stops:
            if s < LIVE_STOP:
                window = [q for q in path if entry_dt <= q[0] <= live_exit]   # fires first
            elif stopped_live:
                window = [q for q in path if q[0] >= live_exit]      # would have stayed open
            else:
                rec['stops'][f'{s}'] = dict(identical=True, pnl=r['pnl'], vs_actual=0)
                cells.append(f'{r["pnl"] or 0:>+13,.0f}')
                continue

            hit = next((q for q in window if q[1] and abs(q[1] - es) / es >= s), None)
            if not hit:
                # No breach found. That means "identical" only if we actually had bars to
                # look at — a gap exit before the recorder's first bar of the day leaves an
                # empty window, and that is unknown, not the same.
                # A trade that stopped at the live level MUST have crossed every tighter
                # band first. If we cannot find it, the recorder missed the decisive
                # minutes — a gap exit stamped before its first bar of the day — and the
                # honest label is unknown, not identical.
                if not window or (stopped_live and s < LIVE_STOP):
                    rec['stops'][f'{s}'] = dict(no_data=True,
                                                note='the move happened outside the recorded '
                                                     'window (gap exit before the first bar)')
                    cells.append(f'{"no data":>13}')
                else:
                    rec['stops'][f'{s}'] = dict(identical=True, pnl=r['pnl'],
                                                vs_actual=0)
                    cells.append(f'{r["pnl"] or 0:>+13,.0f}')
                continue

            pnl, thin = price_at(chain, legs, hit[0])
            if pnl is None:
                rec['stops'][f'{s}'] = dict(no_quote=True)
                cells.append(f'{"no quote":>13}')
                continue
            diff = pnl - (r['pnl'] or 0)
            rec['stops'][f'{s}'] = dict(at=hit[0], spot=hit[1], pnl=round(pnl),
                                        vs_actual=round(diff), thin_legs=thin)
            totals[s].append(diff)
            cells.append(f'{pnl:>+13,.0f}')

        # A trade that stopped at the live 2.0% MUST have crossed every tighter band
        # on its way there. If it did not, the window is wrong, not the market.
        if stopped_live:
            for s in stops:
                if s < LIVE_STOP and rec['stops'].get(f'{s}', {}).get('identical'):
                    print(f"    !! pos {r['id']}: {s:.1%} shows no trigger on a trade that "
                          f"stopped at {LIVE_STOP:.1%} — window or timestamp format is wrong")
        out.append(rec)
        moved = f"moved {rec['max_move_held'] or 0:.2f}% while held"
        if rec['max_move_after']:
            moved += f", {rec['max_move_after']:.2f}% after exit"
        print(f"{r['id']:>4} {r['day']:11} {str(r['exit_reason'])[:12]:<12} "
              f"{r['pnl'] or 0:>+10,.0f}  {' '.join(cells)}   [{moved}]")

    print()
    print(f"{'live 2.0%':<26} {'baseline':>14}   net {sum(r['actual_pnl'] or 0 for r in out):>+12,.0f}")
    for s in stops:
        e = totals[s]
        tag = 'tighter' if s < LIVE_STOP else 'wider'
        if e:
            print(f"{f'{s:.1%} ({tag})':<26} changed {len(e):>2} of {len(out)}   "
                  f"effect {sum(e):>+12,.0f}   mean {sum(e)/len(e):>+10,.0f}")
        else:
            print(f"{f'{s:.1%} ({tag})':<26} never differed from the live book")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_JSON.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(dict(
        generated_at=__import__('datetime').datetime.now().isoformat(timespec='seconds'),
        live_stop=LIVE_STOP, stops=stops, source='recorded option chain since 2026-04-20',
        entry_filters=('VIX>=13 and the CPR/inside-week combo skip are ENTRY gates. This replays the positions the book actually opened, so those gates are already applied — changing a stop cannot change which trades existed.'),
        note=('Backfilled from the recorded chain, not from the live shadow. Tighter stops are '
              'searched between entry and the actual exit; wider stops only where the live stop '
              'ended the trade, then forward to expiry. "same" means that level would not have '
              'changed the outcome.'),
        trades=out), indent=1, default=str), encoding='utf-8')
    tmp.replace(OUT_JSON)
    print(f'\nwrote {OUT_JSON}')
    v2.close(); chain.close()


if __name__ == '__main__':
    main()
