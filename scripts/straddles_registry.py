"""Build the straddles systems registry — one normalised feed for the redesigned page.

The approved mock (docs/mockups/straddles-systems-table.html) needs per system:
kind, size, window, today's state, running P&L, open risk, distance to stop, lifetime
record, the open legs, the closed trades, the evidence behind it, and its rules.

None of that exists in one place. Each book publishes through its own feed
(csl_paper.json, the v2 sqlite store, condor_paper.json, the v1 replay JSON) and none
of them carry kind, window, open risk or backtest provenance. This joins them.

The split that keeps it honest:

  * SPEC below is EDITORIAL and hand-authored - what a system is, what it does, what
    it deliberately does not do, which study justifies it and how that study was run.
    It changes when a rule changes, never automatically.
  * everything else is READ from the live feeds at build time.

Read-only against every source. Writes static/app/straddles_systems.json, which the
page fetches - so this is a cron artefact and needs no Flask restart.

Run: venv/bin/python3 scripts/straddles_registry.py
"""
from __future__ import annotations

import json, sqlite3, statistics as st, sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
OUT = ROOT / 'static' / 'app' / 'straddles_systems.json'

NIFTY_WD2DTE = {0: 1, 1: 0, 2: 4, 3: 3, 4: 2}
SENSEX_WD2DTE = {0: 3, 1: 2, 2: 1, 3: 0, 4: 4}
TODAY = date.today().isoformat()
DAYNAME = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
DAYORD = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

# ---------------------------------------------------------------- editorial spec
# money: 'real' | 'paper' | 'refuted'.  kind: 'intraday' | 'positional'.
SPEC = {
    'NAS_COMB20': dict(
        name='NAS_COMB20', kind='intraday', venue='NIFTY', money='real',
        subtitle='Combined-premium stop · Tuesday only',
        rules_doc='research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py (BOOKS)',
        does=[('Universe', 'NIFTY weekly ATM straddle; strike = nearest 50 to spot at entry.'),
              ('Entry', '<b>09:16</b>, one straddle — sell ATM CE + ATM PE together.'),
              ('Days', '<b>Tuesday only (DTE 0)</b> since 31 Aug 2026. The Monday cell moved to '
                       'paper as NAS_COMB20_MON.'),
              ('Stop', '<b>Combined premium +25%</b> — both legs added and watched as one number. '
                       'Needs <b>2 consecutive polls</b> above the level (dwell) before it acts.'),
              ('Exit', '<b>15:20</b> time exit, or the combined stop, whichever comes first.')],
        doesnt=['No per-leg stop — one leg can run to any price while the pair stays inside 25%.',
                'No re-entry. Once stopped, the day is over.',
                'No adjustment, no rolling, no re-centring.',
                'NOT covered by the NAS book-level portfolio stop — this sleeve carries its own.'],
        evidence=dict(method=['Our recorded chain'], period='92 days · Apr–Aug 2026',
                      nums={'Net @10 lots': '+₹1.45L', 't': '1.23', 'Max DD': '−₹59,630',
                            'Stops': '2 / 20'},
                      how='Replayed on the 1-minute option chain recorded since 20 Apr 2026.',
                      caveat='<b>Short sample</b> — 92 days cannot settle a mean this size. The '
                             'Monday cell was moved to paper on 31 Aug on this evidence.',
                      links=[('research/138', '/app/backtest/sensex-nifty-stop-by-dte')])),
    'NAS_COMB20_MON': dict(
        name='NAS_COMB20_MON', kind='intraday', venue='NIFTY', money='paper',
        subtitle='COMB’s retired Monday cell, kept on paper',
        rules_doc='research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py',
        does=[('Entry', '09:16, ATM straddle, Monday (DTE 1) only.'),
              ('Stop', 'Combined premium +30% — the level it ran live on.'),
              ('Exit', '15:20 time exit, or the stop.')],
        doesnt=['Not real money since 31 Aug 2026.',
                'Size and stop carried over unchanged, so the paper record continues the live one.'],
        evidence=dict(method=['Our recorded chain'], period='92 days',
                      nums={'Net @10 lots': '+₹1.45L', 't': '1.23'},
                      how='Same replay as COMB20, sliced to DTE 1.',
                      caveat='Moved off real money because it duplicated the 9:16 suite’s '
                             'strongest day while carrying ~4× the drawdown of the best cell.',
                      links=[('research/138', '#')])),
    'NAS_COMB20_THU': dict(
        name='NAS_COMB20_THU', kind='intraday', venue='NIFTY', money='paper',
        subtitle='Thursday DTE3 cell · paper twin',
        rules_doc='research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py',
        does=[('Entry', '09:16, ATM straddle, Thursday (DTE 3) only.'),
              ('Stop', 'Combined premium +20%.'), ('Exit', '15:20.')],
        doesnt=['Not promoted to live: NIFTY came off Thursdays on 27 Aug for SENSEX-expiry margin.'],
        evidence=dict(method=['Our recorded chain'], period='18 Thursdays',
                      nums={'Net @10 lots': '+₹1.55L', 't': '3.85', 'Max DD': '−₹15,790',
                            'Stops': '0 / 18'},
                      how='Replayed per DTE on the recorded chain.',
                      caveat='Steadiest cell in the grid and <b>stop-invariant from 20% upward</b> '
                             '— the combined premium never rose 20% on any recorded Thursday.',
                      links=[('research/138', '#')])),
    'NAS_COMB20_FRI': dict(
        name='NAS_COMB20_FRI', kind='intraday', venue='NIFTY', money='paper',
        subtitle='Friday DTE2 cell · paper twin',
        rules_doc='research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py',
        does=[('Entry', '09:16, ATM straddle, Friday (DTE 2) only.'),
              ('Stop', 'Combined premium +30%.'), ('Exit', '15:20.')],
        doesnt=['Parked on paper because NIFTY is live Mon+Tue only.'],
        evidence=dict(method=['Our recorded chain'], period='18 Fridays',
                      nums={'Net @10 lots': '+₹1.06L', 't': '0.94'},
                      how='Replayed per DTE on the recorded chain.', caveat='',
                      links=[('research/138', '#')])),
    'CSL30F_NIFTY': dict(
        name='CSL30F · NIFTY', kind='intraday', venue='NIFTY', money='paper',
        subtitle='Flat 30% combined stop, every DTE — the control',
        rules_doc='research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py',
        does=[('Purpose', 'The <b>flat control</b> for the combined-stop family: one rule on every '
                          'day, so tuned books can be measured against something un-tuned.'),
              ('Entry', '09:16, ATM straddle, all five DTEs.'),
              ('Stop', 'Combined premium <b>+30% on every day</b> — deliberately not tuned.'),
              ('Exit', '15:20, or the stop.')],
        doesnt=['Not optimised, by design. If a tuned book cannot beat this, the tuning is noise.',
                'Never traded with real money.'],
        evidence=dict(method=['Our recorded chain'], period='92 days',
                      nums={}, how='Replayed on the recorded chain.',
                      caveat='Exists to be beaten, not to be deployed.',
                      links=[('research/111', '#')])),
    'CSL30F_SENSEX': dict(
        name='CSL30F · SENSEX', kind='intraday', venue='SENSEX', money='paper',
        subtitle='Flat combined stop, every DTE',
        rules_doc='research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py',
        does=[('Entry', '09:16, SENSEX ATM straddle, all DTEs.'),
              ('Stop', 'Combined premium, per-DTE 20–30%.'), ('Exit', '15:20.')],
        doesnt=['Never traded with real money.'],
        evidence=dict(method=['Our recorded chain'], period='92 days', nums={},
                      how='Replayed on the recorded SENSEX chain (research/139).',
                      caveat='<b>SENSEX has no day-allocation edge</b> — the best cell reaches '
                             't 1.21 against a ~1.5 noise bar for the cells tested.',
                      links=[('research/139', '#')])),
    'CSL_TIMEB_NIFTY': dict(
        name='TimeB · NIFTY', kind='intraday', venue='NIFTY', money='paper',
        subtitle='Windowed entry — pulled from live 28 Aug',
        rules_doc='research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py',
        does=[('Entry', 'A per-DTE <b>window</b>, not the whole day — e.g. 10:00→12:00.'),
              ('Stop', 'Combined premium, per-DTE 20–25%.'),
              ('Exit', 'End of the window.')],
        doesnt=['<b>Not real money since 28 Aug 2026</b> — pulled after −₹8,152 in one 10:00–12:00 '
                'window at 6 lots, where the 20% stop never fired.'],
        evidence=dict(method=['AlgoTest', 'Our recorded chain'], period='3 yrs · 739 trades',
                      nums={'Net': '−₹75,468', 'Expectancy': '−0.01', 'Max DD': '−₹6.30L'},
                      how='AlgoTest 3-year NIFTY run of the same structure with real costs.',
                      caveat='<b class="neg">The long-sample evidence is NEGATIVE.</b> Our 85-day '
                             'record disagreed; the larger sample won and the book came off live.',
                      links=[('Ops review', '#')])),
    'CSL_TIMEB_SENSEX': dict(
        name='TimeB · SENSEX', kind='intraday', venue='SENSEX', money='paper',
        subtitle='Windowed entry on SENSEX',
        rules_doc='research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py',
        does=[('Entry', 'Per-DTE window — DTE0 13:00→15:20, DTE1 10:30→12:00.'),
              ('Stop', 'Combined premium; the DTE0 cell runs a wide 50% disaster backstop.'),
              ('Exit', 'End of the window.')],
        doesnt=['Not real money since 28 Aug 2026.',
                'The DTE0 backstop is deliberately wide — on SENSEX expiry a tight stop is tripped '
                'by noise that then reverts.'],
        evidence=dict(method=['Our recorded chain'], period='92 days',
                      nums={}, how='Replayed on the recorded SENSEX chain.',
                      caveat='On SENSEX expiry every stop from 15% to 40% costs ~70% of the cell '
                             'and still fires 7/18 times (research/139).',
                      links=[('research/139', '#')])),
    'CSL_TIMEB2_LIVE': dict(
        name='TimeB2 · expiry-Tue', kind='intraday', venue='NIFTY', money='paper',
        subtitle='Second afternoon slot on expiry Tuesday',
        rules_doc='research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py',
        does=[('Why it exists', 'research/125 found a <b>second earnings pocket</b> in the '
                                'afternoon of expiry Tuesday, separate from the 09:16 entry.'),
              ('Entry', '<b>13:15</b> on expiry Tuesday (DTE 0) only.'),
              ('Stop', 'Combined premium +30%.'),
              ('Exit', '<b>14:30</b> — a 75-minute window, not the rest of the day.')],
        doesnt=['Trades one day a week and nothing else.',
                'Despite the name, it carries no live-money flag — see the 05-Sep ops review.'],
        evidence=dict(method=['Our recorded chain'], period='research/125',
                      nums={}, how='Window sweep over the recorded chain.',
                      caveat='A single-cell book on a handful of sessions — treat the record as '
                             'indicative only.',
                      links=[('research/125', '#')])),
    'CSL_TIMEB2_NIFTY': dict(
        name='2nd Slots · NIFTY', kind='intraday', venue='NIFTY', money='paper',
        subtitle='Second-slot evidence book, Mon + Tue',
        rules_doc='research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py',
        does=[('Why it exists', 'Tests whether the second afternoon pocket exists on <b>Monday '
                                'as well as Tuesday</b>, at token size.'),
              ('Entry', 'DTE0 13:00→14:00 · DTE1 10:00→12:00.'),
              ('Stop', 'Combined premium +25%.')],
        doesnt=['2 lots — sized to gather evidence, not to earn.',
                'Cells unchanged pending the 05-Sep review.'],
        evidence=dict(method=['Our recorded chain'], period='sweep sec-18',
                      nums={}, how='Second-slot sweep on the recorded chain.',
                      caveat='', links=[('research/111', '#')])),
    'CSL_TIMEB_NIFTY_MON': dict(
        name='TimeB · Monday', kind='intraday', venue='NIFTY', money='paper',
        subtitle='TimeB’s dropped Monday cell, kept for the Nov re-run',
        rules_doc='research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py',
        does=[('Entry', '13:00 → 14:00, Monday (DTE 1) only.'),
              ('Stop', 'Combined premium +20%.')],
        doesnt=['<b>Dropped from the live TimeB book on 23 Aug</b> — condemned by research/120, '
                '121 and 122 independently.',
                'Kept trading on paper so the November re-run has live-shaped evidence.'],
        evidence=dict(method=['Our recorded chain'], period='window atlas r/122',
                      nums={'R:R @p95': '1 : 11.8', 'Modelled P(loss)': '52%'},
                      how='Window atlas across the recorded chain.',
                      caveat='Condemned by three independent routes. It exists to be re-checked '
                             'in November, not to be revived.',
                      links=[('research/122', '#')])),
    'CSL_TIMEB_NIFTY_MON_AM': dict(
        name='TimeB · Monday AM', kind='intraday', venue='NIFTY', money='paper',
        subtitle='Morning Monday cell · user override vs the study',
        rules_doc='research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py',
        does=[('Entry', '<b>09:16 → 11:16</b>, Monday (DTE 1) only.'),
              ('Stop', 'A <b>rupee</b> stop — ₹1,000 per lot — not a percentage. It caps the worst '
                       'day best of the three options tested.')],
        doesnt=['<b>Fails its own null test.</b> research/124 makes this the best Monday cell '
                'on median, but it does not survive a label shuffle (p = 0.376, n = 18) — '
                'indistinguishable from mined noise.',
                'Run anyway, knowingly, at Arun’s call. Review after 4 live Mondays.'],
        evidence=dict(method=['Our recorded chain'], period='n = 18 Mondays',
                      nums={'Median @8L': '+₹6,920', 'Win': '88.9%', 'p (shuffle)': '0.376'},
                      how='Re-run of the Monday window grid on the recorded chain.',
                      caveat='<b class="neg">Chosen against the evidence.</b> The shuffle test '
                             'cannot separate it from noise; it is live on judgement, not proof.',
                      links=[('research/124', '#')])),
    'CSL30F_SENSEX_WED': dict(
        name='CSL30F · SENSEX Wed', kind='intraday', venue='SENSEX', money='paper',
        subtitle='Wednesday full-day cell · user override vs the study',
        rules_doc='research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py',
        does=[('Entry', '09:16 → 15:20, Wednesday (SENSEX DTE 1) only.'),
              ('Stop', 'Combined premium +30%.')],
        doesnt=['<b>Chosen against the study.</b> The Wednesday full-day cell measured −₹571/day '
                'at 64% over 11 days, and the verdict said windows-only.',
                'Run anyway after Arun saw the table. Review after 4 live Wednesdays.'],
        evidence=dict(method=['Our recorded chain'], period='n = 11 Wednesdays',
                      nums={'Mean/day': '−₹571', 'Win': '64%'},
                      how='Per-DTE cell sweep on the recorded SENSEX chain.',
                      caveat='<b class="neg">Negative in the study that motivated it.</b> On '
                             'paper by design until the review.',
                      links=[('research/139', '#')])),
    'NAS_C20_TRAIL': dict(
        name='COMB + Trail', kind='intraday', venue='NIFTY', money='paper',
        subtitle='On stop, trail instead of quit',
        rules_doc='research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py',
        does=[('Entry', '09:16 ATM straddle, combined 20% stop.'),
              ('Management', 'When the stop trips, <b>trail</b> the winner instead of closing — '
                             'exit on a 30% bounce off the post-trigger low.')],
        doesnt=['A management A/B, never a deployment candidate on its own.'],
        evidence=dict(method=['Our recorded chain'], period='92 days', nums={},
                      how='Forward A/B against the plain COMB rule.',
                      caveat='', links=[('research/111 §14', '#')])),
    'NAS_C20_SHIFT': dict(
        name='COMB + Shift', kind='intraday', venue='NIFTY', money='paper',
        subtitle='On stop, re-centre instead of quit',
        rules_doc='research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py',
        does=[('Entry', '09:16 ATM straddle, combined 20% stop.'),
              ('Management', 'When the stop trips, <b>re-centre</b> to the new ATM — up to 3 '
                             'shifts, none after 14:30.')],
        doesnt=['A management A/B, never a deployment candidate on its own.'],
        evidence=dict(method=['Our recorded chain'], period='92 days', nums={},
                      how='Forward A/B against the plain COMB rule.',
                      caveat='', links=[('research/111 §14', '#')])),
}

# books present in the state file but without an editorial entry get a stub
STUB = dict(kind='intraday', venue='NIFTY', money='paper', subtitle='',
            rules_doc='research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py',
            does=[], doesnt=[], evidence=dict(method=[], period='', nums={}, how='', caveat='',
                                              links=[]))


def dd(v):
    cum = peak = w = 0.0
    for x in v:
        cum += x; peak = max(peak, cum); w = min(w, cum - peak)
    return round(w)


def tstat(v):
    return round(st.mean(v) / (st.stdev(v) / len(v) ** 0.5), 2) if len(v) > 2 and st.stdev(v) else None


systems = []

# ---------------------------------------------------------------- CSL / COMB family
state = json.loads((ROOT / 'backtest_data' / 'csl_paper_state.json').read_text())
cfg = json.loads((ROOT / 'backtest_data' / 'csl_paper_config.json').read_text())['books']
recs = [r for r in state['records'] if r.get('pnl') is not None]
by_book = defaultdict(list)
for r in recs:
    by_book[r['book']].append(r)

for book, rows in sorted(by_book.items()):
    spec = dict(STUB, **SPEC.get(book, {}))
    spec.setdefault('name', book)
    rows.sort(key=lambda r: r['day'])
    v = [float(r['pnl']) for r in rows]
    cells = cfg.get(book) or {}
    wd2 = SENSEX_WD2DTE if spec['venue'] == 'SENSEX' else NIFTY_WD2DTE
    dte_today = wd2.get(date.today().weekday())
    cell = cells.get(str(dte_today))
    lots = rows[-1].get('lots')
    today_row = next((r for r in rows if r['day'][:10] == TODAY), None)

    # WINDOW: what the book's window IS, not merely whether it fires today.
    if cell:
        window = f"{cell['entry']} → {cell['exit']}"
    else:
        wins = sorted({f"{c['entry']} → {c['exit']}" for c in cells.values()
                       if isinstance(c, dict) and c.get('entry')})
        days = sorted({DAYNAME[w] for w in range(5) for d in (wd2.get(w),)
                       if str(d) in cells}, key=lambda x: DAYORD.index(x))
        window = (wins[0] if len(wins) == 1 else (wins[0] + ' +' + str(len(wins) - 1)
                                                 if wins else '—'))
        if days:
            window += '  · ' + '/'.join(days)

    # STATE: a time exit and a stop-out are different events; do not flatten them.
    if today_row:
        why = (today_row.get('reason') or '').upper()
        # exit_ts is a BARE clock time ('11:28:49'), not an ISO timestamp
        hhmm = str(today_row.get('exit_ts') or '')[:5]
        if 'SL' in why or 'STOP' in why:
            stlabel, tone = f"Stopped {hhmm}".strip(), 'neg'
        elif why:
            stlabel, tone = f"Closed {hhmm}".strip(), 'neutral'
        else:
            stlabel, tone = f"Closed {hhmm}".strip(), 'neutral'
    elif cell:
        stlabel, tone = 'Waiting', 'neutral'
    else:
        stlabel, tone = 'Not scheduled today', 'muted'

    systems.append(dict(
        key=book, name=spec['name'], subtitle=spec['subtitle'], kind=spec['kind'],
        venue=spec['venue'], money=spec['money'],
        size_lots=lots, size_qty=rows[-1].get('qty'),
        window=window,
        state=dict(label=stlabel, tone=tone),
        today_pnl=(round(float(today_row['pnl'])) if today_row else None),
        running_pnl=None, risk_open=None, to_stop=None,
        lifetime=dict(net=round(sum(v)), n=len(v),
                      win=round(100 * sum(1 for x in v if x > 0) / len(v)) if v else None,
                      maxdd=dd(v), t=tstat(v)),
        legs=[], curve=[],
        closed=[dict(day=r['day'][:10], dte=r.get('dte'), strike=r.get('strike'),
                     credit=r.get('credit'), exit=r.get('exit_comb'),
                     reason=r.get('reason'), pnl=round(float(r['pnl'])))
               for r in rows[::-1][:40]],
        evidence=spec['evidence'],
        rules=dict(does=spec['does'], doesnt=spec['doesnt'], doc=spec['rules_doc']),
    ))

# ---------------------------------------------------------------- V2 iron fly
V2SPEC = dict(
    name='V2 · Iron fly', kind='positional', venue='NIFTY', money='paper',
    subtitle='2nd weekly · ±2% wings · 2% move-stop · 40% target',
    rules_doc='services/v2_ironfly_api.py (CFG) · gates: services/v2_breakout_signals.py',
    does=[('Structure', 'Short iron fly: <b>SELL ATM CE + PE</b>, <b>BUY wings at ±2.0% of ATM</b> '
                        '(≈ ±500 pts at NIFTY 24k). Defined risk both sides.'),
          ('Expiry', '<b>2nd-nearest weekly</b>, needing ≥ 4 calendar days to expiry.'),
          ('Entry', '<b>09:20</b>, when flat and both gates pass.'),
          ('Gates', 'India VIX ≥ 13.0 <b>and</b> the combo skip-filter — it sits out when the '
                    '<b>prior-day CPR width &lt; 0.10%</b> of spot, or when last week was an '
                    '<b>inside week</b>. Every skip is shadow-logged.'),
          ('Manage', '<b>2% underlying move-stop</b> · <b>+40% of credit</b> target · '
                     '<b>roll at DTE ≤ 1</b>.'),
          ('Re-entry', 'Re-enters on the next qualifying day after an exit.')],
    doesnt=['No per-leg stop — the wings are the structural protection.',
            'Does not roll a threatened side or re-centre; it exits and waits.',
            '<b>Not armed.</b> Real trading needs mode = live AND armed = 1; both are off.'],
    evidence=dict(method=['AlgoTest', 'Our bhavcopy'], period='7.3 yrs · 2019–2026',
                  nums={'Net @10 lots': '+₹8.80L', 'Calmar': '1.03', 'Max DD': '−₹1.17L',
                        'Trades': '204'},
                  how='AlgoTest’s 1-minute engine, cross-checked on our own 15-year NSE bhavcopy '
                      '(+₹6.93L vs +₹8.80L, losing years matched).',
                  caveat='<b class="neg">⚠ The backtest traded the FRONT weekly; this book trades '
                         'the 2nd.</b> The expiry lever was never tested — AlgoTest caps entry at '
                         '4 trading days before expiry (research/141).',
                  links=[('V2 study', '/app/backtest/v2-nifty-ironfly-sl-vix'),
                         ('research/141', '#')]))

db = ROOT / 'backtest_data' / 'v2_ironfly_trading.db'
if db.exists():
    c = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    c.row_factory = sqlite3.Row
    pos = [dict(r) for r in c.execute("SELECT * FROM v2_positions WHERE system='v2' ORDER BY id")]
    settings = {r[0]: r[1] for r in c.execute('SELECT key,val FROM v2_settings')} \
        if c.execute("SELECT name FROM sqlite_master WHERE name='v2_settings'").fetchone() else {}
    c.close()
    closed = [p for p in pos if p.get('exit_time')]
    open_ = [p for p in pos if not p.get('exit_time')]
    v = [float(p['pnl']) for p in closed if p.get('pnl') is not None]
    armed = settings.get('armed') == '1' and settings.get('mode') == 'live'
    legs, curve, running = [], [], None
    if open_:
        o = open_[-1]
        # legs_json carries side/instrument_type/strike/entry/ltp but NO qty and NO pnl,
        # so both are derived. SELL profits when the premium falls, BUY when it rises.
        QTY = 650
        try:
            legs = []
            for l in json.loads(o.get('legs_json') or '[]'):
                en, lt = l.get('entry'), (l.get('ltp') if l.get('ltp') is not None else l.get('exit'))
                sgn = 1 if l.get('side') == 'SELL' else -1
                legs.append(dict(side=l.get('side'), type=l.get('instrument_type'),
                                 strike=l.get('strike'), role=l.get('role'), qty=QTY,
                                 entry=en, ltp=lt,
                                 pnl=(round(sgn * (en - lt) * QTY)
                                      if en is not None and lt is not None else None)))
        except Exception:
            legs = []
        running = o.get('pnl_now')
        try:
            curve = [[t, p] for t, p in json.loads(o.get('series_json') or '[]')][-80:]
        except Exception:
            curve = []
    systems.append(dict(
        key='V2_IRONFLY', **{k: V2SPEC[k] for k in ('name', 'subtitle', 'kind', 'venue', 'money')},
        size_lots=10, size_qty=650, window='09:20 · roll DTE≤1',
        state=dict(label=('Holding' if open_ else ('Armed' if armed else 'Flat · unarmed')),
                   tone=('pos' if open_ else 'muted')),
        today_pnl=None, running_pnl=running,
        risk_open=None, to_stop=None,
        lifetime=dict(net=round(sum(v)) if v else 0, n=len(v),
                      win=round(100 * sum(1 for x in v if x > 0) / len(v)) if v else None,
                      maxdd=dd(v), t=tstat(v)),
        legs=legs, curve=curve,
        closed=[dict(day=p['day'], exit=p.get('exit_time'), expiry=p.get('expiry'),
                     credit=p.get('net_entry'), reason=p.get('exit_reason'),
                     pnl=round(float(p['pnl'])) if p.get('pnl') is not None else None)
                for p in closed[::-1][:40]],
        evidence=V2SPEC['evidence'],
        rules=dict(does=V2SPEC['does'], doesnt=V2SPEC['doesnt'], doc=V2SPEC['rules_doc'])))

# ---------------------------------------------------------------- Wed→Fri condor
cp = ROOT / 'static' / 'app' / 'condor_paper.json'
if cp.exists():
    d = json.loads(cp.read_text())
    hist = d.get('history') or []
    v = [float(h['pnl']) for h in hist if h.get('pnl') is not None]
    systems.append(dict(
        key='CONDOR_WEDFRI', name='Wed→Fri iron condor', kind='positional', venue='NIFTY',
        money='refuted', subtitle='±0.8% shorts, wings 1% beyond each short',
        size_lots=d.get('lots'), size_qty=d.get('qty'), window='Wed close → Fri close',
        state=dict(label='Stopped', tone='muted'),
        today_pnl=None, running_pnl=None, risk_open=None, to_stop=None,
        lifetime=dict(net=round(sum(v)) if v else 0, n=len(v),
                      win=round(100 * sum(1 for x in v if x > 0) / len(v)) if v else None,
                      maxdd=dd(v), t=tstat(v)),
        legs=[], curve=[],
        closed=[dict(day=h['entry_day'], exit=h.get('exit_day'), expiry=h.get('expiry'),
                     credit=h.get('credit'), reason='Fri close',
                     pnl=round(float(h['pnl']))) for h in hist[::-1][:40]],
        evidence=dict(method=['Our bhavcopy'], period='15 yrs · 434 campaigns',
                      nums={'Net @2 lots': '−₹83,569', 't': '−1.30', 'Max DD': '−₹1.36L'},
                      how='434 real campaigns priced on NSE bhavcopy closes, untraded contracts '
                          'excluded.',
                      caveat='The original +₹880/campaign came from a <b>no-skew simulation</b>; '
                             'on real prices it is −₹193. The book’s 7 winning cycles sit 1.45 '
                             'standard errors from that — luck, not counter-evidence.',
                      links=[('research/140', '#')]),
        rules=dict(
            does=[('Purpose', 'Built to use the days the 9:16 books leave idle — in Wednesday, out '
                              'Friday, flat before Monday so it never competes for margin.'),
                  ('Structure', 'SELL a strangle <b>~0.8% OTM</b> either side; BUY each wing '
                                '<b>1.0% beyond its own short</b> (≈250-pt verticals).'),
                  ('Entry', 'Wednesday close (~15:10), front-of-next weekly.'),
                  ('Exit', 'Friday close. Never carried over a weekend.'),
                  ('Stop', 'The backtested spec closes if the combined premium doubles.')],
            doesnt=['<b>The running book has no stop code at all</b> — only the wings cap it. A '
                    'known divergence from the tested spec.',
                    'Never held into Monday or Tuesday.',
                    '<b>Refuted 31 Aug 2026</b> on 434 real campaigns: −₹193/campaign at t −1.30.'],
            doc='research/80_farDTE_rescue/scripts/condor_paper.py · verdict: research/140')))

# ---------------------------------------------------------------- write + report
order = {'real': 0, 'paper': 1, 'refuted': 2}
systems.sort(key=lambda s: (order.get(s['money'], 3), s['kind'] != 'intraday', -abs(s['lifetime']['net'] or 0)))

payload = dict(generated_at=datetime.now().isoformat()[:19], date=TODAY,
               n=len(systems), systems=systems)
OUT.write_text(json.dumps(payload), encoding='utf-8')

print(f'wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)  ·  {len(systems)} systems\n')
print(f"{'key':24} {'kind':11} {'money':8} {'lots':>5} {'window':>16} {'net':>11} {'n':>4} {'state'}")
print('-' * 108)
for s in systems:
    print(f"{s['key']:24} {s['kind']:11} {s['money']:8} {str(s['size_lots'] or '—'):>5} "
          f"{s['window']:>16} {s['lifetime']['net']:>11,} {s['lifetime']['n']:>4} "
          f"{s['state']['label']}")
miss = [s['key'] for s in systems if not s['rules']['does']]
if miss:
    print(f"\n  {len(miss)} system(s) still on the stub spec (no rules authored): {miss}")
