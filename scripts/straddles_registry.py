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
# the CSL / COMB / TimeB books are one family on one mechanic — grouped apart
NINE16 = ('NAS_COMB20', 'CSL30F', 'CSL_TIMEB', 'NAS_C20')
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
                      links=[('research/138', '/app/backtest/sensex-nifty-stop-by-dte')])),
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
                      links=[('research/138', '/app/backtest/sensex-nifty-stop-by-dte')])),
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
                      links=[('research/138', '/app/backtest/sensex-nifty-stop-by-dte')])),
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
                      links=[('research/111', '/app/backtest/csl-best-config-straddles')])),
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
                      links=[('research/139', 'https://github.com/castroarun/Quantifyd/tree/main/research/139_sensex_dte_allocation')])),
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
                      links=[('Ops centre', '/app/straddles#ops-center')])),
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
                      links=[('research/139', 'https://github.com/castroarun/Quantifyd/tree/main/research/139_sensex_dte_allocation')])),
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
                      links=[('research/125', 'https://github.com/castroarun/Quantifyd/tree/main/research/125_expiry_afternoon_straddle')])),
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
                      caveat='', links=[('research/111', '/app/backtest/csl-best-config-straddles')])),
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
                      links=[('research/122', 'https://github.com/castroarun/Quantifyd/tree/main/research/122_window_atlas')])),
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
                      links=[('research/124', 'https://github.com/castroarun/Quantifyd/tree/main/research/124_monday_cells')])),
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
                      links=[('research/139', 'https://github.com/castroarun/Quantifyd/tree/main/research/139_sensex_dte_allocation')])),
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
                      caveat='', links=[('research/111', '/app/backtest/csl-best-config-straddles')])),
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
                      caveat='', links=[('research/111', '/app/backtest/csl-best-config-straddles')])),
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
# csl_paper_state holds only CLOSED trades; the open position and its live mark
# live here. Without it a holding book looks flat.
_lf = ROOT / 'static' / 'app' / 'csl_paper_live.json'
LIVE = json.loads(_lf.read_text()) if _lf.exists() else {}
LIVE_BOOKS = (LIVE.get('books') or {}) if LIVE.get('day') == TODAY else {}
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

    # STATE: the live feed wins — a book that is holding right now must say so.
    lv = LIVE_BOOKS.get(book) or {}
    lstate = (lv.get('state') or '').upper()
    running, curve = None, []
    if lstate == 'OPEN':
        ser = lv.get('series') or []
        if ser:
            running = round(float(ser[-1][1]))
            curve = [[t, v] for t, v in ser][-120:]
        stlabel = f"Holding since {lv.get('entry') or '09:16'}"
        tone = 'pos' if (running or 0) >= 0 else 'neg'
    elif lstate == 'WAIT_ENTRY':
        stlabel, tone = f"Waiting · enters {lv.get('entry') or '—'}", 'neutral'
    elif today_row:
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
        group='916',
        venue=spec['venue'], money=spec['money'],
        size_lots=lots, size_qty=rows[-1].get('qty'),
        window=window,
        state=dict(label=stlabel, tone=tone),
        today_pnl=(round(float(today_row['pnl'])) if today_row else None),
        # a quiet row still has a last result; a dash throws that away
        last_pnl=(round(float(rows[-1]['pnl'])) if rows else None),
        last_day=(rows[-1]['day'][:10] if rows else None),
        running_pnl=running, risk_open=None, to_stop=None,
        lifetime=dict(net=round(sum(v)), n=len(v),
                      win=round(100 * sum(1 for x in v if x > 0) / len(v)) if v else None,
                      maxdd=dd(v), t=tstat(v)),
        legs=[], curve=curve,
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
                         ('research/141', 'https://github.com/castroarun/Quantifyd/tree/main/research/141_v2_bhav_pertrade'), ('full dossier', 'https://claude.ai/code/artifact/a3487b7e-e6c4-4a88-8f7f-e006999915fe')]))

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
        key='V2_IRONFLY', group='positional',
        **{k: V2SPEC[k] for k in ('name', 'subtitle', 'kind', 'venue', 'money')},
        size_lots=10, size_qty=650, window='09:20 · roll DTE≤1',
        state=dict(label=('Holding' if open_ else ('Armed' if armed else 'Flat · unarmed')),
                   tone=('pos' if open_ else 'muted')),
        today_pnl=None, running_pnl=running,
        last_pnl=(round(float(closed[-1]['pnl'])) if closed and closed[-1].get('pnl') is not None else None),
        last_day=(closed[-1]['day'] if closed else None),
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
        key='CONDOR_WEDFRI', group='positional',
        name='Wed→Fri iron condor', kind='positional', venue='NIFTY',
        money='refuted', subtitle='±0.8% shorts, wings 1% beyond each short',
        size_lots=d.get('lots'), size_qty=d.get('qty'), window='Wed close → Fri close',
        state=dict(label='Stopped', tone='muted'),
        today_pnl=None, running_pnl=None,
        last_pnl=(round(float(hist[0]['pnl'])) if hist and hist[0].get('pnl') is not None else None),
        last_day=(hist[0].get('exit_day') if hist else None),
        risk_open=None, to_stop=None,
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
                      links=[('research/140', 'https://github.com/castroarun/Quantifyd/tree/main/research/140_condor_real_chain')]),
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

# ---------------------------------------------------------------- V1 books + V2 lab
SA = ROOT / 'static' / 'app' / 'straddles'


def _load(fn):
    f = SA / fn
    return json.loads(f.read_text()) if f.exists() else None


def _add(key, name, subtitle, group, kind, money, lots, qty, window, vals, closed,
         rules, evid, state=('Flat', 'muted'), today=None):
    systems.append(dict(
        key=key, name=name, subtitle=subtitle, group=group, kind=kind, venue='NIFTY',
        money=money, size_lots=lots, size_qty=qty, window=window,
        state=dict(label=state[0], tone=state[1]),
        today_pnl=today,
        last_pnl=(round(vals[-1]) if vals else None),
        last_day=(closed[0]['day'] if closed else None),
        running_pnl=None, risk_open=None, to_stop=None,
        lifetime=dict(net=round(sum(vals)) if vals else 0, n=len(vals),
                      win=round(100 * sum(1 for x in vals if x > 0) / len(vals)) if vals else None,
                      maxdd=dd(vals), t=tstat(vals)),
        legs=[], curve=[], closed=closed, evidence=evid, rules=rules))




def _replay_state(today_pnl, dte_set):
    """Not-scheduled and not-yet-replayed are different facts; do not flatten them."""
    if today_pnl is not None:
        return ('Closed', 'neutral')
    if NIFTY_WD2DTE.get(date.today().weekday()) not in dte_set:
        return ('Not scheduled today', 'muted')
    return ('Replay · post-close', 'muted')

# --- V1 · one-and-done (naked, 0.4% move-stop) --------------------------------
v1 = _load('v1.json')
if v1 and v1.get('per_day'):
    pd_ = v1['per_day']
    days = sorted(pd_)
    vals, closed = [], []
    for d in days:
        r = pd_[d]
        fin = r.get('final')
        if fin is None and r.get('series'):
            fin = r['series'][-1][1]
        if fin is None:
            continue
        vals.append(float(fin))
        closed.append(dict(day=d, strike=r.get('strike'), credit=r.get('credit'),
                           exit=(r.get('exit') or {}).get('time') if isinstance(r.get('exit'), dict) else None,
                           reason=('MOVE-STOP' if r.get('stopped') else 'TIME'),
                           dte=r.get('dte'), pnl=round(float(fin))))
    t_today = next((round(float(c['pnl'])) for c in closed if c['day'] == TODAY), None)
    _add('V1_OAD', 'V1 · One-and-done', 'Naked straddle · 0.4% move-stop · one and done',
         'intraday', 'intraday', 'paper', v1.get('lots'), (v1.get('lots') or 0) * (v1.get('lot') or 65),
         '09:20 → 14:45 · Mon/Tue', vals, closed[::-1][:40],
         dict(does=[('Universe', 'NIFTY weekly ATM straddle, <b>naked</b> — no wings.'),
                    ('Entry', '~<b>09:20</b>, sell ATM CE + ATM PE.'),
                    ('Stop', f"<b>{v1.get('trigger_pct', 0.4)}% underlying move</b> from the entry "
                             "spot — measured on the index, not on premium."),
                    ('Exit', '<b>14:45</b>, or the move-stop.')],
              doesnt=['<b>One and done</b> — after the stop fires it does not re-enter, whatever '
                      'the rest of the day does.',
                      'No wings: the tail is open until the stop acts, which is why it is paper.',
                      'No per-leg stop and no adjustment.'],
              doc='research/58_intraday_recenter_straddle/scripts/v1_oad.py'),
         dict(method=['Our recorded chain'], period=f"{len(vals)} days",
              nums={}, how='Replayed on the 1-minute chain recorded since 20 Apr 2026.',
              caveat='<b>The backtest and the record above are the same days</b> — a replay, not '
                     'an out-of-sample test.',
              links=[('research/58', 'https://github.com/castroarun/Quantifyd/tree/main/research/58_intraday_recenter_straddle')]),
         state=_replay_state(t_today, {r.get('dte') for r in pd_.values()}),
         today=t_today)

# --- V1 + 30% combined-premium SL (the lab variant) ---------------------------
v1s = _load('v1_sl30.json')
if v1s and v1s.get('trades'):
    tr = sorted(v1s['trades'], key=lambda x: x['day'])
    vals = [float(t['final']) for t in tr if t.get('final') is not None]
    closed = [dict(day=t['day'][:10], dte=t.get('dte'), strike=None, credit=None,
                   exit=t.get('exit_time'),
                   reason=('SL 30%' if t.get('stopped') else 'TIME'),
                   pnl=round(float(t['final']))) for t in tr[::-1][:40]]
    stt = v1s.get('stats') or {}
    t_today = next((round(float(t['final'])) for t in tr if t['day'][:10] == TODAY), None)
    _add('V1_SL30', 'V1 variant · 30% combined-premium SL',
         'Same entry as V1, stopped on the pair rather than the move',
         'intraday', 'intraday', 'paper', v1s.get('lots'),
         (v1s.get('lots') or 0) * (v1s.get('lot') or 65), '09:20 → close · all DTEs', vals,
         closed,
         dict(does=[('Entry', 'Identical to V1 — ATM straddle at ~09:20.'),
                    ('Stop', f"<b>Combined premium +{v1s.get('sl_pct', 30):.0f}%</b> — the two legs "
                             "added and watched as one number, instead of V1's underlying move-stop."),
                    ('Exit', 'Held to the last bar unless the combined stop fires.')],
              doesnt=['<b>NOT DEPLOYED.</b> This is a lab variant on the recorded chain — there is '
                      'no live service running a 30% combined stop on the V1 book.',
                      'Exists to answer whether the stop should read the pair or the index.'],
              doc='research/58_intraday_recenter_straddle/scripts/sl30_journeys.py'),
         dict(method=['Our recorded chain'], period=f"{stt.get('n', len(vals))} days",
              nums={'Net @10 lots': f"+₹{stt.get('total', 0):,}" if stt.get('total', 0) > 0
                    else f"−₹{abs(stt.get('total', 0)):,}",
                    'Win': f"{stt.get('win', '—')}%", 'Max DD': f"−₹{abs(stt.get('maxdd', 0)):,}",
                    'SL hit': f"{stt.get('sl_hit_pct', '—')}%"},
              how='Re-priced from the untruncated 5-minute premium path of every recorded day.',
              caveat='Its DTE-3 cell is <b>stop-invariant from 20% upward</b> (0 of 18 stops fired) '
                     '— that cell is about the day, not the stop (research/138).',
              links=[('research/138', '/app/backtest/sensex-nifty-stop-by-dte')]),
         state=_replay_state(t_today, {t.get('dte') for t in tr}), today=t_today)

# --- V2 positional bi-weekly (the recorded-chain lab, NOT the live engine) -----
v2l = _load('v2_2.0.json') or _load('v2.json')
if v2l and v2l.get('trades'):
    tr = sorted(v2l['trades'], key=lambda x: x['entry_day'])
    vals = [float(t['pnl']) for t in tr if t.get('pnl') is not None]
    closed = [dict(day=t['entry_day'], exit=t.get('exit_day'), expiry=t.get('expiry'),
                   strike=t.get('strike'), credit=None, reason=t.get('exit_reason'),
                   pnl=round(float(t['pnl']))) for t in tr[::-1][:40]]
    _add('V2_LAB', 'V2 · Positional bi-weekly',
         f"Recorded-chain replay · {v2l.get('move_stop')}% move-stop · ±{v2l.get('wings')} wings",
         'positional', 'positional', 'paper', v2l.get('lots'),
         (v2l.get('lots') or 0) * (v2l.get('lot') or 65), 'Multi-day carry', vals, closed,
         dict(does=[('What it is', 'A <b>replay of the V2 structure over our recorded chain</b> — '
                                   'the lab the stop and wing sweeps were run in.'),
                    ('Structure', f"Short ATM straddle + ±{v2l.get('wings')} wings."),
                    ('Stop', f"{v2l.get('move_stop')}% underlying move."),
                    ('Target', f"{v2l.get('pt')}% of credit.")],
              doesnt=['<b>This is not the live engine.</b> V2 · Iron fly is the executor; this is '
                      'the replay used to choose its parameters.',
                      'Wings here are priced from the recorder, which goes stale far from the '
                      'money — read it as a straddle-behaviour probe, not a fly validation.'],
              doc='research/58_intraday_recenter_straddle/scripts/v2_curves.py'),
         dict(method=['Our recorded chain'], period=f"{len(vals)} cycles",
              nums={}, how='Replayed over the recorded chain, Apr 2026 onward.',
              caveat='Short window and stale far-OTM wing quotes (research/89). The long-sample '
                     'evidence for this structure is the AlgoTest / bhavcopy pair on the engine row.',
              links=[('V2 study', '/app/backtest/v2-nifty-ironfly-sl-vix')]),
         state=('Replay · post-close', 'muted'))


# ------------------------------------------------- tested, not trading (studies)
# Study RESULTS, not live feeds. They change when the study is re-run, never
# automatically. All three trade the FRONT weekly; the live engine trades the 2nd.
V2_CORE_DOES = [
    ('Structure', 'Short iron fly — SELL ATM CE + PE, BUY wings at <b>±2.0% of ATM</b>.'),
    ('Expiry', '<b>FRONT weekly</b> — the nearest. <b>Not</b> the 2nd-nearest the live '
               'engine trades.'),
    ('Entry', '<b>09:20</b>, 4 trading days before expiry (AlgoTest\'s maximum).'),
    ('Exit', '1 trading day before expiry, 15:15.'),
    ('Gate', 'India VIX ≥ 13 at entry, applied from the export\'s own VIX column.'),
]


def _study(key, name, sub, net, n, win, maxdd, calmar, method, period, how, caveat,
           extra_does, extra_doesnt, links):
    systems.append(dict(
        key=key, name=name, subtitle=sub, group='study', kind='positional', venue='NIFTY',
        money='study', size_lots=10, size_qty=650,
        window='09:20 · 4 TD → 1 TD', state=dict(label='Backtest only', tone='muted'),
        today_pnl=None, running_pnl=None, last_pnl=None, last_day=None,
        risk_open=None, to_stop=None,
        lifetime=dict(net=net, n=n, win=win, maxdd=maxdd, t=None),
        legs=[], curve=[], closed=[],
        evidence=dict(method=method, period=period,
                      nums={'Net @10 lots': ('+₹%s' % f'{net:,}') if net > 0 else ('−₹%s' % f'{abs(net):,}'),
                            'Calmar': calmar, 'Max DD': '−₹%s' % f'{abs(maxdd):,}',
                            'Trades': str(n)},
                      how=how, caveat=caveat, links=links),
        rules=dict(does=V2_CORE_DOES + extra_does,
                   doesnt=extra_doesnt,
                   doc='research/60_v2_straddle_optimization/'
                       'V2_BIWEEKLY_STRADDLE_ALGOTEST_OPTIMIZATION_SWEEP_STATUS.md')))


_study('STUDY_ALGOTEST', 'V2 · AlgoTest, full rules',
       'The tested spec — front weekly, stop + target',
       880110, 204, 56, -116834, '1.03', ['AlgoTest'], '7.3 yrs · 2019–2026',
       'AlgoTest\'s 1-minute positional engine, net of ₹20/order, taxes and 0.25% '
       'slippage (measured: 0.169% median half-spread over 3.47M recorded quotes).',
       '<b class="neg">⚠ Front weekly.</b> The live engine trades the 2nd-nearest — a lever '
       'the sweep listed and never ran, because AlgoTest caps entry at 4 TD (research/141). '
       '<b>The trade CSV was not retained</b>, so streaks and per-year cannot be recomputed.',
       [('Manage', '<b>2% underlying move-stop</b> · <b>+40% of credit</b> target · re-enter '
                    'after either.')],
       ['Not trading. This is the backtest the live book\'s parameters came from.',
        'Its per-trade export is missing — an AlgoTest re-run is the open item.'],
       [('V2 study', '/app/backtest/v2-nifty-ironfly-sl-vix'),
        ('re-run spec', 'https://github.com/castroarun/Quantifyd/tree/main/research/141_v2_bhav_pertrade')])

_study('STUDY_ALGOTEST_CPR', 'V2 · AlgoTest + CPR filter',
       'Best Calmar on the page — but an overlay, on a trade list we no longer hold',
       1100000, 147, None, -95000, '1.59', ['AlgoTest'], '7.3 yrs · 2019–2026',
       'The same AlgoTest run with our CPR skip applied <b>post-hoc to the exported '
       'trades</b> — not a native AlgoTest feature.',
       'Validated by a train/test split: the ≈0.12% threshold is chosen on one half and '
       'improves return AND drawdown on the other. <b>But it is an overlay on a CSV that was '
       'not kept</b>, so it cannot be re-checked, and it is still the front weekly.',
       [('Extra gate', 'Skip when the <b>prior-day CPR width &lt; 0.10%</b> of spot — '
                       'compression precedes expansion.')],
       ['Not trading.',
        'The filter is ours, applied after the fact; AlgoTest never ran it.'],
       [('V2 study', '/app/backtest/v2-nifty-ironfly-sl-vix')])

_study('STUDY_ARM_C', 'Our variant C · front weekly + CPR',
       'Our bhavcopy rebuild — the CONTROL arm, no stop',
       1516346, 213, 54, -344672, '0.59', ['Our bhavcopy'], '7.5 yrs · 2019–2026',
       'Rebuilt on real NSE bhavcopy closes with untraded contracts excluded and the ATM '
       'chosen from the entry-day OPEN (causal). Per-trade output retained, so streaks and '
       'the month grid are computable.',
       '<b>Arm C carries NO STOP.</b> It is a measurement control — it exists to isolate what '
       'the 2% stop does, and is <b>not a proposal to trade without one</b>. Its stopped twin '
       '(arm D) nets +₹9,52,919 at Calmar 0.32.',
       [('Extra gate', 'Skip when the prior-day CPR width &lt; 0.10% of spot.'),
        ('Stop', '<b>None</b> — this is the control.')],
       ['<b>Not a deployable configuration.</b> No stopless book is proposed.',
        'Held to DTE-1 regardless of what the index does in between.'],
       [('research/141', 'https://github.com/castroarun/Quantifyd/tree/main/research/141_v2_bhav_pertrade'), ('full dossier', 'https://claude.ai/code/artifact/a3487b7e-e6c4-4a88-8f7f-e006999915fe')])


# ---------------------------------------------------------------- write + report
GORDER = {'916': 0, 'intraday': 1, 'positional': 2, 'study': 3}
MORDER = {'real': 0, 'paper': 1, 'refuted': 2, 'study': 4}
systems.sort(key=lambda s: (GORDER.get(s.get('group'), 9), MORDER.get(s['money'], 3),
                            -abs(s['lifetime']['net'] or 0)))

# the replay feeds only rebuild after the close; say so rather than let the page
# imply these books sat out a session they simply have not been replayed for yet
_v1f = SA / 'v1.json'
_replay_through = None
if _v1f.exists():
    _d = json.loads(_v1f.read_text()).get('per_day') or {}
    _replay_through = max(_d) if _d else None

payload = dict(generated_at=datetime.now().isoformat()[:19], date=TODAY,
               replay_through=_replay_through,
               n=len(systems), systems=systems)
OUT.write_text(json.dumps(payload), encoding='utf-8')

print(f'wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)  ·  {len(systems)} systems\n')
print(f"{'key':24} {'group':11} {'money':8} {'lots':>5} {'window':>16} {'net':>11} {'n':>4} {'state'}")
print('-' * 108)
for s in systems:
    print(f"{s['key']:24} {s.get('group',''):11} {s['money']:8} {str(s['size_lots'] or '—'):>5} "
          f"{s['window']:>16} {s['lifetime']['net']:>11,} {s['lifetime']['n']:>4} "
          f"{s['state']['label']}")
miss = [s['key'] for s in systems if not s['rules']['does']]
if miss:
    print(f"\n  {len(miss)} system(s) still on the stub spec (no rules authored): {miss}")
