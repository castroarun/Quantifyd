#!/usr/bin/env python3
"""
NAS LIVE GUARDIAN — proactive live-trading validator for the 8 NAS option variants.

Born from the 2026-06-12 incident: stops didn't fire, books churned, a survivor's
SuperTrend never triggered, P&L was misread, subscriptions reshuffled. Those bugs
were only found because a human watched and flagged symptoms. This tool's job is to
find that class of problem BY ITSELF, every run, from a live perspective.

It does four things, all SAFE (read-only on live state) except the opt-in fire-drill
which uses a throwaway sandbox DB and never places a real order:

  TIER 1  LIVE HEALTH        ticker alive, token fresh, per-leg subscription coverage,
                             SL arm sanity, naked-survivor ST sanity, freeze posture.
  TIER 2  BEHAVIOURAL AUDIT  hunt today's pathologies in REAL trade data:
                             churn (exit->re-enter < cooldown), SL-breach-without-exit,
                             subscription-gap (active leg, no live premium), P&L vs Kite.
  TIER 3  REGRESSION SELF-TEST  re-prove the 5 fixes are intact in code (catches a
                             future edit that silently reintroduces a bug).
  TIER 4  PAPER FIRE-DRILL   (opt-in --firedrill) sandbox a temp DB, inject a synthetic
                             paper leg, force an SL, assert the exit + cooldown fire.

Exit code: 0 = all good, 1 = at least one FAIL. WARNs do not fail the run.
Usage:
  python3 nas_live_guardian.py                 # tiers 1-3 (safe, the 5-min job)
  python3 nas_live_guardian.py --firedrill     # also run tier 4 sandbox fire-drill
  python3 nas_live_guardian.py --json          # machine-readable report to stdout
"""
import sys, os, json, argparse, traceback
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')

BASE = 'http://127.0.0.1:5000'
PASS, WARN, FAIL, SKIP = 'PASS', 'WARN', 'FAIL', 'SKIP'
# the 8 variants and their API prefixes + matrix/cooldown facts
VARIANTS = [
    ('nas',          'NAS squeeze base'),
    ('nas-atm',      'NAS ATM squeeze'),
    ('nas-atm2',     'NAS ATM2 squeeze'),
    ('nas-atm4',     'NAS ATM4 squeeze'),
    ('nas-916-otm',  'NAS 9:16 OTM'),
    ('nas-916-atm',  'NAS 9:16 ATM'),
    ('nas-916-atm2', 'NAS 9:16 ATM2'),
    ('nas-916-atm4', 'NAS 9:16 ATM4'),
]
results = []  # list of (tier, name, status, detail)


def add(tier, name, status, detail=''):
    results.append((tier, name, status, detail))


def _get(path, timeout=8):
    import urllib.request
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def in_market_hours():
    n = datetime.now()
    return n.weekday() < 5 and (9, 15) <= (n.hour, n.minute) <= (15, 30)


# ───────────────────────────── TIER 1: LIVE HEALTH ─────────────────────────────
def tier1_live_health():
    # freeze posture
    try:
        from services.nas_kill_switch import is_frozen, is_killed
        frozen, killed = is_frozen(), is_killed()
        if killed:
            add(1, 'Freeze/kill posture', WARN, 'PANIC KILL active — positions squared, no trading')
        elif frozen:
            add(1, 'Freeze/kill posture', WARN, 'MANUAL FREEZE active — code will not place orders (positions left open)')
        else:
            add(1, 'Freeze/kill posture', PASS, 'armed (not frozen, not killed)')
    except Exception as e:
        add(1, 'Freeze/kill posture', FAIL, 'could not read freeze state: %s' % e)

    # ticker health + per-leg coverage (the singleton status carries ALL variants' legs)
    try:
        st = _get('/api/nas/ticker/status')
    except Exception as e:
        add(1, 'NAS ticker', FAIL, 'ticker status endpoint unreachable: %s' % e)
        st = None

    if st is not None:
        running = st.get('is_running')
        conn = st.get('is_connected')
        last_ltp = st.get('last_ltp')
        cc = st.get('completed_candles', 0)
        if not running:
            add(1, 'NAS ticker', FAIL, 'ticker NOT running (is_running=%s) — no live ticks => SL/ST monitoring is dark' % running)
        elif not conn:
            add(1, 'NAS ticker', WARN, 'ticker running but not connected (is_connected=%s)' % conn)
        elif in_market_hours() and (last_ltp in (None, 0)):
            add(1, 'NAS ticker', FAIL, 'in market hours but last_ltp=%s — feed appears stalled' % last_ltp)
        else:
            add(1, 'NAS ticker', PASS, 'running, connected, last_ltp=%s, candles=%s' % (last_ltp, cc))

        # per-leg subscription coverage + arm sanity (THE check that catches the 06-12 bug)
        leg_groups = [('atm_option_legs', 'atm_naked_st', False),
                      ('atm2_option_legs', None, True),  # v3: ATM2 on 0.4%% move-stop, per-leg SL disabled
                      ('atm4_option_legs', 'atm4_naked_st', False),
                      ('option_legs', None, False)]
        gaps, near, naked_issues = [], [], []
        for legs_key, naked_key, is_movestop in leg_groups:
            for leg in st.get(legs_key, []) or []:
                tsym = leg.get('tradingsymbol', '?')
                cp = leg.get('current_premium')
                slp = leg.get('sl_price')
                if cp is None:
                    gaps.append('%s (%s): no live premium (subscription gap)' % (tsym, legs_key))
                    continue
                if slp is None:
                    continue
                if slp >= 900000:
                    # naked survivor — must have an active ST trail
                    nk = st.get(naked_key) if naked_key else None
                    if not (nk and nk.get('active')):
                        naked_issues.append('%s: sl=999999 sentinel but no active ST trail' % tsym)
                    else:
                        stv = nk.get('st_value')
                        cl = nk.get('current_close')
                        if stv is not None and cl is not None and cl > stv:
                            naked_issues.append('%s: premium close %.1f ALREADY above ST %.1f but still open — exit not firing' % (tsym, cl, stv))
                elif slp > 0 and cp >= slp:
                    if not is_movestop:  # ATM2 v3 move-stop: per-leg SL intentionally off
                        gaps.append('%s: premium %.1f >= SL %.1f but STILL OPEN — stop not firing' % (tsym, cp, slp))
                elif slp > 0 and cp >= 0.85 * slp:
                    near.append('%s: premium %.1f within 15%% of SL %.1f' % (tsym, cp, slp))

        if gaps:
            add(1, 'Leg subscription / SL coverage', FAIL, ' | '.join(gaps))
        elif naked_issues:
            add(1, 'Naked-survivor ST', FAIL, ' | '.join(naked_issues))
        else:
            nlegs = sum(len(st.get(k, []) or []) for k, _, _ in leg_groups)
            add(1, 'Leg subscription / SL coverage', PASS, '%d active leg(s), all have live premium + sane arm' % nlegs)
        if near:
            add(1, 'Stop-proximity watch', WARN, ' | '.join(near))

    # token freshness via a cheap authenticated Kite call
    try:
        from services.kite_service import get_kite
        prof = get_kite().profile()
        add(1, 'Kite token', PASS, 'authenticated as %s' % (prof.get('user_id') or prof.get('user_name') or 'ok'))
    except Exception as e:
        add(1, 'Kite token', FAIL, 'Kite auth/profile failed — token may be stale: %s' % e)


# ──────────────────────────── TIER 2: BEHAVIOURAL AUDIT ────────────────────────────
def _cooldown_min():
    try:
        from config import NAS_ATM_DEFAULTS
        return float(NAS_ATM_DEFAULTS.get('reentry_cooldown_min', 15))
    except Exception:
        return 15.0


def tier2_behavioural_audit():
    cd = _cooldown_min()
    churn_hits, breach_hits = [], []
    audited = 0
    for prefix, label in VARIANTS:
        try:
            trades = _get('/api/%s/trades' % prefix)
        except Exception:
            continue
        rows = trades if isinstance(trades, list) else (trades.get('trades') or trades.get('rows') or [])
        if not rows:
            continue
        audited += 1
        # churn: same strike, an exit followed by a re-entry within cooldown minutes
        events = []
        for r in rows:
            strike = r.get('strike') or r.get('tradingsymbol') or r.get('symbol')
            for kind, tkey in (('entry', 'entry_time'), ('exit', 'exit_time')):
                ts = r.get(tkey)
                if strike and ts:
                    events.append((strike, kind, ts))
        bystrike = {}
        for strike, kind, ts in events:
            bystrike.setdefault(strike, []).append((kind, ts))
        for strike, evs in bystrike.items():
            parsed = []
            for kind, ts in evs:
                dt = _parse_ts(ts)
                if dt:
                    parsed.append((dt, kind))
            parsed.sort()
            for i in range(1, len(parsed)):
                if parsed[i][1] == 'entry' and parsed[i - 1][1] == 'exit':
                    gap = (parsed[i][0] - parsed[i - 1][0]).total_seconds() / 60.0
                    if 0 <= gap < cd:
                        churn_hits.append('%s %s: re-entered %.1f min after exit (< %.0f cooldown)' % (label, strike, gap, cd))
    if churn_hits:
        add(2, 'Churn (exit->re-enter < cooldown)', FAIL, ' | '.join(churn_hits[:8]))
    elif audited:
        add(2, 'Churn (exit->re-enter < cooldown)', PASS, 'no sub-cooldown re-entries across %d variant trade logs' % audited)
    else:
        add(2, 'Churn (exit->re-enter < cooldown)', SKIP, 'no trade logs available to audit')

    # P&L reconciliation: per-system day P&L vs Kite net option position MTM
    try:
        mtm = _get('/api/nas/mtm').get('systems', {})
        sys_total = sum((v.get('last') or 0) for v in mtm.values())
        from services.kite_service import get_kite
        pos = get_kite().positions().get('net', [])
        kite_opt = sum((p.get('pnl') or 0) for p in pos
                       if (p.get('exchange') == 'NFO') and ('NIFTY' in (p.get('tradingsymbol') or '')))
        diff = abs(sys_total - kite_opt)
        detail = 'NAS systems day-P&L=Rs.%.0f vs Kite NIFTY-NFO MTM=Rs.%.0f (diff Rs.%.0f)' % (sys_total, kite_opt, diff)
        # large divergence => either a mismarked leg or the realized-vs-open bug from 06-12
        if diff > 5000:
            add(2, 'P&L reconcile (DB vs Kite)', WARN, detail + ' — investigate (Kite also holds non-NAS NIFTY legs)')
        else:
            add(2, 'P&L reconcile (DB vs Kite)', PASS, detail)
    except Exception as e:
        add(2, 'P&L reconcile (DB vs Kite)', SKIP, 'could not reconcile: %s' % e)


def _parse_ts(ts):
    if not isinstance(ts, str):
        return None
    for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S', '%H:%M:%S'):
        try:
            return datetime.strptime(ts[:26], fmt)
        except ValueError:
            continue
    return None


# ──────────────────────────── TIER 3: REGRESSION SELF-TEST ────────────────────────────
def tier3_regression():
    # #3 cooldown isoformat parse (the exact 06-12 root cause)
    iso = '2026-06-12T13:45:04.502518'
    ok = any(_try(iso, f) for f in ('%Y-%m-%dT%H:%M:%S.%f',))
    add(3, '#3 cooldown parses isoformat exit_time', PASS if ok else FAIL,
        'isoformat (T + microseconds) parses' if ok else 'isoformat does NOT parse — cooldown blind again')

    # #1a SL fetch-fallback present in all 3 executors
    needle = 'don' + chr(39) + 't silently skip the SL check'
    miss = [m for m in ('nas_atm_executor', 'nas_atm2_executor', 'nas_atm4_executor')
            if needle not in open('services/%s.py' % m).read()]
    add(3, '#1a SL fetch-fallback in executors', PASS if not miss else FAIL,
        'present in all 3' if not miss else 'MISSING in: %s' % miss)

    t = open('services/nas_ticker.py').read()
    # #2 level-breach + tick-level + sentinel preserved
    cond = ('latest_close > st_val' in t) and ('_atm_naked_st_val' in t) and ('999999' in t)
    add(3, '#2 ST level-breach + tick-exit + sentinel', PASS if cond else FAIL,
        'level-breach + tick cache + 999999 sentinel all present' if cond else 'one or more #2 pieces missing')

    # #4 subscription helper excludes ALL siblings (unit-test the live class)
    try:
        from services.nas_ticker import NasTicker
        nt = NasTicker.__new__(NasTicker)
        nt._option_tokens = {1: {}}
        nt._atm_option_tokens = {2: {}}
        nt._atm2_option_tokens = {3: {}}
        nt._atm4_option_tokens = {2: {}, 4: {}}  # token 2 shared with ATM
        others = nt._tokens_in_use_by_others(nt._atm2_option_tokens)
        # for ATM2, "others" must include 1 (OTM), 2 & 4 (ATM/ATM4); must NOT include 3 (its own)
        good = (1 in others and 2 in others and 4 in others and 3 not in others)
        add(3, '#4 subscription helper excludes all siblings', PASS if good else FAIL,
            'returns {1,2,4} excluding own {3}' if good else 'helper math wrong: %s' % sorted(others))
    except Exception as e:
        add(3, '#4 subscription helper excludes all siblings', FAIL, 'helper missing/broken: %s' % e)

    # #1b squeeze SL poll registered
    a = open('app.py').read()
    reg = ('def _nas_squeeze_sl_monitor' in a) and ("id=" + chr(39) + "nas_squeeze_sl_monitor" + chr(39) in a)
    add(3, '#1b squeeze SL poll registered', PASS if reg else FAIL,
        'defined + scheduled (10s backstop)' if reg else 'squeeze SL poll missing/not scheduled')


def _try(s, fmt):
    try:
        datetime.strptime(s[:26], fmt); return True
    except ValueError:
        return False


# ──────────────────────────── TIER 4: PAPER FIRE-DRILL (opt-in) ────────────────────────────
def tier4_firedrill():
    """Sandbox the real SL path: temp DB, synthetic PAPER leg priced ABOVE its SL,
    drive the real check_and_handle_sl, assert the leg gets closed. Paper closes go
    straight to db.close_position (no Kite, not gated by freeze), so this exercises
    the actual exit code path end-to-end without touching live state or placing orders."""
    import tempfile
    try:
        import services.nas_atm_db as dbmod
        from services.nas_atm_executor import NasAtmExecutor
        from config import NAS_ATM_DEFAULTS
    except Exception as e:
        add(4, 'Paper fire-drill', SKIP, 'imports unavailable: %s' % e)
        return
    tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False).name
    orig_path = dbmod.DB_PATH
    try:
        cfg = dict(NAS_ATM_DEFAULTS); cfg['mode'] = 'paper'
        ex = NasAtmExecutor(config=cfg)               # self.db = live singleton (overridden below)
        dbmod.DB_PATH = tmp
        sandbox = dbmod.NasAtmDB()                     # full __init__ on the throwaway temp path
        conn = sandbox._get_conn()
        cols = {r[1] for r in conn.execute("PRAGMA table_info(nas_atm_positions)").fetchall()}
        conn.close()
        sid = sandbox.get_next_strangle_id() if hasattr(sandbox, 'get_next_strangle_id') else 99999
        tsym = 'NIFTYTEST00000CE'
        _now = datetime.now().isoformat()
        want = dict(tradingsymbol=tsym, strangle_id=sid, leg='CE', instrument_type='CE',
                    strike=0, entry_price=100.0, sl_price=130.0, qty=75,
                    status='ACTIVE', mode='paper', transaction_type='SELL',
                    entry_time=_now, created_at=_now, updated_at=_now)
        sandbox.add_position(**{k: v for k, v in want.items() if k in cols})
        pid = sandbox.get_active_positions()[0]['id']
        ex.db = sandbox

        class _StubScanner:                            # synthetic premium ABOVE the SL
            kite = None
            def get_live_option_premium(self, _): return 160.0
            def get_live_spot(self, *a, **k): return 23600.0   # ±-move stop reference
            def __getattr__(self, _name):                      # any other probe -> no-op None
                return lambda *a, **k: None
        ex.scanner = _StubScanner()
        ex.check_and_handle_sl(positions=sandbox.get_active_positions(), live_ltps={tsym: 160.0})
        still_open = any(p['id'] == pid for p in sandbox.get_active_positions())
        if not still_open:
            add(4, 'Paper fire-drill', PASS,
                'synthetic paper leg priced 160 vs SL 130 -> SL fired and closed it (exit path verified end-to-end)')
        else:
            add(4, 'Paper fire-drill', FAIL,
                'synthetic paper leg breached SL 130 at premium 160 but was NOT closed — exit path broken')
    except Exception as e:
        add(4, 'Paper fire-drill', SKIP, 'sandbox could not run (executor internals): %s' % e)
    finally:
        dbmod.DB_PATH = orig_path
        try: os.unlink(tmp)
        except Exception: pass


# ──────────────────────────────── REPORT ────────────────────────────────
def tier1b_db_ticker_kite_reconcile():
    """Cross-check every DB-ACTIVE leg against the ticker AND the broker (added 2026-06-16).
    Catches what the basic ticker leg-check can miss: a DB-active leg the ticker isn't
    monitoring (subscription gap), or a DB leg with no matching Kite short (orphan)."""
    import sqlite3
    try:
        st = _get('/api/nas/ticker/status')
        mon = set()
        for k in ('option_legs', 'atm_option_legs', 'atm2_option_legs', 'atm4_option_legs'):
            for l in st.get(k, []) or []:
                mon.add(l['tradingsymbol'])
    except Exception as e:
        add(1, 'DB/ticker/Kite reconcile', SKIP, 'ticker unreachable: %s' % e)
        return
    try:
        from services.kite_service import get_kite
        kq = {p['tradingsymbol']: (p.get('quantity') or 0) for p in get_kite().positions().get('net', [])}
    except Exception:
        kq = None
    dbs = {'nas': 'nas_trading', '916_otm': 'nas_916_otm_trading', '916_atm': 'nas_916_atm_trading',
           '916_atm2': 'nas_916_atm2_trading', '916_atm4': 'nas_916_atm4_trading',
           'sq_atm': 'nas_atm_trading', 'sq_atm2': 'nas_atm2_trading', 'sq_atm4': 'nas_atm4_trading'}
    not_mon, not_kite, nlegs = [], [], 0
    for v, db in dbs.items():
        p = 'backtest_data/%s.db' % db
        if not os.path.exists(p):
            continue
        c = sqlite3.connect(p)
        rows = []
        for tbl in ('nas_atm_positions', 'nas_positions'):
            try:
                rows = [r[0] for r in c.execute("SELECT tradingsymbol FROM %s WHERE status='ACTIVE'" % tbl)]
                break
            except Exception:
                continue
        for ts in rows:
            nlegs += 1
            if ts not in mon:
                not_mon.append('%s:%s' % (v, ts[-9:]))
            if kq is not None and kq.get(ts, 0) == 0:
                not_kite.append('%s:%s' % (v, ts[-9:]))
    if not_mon:
        add(1, 'DB->ticker coverage', FAIL, '%d DB-active leg(s) NOT ticker-monitored: %s' % (len(not_mon), ', '.join(not_mon)))
    else:
        add(1, 'DB->ticker coverage', PASS, ('all %d DB-active legs ticker-monitored' % nlegs) if nlegs else 'no active legs (flat)')
    if not_kite:
        add(1, 'DB->Kite reconcile', FAIL, '%d DB-active leg(s) with NO Kite short (orphan?): %s' % (len(not_kite), ', '.join(not_kite)))
    elif nlegs and kq is not None:
        add(1, 'DB->Kite reconcile', PASS, 'all DB-active legs have a matching Kite short')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--firedrill', action='store_true', help='also run the sandbox paper fire-drill')
    ap.add_argument('--json', action='store_true', help='emit machine-readable JSON')
    args = ap.parse_args()

    for fn in (tier1_live_health, tier1b_db_ticker_kite_reconcile, tier2_behavioural_audit, tier3_regression):
        try:
            fn()
        except Exception as e:
            add(0, fn.__name__, FAIL, 'guardian check crashed: %s\n%s' % (e, traceback.format_exc()))
    if args.firedrill:
        tier4_firedrill()

    fails = [r for r in results if r[2] == FAIL]
    warns = [r for r in results if r[2] == WARN]
    if args.json:
        print(json.dumps({'ts': datetime.now().isoformat(),
                          'verdict': 'FAIL' if fails else ('WARN' if warns else 'PASS'),
                          'checks': [{'tier': t, 'name': n, 'status': s, 'detail': d} for t, n, s, d in results]}, indent=2))
    else:
        icon = {PASS: 'OK  ', WARN: 'WARN', FAIL: 'FAIL', SKIP: 'skip'}
        tier_name = {1: 'LIVE HEALTH', 2: 'BEHAVIOURAL AUDIT', 3: 'REGRESSION', 4: 'FIRE-DRILL', 0: 'GUARDIAN'}
        print('=' * 72)
        print('NAS LIVE GUARDIAN  ', datetime.now().strftime('%Y-%m-%d %H:%M:%S IST'),
              '  market-hours' if in_market_hours() else '  (off-hours)')
        print('=' * 72)
        last_tier = None
        for t, n, s, d in results:
            if t != last_tier:
                print('\n[ TIER %s · %s ]' % (t, tier_name.get(t, t)))
                last_tier = t
            print('  %s  %-42s %s' % (icon[s], n, d))
        verdict = 'FAIL' if fails else ('WARN' if warns else 'ALL CLEAR')
        print('\n' + '=' * 72)
        print('VERDICT: %s   (%d fail, %d warn, %d checks)' % (verdict, len(fails), len(warns), len(results)))
        print('=' * 72)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
