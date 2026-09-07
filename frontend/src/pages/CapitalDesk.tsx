import { useEffect, useState } from 'react';
import { apiGet } from '../api/client';
import styles from './BlueskyPaper.module.css';
import LiveTick from '../components/LiveTick/LiveTick';

/* CAPITAL DESK (/app/capital) — the one page that owns every rupee in and out.
   Renamed from "Sleeves 50-50" on 05-Sep-2026: the book is three systems on a
   TN 40 / OA 40 / IPO 20 target, so a name describing a two-way even split had
   stopped being true.

   It carries the target allocation and its drift, the deposit router, deposits and
   withdrawals for every book, the dividend policy, and the comparison of the sleeves
   over their common history. Money controls live HERE and not on the book pages, so
   there is exactly one place to look and one path to audit.

   Every flow dispatches to the book's own hardened implementation; this page derives
   no split of its own and touches no trading logic. */

type MomNav = { d: string; nav: number; bench: number | null };
type MomState = { navcurve: MomNav[]; nav: number; capital: number; total_return_pct: number;
  gate: string; mode?: string; inception?: string };
type BsNav = { date: string; nav: number; bench: number | null };
type BsFeed = {
  nav_curve: BsNav[]; nav: number; capital: number; ret_pct: number; gate_weak: boolean;
  cagr_pct: number; max_dd_pct: number; n_trades: number; win_pct: number;
  n_live_trades: number; provenance: string; study: string;
};
/* The TIME-WEIGHTED curve. True North was funded from Rs2.98L to Rs9.07L inside this
   window, so raw NAV is not a return series -- book_curve() backs each day's flow out
   before chaining, which is the only reason the two sleeves can share an axis. */
type TnBook = { d: string; nav: number; r: number };
type TnBench = { book: TnBook[]; inception: string; series: Record<string, unknown> };

const pct = (n: number | null | undefined) =>
  n == null ? '—' : (n >= 0 ? '+' : '') + n.toFixed(1) + '%';

function monthKey(d: string) { return d.slice(0, 7); }

function blend5050(dates: string[], a: number[], b: number[]) {
  /* 50-50, rebalanced at each month boundary. */
  const out: number[] = [];
  let wA = 0.5, wB = 0.5, base = 100;
  let aRef = a[0], bRef = b[0], lastM = monthKey(dates[0]);
  for (let i = 0; i < dates.length; i++) {
    const m = monthKey(dates[i]);
    if (m !== lastM) {
      base = out[i - 1];
      aRef = a[i - 1]; bRef = b[i - 1];
      wA = 0.5; wB = 0.5; lastM = m;
    }
    out.push(base * (wA * (a[i] / aRef) + wB * (b[i] / bRef)));
  }
  return out;
}

function corrMonthly(dates: string[], a: number[], b: number[]) {
  const idx: number[] = [];
  for (let i = 1; i < dates.length; i++)
    if (monthKey(dates[i]) !== monthKey(dates[i - 1])) idx.push(i - 1);
  idx.push(dates.length - 1);
  const ra: number[] = [], rb: number[] = [];
  for (let k = 1; k < idx.length; k++) {
    ra.push(a[idx[k]] / a[idx[k - 1]] - 1);
    rb.push(b[idx[k]] / b[idx[k - 1]] - 1);
  }
  const mean = (v: number[]) => v.reduce((x, y) => x + y, 0) / v.length;
  const ma = mean(ra), mb = mean(rb);
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < ra.length; i++) {
    num += (ra[i] - ma) * (rb[i] - mb);
    da += (ra[i] - ma) ** 2; db += (rb[i] - mb) ** 2;
  }
  return da && db ? num / Math.sqrt(da * db) : null;
}

function stats(series: number[], dates: string[]) {
  const yrs = (Date.parse(dates[dates.length - 1]) - Date.parse(dates[0])) / 3.15576e10;
  const cagr = (Math.pow(series[series.length - 1] / series[0], 1 / yrs) - 1) * 100;
  let peak = series[0], dd = 0;
  for (const v of series) { peak = Math.max(peak, v); dd = Math.min(dd, v / peak - 1); }
  return { cagr, dd: dd * 100, total: (series[series.length - 1] / series[0] - 1) * 100 };
}

function MultiCurve({ dates, lines }: { dates: string[]; lines: { name: string; v: number[]; color: string; dash?: string }[] }) {
  const W = 780, H = 260, P = 8;
  const all = lines.flatMap((l) => l.v);
  const lo = Math.min(...all), hi = Math.max(...all);
  const x = (i: number) => P + (i / (dates.length - 1)) * (W - 2 * P);
  const y = (v: number) => P + (1 - (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo) || 1)) * (H - 2 * P);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={styles.chart} preserveAspectRatio="none" style={{ height: 260 }}>
      {lines.map((l) => (
        <path key={l.name} fill="none" stroke={l.color} strokeWidth={l.name.includes('blend') ? 2.4 : 1.4}
              strokeDasharray={l.dash} d={l.v.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')} />
      ))}
    </svg>
  );
}

type Flow = { ts: string; kind: string; amount: number; via?: string };
type BookStatus = { name: string; kind?: string; capital?: number | null; cash?: number | null;
  liquid?: number | null; nav?: number | null; positions?: number;
  flows?: Flow[]; error?: string; note?: string };
type AllocRow = { book: string; value: number; target_pct: number; current_pct: number;
  target_value: number; gap: number };
type Allocation = { total: number; base: string; ipo_status: string; rows: AllocRow[];
  changelog: { date: string; text: string }[] };
type FlowsStatus = { books: Record<string, BookStatus>; allocation: Allocation; note: string };
type RouteLeg = { book: string; amount: number };
type RoutePlan = { amount: number; legs: RouteLeg[]; notes: string[] };

const BOOK_LABEL: Record<string, string> = {
  truenorth: 'True North', openalpha: 'Open Alpha', ipo: 'IPO base',
};

function AllocationPanel({ a }: { a: Allocation }) {
  return (
    <div className={styles.card}>
      <div className={styles.cardTitle}>
        Target allocation — {a.rows.map((r) => `${BOOK_LABEL[r.book] ?? r.book} ${r.target_pct}`).join(' / ')}
      </div>
      <div className={styles.sub} style={{ marginBottom: 10 }}>
        {BOOK_LABEL[a.base] ?? a.base} is the base: it is never sold to rebalance. Arriving cash
        goes to whichever book is furthest below its share.
        {a.ipo_status === 'paper' && ' IPO is on paper, so its share is earmarked in the liquid ETF.'}
      </div>
      <table className={styles.tbl}>
        <thead>
          <tr><th className={styles.txt}>Book</th><th>Value</th><th>Now</th>
            <th>Target</th><th>Target ₹</th><th>Gap</th></tr>
        </thead>
        <tbody>
          {a.rows.map((r) => (
            <tr key={r.book}>
              <td className={styles.txt}>{BOOK_LABEL[r.book] ?? r.book}</td>
              <td>{rup(r.value)}</td>
              <td>{r.current_pct}%</td>
              <td className={styles.muted}>{r.target_pct}%</td>
              <td className={styles.muted}>{rup(r.target_value)}</td>
              <td className={r.gap >= 0 ? styles.pos : styles.neg}>
                {r.gap >= 0 ? '+' : '−'}{rup(Math.abs(r.gap)).slice(1)}
              </td>
            </tr>
          ))}
          <tr>
            <td className={styles.txt}><b>Total</b></td>
            <td><b>{rup(a.total)}</b></td>
            <td colSpan={4} className={styles.muted}>
              a positive gap is money the book still needs
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function AllocationDesk() {
  const [a, setA] = useState<Allocation | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    apiGet<Allocation>('/api/sleeves/allocation').then(setA)
      .catch((e) => setErr(String(e)));
  }, []);
  if (err) return (
    <div className={styles.card}>
      <div className={styles.cardTitle}>Target allocation</div>
      <p className={styles.note}>unavailable: {err}</p>
    </div>
  );
  if (!a) return null;
  return <AllocationPanel a={a} />;
}

function FundsPanel() {
  const [st, setSt] = useState<FlowsStatus | null>(null);
  const [amt, setAmt] = useState('');
  const [kind, setKind] = useState<'deposit' | 'withdraw'>('deposit');
  const [target, setTarget] = useState<'auto' | 'truenorth' | 'openalpha' | 'ipo'>('auto');
  const [route, setRoute] = useState<RoutePlan | null>(null);
  const [plans, setPlans] = useState<any[] | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const load = () => apiGet<FlowsStatus>('/api/sleeves/status').then(setSt)
    .catch((e) => setMsg('status unavailable: ' + String(e)));
  useEffect(() => { load(); }, []);

  const call = (url: string, body: any) =>
    fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                 body: JSON.stringify(body), credentials: 'include' })
      .then(async (r) => ({ ok: r.ok, data: await r.json().catch(() => ({})) }));

  /* Legs come from the SERVER's router when target is auto, so the split that is
     previewed is the split that executes — the page never re-derives it. */
  const buildLegs = async (n: number): Promise<RouteLeg[]> => {
    if (target !== 'auto') return [{ book: target, amount: n }];
    const r = await call('/api/sleeves/allocation/route', { amount: n });
    if (!r.ok) throw new Error(r.data?.error || 'router failed');
    setRoute(r.data as RoutePlan);
    return (r.data as RoutePlan).legs;
  };

  const preview = async () => {
    const n = Number(amt);
    if (!n || n <= 0) { setMsg('enter a positive amount'); return; }
    setBusy(true); setMsg(null); setPlans(null); setRoute(null);
    try {
      const legs = await buildLegs(n);
      const out = [];
      for (const l of legs) {
        const r = await call('/api/sleeves/' + l.book + '/' + kind,
                             { amount: Math.round(l.amount), dry_run: true });
        out.push({ book: l.book, amount: l.amount, ...r });
      }
      setPlans(out);
    } catch (e) { setMsg(String(e)); }
    setBusy(false);
  };

  /* Multi-leg safety. Previously two legs fired sequentially under a single confirm, so
     a failure on the second left the first applied with no record and no reversal. Now
     every leg must pass its own dry run BEFORE anything executes, and if a leg still
     fails mid-flight we stop immediately and report exactly what was applied and what
     was not. The books are separate stores, so true atomicity is not available —
     pretending otherwise would be worse than saying so plainly. */
  const execute = async () => {
    if (!plans || !plans.length) return;
    const bad = plans.filter((p) => !p.ok || p.data?.feasible === false);
    if (bad.length) {
      setMsg('cannot execute — ' + bad.map((b) => BOOK_LABEL[b.book] ?? b.book).join(', ')
             + ' failed the dry run. Nothing was sent.');
      return;
    }
    const n = Number(amt);
    const live = plans.some((p) => p.book === 'truenorth');
    const warn = live ? 'True North is a LIVE book and its leg may place REAL orders.\n\n' : '';
    const lines = plans.map((p) => '  ' + (BOOK_LABEL[p.book] ?? p.book) + ': Rs '
      + Math.round(p.amount).toLocaleString('en-IN')).join('\n');
    if (!window.confirm(warn + kind + ' Rs ' + n.toLocaleString('en-IN')
        + ' split as:\n' + lines + '\n\nProceed?')) return;
    setBusy(true);
    const applied: string[] = [], skipped: string[] = [];
    let halted = false;
    for (const p of plans) {
      const label = BOOK_LABEL[p.book] ?? p.book;
      if (halted) { skipped.push(label); continue; }
      const r = await call('/api/sleeves/' + p.book + '/' + kind,
                           { amount: Math.round(p.amount), dry_run: false })
        .catch((e) => ({ ok: false, data: { error: String(e) } }));
      if (r.ok) applied.push(label + ' ' + rup(p.amount));
      else { halted = true; skipped.push(label + ' FAILED: ' + (r.data?.error || 'error')); }
    }
    setMsg(halted
      ? 'PARTIAL — applied: ' + (applied.join(', ') || 'nothing') + ' · NOT applied: '
        + skipped.join(', ') + '. Reverse the applied legs manually to undo the whole flow.'
      : 'Done — ' + applied.join(' · '));
    setPlans(null); setRoute(null); setAmt(''); setBusy(false); load();
  };

  const books = st?.books ?? {};
  const sel: React.CSSProperties = { padding: '7px 10px', borderRadius: 6,
    border: '1px solid var(--hairline, #ccc)', background: 'var(--surface)',
    color: 'var(--ink)', fontSize: 13 };
  return (
    <div className={styles.card}>
      <div className={styles.cardTitle}>Funds — every rupee in and out</div>
      <div className={styles.sub} style={{ marginBottom: 10 }}>
        {Object.entries(books).filter(([k]) => k !== 'openalpha_model').map(([k, b]) => (
          <span key={k} style={{ marginRight: 14 }}>
            <b>{b.name}</b>{' '}
            {b.error ? <span className={styles.neg}>unavailable</span>
              : <>capital {rup(b.capital)} · liquid {rup(b.liquid ?? b.cash)}</>}
          </span>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <select value={kind} onChange={(e) => { setKind(e.target.value as any); setPlans(null); }} style={sel}>
          <option value="deposit">Deposit</option>
          <option value="withdraw">Withdraw</option>
        </select>
        <select value={target} onChange={(e) => { setTarget(e.target.value as any); setPlans(null); }} style={sel}>
          <option value="auto">Route to target (40/40/20)</option>
          <option value="truenorth">True North only</option>
          <option value="openalpha">Open Alpha only</option>
          <option value="ipo">IPO base only</option>
        </select>
        <input value={amt} onChange={(e) => { setAmt(e.target.value); setPlans(null); }}
               placeholder="amount Rs" style={{ ...sel, width: 130 }} />
        <button style={{ ...sel, cursor: 'pointer', fontWeight: 600 }} disabled={busy} onClick={preview}>Preview</button>
        {plans && <button style={{ ...sel, cursor: 'pointer', fontWeight: 700 }} disabled={busy} onClick={execute}>Confirm &amp; execute</button>}
      </div>
      {route && route.notes.map((n, i) => (
        <p key={i} className={styles.note} style={{ marginBottom: 2 }}>{n}</p>
      ))}
      {plans && plans.map((pl, i) => (
        <p key={i} className={styles.note}>
          <b>{BOOK_LABEL[pl.book] ?? pl.book} — {rup(pl.amount)}:</b>{' '}
          {pl.ok
            ? (Array.isArray(pl.data.plan)
                ? pl.data.plan.map((s: any) => typeof s === 'string' ? s
                    : [s.action, s.qty, s.source ?? s.symbol, '(' + rup(s.value) + ')']
                        .filter(Boolean).join(' ')).join(' → ')
                : JSON.stringify(pl.data).slice(0, 220))
            : (pl.data.error || 'preview failed')}
        </p>
      ))}
      {msg && <p className={styles.note}><b>{msg}</b></p>}
      {Object.entries(books).filter(([, b]) => (b.flows ?? []).length).map(([k, b]) => (
        <p key={k} className={styles.note}>
          {b.name} flows: {(b.flows ?? []).slice(-4).map((f) =>
            f.kind + ' ' + rup(f.amount) + ' (' + String(f.ts).slice(0, 10) + ')').join(' · ')}
        </p>
      ))}
      <p className={styles.note}>
        Preview → confirm → execute, the same contract True North's own cash panel uses. Each leg
        runs that book's hardened flow: positions are never force-sold, and a withdrawal that
        cannot be funded is refused rather than partially filled.
      </p>
    </div>
  );
}

type DivBook = { book: string; initialized: boolean; note?: string; hwm?: number;
  cap?: number | null; reserve?: number; ledger?: any[] };
type DivStatus = { policy: { baseline: number; cap_growth_q: number; reserve_rate_pa: number };
  truenorth: DivBook; openalpha: DivBook };

function rup(n: number | null | undefined) {
  return n == null ? '—' : `₹${Math.round(n).toLocaleString('en-IN')}`;
}

function HowItWorksModal({ onClose }: { onClose: () => void }) {
  return (
    <div onClick={onClose}
      style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)', zIndex: 60,
               display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16 }}>
      <div onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: 640, maxHeight: '85vh', overflowY: 'auto', borderRadius: 10,
                 background: 'var(--surface, #16181d)', color: 'var(--ink, #e8e8e8)',
                 border: '1px solid var(--hairline, #333)', padding: '22px 26px',
                 fontSize: 13.5, lineHeight: 1.65 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ fontWeight: 700, fontSize: 15 }}>Where the 25% carve-out comes in</div>
          <button onClick={onClose}
            style={{ border: 'none', background: 'transparent', color: 'inherit',
                     fontSize: 20, cursor: 'pointer', lineHeight: 1 }}>×</button>
        </div>
        <p><b>It comes in nowhere in the trading loop — and that's deliberate.</b> Both
        engines size positions as a % of current NAV: a closed trade's profit lands in
        cash, sweeps into CASHIETF, and the next entry is sized off the bigger book. The
        engines have no concept of "distributable."</p>
        <ol style={{ paddingLeft: 20, margin: '10px 0' }}>
          <li style={{ marginBottom: 8 }}><b>Between record dates (91 days at a time):
          nothing changes.</b> The engine trades the full book and 100% of booked profit
          reinvests, exactly as coded. No per-trade skimming — that would starve
          compounding and add churn.</li>
          <li style={{ marginBottom: 8 }}><b>On the quarter-end record date only</b>, the
          dividend engine (a separate 19:15 cron) does the accounting: NAV vs the
          high-water mark — flow-adjusted, so your own deposits never count as "profit" —
          then entitlement = 25% of the excess, then the smoothed cap (last dividend
          +7.5%/qtr; surplus banks into the equalization reserve).</li>
          <li style={{ marginBottom: 8 }}><b>The money physically leaves the way a
          withdrawal does</b>: from cash + CASHIETF redemption only. Positions are never
          force-sold. The paid amount goes to the distribution pool (the notice carries
          the Zerodha Console amount for the bank leg); the reserve sits in its own
          liquid pocket outside book NAV.</li>
          <li style={{ marginBottom: 8 }}><b>From the next cycle the engine simply sizes
          off the smaller NAV.</b> To the trading loop a dividend is indistinguishable
          from a withdrawal you made yourself — which is why no engine code was touched.</li>
          <li><b>Edge rule:</b> if the book is fully deployed and liquid cash is less
          than the entitlement, the outflow is clipped to what's liquid — capital is
          never invaded and nothing is ever force-sold to pay a dividend.</li>
        </ol>
        <p style={{ opacity: 0.75, marginBottom: 0 }}>Policy evidence: research/142
        <code> dividend_sim_v2.py</code> variant E — 10-yr sim on ₹10L: ₹21.7L paid,
        ending NAV ₹1.14Cr + ₹15.6L reserve, 24 consecutive rising quarterly payouts
        2020-Q4 → 2026 through two drawdowns.</p>
      </div>
    </div>
  );
}

type SimRow = { year: number; q: (number | null)[]; src: string[]; profit: number;
  total: number; reserve: number };
type SimFeed = { policy: string; seed_capital: number; rows: SimRow[]; total_paid: number;
  total_profit: number; end_nav: number; end_reserve: number; note: string };

function DividendsCard() {
  const [dv, setDv] = useState<DivStatus | null>(null);
  const [sim, setSim] = useState<SimFeed | null>(null);
  const [showSim, setShowSim] = useState(false);
  const [prev, setPrev] = useState<any | null>(null);
  const [busy, setBusy] = useState(false);
  const [showHow, setShowHow] = useState(false);
  useEffect(() => {
    apiGet<DivStatus>('/api/sleeves/dividends').then(setDv).catch(() => setDv(null));
    fetch('/app/dividend_sim.json').then((x) => x.json()).then(setSim).catch(() => setSim(null));
  }, []);
  const preview = () => {
    setBusy(true);
    fetch('/api/sleeves/dividends/preview', { method: 'POST', credentials: 'include' })
      .then((r) => r.json()).then(setPrev).finally(() => setBusy(false));
  };
  const row = (b: DivBook, label: string) => (
    <div style={{ flex: '1 1 260px' }}>
      <div style={{ fontWeight: 700, marginBottom: 4 }}>{label}</div>
      {b.initialized ? (
        <div className={styles.sub}>
          High-water mark {rup(b.hwm)} · dividend line {b.cap ? `${rup(b.cap)}/qtr` : 'not yet seeded'} ·
          reserve {rup(b.reserve)}
          {b.ledger && b.ledger.length > 0 && (
            <div style={{ marginTop: 4 }}>
              {b.ledger.slice(-4).map((r: any, i: number) => (
                <div key={i}>{r.quarter}: paid {rup(r.paid)} ({r.source}
                  {r.liquidity_clipped ? ', liquidity-clipped' : ''})</div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className={styles.sub}>Not yet initialized — HWM seeds at contributed capital on the
          first declaration run.</div>
      )}
    </div>
  );
  return (
    <div className={styles.card} id="dividends">
      <div className={styles.cardTitle} style={{ display: 'flex', gap: 10, alignItems: 'baseline' }}>
        Dividends — quarterly high-water-mark policy
        <a onClick={(e) => { e.preventDefault(); setShowHow(true); }} href="#"
          style={{ fontSize: 12, fontWeight: 500, textDecoration: 'underline', cursor: 'pointer' }}>
          how the carve-out works
        </a>
      </div>
      {showHow && <HowItWorksModal onClose={() => setShowHow(false)} />}
      <p className={styles.note}>
        25% of new profit above the flow-adjusted HWM leaves the book each quarter; the payout is
        capped at last dividend +7.5%/qtr (a smooth, stepping income line); boom surplus banks into
        a liquid equalization reserve (~6% p.a.) that keeps the line paying through profitless
        quarters. Capital is never invaded and positions are never force-sold. Declarations run
        automatically after each quarter end and fire the intimation email / desktop alert with the
        Console withdrawal amount.
      </p>
      {dv ? (
        <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap' }}>
          {row(dv.truenorth, 'True North')}
          {row(dv.openalpha, 'Open Alpha')}
        </div>
      ) : (
        <p className={styles.note}>
          Live HWM / reserve / ledger state arrives with the 15:40 IST service reload
          (deferred-restart armed). Anchored today: True North HWM ₹9,37,525 (contributed
          capital — currently underwater, so it correctly pays nothing until recovery),
          Open Alpha HWM ₹9,17,628 (NAV at adoption — backfilled history is capital, never
          distributable). First declaration: 30-Sep-2026, 19:15 cron.
        </p>
      )}
      <div style={{ marginTop: 10 }}>
        <button disabled={busy || !dv} onClick={preview}
          style={{ padding: '6px 12px', borderRadius: 6, cursor: 'pointer', fontSize: 13,
                   border: '1px solid var(--hairline, #ccc)', background: 'var(--surface)', color: 'var(--ink)' }}>
          Preview next declaration (dry run)
        </button>
      </div>
      {sim && (
        <div style={{ marginTop: 16, borderTop: '1px solid var(--hairline, rgba(0,0,0,0.12))', paddingTop: 12 }}>
          <a href="#" onClick={(e) => { e.preventDefault(); setShowSim(!showSim); }}
             style={{ fontSize: 13, fontWeight: 600, textDecoration: 'none' }}>
            {showSim ? '▾' : '▸'} What this policy would have paid — 10-year rehearsal
          </a>
          <span style={{ fontSize: 12.5, color: 'var(--ink-muted,#888)', marginLeft: 8 }}>
            ₹10L seed → ₹{(sim.total_paid / 100000).toFixed(1)}L paid out · book ends
            ₹{(sim.end_nav / 100000).toFixed(1)}L + ₹{(sim.end_reserve / 100000).toFixed(1)}L reserve
          </span>
          {showSim && (
            <div style={{ marginTop: 10 }}>
              <div style={{ overflowX: 'auto' }}>
                <table className={styles.tbl}
                       style={{ fontSize: 12.5, fontVariantNumeric: 'tabular-nums', width: 'auto' }}>
                  <thead>
                    <tr>
                      <th style={{ textAlign: 'left', paddingRight: 14 }}>Year</th>
                      <th style={{ textAlign: 'right', paddingLeft: 18 }}>Q1</th>
                      <th style={{ textAlign: 'right', paddingLeft: 18 }}>Q2</th>
                      <th style={{ textAlign: 'right', paddingLeft: 18 }}>Q3</th>
                      <th style={{ textAlign: 'right', paddingLeft: 18 }}>Q4</th>
                      <th style={{ textAlign: 'right', paddingLeft: 26, whiteSpace: 'nowrap' }}>Year paid</th>
                      <th style={{ textAlign: 'right', paddingLeft: 26, whiteSpace: 'nowrap' }}>New profit<br />above HWM</th>
                      <th style={{ textAlign: 'right', paddingLeft: 26, whiteSpace: 'nowrap' }}>Reserve<br />at year end</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sim.rows.map((r) => (
                      <tr key={r.year}>
                        <td style={{ fontWeight: 600, textAlign: 'left', paddingRight: 14 }}>{r.year}</td>
                        {r.q.map((v, i) => (
                          <td key={i} style={{ textAlign: 'right', paddingLeft: 18, whiteSpace: 'nowrap' }}>
                            {v == null ? '—' : Math.round(v).toLocaleString('en-IN')}
                            {r.src[i] === 'reserve' && v ? (
                              <sup title="paid from the equalization reserve — no new profit that quarter"
                                   style={{ color: 'var(--brand-amber,#C97B20)' }}>r</sup>
                            ) : null}
                          </td>
                        ))}
                        <td style={{ textAlign: 'right', paddingLeft: 26, fontWeight: 600, whiteSpace: 'nowrap' }}>
                          {r.total.toLocaleString('en-IN')}</td>
                        <td style={{ textAlign: 'right', paddingLeft: 26, whiteSpace: 'nowrap',
                                     color: r.profit === 0 ? 'var(--accent-neg,#A32D2D)' : 'inherit' }}>
                          {r.profit.toLocaleString('en-IN')}</td>
                        <td style={{ textAlign: 'right', paddingLeft: 26, whiteSpace: 'nowrap',
                                     color: 'var(--ink-muted,#888)' }}>
                          {r.reserve.toLocaleString('en-IN')}</td>
                      </tr>
                    ))}
                    <tr style={{ borderTop: '2px solid var(--hairline,rgba(0,0,0,0.14))', fontWeight: 700 }}>
                      <td style={{ textAlign: 'left', paddingRight: 14 }}>Total</td>
                      <td colSpan={4} />
                      <td style={{ textAlign: 'right', paddingLeft: 26, whiteSpace: 'nowrap' }}>
                        {sim.total_paid.toLocaleString('en-IN')}</td>
                      <td style={{ textAlign: 'right', paddingLeft: 26, whiteSpace: 'nowrap' }}>
                        {sim.total_profit.toLocaleString('en-IN')}</td>
                      <td style={{ textAlign: 'right', paddingLeft: 26, whiteSpace: 'nowrap' }}>
                        {sim.end_reserve.toLocaleString('en-IN')}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p className={styles.note}>
                <sup style={{ color: 'var(--brand-amber,#C97B20)' }}>r</sup> = bridged by the equalization
                reserve (a quarter with no new profit). Read the profit column against the payouts: the policy
                distributes ~25% of genuinely new profit, so a zero-profit year pays only what the reserve can
                bridge — 2018–19 and 2022 show that working. {sim.note}
              </p>
            </div>
          )}
        </div>
      )}

      {prev && ['truenorth', 'openalpha'].map((k) => {
        const p = prev[k];
        return (
          <p key={k} className={styles.note}>
            <b>{k === 'truenorth' ? 'True North' : 'Open Alpha'}:</b>{' '}
            {p?.skipped ? p.skipped :
              p?.declaration ? `NAV ${rup(p.declaration.nav)} vs HWM ${rup(p.declaration.flow_adjusted_hwm_before)} → ` +
                `new profit ${rup(p.declaration.new_profit)} → would pay ${rup(p.declaration.paid)} ` +
                `(${p.declaration.source}), reserve after ${rup(p.declaration.reserve_after)}`
              : 'no data'}
          </p>
        );
      })}
    </div>
  );
}

/* ── the desk's own view: three live books, one portfolio ────────────────────────
   This page used to read bluesky_paper.json for Open Alpha and show Rs 10,30,361 for a
   book holding Rs 4,45,774 — the retired paper model, the same defect the money paths
   had (D1) surviving in the display layer. IPO was absent entirely.

   Every number below now comes from a live book feed:
     True North  /app/momentum_live.json   (cron-baked marks, ~2ms)
     Open Alpha  /app/oa_real.json         (the REAL book)
     IPO Base    /app/ipo_paper.json       (paper until the desk funds it)
   NIFTY for the curve comes from the True North benchmark endpoint, which already
   carries a time-weighted book curve — deposits are not returns, and True North was
   funded from Rs 2.98L to Rs 9.38L inside this window, so raw NAV would not be a
   return series. */

type LiveTN = { updated: string; nav: number; capital: number; value: number; cash: number;
  swept: number; pnl: number; n: number };
type LiveOA = { updated: string; nav: number; capital: number; value: number; cash: number;
  pnl: number; realized: number; gain: number; return_pct: number;
  navcurve: { d: string; nav: number }[]; positions: unknown[] };
type LiveIPO = { updated: string; mode: string; nav: number; capital: number; value: number;
  cash: number; pnl: number; realized: number; gain: number; return_pct: number;
  slots_used: number; slots: number; navcurve: { d: string; nav: number }[];
  pending: unknown[] };

function Tile({ label, value, sub, tone }:
  { label: string; value: string; sub?: string; tone?: 'pos' | 'neg' }) {
  return (
    <div className={styles.tile}>
      <div>{label}</div>
      <b className={tone === 'pos' ? styles.pos : tone === 'neg' ? styles.neg : undefined}>{value}</b>
      {sub && <div className={styles.muted} style={{ fontSize: 11, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

/* Growth of 100 for each book and the portfolio, on the days they share. Books started
   on different dates, so the common window is the shortest of them — stated on the card
   rather than quietly padded. */
function CombinedCurve({ tn, oa, ipo }:
  { tn: TnBench | null; oa: LiveOA | null; ipo: LiveIPO | null }) {
  const series: { name: string; color: string; pts: { d: string; v: number }[] }[] = [];
  if (tn?.book?.length)
    series.push({ name: 'True North', color: '#2563EB',
                  pts: tn.book.map((r) => ({ d: r.d, v: 1 + r.r / 100 })) });
  if (oa?.navcurve?.length)
    series.push({ name: 'Open Alpha', color: '#0891B2',
                  pts: oa.navcurve.map((r) => ({ d: r.d, v: r.nav })) });
  if (ipo?.navcurve?.length)
    series.push({ name: 'IPO Base', color: '#D946A0',
                  pts: ipo.navcurve.map((r) => ({ d: r.d, v: r.nav })) });
  const bench = (tn?.book ?? []) as unknown as { d: string; bench?: number }[];
  const nb = (tn?.series as any)?.NIFTYBEES as { d: string; v: number }[] | undefined;

  const dateSets = series.map((s) => new Set(s.pts.map((p) => p.d)));
  const common = series.length
    ? [...dateSets[0]].filter((d) => dateSets.every((s) => s.has(d))).sort()
    : [];

  if (common.length < 3) {
    return (
      <div className={styles.card}>
        <div className={styles.cardTitle}>Combined curve</div>
        <p className={styles.note}>
          The books start on different dates and share only {common.length} common
          {common.length === 1 ? ' day' : ' days'} so far — Open Alpha since 04-Sep and
          IPO Base since 06-Sep. The curve draws once there are a few days they all cover.
          Each book&apos;s own curve is on its tab in the meantime.
        </p>
      </div>
    );
  }

  const reb = (s: typeof series[0]) => {
    const m = new Map(s.pts.map((p) => [p.d, p.v]));
    const base = m.get(common[0])!;
    return common.map((d) => (m.get(d)! / base) * 100);
  };
  const lines = series.map((s) => ({ name: s.name, color: s.color, v: reb(s) }));
  const port = common.map((_, i) => lines.reduce((a, l) => a + l.v[i], 0) / lines.length);
  lines.unshift({ name: 'Portfolio (blend)', color: '#111', v: port });
  if (nb?.length) {
    const m = new Map(nb.map((r) => [r.d, r.v]));
    if (common.every((d) => m.has(d))) {
      const b0 = m.get(common[0])!;
      lines.push({ name: 'NIFTY', color: '#B4B2A9', v: common.map((d) => (m.get(d)! / b0) * 100) });
    }
  }
  const st = stats(port, common);
  return (
    <div className={styles.card}>
      <div className={styles.cardTitle}>
        Combined curve — growth of 100 over the {common.length} days all books share
      </div>
      <MultiCurve dates={common} lines={lines.map((l) => ({
        name: l.name, v: l.v, color: l.color,
        dash: l.name === 'NIFTY' ? '4 3' : undefined,
      }))} />
      <div className={styles.legend} style={{ marginTop: 8 }}>
        {lines.map((l) => (
          <span key={l.name} className={styles.legendItem}>
            <i className={styles.swatch} style={{ background: l.color }} />
            {l.name} <b>{pct(l.v[l.v.length - 1] - 100)}</b>
          </span>
        ))}
      </div>
      <p className={styles.note}>
        Equal-weighted blend of the books that have started, rebased to 100 on the first
        shared day. True North is drawn from its TIME-WEIGHTED curve — it was funded from
        Rs 2.98L to Rs 9.38L inside this window, so raw NAV is not a return series.
        Portfolio over this window: {pct(st.total)} · worst drawdown {pct(st.dd)}.
      </p>
    </div>
  );
}

export default function CapitalDesk() {
  const [tnLive, setTnLive] = useState<LiveTN | null>(null);
  const [oa, setOa] = useState<LiveOA | null>(null);
  const [ipo, setIpo] = useState<LiveIPO | null>(null);
  const [tn, setTn] = useState<TnBench | null>(null);
  const [showEvidence, setShowEvidence] = useState(false);

  useEffect(() => {
    const j = (u: string) => fetch(u + '?t=' + Date.now()).then((r) => (r.ok ? r.json() : null));
    const load = () => {
      j('/app/momentum_live.json').then(setTnLive).catch(() => {});
      j('/app/oa_real.json').then(setOa).catch(() => {});
      j('/app/ipo_paper.json').then(setIpo).catch(() => {});
    };
    load();
    apiGet<TnBench>('/api/momentum-paper/benchmarks').then(setTn).catch(() => setTn(null));
    const id = setInterval(load, 10000);
    return () => clearInterval(id);
  }, []);

  const navTN = tnLive?.nav ?? 0;
  const navOA = oa?.nav ?? 0;
  const navIPO = ipo?.nav ?? 0;
  const capTN = tnLive?.capital ?? 0;
  const capOA = oa?.capital ?? 0;
  /* IPO on paper runs a notional Rs 10L, which is NOT the portfolio's money — counting it
     would inflate every total on this page. Only real money committed to it counts. */
  const ipoLive = ipo?.mode === 'live';
  const navIPOreal = ipoLive ? navIPO : 0;
  const capIPOreal = ipoLive ? (ipo?.capital ?? 0) : 0;
  const portNav = navTN + navOA + navIPOreal;
  const portCap = capTN + capOA + capIPOreal;
  const gain = portNav - portCap;
  const dayPnl = (tnLive?.pnl ?? 0) + (oa?.pnl ?? 0) + (ipoLive ? (ipo?.pnl ?? 0) : 0);

  return (
    <div className={styles.page}>
      <div className={styles.head}>
        <div>
          <h1>Capital Desk</h1>
          <div className={styles.sub}>
            Every rupee in and out, and the target it is working toward · live book values
          </div>
        </div>
        <span style={{ marginLeft: 'auto', alignSelf: 'center' }}>
          <LiveTick updated={tnLive?.updated || oa?.updated} />
        </span>
      </div>

      <div className={styles.tiles}>
        <Tile label="Portfolio NAV" value={rup(portNav)}
              sub={`on ${rup(portCap)} of capital`} />
        <Tile label="Total return" value={`${gain >= 0 ? '+' : '−'}${rup(Math.abs(gain)).slice(1)}`}
              sub={portCap ? pct((gain / portCap) * 100) : '—'}
              tone={gain >= 0 ? 'pos' : 'neg'} />
        <Tile label="Open P&L today" value={`${dayPnl >= 0 ? '+' : '−'}${rup(Math.abs(dayPnl)).slice(1)}`}
              sub="unrealised, across the live books" tone={dayPnl >= 0 ? 'pos' : 'neg'} />
        <Tile label="True North" value={rup(navTN)}
              sub={tnLive ? `${tnLive.n} holdings · liquid ${rup(tnLive.swept + tnLive.cash)}` : '—'} />
        <Tile label="Open Alpha" value={rup(navOA)}
              sub={oa ? `${oa.positions?.length ?? 0} holdings · REAL money` : '—'} />
        <Tile label="IPO Base" value={ipoLive ? rup(navIPO) : 'on paper'}
              sub={ipo ? `${ipo.slots_used}/${ipo.slots} slots · ${ipo.pending?.length ?? 0} armed`
                       : 'not started'} />
      </div>

      <AllocationDesk />
      <CombinedCurve tn={tn} oa={oa} ipo={ipo} />
      <FundsPanel />
      <DividendsCard />

      <div className={styles.card}>
        <div className={styles.cardTitle}>
          Backtest evidence
          <button onClick={() => setShowEvidence(!showEvidence)}
                  style={{ marginLeft: 10, cursor: 'pointer', font: '500 11px inherit',
                           padding: '3px 8px', borderRadius: 5, background: 'transparent',
                           border: '1px solid var(--hairline,rgba(0,0,0,0.16))',
                           color: 'var(--ink-muted,#8a8a85)' }}>
            {showEvidence ? 'hide' : 'show'}
          </button>
        </div>
        {!showEvidence
          ? <p className={styles.note}>
              Everything above is live. The studies behind these books are one click away.
            </p>
          : (
            <table className={styles.tbl}>
              <thead><tr><th className={styles.txt}>Book</th><th>Study</th><th>Headline</th></tr></thead>
              <tbody>
                <tr><td className={styles.txt}>True North</td>
                  <td><a href="/app/backtest/momentum30-etf-subselection-research62">research/62</a></td>
                  <td className={styles.muted}>Nifty-200 momentum, top-8, 100-SMA gate</td></tr>
                <tr><td className={styles.txt}>Open Alpha</td>
                  <td><a href="/app/backtest/bluesky-ath-breakout-research142">research/142</a></td>
                  <td className={styles.muted}>30.4% CAGR / −31.5% DD, 20-year ensemble</td></tr>
                <tr><td className={styles.txt}>IPO Base</td>
                  <td><a href="/app/backtest/ipo-base-breakout-research153">research/153</a></td>
                  <td className={styles.muted}>31.0% CAGR / −20.9% DD, corr 0.16 to OA</td></tr>
                <tr><td className={styles.txt}>The blend</td>
                  <td><a href="/app/backtest/multi-system-blends-research154">research/154</a></td>
                  <td className={styles.muted}>8,172 weight vectors on 360 paired paths</td></tr>
              </tbody>
            </table>
          )}
      </div>
    </div>
  );
}