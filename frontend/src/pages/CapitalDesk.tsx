import { useEffect, useState } from 'react';
import { apiGet } from '../api/client';
import styles from './MomentumPaper.module.css';
import LiveTick from '../components/LiveTick/LiveTick';
import BookCurve from '../components/BookPanel/BookCurve';

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
        {a.ipo_status === 'paper'
          ? ' IPO is on paper, so its share is earmarked in the liquid ETF.'
          : ' IPO is LIVE — real money, and its signals are real orders.'}
      </div>
      <table className={styles.table}>
        <thead>
          <tr><th className={styles.sym}>Book</th><th>Value</th><th>Now</th>
            <th>Target</th><th>Target ₹</th><th>Gap</th></tr>
        </thead>
        <tbody>
          {a.rows.map((r) => (
            <tr key={r.book}>
              <td className={styles.sym}>{BOOK_LABEL[r.book] ?? r.book}</td>
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
            <td className={styles.sym}><b>Total</b></td>
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
  /* This fetched once on mount and never again. After the Rs 4,00,000 deposit on
     08-Sep the table still showed Open Alpha at Rs 4,46,349 and IPO at zero, and still
     described IPO as being on paper minutes after that same deposit had taken it live.
     A table about where the money IS cannot be a snapshot of where it WAS. */
  useEffect(() => {
    const load = () => apiGet<Allocation>('/api/sleeves/allocation').then(setA)
      .catch((e) => setErr(String(e)));
    load();
    const id = setInterval(load, 10000);
    return () => clearInterval(id);
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
  const [receipt, setReceipt] = useState<{
    kind: string; total: number; halted: boolean; ts: Date;
    done: { book: string; amount: number; data: any }[]; skipped: string[];
  } | null>(null);
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
    const done: { book: string; amount: number; data: any }[] = [];
    const skipped: string[] = [];
    let halted = false;
    for (const p of plans) {
      const label = BOOK_LABEL[p.book] ?? p.book;
      if (halted) { skipped.push(label); continue; }
      const r = await call('/api/sleeves/' + p.book + '/' + kind,
                           { amount: Math.round(p.amount), dry_run: false })
        .catch((e) => ({ ok: false, data: { error: String(e) } }));
      if (r.ok) done.push({ book: p.book, amount: p.amount, data: r.data });
      else { halted = true; skipped.push(label + ' FAILED: ' + (r.data?.error || 'error')); }
    }
    setReceipt({ kind, total: n, done, skipped, halted, ts: new Date() });
    setMsg(null);
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
      {receipt && (
        <div style={{
          border: `1px solid ${receipt.halted ? 'var(--accent-neg,#A32D2D)' : 'var(--accent-pos,#0F6E56)'}`,
          borderLeftWidth: 4, borderRadius: 7, padding: '11px 14px', margin: '4px 0 14px',
          background: 'var(--surface,#fff)',
        }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap' }}>
            <b style={{ fontSize: 14 }}>
              {receipt.halted ? 'Partly applied' : `${receipt.kind === 'deposit' ? 'Deposited' : 'Withdrawn'} ${rup(receipt.total)}`}
            </b>
            <span className={styles.muted} style={{ fontSize: 11.5 }}>
              {receipt.ts.toLocaleTimeString('en-IN', { hour12: false })} IST
            </span>
            <button onClick={() => setReceipt(null)}
                    style={{ marginLeft: 'auto', cursor: 'pointer', font: '500 11px inherit',
                             padding: '2px 8px', borderRadius: 5, background: 'transparent',
                             border: '1px solid var(--hairline,rgba(0,0,0,0.16))',
                             color: 'var(--ink-muted,#8a8a85)' }}>dismiss</button>
          </div>

          <table className={styles.table} style={{ marginTop: 8 }}>
            <tbody>
              {receipt.done.map((d) => (
                <tr key={d.book}>
                  <td className={styles.sym}>{BOOK_LABEL[d.book] ?? d.book}</td>
                  <td><b>{rup(d.amount)}</b></td>
                  <td className={styles.muted}>
                    {d.data?.capital_after != null && `capital now ${rup(d.data.capital_after)}`}
                    {d.data?.cash_after != null && ` · cash ${rup(d.data.cash_after)}`}
                  </td>
                </tr>
              ))}
              {receipt.skipped.map((s, i) => (
                <tr key={'s' + i}><td className={styles.neg} colSpan={3}>{s}</td></tr>
              ))}
            </tbody>
          </table>

          {/* The consequences worth saying out loud. */}
          {receipt.done.some((d) => d.data?.arms_live) && (
            <p className={styles.note} style={{ color: 'var(--accent-pos,#0F6E56)', fontWeight: 600 }}>
              IPO Base is now LIVE. It has left paper permanently — every signal from the next
              scan is a real-money instruction. Withdrawing does not put it back on paper.
            </p>
          )}
          {receipt.done.filter((d) => (d.data?.cash_after ?? 0) > 1000).map((d) => (
            <p key={'c' + d.book} className={styles.note}>
              <b>{rup(d.data.cash_after)} sits as cash in {BOOK_LABEL[d.book] ?? d.book}.</b>{' '}
              Neither live book has an automated executor, so it stays undeployed until you
              place the buys yourself — the book alerts the exact orders, it never sends them.
            </p>
          ))}
          {receipt.halted && (
            <p className={styles.note} style={{ color: 'var(--accent-neg,#A32D2D)' }}>
              Some legs did not run. The books are separate stores, so this cannot be rolled
              back automatically — reverse the applied legs by hand if you want the whole
              flow undone.
            </p>
          )}
        </div>
      )}

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
                <table className={styles.table}
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
  swept: number; pnl: number; n: number; slots?: number; interest?: number; realized?: number;
  positions?: { symbol: string; to_stop_pct?: number | null }[] };
type LiveOA = { updated: string; nav: number; capital: number; value: number; cash: number;
  pnl: number; realized: number; gain: number; return_pct: number;
  navcurve: { d: string; nav: number }[]; slots?: number;
  positions: { symbol: string; to_stop_pct?: number | null;
               to_trail_pct?: number | null }[] };
type LiveIPO = { updated: string; mode: string; nav: number; capital: number; value: number;
  cash: number; pnl: number; realized: number; gain: number; return_pct: number;
  slots_used: number; slots: number; navcurve: { d: string; nav: number }[];
  pending: unknown[];
  positions?: { symbol: string; to_stop_pct?: number | null;
                to_trail_pct?: number | null }[] };

/* ROOM BEFORE A SALE — how far each holding is from the rule that would sell it.
 *
 * The one risk question the desk could not answer: not "what am I worth" but "what is
 * about to go". Each book has its own exit, so the distance is measured against whichever
 * of that book's rules is nearest, and the row says which one:
 *   True North  15-day-low Donchian stop
 *   Open Alpha  the -8% stop, or the 15-SMA trail, whichever is closer
 *   IPO Base    the -8% stop, or the 20-SMA trail, whichever is closer
 *
 * Sorted closest first and cut to a handful, because a name with 30% of room is not news.
 * A NEGATIVE distance means the rule is already breached and the exit is due at the next
 * check - that is the case worth seeing above all others, so it is never truncated away.
 */
type Room = { sym: string; book: string; pct: number; rule: string };

function roomRows(tnLive: LiveTN | null, oa: LiveOA | null, ipo: LiveIPO | null,
                  ipoLive: boolean): Room[] {
  const out: Room[] = [];
  const push = (sym: string, book: string, cands: { v: number | null | undefined; rule: string }[]) => {
    const live = cands.filter((c) => c.v != null) as { v: number; rule: string }[];
    if (!live.length) return;
    const nearest = live.reduce((a, b) => (b.v < a.v ? b : a));
    out.push({ sym, book, pct: nearest.v, rule: nearest.rule });
  };
  (tnLive?.positions ?? []).forEach((p: any) =>
    push(p.symbol, 'TN', [{ v: p.to_stop_pct, rule: 'Donchian stop' }]));
  (oa?.positions ?? []).forEach((p: any) =>
    push(p.symbol, 'OA', [{ v: p.to_stop_pct, rule: '−8% stop' },
                          { v: p.to_trail_pct, rule: '15-SMA trail' }]));
  if (ipoLive) {
    (ipo?.positions ?? []).forEach((p: any) =>
      push(p.symbol, 'IPO', [{ v: p.to_stop_pct, rule: '−8% stop' },
                             { v: p.to_trail_pct, rule: '20-SMA trail' }]));
  }
  return out.sort((a, b) => a.pct - b.pct);
}

function RoomBlock({ rows }: { rows: Room[] }) {
  if (!rows.length) return null;
  const SHOW = 6;
  /* Anything already past its rule is shown however many there are — that is the point. */
  const due = rows.filter((r) => r.pct < 0);
  const head = rows.slice(0, Math.max(SHOW, due.length));
  const rest = rows.slice(head.length);
  /* The gauge is ROOM LEFT out of 10%, not distance: a longer bar must mean safer.
     Drawing |distance| made a breached rule the longest bar on the chart. */
  const FULL = 10;
  const tone = (v: number) =>
    v < 0 ? 'var(--accent-neg,#A32D2D)'
      : v < 5 ? 'var(--accent-neg,#A32D2D)'
      : v < 10 ? 'var(--accent-warn,#B45309)'
      : 'var(--accent-pos,#0F6E56)';

  return (
    <div className={styles.roomBlock}>
      <div className={styles.moneySub}>
        Room before a sale
        <span className={styles.cardCount} style={{ textTransform: 'none', letterSpacing: 0 }}>
          the bar is room left, out of 10% — a full track is comfortable, an empty one is next
        </span>
      </div>
      {head.map((r) => (
        <div key={r.book + r.sym} className={styles.roomRow}>
          <span className={styles.roomSym}>{r.sym}</span>
          <span className={styles.roomBook}>{r.book}</span>
          <span className={styles.roomTrack}>
            <i style={{ width: `${Math.max(0, Math.min(100, (r.pct / FULL) * 100))}%`,
                        background: tone(r.pct) }} />
          </span>
          <b style={{ color: tone(r.pct) }}>
            {r.pct < 0 ? 'due' : r.pct.toFixed(1) + '%'}
          </b>
          <span className={styles.roomRule}>
            {r.pct < 0 ? <span className={styles.roomDue}>alerted · place the sell</span> : r.rule}
          </span>
        </div>
      ))}
      {rest.length > 0 && (
        <div className={styles.roomMore}>
          {rest.length} more, all with over {rest[0].pct.toFixed(0)}% of room
        </div>
      )}
      {due.length > 0 && (
        <p className={styles.note} style={{ marginTop: 9 }}>
          <b>{due.length === 1 ? 'That name is' : 'Those names are'} past the exit rule and
          still held.</b> Open Alpha and IPO Base alert their exits — neither has an exit
          executor, so the book raises the order and you place it. The entry side is
          automated; the exit side is not.
        </p>
      )}
    </div>
  );
}

/* WHERE THE MONEY SITS — the cut the page did not already have.
 *
 * The bar in the panel above splits the portfolio BY BOOK. This splits it by what the
 * money is actually doing: at work in stocks, parked in the liquid fund earning ~5%, or
 * sitting free. Same rupees, different question - "am I deployed?" rather than "who holds
 * what" - so the two are not the same picture twice.
 *
 * P&L IS DELIBERATELY NOT A DONUT. A ring divides a whole into parts, and P&L has no
 * whole: True North can be down while Open Alpha is up, and a pie of mixed signs is
 * meaningless - the slices would not sum to the total and a bigger loss would draw as a
 * bigger share of "profit". It is drawn as bars from a shared zero instead, which is what
 * a signed quantity needs.
 */
function Donut({ segs, size = 132, thickness = 20, centre, sub }:
  { segs: { k: string; v: number; c: string }[]; size?: number; thickness?: number;
    centre: string; sub: string }) {
  const total = segs.reduce((a, x) => a + x.v, 0);
  const r = (size - thickness) / 2;
  const circ = 2 * Math.PI * r;
  let offset = 0;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img"
         aria-label={segs.map((s) => `${s.k} ${Math.round((s.v / (total || 1)) * 100)}%`).join(', ')}>
      <g transform={`rotate(-90 ${size / 2} ${size / 2})`}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth={thickness}
                stroke="var(--hairline-soft,rgba(0,0,0,0.06))" />
        {segs.map((s) => {
          const frac = total ? s.v / total : 0;
          const el = (
            <circle key={s.k} cx={size / 2} cy={size / 2} r={r} fill="none"
                    strokeWidth={thickness} stroke={s.c}
                    strokeDasharray={`${circ * frac} ${circ * (1 - frac)}`}
                    strokeDashoffset={-circ * offset} />
          );
          offset += frac;
          return el;
        })}
      </g>
      <text x={size / 2} y={size / 2 - 1} textAnchor="middle" fontSize={21} fontWeight={700}
            fill="var(--ink,#1B1B1A)">{centre}</text>
      <text x={size / 2} y={size / 2 + 15} textAnchor="middle" fontSize={9.5}
            fill="var(--ink-muted,#888780)">{sub}</text>
    </svg>
  );
}

function MoneyCard({ tnLive, oa, ipo, ipoLive }:
  { tnLive: LiveTN | null; oa: LiveOA | null; ipo: LiveIPO | null; ipoLive: boolean }) {
  /* IPO Base on paper runs a notional Rs10L that is not the portfolio's money, so it is
     counted only once it is live - the same rule the totals above use. */
  const stocks = (tnLive?.value ?? 0) + (oa?.value ?? 0) + (ipoLive ? (ipo?.value ?? 0) : 0);
  const parked = tnLive?.swept ?? 0;
  /* Inside True North's own P&L already — shown, but never added again. */
  const yieldRs = tnLive?.interest ?? 0;
  const free = (tnLive?.cash ?? 0) + (oa?.cash ?? 0) + (ipoLive ? (ipo?.cash ?? 0) : 0);
  const total = stocks + parked + free;
  if (total <= 0) return null;

  const where = [
    { k: 'At work in stocks', v: stocks, c: '#2563EB' },
    { k: 'Liquid fund', v: parked, c: '#0891B2' },
    { k: 'Free cash', v: free, c: 'var(--ink-faint,#B4B2A9)' },
  ].filter((x) => x.v > 0);

  const books = [
    { k: 'True North', v: (tnLive?.nav ?? 0) - (tnLive?.capital ?? 0), c: '#0F6E56' },
    { k: 'Open Alpha', v: oa?.gain ?? 0, c: '#A21CAF' },
    ...(ipoLive ? [{ k: 'IPO Base', v: ipo?.gain ?? 0, c: '#0E7490' }] : []),
  ];
  const worst = Math.max(1, ...books.map((b) => Math.abs(b.v)));
  const net = books.reduce((a, b) => a + b.v, 0);

  return (
    <div className={styles.card}>
      <div className={styles.cardTitle}>
        Where the money sits
        <span className={styles.cardCount}>
          by what it is doing, not by which book holds it
        </span>
      </div>

      <div className={styles.moneyGrid}>
        <div>
        <div className={styles.moneySplit}>
          <Donut segs={where} centre={`${Math.round((stocks / total) * 100)}%`} sub="at work" />
          <div className={styles.moneyKeys}>
            {where.map((x) => (
              <div key={x.k} className={styles.moneyKey}>
                <i style={{ background: x.c }} />
                <span className={styles.moneyKeyName}>{x.k}</span>
                <b>{rup(x.v)}</b>
                <span className={styles.muted}>{((x.v / total) * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>

        <RoomBlock rows={roomRows(tnLive, oa, ipo, ipoLive)} />
        </div>

        <div className={styles.moneyPnl}>
          <div className={styles.moneySub}>Profit and loss, by book</div>
          {books.map((b) => (
            <div key={b.k} className={styles.pnlBarRow}>
              <span className={styles.pnlBarName}>{b.k}</span>
              <span className={styles.pnlBarTrack}>
                {/* both halves of a shared zero, so a loss reads as a loss */}
                <i className={styles.pnlBarZero} />
                <i style={{
                  background: b.v >= 0 ? 'var(--accent-pos,#0F6E56)' : 'var(--accent-neg,#A32D2D)',
                  width: `${(Math.abs(b.v) / worst) * 50}%`,
                  left: b.v >= 0 ? '50%' : undefined,
                  right: b.v < 0 ? '50%' : undefined,
                }} />
              </span>
              <b className={b.v >= 0 ? styles.pos : styles.neg}>
                {b.v >= 0 ? '+' : '−'}{rup(Math.abs(b.v)).slice(1)}
              </b>
            </div>
          ))}
          <div className={`${styles.pnlBarRow} ${styles.pnlBarTotal}`}>
            <span className={styles.pnlBarName}>Together</span>
            <span className={styles.pnlBarTrack} />
            <b className={net >= 0 ? styles.pos : styles.neg}>
              {net >= 0 ? '+' : '−'}{rup(Math.abs(net)).slice(1)}
            </b>
          </div>
          {parked > 0 && (
            <div className={styles.pnlBarRow} style={{ paddingTop: 2 }}>
              <span className={styles.pnlBarName} style={{ width: 'auto' }}>
                <span className={styles.muted}>of which</span> liquid fund
              </span>
              <span className={styles.pnlBarTrack} style={{ flex: 'none' }} />
              <span className={styles.muted} style={{ marginLeft: 'auto', fontSize: 11.5 }}>
                {rup(parked)} parked{yieldRs ? ' · earned ' : ''}
                {yieldRs ? <b className={yieldRs >= 0 ? styles.pos : styles.neg}>
                  {yieldRs >= 0 ? '+' : '−'}{rup(Math.abs(yieldRs)).slice(1)}</b> : null}
              </span>
            </div>
          )}
          <p className={styles.note} style={{ marginTop: 8 }}>
            Drawn as bars from a shared zero rather than a ring: one book can be down while
            another is up, and a ring divides a whole into parts that a signed quantity has
            not got. The liquid fund is True North's swept cash, so its yield is already
            inside True North's figure — shown here, never added twice.
          </p>
        </div>
      </div>
    </div>
  );
}

/* WHAT HAPPENS NEXT — the desk's answer to "should I be doing something?"
 *
 * The page showed the portfolio's state but never its next move, so the only way to know
 * whether a deposit would be deployed this morning or next Tuesday was to read a crontab.
 *
 * ─────────────────────────────────────────────────────────────────────────────────────
 * THIS LIST MIRRORS THE VPS CRONTAB AND momentum_paper.register(). It is display only —
 * nothing here schedules anything — so if a job moves, MOVE IT HERE TOO or the page will
 * calmly state a falsehood. Sources, as of 08-Sep-2026:
 *   crontab -l                     : the 09:20 executor, the IPO reconciles, 15:18, 15:50,
 *                                    16:20, 17:30, 17:45, 18:45
 *   services/momentum_paper.py     : 09:20 reconcile, 14:45 rebalance, 15:05 exits
 * ─────────────────────────────────────────────────────────────────────────────────────
 *
 * `acts` marks the steps that can move money or raise an order alert, as against the ones
 * that only refresh data. That is the distinction worth seeing at a glance.
 */
type Step = { at: string; what: string; who: string; acts?: boolean };

const DAY: Step[] = [
  { at: '09:20', who: 'all three', acts: true,
    what: 'Deploy any cash you deposited — the executor places the buys and alerts anything it could not fill' },
  { at: '09:35', who: 'IPO Base', what: 'Confirm overnight buy-stop fills against the broker (again at 11:35, 13:35, 15:35)' },
  { at: '14:45', who: 'True North', acts: true,
    what: 'Monthly re-rank — only on the rebalance day, and it runs early to leave runway' },
  { at: '15:05', who: 'True North', acts: true,
    what: 'Exit check — Donchian 15-day-low stop and the 100-DMA gate' },
  { at: '15:18', who: 'Open Alpha', acts: true,
    what: 'Exit check — the −8% stop and the 15-SMA trail' },
  { at: '15:50', who: 'Open Alpha', what: 'Reconcile the day’s own fills into the book' },
  { at: '16:20', who: 'True North', what: 'Momentum scan for the next rebalance' },
  { at: '17:30', who: 'IPO Base', what: 'Onboard newly listed NSE names into the universe' },
  { at: '17:45', who: 'all three', what: 'Daily price refresh for every name in the universe' },
  { at: '18:45', who: 'IPO Base', acts: true,
    what: 'The book runs: exits, entries, and tomorrow’s buy-stops are armed' },
];

function istNow() {
  const d = new Date();
  return new Date(d.getTime() + (d.getTimezoneOffset() + 330) * 60000);
}

function NextUp() {
  const [, tick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => tick((n) => n + 1), 30000);
    return () => clearInterval(id);
  }, []);

  const now = istNow();
  const mins = now.getHours() * 60 + now.getMinutes();
  const weekend = now.getDay() === 0 || now.getDay() === 6;
  const mm = (s: string) => parseInt(s.slice(0, 2), 10) * 60 + parseInt(s.slice(3), 10);

  /* On a weekend, or once the day's last job has run, the whole list is "tomorrow" —
     saying "next: 09:20" on a Saturday evening would be true only in a useless sense. */
  const nextIdx = weekend ? -1 : DAY.findIndex((s) => mm(s.at) > mins);
  const when = weekend
    ? 'Nothing runs at the weekend — the list below resumes Monday'
    : nextIdx === -1
      ? 'Done for today — the list below resumes tomorrow'
      : `Next in ${(() => {
          const d = mm(DAY[nextIdx].at) - mins;
          return d < 60 ? `${d} min` : `${Math.floor(d / 60)}h ${d % 60}m`;
        })()}`;

  return (
    <div className={styles.card}>
      <div className={styles.cardTitle}>
        What happens next
        <span className={styles.cardCount}>{when} · all times IST, weekdays</span>
      </div>
      <table className={styles.table}>
        <tbody>
          {DAY.map((s, i) => {
            const done = !weekend && mm(s.at) <= mins;
            const isNext = i === nextIdx;
            return (
              <tr key={s.at + s.who}
                  style={{ opacity: done ? 0.45 : 1,
                           fontWeight: isNext ? 600 : undefined }}>
                <td style={{ width: 62, whiteSpace: 'nowrap' }}>
                  <b>{s.at}</b>
                </td>
                <td style={{ width: 96, whiteSpace: 'nowrap' }} className={styles.muted}>{s.who}</td>
                <td>
                  {s.what}
                  {s.acts && <span className={styles.actsTag} title="This step can place an order or raise an alert">acts</span>}
                </td>
                <td style={{ width: 74, textAlign: 'right' }} className={styles.muted}>
                  {isNext ? 'next' : done ? 'done' : ''}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className={styles.note}>
        Deposits are the only thing that waits on you, and only until the next 09:20 — the
        executor deploys them, places the orders and alerts anything it could not fill by
        email and WhatsApp. Steps marked <b>acts</b> can move money or raise an alert; the
        rest only refresh data.
      </p>
    </div>
  );
}

/* Growth of 100 for each book and the portfolio, on the days they share. Books started
   on different dates, so the common window is the shortest of them — stated on the card
   rather than quietly padded. */

export default function CapitalDesk() {
  const [tnLive, setTnLive] = useState<LiveTN | null>(null);
  const [oa, setOa] = useState<LiveOA | null>(null);
  const [ipo, setIpo] = useState<LiveIPO | null>(null);
  const [showEvidence, setShowEvidence] = useState(false);
  const [showCurve, setShowCurve] = useState(false);

  useEffect(() => {
    const j = (u: string) => fetch(u + '?t=' + Date.now()).then((r) => (r.ok ? r.json() : null));
    const load = () => {
      j('/app/momentum_live.json').then(setTnLive).catch(() => {});
      j('/app/oa_real.json').then(setOa).catch(() => {});
      j('/app/ipo_paper.json').then(setIpo).catch(() => {});
    };
    load();
    const id = setInterval(load, 10000);
    return () => clearInterval(id);
  }, []);

  const tnSlots = tnLive?.slots ?? 0;
  const armed = ipo?.pending?.length ?? 0;
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
  /* Same three colours the books use for their own allocation bars, so a segment means
     the same thing wherever it appears. */
  const segs = [
    { k: 'True North', v: navTN, c: '#2563EB' },
    { k: 'Open Alpha', v: navOA, c: '#0891B2' },
    ...(ipoLive ? [{ k: 'IPO Base', v: navIPO, c: '#D946A0' }] : []),
  ].filter((x) => x.v > 0);
  const segTotal = segs.reduce((a, x) => a + x.v, 0);

  return (
    <div className={styles.root}>
      <div className={styles.studyBar}>
        <span className={styles.studyBarLabel}>The desk</span>
        <a className={styles.studyLink} href="/app/strategies">Strategies register</a>
        <a className={styles.studyLink} href="/app/holdings">Broker holdings</a>
      </div>

      <div className={styles.headerRow}>
        <div>
          <h1 className={styles.title}>Capital Desk</h1>
          <p className={styles.sub}>
            Every rupee in and out, and the target it is working toward · live book values ·
            True North is the base and is never sold to rebalance
          </p>
        </div>
        <span style={{ marginLeft: 'auto', alignSelf: 'center' }}>
          <LiveTick updated={tnLive?.updated || oa?.updated} />
        </span>
      </div>

      <div className={`${styles.sumWrap} ${showCurve ? styles.sumWrapOpen : ''}`}>
      <div className={styles.bookSummary}>
        <div className={styles.sumMain}>
          <div className={styles.sumLabel}>Portfolio value</div>
          <div className={styles.sumHero}>{rup(portNav)}</div>
          <div className={styles.sumSub}>
            on <b>{rup(portCap)}</b> of capital{' '}
            <span className={gain >= 0 ? styles.pos : styles.neg} style={{ fontWeight: 700 }}>
              {gain >= 0 ? '+' : '−'}{rup(Math.abs(gain)).slice(1)}
              {portCap ? ' · ' + pct((gain / portCap) * 100) : ''}
            </span>
          </div>
          <div className={styles.barWrap} role="img" aria-label="allocation by book">
            {segs.map((x) => (
              <div key={x.k} className={styles.barSeg}
                   style={{ width: `${(x.v / (segTotal || 1)) * 100}%`, background: x.c }} />
            ))}
          </div>
          <div className={styles.legend}>
            {segs.map((x) => (
              <span key={x.k} className={styles.legendItem}>
                <i className={styles.swatch} style={{ background: x.c }} />
                {x.k} <b>{rup(x.v)}</b>
                <span className={styles.legendPct}>
                  {((x.v / (segTotal || 1)) * 100).toFixed(0)}%
                </span>
              </span>
            ))}
          </div>
          <div className={styles.sumStatus}>
            <span><b>{tnLive?.n ?? 0}{tnSlots ? `/${tnSlots}` : ''}</b> True North holdings</span>
            <span><b>{oa?.positions?.length ?? 0}{oa?.slots ? `/${oa.slots}` : ''}</b> Open Alpha holdings</span>
            <span><b>{ipo ? `${ipo.slots_used}/${ipo.slots}` : '—'}</b>{' '}
              IPO {ipoLive ? 'live' : 'on paper'}</span>
            <span title="Buy-stop orders queued for the next session: names that closed above their pivot today, which IPO Base will buy tomorrow if they trade there.">
              {armed === 0
                ? <span className={styles.muted}>no buy-stops for tomorrow</span>
                : <><b>{armed}</b> buy-stop{armed === 1 ? '' : 's'} for tomorrow</>}
            </span>
            <button type="button" className={styles.sumTog} onClick={() => setShowCurve((v) => !v)}
                    aria-expanded={showCurve} aria-controls="desk-curve"
                    title={showCurve ? 'Hide the curve'
                                     : 'The portfolio against Nifty 50, with each book beside it'}>
              {'\u25be'}
            </button>
          </div>
        </div>
        <div className={styles.sumPnl}>
          <div className={styles.sumLabel}>Profit &amp; loss</div>
          {(() => {
            /* The parts, in the same order and language as every book page. Costs are the
               residual - unrealised is measured before entry costs while realised is
               already net of its own, so naming the gap is the honest thing to do. */
            const unreal = (tnLive?.pnl ?? 0) + (oa?.pnl ?? 0) + (ipoLive ? (ipo?.pnl ?? 0) : 0);
            const real = (tnLive?.realized ?? 0) + (oa?.realized ?? 0)
                       + (ipoLive ? (ipo?.realized ?? 0) : 0);
            const yld = tnLive?.interest ?? 0;
            const costs = gain - (unreal + real + yld);
            return [
              { k: 'Unrealised', v: unreal, hint: 'open positions, before entry costs' },
              { k: 'Realised (net)', v: real, hint: 'closed trades, after their own costs' },
              { k: 'Liquid fund yield', v: yld, hint: 'gain on True North\u2019s swept cash' },
              { k: 'Costs & fees', v: costs, hint: 'the residual: brokerage, STT and stamp duty not already netted above' },
            ].map((x) => (
              <div key={x.k} className={styles.pnlRow} title={x.hint}>
                <span>{x.k}</span>
                <b className={x.v > 0 ? styles.pos : x.v < 0 ? styles.neg : styles.muted}>
                  {x.v >= 0 ? '+' : '\u2212'}{rup(Math.abs(x.v)).slice(1)}
                </b>
              </div>
            ));
          })()}
          <div className={`${styles.pnlRow} ${styles.pnlTotal}`}>
            <span>Total return</span>
            <b className={gain >= 0 ? styles.pos : styles.neg}>
              {gain >= 0 ? '+' : '−'}{rup(Math.abs(gain)).slice(1)}
              {portCap ? ' · ' + pct((gain / portCap) * 100) : ''}</b>
          </div>
        </div>
      </div>
      {showCurve && (
        <div className={styles.sumReveal} id="desk-curve">
          <BookCurve url="/api/books/portfolio/benchmarks" label="Momentum Portfolio" />
        </div>
      )}
      </div>

      <MoneyCard tnLive={tnLive} oa={oa} ipo={ipo} ipoLive={ipoLive} />

      <NextUp />

      <AllocationDesk />
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
            <table className={styles.table}>
              <thead><tr><th className={styles.sym}>Book</th><th>Study</th><th>Headline</th></tr></thead>
              <tbody>
                <tr><td className={styles.sym}>True North</td>
                  <td><a href="/app/backtest/momentum30-etf-subselection-research62">research/62</a></td>
                  <td className={styles.muted}>Nifty-200 momentum, top-8, 100-SMA gate</td></tr>
                <tr><td className={styles.sym}>Open Alpha</td>
                  <td><a href="/app/backtest/bluesky-ath-breakout-research142">research/142</a></td>
                  <td className={styles.muted}>30.4% CAGR / −31.5% DD, 20-year ensemble</td></tr>
                <tr><td className={styles.sym}>IPO Base</td>
                  <td><a href="/app/backtest/ipo-base-breakout-research153">research/153</a></td>
                  <td className={styles.muted}>31.0% CAGR / −20.9% DD, corr 0.16 to OA</td></tr>
                <tr><td className={styles.sym}>The blend</td>
                  <td><a href="/app/backtest/multi-system-blends-research154">research/154</a></td>
                  <td className={styles.muted}>8,172 weight vectors on 360 paired paths</td></tr>
              </tbody>
            </table>
          )}
      </div>
    </div>
  );
}