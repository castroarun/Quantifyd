import { useEffect, useState } from 'react';
import { apiGet } from '../api/client';
import styles from './BlueskyPaper.module.css';

/* Combined sleeves view: Momentum (live/paper) + BlueSky (paper) + the 50-50
   monthly-rebalanced blend, all rebased to 100 at the common start. Read-only
   projection over the two books' own feeds — touches no trading logic. */

type MomNav = { d: string; nav: number; bench: number | null };
type MomState = { navcurve: MomNav[]; nav: number; capital: number; total_return_pct: number; gate: string };
type BsNav = { date: string; nav: number; bench: number | null };
type BsFeed = { nav_curve: BsNav[]; nav: number; capital: number; ret_pct: number; gate_weak: boolean };

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

type FlowsStatus = {
  open_alpha: { nav: number | null; cash: number | null; liquid: number;
    capital: number | null; sweep: { units: number; cost: number } | null;
    flows: { ts: string; kind: string; amount: number }[] };
  note: string;
};

function FundsPanel() {
  const [st, setSt] = useState<FlowsStatus | null>(null);
  const [amt, setAmt] = useState('');
  const [msg, setMsg] = useState<string | null>(null);
  const load = () => apiGet<FlowsStatus>('/api/sleeves/status').then(setSt).catch((e) => setMsg(String(e)));
  useEffect(() => { load(); }, []);
  const act = (kind: 'deposit' | 'withdraw') => {
    const n = Number(amt);
    if (!n || n <= 0) { setMsg('enter a positive amount'); return; }
    fetch(`/api/sleeves/openalpha/${kind}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ amount: n }), credentials: 'include',
    }).then(async (r) => {
      const d = await r.json();
      setMsg(r.ok ? `${kind} of ₹${n.toLocaleString('en-IN')} done — ${d.note}` : d.error);
      setAmt(''); load();
    }).catch((e) => setMsg(String(e)));
  };
  const oa = st?.open_alpha;
  return (
    <div className={styles.card}>
      <div className={styles.cardTitle}>Funds — Open Alpha sleeve (paper) · True North funds move via its own page</div>
      {oa && (
        <div className={styles.sub} style={{ marginBottom: 10 }}>
          Liquid (cash + CASHIETF sweep): <b>₹{Math.round(oa.liquid).toLocaleString('en-IN')}</b> ·
          capital contributed: ₹{Math.round(oa.capital ?? 0).toLocaleString('en-IN')} ·
          withdrawals never force-sell positions
        </div>
      )}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <input value={amt} onChange={(e) => setAmt(e.target.value)} placeholder="amount ₹"
               style={{ padding: '8px 10px', borderRadius: 6, border: '1px solid var(--hairline, #ccc)',
                        background: 'var(--surface)', color: 'var(--ink)', width: 140 }} />
        <button className={styles.tile} style={{ cursor: 'pointer' }} onClick={() => act('deposit')}>Deposit</button>
        <button className={styles.tile} style={{ cursor: 'pointer' }} onClick={() => act('withdraw')}>Withdraw</button>
      </div>
      {msg && <p className={styles.note}>{msg}</p>}
      {oa && oa.flows.length > 0 && (
        <p className={styles.note}>
          Recent flows: {oa.flows.slice(-5).map((f) => `${f.kind} ₹${Math.round(f.amount).toLocaleString('en-IN')} (${f.ts.slice(0, 10)})`).join(' · ')}
        </p>
      )}
      <p className={styles.note}>
        Deposits land in cash, sweep to CASHIETF, and fund new signals from the next nightly run.
        Book-page figures refresh with the nightly cycle. Real-money flows arrive with the go-live
        allocator after the Dec-5 soak review.
      </p>
    </div>
  );
}

export default function Sleeves() {
  const [mom, setMom] = useState<MomState | null>(null);
  const [bs, setBs] = useState<BsFeed | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    apiGet<MomState>('/api/momentum-paper/state').then(setMom).catch((e) => setErr('momentum: ' + e));
    fetch('/app/bluesky_paper.json').then((r) => r.json()).then(setBs).catch((e) => setErr('bluesky: ' + e));
  }, []);
  if (err) return <div className={styles.page}><div className={styles.empty}>{err}</div></div>;
  if (!mom || !bs) return <div className={styles.page}><div className={styles.empty}>Loading both books…</div></div>;

  const mMap = new Map(mom.navcurve.map((r) => [r.d, r]));
  const rows = bs.nav_curve.filter((r) => mMap.has(r.date));
  if (rows.length < 25)
    return <div className={styles.page}>
      <h1>Sleeves — True North × Open Alpha</h1>
      <FundsPanel />
      <div className={styles.empty}>
        Only {rows.length} overlapping trading days so far — the combined view becomes meaningful as the
        Open Alpha soak accumulates history alongside True North. Both books are shown on their own
        pages meanwhile.
      </div></div>;

  const dates = rows.map((r) => r.date);
  const bV = rows.map((r) => 100 * r.nav / rows[0].nav);
  const mV = rows.map((r) => 100 * (mMap.get(r.date)!.nav) / (mMap.get(dates[0])!.nav));
  const benchRaw = rows.map((r) => r.bench);
  const b0 = benchRaw.find((x) => x != null) ?? 1;
  const nV = benchRaw.map((v) => (v == null ? NaN : 100 * v / (b0 as number)));
  const blend = blend5050(dates, mV, bV);
  const sM = stats(mV, dates), sB = stats(bV, dates), sX = stats(blend, dates);
  const corr = corrMonthly(dates, mV, bV);

  return (
    <div className={styles.page}>
      <div className={styles.head}>
        <div>
          <h1>Sleeves — Momentum × BlueSky (50-50, monthly rebalanced)</h1>
          <div className={styles.sub}>
            Read-only combined view over both books · common window {dates[0]} → {dates[dates.length - 1]} ·
            monthly correlation {corr == null ? '—' : corr.toFixed(2)}
          </div>
        </div>
      </div>
      <FundsPanel />
      <div className={styles.tiles}>
        <div className={styles.tile}><div>Momentum</div>
          <b className={sM.total >= 0 ? styles.pos : styles.neg}>{pct(sM.total)}</b></div>
        <div className={styles.tile}><div>BlueSky</div>
          <b className={sB.total >= 0 ? styles.pos : styles.neg}>{pct(sB.total)}</b></div>
        <div className={styles.tile}><div>50-50 blend</div>
          <b className={sX.total >= 0 ? styles.pos : styles.neg}>{pct(sX.total)}</b></div>
        <div className={styles.tile}><div>Blend CAGR</div><b>{pct(sX.cagr)}</b></div>
        <div className={styles.tile}><div>Blend MaxDD</div><b className={styles.neg}>{pct(sX.dd)}</b></div>
        <div className={styles.tile}><div>DD: Mom / BlueSky</div>
          <b>{pct(sM.dd)} / {pct(sB.dd)}</b></div>
      </div>
      <div className={styles.card}>
        <div className={styles.cardTitle}>
          Growth of 100 (log) — gold = 50-50 blend · green = Momentum · blue = BlueSky · dashed = NIFTYBEES
        </div>
        <MultiCurve dates={dates} lines={[
          { name: 'NIFTYBEES', v: nV.map((v) => (isNaN(v) ? 100 : v)), color: 'var(--ink-muted)', dash: '4 3' },
          { name: 'Momentum', v: mV, color: '#1f9d55' },
          { name: 'BlueSky', v: bV, color: '#3b82d6' },
          { name: '50-50 blend', v: blend, color: '#d4a017' },
        ]} />
        <p className={styles.note}>
          The backtested version of this blend (2006 → Jul 2026): 33.0% CAGR at −27.5% max drawdown,
          beating both legs — see the correlation matrix and capstone table on the
          {' '}<a className={styles.studyLink} href="/app/backtest/bluesky-ath-breakout-research142">study page</a>.
          This live view accumulates the forward evidence for the same construction.
        </p>
      </div>
    </div>
  );
}
