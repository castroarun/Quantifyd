/**
 * StraddleSystems — every short-premium system in one table.
 *
 * Reads static/app/straddles_systems.json, built after the close by
 * scripts/straddles_registry.py. That file is the only source here; this component
 * renders and does not compute, so a rule change reaches the page by changing the
 * registry rather than this file.
 *
 * Layout follows the approved mock (docs/mockups/straddles-systems-table.html):
 * grouped Intraday / Positional, a row per system, expanding to position + history
 * on the left and track record + evidence on the right, with an R badge on the
 * master row opening that system's rules.
 */
import { useEffect, useMemo, useState } from 'react';
import s from './StraddleSystems.module.css';

type Leg = { side: string; type: string; strike: number; role?: string; qty: number;
             entry: number | null; ltp: number | null; pnl: number | null };
type Closed = { day: string; exit?: string | null; expiry?: string | null; dte?: number | null;
                strike?: number | null; credit?: number | null; reason?: string | null;
                pnl: number | null };
type Evidence = { method: string[]; period: string; nums: Record<string, string>;
                  how: string; caveat: string; links: [string, string][] };
type Rules = { does: [string, string][]; doesnt: string[]; doc: string };
type Sys = {
  key: string; name: string; subtitle: string; kind: 'intraday' | 'positional';
  venue: string; money: 'real' | 'paper' | 'refuted';
  size_lots: number | null; size_qty: number | null; window: string;
  state: { label: string; tone: string };
  today_pnl: number | null; running_pnl: number | null;
  risk_open: number | null; to_stop: { pct: number; of: number } | null;
  lifetime: { net: number; n: number; win: number | null; maxdd: number; t: number | null };
  legs: Leg[]; curve: [string, number][]; closed: Closed[];
  evidence: Evidence; rules: Rules;
};
type Feed = { generated_at: string; date: string; n: number; systems: Sys[] };

const inr = (v: number | null | undefined) =>
  v === null || v === undefined ? '—'
    : (v < 0 ? '−' : '') + '₹' + Math.abs(Math.round(v)).toLocaleString('en-IN');
const cls = (v: number | null | undefined) =>
  v === null || v === undefined ? s.flat : v > 0 ? s.pos : v < 0 ? s.neg : '';

/** Sparkline of P&L since entry. Inline SVG — no chart library for 40 points. */
function Curve({ data }: { data: [string, number][] }) {
  if (!data || data.length < 2) return null;
  const ys = data.map((d) => d[1]);
  const lo = Math.min(...ys, 0), hi = Math.max(...ys, 0);
  const span = hi - lo || 1;
  const W = 320, H = 76;
  const x = (i: number) => (i / (data.length - 1)) * W;
  const y = (v: number) => H - ((v - lo) / span) * (H - 8) - 4;
  const line = data.map((d, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(d[1]).toFixed(1)}`).join(' ');
  const last = ys[ys.length - 1];
  const col = last >= 0 ? 'var(--accent-pos)' : 'var(--accent-neg)';
  return (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none"
           style={{ display: 'block' }}>
        <line x1="0" y1={y(0)} x2={W} y2={y(0)} stroke="var(--hairline)" strokeWidth="1"
              strokeDasharray="3 3" />
        <path d={`${line} L${W},${H} L0,${H} Z`} fill={col} fillOpacity="0.12" />
        <path d={line} fill="none" stroke={col} strokeWidth="1.6" />
        <circle cx={W} cy={y(last)} r="3" fill={col} />
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11,
                    color: 'var(--ink-faint)', marginTop: 4, fontVariantNumeric: 'tabular-nums' }}>
        <span>{data[0][0]}</span>
        <span>peak {inr(Math.max(...ys))}</span>
        <span>now {inr(last)}</span>
      </div>
    </>
  );
}

function RulesModal({ sys, onClose }: { sys: Sys; onClose: () => void }) {
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', k);
    return () => document.removeEventListener('keydown', k);
  }, [onClose]);
  return (
    <div className={s.backdrop} role="dialog" aria-modal="true"
         onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className={s.modal}>
        <div className={s.mhead}>
          <div>
            <div className={s.mtitle}>{sys.name} — rules</div>
            <div className={s.mtag}>
              {sys.money === 'real' ? 'Real money' : sys.money === 'refuted' ? 'Refuted' : 'Paper'}
              {sys.size_lots ? ` · ${sys.size_lots} lots` : ''}
              {sys.size_qty ? ` (qty ${sys.size_qty})` : ''} · {sys.kind} · {sys.venue}
            </div>
          </div>
          <button className={s.mclose} onClick={onClose} aria-label="Close">&times;</button>
        </div>
        <div className={s.mbody}>
          {sys.rules.does.length > 0 && <>
            <div className={s.rgrp}>What it does</div>
            {sys.rules.does.map(([k, v], i) => (
              <div className={s.rrow} key={i}>
                <div className={s.rk}>{k}</div>
                <div className={s.rv} dangerouslySetInnerHTML={{ __html: v }} />
              </div>
            ))}
          </>}
          {sys.rules.doesnt.length > 0 && <>
            <div className={s.rgrp}>What it deliberately does not do</div>
            <div className={s.rnot}><ul>
              {sys.rules.doesnt.map((x, i) => (
                <li key={i} dangerouslySetInnerHTML={{ __html: x }} />
              ))}
            </ul></div>
          </>}
        </div>
        <div className={s.mfoot}>Rules doc: {sys.rules.doc}</div>
      </div>
    </div>
  );
}

function Evid({ e }: { e: Evidence }) {
  return (
    <div className={s.box}>
      <div style={{ marginBottom: 9 }}>
        {e.method.map((m) => (
          <span key={m} className={`${s.chip} ${m === 'AlgoTest' ? s.chipAmber
            : m.includes('bhavcopy') ? s.chipPos : ''}`}>{m}</span>
        ))}
        {e.period && <span className={s.chip}>{e.period}</span>}
      </div>
      {Object.keys(e.nums).length > 0 && (
        <div className={s.mini} style={{ marginBottom: 0 }}>
          {Object.entries(e.nums).map(([k, v]) => (
            <div className={s.m} key={k}>
              <div className={s.mK}>{k}</div>
              <div className={`${s.mV} ${v.startsWith('+') ? s.pos : v.startsWith('−') ? s.neg : ''}`}>{v}</div>
            </div>
          ))}
        </div>
      )}
      <div className={s.note}>
        {e.how && <><b>How:</b> <span dangerouslySetInnerHTML={{ __html: e.how }} /><br /></>}
        {e.caveat && <span dangerouslySetInnerHTML={{ __html: e.caveat }} />}
        {e.links.length > 0 && <div style={{ marginTop: 6 }}>
          {e.links.map(([label, href], i) => (
            <span key={label}>
              {i > 0 && ' · '}<a className={s.lnk} href={href}>{label} ↗</a>
            </span>
          ))}
        </div>}
      </div>
    </div>
  );
}

export default function StraddleSystems() {
  const [feed, setFeed] = useState<Feed | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [rules, setRules] = useState<Sys | null>(null);

  useEffect(() => {
    fetch('/app/straddles_systems.json?t=' + Date.now())
      .then((r) => r.json()).then(setFeed).catch(() => {});
  }, []);

  const kpi = useMemo(() => {
    if (!feed) return null;
    const S = feed.systems;
    const today = S.reduce((a, x) => a + (x.today_pnl || 0), 0);
    const real = S.filter((x) => x.money === 'real')
                  .reduce((a, x) => a + (x.today_pnl || 0), 0);
    const traded = S.filter((x) => x.today_pnl !== null).length;
    const holding = S.filter((x) => x.legs.length > 0).length;
    const idle = S.filter((x) => x.today_pnl === null && x.money !== 'refuted').length;
    return { today, real, traded, holding, idle, n: S.length };
  }, [feed]);

  if (!feed) return null;
  const groups: [string, string, Sys[]][] = [
    ['intraday', 'Intraday · entered and squared off the same day',
     feed.systems.filter((x) => x.kind === 'intraday')],
    ['positional', 'Positional · carried overnight across sessions',
     feed.systems.filter((x) => x.kind === 'positional')],
  ];
  const tot = feed.systems.reduce((a, x) => a + (x.lifetime.net || 0), 0);

  return (
    <div className={s.wrap}>
      <div className={s.head}>
        <span className={s.h1}>Systems</span>
        <span className={s.live}><span className={s.dot} />{feed.n} books</span>
      </div>
      <p className={s.sub}>
        Every short-premium system, what it is doing today, and the evidence behind it.
        Click a row for its position, history and backtest · <b>R</b> for its rules.
        Built {feed.generated_at.replace('T', ' ')}.
      </p>

      {kpi && (
        <div className={s.strip}>
          <div className={s.kpi}><div className={s.kpiK}>Today · all books</div>
            <div className={`${s.kpiV} ${cls(kpi.today)}`}>{inr(kpi.today)}</div>
            <div className={s.kpiN}>{kpi.traded} of {kpi.n} traded</div></div>
          <div className={s.kpi}><div className={s.kpiK}>Real money today</div>
            <div className={`${s.kpiV} ${cls(kpi.real)}`}>{inr(kpi.real)}</div>
            <div className={s.kpiN}>NAS_COMB20 only</div></div>
          <div className={s.kpi}><div className={s.kpiK}>Holding now</div>
            <div className={s.kpiV}>{kpi.holding}</div>
            <div className={s.kpiN}>open positions</div></div>
          <div className={s.kpi}><div className={s.kpiK}>Not scheduled</div>
            <div className={s.kpiV}>{kpi.idle}</div>
            <div className={s.kpiN}>no cell today, or gated out</div></div>
          <div className={s.kpi}><div className={s.kpiK}>Lifetime · all books</div>
            <div className={`${s.kpiV} ${cls(tot)}`}>{inr(tot)}</div>
            <div className={s.kpiN}>at each book's own size</div></div>
        </div>
      )}

      <div className={s.panel}>
        <div className={s.ptitle}>Straddle &amp; fly systems</div>
        <div className={s.scroll}>
          <table className={s.table}>
            <thead><tr>
              <th>System</th><th>Size</th><th>Window</th><th>State</th>
              <th>Today</th><th>Running</th><th>Lifetime</th><th>Record</th><th>Evidence</th>
            </tr></thead>
            <tbody>
              {groups.map(([gk, glabel, list]) => list.length === 0 ? null : (
                <>
                  <tr className={s.groupRow} key={gk}><td colSpan={9}>{glabel}</td></tr>
                  {list.map((x) => {
                    const isOpen = open === x.key;
                    return (
                      <>
                        <tr className={s.sysRow} key={x.key} aria-expanded={isOpen}
                            onClick={() => setOpen(isOpen ? null : x.key)}>
                          <td>
                            <span className={`${s.caret} ${isOpen ? s.caretOpen : ''}`}>▶</span>
                            <span className={s.name}>{x.name}</span>
                            <span className={`${s.chip} ${x.money === 'real' ? s.chipReal
                              : x.money === 'refuted' ? s.chipDead : s.chipPaper}`}>
                              {x.money === 'real' ? '● Real money'
                                : x.money === 'refuted' ? 'Refuted' : 'Paper'}
                            </span>
                            <button className={s.rbtn} title="Rules" aria-label="Rules"
                                    onClick={(e) => { e.stopPropagation(); setRules(x); }}>R</button>
                            {x.subtitle && <div className={s.sub2} style={{ paddingLeft: 13 }}>{x.subtitle}</div>}
                          </td>
                          <td>{x.size_lots ?? '—'} lots
                            {x.size_qty ? <div className={s.sub2}>qty {x.size_qty}</div> : null}</td>
                          <td>{x.window}</td>
                          <td><span className={`${s.chip} ${x.state.tone === 'neg' ? s.chipReal
                            : x.state.tone === 'pos' ? s.chipPos : ''}`}>{x.state.label}</span></td>
                          <td className={`${cls(x.today_pnl)} ${x.today_pnl === null ? ''
                            : x.today_pnl > 0 ? s.washPos : s.washNeg}`}>{inr(x.today_pnl)}</td>
                          <td className={cls(x.running_pnl)}>{inr(x.running_pnl)}</td>
                          <td className={cls(x.lifetime.net)}>{inr(x.lifetime.net)}</td>
                          <td>{x.lifetime.n}{x.lifetime.win !== null ? ` · ${x.lifetime.win}%` : ''}
                            <div className={s.sub2}>
                              DD {inr(x.lifetime.maxdd)}{x.lifetime.t !== null ? ` · t ${x.lifetime.t}` : ''}
                            </div></td>
                          <td>{x.evidence.method.map((m) => (
                            <span key={m} className={`${s.chip} ${m === 'AlgoTest' ? s.chipAmber
                              : m.includes('bhavcopy') ? s.chipPos : ''}`}>{m}</span>))}
                            {x.evidence.period && <div className={s.sub2}>{x.evidence.period}</div>}</td>
                        </tr>

                        {isOpen && (
                          <tr key={x.key + '-exp'}><td className={s.expTd} colSpan={9}>
                            <div className={s.expIn}>
                              <div className={s.blk}>
                                <div className={s.blab}>Position<span className={s.blabRule} /></div>
                                {x.legs.length > 0 ? (
                                  <>
                                    <div className={`${s.box} ${s.boxFlush}`}>
                                      <table className={s.inner}>
                                        <thead><tr><th>Leg</th><th>Strike</th><th>Qty</th>
                                          <th>Entry</th><th>LTP</th><th>P&amp;L</th></tr></thead>
                                        <tbody>{x.legs.map((l, i) => (
                                          <tr key={i}>
                                            <td><b className={l.side === 'SELL' ? s.neg : s.pos}>
                                              {l.side} {l.type}</b></td>
                                            <td>{l.strike}</td><td>{l.qty}</td>
                                            <td>{l.entry}</td><td>{l.ltp}</td>
                                            <td className={cls(l.pnl)}>{inr(l.pnl)}</td>
                                          </tr>))}</tbody>
                                      </table>
                                    </div>
                                    {x.curve.length > 1 && (
                                      <>
                                        <div className={s.blab} style={{ marginTop: 14 }}>
                                          P&amp;L since entry<span className={s.blabRule} /></div>
                                        <div className={s.box}><Curve data={x.curve} /></div>
                                      </>
                                    )}
                                  </>
                                ) : (
                                  <div className={s.box}><div className={s.empty}>
                                    Flat — {x.state.label.toLowerCase()}.</div></div>
                                )}

                                <div className={s.blab} style={{ marginTop: 14 }}>
                                  Closed trades<span className={s.blabRule} /></div>
                                <div className={`${s.box} ${s.boxFlush}`}>
                                  <div className={s.hist}>
                                    {x.closed.length === 0
                                      ? <div className={s.empty} style={{ padding: '10px 12px' }}>
                                          No closed trades yet.</div>
                                      : <table className={s.inner}>
                                          <thead><tr><th>Date</th><th>Strike</th><th>Credit</th>
                                            <th>Exit</th><th>Reason</th><th>P&amp;L</th></tr></thead>
                                          <tbody>{x.closed.map((t, i) => (
                                            <tr key={i}>
                                              <td>{t.day}</td><td>{t.strike ?? '—'}</td>
                                              <td>{t.credit ?? '—'}</td><td>{t.exit ?? '—'}</td>
                                              <td>{t.reason ?? '—'}</td>
                                              <td className={cls(t.pnl)}>{inr(t.pnl)}</td>
                                            </tr>))}</tbody>
                                        </table>}
                                  </div>
                                </div>
                              </div>

                              <div className={s.blk}>
                                <div className={s.blab}>
                                  Track record · {x.money === 'real' ? 'real money' : 'paper'}
                                  <span className={s.blabRule} /></div>
                                <div className={s.mini}>
                                  <div className={s.m}><div className={s.mK}>Net</div>
                                    <div className={`${s.mV} ${cls(x.lifetime.net)}`}>{inr(x.lifetime.net)}</div></div>
                                  <div className={s.m}><div className={s.mK}>Trades</div>
                                    <div className={s.mV}>{x.lifetime.n}</div></div>
                                  {x.lifetime.win !== null && <div className={s.m}>
                                    <div className={s.mK}>Win</div>
                                    <div className={s.mV}>{x.lifetime.win}%</div></div>}
                                  <div className={s.m}><div className={s.mK}>Max DD</div>
                                    <div className={s.mV}>{inr(x.lifetime.maxdd)}</div></div>
                                  {x.lifetime.t !== null && <div className={s.m}>
                                    <div className={s.mK}>t</div>
                                    <div className={`${s.mV} ${cls(x.lifetime.t)}`}>{x.lifetime.t}</div></div>}
                                </div>
                                <div className={s.blab}>Evidence<span className={s.blabRule} /></div>
                                <Evid e={x.evidence} />
                              </div>
                            </div>
                          </td></tr>
                        )}
                      </>
                    );
                  })}
                </>
              ))}
              <tr className={s.totalRow}>
                <td>Total · {feed.n} systems</td><td /><td /><td />
                <td className={cls(kpi?.today ?? 0)}>{inr(kpi?.today ?? 0)}</td>
                <td /><td className={cls(tot)}>{inr(tot)}</td><td /><td />
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {rules && <RulesModal sys={rules} onClose={() => setRules(null)} />}
    </div>
  );
}
