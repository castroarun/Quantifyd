/**
 * Strategies — the register of record for every system we run.
 *
 * Two views over the same registry (frontend/src/data/strategies.ts):
 *   Register — one row per system, grouped by whose money is at risk.
 *   Spec     — rail + full spec: rules as they run, evidence, change log.
 *
 * Day P&L is shown only for systems with a live feed wired (see dayPnlFeed);
 * everything else reads "—" and is read on its own dashboard, rather than
 * inventing a number here.
 */

import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './Strategies.module.css';
import { apiGet } from '../api/client';
import type { ORBState, NASState, StrangleState } from '../api/types';
import { formatPnl, pnlClass } from '../utils/format';
import {
  SYSTEMS,
  STATUS_LABEL,
  REGISTER_UPDATED,
  LAB_PAGES,
} from '../data/strategies';
import type { DayPnlFeed, StrategySystem, SystemStatus } from '../data/strategies';

/** One book's own trade record, from /api/books/liveness (read-only projection). */
export interface BookLiveness {
  trades: number;
  last_trade: string | null;
  days_idle: number | null;
  trades_30d: number;
  net_total: number | null;
  net_30d: number | null;
  win_rate: number | null;
  series: Array<{ d: string; c: number }>;
  /** Positions held right now — a book with no exits yet is still working. */
  open_positions?: number;
  last_entry?: string | null;
  /** Days since the last entry OR exit, whichever is later. */
  days_since_activity?: number | null;
}

/** Tiny cumulative-P&L sparkline. No axes — it answers "shape?", not "how much?". */
function Spark({ series }: { series: Array<{ d: string; c: number }> }) {
  if (!series || series.length < 2) return null;
  const vals = series.map((p) => p.c);
  const min = Math.min(...vals, 0);
  const max = Math.max(...vals, 0);
  const span = max - min || 1;
  const w = 132;
  const h = 30;
  const pts = series.map((p, i) => {
    const x = (i / (series.length - 1)) * w;
    const y = h - ((p.c - min) / span) * h;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const last = vals[vals.length - 1];
  const zeroY = h - ((0 - min) / span) * h;
  return (
    <svg className={styles.spark} width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden="true">
      <line x1="0" y1={zeroY} x2={w} y2={zeroY} className={styles.sparkZero} />
      <polyline points={pts.join(' ')} className={last >= 0 ? styles.sparkPos : styles.sparkNeg} />
    </svg>
  );
}

/** "3d ago" / "today" / "—", with a warning tone once a book goes quiet.
 *  A book that has entered but never exited reports its entry, not a dash —
 *  holding a position is activity, and calling it "—" reads as broken. */
function IdleCell({ lv }: { lv?: BookLiveness }) {
  const held = lv?.open_positions ?? 0;
  if (!lv || (!lv.last_trade && !held)) return <span className={styles.dash}>—</span>;
  const d = lv.days_since_activity ?? lv.days_idle;
  const label = d === 0 ? 'today' : d === 1 ? 'yesterday' : `${d}d ago`;
  const cls = d != null && d >= 30 ? styles.idleStale : d != null && d >= 7 ? styles.idleWarn : '';
  const title = [
    lv.last_trade ? `last exit ${lv.last_trade}` : 'no exit yet',
    lv.last_entry ? `last entry ${lv.last_entry}` : null,
    held ? `holding ${held}` : null,
    `${lv.trades} closed all-time`,
  ].filter(Boolean).join(' · ');
  return (
    <span className={cls} title={title}>
      {label}
      {held ? <span className={styles.dash}> · holds {held}</span> : null}
    </span>
  );
}

// All 8 NIFTY NAS sub-system endpoints — the suite's day P&L is their sum.
const NAS_SYSTEMS = [
  'nas',
  'nas-atm',
  'nas-atm2',
  'nas-atm4',
  'nas-916-otm',
  'nas-916-atm',
  'nas-916-atm2',
  'nas-916-atm4',
] as const;

const VIEW_KEY = 'qf.strategies.view';
const GROUPS: SystemStatus[] = ['live', 'paper', 'parked'];

// Polled /api/{key}/state returns p.ltp=None for some open legs (the 916
// executors write to separate ticker caches). The NAS page masks this by
// reading LTPs off /api/nas/stream; this aggregator does the same or it
// under-reports unrealized and shows only the realized loss.
function aggregateNasDayPnl(
  states: (NASState | null)[],
  liveLegLtps: Record<string, number>,
): number {
  let dayPnl = 0;
  for (const s of states) {
    if (!s) continue;
    dayPnl += (s.stats?.today_pnl as number | undefined) ?? 0;
    const legs = [...(s.positions?.ce ?? []), ...(s.positions?.pe ?? [])];
    for (const p of legs) {
      const entry = p.entry_price ?? p.entry_premium;
      const liveLtp = p.tradingsymbol ? liveLegLtps[p.tradingsymbol] : undefined;
      const ltp = liveLtp ?? p.ltp;
      const qty = p.qty ?? 0;
      if (entry != null && ltp != null && qty) dayPnl += (entry - ltp) * qty;
    }
  }
  return Math.round(dayPnl * 100) / 100;
}

export default function Strategies() {
  const [view, setView] = useState<'register' | 'spec'>(() => {
    try {
      return localStorage.getItem(VIEW_KEY) === 'spec' ? 'spec' : 'register';
    } catch {
      return 'register';
    }
  });
  const [selectedId, setSelectedId] = useState<string>(SYSTEMS[0].id);
  const [orb, setOrb] = useState<ORBState | null>(null);
  const [strangle, setStrangle] = useState<StrangleState | null>(null);
  const [nasStates, setNasStates] = useState<(NASState | null)[]>([]);
  const [sensexDayPnl, setSensexDayPnl] = useState<number | null>(null);
  const [liveness, setLiveness] = useState<Record<string, BookLiveness>>({});
  const [liveLegLtps, setLiveLegLtps] = useState<Record<string, number>>({});
  const [err, setErr] = useState<string | null>(null);
  const evtRef = useRef<EventSource | null>(null);

  useEffect(() => {
    try {
      localStorage.setItem(VIEW_KEY, view);
    } catch {
      /* ignore */
    }
  }, [view]);

  // Each book's own trade record. Absent until the backend restarts, so a
  // failure here must leave the page working with em-dashes.
  useEffect(() => {
    let cancelled = false;
    const load = () =>
      fetch(`/api/books/liveness?t=${Date.now()}`, { cache: 'no-store' })
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => {
          if (!cancelled && d?.books) setLiveness(d.books);
        })
        .catch(() => undefined);
    load();
    const id = setInterval(load, 120_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const [o, st, ...ns] = await Promise.all([
          apiGet<ORBState>('/api/orb/state').catch(() => null),
          apiGet<StrangleState>('/api/strangle/state').catch(() => null),
          ...NAS_SYSTEMS.map((s) => apiGet<NASState>(`/api/${s}/state`).catch(() => null)),
        ]);
        // SENSEX arms are written by a standalone writer, not a Flask route.
        const sx = await fetch(`/app/sensex_live.json?t=${Date.now()}`, { cache: 'no-store' })
          .then((r) => (r.ok ? (r.json() as Promise<{ day_pnl?: number }>) : null))
          .catch(() => null);
        if (cancelled) return;
        setOrb(o);
        setStrangle(st);
        setSensexDayPnl(typeof sx?.day_pnl === 'number' ? sx.day_pnl : null);
        setNasStates(ns);
      } catch (e) {
        if (!cancelled) setErr(e instanceof Error ? e.message : 'Failed to load');
      }
    };
    load();
    const id = setInterval(load, 5_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  // NAS SSE tick stream — live LTPs for every open leg.
  useEffect(() => {
    let cancelled = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    const open = () => {
      if (cancelled) return;
      const es = new EventSource('/api/nas/stream');
      evtRef.current = es;
      es.onmessage = (ev) => {
        if (cancelled) return;
        try {
          const d = JSON.parse(ev.data);
          if (d.type === 'tick' && d.legs) {
            const next: Record<string, number> = {};
            for (const [tsym, info] of Object.entries(d.legs)) {
              const ltp = (info as { ltp?: number }).ltp;
              if (typeof ltp === 'number') next[tsym] = ltp;
            }
            setLiveLegLtps(next);
          }
        } catch {
          /* ignore malformed payload */
        }
      };
      es.onerror = () => {
        if (cancelled) return;
        es.close();
        evtRef.current = null;
        reconnectTimer = setTimeout(open, 3000);
      };
    };
    open();
    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (evtRef.current) {
        evtRef.current.close();
        evtRef.current = null;
      }
    };
  }, []);

  const dayPnl = useMemo<Partial<Record<DayPnlFeed, number>>>(() => {
    const strangleTotal = (strangle?.variants ?? []).reduce(
      (s, v) => s + (v.today_pnl ?? 0),
      0,
    );
    const out: Partial<Record<DayPnlFeed, number>> = {
      'nas-nifty': aggregateNasDayPnl(nasStates, liveLegLtps),
    };
    if (orb) out.orb = orb.today_pnl ?? 0;
    if (strangle) out.strangle = strangleTotal;
    if (sensexDayPnl != null) out['nas-sensex'] = sensexDayPnl;
    return out;
  }, [orb, strangle, nasStates, liveLegLtps, sensexDayPnl]);

  const counts = useMemo(() => {
    const c: Record<SystemStatus, number> = { live: 0, paper: 0, parked: 0 };
    for (const s of SYSTEMS) c[s.status] += 1;
    return c;
  }, []);

  const openSpec = useCallback((id: string) => {
    setSelectedId(id);
    setView('spec');
  }, []);

  const selected = SYSTEMS.find((s) => s.id === selectedId) ?? SYSTEMS[0];

  return (
    <div className={styles.root}>
      <div className={styles.head}>
        <div>
          <div className="page-title">Strategies</div>
          <div className="page-subtitle">
            The register of record — {counts.live} live with real money · {counts.paper} on paper ·{' '}
            {counts.parked} parked. Updated {REGISTER_UPDATED}.
          </div>
        </div>
        <div className={styles.switch} role="group" aria-label="View">
          <button
            type="button"
            className={`${styles.switchBtn} ${view === 'register' ? styles.switchOn : ''}`}
            onClick={() => setView('register')}
            aria-pressed={view === 'register'}
          >
            Register
          </button>
          <button
            type="button"
            className={`${styles.switchBtn} ${view === 'spec' ? styles.switchOn : ''}`}
            onClick={() => setView('spec')}
            aria-pressed={view === 'spec'}
          >
            Spec
          </button>
        </div>
      </div>

      {err ? <div className={styles.error}>Failed to load live state: {err}</div> : null}

      {view === 'register' ? (
        <RegisterView dayPnl={dayPnl} liveness={liveness} onOpenSpec={openSpec} />
      ) : (
        <SpecView
          selected={selected}
          onSelect={setSelectedId}
          dayPnl={dayPnl}
          liveness={liveness}
        />
      )}

      <div className={styles.labs}>
        <span className={styles.labsLabel}>Labs &amp; research pages — no capital</span>
        {LAB_PAGES.map((l) => (
          <Link key={l.to} to={l.to} className={styles.labLink} title={l.what}>
            {l.name}
          </Link>
        ))}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ register */

function RegisterView({
  dayPnl,
  liveness,
  onOpenSpec,
}: {
  dayPnl: Partial<Record<DayPnlFeed, number>>;
  liveness: Record<string, BookLiveness>;
  onOpenSpec: (id: string) => void;
}) {
  return (
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th className={styles.stripeCell} />
            <th>System</th>
            <th>Size</th>
            <th>The rule, in one line</th>
            <th className={styles.r}>Day</th>
            <th className={styles.r}>Last trade</th>
            <th className={styles.r}>30d</th>
            <th className={styles.r}>Book net</th>
            <th>Rules &amp; evidence</th>
          </tr>
        </thead>
        <tbody>
          {GROUPS.map((group) => {
            const rows = SYSTEMS.filter((s) => s.status === group);
            if (!rows.length) return null;
            return (
              <Fragment key={group}>
                <tr className={styles.groupRow}>
                  <td className={styles.stripeCell} />
                  <td colSpan={8}>
                    <div className={styles.groupLabel}>
                      {STATUS_LABEL[group]}
                      <span className={styles.groupCount}>
                        {rows.length} {rows.length === 1 ? 'system' : 'systems'}
                      </span>
                    </div>
                  </td>
                </tr>
                {rows.map((s) => {
                  const pnl = s.dayPnlFeed ? dayPnl[s.dayPnlFeed] : undefined;
                  return (
                    <tr key={s.id} className={styles[group]}>
                      <td className={styles.stripeCell}>
                        <span className={styles.stripe} />
                      </td>
                      <td>
                        <button
                          type="button"
                          className={styles.sysName}
                          onClick={() => onOpenSpec(s.id)}
                          title="Open the full spec"
                        >
                          {s.name}
                        </button>
                        <div className={styles.sysSub}>{s.subtitle}</div>
                      </td>
                      <td className={styles.num}>{s.size}</td>
                      <td className={styles.rule}>
                        {s.rule}
                        {s.note ? <div className={styles.rowNote}>{s.note}</div> : null}
                      </td>
                      <td className={`${styles.r} ${styles.num}`}>
                        {pnl == null ? (
                          <span className={styles.dash}>—</span>
                        ) : (
                          <span className={pnlClass(pnl)}>{formatPnl(pnl)}</span>
                        )}
                      </td>
                      <td className={`${styles.r} ${styles.num} ${styles.since}`}>
                        <IdleCell lv={liveness[s.id]} />
                      </td>
                      <td className={`${styles.r} ${styles.num} ${styles.since}`}>
                        {liveness[s.id]?.trades_30d ?? <span className={styles.dash}>—</span>}
                      </td>
                      <td className={`${styles.r} ${styles.num}`}>
                        {liveness[s.id]?.net_total == null ? (
                          <span className={styles.dash}>—</span>
                        ) : (
                          <span className={pnlClass(liveness[s.id].net_total as number)}>
                            {formatPnl(liveness[s.id].net_total as number)}
                          </span>
                        )}
                      </td>
                      <td>
                        <div className={styles.links}>
                          <button
                            type="button"
                            className={styles.lk}
                            onClick={() => onOpenSpec(s.id)}
                          >
                            Rules
                          </button>
                          <StudyLink system={s} onOpenSpec={onOpenSpec} />
                          {s.dashboard ? (
                            <Link className={styles.lk} to={s.dashboard}>
                              Dashboard
                            </Link>
                          ) : null}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function StudyLink({
  system,
  onOpenSpec,
}: {
  system: StrategySystem;
  onOpenSpec: (id: string) => void;
}) {
  if (!system.studies.length) {
    return (
      <span className={`${styles.lk} ${styles.lkGap}`} title={system.studyGap ?? 'No published study'}>
        Study pending
      </span>
    );
  }
  if (system.studies.length === 1) {
    return (
      <Link className={styles.lk} to={`/backtest/${system.studies[0].slug}`}>
        Study
      </Link>
    );
  }
  return (
    <button type="button" className={styles.lk} onClick={() => onOpenSpec(system.id)}>
      Study ×{system.studies.length}
    </button>
  );
}

/* ---------------------------------------------------------------------- spec */

function SpecView({
  selected,
  onSelect,
  dayPnl,
  liveness,
}: {
  selected: StrategySystem;
  onSelect: (id: string) => void;
  dayPnl: Partial<Record<DayPnlFeed, number>>;
  liveness: Record<string, BookLiveness>;
}) {
  const pnl = selected.dayPnlFeed ? dayPnl[selected.dayPnlFeed] : undefined;
  const lv = liveness[selected.id];
  return (
    <div className={styles.split}>
      <div className={styles.rail}>
        {GROUPS.map((group) => {
          const rows = SYSTEMS.filter((s) => s.status === group);
          if (!rows.length) return null;
          return (
            <div key={group}>
              <div className={styles.railGroup}>
                <div className={styles.groupLabel}>
                  {STATUS_LABEL[group]}
                  <span className={styles.groupCount}>{rows.length}</span>
                </div>
              </div>
              {rows.map((s) => {
                const rp = s.dayPnlFeed ? dayPnl[s.dayPnlFeed] : undefined;
                return (
                  <button
                    type="button"
                    key={s.id}
                    className={`${styles.railRow} ${s.id === selected.id ? styles.railOn : ''}`}
                    onClick={() => onSelect(s.id)}
                  >
                    <span className={`${styles.dot} ${styles[`dot_${group}`]}`} />
                    <span className={styles.railName}>{s.name}</span>
                    <span className={`${styles.num} ${rp == null ? styles.dash : pnlClass(rp)}`}>
                      {rp == null ? '—' : formatPnl(rp)}
                    </span>
                  </button>
                );
              })}
            </div>
          );
        })}
      </div>

      <div className={styles.spec}>
        <div className={styles.specHead}>
          <div>
            <h2 className={styles.specName}>{selected.name}</h2>
            <p className={styles.specSub}>{selected.subtitle}</p>
          </div>
          <div className={styles.specActions}>
            {selected.dashboard ? (
              <Link className={styles.lk} to={selected.dashboard}>
                Dashboard
              </Link>
            ) : null}
          </div>
        </div>

        {/* Inline stat line — no boxes. */}
        <div className={styles.statLine}>
          <div className={styles.stat}>
            <span className={styles.statK}>Status</span>
            <span className={`${styles.statV} ${styles[`ink_${selected.status}`]}`}>
              {STATUS_LABEL[selected.status]}
            </span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statK}>Size</span>
            <span className={`${styles.statV} ${styles.num}`}>{selected.size}</span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statK}>Day P&amp;L</span>
            <span className={`${styles.statV} ${styles.num} ${pnl == null ? styles.dash : pnlClass(pnl)}`}>
              {pnl == null ? '—' : formatPnl(pnl)}
            </span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statK}>Since</span>
            <span className={`${styles.statV} ${styles.num}`}>{selected.since}</span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statK}>Last trade</span>
            <span className={`${styles.statV} ${styles.num}`}>
              <IdleCell lv={lv} />
            </span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statK}>Book net</span>
            <span className={`${styles.statV} ${styles.num} ${lv?.net_total == null ? styles.dash : pnlClass(lv.net_total)}`}>
              {lv?.net_total == null ? '—' : formatPnl(lv.net_total)}
            </span>
          </div>
        </div>

        {lv && lv.trades > 0 ? (
          <div className={styles.activity}>
            <Spark series={lv.series} />
            <div className={styles.activityText}>
              <b>{lv.trades}</b> trades all-time · <b>{lv.trades_30d}</b> in the last 30 days
              {lv.win_rate != null ? <> · <b>{lv.win_rate.toFixed(0)}%</b> wins</> : null}
              {lv.net_30d != null ? <> · 30d {formatPnl(lv.net_30d)}</> : null}
            </div>
          </div>
        ) : null}

        {selected.note ? <p className={styles.specNote}>{selected.note}</p> : null}

        <section className={styles.block}>
          <div className={styles.blockTitle}>The rules, as they run today</div>
          <dl className={styles.ruleList}>
            {selected.rules.map(([k, v]) => (
              <div className={styles.ruleRow} key={k}>
                <dt>{k}</dt>
                <dd>{v}</dd>
              </div>
            ))}
          </dl>
          {selected.rulesDoc ? (
            <div className={styles.docPath}>
              Source of truth: <code>{selected.rulesDoc}</code>
            </div>
          ) : null}
        </section>

        <section className={styles.block}>
          <div className={styles.blockTitle}>Evidence</div>
          {selected.studies.length ? (
            <div className={styles.evid}>
              {selected.studies.map((st) => (
                <Link key={st.slug} className={styles.ev} to={`/backtest/${st.slug}`}>
                  {st.verdict ? (
                    <span className={`${styles.vd} ${styles[`vd_${st.verdict.split(/[ -]/)[0].toLowerCase()}`]}`}>
                      {st.verdict}
                    </span>
                  ) : null}
                  <span className={styles.evTitle}>{st.title}</span>
                </Link>
              ))}
            </div>
          ) : (
            <p className={styles.gap}>{selected.studyGap ?? 'No published study yet.'}</p>
          )}
        </section>

        <section className={styles.block}>
          <div className={styles.blockTitle}>Change log</div>
          <div className={styles.log}>
            {selected.changeLog.map((c, i) => (
              <div className={styles.logRow} key={`${c.date}-${i}`}>
                <span className={`${styles.logDate} ${styles.num}`}>{c.date}</span>
                <span>{c.text}</span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
