/**
 * /app/mpf-report — THE MOMENTUM PORTFOLIO, one page.
 *
 * This is the entry point to the book. The individual study pages under /app/backtest/<slug>
 * remain the archive and every section here links back to them; nothing was removed.
 *
 * Every number rendered below comes from /app/mpf_report.json, written by
 * research/_utilities/mpf_report_build.py. The narrative comes from data/mpf_report.ts.
 * Nothing on this page is a hand-typed portfolio figure.
 *
 * Editorial rules, binding (Arun, 11-Sep-2026):
 *   POST-TAX ONLY · every table states WHICH SYSTEMS / WHICH WINDOW / WHICH BASIS ·
 *   systems are named, never versioned · test period, average invested and average in cash
 *   go on the table · questions are answered in Q&A form · the correction leads, before
 *   anyone is shown a CAGR.
 */
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './MpfReport.module.css';
import Chip from '../components/Chip/Chip';
import StatusDot from '../components/StatusDot/StatusDot';
import { SYSTEMS, REVIEWS, OWED, HIGHLIGHTS, REPORT_DATE } from '../data/mpf_report';
import type { SystemReport } from '../data/mpf_report';

/* ---------------------------------------------------------------- json shape */

interface Row {
  cagr: number;
  maxdd: number;
  calmar: number;
  growth100: number;
  invested: number | null;
  cash: number | null;
  seriesSpan: string;
}
interface Block {
  window: [string, string];
  years: number;
  windowWhy: string;
  source: string;
  basis: string;
  order: string[];
  contenders: string[];
  benchmarks: string[];
  rows: Record<string, Row>;
  yoy: Record<string, Record<string, [number, number]>>;
  bestOf: Record<string, { bestCagr: string; leastDD: string; bestOverall: string }>;
  yearList: string[];
  corr: Record<string, Record<string, number>>;
  corrOrder: string[];
  weeks: number;
  qsOffsets?: { drawn: string; n: number; median: number; min: number; max: number };
}
interface Report {
  generated: string;
  generator: string;
  standard: string;
  names: Record<string, string>;
  blendNote: string;
  investedSources: Record<string, string>;
  headline: Block;
  window2018: Block;
  correction: {
    headline: string;
    costs: { book: string; published: string; honest: string }[];
    proof: string[];
    entrySurface: { trails: number[]; rows: { label: string; placeable: boolean; cagr: (number | null)[] }[] };
    nullControl: { trails: number[]; rows: { label: string; cagr: (number | null)[] }[]; edge: (number | null)[] };
    gateBakeoff: { gate: string; cagr: number; maxdd: number; calmar: number; blockedPct: number }[];
    vixGates: { gate: string; cagr: number; maxdd: number; calmar: number; blockedPct: number }[];
    athVix: { label: string; cagr: number; maxdd: number; calmar: number; window: string; source: string };
  };
  notes: Record<string, string>;
  sources: Record<string, string>;
  charts: {
    curves20y: string;
    curves2018: string;
    yearlyBars: string;
    rolling3y: string;
    corr20y: string;
    corr2018: string;
    invested: string;
    heat: Record<string, string>;
  };
}

/* ---------------------------------------------------------------- helpers */

const pc = (n: number | null | undefined, d = 1) =>
  n === null || n === undefined ? 'not measured' : `${n > 0 ? '' : ''}${n.toFixed(d)}%`;

const sgn = (n: number, d = 1) => `${n > 0 ? '+' : ''}${n.toFixed(d)}`;

function Section({
  num,
  label,
  title,
  intro,
  children,
}: {
  num: string;
  label: string;
  title: string;
  intro?: string;
  children: React.ReactNode;
}) {
  return (
    <section className={styles.section}>
      <div className={styles.sectionHead}>
        <span className={styles.sectionNum}>{num}</span>
        <span className={styles.sectionLabel}>{label}</span>
      </div>
      <h2 className={styles.sectionTitle}>{title}</h2>
      {intro ? <p className={styles.sectionIntro}>{intro}</p> : null}
      {children}
    </section>
  );
}

function Figure({ src, caption }: { src: string; caption: string }) {
  return (
    <figure className={styles.figure}>
      <img src={src} alt={caption} className={styles.figureImg} loading="lazy" />
      <figcaption className={styles.figureCap}>{caption}</figcaption>
    </figure>
  );
}

/** Every table on this page is wrapped so it scrolls inside its own box on a phone
 *  instead of squeezing the page or running off-screen. */
function TableBox({
  title,
  caption,
  children,
}: {
  title: string;
  caption?: string;
  children: React.ReactNode;
}) {
  return (
    <div className={styles.tableBlock}>
      <div className={styles.tableTitle}>{title}</div>
      {caption ? <div className={styles.tableCaption}>{caption}</div> : null}
      <div className={styles.tableWrap}>{children}</div>
    </div>
  );
}

/* ---------------------------------------------------------------- summary table */

function SummaryTable({
  block,
  blockLabel,
  smallRows,
  blendKey,
  blendNote,
  investedNote,
}: {
  block: Block;
  blockLabel: string;
  smallRows: string[];
  blendKey: string;
  blendNote: string;
  investedNote: string;
}) {
  return (
    <TableBox
      title={blockLabel}
      caption={`${block.windowWhy} BASIS: ${block.basis} SOURCE: ${block.source}`}
    >
      <table className={styles.tbl}>
        <thead>
          <tr>
            <th className={styles.thLeft}>System</th>
            <th>Series available</th>
            <th>CAGR</th>
            <th>Max DD</th>
            <th>Calmar</th>
            <th>Growth of 100</th>
            <th>Avg invested</th>
            <th>Avg in cash</th>
          </tr>
        </thead>
        <tbody>
          {block.order.map((k) => {
            const r = block.rows[k];
            const isBench = block.benchmarks.includes(k);
            const cls = [
              smallRows.includes(k) ? styles.rowSmall : '',
              isBench ? styles.rowBench : '',
              k === blendKey ? styles.rowBlend : '',
            ]
              .filter(Boolean)
              .join(' ');
            return (
              <tr key={k} className={cls}>
                <td className={styles.thLeft}>
                  <span className={styles.sysName}>{k}</span>
                  {k === blendKey ? <span className={styles.tagComputed}>computed here</span> : null}
                  {isBench ? <span className={styles.tagBench}>benchmark</span> : null}
                </td>
                <td className={styles.mut}>{r.seriesSpan}</td>
                <td className={styles.num}>{r.cagr.toFixed(2)}%</td>
                <td className={styles.numNeg}>{r.maxdd.toFixed(2)}%</td>
                <td className={styles.num}>{r.calmar.toFixed(2)}</td>
                <td className={styles.num}>{r.growth100.toLocaleString()}</td>
                <td className={r.invested === null ? styles.gap : styles.num}>
                  {r.invested === null ? 'not measured' : pc(r.invested, 0)}
                </td>
                <td className={r.cash === null ? styles.gap : styles.num}>
                  {r.cash === null ? '—' : pc(r.cash, 0)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className={styles.tableFoot}>
        <strong>Cash-yield caveat.</strong> {investedNote}
      </div>
      <div className={styles.tableFoot}>
        <strong>The blend row.</strong> {blendNote}
      </div>
    </TableBox>
  );
}

/* ---------------------------------------------------------------- YoY table */

function YoyTable({ block, label }: { block: Block; label: string }) {
  const cols = block.order;
  return (
    <TableBox
      title={label}
      caption={`Each cell is the year’s return with that year’s worst fall beneath it in small type — measured from the running peak of the FULL curve, never from the year’s first bar. The three best-of columns exclude the benchmark. WINDOW ${block.window[0]} to ${block.window[1]}. BASIS: ${block.basis}`}
    >
      <table className={`${styles.tbl} ${styles.tblYoy}`}>
        <thead>
          <tr>
            <th className={styles.thLeft}>Year</th>
            {cols.map((c) => (
              <th key={c} className={block.benchmarks.includes(c) ? styles.thBench : undefined}>
                {c}
              </th>
            ))}
            <th className={styles.thBest}>Best CAGR</th>
            <th className={styles.thBest}>Least DD</th>
            <th className={styles.thBest}>Best overall</th>
          </tr>
        </thead>
        <tbody>
          {block.yearList.map((y) => {
            const bo = block.bestOf[y];
            return (
              <tr key={y}>
                <td className={styles.thLeft}>{y}</td>
                {cols.map((c) => {
                  const cell = block.yoy[c]?.[y];
                  if (!cell) return <td key={c} className={styles.mut}>—</td>;
                  const [ret, dd] = cell;
                  return (
                    <td key={c}>
                      <div className={ret >= 0 ? styles.cellPos : styles.cellNeg}>{sgn(ret)}</div>
                      <div className={styles.cellDd}>({dd.toFixed(1)}%)</div>
                    </td>
                  );
                })}
                <td className={styles.best}>{bo?.bestCagr ?? '—'}</td>
                <td className={styles.best}>{bo?.leastDD ?? '—'}</td>
                <td className={styles.best}>{bo?.bestOverall ?? '—'}</td>
              </tr>
            );
          })}
          <tr className={styles.rowTotal}>
            <td className={styles.thLeft}>FULL WINDOW</td>
            {cols.map((c) => {
              const r = block.rows[c];
              return (
                <td key={c}>
                  <div className={styles.cellPos}>{r.cagr.toFixed(1)}</div>
                  <div className={styles.cellDd}>({r.maxdd.toFixed(1)}%)</div>
                </td>
              );
            })}
            <td className={styles.best} colSpan={3}>
              {block.years} years · Calmar leader:{' '}
              {
                block.contenders.reduce((a, b) =>
                  block.rows[a].calmar >= block.rows[b].calmar ? a : b,
                )
              }
            </td>
          </tr>
        </tbody>
      </table>
    </TableBox>
  );
}

/* ---------------------------------------------------------------- gate bake-off */

function GateTable({
  rows,
  title,
  caption,
}: {
  rows: { gate: string; cagr: number; maxdd: number; calmar: number; blockedPct: number }[];
  title: string;
  caption: string;
}) {
  return (
    <TableBox title={title} caption={caption}>
      <table className={styles.tbl}>
        <thead>
          <tr>
            <th className={styles.thLeft}>Gate</th>
            <th>CAGR</th>
            <th>Max DD</th>
            <th>Calmar</th>
            <th>Days blocked</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((g) => {
            const base = g.gate === 'none';
            return (
              <tr key={g.gate} className={base ? styles.rowTotal : undefined}>
                <td className={styles.thLeft}>{base ? 'no gate (baseline)' : g.gate}</td>
                <td className={styles.num}>{g.cagr.toFixed(2)}%</td>
                <td className={styles.numNeg}>{g.maxdd.toFixed(2)}%</td>
                <td className={styles.num}>{g.calmar.toFixed(3)}</td>
                <td className={styles.num}>{g.blockedPct.toFixed(1)}%</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </TableBox>
  );
}

/* ---------------------------------------------------------------- correlation */

function CorrTable({ block, label }: { block: Block; label: string }) {
  const keys = block.corrOrder.filter((k) => block.corr[k]);
  return (
    <TableBox
      title={label}
      caption={`Weekly returns, ${block.weeks} weeks, ${block.window[0]} to ${block.window[1]}, after tax.`}
    >
      <table className={styles.tbl}>
        <thead>
          <tr>
            <th className={styles.thLeft} />
            {keys.map((k) => (
              <th key={k}>{k}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {keys.map((r) => (
            <tr key={r}>
              <td className={styles.thLeft}>{r}</td>
              {keys.map((c) => {
                const v = block.corr[r]?.[c];
                return (
                  <td key={c} className={r === c ? styles.mut : styles.num}>
                    {v === undefined ? '—' : v.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </TableBox>
  );
}

/* ---------------------------------------------------------------- system card */

function SystemCard({ s, r, window2018 }: { s: SystemReport; r?: Row; window2018?: Row }) {
  const shown = r ?? window2018;
  return (
    <div className={`${styles.card} ${styles[`ac_${s.accent}`]}`}>
      <div className={styles.cardHead}>
        <div className={styles.cardName}>{s.name}</div>
        <StatusDot
          kind={s.kind === 'live' ? 'live' : s.kind === 'paper' ? 'paper' : 'off'}
          title={s.statusLabel}
        />
      </div>
      <Chip className={styles[`chip_${s.accent}`]}>{s.statusLabel}</Chip>
      <div className={styles.cardRule}>{s.rule}</div>
      {shown ? (
        <div className={styles.cardStats}>
          <div>
            <div className={styles.statLabel}>CAGR</div>
            <div className={styles.statValue}>{shown.cagr.toFixed(1)}%</div>
          </div>
          <div>
            <div className={styles.statLabel}>Max DD</div>
            <div className={styles.statValue}>{shown.maxdd.toFixed(1)}%</div>
          </div>
          <div>
            <div className={styles.statLabel}>Calmar</div>
            <div className={styles.statValue}>{shown.calmar.toFixed(2)}</div>
          </div>
        </div>
      ) : null}
      <div className={styles.cardFoot}>
        {r
          ? 'On the full 20.4-year window, after tax.'
          : 'WINDOW: Aug-2018 → Sep-2026 ONLY, after tax — it cannot be measured earlier (needs four filed fiscal years; Screener history starts FY2015). Not comparable to the 20-year cards beside it; see the 2018 section for every book on this window.'}
        <br />
        <span className={styles.mut}>Size: {s.size}</span>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- per-system section */

function Block6({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className={styles.block6}>
      <div className={styles.block6Title}>{title}</div>
      {children}
    </div>
  );
}

function KvList({ rows }: { rows: { k: string; v: string }[] }) {
  return (
    <dl className={styles.kv}>
      {rows.map((r, i) => (
        <div className={styles.kvRow} key={i}>
          <dt>{r.k}</dt>
          <dd>{r.v}</dd>
        </div>
      ))}
    </dl>
  );
}

function SystemSection({
  s,
  data,
  defaultOpen,
}: {
  s: SystemReport;
  data: Report;
  defaultOpen: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const h = data.headline.rows[s.key];
  const w = data.window2018.rows[s.key];
  const heat = s.heatKey ? data.charts.heat[s.heatKey] : undefined;

  return (
    <div className={`${styles.sysSection} ${styles[`ac_${s.accent}`]}`}>
      <button
        className={styles.sysToggle}
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span className={styles.sysCaret}>{open ? '▾' : '▸'}</span>
        <span className={styles.sysTitle}>{s.name}</span>
        <Chip className={styles[`chip_${s.accent}`]}>{s.statusLabel}</Chip>
        <span className={styles.sysNums}>
          {h
            ? `${h.cagr.toFixed(1)}% / ${h.maxdd.toFixed(1)}% / Calmar ${h.calmar.toFixed(2)} — 20.4y`
            : w
              ? `${w.cagr.toFixed(1)}% / ${w.maxdd.toFixed(1)}% / Calmar ${w.calmar.toFixed(2)} — 2018 window`
              : ''}
        </span>
      </button>

      {open ? (
        <div className={styles.sysBody}>
          {s.alert ? <div className={styles.alert}>{s.alert}</div> : null}
          <div className={styles.sysMeta}>
            <span>
              <strong>Size as traded:</strong> {s.size}
            </span>
            <span>
              <strong>Window:</strong> {s.windowNote}
            </span>
          </div>

          <Block6 title="1 · Rules">
            <KvList rows={s.rules} />
          </Block6>

          <Block6 title="2 · Entry and exit mechanics">
            <KvList rows={s.mechanics} />
          </Block6>

          <Block6 title="3 · Evidence">
            <KvList rows={s.evidence} />
            {h || w ? (
              <div className={styles.tableWrap}>
                <table className={styles.tbl}>
                  <thead>
                    <tr>
                      <th className={styles.thLeft}>Measured on</th>
                      <th>CAGR</th>
                      <th>Max DD</th>
                      <th>Calmar</th>
                      <th>Growth of 100</th>
                      <th>Avg invested</th>
                    </tr>
                  </thead>
                  <tbody>
                    {h ? (
                      <tr>
                        <td className={styles.thLeft}>
                          20.4 years · {data.headline.window[0]} to {data.headline.window[1]}
                        </td>
                        <td className={styles.num}>{h.cagr.toFixed(2)}%</td>
                        <td className={styles.numNeg}>{h.maxdd.toFixed(2)}%</td>
                        <td className={styles.num}>{h.calmar.toFixed(2)}</td>
                        <td className={styles.num}>{h.growth100.toLocaleString()}</td>
                        <td className={h.invested === null ? styles.gap : styles.num}>
                          {h.invested === null ? 'not measured' : pc(h.invested, 0)}
                        </td>
                      </tr>
                    ) : null}
                    {w ? (
                      <tr>
                        <td className={styles.thLeft}>
                          2018 window · {data.window2018.window[0]} to {data.window2018.window[1]}
                        </td>
                        <td className={styles.num}>{w.cagr.toFixed(2)}%</td>
                        <td className={styles.numNeg}>{w.maxdd.toFixed(2)}%</td>
                        <td className={styles.num}>{w.calmar.toFixed(2)}</td>
                        <td className={styles.num}>{w.growth100.toLocaleString()}</td>
                        <td className={w.invested === null ? styles.gap : styles.num}>
                          {w.invested === null ? 'not measured' : pc(w.invested, 0)}
                        </td>
                      </tr>
                    ) : null}
                  </tbody>
                </table>
              </div>
            ) : null}
            {heat ? (
              <Figure
                src={heat}
                caption={`${s.name} — monthly returns after tax. Green is a month made, red is a month lost; the annotation is the percentage.`}
              />
            ) : null}
          </Block6>

          <Block6 title="4 · What its distinctive piece is worth">
            <div className={styles.highlight}>
              <div className={styles.highlightTitle}>{s.distinctiveTitle}</div>
              <p>{s.distinctive}</p>
            </div>
          </Block6>

          <Block6 title="5 · Caveats">
            <ul className={styles.bullets}>
              {s.caveats.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ul>
          </Block6>

          {s.rejected ? (
            <Block6 title="Rejected alternatives">
              <div className={styles.tableCaption}>{s.rejected.caption}</div>
              <div className={styles.tableWrap}>
                <table className={styles.tbl}>
                  <thead>
                    <tr>
                      <th className={styles.thLeft}>Variant</th>
                      <th className={styles.thLeft}>After tax</th>
                      <th className={styles.thLeft}>Why it is not on the headline</th>
                    </tr>
                  </thead>
                  <tbody>
                    {s.rejected.rows.map((r, i) => (
                      <tr key={i}>
                        <td className={styles.thLeft}>
                          <strong>{r.name}</strong>
                        </td>
                        <td className={styles.thLeft}>{r.numbers}</td>
                        <td className={styles.thLeft}>{r.why}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Block6>
          ) : null}

          <Block6 title="6 · Links">
            <ul className={styles.links}>
              {s.links.map((l, i) => (
                <li key={i}>
                  {l.href ? (
                    l.href.startsWith('/app/') ? (
                      <Link to={l.href.replace('/app', '')}>{l.label}</Link>
                    ) : (
                      <Link to={l.href}>{l.label}</Link>
                    )
                  ) : (
                    <span className={styles.gap}>{l.label}</span>
                  )}
                </li>
              ))}
            </ul>
          </Block6>
        </div>
      ) : null}
    </div>
  );
}

/* ---------------------------------------------------------------- page */

export default function MpfReport() {
  const [d, setD] = useState<Report | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    fetch('/app/mpf_report.json?t=' + Date.now())
      .then((r) => r.json())
      .then(setD)
      .catch(() => setErr(true));
  }, []);

  if (err) {
    return (
      <div className={styles.page}>
        <h1 className={styles.h1}>The Momentum Portfolio</h1>
        <p className={styles.sectionIntro}>
          The numbers have not been generated yet. Run{' '}
          <code>venv/bin/python3 research/_utilities/mpf_report_build.py</code> on the VPS.
        </p>
      </div>
    );
  }
  if (!d) return <div className={styles.page}>Loading…</div>;

  const H = d.headline;
  const W = d.window2018;
  const r = (k: string) => H.rows[k];
  const w = (k: string) => W.rows[k];
  const TN = d.names.TN;
  const BA = d.names.BA;
  const IPO = d.names.IPO;
  const QS = d.names.QS;
  const BM = d.names.BM;
  const BLEND = d.names.BLEND;

  const bestCagr = H.contenders.reduce((a, b) => (r(a).cagr >= r(b).cagr ? a : b));
  const leastDd = H.contenders.reduce((a, b) => (r(a).maxdd >= r(b).maxdd ? a : b));
  const bestCalmar = H.contenders.reduce((a, b) => (r(a).calmar >= r(b).calmar ? a : b));
  const ipoCorr = H.corr[IPO] ?? {};
  const ipoMaxCorr = Math.max(
    ...Object.entries(ipoCorr)
      .filter(([k]) => k !== IPO)
      .map(([, v]) => v),
  );

  return (
    <div className={styles.page}>
      {/* ---------------------------------------------------------- header */}
      <header className={styles.header}>
        <h1 className={styles.h1}>The Momentum Portfolio</h1>
        <p className={styles.lede}>{d.standard}</p>
        <div className={styles.headerMeta}>
          <span>Report compiled {REPORT_DATE}</span>
          <span className={styles.dot}>·</span>
          <span>Numbers regenerated {d.generated}</span>
          <span className={styles.dot}>·</span>
          <span className={styles.mono}>{d.generator}</span>
        </div>
        <div className={styles.headerMeta}>
          <Chip>Post-tax only</Chip>
          <Chip>Placeable entries only</Chip>
          <Chip>
            Headline window {H.window[0]} → {H.window[1]} ({H.years} years)
          </Chip>
        </div>
      </header>

      {/* ---------------------------------------------------------- the correction leads */}
      <div className={styles.correction}>
        <div className={styles.correctionTag}>READ THIS BEFORE ANY RETURN FIGURE</div>
        <p className={styles.correctionLede}>{d.correction.headline}</p>
        <div className={styles.tableWrap}>
          <table className={`${styles.tbl} ${styles.tblTight}`}>
            <thead>
              <tr>
                <th className={styles.thLeft}>Book</th>
                <th>Published</th>
                <th>Honest, same spec and window</th>
              </tr>
            </thead>
            <tbody>
              {d.correction.costs.map((c) => (
                <tr key={c.book}>
                  <td className={styles.thLeft}>{c.book}</td>
                  <td className={styles.num}>{c.published}</td>
                  <td className={styles.numHot}>{c.honest}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <ul className={styles.bullets}>
          {d.correction.proof.map((p, i) => (
            <li key={i}>{p}</li>
          ))}
        </ul>
        <div className={styles.correctionFoot}>
          The audit:{' '}
          <span className={styles.mono}>
            research/158_oa_arming_width/OA_ARMING_WIDTH_AND_POKE_FILL_DAILY_SWEEP_STATUS.md
          </span>{' '}
          and the reusable template{' '}
          <span className={styles.mono}>
            research/158_oa_arming_width/scripts/verify_published_trades.py
          </span>
          . Full record:{' '}
          <Link to="/backtest/mpf-honest-entries-roster-2026-09">the 11-Sep-2026 audit page</Link>.
        </div>
      </div>

      {/* ---------------------------------------------------------- Q&A */}
      <Section
        num="00"
        label="Q & A"
        title="The five questions this page exists to answer"
        intro="Each answered in one or two lines, with the window stated. Every figure below is generated, not typed."
      >
        <div className={styles.qa}>
          <div className={styles.qaItem}>
            <div className={styles.q}>Q1. Which book earns the most?</div>
            <div className={styles.a}>
              <strong>{bestCagr}</strong>, at {r(bestCagr).cagr.toFixed(1)}% a year after tax over
              the full {H.years} years ({H.window[0]} to {H.window[1]}). {TN} is close behind at{' '}
              {r(TN).cagr.toFixed(1)}%. On the shorter 2018 window the ranking widens — {BA} reads{' '}
              {w(BA).cagr.toFixed(1)}% against {TN}’s {w(TN).cagr.toFixed(1)}% — which is a reason
              to read the long window, not the short one.
            </div>
          </div>
          <div className={styles.qaItem}>
            <div className={styles.q}>Q2. Which one loses the least?</div>
            <div className={styles.a}>
              <strong>{leastDd}</strong>, at {r(leastDd).maxdd.toFixed(1)}% worst fall against{' '}
              {r(BA).maxdd.toFixed(1)}% for {BA} and {r(BM).maxdd.toFixed(1)}% for the index, over
              the full {H.years} years. On return-per-unit-of-pain the leader is{' '}
              <strong>{bestCalmar}</strong> at Calmar {r(bestCalmar).calmar.toFixed(2)}.
            </div>
          </div>
          <div className={styles.qaItem}>
            <div className={styles.q}>Q3. Do they diversify each other?</div>
            <div className={styles.a}>
              Partly. {TN} and {BA} run at {H.corr[TN]?.[BA]?.toFixed(2)} weekly correlation — related, but far
              from the same book, which is why a 50-50 of them earns {r(BLEND).cagr.toFixed(1)}% at
              only {r(BLEND).maxdd.toFixed(1)}%. {IPO} is the genuine diversifier: its highest
              correlation to anything here is {ipoMaxCorr.toFixed(2)}. {QS}, by contrast, runs at{' '}
              {W.corr[QS]?.[BA]?.toFixed(2)} to {BA} on the 2018 window — it is a weaker sampling of
              a family the book already trades.
            </div>
          </div>
          <div className={styles.qaItem}>
            <div className={styles.q}>Q4. Does any of them reach 25%?</div>
            <div className={styles.a}>
              <strong>No.</strong> Over the full {H.years} years the best single book is{' '}
              {r(bestCagr).cagr.toFixed(1)}% and the 50-50 blend is {r(BLEND).cagr.toFixed(1)}%, both
              after tax. Only on the shorter 2018 window — which throws away 2008 and 2020 — does
              anything reach it, and that is a window effect rather than a result. The honest answer
              is that the 25% target needs a blend and allocation study that has not been started.
            </div>
          </div>
          <div className={styles.qaItem}>
            <div className={styles.q}>Q5. What is still owed?</div>
            <div className={styles.a}>
              Nine items, listed in full in the last section. The three that matter most: IPO Base
              has NOT been re-optimised on the honest entry; the inverted entry condition in{' '}
              <span className={styles.mono}>services/oa_entry.py</span> is not fixed and its buying
              stays paused; and the blend work — the only structure that plausibly clears 25% — has
              not been started.
            </div>
          </div>
        </div>
      </Section>

      {/* ---------------------------------------------------------- 1. headline */}
      <Section
        num="01"
        label="The book"
        title={`Every book being chosen between — ${H.years} years, after tax`}
        intro={H.windowWhy}
      >
        <div className={styles.cards}>
          {/* Arun (11-Sep-2026, late): every system gets a card at the top, Quality Summit
              included — its card carries its own window because it cannot exist on the
              20-year one. The 20-year TABLE below still excludes it, deliberately. */}
          {SYSTEMS.filter((s) => H.rows[s.key] || W.rows[s.key]).map((s) => (
            <SystemCard key={s.key} s={s} r={H.rows[s.key]} window2018={W.rows[s.key]} />
          ))}
        </div>

        <SummaryTable
          block={H}
          blockLabel={`WHICH SYSTEMS: True North, Open Alpha · Base Age, IPO Base, the 50-50 blend, and NIFTYBEES. WHICH WINDOW: ${H.window[0]} to ${H.window[1]} (${H.years} years). WHICH BASIS: after tax, placeable entries only.`}
          smallRows={[IPO]}
          blendKey={BLEND}
          blendNote={d.blendNote}
          investedNote={d.notes.cash_yield}
        />

        <div className={styles.noteBox}>
          <strong>Average invested for {BA} is a gap, not a zero.</strong>{' '}
          {d.notes.invested_gap}
        </div>
      </Section>

      {/* ---------------------------------------------------------- 2. portfolio view */}
      <Section
        num="02"
        label="The portfolio view"
        title="What the book looks like as one thing"
        intro="The charts and the year table are the part no individual study page can show: which book carried which year, how long each can disappoint, and whether holding two of them is better than holding the best one."
      >
        <Figure
          src={d.charts.curves20y}
          caption={`Growth of 100 on a log scale with the drawdown panel beneath, ${H.window[0]} to ${H.window[1]}. The lower panel is the reason the chart exists: two books ending at a similar multiple with −24% and −32% worst falls are not the same product.`}
        />

        <div className={styles.highlight}>
          <div className={styles.highlightTitle}>What this says</div>
          <ul className={styles.bullets}>
            {HIGHLIGHTS.map((h, i) => (
              <li key={i}>{h}</li>
            ))}
          </ul>
        </div>

        <YoyTable
          block={H}
          label={`YEAR BY YEAR — WHICH SYSTEMS: True North, Open Alpha · Base Age, IPO Base, the 50-50 blend, NIFTYBEES. WHICH WINDOW: ${H.window[0]} to ${H.window[1]}. WHICH BASIS: after tax.`}
        />

        <Figure
          src={d.charts.yearlyBars}
          caption="The same year table as a picture. Look for the years where only one bar is green — those are the years that justify holding more than one book."
        />

        <Figure
          src={d.charts.rolling3y}
          caption="Trailing three-year return. Every book spends multi-year stretches below the 25% bar and below zero; a three-year run of nothing is normal behaviour for these systems, not evidence that one has broken."
        />

        <CorrTable
          block={H}
          label={`WEEKLY-RETURN CORRELATION — WHICH SYSTEMS: True North, Open Alpha · Base Age, IPO Base, the blend, NIFTYBEES. WHICH WINDOW: ${H.window[0]} to ${H.window[1]}. WHICH BASIS: after tax.`}
        />

        <Figure
          src={d.charts.corr20y}
          caption="The same correlations as a heatmap. IPO Base is the pale column — the only book here that is genuinely doing something else."
        />

        <Figure
          src={d.charts.invested}
          caption="How much of each book is actually in the market. True North holds cash 57% of the time and still finishes near the top; Open Alpha · Base Age is a measurement gap, not a zero."
        />
        <div className={styles.noteBox}>{d.notes.invested_timeseries}</div>
      </Section>

      {/* ---------------------------------------------------------- 3. the 2018 window */}
      <Section
        num="03"
        label="Second window"
        title="The 2018-2026 window — where Quality Summit can be compared"
        intro={W.windowWhy}
      >
        <div className={styles.cards}>
          {SYSTEMS.filter((s) => !H.rows[s.key] && W.rows[s.key]).map((s) => (
            <SystemCard key={s.key} s={s} window2018={W.rows[s.key]} />
          ))}
        </div>

        <SummaryTable
          block={W}
          blockLabel={`WHICH SYSTEMS: all five plus NIFTYBEES, re-measured. WHICH WINDOW: ${W.window[0]} to ${W.window[1]}. WHICH BASIS: after tax; Quality Summit is the median-CAGR rebalance offset of ${W.qsOffsets?.n ?? 12}.`}
          smallRows={[IPO]}
          blendKey={BLEND}
          blendNote={d.blendNote}
          investedNote={d.notes.cash_yield}
        />

        {W.qsOffsets ? (
          <div className={styles.noteBox}>
            <strong>Quality Summit path drawn:</strong> {W.qsOffsets.drawn} — the median-CAGR offset
            of {W.qsOffsets.n} (median {W.qsOffsets.median}%, range {W.qsOffsets.min}% to{' '}
            {W.qsOffsets.max}%). Never an average of the twelve: averaging equity curves
            manufactures a smoother line than any single book could have run.
          </div>
        ) : null}

        <Figure
          src={d.charts.curves2018}
          caption="The same chart on the only window Quality Summit can exist in. It throws away 2008 and 2020, so it flatters everything — read it alongside the twenty-year chart, never instead of it."
        />

        <YoyTable
          block={W}
          label={`YEAR BY YEAR ON THE 2018 WINDOW — WHICH SYSTEMS: all five plus NIFTYBEES. WHICH WINDOW: ${W.window[0]} to ${W.window[1]}. WHICH BASIS: after tax. These figures are NOT comparable with the year table above.`}
        />

        <CorrTable
          block={W}
          label={`WEEKLY-RETURN CORRELATION ON THE 2018 WINDOW — WHICH SYSTEMS: all five plus NIFTYBEES. WHICH BASIS: after tax.`}
        />

        <Figure
          src={d.charts.corr2018}
          caption="Quality Summit sits closest to Open Alpha · Base Age and to the index. It is not a third source of return."
        />

        <div className={styles.noteBox}>{d.notes.two_curve_files}</div>
      </Section>

      {/* ---------------------------------------------------------- 4. per-system */}
      <Section
        num="04"
        label="Each book"
        title="The four books, six blocks each"
        intro="Identical structure for every system so nothing is ambiguous: the rules, the mechanics, the evidence, what its distinctive piece is worth, the caveats, and the links back to the full study. Collapsed by default — open the one you want."
      >
        {SYSTEMS.map((s, i) => (
          <SystemSection key={s.key} s={s} data={d} defaultOpen={i === 0} />
        ))}
      </Section>

      {/* ---------------------------------------------------------- 5. the correction's evidence */}
      <Section
        num="05"
        label="Evidence"
        title="The correction, measured — after tax"
        intro={`These are the tables behind the numbers at the top of the page. All three are AFTER TAX on 2006-2026. ${d.notes.aftertax_incomplete}`}
      >
        <TableBox
          title="WHICH SYSTEM: Open Alpha entry mechanics. WHICH WINDOW: 2006-2026. WHICH BASIS: after tax, median of seeds, CAGR %."
          caption="Every placeable entry improves as the trail lengthens. The unplaceable one degrades over the same range. The surfaces do not merely shift — they invert, so a parameter fitted on the second is wrong for the first."
        >
          <table className={styles.tbl}>
            <thead>
              <tr>
                <th className={styles.thLeft}>Entry mechanic</th>
                {d.correction.entrySurface.trails.map((t) => (
                  <th key={t}>trail-{t}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {d.correction.entrySurface.rows.map((row) => (
                <tr key={row.label} className={row.placeable ? undefined : styles.rowHot}>
                  <td className={styles.thLeft}>
                    {row.label}
                    {row.placeable ? (
                      <span className={styles.tagOk}>placeable</span>
                    ) : (
                      <span className={styles.tagBad}>not placeable</span>
                    )}
                  </td>
                  {row.cagr.map((v, i) => (
                    <td key={i} className={styles.num}>
                      {v === null ? '—' : `${v.toFixed(2)}%`}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </TableBox>

        <TableBox
          title="WHICH SYSTEM: Open Alpha, buy at the breakout close. WHICH WINDOW: 2006-2026. WHICH BASIS: after tax, CAGR %."
          caption="The null control: same days, same number of entries, same trail, stop, slots and costs. Only the names differ — chosen by the rule, or drawn at random from the eligible universe."
        >
          <table className={styles.tbl}>
            <thead>
              <tr>
                <th className={styles.thLeft}>Arm</th>
                {d.correction.nullControl.trails.map((t) => (
                  <th key={t}>trail-{t}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {d.correction.nullControl.rows.map((row) => (
                <tr key={row.label}>
                  <td className={styles.thLeft}>{row.label}</td>
                  {row.cagr.map((v, i) => (
                    <td key={i} className={styles.num}>
                      {v === null ? '—' : `${v.toFixed(2)}%`}
                    </td>
                  ))}
                </tr>
              ))}
              <tr className={styles.rowTotal}>
                <td className={styles.thLeft}>Edge over random</td>
                {d.correction.nullControl.edge.map((v, i) => (
                  <td key={i} className={styles.num}>
                    {v === null ? '—' : sgn(v, 2)}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </TableBox>

        <GateTable
          rows={d.correction.gateBakeoff}
          title="WHICH SYSTEM: Open Alpha, buy at the breakout close, trail-75. WHICH WINDOW: 2006-2026, the full period. WHICH BASIS: after tax. PRICE GATES ONLY."
          caption="Price gates barely move the book even on an honest entry: the best of them adds under a point of CAGR while blocking a quarter to a third of the days, and several take return away. This is the opposite of True North, where the gate is the whole product — a breakout book and a momentum book do not want the same gate."
        />

        <GateTable
          rows={d.correction.vixGates}
          title="WHICH SYSTEM: the same book and trail. WHICH WINDOW: 2016-2026 ONLY. WHICH BASIS: after tax. VIX GATES ONLY."
          caption="A SEPARATE TABLE because it is a SEPARATE WINDOW: INDIA VIX begins in 2015, so no VIX construction can be measured before 2016 and none of these figures may be read across to the price-gate table above. Every fixed VIX level loses and every relative construction wins, which makes sense — VIX drifts over a decade while a percentile adapts. The gate the ATH + VIX variant adopted is the 1-year 70th percentile."
        />

        <div className={styles.noteBox}>
          <strong>{d.correction.athVix.label}</strong> — the one after-tax summary figure that
          exists for the VIX-gated variant: {d.correction.athVix.cagr.toFixed(1)}% /{' '}
          {d.correction.athVix.maxdd.toFixed(1)}% / Calmar {d.correction.athVix.calmar.toFixed(2)},
          on {d.correction.athVix.window} ONLY, because INDIA VIX does not exist before 2015 in our
          data. It is evidence for the correction, not a candidate. Source:{' '}
          <span className={styles.mono}>{d.correction.athVix.source}</span>.
        </div>
      </Section>

      {/* ---------------------------------------------------------- 6. decisions & reviews */}
      <Section
        num="06"
        label="Decisions"
        title="Dated reviews, and what is still owed"
        intro="Every review below is registered in the Ops & Review Centre with its due date and its pass criterion. The owed list is the entry-audit session's, in its own priority order."
      >
        <TableBox
          title="DATED REVIEWS — the ones that touch these four books"
          caption="Registered in research/111_sensex_manual_mgmt/scripts/ops_center.py and rendered at /app/straddles#ops-center with automatic DUE SOON / OVERDUE badges."
        >
          <table className={styles.tbl}>
            <thead>
              <tr>
                <th className={styles.thLeft}>Review</th>
                <th>Due</th>
                <th>Status</th>
                <th className={styles.thLeft}>Pass criterion / what it must answer</th>
              </tr>
            </thead>
            <tbody>
              {REVIEWS.map((rv) => (
                <tr key={rv.title}>
                  <td className={styles.thLeft}>
                    <strong>{rv.title}</strong>
                  </td>
                  <td className={styles.num}>{rv.due}</td>
                  <td className={styles.num}>{rv.status}</td>
                  <td className={styles.thLeft}>{rv.what}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableBox>

        <div className={styles.owed}>
          {OWED.map((o) => (
            <div className={styles.owedItem} key={o.title}>
              <div className={styles.owedHead}>
                <span className={styles.owedTitle}>{o.title}</span>
                <Chip className={styles.chipWarn}>{o.state}</Chip>
              </div>
              <p>{o.what}</p>
            </div>
          ))}
        </div>

        <TableBox title="WHERE EVERY NUMBER ON THIS PAGE COMES FROM" caption="Regenerate all of it with the command below; nothing here is hand-typed.">
          <table className={styles.tbl}>
            <thead>
              <tr>
                <th className={styles.thLeft}>Figure group</th>
                <th className={styles.thLeft}>Source file</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(d.sources).map(([k, v]) => (
                <tr key={k}>
                  <td className={styles.thLeft}>{k}</td>
                  <td className={`${styles.thLeft} ${styles.mono}`}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableBox>

        <div className={styles.noteBox}>
          <strong>Regenerate:</strong>{' '}
          <span className={styles.mono}>
            cd /home/arun/quantifyd &amp;&amp; venv/bin/python3 research/_utilities/mpf_report_build.py
          </span>
          , then rebuild the frontend. Registered as an on-demand job in the Ops &amp; Review
          Centre and mirrored in <span className={styles.mono}>docs/LABS_AND_JOBS_REFERENCE.md</span>.
          Build log:{' '}
          <span className={styles.mono}>
            research/160_quality_growth_near_ath/MPF_REPORT_PAGE_BUILD_STATUS.md
          </span>
          .
        </div>
      </Section>
    </div>
  );
}
