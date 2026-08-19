import styles from './ScaleUp.module.css';
import { CONFIG_PLAN, TARGET, WEEKS, LADDER, PAGE_UPDATED } from '../data/scaleup';

const inr = (v: number) => (v >= 0 ? '+₹' : '−₹') + Math.abs(Math.round(v)).toLocaleString('en-IN');

export default function ScaleUp() {
  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>Scale-Up · Weekly Targets &amp; Progress</h1>
          <p className={styles.sub}>
            NIFTY + SENSEX combined, live books only. Row 1 is the locked 1.0× target; a new
            actuals row lands every Friday evening. Targets move only via the weekly
            reassessment — never off a single week.
          </p>
        </div>
        <span className={styles.updated}>updated {PAGE_UPDATED}</span>
      </div>

      {/* links to the validating machinery */}
      <div className={styles.linkRow}>
        <a className={styles.linkChip} href="/app/straddles#portfolio-lab">
          🧪 Weekly validation — stack reassessment (Fri 16:35) &amp; Portfolio Lab
        </a>
        <a className={styles.linkChip} href="/app/straddles#ops-center">
          🛠 Ops &amp; Reviews — the Friday append ritual is registered here
        </a>
        <a className={styles.linkChip} href="/app/nas-config">
          ⚙ Rules Matrix — what each system runs
        </a>
      </div>

      {/* the locked config */}
      <div className={styles.groupLabel}>The locked 1.0× week (Option B · venue-split, reduced NIFTY Thursday)</div>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr><th>Day</th><th>Books live</th><th>Lots</th><th>Concurrent margin</th></tr>
          </thead>
          <tbody>
            {CONFIG_PLAN.map((d) => (
              <tr key={d.day} className={d.day === 'Thu' ? styles.peakRow : ''}>
                <td className={styles.dayCell}>{d.day}</td>
                <td>{d.books}</td>
                <td className={styles.mono}>{d.lots}</td>
                <td className={styles.mono}>₹{d.margin.toFixed(1)}L{d.day === 'Thu' ? ' · peak' : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* weekly progress */}
      <div className={styles.groupLabel}>Weekly progress vs target</div>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Week</th><th>P&amp;L (combined)</th><th>vs target</th>
              <th>Worst DD</th><th>vs floor</th><th>Verdict</th><th className={styles.noteCol}>Notes</th>
            </tr>
          </thead>
          <tbody>
            <tr className={styles.targetRow}>
              <td className={styles.dayCell}>TARGET · {TARGET.label}</td>
              <td className={styles.mono}>{inr(TARGET.pnlWeek)} / wk</td>
              <td>—</td>
              <td className={styles.mono}>{inr(TARGET.ddFloor)} floor</td>
              <td>—</td>
              <td><span className={styles.chipTarget}>row 1 · the bar</span></td>
              <td className={styles.noteCol}>
                Peak margin ₹{TARGET.peakMarginL}L (Thu) on capital ₹{TARGET.capitalL}L
                (cash-equiv ₹{TARGET.cashEquivL}L). <span className={styles.basis}>{TARGET.basis}</span>
              </td>
            </tr>
            {WEEKS.map((w) => {
              const chip =
                w.status === 'transition' ? <span className={styles.chipMuted}>transition · unscored</span>
                : w.status === 'progress' ? <span className={styles.chipProgress}>in progress</span>
                : w.status === 'ontrack' ? <span className={styles.chipOn}>on track</span>
                : <span className={styles.chipMiss}>behind</span>;
              const vsT = w.pnl != null ? inr(w.pnl - TARGET.pnlWeek) : '—';
              const vsD = w.dd != null ? (w.dd >= TARGET.ddFloor ? 'inside floor' : 'FLOOR BREACHED') : '—';
              return (
                <tr key={w.week}>
                  <td className={styles.dayCell}>{w.week}</td>
                  <td className={styles.mono}>{w.pnl != null ? inr(w.pnl) : '—'}</td>
                  <td className={styles.mono}>{vsT}</td>
                  <td className={styles.mono}>{w.dd != null ? inr(w.dd) : '—'}</td>
                  <td>{vsD}</td>
                  <td>{chip}</td>
                  <td className={styles.noteCol}>{w.note}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* future ladder */}
      <div className={styles.groupLabel}>
        Future scale-up ladder <span className={styles.caveat}>— entire table subject to the weekly reassessment and adjustments</span>
      </div>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Step</th><th>NIFTY (suite-COMB-TB)</th><th>SENSEX (suite/TB)</th>
              <th>Peak margin</th><th>Capital needed</th><th>Add vs today</th>
              <th>of which liquid</th><th>Est. P&amp;L/wk</th><th>DD guide</th>
            </tr>
          </thead>
          <tbody>
            {LADDER.map((l) => (
              <tr key={l.step} className={l.step.startsWith('1.0') ? styles.currentRow : ''}>
                <td className={styles.dayCell}>{l.step}</td>
                <td className={styles.mono}>{l.nifty}</td>
                <td className={styles.mono}>{l.sensex}</td>
                <td className={styles.mono}>₹{l.peakL.toFixed(1)}L</td>
                <td className={styles.mono}>₹{l.capitalL.toFixed(1)}L</td>
                <td className={styles.mono}>{l.addL > 0 ? `+₹${l.addL.toFixed(1)}L` : '—'}</td>
                <td className={styles.mono}>{l.liquidAddL > 0 ? `+₹${l.liquidAddL.toFixed(1)}L` : '—'}</td>
                <td className={styles.mono}>{inr(l.pnlWeek)}</td>
                <td className={styles.mono}>{inr(l.dd)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className={styles.foot}>
        Margins measured 19-Aug (NIFTY ≈ ₹1.65L/lot · SENSEX ≈ ₹2.04L/lot straddle, MIS). 50%
        cash rule: liquid-fund collateral counts as cash-equivalent. A step is earned by weeks of
        actuals on this page, not by the lab means alone — SENSEX live history is still thin.
      </p>
    </div>
  );
}
