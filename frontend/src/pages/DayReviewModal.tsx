/**
 * Day in review — popout dialog opened from the Journal calendar.
 *
 * Same content as the /journal/day/:date page, but as an overlay so you can
 * stay on the calendar. Traverse days with the ← → buttons inside the popout
 * or the keyboard arrow keys; Esc closes.
 */

import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import styles from './Journal.module.css';
import { apiGet, apiPost } from '../api/client';
import type { JournalDayResponse } from '../api/types';
import { formatPnl, formatInt } from '../utils/format';

function shiftDate(date: string, days: number): string {
  const d = new Date(date + 'T00:00:00');
  d.setDate(d.getDate() + days);
  return d.toISOString().slice(0, 10);
}

export default function DayReviewModal({
  date,
  onNavigate,
  onClose,
}: {
  date: string;
  onNavigate: (d: string) => void;
  onClose: () => void;
}) {
  const navigate = useNavigate();
  const [data, setData] = useState<JournalDayResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [postClose, setPostClose] = useState('');
  const [preMarket, setPreMarket] = useState('');
  const [discipline, setDiscipline] = useState<number | null>(null);
  const [violations, setViolations] = useState('0');
  const [savedTick, setSavedTick] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setErr(null);
    apiGet<JournalDayResponse>(`/api/journal/day/${date}`)
      .then((d) => {
        if (cancelled) return;
        setData(d);
        setPreMarket(d.review?.pre_market_md || '');
        setPostClose(d.review?.post_close_md || '');
        setDiscipline(d.review?.discipline_score ?? null);
        setViolations(String(d.review?.rule_violations || 0));
        setLoading(false);
      })
      .catch((e: Error) => {
        if (cancelled) return;
        setErr(e.message);
        setLoading(false);
      });
    return () => { cancelled = true; };
  }, [date]);

  const go = useCallback((delta: number) => onNavigate(shiftDate(date, delta)), [date, onNavigate]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'TEXTAREA' || tag === 'INPUT') return;   // don't hijack typing
      if (e.key === 'ArrowLeft') { e.preventDefault(); go(-1); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); go(1); }
      else if (e.key === 'Escape') { e.preventDefault(); onClose(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [go, onClose]);

  function saveReview() {
    apiPost(`/api/journal/day/${date}/review`, {
      pre_market_md: preMarket,
      post_close_md: postClose,
      discipline_score: discipline,
      rule_violations: parseInt(violations, 10) || 0,
    })
      .then(() => {
        setSavedTick(new Date().toLocaleTimeString());
        setTimeout(() => setSavedTick(null), 2200);
      })
      .catch((e: Error) => setErr(e.message));
  }

  const niceDate = new Date(date + 'T00:00:00').toLocaleDateString('en-US', {
    weekday: 'long', day: 'numeric', month: 'long', year: 'numeric',
  });
  const m = data?.metrics;
  const trades = data?.trades ?? [];

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modalDialog} onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHead}>
          <div className={styles.modalNav}>
            <button className={styles.iconBtn} onClick={() => go(-1)} title="Previous day (←)">‹</button>
            <button className={styles.iconBtn} onClick={() => go(1)} title="Next day (→)">›</button>
          </div>
          <div style={{ flex: 1 }}>
            <div className={styles.eyebrow}>Day in review</div>
            <h2 className={styles.modalTitle}>{niceDate}</h2>
            <div className={styles.sub}>
              {loading ? 'Loading…'
                : m ? `${m.trades_count} trades · ${m.wins} wins · ${m.losses} losses`
                : 'No data'}
            </div>
          </div>
          <div className={styles.modalHeadActions}>
            <button className={styles.btnSecondary} onClick={() => navigate(`/journal/day/${date}`)}>
              Open full page
            </button>
            <button className={styles.iconBtn} onClick={onClose} title="Close (Esc)">✕</button>
          </div>
        </div>

        <div className={styles.modalBody}>
          {err && <div className={styles.error}>{err}</div>}

          {m && (
            <div className={styles.dayMetrics}>
              <div className={styles.metric}>
                <div className={styles.metricLabel}>Net P&amp;L</div>
                <div className={`${styles.metricValue} ${(m.pnl_net ?? 0) >= 0 ? styles.pos : styles.neg}`}>
                  {formatPnl(m.pnl_net)}
                </div>
                <div className={styles.metricSub}>Gross {formatPnl(m.pnl_gross)}</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricLabel}>Trades</div>
                <div className={styles.metricValue}>{m.trades_count}</div>
                <div className={styles.metricSub}>{m.wins} W · {m.losses} L</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricLabel}>Hit rate</div>
                <div className={styles.metricValue}>
                  {m.trades_count ? ((m.wins / m.trades_count) * 100).toFixed(0) + '%' : '—'}
                </div>
                <div className={styles.metricSub}>
                  {(data?.review?.discipline_score ?? null) != null
                    ? `Discipline ${data!.review!.discipline_score}/5` : 'Discipline not scored'}
                </div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricLabel}>Rule violations</div>
                <div className={styles.metricValue}>{data?.review?.rule_violations ?? 0}</div>
                <div className={styles.metricSub}>From daily review</div>
              </div>
            </div>
          )}

          <section className={styles.panel} style={{ marginBottom: 18 }}>
            <div className={styles.panelHead}>
              <div className={styles.panelTitle}>Trades · timeline</div>
            </div>
            <div className={styles.timeline}>
              {loading ? (
                <div className={styles.empty}>Loading…</div>
              ) : trades.length === 0 ? (
                <div className={styles.empty}>No trades on this day.</div>
              ) : (
                trades.map((t) => {
                  const time = (t.entry_time || '').slice(11, 16) || '—';
                  const exitTime = (t.exit_time || '').slice(11, 16);
                  return (
                    <div className={styles.tradeTimelineRow} key={t.id}
                         onClick={() => navigate(`/journal/trade/${t.id}`)}>
                      <div style={{ fontVariantNumeric: 'tabular-nums', fontSize: 12, color: 'var(--ink-muted)' }}>
                        {time}{exitTime ? ` → ${exitTime}` : ''}
                      </div>
                      <div>
                        <div className={styles.sym} style={{ fontWeight: 500 }}>{t.instrument}</div>
                        <div><span className={styles.stratChip}>{t.strategy}</span></div>
                      </div>
                      <div>
                        <span className={`${styles.side} ${t.direction === 'LONG' ? styles.long : styles.short}`}>
                          {t.direction === 'LONG' ? 'Long' : 'Short'}
                        </span>
                      </div>
                      <div style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums', fontSize: 13 }}>
                        @ {t.entry_price?.toFixed(2)}
                      </div>
                      <div style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums', fontSize: 13 }}>
                        → {t.exit_price?.toFixed(2) ?? '—'}
                      </div>
                      <div className={`${styles.pnlCell} ${(t.pnl_net ?? 0) >= 0 ? styles.pos : styles.neg}`}>
                        {formatPnl(t.pnl_net ?? 0)}
                      </div>
                      <div className={styles.rMult}>
                        {t.r_multiple != null
                          ? `${t.r_multiple > 0 ? '+' : ''}${t.r_multiple.toFixed(2)}R`
                          : `${formatInt(t.qty)}q`}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </section>

          <section className={styles.panel}>
            <div className={styles.panelHead}>
              <div className={styles.panelTitle}>
                Daily review
                <span className={styles.panelTitleAccent}>· five questions, three minutes</span>
              </div>
              {savedTick && <span style={{ fontSize: 12, color: 'var(--accent-pos)' }}>Saved {savedTick}</span>}
            </div>
            <div className={styles.reviewForm}>
              <div>
                <div className={styles.reviewLabel}>1. Pre-market intent (what did I expect today?)</div>
                <textarea className={styles.textarea} value={preMarket}
                          onChange={(e) => setPreMarket(e.target.value)}
                          placeholder="NAS premium 8-12 acceptable, planned 1-2 trades..." />
              </div>
              <div>
                <div className={styles.reviewLabel}>2. Post-close review (did I follow the system? what surprised me?)</div>
                <textarea className={styles.textarea} value={postClose}
                          onChange={(e) => setPostClose(e.target.value)}
                          style={{ minHeight: 120 }}
                          placeholder="One mistake today..." />
              </div>
              <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'flex-end' }}>
                <div>
                  <div className={styles.reviewLabel}>3. Discipline score (1-5)</div>
                  <div className={styles.scoreRow}>
                    {[1, 2, 3, 4, 5].map((n) => (
                      <button key={n}
                              className={`${styles.scoreBtn} ${discipline === n ? styles.active : ''}`}
                              onClick={() => setDiscipline(n)}>{n}</button>
                    ))}
                  </div>
                </div>
                <div>
                  <div className={styles.reviewLabel}>4. Rule violations</div>
                  <input className={styles.input} type="number" min={0} value={violations}
                         onChange={(e) => setViolations(e.target.value)} style={{ width: 80 }} />
                </div>
                <button className={styles.btnPrimary} onClick={saveReview} style={{ marginLeft: 'auto' }}>
                  Save review
                </button>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
