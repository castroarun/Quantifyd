/**
 * Journal Day — single-day timeline + per-trade list + daily review form.
 *
 * Route: /journal/day/:date (date in YYYY-MM-DD)
 */

import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import styles from './Journal.module.css';
import { apiGet, apiPost } from '../api/client';
import type { JournalDayResponse } from '../api/types';
import { formatPnl, formatInt } from '../utils/format';

export default function JournalDay() {
  const { date } = useParams<{ date: string }>();
  const navigate = useNavigate();
  const [data, setData] = useState<JournalDayResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  // Review form state
  const [postClose, setPostClose] = useState('');
  const [preMarket, setPreMarket] = useState('');
  const [discipline, setDiscipline] = useState<number | null>(null);
  const [violations, setViolations] = useState<string>('0');
  const [savedTick, setSavedTick] = useState<string | null>(null);

  useEffect(() => {
    if (!date) return;
    let cancelled = false;
    setLoading(true);
    apiGet<JournalDayResponse>(`/api/journal/day/${date}`)
      .then((d) => {
        if (cancelled) return;
        setData(d);
        if (d.review) {
          setPreMarket(d.review.pre_market_md || '');
          setPostClose(d.review.post_close_md || '');
          setDiscipline(d.review.discipline_score);
          setViolations(String(d.review.rule_violations || 0));
        } else {
          setPreMarket('');
          setPostClose('');
          setDiscipline(null);
          setViolations('0');
        }
        setLoading(false);
      })
      .catch((e: Error) => {
        if (cancelled) return;
        setErr(e.message);
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [date]);

  function saveReview() {
    if (!date) return;
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

  if (loading || !data) {
    return (
      <div className={styles.root}>
        <div className={styles.loading}>Loading day…</div>
      </div>
    );
  }

  const trades = data.trades;
  const m = data.metrics;
  const niceDate = new Date(data.date + 'T00:00:00').toLocaleDateString('en-US', {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  });

  return (
    <div className={styles.root}>
      <div className={styles.headerRow}>
        <div>
          <div className={styles.eyebrow}>Day in review</div>
          <h1 className={styles.title}>{niceDate}</h1>
          <div className={styles.sub}>
            {m.trades_count} trades · {m.wins} wins · {m.losses} losses
          </div>
        </div>
        <div className={styles.actions}>
          <button className={styles.btnSecondary} onClick={() => navigate('/journal')}>
            &#8592; Back to calendar
          </button>
        </div>
      </div>

      {err && <div className={styles.error}>{err}</div>}

      {/* Day metrics */}
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
            {(data.review?.discipline_score ?? null) != null ? `Discipline ${data.review!.discipline_score}/5` : 'Discipline not scored'}
          </div>
        </div>
        <div className={styles.metric}>
          <div className={styles.metricLabel}>Rule violations</div>
          <div className={styles.metricValue}>{data.review?.rule_violations ?? 0}</div>
          <div className={styles.metricSub}>From daily review</div>
        </div>
      </div>

      {/* Trades timeline */}
      <section className={styles.panel} style={{ marginBottom: 22 }}>
        <div className={styles.panelHead}>
          <div className={styles.panelTitle}>Trades · timeline</div>
        </div>
        <div className={styles.timeline}>
          {trades.length === 0 ? (
            <div className={styles.empty}>No trades on this day.</div>
          ) : (
            trades.map((t) => {
              const time = (t.entry_time || '').slice(11, 16) || '—';
              const exitTime = (t.exit_time || '').slice(11, 16);
              return (
                <div
                  className={styles.tradeTimelineRow}
                  key={t.id}
                  onClick={() => navigate(`/journal/trade/${t.id}`)}
                >
                  <div style={{ fontVariantNumeric: 'tabular-nums', fontSize: 12, color: 'var(--ink-muted)' }}>
                    {time}{exitTime ? ` → ${exitTime}` : ''}
                  </div>
                  <div>
                    <div className={styles.sym} style={{ fontWeight: 500 }}>{t.instrument}</div>
                    <div>
                      <span className={styles.stratChip}>{t.strategy}</span>
                    </div>
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
                    {t.r_multiple != null ? `${t.r_multiple > 0 ? '+' : ''}${t.r_multiple.toFixed(2)}R` : `${formatInt(t.qty)}q`}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </section>

      {/* Daily review form */}
      <section className={styles.panel}>
        <div className={styles.panelHead}>
          <div className={styles.panelTitle}>
            Daily review
            <span className={styles.panelTitleAccent}>· five questions, three minutes</span>
          </div>
          {savedTick && (
            <span style={{ fontSize: 12, color: 'var(--accent-pos)' }}>Saved {savedTick}</span>
          )}
        </div>
        <div className={styles.reviewForm}>
          <div>
            <div className={styles.reviewLabel}>1. Pre-market intent (what did I expect today?)</div>
            <textarea
              className={styles.textarea}
              value={preMarket}
              onChange={(e) => setPreMarket(e.target.value)}
              placeholder="ORB long bias, NAS premium 8-12 acceptable, planned 1-2 trades..."
            />
          </div>
          <div>
            <div className={styles.reviewLabel}>2. Post-close review (did I follow the system? what surprised me? tomorrow's priority?)</div>
            <textarea
              className={styles.textarea}
              value={postClose}
              onChange={(e) => setPostClose(e.target.value)}
              placeholder="One mistake today: HDFC long where I moved SL up before the OR tested. Cost a clean stop-out..."
              style={{ minHeight: 140 }}
            />
          </div>
          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'flex-end' }}>
            <div>
              <div className={styles.reviewLabel}>3. Discipline score (1-5)</div>
              <div className={styles.scoreRow}>
                {[1, 2, 3, 4, 5].map((n) => (
                  <button
                    key={n}
                    className={`${styles.scoreBtn} ${discipline === n ? styles.active : ''}`}
                    onClick={() => setDiscipline(n)}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <div className={styles.reviewLabel}>4. Rule violations</div>
              <input
                className={styles.input}
                type="number"
                min={0}
                value={violations}
                onChange={(e) => setViolations(e.target.value)}
                style={{ width: 80 }}
              />
            </div>
            <button className={styles.btnPrimary} onClick={saveReview} style={{ marginLeft: 'auto' }}>
              Save review
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
