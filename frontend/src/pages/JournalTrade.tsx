/**
 * Journal Trade — single-trade detail.
 *
 * Route: /journal/trade/:id
 *
 * Layout: header (instrument, strategy, mode), metric strip (entry/exit/P&L/R/hold/grade),
 * placeholder for chart, tags cloud, notes editor, screenshots placeholder.
 */

import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import styles from './Journal.module.css';
import { apiGet, apiPost } from '../api/client';
import type { JournalTradeDetail, JournalTag } from '../api/types';
import { formatPnl } from '../utils/format';

export default function JournalTrade() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [trade, setTrade] = useState<JournalTradeDetail | null>(null);
  const [tags, setTags] = useState<JournalTag[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [notes, setNotes] = useState('');
  const [savedAt, setSavedAt] = useState<string | null>(null);

  function load() {
    if (!id) return;
    setLoading(true);
    Promise.all([
      apiGet<JournalTradeDetail>(`/api/journal/trades/${id}`),
      apiGet<{ tags: JournalTag[] }>(`/api/journal/tags`),
    ])
      .then(([t, tg]) => {
        setTrade(t);
        setNotes(t.notes?.body_md || '');
        setTags(tg.tags);
        setLoading(false);
      })
      .catch((e: Error) => {
        setErr(e.message);
        setLoading(false);
      });
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  function saveNotes() {
    if (!trade) return;
    apiPost(`/api/journal/trades/${trade.id}/notes`, { body_md: notes })
      .then(() => {
        setSavedAt(new Date().toLocaleTimeString());
        setTimeout(() => setSavedAt(null), 2200);
      })
      .catch((e: Error) => setErr(e.message));
  }

  function setGrade(n: number) {
    if (!trade) return;
    apiPost(`/api/journal/trades/${trade.id}`, { grade: n }).catch(() => undefined);
    // PATCH semantically; our blueprint accepts PATCH; but apiPost helpers
    // don't have a PATCH variant — call fetch directly.
    fetch(`/api/journal/trades/${trade.id}`, {
      method: 'PATCH',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ grade: n }),
    }).then(() => load()).catch(() => undefined);
  }

  function toggleTag(tag: JournalTag) {
    if (!trade) return;
    const has = trade.tags.find((t) => t.id === tag.id);
    if (has) {
      fetch(`/api/journal/trades/${trade.id}/tags/${tag.id}`, {
        method: 'DELETE',
        credentials: 'include',
      }).then(load);
    } else {
      apiPost(`/api/journal/trades/${trade.id}/tags`, { tag_ids: [tag.id] }).then(load);
    }
  }

  function toggleMistakeFlag() {
    if (!trade) return;
    const newVal = trade.mistake_flag ? 0 : 1;
    fetch(`/api/journal/trades/${trade.id}`, {
      method: 'PATCH',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mistake_flag: newVal }),
    }).then(load);
  }

  if (loading || !trade) {
    return (
      <div className={styles.root}>
        <div className={styles.loading}>Loading trade…</div>
      </div>
    );
  }

  const tagsByCat: Record<string, JournalTag[]> = {};
  tags.forEach((t) => {
    if (!tagsByCat[t.category]) tagsByCat[t.category] = [];
    tagsByCat[t.category].push(t);
  });
  const tradeTagIds = new Set(trade.tags.map((t) => t.id));

  const entryDate = new Date((trade.entry_time || '').replace(' ', 'T'));
  const dayStr = isNaN(entryDate.getTime())
    ? trade.entry_time
    : entryDate.toLocaleDateString('en-US', { day: 'numeric', month: 'long', year: 'numeric' });

  return (
    <div className={styles.root}>
      <div className={styles.headerRow}>
        <div>
          <div className={styles.eyebrow}>Trade detail</div>
          <h1 className={styles.title}>{trade.instrument}</h1>
          <div className={styles.sub}>
            {trade.strategy} · {trade.direction} · {dayStr}
          </div>
        </div>
        <div className={styles.actions}>
          <span
            className={styles.chipFilter}
            style={{ cursor: 'pointer' }}
            onClick={toggleMistakeFlag}
          >
            {trade.mistake_flag ? 'Unmark mistake' : 'Mark as mistake'}
          </span>
          <button className={styles.btnSecondary} onClick={() => navigate(-1)}>
            &#8592; Back
          </button>
        </div>
      </div>

      {err && <div className={styles.error}>{err}</div>}

      {/* Metric strip */}
      <div className={styles.metrics}>
        <div className={styles.metric}>
          <div className={styles.metricLabel}>Entry</div>
          <div className={styles.metricValue}>{trade.entry_price?.toFixed(2)}</div>
          <div className={styles.metricSub}>
            {(trade.entry_time || '').slice(0, 16).replace('T', ' ')}
          </div>
        </div>
        <div className={styles.metric}>
          <div className={styles.metricLabel}>Exit</div>
          <div className={styles.metricValue}>
            {trade.exit_price != null ? trade.exit_price.toFixed(2) : '—'}
          </div>
          <div className={styles.metricSub}>{trade.exit_reason || 'Open'}</div>
        </div>
        <div className={styles.metric}>
          <div className={styles.metricLabel}>P&amp;L net</div>
          <div className={`${styles.metricValue} ${(trade.pnl_net ?? 0) >= 0 ? styles.pos : styles.neg}`}>
            {formatPnl(trade.pnl_net ?? 0)}
          </div>
          <div className={styles.metricSub}>
            Gross {formatPnl(trade.pnl_gross ?? 0)} · costs {formatPnl(-(trade.pnl_charges ?? 0))}
          </div>
        </div>
        <div className={styles.metric}>
          <div className={styles.metricLabel}>R-multiple</div>
          <div className={styles.metricValue}>
            {trade.r_multiple != null ? `${trade.r_multiple > 0 ? '+' : ''}${trade.r_multiple.toFixed(2)}R` : '—'}
          </div>
          <div className={styles.metricSub}>
            Risk {formatPnl(trade.initial_risk ?? 0)}
          </div>
        </div>
        <div className={styles.metric}>
          <div className={styles.metricLabel}>Hold time</div>
          <div className={styles.metricValue}>
            {trade.hold_minutes != null ? formatHold(trade.hold_minutes) : '—'}
          </div>
          <div className={styles.metricSub}>
            <span className={`${styles.modeBadge} ${trade.mode === 'LIVE' ? styles.live : ''}`}>
              {trade.mode}
            </span>
          </div>
        </div>
      </div>

      <div className={styles.tradeBody}>
        {/* Left column: chart placeholder + notes */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div className={styles.section}>
            <div className={styles.sectionHead}>Entry / exit chart</div>
            <div className={styles.sectionBody}>
              <div className={styles.placeholder}>
                Chart playback (Phase 2). Render 5-min OHLC around the trade
                window with entry / SL / target / exit markers from
                <code> market_data_unified</code>.
              </div>
            </div>
          </div>

          <div className={styles.section}>
            <div className={styles.sectionHead}>
              Notes
              {savedAt && (
                <span style={{ float: 'right', fontSize: 12, color: 'var(--accent-pos)', fontWeight: 400 }}>
                  Saved {savedAt}
                </span>
              )}
            </div>
            <div className={styles.sectionBody}>
              <textarea
                className={styles.textarea}
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                onBlur={saveNotes}
                placeholder="What was the thesis? What surprised you? What would you do differently? (Markdown supported, saves on blur.)"
                style={{ minHeight: 180 }}
              />
              <div style={{ marginTop: 8 }}>
                <button className={styles.btnSecondary} onClick={saveNotes}>
                  Save notes
                </button>
              </div>
            </div>
          </div>

          <div className={styles.section}>
            <div className={styles.sectionHead}>Screenshots</div>
            <div className={styles.sectionBody}>
              <div className={styles.placeholder}>
                Drag-drop to attach (Phase 2). Up to 5 images per trade,
                stored under <code>backtest_data/journal_screenshots/</code>.
              </div>
            </div>
          </div>
        </div>

        {/* Right column: grade + tags + source link */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div className={styles.section}>
            <div className={styles.sectionHead}>Process grade</div>
            <div className={styles.sectionBody}>
              <div className={styles.gradePicker}>
                {[1, 2, 3, 4, 5].map((n) => (
                  <button
                    key={n}
                    className={`${styles.scoreBtn} ${trade.grade === n ? styles.active : ''}`}
                    onClick={() => setGrade(n)}
                  >
                    {n}
                  </button>
                ))}
              </div>
              <div style={{ marginTop: 10, fontSize: 12, color: 'var(--ink-muted)' }}>
                Score the <em>process</em>, not the outcome. Did you follow the
                rules? Was sizing right? Was entry on the planned signal?
              </div>
            </div>
          </div>

          <div className={styles.section}>
            <div className={styles.sectionHead}>Tags</div>
            <div className={styles.sectionBody}>
              {Object.keys(tagsByCat).sort().map((cat) => (
                <div key={cat} style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 11, color: 'var(--ink-muted)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 6 }}>
                    {cat}
                  </div>
                  <div className={styles.tagsCloud}>
                    {tagsByCat[cat].map((t) => {
                      const selected = tradeTagIds.has(t.id);
                      const bg = t.color_hex || undefined;
                      return (
                        <span
                          key={t.id}
                          className={`${styles.tagPill} ${selected ? styles.selected : ''}`}
                          onClick={() => toggleTag(t)}
                          style={selected && bg ? { background: bg, color: 'white' } : undefined}
                        >
                          {t.name}
                        </span>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className={styles.section}>
            <div className={styles.sectionHead}>Source</div>
            <div className={styles.sectionBody}>
              <div className={styles.kvList}>
                <div className={styles.kv}>
                  <div className="k">Source DB</div>
                  <div className="v">{trade.source_db}</div>
                </div>
                <div className={styles.kv}>
                  <div className="k">Source row</div>
                  <div className="v">{trade.source_id ?? 'manual'}</div>
                </div>
                <div className={styles.kv}>
                  <div className="k">Qty</div>
                  <div className="v">{trade.qty}</div>
                </div>
                <div className={styles.kv}>
                  <div className="k">Mode</div>
                  <div className="v">{trade.mode}</div>
                </div>
              </div>
              <div style={{ marginTop: 12 }}>
                <a
                  className={styles.chipFilter}
                  href={sourceLink(trade.source_db)}
                  style={{ display: 'inline-block' }}
                >
                  View in {sourceLinkLabel(trade.source_db)} dashboard &#8594;
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function formatHold(minutes: number): string {
  if (minutes < 60) return `${minutes}m`;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  if (h < 24) return `${h}h${m > 0 ? ` ${m}m` : ''}`;
  const d = Math.floor(h / 24);
  return `${d}d ${h % 24}h`;
}

function sourceLink(src: string): string {
  switch (src) {
    case 'orb_trading':
      return '/app/orb';
    case 'kc6_trading':
      return '/kc6';
    case 'nas_trading':
      return '/app/nas';
    case 'strangle_trading':
      return '/app/strangle';
    default:
      return '#';
  }
}

function sourceLinkLabel(src: string): string {
  if (src.startsWith('orb')) return 'ORB';
  if (src.startsWith('kc6')) return 'KC6';
  if (src.startsWith('nas')) return 'NAS';
  if (src.startsWith('strangle')) return 'Strangle';
  return 'source';
}
