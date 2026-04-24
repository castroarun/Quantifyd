import { useEffect, useState } from 'react';
import { apiGet, apiPost } from '../api/client';
import styles from './Nwv.module.css';

type StackedLevel = {
  price: number;
  components: string[];
  weekly_price?: number;
  daily_price?: number;
};

type NwvView = {
  id?: number;
  week_start: string;
  generated_at: string;
  mon_open?: number;
  first_candle_open?: number;
  first_candle_high?: number;
  first_candle_low?: number;
  first_candle_close?: number;
  first_candle_body?: string;
  first_candle_pos?: string;
  first_candle_range_pct?: number;
  first_candle_wick_pos_pct?: number;
  gap_pct?: number;
  gap_tier?: string;
  gap_direction?: string;
  vix_value?: number;
  vix_pct_rank?: number;
  adx_daily?: number;
  adx_bucket?: string;
  monthly_override_side?: string | null;
  monthly_override_applied?: number;
  stacked_supports?: StackedLevel[];
  stacked_resistances?: StackedLevel[];
  base_view: string;
  final_view: string;
  conviction: number;
  instrument_choice?: string;
  expected_range_low?: number;
  expected_range_high?: number;
  time_stop?: string;
  notes?: string;
};

type NwvWeeklyState = {
  week_start: string;
  prev_week_high: number;
  prev_week_low: number;
  prev_week_close: number;
  prev_fri_close?: number;
  pivot_pp: number;
  pivot_s1: number;
  pivot_s2: number;
  pivot_r1: number;
  pivot_r2: number;
  cpr_tc: number;
  cpr_bc: number;
  cpr_width_pct: number;
  cpr_bucket: string;
  monthly_tc?: number;
  monthly_bc?: number;
  monthly_pivot?: number;
};

type ViewResp = {
  view: NwvView | null;
  weekly_state: NwvWeeklyState | null;
  message?: string;
};

function fmtRs(v?: number | null, dp = 2): string {
  if (v == null || !Number.isFinite(v)) return '—';
  return v.toLocaleString('en-IN', { maximumFractionDigits: dp, minimumFractionDigits: dp });
}

function fmtPct(v?: number | null, dp = 2): string {
  if (v == null || !Number.isFinite(v)) return '—';
  return `${v >= 0 ? '+' : ''}${v.toFixed(dp)}%`;
}

function viewLabel(v?: string): string {
  if (!v) return '—';
  return v.replace(/_/g, ' ').toUpperCase();
}

function viewClass(v?: string): string {
  switch (v) {
    case 'bearish':             return styles.vBear;
    case 'neutral_to_bearish':  return styles.vNtB;
    case 'neutral':             return styles.vNeu;
    case 'neutral_to_bullish':  return styles.vNtBull;
    case 'bullish':             return styles.vBull;
    case 'ignore':              return styles.vIgnore;
    default:                    return '';
  }
}

export default function Nwv() {
  const [data, setData] = useState<ViewResp | null>(null);
  const [err, setErr]   = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = async () => {
    try {
      const r = await apiGet<ViewResp>('/api/nwv/view');
      setData(r); setErr(null);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    }
  };

  useEffect(() => { void load(); }, []);

  const recompute = async () => {
    setBusy(true);
    try {
      await apiPost('/api/nwv/recompute', {});
      await load();
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const view = data?.view;
  const ws   = data?.weekly_state;
  const cprWidth = ws?.cpr_width_pct;
  const bucketCls =
    ws?.cpr_bucket === 'wide'   ? styles.bucketWide   :
    ws?.cpr_bucket === 'narrow' ? styles.bucketNarrow :
                                   styles.bucketNormal;

  return (
    <div className={styles.root}>
      <div className={styles.headerRow}>
        <div>
          <div className="page-title">Nifty Weekly View</div>
          <div className="page-subtitle">
            Phase 0 · view-only · no trades. Fires every Monday 09:46 IST.
          </div>
        </div>
        <div>
          <button className={styles.btn} onClick={recompute} disabled={busy}>
            {busy ? 'Computing…' : 'Recompute'}
          </button>
        </div>
      </div>

      {err ? <div className={styles.error}>{err}</div> : null}

      {/* Final view card */}
      <div className={styles.finalCard}>
        <div className={styles.finalLeft}>
          <div className={styles.finalLabel}>Final view</div>
          <div className={`${styles.finalValue} ${viewClass(view?.final_view)}`}>
            {viewLabel(view?.final_view)}
          </div>
          <div className={styles.conviction}>
            {view ? `Conviction ${view.conviction}/5` : ''}
          </div>
        </div>
        <div className={styles.finalRight}>
          <div className={styles.tradeLine}>
            <span className={styles.k}>Suggested:</span>{' '}
            <span className={styles.v}>{view?.instrument_choice || '—'}</span>
          </div>
          <div className={styles.tradeLine}>
            <span className={styles.k}>Expected range:</span>{' '}
            <span className={styles.v}>
              {view?.expected_range_low != null && view?.expected_range_high != null
                ? `${fmtRs(view.expected_range_low, 0)} – ${fmtRs(view.expected_range_high, 0)}`
                : view?.expected_range_low != null
                ? `above ${fmtRs(view.expected_range_low, 0)}`
                : view?.expected_range_high != null
                ? `below ${fmtRs(view.expected_range_high, 0)}`
                : '—'}
            </span>
          </div>
          <div className={styles.tradeLine}>
            <span className={styles.k}>Time stop:</span>{' '}
            <span className={styles.v}>{view?.time_stop || 'Fri 15:15'}</span>
          </div>
          <div className={styles.tradeLineSm}>
            Week of {view?.week_start || ws?.week_start || '—'} ·
            generated {view ? new Date(view.generated_at).toLocaleString() : '—'}
          </div>
        </div>
      </div>

      {!view && (
        <div className={styles.waiting}>
          {data?.message || 'No view computed yet. Monday 09:46 IST is the next auto-run.'}
        </div>
      )}

      {view && (
        <>
          {/* Derivation card */}
          <div className={styles.grid2}>
            <div className={styles.card}>
              <div className={styles.cardTitle}>Derivation</div>
              <div className={styles.kv}>
                <span>Base view</span>
                <b className={viewClass(view.base_view)}>{viewLabel(view.base_view)}</b>
              </div>
              <div className={styles.kv}>
                <span>Gap {fmtPct(view.gap_pct)}</span>
                <b>{view.gap_tier}{view.gap_direction ? ` / ${view.gap_direction}` : ''}</b>
              </div>
              <div className={styles.kv}>
                <span>Monthly override</span>
                <b>{view.monthly_override_applied ? view.monthly_override_side : 'none'}</b>
              </div>
              <div className={styles.kv}>
                <span>Final view</span>
                <b className={viewClass(view.final_view)}>{viewLabel(view.final_view)}</b>
              </div>
            </div>

            <div className={styles.card}>
              <div className={styles.cardTitle}>Regime filters</div>
              <div className={styles.kv}>
                <span>CPR width</span>
                <b>
                  <span className={`${styles.bucket} ${bucketCls}`}>{ws?.cpr_bucket}</span>
                  {cprWidth != null ? ` ${cprWidth.toFixed(2)}%` : ''}
                </b>
              </div>
              <div className={styles.kv}>
                <span>VIX</span>
                <b>
                  {view.vix_value != null ? view.vix_value.toFixed(2) : '—'}
                  {view.vix_pct_rank != null ? ` (${view.vix_pct_rank.toFixed(0)} %ile)` : ''}
                </b>
              </div>
              <div className={styles.kv}>
                <span>ADX daily</span>
                <b>
                  {view.adx_daily != null ? view.adx_daily.toFixed(1) : '—'}
                  {view.adx_bucket ? ` · ${view.adx_bucket}` : ''}
                </b>
              </div>
              <div className={styles.kv}>
                <span>1st candle range</span>
                <b>{view.first_candle_range_pct != null ? `${view.first_candle_range_pct.toFixed(2)}%` : '—'}</b>
              </div>
              <div className={styles.kv}>
                <span>Wick position</span>
                <b>
                  {view.first_candle_wick_pos_pct != null
                    ? `${view.first_candle_wick_pos_pct.toFixed(0)}% ${
                        view.first_candle_wick_pos_pct < 30 ? '(near low)' :
                        view.first_candle_wick_pos_pct > 70 ? '(near high)' : '(mid)'
                      }`
                    : '—'}
                </b>
              </div>
            </div>
          </div>

          {/* First candle + pivots */}
          <div className={styles.grid2}>
            <div className={styles.card}>
              <div className={styles.cardTitle}>Monday 1st 30-min candle</div>
              <div className={styles.kv}>
                <span>Open → Close</span>
                <b>{fmtRs(view.first_candle_open, 2)} → {fmtRs(view.first_candle_close, 2)}</b>
              </div>
              <div className={styles.kv}>
                <span>High / Low</span>
                <b>{fmtRs(view.first_candle_high, 2)} / {fmtRs(view.first_candle_low, 2)}</b>
              </div>
              <div className={styles.kv}>
                <span>Body</span>
                <b>{view.first_candle_body}</b>
              </div>
              <div className={styles.kv}>
                <span>Position vs CPR</span>
                <b>{view.first_candle_pos}</b>
              </div>
            </div>

            <div className={styles.card}>
              <div className={styles.cardTitle}>Weekly pivots</div>
              <div className={styles.kv}><span>R2</span><b>{fmtRs(ws?.pivot_r2, 2)}</b></div>
              <div className={styles.kv}><span>R1</span><b>{fmtRs(ws?.pivot_r1, 2)}</b></div>
              <div className={styles.kv}><span>PP / TC</span><b>{fmtRs(ws?.pivot_pp, 2)} / {fmtRs(ws?.cpr_tc, 2)}</b></div>
              <div className={styles.kv}><span>BC</span><b>{fmtRs(ws?.cpr_bc, 2)}</b></div>
              <div className={styles.kv}><span>S1</span><b>{fmtRs(ws?.pivot_s1, 2)}</b></div>
              <div className={styles.kv}><span>S2</span><b>{fmtRs(ws?.pivot_s2, 2)}</b></div>
            </div>
          </div>

          {/* Stacked levels */}
          <div className={styles.grid2}>
            <div className={styles.card}>
              <div className={styles.cardTitle}>Stacked supports (weekly + daily clusters)</div>
              {view.stacked_supports && view.stacked_supports.length
                ? view.stacked_supports.map((s, i) => (
                    <div className={styles.stack} key={i}>
                      <b>{fmtRs(s.price, 2)}</b>
                      <span className={styles.muted}>
                        {s.components.join(' + ')}
                        {s.weekly_price && s.daily_price
                          ? ` · w${fmtRs(s.weekly_price, 0)} d${fmtRs(s.daily_price, 0)}`
                          : ''}
                      </span>
                    </div>
                  ))
                : <div className={styles.mute}>No clusters within 0.25% of spot.</div>}
            </div>
            <div className={styles.card}>
              <div className={styles.cardTitle}>Stacked resistances</div>
              {view.stacked_resistances && view.stacked_resistances.length
                ? view.stacked_resistances.map((s, i) => (
                    <div className={styles.stack} key={i}>
                      <b>{fmtRs(s.price, 2)}</b>
                      <span className={styles.muted}>
                        {s.components.join(' + ')}
                        {s.weekly_price && s.daily_price
                          ? ` · w${fmtRs(s.weekly_price, 0)} d${fmtRs(s.daily_price, 0)}`
                          : ''}
                      </span>
                    </div>
                  ))
                : <div className={styles.mute}>No clusters within 0.25% of spot.</div>}
            </div>
          </div>

          {view.notes ? <div className={styles.notes}>Notes: {view.notes}</div> : null}
        </>
      )}
    </div>
  );
}
