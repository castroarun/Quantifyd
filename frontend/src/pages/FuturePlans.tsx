import { useEffect, useState } from 'react';
import styles from './FuturePlans.module.css';
import { apiGet } from '../api/client';

interface PlanItem {
  label?: string;
  body?: string;
  pros?: string[];
  cons?: string[];
}

interface PlanSection {
  title: string;
  body?: string;
  items?: (string | PlanItem)[];
}

interface Plan {
  id: string;
  title: string;
  subtitle?: string;
  status: 'idea' | 'designing' | 'backtesting' | 'building' | 'live' | 'parked';
  created: string;
  tags?: string[];
  sections: PlanSection[];
}

interface Resp {
  plans: Plan[];
}

const STATUS_LABEL: Record<Plan['status'], string> = {
  idea: 'Idea',
  designing: 'Designing',
  backtesting: 'Backtesting',
  building: 'Building',
  live: 'Live',
  parked: 'Parked',
};

function fmtDate(d: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(d);
  if (!m) return d;
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${parseInt(m[3], 10)} ${months[parseInt(m[2], 10) - 1]} ${m[1]}`;
}

function isItemObject(x: string | PlanItem): x is PlanItem {
  return typeof x === 'object' && x !== null;
}

function Section({ s }: { s: PlanSection }) {
  return (
    <div className={styles.section}>
      <div className={styles.sectionTitle}>{s.title}</div>
      {s.body && <div className={styles.sectionBody}>{s.body}</div>}
      {s.items && (
        <div className={styles.itemList}>
          {s.items.map((it, idx) => {
            if (!isItemObject(it)) {
              return (
                <div key={idx} className={styles.itemLine}>
                  <span className={styles.bullet}>•</span>
                  <span>{it}</span>
                </div>
              );
            }
            return (
              <div key={idx} className={styles.itemCard}>
                {it.label && <div className={styles.itemLabel}>{it.label}</div>}
                {it.body && <div className={styles.itemBody}>{it.body}</div>}
                {(it.pros || it.cons) && (
                  <div className={styles.pcWrap}>
                    {it.pros && (
                      <div className={styles.pcCol}>
                        <div className={`${styles.pcHead} ${styles.pcHeadPos}`}>Pros</div>
                        <ul className={styles.pcList}>
                          {it.pros.map((p, i) => <li key={i}>{p}</li>)}
                        </ul>
                      </div>
                    )}
                    {it.cons && (
                      <div className={styles.pcCol}>
                        <div className={`${styles.pcHead} ${styles.pcHeadNeg}`}>Cons</div>
                        <ul className={styles.pcList}>
                          {it.cons.map((c, i) => <li key={i}>{c}</li>)}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function PlanCard({ plan, defaultOpen }: { plan: Plan; defaultOpen: boolean }) {
  const statusClass = styles[`status_${plan.status}`] || styles.status_idea;
  return (
    <details className={styles.plan} open={defaultOpen}>
      <summary className={styles.planHead}>
        <span className={styles.planArrow} aria-hidden>›</span>
        <div className={styles.planHeadMain}>
          <div className={styles.planTitleRow}>
            <span className={styles.planTitle}>{plan.title}</span>
            <span className={`${styles.statusChip} ${statusClass}`}>
              {STATUS_LABEL[plan.status]}
            </span>
          </div>
          {plan.subtitle && <div className={styles.planSubtitle}>{plan.subtitle}</div>}
          <div className={styles.planMeta}>
            <span className={styles.metaDate}>Added {fmtDate(plan.created)}</span>
            {plan.tags && plan.tags.length > 0 && (
              <span className={styles.tagList}>
                {plan.tags.map((t) => (
                  <span key={t} className={styles.tag}>#{t}</span>
                ))}
              </span>
            )}
          </div>
        </div>
      </summary>
      <div className={styles.planBody}>
        {plan.sections.map((s, i) => <Section key={i} s={s} />)}
      </div>
    </details>
  );
}

export default function FuturePlans() {
  const [data, setData] = useState<Resp | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiGet<Resp>('/api/plans/future')
      .then((r) => { setData(r); setLoading(false); })
      .catch((e) => { setErr(String(e)); setLoading(false); });
  }, []);

  if (loading) return <div className={styles.loading}>Loading plans…</div>;
  if (err) return <div className={styles.error}>Failed to load: {err}</div>;
  if (!data || !data.plans?.length) {
    return (
      <div className={styles.page}>
        <div className={styles.header}>
          <div className={styles.title}>Future implementation plans</div>
          <div className={styles.subtitle}>No plans logged yet.</div>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div className={styles.title}>Future implementation plans</div>
        <div className={styles.subtitle}>
          Ideas and design sketches for strategies yet to be built.
          {' '}Edit <code>data/future_plans.json</code> to add more.
        </div>
      </div>
      <div className={styles.list}>
        {data.plans.map((p, idx) => (
          <PlanCard key={p.id} plan={p} defaultOpen={idx === 0} />
        ))}
      </div>
    </div>
  );
}
