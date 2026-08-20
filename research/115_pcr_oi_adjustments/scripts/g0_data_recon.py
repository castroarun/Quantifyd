#!/usr/bin/env python3
"""
research/115_pcr_oi_adjustments -- G0 DATA SANITY

Question: is the recorded option chain's OI good enough to support *intraday* signals?
Exchange OI is disseminated with lag; if it only refreshes every N minutes (or not at all
within a session) then every dOI-based intraday signal is mechanically stale and the study
ends here.

Outputs (results/):
  g0_day_coverage.csv   -- per (date, venue): minutes, expiries, null rates, first/last stamp
  g0_oi_staleness.csv   -- per (date, venue): how often OI actually changes minute-to-minute
  g0_summary.json       -- pooled headline numbers

READ-ONLY. No writes to any DB.
"""
import sqlite3, os, json, csv, sys
from collections import defaultdict

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
                  'backtest_data', 'options_data.db')
DB = os.path.abspath(DB)
OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
os.makedirs(OUT, exist_ok=True)

VENUES = ('NIFTY', 'SENSEX')


def conn():
    c = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
    c.execute('PRAGMA query_only=ON')
    return c


def trading_days(c):
    """Distinct dates present, per venue. Cheap: use the download_log if usable else derive."""
    days = defaultdict(set)
    for sym, st in c.execute(
            "select symbol, substr(snapshot_time,1,10) d from download_log "
            "where symbol in ('NIFTY','SENSEX') group by symbol, d"):
        days[sym].add(st)
    if not days:
        for sym in VENUES:
            for (d,) in c.execute(
                    "select distinct substr(snapshot_time,1,10) from option_chain where symbol=?", (sym,)):
                days[sym].add(d)
    return {k: sorted(v) for k, v in days.items()}


def main():
    c = conn()
    days = trading_days(c)
    all_days = sorted(set(days.get('NIFTY', [])) | set(days.get('SENSEX', [])))
    print('days found: %d  %s .. %s' % (len(all_days), all_days[0], all_days[-1]))

    cov_f = open(os.path.join(OUT, 'g0_day_coverage.csv'), 'w', newline='')
    cov = csv.DictWriter(cov_f, fieldnames=[
        'date', 'venue', 'rows', 'minutes', 'first_stamp', 'last_stamp', 'n_expiries',
        'near_expiry', 'dte', 'oi_null_pct', 'oi_zero_pct', 'iv_null_pct', 'bid_null_pct',
        'n_strikes_near', 'strike_step'])
    cov.writeheader()

    st_f = open(os.path.join(OUT, 'g0_oi_staleness.csv'), 'w', newline='')
    st = csv.DictWriter(st_f, fieldnames=[
        'date', 'venue', 'dte', 'series_tested', 'minute_pairs',
        'pct_minutes_oi_changed', 'median_run_len', 'p90_run_len', 'max_run_len',
        'pct_minutes_ltp_changed', 'pct_minutes_vol_changed'])
    st.writeheader()

    pooled = defaultdict(list)

    for d in all_days:
        lo, hi = d + 'T00:00:00', d + 'T23:59:59'
        for sym in VENUES:
            rows = list(c.execute(
                "select snapshot_time, expiry_date, strike, instrument_type, ltp, bid, oi, volume, iv, "
                "underlying_spot from option_chain where symbol=? and snapshot_time>=? and snapshot_time<=?",
                (sym, lo, hi)))
            if not rows:
                continue
            stamps = sorted({r[0] for r in rows})
            exps = sorted({r[1] for r in rows})
            near = exps[0]
            dte = (int(near[:4]) * 372 + int(near[5:7]) * 31 + int(near[8:10])) - \
                  (int(d[:4]) * 372 + int(d[5:7]) * 31 + int(d[8:10]))
            nr = [r for r in rows if r[1] == near]
            strikes = sorted({r[2] for r in nr})
            step = min((strikes[i + 1] - strikes[i]) for i in range(len(strikes) - 1)) if len(strikes) > 1 else 0
            n = len(rows)
            cov.writerow(dict(
                date=d, venue=sym, rows=n, minutes=len(stamps),
                first_stamp=stamps[0][11:], last_stamp=stamps[-1][11:], n_expiries=len(exps),
                near_expiry=near, dte=dte,
                oi_null_pct=round(100.0 * sum(1 for r in rows if r[6] is None) / n, 2),
                oi_zero_pct=round(100.0 * sum(1 for r in rows if r[6] == 0) / n, 2),
                iv_null_pct=round(100.0 * sum(1 for r in rows if r[8] is None) / n, 2),
                bid_null_pct=round(100.0 * sum(1 for r in rows if r[5] is None) / n, 2),
                n_strikes_near=len(strikes), strike_step=step))

            # ---- staleness on the near-expiry ATM +/-5 strikes -------------------
            spot = None
            for r in nr:
                if r[9]:
                    spot = r[9]
                    break
            if spot is None or not strikes:
                continue
            atm = min(strikes, key=lambda k: abs(k - spot))
            ai = strikes.index(atm)
            keep = set(strikes[max(0, ai - 5): ai + 6])
            ser = defaultdict(dict)   # (strike,type) -> {stamp: (oi, ltp, vol)}
            for s_, e_, k_, t_, ltp_, bid_, oi_, vol_, iv_, sp_ in nr:
                if k_ in keep:
                    ser[(k_, t_)][s_] = (oi_, ltp_, vol_)
            pairs = oich = ltpch = volch = 0
            runs = []
            for key, m in ser.items():
                ss = sorted(m)
                run = 1
                for i in range(1, len(ss)):
                    a, b = m[ss[i - 1]], m[ss[i]]
                    pairs += 1
                    if a[0] is not None and b[0] is not None:
                        if b[0] != a[0]:
                            oich += 1
                            runs.append(run)
                            run = 1
                        else:
                            run += 1
                    if a[1] is not None and b[1] is not None and b[1] != a[1]:
                        ltpch += 1
                    if a[2] is not None and b[2] is not None and b[2] != a[2]:
                        volch += 1
                runs.append(run)
            runs.sort()
            if pairs:
                row = dict(date=d, venue=sym, dte=dte, series_tested=len(ser), minute_pairs=pairs,
                           pct_minutes_oi_changed=round(100.0 * oich / pairs, 2),
                           median_run_len=runs[len(runs) // 2] if runs else '',
                           p90_run_len=runs[int(0.9 * (len(runs) - 1))] if runs else '',
                           max_run_len=runs[-1] if runs else '',
                           pct_minutes_ltp_changed=round(100.0 * ltpch / pairs, 2),
                           pct_minutes_vol_changed=round(100.0 * volch / pairs, 2))
                st.writerow(row)
                pooled[sym].append(row)
        cov_f.flush(); st_f.flush()
        print('  %s done' % d, flush=True)

    cov_f.close(); st_f.close()

    summ = {}
    for sym, rr in pooled.items():
        if not rr:
            continue
        summ[sym] = dict(
            days=len(rr),
            mean_pct_minutes_oi_changed=round(sum(r['pct_minutes_oi_changed'] for r in rr) / len(rr), 2),
            mean_pct_minutes_ltp_changed=round(sum(r['pct_minutes_ltp_changed'] for r in rr) / len(rr), 2),
            mean_pct_minutes_vol_changed=round(sum(r['pct_minutes_vol_changed'] for r in rr) / len(rr), 2),
            mean_median_run_len=round(sum(float(r['median_run_len'] or 0) for r in rr) / len(rr), 2),
        )
    json.dump(summ, open(os.path.join(OUT, 'g0_summary.json'), 'w'), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == '__main__':
    main()
