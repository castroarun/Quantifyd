"""Straddle Intraday Study - dynamic query API over the AlgoTest trade archive.

Read-only blueprint over backtest_data/algotest_studies.db (built by
scripts/load_algotest_studies.py). Stored rows keep GROSS P/L and premium
turnover, so the cost model is a query-time parameter and every metric is
recomputed per request rather than pre-baked.

Register in app.py:
    from services.straddle_study_api import straddle_study_bp
    app.register_blueprint(straddle_study_bp)
"""
from flask import Blueprint, jsonify, request
from pathlib import Path
from collections import defaultdict
import sqlite3, math, statistics

straddle_study_bp = Blueprint('straddle_study', __name__)
DB = Path(__file__).parent.parent / 'backtest_data' / 'algotest_studies.db'

DEF_COST_RATE = 0.59      # % of premium turnover
DEF_COST_FLAT = 80.0      # Rs. per trade
DEF_WR_MIN = 45.0
DEF_STREAK_MAX = 7


def _con():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def _csv_param(name, cast=str):
    raw = request.args.get(name, '').strip()
    if not raw:
        return None
    out = []
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(cast(part))
        except (TypeError, ValueError):
            continue
    return out or None


def _maxdd(seq):
    peak = cum = mdd = 0.0
    for v in seq:
        cum += v
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    return mdd


def _metrics(rows, label, meta=None):
    """rows: list of (entry_date, year, net) ordered by date."""
    v = [r[2] for r in rows]
    n = len(v)
    if n < 2:
        return None
    wins = [x for x in v if x > 0]
    loss = [x for x in v if x <= 0]
    net = sum(v)
    mean = net / n
    sd = statistics.stdev(v) if n > 1 else 0.0
    lose = win = cl = cw = 0
    for x in v:
        if x <= 0:
            cl += 1
            cw = 0
        else:
            cw += 1
            cl = 0
        lose = max(lose, cl)
        win = max(win, cw)
    dd = _maxdd(v)
    per_year = defaultdict(float)
    for r in rows:
        per_year[r[1]] += r[2]
    years = sorted(per_year)
    span = max(len(years), 1)
    aw = sum(wins) / len(wins) if wins else 0.0
    al = sum(loss) / len(loss) if loss else 0.0
    out = dict(
        label=label,
        n=n,
        net=round(net),
        mean=round(mean),
        median=round(statistics.median(v)),
        win_pct=round(100.0 * len(wins) / n, 1),
        avg_win=round(aw),
        avg_loss=round(al),
        rr=round(aw / abs(al), 2) if al else None,
        pf=round(sum(wins) / abs(sum(loss)), 2) if loss and sum(loss) else None,
        maxdd=round(dd),
        net_dd=round(net / abs(dd), 2) if dd else None,
        calmar=round((net / span) / abs(dd), 2) if dd else None,
        t=round(mean / (sd / math.sqrt(n)), 2) if sd else 0.0,
        worst=round(min(v)),
        best=round(max(v)),
        lose_streak=lose,
        win_streak=win,
        years_positive=sum(1 for y in years if per_year[y] > 0),
        years_total=len(years),
        per_year={str(y): round(per_year[y]) for y in years},
    )
    if meta:
        out.update(meta)
    return out


def _gate(m, wr_min, streak_max):
    if m is None:
        return m
    stats_ok = (m['t'] >= 2.0
                and (m['pf'] or 0) >= 1.3
                and m['years_total'] > 0
                and m['years_positive'] / m['years_total'] >= 0.8)
    trade_ok = m['win_pct'] >= wr_min and m['lose_streak'] <= streak_max
    m['gate_stats'] = bool(stats_ok)
    m['gate_tradeable'] = bool(trade_ok)
    if stats_ok and trade_ok:
        m['verdict'] = 'PASS'
    elif stats_ok:
        m['verdict'] = 'rej: WR/streak'
    else:
        m['verdict'] = 'rej: stats'
    return m


@straddle_study_bp.route('/api/straddle-study/runs')
def runs():
    with _con() as c:
        rs = [dict(r) for r in c.execute(
            "SELECT * FROM at_runs ORDER BY index_name, sl_pct")]
        yrs = [r[0] for r in c.execute(
            "SELECT DISTINCT year FROM at_trades ORDER BY year")]
        dtes = [r[0] for r in c.execute(
            "SELECT DISTINCT dte FROM at_trades ORDER BY dte")]
    return jsonify(ok=True, runs=rs,
                   indices=sorted({r['index_name'] for r in rs}),
                   years=yrs, dtes=dtes,
                   defaults=dict(cost_rate=DEF_COST_RATE, cost_flat=DEF_COST_FLAT,
                                 wr_min=DEF_WR_MIN, streak_max=DEF_STREAK_MAX))


@straddle_study_bp.route('/api/straddle-study/query')
def query():
    idxs = _csv_param('index')
    sls = _csv_param('sl', float)
    dtes = _csv_param('dte', int)
    y_from = request.args.get('year_from', type=int)
    y_to = request.args.get('year_to', type=int)
    excl = request.args.get('exclude_events', '1') != '0'
    rate = request.args.get('cost_rate', DEF_COST_RATE, type=float) / 100.0
    flat = request.args.get('cost_flat', DEF_COST_FLAT, type=float)
    scale = request.args.get('lots_scale', 1.0, type=float)
    group = request.args.get('group_by', 'run')
    sort = request.args.get('sort', 'net')
    wr_min = request.args.get('wr_min', DEF_WR_MIN, type=float)
    stk_max = request.args.get('streak_max', DEF_STREAK_MAX, type=int)

    where, args = ["1=1"], []
    if idxs:
        where.append("r.index_name IN (%s)" % ",".join(["?"] * len(idxs)))
        args += idxs
    if sls:
        where.append("r.sl_pct IN (%s)" % ",".join(["?"] * len(sls)))
        args += sls
    if dtes:
        where.append("t.dte IN (%s)" % ",".join(["?"] * len(dtes)))
        args += dtes
    if y_from:
        where.append("t.year >= ?")
        args.append(y_from)
    if y_to:
        where.append("t.year <= ?")
        args.append(y_to)
    if excl:
        where.append("t.is_event = 0")

    sql = ("SELECT r.run_id, r.index_name, r.sl_pct, r.lots, t.entry_date, t.year, "
           "t.dte, t.weekday, t.gross, t.turnover "
           "FROM at_trades t JOIN at_runs r ON r.run_id = t.run_id "
           "WHERE " + " AND ".join(where) + " ORDER BY t.entry_date")

    buckets, meta = defaultdict(list), {}
    with _con() as c:
        for r in c.execute(sql, args):
            net = (r['gross'] - rate * r['turnover'] - flat) * scale
            if group == 'dte':
                key = "DTE %d" % r['dte']
                mk = dict(dte=r['dte'])
            elif group == 'run_dte':
                key = "%s %.0f%% - DTE %d" % (r['index_name'], r['sl_pct'], r['dte'])
                mk = dict(index_name=r['index_name'], sl_pct=r['sl_pct'], dte=r['dte'])
            elif group == 'year':
                key = str(r['year'])
                mk = dict(year=r['year'])
            elif group == 'weekday':
                key = r['weekday']
                mk = dict(weekday=r['weekday'])
            else:
                key = "%s %.0f%%" % (r['index_name'], r['sl_pct'])
                mk = dict(index_name=r['index_name'], sl_pct=r['sl_pct'], lots=r['lots'])
            buckets[key].append((r['entry_date'], r['year'], net))
            meta.setdefault(key, mk)

    rows = []
    for k, vals in buckets.items():
        m = _gate(_metrics(vals, k, meta[k]), wr_min, stk_max)
        if m:
            rows.append(m)

    keyf = {
        'net': lambda m: m['net'],
        'per_trade': lambda m: m['mean'],
        'median': lambda m: m['median'],
        'win': lambda m: m['win_pct'],
        'net_dd': lambda m: m['net_dd'] if m['net_dd'] is not None else -9e9,
        'calmar': lambda m: m['calmar'] if m['calmar'] is not None else -9e9,
        'pf': lambda m: m['pf'] if m['pf'] is not None else -9e9,
        't': lambda m: m['t'],
        'worst': lambda m: m['worst'],
        'lose_streak': lambda m: -m['lose_streak'],
        'maxdd': lambda m: m['maxdd'],
    }.get(sort, lambda m: m['net'])
    rows.sort(key=keyf, reverse=True)

    return jsonify(ok=True, rows=rows, n_groups=len(rows),
                   filters=dict(index=idxs, sl=sls, dte=dtes, year_from=y_from,
                                year_to=y_to, exclude_events=excl,
                                cost_rate=rate * 100, cost_flat=flat,
                                lots_scale=scale, group_by=group, sort=sort,
                                wr_min=wr_min, streak_max=stk_max))
