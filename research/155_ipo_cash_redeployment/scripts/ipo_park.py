"""research/155 — the IPO-Base sleeve with an EXTERNAL CASH SINK/SOURCE.

This is r/153's `simulate_ipo` (position-level, cash-constrained, FY-netted tax) extended so
that idle sleeve cash can be PARKED in an external NAV series (Open Alpha, True North, a
50/50 of the two, or NIFTYBEES) and PULLED BACK when an IPO candidate triggers.

NAV-level blending is NOT sufficient for this question: redeployment changes the sleeve's own
cash path, its position sizes and its trade set. So the sleeve is simulated at position level
and only the finished sleeve NAV is blended.

EVERY pull-back friction is charged (Arun, 2026-09-05: "Pull-back friction must be modelled,
not waived"):
  * transaction cost on BOTH the redemption and the re-parking of the parked leg
  * tax on the realised NAV-lot gain (20% STCG / 12.5% LTCG, FY-netted) -- `park_tax='full'`,
    the conservative UPPER bound, because the parked NAV series is itself already after-tax,
    so this double-counts.  `park_tax='txn'` is the lower bound (transaction cost only)
  * T+1 settlement: cash from a sale on day t arrives on t+1, so the entry that forced the
    sale is MISSED that day.  `settle_days=0` isolates how much of any edge is a settlement
    artefact
  * an explicit lot-selection policy: pro-rata / LIFO / FIFO

Reducing `park_nav=None` gives back r/153's sleeve bit-for-bit (validated in run_sweep.py
phase R against results/ipo_equity_seeds.csv).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/arun/quantifyd")
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "153_ipo_base" / "scripts"))
import ipo_replay as ir  # noqa: E402

CAPITAL = ir.CAPITAL


def forward_pool_empty(ctx, horizon, max_age_m=6, min_bars=25):
    """park_allowed[t] for arm E: True when NO name can possibly become an eligible IPO-base
    candidate within the next `horizon` trading days.

    CAUSAL by construction.  `young` depends only on (a) bars since listing and (b) calendar
    age -- both deterministic once a listing date is in the past -- and the liquidity leg is
    taken AS OF t and held constant forward.  No future price is consulted.
    """
    young = (ctx.AGE > 0) & (ctx.AGE <= max_age_m * 30.44) & (ctx.BARS >= min_bars)
    T, N = young.shape
    # fwd[t, c] = any(young[t+1 : t+1+horizon, c])
    fwd = np.zeros((T, N), dtype=bool)
    # reverse sliding OR: last_true[c] = smallest u >= t+1 with young[u, c] == True
    BIG = 1 << 40
    last_true = np.full(N, BIG, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        fwd[t] = (last_true - t) <= horizon
        last_true = np.where(young[t], t, last_true)
    pool = fwd & ctx.ELIG          # liquid AS OF t, and young at some point in the horizon
    return ~pool.any(axis=1)


def simulate_park(seed, days_idx, dates, C, O, PIV, LO, SMA, TVp, TRIG, weak,
                  park_lvl=None, park_allowed=None,
                  *, cost=0.0025, park_cost=None, stop=0.08, slots=8, size_pct=0.1875,
                  target=0.25, stcg=0.20, ltcg=0.125, cash_yield=0.05,
                  settle_days=1, reserve_slots=0, cadence="daily",
                  sell_policy="prorata", park_tax="full", min_park_frac=0.02,
                  capital=CAPITAL, frictionless=False):
    """Returns dict(nav, invested, parked, trades, diag).

    park_lvl : float[T] price level of the parked asset aligned to `dates` (NaN where absent)
    park_allowed : bool[T]  may hold the parked leg on that day (arm E gate); None -> always
    """
    if park_cost is None:
        park_cost = cost
    if frictionless:
        park_cost = 0.0
        settle_days = 0
        park_tax = "none"
    rng = np.random.default_rng(seed)
    T = len(dates)
    if park_allowed is None:
        park_allowed = np.ones(T, dtype=bool)
    parking = park_lvl is not None

    cash = float(capital)
    pending = {}                 # settle_day_index -> cash arriving
    lots: list[list] = []        # [units, cost_price, entry_day_index]
    positions = []               # (col, entry_i, buy, qty, stop_px, tvp)
    trades = []
    nav = np.empty(len(days_idx))
    inv = np.empty(len(days_idx))
    pk = np.empty(len(days_idx))
    pull_n_d = np.zeros(len(days_idx))
    pull_c_d = np.zeros(len(days_idx))
    miss_d = np.zeros(len(days_idx))
    y_day = 1.0 + cash_yield / 252.0
    fy_st = fy_lt = 0.0
    carry = 0.0
    passed_up = 0
    n_pull = 0; pull_val = 0.0; pull_cost = 0.0; pull_tax = 0.0
    n_park = 0; park_val = 0.0; park_cost_paid = 0.0
    n_missed_settle = 0
    park_days = 0

    def fy_of(d):
        return d.year if d.month >= 4 else d.year - 1

    cur_fy = fy_of(dates[days_idx[0]])

    def parked_units():
        return sum(l[0] for l in lots)

    def sell_park(target_gross, i, lvl):
        """Liquidate `target_gross` of parked VALUE (capped at what we hold).
        Returns net proceeds after cost.  Books tax on the realised lot gain."""
        nonlocal fy_st, fy_lt, pull_val, pull_cost, pull_tax
        tot_u = parked_units()
        if tot_u <= 0 or not np.isfinite(lvl) or lvl <= 0:
            return 0.0
        want_u = min(target_gross / lvl, tot_u)
        if want_u <= 0:
            return 0.0
        order = (list(range(len(lots))) if sell_policy == "fifo"
                 else list(range(len(lots) - 1, -1, -1)) if sell_policy == "lifo"
                 else None)
        gross = 0.0
        gain_st = gain_lt = 0.0
        if order is None:                                   # pro-rata across every lot
            frac = want_u / tot_u
            for L in lots:
                u = L[0] * frac
                g = u * (lvl - L[1])
                held = (dates[i] - dates[L[2]]).days
                if held > 365:
                    gain_lt += g
                else:
                    gain_st += g
                gross += u * lvl
                L[0] -= u
        else:
            left = want_u
            for k in order:
                if left <= 1e-12:
                    break
                L = lots[k]
                u = min(L[0], left)
                if u <= 0:
                    continue
                g = u * (lvl - L[1])
                held = (dates[i] - dates[L[2]]).days
                if held > 365:
                    gain_lt += g
                else:
                    gain_st += g
                gross += u * lvl
                L[0] -= u
                left -= u
        lots[:] = [L for L in lots if L[0] > 1e-9]
        c_ = gross * park_cost
        net = gross - c_
        pull_val += gross
        pull_cost += c_
        if park_tax == "full":
            # conservative: the pre-cost gain is booked (the cost already reduced cash)
            fy_st += gain_st
            fy_lt += gain_lt
            pull_tax += stcg * max(gain_st, 0.0) + ltcg * max(gain_lt, 0.0)
        return net

    for k, i in enumerate(days_idx):
        # 1. idle-cash yield on SETTLED cash only
        if cash_yield and cash > 0:
            cash *= y_day
        # 2. settlement pipe
        if pending:
            arr = pending.pop(i, 0.0)
            if arr:
                cash += arr
        # 3. financial-year tax settlement
        d = dates[i]
        if stcg and fy_of(d) != cur_fy:
            st, lt, cf = fy_st, fy_lt, carry
            if cf < 0:
                u = min(-cf, max(st, 0.0)); st -= u; cf += u
                u = min(-cf, max(lt, 0.0)); lt -= u; cf += u
            if st < 0:
                lt += st; st = 0.0
            if lt < 0:
                cf += lt; lt = 0.0
            cash -= stcg * max(st, 0.0) + ltcg * max(lt, 0.0)
            carry = cf
            fy_st = fy_lt = 0.0
            cur_fy = fy_of(d)

        lvl = float(park_lvl[i]) if parking else np.nan
        lvl_ok = parking and np.isfinite(lvl) and lvl > 0
        pv = parked_units() * lvl if lvl_ok else 0.0
        pend_tot = sum(pending.values()) if pending else 0.0

        # 4. entries
        if not weak[i]:
            cand = np.nonzero(TRIG[i])[0]
            if len(cand):
                mtm = sum(q * (C[i, c] if not np.isnan(C[i, c]) else b)
                          for c, _, b, q, _, _ in positions)
                eq = cash + pend_tot + pv + mtm
                cand = rng.permutation(cand)
                for c in cand:
                    if len(positions) >= slots:
                        passed_up += 1
                        continue
                    piv = float(PIV[i, c])
                    fill = max(piv, float(O[i, c]))
                    if not np.isfinite(fill) or fill <= 0:
                        continue
                    sp = fill * (1 - stop)
                    size = size_pct * eq
                    qty = int(size / fill)
                    if qty < 1:
                        passed_up += 1
                        continue
                    need = qty * fill * (1 + cost)
                    if cash < need and lvl_ok and parked_units() > 0:
                        short = need - cash
                        gross_needed = short / max(1e-9, (1 - park_cost))
                        c_before = pull_cost + pull_tax
                        net = sell_park(gross_needed, i, lvl)
                        n_pull += 1
                        pull_n_d[k] += 1
                        pull_c_d[k] += (pull_cost + pull_tax) - c_before
                        pv = parked_units() * lvl
                        if settle_days <= 0:
                            cash += net
                        else:
                            pending[i + settle_days] = pending.get(i + settle_days, 0.0) + net
                            pend_tot += net
                            n_missed_settle += 1
                            miss_d[k] += 1
                            passed_up += 1
                            continue                        # entry missed today
                    if cash < need:
                        passed_up += 1
                        continue
                    cash -= need
                    positions.append((c, i, fill, qty, sp, float(TVp[i, c])))

        # 5. exits at the close
        still = []
        for c, ei, b, q, sp, tv in positions:
            cl = C[i, c]
            if np.isnan(cl):
                still.append((c, ei, b, q, sp, tv)); continue
            reason = None
            if cl <= sp:
                reason = "stop"
            elif target is not None and cl >= b * (1 + target):
                reason = "target"
            elif i > ei and not np.isnan(SMA[i, c]) and cl < SMA[i, c]:
                reason = "trail"
            if reason:
                cash += q * float(cl) * (1 - cost)
                pnl = q * (float(cl) * (1 - cost) - b * (1 + cost))
                held = (dates[i] - dates[ei]).days
                if stcg:
                    if held > 365:
                        fy_lt += pnl
                    else:
                        fy_st += pnl
                trades.append(dict(col=int(c), ei=int(ei), xi=int(i), buy=b, sell=float(cl),
                                   qty=q, reason=reason, held=held, tv=tv,
                                   ret=float(cl) / b - 1.0, notional=q * b))
            else:
                still.append((c, ei, b, q, sp, tv))
        positions = still

        # 6. re-park / gate unwind
        if parking and lvl_ok:
            mtm = sum(q * (C[i, c] if not np.isnan(C[i, c]) else b)
                      for c, _, b, q, _, _ in positions)
            pv = parked_units() * lvl
            pend_tot = sum(pending.values()) if pending else 0.0
            navnow = cash + pend_tot + pv + mtm
            if not park_allowed[i]:
                if pv > 0:                                   # gate closed -> unwind fully
                    c_before = pull_cost + pull_tax
                    net = sell_park(pv, i, lvl)
                    n_pull += 1
                    pull_n_d[k] += 1
                    pull_c_d[k] += (pull_cost + pull_tax) - c_before
                    if settle_days <= 0:
                        cash += net
                    else:
                        pending[i + settle_days] = pending.get(i + settle_days, 0.0) + net
            else:
                do = (cadence == "daily"
                      or (cadence == "weekly" and (k % 5 == 0))
                      or (cadence == "monthly" and
                          (k == 0 or dates[days_idx[k - 1]].month != d.month)))
                if do:
                    reserve = reserve_slots * size_pct * navnow
                    excess = cash - reserve
                    if excess > min_park_frac * navnow and excess > 0:
                        units = excess * (1 - park_cost) / lvl
                        lots.append([units, lvl, i])
                        cash -= excess
                        n_park += 1
                        park_val += excess
                        park_cost_paid += excess * park_cost

        mtm = sum(q * (C[i, c] if not np.isnan(C[i, c]) else b)
                  for c, _, b, q, _, _ in positions)
        pv = (parked_units() * lvl) if lvl_ok else 0.0
        pend_tot = sum(pending.values()) if pending else 0.0
        nav[k] = cash + pend_tot + pv + mtm
        inv[k] = mtm
        pk[k] = pv
        if pv > 0:
            park_days += 1

    last = days_idx[-1]
    for c, ei, b, q, sp, tv in positions:
        cl = C[last, c]
        px = float(cl) if not np.isnan(cl) else b
        trades.append(dict(col=int(c), ei=int(ei), xi=int(last), buy=b, sell=px, qty=q,
                           reason="open_marked", held=(dates[last] - dates[ei]).days, tv=tv,
                           ret=px / b - 1.0, notional=q * b))

    diag = dict(passed_up=passed_up, n_pull=n_pull, pull_val=pull_val, pull_cost=pull_cost,
                pull_tax=pull_tax, n_park=n_park, park_val=park_val,
                park_cost_paid=park_cost_paid, n_missed_settle=n_missed_settle,
                park_days=park_days, park_days_pct=100.0 * park_days / len(days_idx))
    return dict(nav=nav, invested=inv, parked=pk, trades=trades, diag=diag,
                pull_n_d=pull_n_d, pull_c_d=pull_c_d, miss_d=miss_d)
