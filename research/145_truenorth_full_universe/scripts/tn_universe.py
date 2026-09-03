"""research/145 — True North rules on the Open Alpha universe (adversarial revalidation of
the r/62 wider-universe rejection).

TN mechanics EXACTLY as deployed (r/144 engine reused); ONLY the universe swaps:
  top200 (control) | top500 | tvfloor (OA: 20d-median traded value >= Rs 5cr at t-1,
  ETFs excluded via bluesky_replay.ETF_RE + our ETF set, NO mcap floor).

Phases: sweep (12 offsets x tax x 3 universes + cost tiers 0.5/0.75% + cash5) |
capacity (held-name traded-value distributions, no new backtests) |
blend (OA adopted-spec 10 seeds after-tax x {U-OA TN, U-200 TN} + corr + holdings overlap).
"""
from __future__ import annotations
import sys, csv, time, pickle, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
R144 = Path("/home/arun/quantifyd/research/144_truenorth_reassessment/scripts")
OA = Path("/home/arun/quantifyd/research/142_bananapatterns_replication/scripts")
sys.path.insert(0, str(OA))
import bluesky_replay as br

_s = importlib.util.spec_from_file_location("tn", str(R144 / "tn_sweep.py"))
tn = importlib.util.module_from_spec(_s); _s.loader.exec_module(tn)

MODES = ["top200", "top500", "tvfloor"]


def _excl(sym):
    return tn._is_etf(sym) or bool(br.ETF_RE.search(sym))


class UCtx(tn.Ctx):
    """r/144 Ctx with a switchable universe for the ranking step."""

    def set_mode(self, mode):
        if getattr(self, "uni_mode", None) == mode:
            return
        self.save_ranks()
        self.uni_mode = mode
        self._rank_cache_file = (
            tn.RES / "ranks_cache.pkl" if mode == "top200"
            else RES / f"ranks_{mode}.pkl")
        self.rank_cache = (pickle.load(open(self._rank_cache_file, "rb"))
                           if self._rank_cache_file.exists() else {})
        self.uni_sizes = {}

    def ranking(self, i):
        if self.uni_mode == "top200":
            return super().ranking(i)
        if i in self.rank_cache:
            return self.rank_cache[i]
        if i <= 253:
            return None
        fresh = self.rawnn[max(0, i - 4):i + 1].any(axis=0)
        if self.uni_mode == "top500":
            w = self.tv.iloc[max(0, i - tn.LIQ_LB + 1):i + 1]
            cnt = w.notna().sum(); med = w.median()
            elig = med[(cnt >= 75) & (med > 0)].sort_values(ascending=False)
            uni = [s for s in elig.index
                   if not _excl(s) and fresh[self.sidx[s]]][:500]
        else:  # tvfloor — OA convention: 20d median TV >= 5cr as of t-1
            w = self.tv.iloc[max(0, i - 20):i]
            med = w.median()
            elig = med[med >= br.TV_FLOOR]
            uni = [s for s in elig.index
                   if not _excl(s) and fresh[self.sidx[s]]]
        self.uni_sizes[i] = len(uni)
        p1 = self.C[i]; p126 = self.C[i - 126]; p252 = self.C[i - 252]
        jb = self.sidx[tn.BENCH]
        nf126 = p1[jb] / p126[jb]; nf252 = p1[jb] / p252[jb]
        sc = {}
        for s in uni:
            j = self.sidx[s]
            a, b, c = p1[j], p126[j], p252[j]
            if a == a and b == b and c == c and b > 0 and c > 0:
                sc[s] = 0.5 * (a / b) / nf126 + 0.5 * (a / c) / nf252
        rank = [s for s, _ in sorted(sc.items(), key=lambda kv: -kv[1])][:tn.RANK_KEEP]
        self.rank_cache[i] = rank
        self._rank_dirty += 1
        if self._rank_dirty % 200 == 0:
            self.save_ranks()
        return rank


FIELDS = ["label", "mode", "offset", "tax", "rt", "cash_y"]
for w in ("w0", "wa", "w1", "w2"):
    FIELDS += [f"{w}_cagr", f"{w}_dd", f"{w}_sharpe", f"{w}_calmar"]
FIELDS += ["fills", "donch_exits", "gate_events", "avg_inv", "secs"]


def _emit(path, done, label, row):
    if label in done:
        return
    out = {k: row.get(k, "") for k in FIELDS}
    out["label"] = label
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(out)
    print(f"{label:42} waCAGR={out['wa_cagr']:>7} waDD={out['wa_dd']:>7} "
          f"waCal={out['wa_calmar']:>5} w1={out['w1_cagr']:>7} w2={out['w2_cagr']:>7} "
          f"[{out['secs']}s]", flush=True)


def phase_sweep(ctx):
    path = RES / "universe_sweep.csv"
    done = set()
    if path.exists():
        with open(path) as f:
            done = {r["label"] for r in csv.DictReader(f)}
    else:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    py = {}
    for mode in MODES:
        ctx.set_mode(mode)
        for off in range(12):
            for txm in (0, 1):
                lbl = f"U_{mode}_off{off}_tax{txm}"
                if lbl in done:
                    continue
                rec = [] if (off == 0 and txm == 1) else None
                r = tn.run(ctx, offset=off, tax=bool(txm), record=rec)
                r.update(mode=mode)
                if rec is not None:
                    nav = r["_nav"]
                    nav.to_csv(RES / f"nav_{mode}_tax1.csv")
                    pickle.dump(rec, open(RES / f"holdings_{mode}.pkl", "wb"))
                    last = nav.groupby(nav.index.year).last()
                    yr = last.pct_change() * 100
                    yr.iloc[0] = (last.iloc[0] / nav.iloc[0] - 1) * 100
                    py[mode] = yr.round(1)
                _emit(path, done, lbl, r)
        for rt in (0.005, 0.0075):
            for txm in (0, 1):
                lbl = f"U_{mode}_rt{int(rt*10000)}_tax{txm}"
                if lbl not in done:
                    r = tn.run(ctx, tax=bool(txm), rt=rt); r.update(mode=mode)
                    _emit(path, done, lbl, r)
        lbl = f"U_{mode}_cash5_tax1"
        if lbl not in done:
            r = tn.run(ctx, tax=True, cash_y=0.05); r.update(mode=mode)
            _emit(path, done, lbl, r)
        ctx.save_ranks()
    if py:
        pd.DataFrame(py).to_csv(RES / "peryear.csv")


def phase_capacity(ctx):
    """Held-name 20d-median traded value distributions + universe sizes. No backtests."""
    rows = []
    med20 = ctx.tv.rolling(20, min_periods=10).median()
    for mode in MODES:
        f = RES / f"holdings_{mode}.pkl"
        if not f.exists():
            print(f"capacity: missing holdings for {mode}", flush=True); continue
        rec = pickle.load(open(f, "rb"))
        vals_p10, vals_min, vals_p50 = [], [], []
        for i, held in rec[::5]:                       # weekly-ish sampling
            if not held:
                continue
            tvs = [med20.iat[i, ctx.sidx[s]] for s in held]
            tvs = [t for t in tvs if t == t]
            if tvs:
                vals_p50.append(np.median(tvs))
                vals_p10.append(np.percentile(tvs, 10))
                vals_min.append(min(tvs))
        p10_typ = float(np.median(vals_p10)); p10_worst = float(np.percentile(vals_p10, 5))
        rows.append(dict(
            mode=mode,
            held_tv_p50_cr=round(float(np.median(vals_p50)) / 1e7, 1),
            held_tv_p10_typ_cr=round(p10_typ / 1e7, 1),
            held_tv_p10_worst5pct_cr=round(p10_worst / 1e7, 1),
            held_tv_min_ever_cr=round(float(min(vals_min)) / 1e7, 2),
            # max book so one 12.5% slot = 10% of the p10-name's daily traded value
            max_book_10pct_participation_cr=round(0.10 * p10_typ * 8 / 1e7, 1),
            max_book_worstcase_cr=round(0.10 * p10_worst * 8 / 1e7, 1),
        ))
        print(rows[-1], flush=True)
    # universe sizes over time (tvfloor + top500 recorded during ranking)
    for mode in MODES[1:]:
        ctx.set_mode(mode)
        sizes = {}
        for y in (2006, 2009, 2012, 2016, 2020, 2023, 2026):
            i = int(np.searchsorted(ctx.dates.values, np.datetime64(f"{y}-06-01")))
            i = min(i, len(ctx.dates) - 1)
            ctx.rank_cache.pop(i, None)                 # force fresh to capture size
            ctx.ranking(i)
            sizes[y] = ctx.uni_sizes.get(i, "")
        print(f"universe size {mode}: {sizes}", flush=True)
        rows.append(dict(mode=f"{mode}_unisize", held_tv_p50_cr=str(sizes)))
    pd.DataFrame(rows).to_csv(RES / "capacity.csv", index=False)


def phase_blend(ctx):
    print("loading OA frames (trail_sma=15) ...", flush=True)
    w = br.load_frames("2004-06-01", trail_sma=15)
    close, high, open_, athcp, sma, tv20 = (w[k] for k in
        ("close", "high", "open", "athcp", "sma50", "tv20"))
    etf = [c for c in close.columns if br.ETF_RE.search(c)]
    tv_prev = tv20.shift(1)
    prev_close = close.shift(1)
    elig = tv_prev >= br.TV_FLOOR
    elig[etf] = False
    score = 2 * (close / close.shift(63) - 1) + (close / close.shift(126) - 1) \
        + (close / close.shift(189) - 1) + (close / close.shift(252) - 1)
    rs = (score.where(elig).rank(axis=1, pct=True) * 100).shift(1)
    setup = (prev_close < athcp) & (prev_close >= 0.8 * athcp) & elig & (rs >= 70.0)
    trig = (setup & (close > athcp) & athcp.notna()).fillna(False).values
    odates = close.index
    C, H, O, ATH, S = close.values, high.values, open_.values, athcp.values, sma.values
    RSv, TVv = rs.values, tv_prev.values
    days = np.array([i for i, d in enumerate(odates) if str(d.date()) >= "2006-01-01"])
    weak = np.zeros(len(odates), dtype=bool)
    oa_navs, oa_holdsets = [], []
    for seed in range(1, 11):
        eq, trades, _ = br.simulate(seed, "random", days, odates, C, H, O, ATH, S, RSv,
                                    TVv, trig, weak, True, 0.0025, stop=0.08, slots=16,
                                    size_pct=0.0625, stcg=0.20, ltcg=0.125, cash_yield=0.05)
        oa_navs.append(pd.Series(np.asarray(eq, float), index=odates[days]))
        if seed <= 3:                                   # holdings for overlap
            hs = {}                                     # date -> set(sym)
            for (c, ei, xi, *_rest) in trades:
                sym = close.columns[c]
                for i in range(ei, xi + 1):
                    hs.setdefault(i, set()).add(sym)
            oa_holdsets.append(hs)
    print("OA 10 seeds done", flush=True)

    tn_navs = {"U200": pd.read_csv(
        Path("/home/arun/quantifyd/research/144_truenorth_reassessment/results/"
             "nav_INC_cash_n8_d15_tax1.csv"), index_col=0, parse_dates=True).iloc[:, 0],
        "UOA": pd.read_csv(RES / "nav_tvfloor_tax1.csv", index_col=0,
                           parse_dates=True).iloc[:, 0]}

    def stats(nav):
        yrs = (nav.index[-1] - nav.index[0]).days / 365.25
        cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
        dd = float((nav / nav.cummax() - 1).min())
        return cagr * 100, dd * 100, (cagr / abs(dd) if dd < 0 else np.nan)

    rows = []
    for tag, mnav in tn_navs.items():
        cl, dl, kl, cd, cm = [], [], [], [], []
        for onav in oa_navs:
            idx = onav.index.intersection(mnav.index)
            o, m = onav.loc[idx], mnav.loc[idx]
            b_m = o.resample("ME").last().pct_change().fillna(0)
            m_m = m.resample("ME").last().pct_change().fillna(0)
            blend = (1 + 0.5 * b_m + 0.5 * m_m).cumprod()
            c, d, k = stats(blend)
            cl.append(c); dl.append(d); kl.append(k)
            cd.append(o.pct_change().corr(m.pct_change())); cm.append(b_m.corr(m_m))
        rows.append(dict(tn_leg=tag,
                         blend_cagr_med=round(float(np.median(cl)), 2),
                         blend_cagr_min=round(min(cl), 2), blend_cagr_max=round(max(cl), 2),
                         blend_dd_med=round(float(np.median(dl)), 2),
                         blend_dd_worst=round(min(dl), 2),
                         blend_calmar_med=round(float(np.median(kl)), 2),
                         corr_daily_med=round(float(np.median(cd)), 3),
                         corr_monthly_med=round(float(np.median(cm)), 3)))
        print(rows[-1], flush=True)
    # TN-full vs TN-std correlation
    idx = tn_navs["U200"].index.intersection(tn_navs["UOA"].index)
    a, b = tn_navs["U200"].loc[idx], tn_navs["UOA"].loc[idx]
    print("corr U200 vs UOA: daily "
          f"{a.pct_change().corr(b.pct_change()):.3f}, monthly "
          f"{a.resample('ME').last().pct_change().corr(b.resample('ME').last().pct_change()):.3f}",
          flush=True)
    # holdings overlap: TN(UOA) names also held by OA (same calendar via date match)
    hold = pickle.load(open(RES / "holdings_tvfloor.pkl", "rb"))
    od_index = {d: i for i, d in enumerate(odates)}
    ovl = []
    for i, held in hold[::5]:
        if not held:
            continue
        d = ctx.dates[i]
        oi = od_index.get(d)
        if oi is None:
            continue
        fr = [len(set(held) & hs.get(oi, set())) / len(held) for hs in oa_holdsets]
        ovl.append(np.mean(fr))
    print(f"TN(U-OA) holdings overlap with OA positions (mean of 3 seeds): "
          f"median {np.median(ovl)*100:.1f}%  p90 {np.percentile(ovl, 90)*100:.1f}%", flush=True)
    rows.append(dict(tn_leg="overlap_UOA_vs_OA",
                     blend_cagr_med=round(float(np.median(ovl)) * 100, 1),
                     blend_dd_med=round(float(np.percentile(ovl, 90)) * 100, 1)))
    pd.DataFrame(rows).to_csv(RES / "blend_universe.csv", index=False)


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    ctx = UCtx()
    ctx.uni_mode = "top200"
    ctx.uni_sizes = {}
    if phase == "sweep":
        phase_sweep(ctx)
    elif phase == "capacity":
        phase_capacity(ctx)
    elif phase == "blend":
        phase_blend(ctx)
    print("PHASE", phase, "DONE", flush=True)


if __name__ == "__main__":
    main()
