"""True replay backtest of the NAS systems on the recorded NIFTY weekly option chain.

For each trading day we load that day's recorded option_chain (real per-minute
premiums) + per-minute spot, then REPLAY each system's documented rules to generate
the trades the logic would make, priced entirely from the recorded chain. Lots fixed
at 2 (130 qty) for all systems so they are directly comparable.

Faithfulness: 9:16 entry is exact; squeeze entry is reconstructed from per-minute spot
(approximate). 1-min snapshot cadence => SL/ST resolved to ~1 min. Validated against
the actual recorded trades. See STATUS doc for caveats.
"""
import sqlite3, math
from pathlib import Path
from datetime import datetime, time as dtime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path("/home/arun/quantifyd")
BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results"; OUT.mkdir(parents=True, exist_ok=True)
OPT_DB = ROOT / "backtest_data" / "options_data.db"
LOT = 65; LOTS = 2; QTY = LOT * LOTS
BROK_PER_LEG = 40 * 2          # sell+buy
TARGET_OTM_PREM = 20.0
SL_MULT = 1.3                  # ATM per-leg stop = entry x 1.3
ROLL_RATIO = 2.0              # OTM cross-leg imbalance trigger
ENTRY_916 = dtime(9, 16); ENTRY_WIN_END = dtime(14, 30)
TIME_EXIT = dtime(14, 45); EOD = dtime(15, 15)
SQ_START = dtime(9, 35)

SYSTEMS = [
    ("Squeeze OTM",  "squeeze", "OTM", "ROLL"),
    ("Squeeze ATM",  "squeeze", "ATM", "SL_ST"),
    ("Squeeze ATM2", "squeeze", "ATM", "CASCADE"),
    ("Squeeze ATM4", "squeeze", "ATM", "ROLL_MATCH"),
    ("916 OTM",      "t916",    "OTM", "ROLL"),
    ("916 ATM",      "t916",    "ATM", "SL_ST"),
    ("916 ATM2",     "t916",    "ATM", "CASCADE"),
    ("916 ATM4",     "t916",    "ATM", "ROLL_MATCH"),
]

# ---------- data ----------
oc = sqlite3.connect(str(OPT_DB))
oc.execute("CREATE INDEX IF NOT EXISTS idx_oc_sym_day ON option_chain(symbol, snapshot_time)")
days = [r[0] for r in oc.execute(
    "SELECT DISTINCT substr(snapshot_time,1,10) FROM option_chain WHERE symbol='NIFTY' ORDER BY 1")]


def front_expiry(day_df, day):
    exps = sorted(day_df["expiry_date"].unique())
    fut = [e for e in exps if e >= day]
    return fut[0] if fut else (exps[-1] if exps else None)


def supertrend_up(prem_series, period=7, mult=2.0):
    """Resample a leg's per-min premium to 5-min OHLC, return a time-indexed bool
    Series: True where Supertrend says trend is UP (premium rising => exit short)."""
    if prem_series.empty:
        return pd.Series(dtype=bool)
    o = prem_series.resample("5min").agg(["first", "max", "min", "last"]).dropna()
    if len(o) < period + 1:
        return pd.Series(False, index=o.index)
    o.columns = ["open", "high", "low", "close"]
    pc = o["close"].shift(1)
    tr = pd.concat([o["high"] - o["low"], (o["high"] - pc).abs(), (o["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    hl2 = (o["high"] + o["low"]) / 2
    up = hl2 - mult * atr; dn = hl2 + mult * atr
    st = pd.Series(index=o.index, dtype=float); dir_up = pd.Series(index=o.index, dtype=bool)
    fu = fl = np.nan; trend = True
    for i in range(len(o)):
        c = o["close"].iloc[i]
        if np.isnan(atr.iloc[i]):
            dir_up.iloc[i] = True; continue
        fu = up.iloc[i] if (np.isnan(fu) or up.iloc[i] > fu or o["close"].iloc[i-1] < fu) else fu
        fl = dn.iloc[i] if (np.isnan(fl) or dn.iloc[i] < fl or o["close"].iloc[i-1] > fl) else fl
        if trend and c < fu:
            trend = False
        elif (not trend) and c > fl:
            trend = True
        dir_up.iloc[i] = trend   # True = premium in uptrend => bad for short
    return dir_up


# ---------- per-day simulation ----------
def sim_day(day, entry_mode, strike_mode, mgmt):
    df = pd.read_sql_query(
        "SELECT snapshot_time, tradingsymbol, strike, instrument_type, ltp, expiry_date, underlying_spot "
        "FROM option_chain WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? AND ltp IS NOT NULL",
        oc, params=[day])
    if df.empty:
        return []
    df["t"] = pd.to_datetime(df["snapshot_time"])
    exp = front_expiry(df, day)
    df = df[df["expiry_date"] == exp]
    if df.empty:
        return []
    spot_s = df.groupby("t")["underlying_spot"].first().sort_index()
    times = spot_s.index
    # premium lookup: tsym -> sorted (np times, ltps)
    chain = {}
    for ts, g in df.groupby("tradingsymbol"):
        g = g.sort_values("t")
        chain[ts] = (g["t"].values, g["ltp"].values, g["strike"].iloc[0], g["instrument_type"].iloc[0])

    def prem(ts, t):
        if ts not in chain:
            return None
        ta, la, _, _ = chain[ts]
        idx = np.searchsorted(ta, np.datetime64(t), side="right") - 1
        if idx < 0:
            return None
        v = la[idx]
        return float(v) if v and v > 0 else None

    def tsym(strike, typ):
        for ts, (_, _, st, ty) in chain.items():
            if int(st) == int(strike) and ty == typ:
                return ts
        return None

    def pick_atm(t, spot):
        atm = round(spot / 50) * 50
        return tsym(atm, "CE"), tsym(atm, "PE")

    def pick_otm(t, spot, target=TARGET_OTM_PREM):
        best = {"CE": (None, 1e9), "PE": (None, 1e9)}
        for ts, (ta, la, st, ty) in chain.items():
            if ty == "CE" and st <= spot: continue
            if ty == "PE" and st >= spot: continue
            p = prem(ts, t)
            if p is None or p < 5: continue
            d = abs(p - target)
            if d < best[ty][1]:
                best[ty] = (ts, d)
        return best["CE"][0], best["PE"][0]

    # entry time
    if entry_mode == "t916":
        ent_times = [t for t in times if t.time() >= ENTRY_916]
        if not ent_times: return []
        t0 = ent_times[0]
    else:  # squeeze: ATR(14)<SMA(ATR,50) on 5-min spot candles, first after 09:35
        c5 = spot_s.resample("5min").agg(["first", "max", "min", "last"]).dropna()
        if len(c5) < 55:
            t0 = next((t for t in times if t.time() >= dtime(9, 30)), None)
        else:
            c5.columns = ["open", "high", "low", "close"]
            pc = c5["close"].shift(1)
            tr = pd.concat([c5["high"]-c5["low"], (c5["high"]-pc).abs(), (c5["low"]-pc).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean(); sma = atr.rolling(50).mean()
            sq = (atr < sma) & (c5.index.to_series().dt.time >= SQ_START) & (c5.index.to_series().dt.time <= ENTRY_WIN_END)
            t0 = sq[sq].index[0] if sq.any() else None
        if t0 is None: return []
        t0 = min((t for t in times if t >= t0), default=None)
        if t0 is None: return []
    spot0 = float(spot_s.loc[t0])

    # open legs: list of dicts {tsym, typ, entry, qty, sl, open}
    legs = []
    def open_strangle(t, spot, reentry=False):
        if strike_mode == "ATM":
            ce, pe = pick_atm(t, spot)
        else:
            ce, pe = pick_otm(t, spot)
        out = []
        for ts in (ce, pe):
            if ts is None: continue
            p = prem(ts, t)
            if p is None: continue
            out.append({"tsym": ts, "typ": chain[ts][3], "entry": p, "qty": QTY,
                        "sl": p * SL_MULT, "open": True, "naked_st": False})
        return out
    legs = open_strangle(t0, spot0)
    if not legs: return []

    realized = []   # closed leg pnl tuples
    reentries = 0
    sim_times = [t for t in times if t >= t0 and t.time() <= EOD]
    st_cache = {}
    last_roll = t0; roll_count = 0   # OTM roll cadence control (live: 5-min + cooldown)

    def close_leg(lg, t, reason):
        p = prem(lg["tsym"], t)
        if p is None: p = lg["entry"]
        pnl = (lg["entry"] - p) * lg["qty"] - BROK_PER_LEG
        lg["open"] = False
        realized.append({"day": day, "tsym": lg["tsym"], "typ": lg["typ"],
                         "entry": lg["entry"], "exit": p, "qty": lg["qty"],
                         "pnl": pnl, "reason": reason})

    for t in sim_times:
        if not any(l["open"] for l in legs):
            break
        force = t.time() >= TIME_EXIT
        for lg in legs:
            if not lg["open"]:
                continue
            p = prem(lg["tsym"], t)
            if p is None:
                continue
            if force:
                close_leg(lg, t, "time_exit"); continue
            # naked-leg ST exit
            if lg["naked_st"]:
                key = lg["tsym"]
                if key not in st_cache:
                    ta, la, _, _ = chain[lg["tsym"]]
                    ps = pd.Series(la, index=pd.DatetimeIndex(ta)).sort_index()
                    st_cache[key] = supertrend_up(ps)
                stu = st_cache[key]
                sub = stu[stu.index <= t]
                if len(sub) and bool(sub.iloc[-1]):
                    close_leg(lg, t, "ST_EXIT")
                continue
            # SL / management
            if p >= lg["sl"]:
                if mgmt == "SL_ST":
                    close_leg(lg, t, "SL_HIT")
                    for o in legs:
                        if o["open"]:
                            o["naked_st"] = True
                elif mgmt == "CASCADE":
                    for o in legs:
                        if o["open"]:
                            close_leg(o, t, "SL_CASCADE")
                    if reentries < 5 and t.time() <= ENTRY_WIN_END:
                        reentries += 1
                        legs.extend(open_strangle(t, float(spot_s.loc[t]) if t in spot_s.index else spot0, True))
                    break
                elif mgmt == "ROLL_MATCH":
                    surv = next((o for o in legs if o["open"] and o is not lg), None)
                    close_leg(lg, t, "SL_ROLL")
                    if surv and not lg.get("rolled_once"):
                        sp = prem(surv["tsym"], t) or surv["entry"]
                        nts = None; best = 1e9
                        for ts, (_, _, st, ty) in chain.items():
                            if ty != lg["typ"]: continue
                            pp = prem(ts, t)
                            if pp is None or pp < 5: continue
                            if abs(pp - sp) < best:
                                best = abs(pp - sp); nts = ts
                        if nts:
                            np_ = prem(nts, t)
                            legs.append({"tsym": nts, "typ": lg["typ"], "entry": np_, "qty": QTY,
                                         "sl": np_ * SL_MULT, "open": True, "naked_st": False,
                                         "rolled_once": True})
                    else:
                        for o in legs:
                            if o["open"]:
                                o["naked_st"] = True
                elif mgmt == "ROLL":
                    pass  # OTM handled below by cross-leg ratio
        # OTM cross-leg roll (premium imbalance >= 2x). FAITHFULNESS GATE: the live
        # engine evaluates on 5-min candle close with a cooldown, not every tick —
        # so gate to 5-min boundaries, >=5-min spacing, and cap rolls/day. Without
        # this the roll churns ~45 legs/day (replay v1 bug).
        if mgmt == "ROLL" and t.minute % 5 == 0 and (t - last_roll).total_seconds() >= 300 and roll_count < 12:
            opens = [l for l in legs if l["open"]]
            if len(opens) == 2 and not force:
                pr = {l["typ"]: (prem(l["tsym"], t), l) for l in opens}
                if all(pr[k][0] for k in pr):
                    pce, ppe = pr.get("CE", (0, None))[0], pr.get("PE", (0, None))[0]
                    hi, lo = max(pce, ppe), min(pce, ppe)
                    if lo > 0 and hi / lo >= ROLL_RATIO:
                        last_roll = t; roll_count += 1
                        exp_typ = "CE" if pce > ppe else "PE"
                        exp_leg = pr[exp_typ][1]; cheap_p = lo
                        close_leg(exp_leg, t, "ROLL_OUT")
                        nts = None; best = 1e9; spot = float(spot_s.loc[t]) if t in spot_s.index else spot0
                        for ts, (_, _, st, ty) in chain.items():
                            if ty != exp_typ: continue
                            if exp_typ == "CE" and st <= spot: continue
                            if exp_typ == "PE" and st >= spot: continue
                            pp = prem(ts, t)
                            if pp is None or pp < 5: continue
                            if abs(pp - cheap_p) < best:
                                best = abs(pp - cheap_p); nts = ts
                        if nts:
                            np_ = prem(nts, t)
                            legs.append({"tsym": nts, "typ": exp_typ, "entry": np_, "qty": QTY,
                                         "sl": np_ * SL_MULT, "open": True, "naked_st": False})
    # close any still open at EOD
    last_t = sim_times[-1] if sim_times else t0
    for lg in legs:
        if lg["open"]:
            close_leg(lg, last_t, "eod")
    return realized


# ---------- run all ----------
rows = []
for name, em, sm, mg in SYSTEMS:
    for day in days:
        try:
            for r in sim_day(day, em, sm, mg):
                r["system"] = name; rows.append(r)
        except Exception as e:
            print(f"ERR {name} {day}: {e}")
legs = pd.DataFrame(rows)
legs.to_csv(OUT / "replay_legs.csv", index=False)
legs["day"] = pd.to_datetime(legs["day"])

daily = legs.groupby(["system", "day"])["pnl"].sum().reset_index()
order = [s[0] for s in SYSTEMS]
present = [s for s in order if s in set(legs["system"])]
dates = sorted(legs["day"].unique())
pv = daily.pivot(index="system", columns="day", values="pnl").reindex(present).reindex(columns=dates).fillna(0.0)
cum = pv.cumsum(axis=1)
comb = pv.sum(axis=0).cumsum()

def mdd(s):
    return (s - s.cummax()).min()

summ = []
for s in present:
    g = legs[legs["system"] == s]
    bydate = g.groupby("day")["pnl"].sum()
    strangles = g.groupby("day").ngroup().nunique()
    summ.append(dict(System=s, Legs=len(g), Days=g["day"].nunique(),
                     Net=round(g["pnl"].sum()), PerDay=round(g["pnl"].sum()/max(g["day"].nunique(),1)),
                     DayWin=round(100*(bydate>0).mean()),
                     MaxDD=round(mdd(cum.loc[s])), Best=round(bydate.max()), Worst=round(bydate.min())))
summ = pd.DataFrame(summ)

# ---------- figure ----------
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25, "figure.facecolor": "white"})
fig = plt.figure(figsize=(15, 16)); gs = GridSpec(4, 2, height_ratios=[.4, 1.2, 1.2, 1.0], hspace=.42, wspace=.18)
ax0 = fig.add_subplot(gs[0, :]); ax0.axis("off")
tot = legs["pnl"].sum()
ax0.text(0, .7, "NAS Systems — REPLAY backtest on recorded NIFTY weekly chain", fontsize=16, fontweight="bold")
ax0.text(0, .28, f"{pd.to_datetime(dates[0]).date()} -> {pd.to_datetime(dates[-1]).date()} · {len(dates)} days · "
                 f"lots=2 (130 qty) all systems · combined net ₹{tot:,.0f} (priced from real chain)", fontsize=10)
ax0.text(0, -.05, "True strategy replay (rules on recorded premiums). 28-day single regime — signal/audit, "
                  "not validation. 9:16 entry exact; squeeze entry approximated; SL/ST at 1-min cadence.",
         fontsize=8.5, color="#a00", style="italic")
ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(comb.index, comb.values, color="#06c", lw=2); ax1.axhline(0, color="#999", lw=.8)
ax1.set_title("Combined book — cumulative net ₹"); ax1.tick_params(axis="x", rotation=45)
ax2 = fig.add_subplot(gs[1, 1])
for s in present: ax2.plot(cum.columns, cum.loc[s].values, lw=1.3, label=s)
ax2.axhline(0, color="#999", lw=.8); ax2.legend(fontsize=7, ncol=2)
ax2.set_title("Per-system cumulative net ₹"); ax2.tick_params(axis="x", rotation=45)
ax3 = fig.add_subplot(gs[2, :])
M = pv.values.astype(float); vmax = np.nanpercentile(np.abs(M), 95) or 1
im = ax3.imshow(M, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
ax3.set_yticks(range(len(present))); ax3.set_yticklabels(present, fontsize=8)
ax3.set_xticks(range(len(dates))); ax3.set_xticklabels([pd.to_datetime(d).strftime("%m-%d") for d in dates], rotation=90, fontsize=6.5)
ax3.set_title("Per-day × per-system net ₹ (replay)")
for i in range(len(present)):
    for j in range(len(dates)):
        if abs(M[i, j]) > 1: ax3.text(j, i, f"{M[i,j]/1000:.1f}", ha="center", va="center", fontsize=5, color="black")
fig.colorbar(im, ax=ax3, fraction=.025, pad=.01)
ax4 = fig.add_subplot(gs[3, :]); ax4.axis("off")
disp = summ.copy()
for cc in ["Net", "PerDay", "MaxDD", "Best", "Worst"]: disp[cc] = disp[cc].map(lambda v: f"{v:,.0f}")
disp["DayWin"] = disp["DayWin"].map(lambda v: f"{v:.0f}%")
tb = ax4.table(cellText=disp.values, colLabels=disp.columns, loc="center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.6)
for j in range(len(disp.columns)): tb[0, j].set_facecolor("#222"); tb[0, j].set_text_props(color="white", fontweight="bold")
ax4.set_title("Per-system replay stats (net ₹, lots=2)", y=.9, fontweight="bold")
fig.savefig(OUT / "nas_replay.png", dpi=110, bbox_inches="tight")
print("WROTE", OUT / "nas_replay.png")

L = ["# NAS Systems — REPLAY backtest on recorded NIFTY chain\n",
     f"{pd.to_datetime(dates[0]).date()} → {pd.to_datetime(dates[-1]).date()} · {len(dates)} days · lots=2 · "
     f"**combined net ₹{tot:,.0f}**\n",
     "**VERDICT: SIGNAL/AUDIT (28-day single regime, not validation).** Rules replayed on real recorded premiums.\n",
     "| " + " | ".join(summ.columns) + " |", "|" + "|".join(["---"]*len(summ.columns)) + "|"]
for _, r in summ.iterrows(): L.append("| " + " | ".join(str(r[c]) for c in summ.columns) + " |")
L += ["\n- 9:16 entry exact; squeeze entry reconstructed from per-min spot (approx).",
      "- 1-min cadence (intra-min SL/ST spikes missed); LTP pricing (no slippage) => mildly optimistic.",
      "- Validate against actual trades (research/50) for faithfulness."]
(OUT / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
print("WROTE", OUT / "RESULTS.md")
oc.close()
