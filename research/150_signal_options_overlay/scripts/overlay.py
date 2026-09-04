"""research/150 — defined-risk options structures triggered by the high-WR killed cash
signals (Connors RSI2, KC6, pullback-50SMA), priced from real NSE bhavcopy (traded strikes
only), held to expiry. Window 2024-01+ (stock-option bhav density). Read-only."""
import sqlite3, math, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
ENG = Path("/home/arun/quantifyd/research/146_complementary_third_sleeve/scripts/sleeve_engine.py")
_s = importlib.util.spec_from_file_location("se", str(ENG))
se = importlib.util.module_from_spec(_s); _s.loader.exec_module(se)

START = "2024-01-01"
INDEXES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "SENSEX", "BANKEX"}


def build_signals(ctx):
    D = ctx.dates; C = ctx.C; O = ctx.O; Lo = ctx.Lo
    i0 = int(np.searchsorted(D.values, np.datetime64(START)))
    s200 = ctx.SMA200
    sigs = {"conn": [], "kc6": [], "pull": []}
    for i in range(i0, len(D)):
        up = C[i] > s200[i]
        m1 = (ctx.RSI2[i] < 10) & up
        lower = ctx.EMA6[i] - 1.3 * ctx.ATR6[i]
        m2 = (C[i] < lower) & up
        green = C[i] > O[i]; red1 = C[i - 1] < O[i - 1]
        touched = (Lo[i] <= ctx.SMA50[i]) | (Lo[i - 1] <= ctx.SMA50[i - 1]) | \
                  (Lo[i - 2] <= ctx.SMA50[i - 2])
        rising = ctx.SMA50[i] > ctx.SMA50[i - 10]
        m3 = green & red1 & touched & rising & up
        for name, m in (("conn", m1), ("kc6", m2), ("pull", m3)):
            for j in np.nonzero(m)[0]:
                sigs[name].append((i, j))
    return sigs, i0


def main():
    ctx = se.SCtx()
    con = sqlite3.connect(DB)
    syms = {r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM nse_options_bhav WHERE trade_date>=?", (START,))}
    fno = sorted((syms - INDEXES) & set(ctx.sidx))
    print(f"F&O stock underlyings usable: {len(fno)}", flush=True)
    fset = {ctx.sidx[s]: s for s in fno}
    sigs, i0 = build_signals(ctx)
    D = ctx.dates
    iso = D.isocalendar()
    wk = (iso.year * 100 + iso.week).to_numpy()

    bhav_cache = {}

    def bhav(sym):
        if sym not in bhav_cache:
            bhav_cache[sym] = pd.read_sql(
                "SELECT trade_date, expiry_date, strike, option_type, close, contracts "
                "FROM nse_options_bhav WHERE symbol=? AND trade_date>=? AND contracts>0",
                con, params=(sym, START))
        return bhav_cache[sym]

    def uclose_at(j, dstr):
        # underlying close on (or last before) dstr
        pos = int(np.searchsorted(D.values, np.datetime64(dstr), side="right")) - 1
        for k in range(pos, max(pos - 5, 0), -1):
            v = ctx.C[k, j]
            if v == v:
                return float(v), D[k]
        return None, None

    def pick(chain, typ, target):
        c = chain[chain.option_type == typ]
        if not len(c):
            return None
        c = c.assign(dist=(c.strike - target).abs()).sort_values("dist")
        r = c.iloc[0]
        return dict(strike=float(r.strike), px=float(r.close), vol=int(r.contracts))

    rows = []
    last_date = str(D[-1].date())
    for name, lst in sigs.items():
        seen = set()
        for i, j in lst:
            if j not in fset:
                continue
            key = (j, wk[i])
            if key in seen:
                continue
            seen.add(key)
            sym = fset[j]
            d = str(D[i].date())
            spot = float(ctx.C[i, j])
            b = bhav(sym)
            day = b[b.trade_date == d]
            if not len(day):
                continue
            exps = sorted(day.expiry_date.unique())
            exp = None
            for e in exps:
                dte = (pd.Timestamp(e) - D[i]).days
                if 20 <= dte <= 45:
                    exp = e; break
            if exp is None or exp > last_date:
                continue
            chain = day[day.expiry_date == exp]
            es, _ = uclose_at(j, exp)
            if es is None:
                continue
            for struct in ("S1", "S2", "S3"):
                if struct == "S1":
                    legs = [("PE", 0.97, -1), ("PE", 0.90, +1)]
                elif struct == "S2":
                    legs = [("PE", 1.00, -1), ("PE", 0.95, +1)]
                else:
                    legs = [("PE", 0.97, -1), ("PE", 0.90, +1),
                            ("CE", 1.07, -1), ("CE", 1.12, +1)]
                lg = []
                ok = True
                for typ, m, side in legs:
                    p = pick(chain, typ, m * spot)
                    if p is None:
                        ok = False; break
                    p["side"] = side; p["typ"] = typ
                    lg.append(p)
                if not ok:
                    continue
                strikes = [(x["typ"], x["strike"], x["side"]) for x in lg]
                if len({(t, k) for t, k, _ in strikes}) < len(strikes):
                    continue                     # degenerate: same strike picked twice
                credit = -sum(x["side"] * x["px"] for x in lg)   # short legs collect
                if credit <= 0:
                    continue
                pw = [x for x in lg if x["typ"] == "PE"]
                if len(pw) == 2 and pw[0]["strike"] <= pw[1]["strike"]:
                    continue                     # short put must be higher strike
                width_p = abs(pw[0]["strike"] - pw[1]["strike"]) if len(pw) == 2 else 0
                cw = [x for x in lg if x["typ"] == "CE"]
                width_c = abs(cw[0]["strike"] - cw[1]["strike"]) if len(cw) == 2 else 0
                maxw = max(width_p, width_c)
                if maxw <= 0 or credit >= maxw:
                    continue
                payout = 0.0
                for x in lg:
                    iv = max(0.0, (x["strike"] - es) if x["typ"] == "PE" else (es - x["strike"]))
                    payout += -x["side"] * iv    # short leg pays out intrinsic
                pnl = credit - payout
                rows.append(dict(signal=name, structure=struct, symbol=sym, date=d,
                                 expiry=exp, spot=round(spot, 2), exp_spot=round(es, 2),
                                 credit=round(credit, 2), max_risk=round(maxw - credit, 2),
                                 pnl=round(pnl, 2), min_vol=min(x["vol"] for x in lg),
                                 year=d[:4]))
    con.close()
    tr = pd.DataFrame(rows)
    tr.to_csv(RES / "overlay_trades.csv", index=False)
    print(f"structures priced: {len(tr)}", flush=True)
    out = []
    for (sig, st), g in tr.groupby(["signal", "structure"]):
        for hc in (0.0, 0.05, 0.10):
            cr = g.credit * (1 - hc)
            pnl = cr - (g.credit - g.pnl)        # payout unchanged
            ror = pnl / (g.max_risk + g.credit * hc)
            n = len(g)
            t = ror.mean() / (ror.std(ddof=1) / math.sqrt(n)) if n > 2 else np.nan
            row = dict(signal=sig, structure=st, haircut=hc, n=n,
                       mean_ror=round(ror.mean() * 100, 2), t=round(t, 2),
                       wr=round((pnl > 0).mean() * 100, 1),
                       med_min_vol=int(g.min_vol.median()))
            for y, gy in g.groupby("year"):
                cy = gy.credit * (1 - hc)
                py_ = cy - (gy.credit - gy.pnl)
                row[f"ror_{y}"] = round((py_ / (gy.max_risk + gy.credit * hc)).mean() * 100, 2)
            out.append(row)
            print(row, flush=True)
    pd.DataFrame(out).to_csv(RES / "overlay_summary.csv", index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
