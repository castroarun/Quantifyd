"""research/149 — pullback family done properly: MA {20,50,100,200} x {SMA,EMA} x
confluence {none, RSI14<40, RSI2<10, RS-pct>=70, Stoch<20 turning up, CCI<-100 turning up}
x exits {2R/10d, 2R/15d} = 96 cells, gross + after-tax. Runs on the r/146 sleeve engine
(candle mechanics identical; pre-registered resurrection bar in the STATUS doc)."""
import sys, csv, time, importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
ENG = Path("/home/arun/quantifyd/research/146_complementary_third_sleeve/scripts/sleeve_engine.py")
_s = importlib.util.spec_from_file_location("se", str(ENG))
se = importlib.util.module_from_spec(_s); _s.loader.exec_module(se)

FIELDS = ["label", "ma", "conf", "exit", "tax", "n_trades", "win_rate", "avg_win",
          "avg_loss", "expectancy", "max_lose_streak", "corr_tn_d", "secs"]
for w, _, _ in se.WINDOWS:
    FIELDS += [f"{w}_cagr", f"{w}_dd", f"{w}_sharpe", f"{w}_calmar"]


def main():
    ctx = se.SCtx()
    ctx.ensure_pullback_extras()
    path = RES / "pullback_grid.csv"
    done = set()
    if path.exists():
        with open(path) as f:
            done = {r["label"] for r in csv.DictReader(f)}
    else:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    confs = ["none", "rsi14", "rsi2", "rs70", "stoch", "cci"]
    exits = [("2R_t10", dict(rr=2.0, time=10)), ("2R_t15", dict(rr=2.0, time=15))]
    t0 = time.time()
    for mt in ("sma", "ema"):
        for L in (20, 50, 100, 200):
            for conf in confs:
                for exl, exp_ in exits:
                    for tax in (False, True):
                        lbl = f"{mt}{L}_{conf}_{exl}_tax{int(tax)}"
                        if lbl in done:
                            continue
                        p = dict(ma_len=L, ma_type=mt, conf=(None if conf == "none" else conf),
                                 **exp_)
                        r = se.run_sleeve(ctx, "pull", p, tax=tax)
                        row = {k: r.get(k, "") for k in FIELDS}
                        row.update(label=lbl, ma=f"{mt}{L}", conf=conf, exit=exl)
                        with open(path, "a", newline="") as f:
                            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
                        print(f"{lbl:28} n={row['n_trades']:>5} exp={row['expectancy']:>7} "
                              f"waCAGR={row['wa_cagr']:>7} w1={row['w1_cagr']:>7} "
                              f"w2={row['w2_cagr']:>7} [{row['secs']}s]", flush=True)
    print(f"GRID DONE 96x2 [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
