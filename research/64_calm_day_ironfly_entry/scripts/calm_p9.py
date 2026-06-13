"""research/64 P9 — DAILY-CLOSE breach (our calm study) vs INTRADAY stop (the live 1-min engine).
Proxy for an intraday breach = the day's HIGH/LOW touched ±2% from entry (we lack NIFTY 1-min/5-min in
the DB). Classify each 5-day hold:
  clean   : never touched ±2%            -> both keep (winner)
  whipsaw : touched ±2% intraday BUT no daily CLOSE ever breached -> intraday stop EXITS (locks ~-2%),
            daily-close rule KEEPS it (finishes calm) => the COST of the intraday stop
  continued: touched ±2% AND a daily CLOSE breached -> both exit; intraday caps at -2%, daily rides to
            the (worse) close => the SAVING of the intraday stop
Fly P&L proxy: stop@2% ≈ -₹34k (verified); daily-close exit ≈ -₹34k.. capped -₹93k (wings) as the move
runs; a calm finish ≈ +credit win. Cached NIFTY daily."""
import numpy as np, pandas as pd
n, vix = pd.read_pickle("/tmp/nifty_vix_cache.pkl")
H = n.high.values; L = n.low.values; C = n.close.values
rma = lambda x, p: x.ewm(alpha=1/p, adjust=False).mean()
prevC = pd.Series(C).shift(1).values
atr = rma(pd.Series(np.maximum.reduce([H-L, np.abs(H-prevC), np.abs(L-prevC)])), 14).values
piv = (H+L+C)/3; bc = (H+L)/2; tc = 2*piv-bc
cprw = np.abs(tc-bc)/C*100
lo14 = pd.Series(L).rolling(14).min().values; hi14 = pd.Series(H).rolling(14).max().values
st = 100*(C-lo14)/(hi14-lo14)
vx = vix.reindex(n.index, method="ffill").values
HH = 5; rows = []
for i in range(20, len(C)-HH):
    es = C[i-1]; up = es*1.02; dn = es*0.98
    touch_d = None; close_d = None; close_move = None
    for d in range(HH):
        j = i+d
        if touch_d is None and (H[j] >= up or L[j] <= dn):
            touch_d = d
        if close_d is None and abs(C[j]-es)/es >= 0.02:
            close_d = d; close_move = abs(C[j]-es)/es*100
            break
    gated = (((atr[i-1]/C[i-1]*100 < 1.1) + (cprw[i-1] < 0.16) + (st[i-1] > 65)) >= 2) and (13 <= vx[i-1] <= 22)
    rows.append([touch_d, close_d, close_move, gated])
d = pd.DataFrame(rows, columns=["touch", "close", "cmove", "gated"])
N = len(d)

def report(df, tag):
    n = len(df)
    clean = df.touch.isna().sum()
    touched = df[df.touch.notna()]
    whip = touched[touched.close.isna()]
    cont = touched[touched.close.notna()]
    print(f"\n=== {tag}: N={n} ===")
    print(f"  clean (never touch ±2%) : {clean} ({clean/n*100:.0f}%)")
    print(f"  touched ±2% intraday    : {len(touched)} ({len(touched)/n*100:.0f}%)")
    print(f"    -> WHIPSAW (close stayed inside, daily-rule KEEPS, intraday-stop EXITS): {len(whip)} ({len(whip)/n*100:.1f}%)")
    print(f"    -> CONTINUED (daily close also breached, both exit)                   : {len(cont)} ({len(cont)/n*100:.1f}%)")
    if len(touched):
        print(f"  of intraday touches: {len(whip)/len(touched)*100:.0f}% were WHIPSAWS (reverted by close), {len(cont)/len(touched)*100:.0f}% CONTINUED")
    if len(cont):
        print(f"  CONTINUED daily-close over-run beyond 2%: median {cont.cmove.median()-2:.2f}pp, mean {cont.cmove.mean()-2:.2f}pp (how much worse daily rides vs the -2% intraday cap)")
        print(f"    touch precedes the close-breach day in {(cont.touch < cont.close).mean()*100:.0f}% of continued trades (intraday exits earlier)")
    # crude fly-P&L net (proxy): saving on continued ≈ (cmove-2)/2 * (93k-34k) capped at 93k-34k; cost on whipsaw ≈ 34k stop + a ~40k forgone win
    if len(cont) and len(whip):
        save_each = np.clip((cont.cmove.values-2.0)/2.0, 0, 1)*(93000-34000)
        saving = save_each.sum()
        cost = len(whip)*(34000+40000)   # -34k stop taken + ~40k median fly win forgone
        print(f"  PROXY net: saving≈+₹{saving:,.0f} (cap trend/gap losses) vs cost≈−₹{cost:,.0f} ({len(whip)} whipsaw stops) → NET ₹{saving-cost:,.0f}")

report(d, "ALL entries")
report(d[d.gated], "GATED (compression + VIX 13–22)")
