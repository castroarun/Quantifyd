"""How many days/year: compression+VIX gate passes, realized ENTRIES (one-trade-at-a-time, 5-day hold),
and IN-TRADE days. Neutral fly. Cached NIFTY+VIX daily."""
import numpy as np, pandas as pd
n, vix = pd.read_pickle("/tmp/nifty_vix_cache.pkl")
H, L, C = n.high, n.low, n.close; prevC = C.shift(1)
rma = lambda x, p: x.ewm(alpha=1/p, adjust=False).mean()
tr = pd.concat([(H-L), (H-prevC).abs(), (L-prevC).abs()], axis=1).max(axis=1); atr14 = rma(tr, 14)
lo14, hi14 = L.rolling(14).min(), H.rolling(14).max(); piv = (H+L+C)/3; bc = (H+L)/2; tc = 2*piv-bc
f = pd.DataFrame(index=n.index)
f["atr"] = atr14/C*100; f["cpr"] = (tc-bc).abs()/C*100; f["st"] = 100*(C-lo14)/(hi14-lo14)
f["vix"] = vix.reindex(n.index, method="ffill")
f = f.shift(1)   # causal (prior close)
gate = (((f.atr < 1.1).astype(int) + (f.cpr < 0.16).astype(int) + (f.st > 65).astype(int)) >= 2) & (f.vix >= 13) & (f.vix <= 22)
gate = gate.fillna(False)
idx = list(n.index); HOLD = 5
entry = pd.Series(False, index=n.index); intrade = pd.Series(False, index=n.index)
i = 0
while i < len(idx) - HOLD:
    if gate.iloc[i]:
        entry.iloc[i] = True
        for j in range(i, min(i + HOLD, len(idx))):
            intrade.iloc[j] = True
        i += HOLD                # one-trade-at-a-time: locked for the hold
    else:
        i += 1
df = pd.DataFrame({"year": n.index.year, "gate": gate.values, "entry": entry.values, "intrade": intrade.values})
g = df.groupby("year").agg(sessions=("gate", "size"), gate_pass_days=("gate", "sum"),
                           entries=("entry", "sum"), intrade_days=("intrade", "sum"))
g["gate_pass_%"] = (g.gate_pass_days / g.sessions * 100).round(0)
print(g.to_string())
print(f"\nAVG/yr: gate-pass days {g.gate_pass_days.mean():.0f} ({g['gate_pass_%'].mean():.0f}% of sessions) | "
      f"entries {g.entries.mean():.0f} | in-trade days {g.intrade_days.mean():.0f} (~{g.intrade_days.mean()/250*100:.0f}% of the year)")
print("(one-trade-at-a-time, 5-day hold. 2026 is a partial year. Entries < gate-pass days because you")
print(" are locked for 5 days after each entry; in-trade days = entries x ~5.)")
