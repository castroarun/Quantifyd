"""research/64 P7 — day-3 adjustment: data-only part. When flagged at day-3 (near-band drift >1.4%, or
low-drift+wide-chop), does the move CONTINUE (→ defend/convert directional) or REVERT (→ hold)? The ₹ of
each adjustment needs premiums (spec below). Cached NIFTY daily."""
import numpy as np, pandas as pd
n, vix = pd.read_pickle("/tmp/nifty_vix_cache.pkl")
H = n.high.values; L = n.low.values; C = n.close.values; clv = C
rows = []
for i in range(20, len(clv)-5):
    es = clv[i-1]; m = [(clv[i+dd]-es)/es*100 for dd in range(5)]
    if max(abs(m[0]), abs(m[1]), abs(m[2])) >= 2: continue          # calm through day-3
    drift3 = m[2]; hr3 = (max(H[i],H[i+1],H[i+2])-min(L[i],L[i+1],L[i+2]))/es*100
    # outcome over days 4-5 (relative to entry)
    fut = [m[3], m[4]]; maxup = max(fut); maxdn = min(fut)
    breach_up = max(m[3], m[4]) >= 2 or maxup >= 2
    breach_dn = min(m[3], m[4]) <= -2
    rows.append([drift3, hr3, breach_up, breach_dn, m[4]])
d = pd.DataFrame(rows, columns=["drift3", "hr3", "bup", "bdn", "end"])

def split(df, tag):
    n = len(df)
    if not n: print(f"{tag}: n=0"); return
    # direction of the day-3 drift
    up = df[df.drift3 > 0]; dn = df[df.drift3 < 0]
    def o(s, dirn):
        if not len(s): return
        same = (s.bup if dirn > 0 else s.bdn).mean()       # breach in the drift direction (continue)
        opp = (s.bdn if dirn > 0 else s.bup).mean()        # breach opposite (reverse-and-break)
        calm = (~(s.bup | s.bdn)).mean()
        print(f"   {('UP' if dirn>0 else 'DOWN')}-drift (n={len(s)}): finish-calm {calm*100:.0f}% | continue-breach {same*100:.0f}% | reverse-breach {opp*100:.0f}%")
    print(f"\n{tag}: n={n}")
    o(up, 1); o(dn, -1)

print("=== FLAGGED at day-3: near-band drift 1.4-2.0% (the P6 'roll/close' zone) ===")
split(d[(d.drift3.abs() >= 1.4)], "near-band flagged")
print("\n=== FLAGGED: low-drift (<0.6%) BUT wide chop (range>1.5%) — the P8 hidden danger ===")
split(d[(d.drift3.abs() < 0.6) & (d.hr3 > 1.5)], "chop-flagged")
print("\n=== reference: NOT flagged (drift<0.6 and tight) ===")
split(d[(d.drift3.abs() < 0.6) & (d.hr3 <= 1.5)], "calm/tight")
print("""
READ: if 'continue-breach' >> 'reverse-breach' when flagged, the move tends to keep going in the drift
direction -> the right adjustment is DEFEND the threatened side / CONVERT to a directional skew away from
the band (the breach, if it comes, is same-side). If finish-calm is high, HOLDING is fine. The actual ₹
of defend vs convert vs close vs hold needs option premiums (see spec).""")
