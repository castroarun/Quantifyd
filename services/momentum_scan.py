"""Momentum-30 scanner (research/62) — the ranking the live momentum_paper book uses,
exposed for the Holdings-page scanner UI. Ranks the official Nifty-200 by
    rsblend = 0.5 * RS(126d) + 0.5 * RS(252d)   where RS(L) = (px/px[-L]) / (NIFTYBEES/NIFTYBEES[-L])
Read-only over market_data.db (reuses momentum_paper._panel/_universe/_mcap_tier). No new data path.
"""
import pandas as pd  # noqa: F401


def scan():
    from services import momentum_paper as mp

    close, tv = mp._panel()
    if close is None or close.empty:
        return {"error": "no price panel", "rows": []}
    asof = close.index[-1]
    bench = mp.BENCH
    h = close.loc[:asof].ffill()
    if bench not in h.columns or len(h) <= 253:
        return {"error": "benchmark/history missing", "asof": str(asof.date()), "rows": []}

    uni = [s for s in mp._universe(close, tv, asof) if s in h.columns and s not in mp.EXCLUDE]
    nb = h[bench]
    nf6 = nb.iloc[-1] / nb.iloc[-127]
    nf12 = nb.iloc[-1] / nb.iloc[-253]

    rows = []
    for s in uni:
        col = h[s]
        p_now, p6, p12 = col.iloc[-1], col.iloc[-127], col.iloc[-253]
        if pd.isna(p_now) or pd.isna(p6) or pd.isna(p12) or p6 <= 0 or p12 <= 0:
            continue
        rs6 = (p_now / p6) / nf6
        rs12 = (p_now / p12) / nf12
        score = 0.5 * rs6 + 0.5 * rs12
        try:
            tier = mp._mcap_tier(s)
        except Exception:
            tier = ""
        rows.append({
            "symbol": s,
            "score": round(float(score), 3),
            "ret_6m": round(float((p_now / p6 - 1) * 100), 1),
            "ret_12m": round(float((p_now / p12 - 1) * 100), 1),
            "price": round(float(p_now), 1),
            "tier": tier,
        })

    rows.sort(key=lambda r: r["score"], reverse=True)
    top_n = int(mp.CFG.get("etf_size", 30))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
        r["in_top"] = i < top_n
    return {"asof": str(asof.date()), "bench": bench, "top_n": top_n, "count": len(rows), "rows": rows}


if __name__ == "__main__":
    import json
    d = scan()
    print("asof", d.get("asof"), "count", d.get("count"), "top_n", d.get("top_n"))
    for r in d.get("rows", [])[:12]:
        print(f"  #{r['rank']:>3} {r['symbol']:<14} score={r['score']:.3f} 6m={r['ret_6m']:+.1f}% 12m={r['ret_12m']:+.1f}% {r['tier']}")
