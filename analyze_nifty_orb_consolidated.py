"""
Consolidated Nifty ORB obedience stats — single comprehensive table.
Reads nifty_orb_signals.csv produced by backtest_nifty_orb_stats.py.
"""

import pandas as pd

CSV = 'nifty_orb_signals.csv'


def filter_describe(df, label, mask):
    sub = df[mask].copy()
    n = len(sub)
    if n == 0:
        return {'variant': label, 'N': 0, 'Wins': 0, 'Losses': 0, 'WR_pct': 0.0,
                'Med_min_to_SL': None, 'Med_min_winner_held': None,
                'Avg_win_move_pct': None, 'Avg_loss_move_pct': None}
    wins = sub[~sub['sl_hit']]
    losses = sub[sub['sl_hit']].copy()

    # Time stats
    sub['entry_time'] = pd.to_datetime(sub['entry_time'])
    if len(losses):
        losses['entry_time'] = pd.to_datetime(losses['entry_time'])
        losses['sl_time'] = pd.to_datetime(losses['sl_time'])
        losses['mins_to_sl'] = (losses['sl_time'] - losses['entry_time']).dt.total_seconds() / 60
        med_min_to_sl = round(float(losses['mins_to_sl'].median()), 0)
    else:
        med_min_to_sl = None

    if len(wins):
        wins = wins.copy()
        wins['entry_time'] = pd.to_datetime(wins['entry_time'])
        # Winner held until 15:30 EOD
        wins['mins_held'] = (
            wins['entry_time'].dt.normalize() + pd.Timedelta(hours=15, minutes=30) - wins['entry_time']
        ).dt.total_seconds() / 60
        med_min_held = round(float(wins['mins_held'].median()), 0)
    else:
        med_min_held = None

    return {
        'variant': label,
        'N': n,
        'Wins': len(wins),
        'Losses': len(losses),
        'WR_pct': round(len(wins) / n * 100, 1),
        'Med_min_to_SL': med_min_to_sl,
        'Med_min_winner_held': med_min_held,
        'Avg_win_move_pct': round(float(wins['pnl_pct_underlying'].mean()), 2) if len(wins) else None,
        'Avg_loss_move_pct': round(float(losses['pnl_pct_underlying'].mean()), 2) if len(losses) else None,
    }


def main():
    df = pd.read_csv(CSV)
    print(f"Loaded {len(df)} raw signals from {CSV}")
    print(f"Sessions covered: ~{df['date'].nunique()}\n")

    rows = []

    # Direction-aware filters
    is_long = df['direction'] == 'LONG'
    is_short = df['direction'] == 'SHORT'

    def rsi5m(lo, hi):
        return ((is_long & (df['rsi5m'] > lo)) | (is_short & (df['rsi5m'] < hi)))

    def rsi15m(lo, hi):
        return ((is_long & (df['rsi15m'] > lo)) | (is_short & (df['rsi15m'] < hi)))

    # =========== Section A: baseline by OR window ===========
    for w in [5, 10, 15, 30]:
        rows.append(filter_describe(df, f"OR{w}m  (no filter)", df['or_min'] == w))

    # =========== Section B: OR15 with 5-min RSI ladder ===========
    for lo, hi in [(50, 50), (55, 45), (60, 40), (65, 35), (70, 30)]:
        rows.append(filter_describe(df, f"OR15m  RSI5m>{lo} L  /  <{hi} S",
                                    (df['or_min'] == 15) & rsi5m(lo, hi)))

    # =========== Section C: OR15 with 15-min RSI ladder (for comparison) ===========
    for lo, hi in [(60, 40), (65, 35)]:
        rows.append(filter_describe(df, f"OR15m  RSI15m>{lo} L  /  <{hi} S",
                                    (df['or_min'] == 15) & rsi15m(lo, hi)))

    # =========== Section D: OR15 + RSI>60/<40 + gap slicing ===========
    base_rsi = (df['or_min'] == 15) & rsi5m(60, 40)
    rows.append(filter_describe(df, "OR15m  RSI5m>60/<40  any gap",
                                base_rsi))
    rows.append(filter_describe(df, "OR15m  RSI5m>60/<40  |gap|<=0.5%",
                                base_rsi & (df['gap_pct'].abs() <= 0.5)))
    rows.append(filter_describe(df, "OR15m  RSI5m>60/<40  |gap|>0.5%",
                                base_rsi & (df['gap_pct'].abs() > 0.5)))
    rows.append(filter_describe(df, "OR15m  RSI5m>60/<40  |gap|>1.0%",
                                base_rsi & (df['gap_pct'].abs() > 1.0)))

    # =========== Section E: rescue check on the bad gap bucket ===========
    bad_gap = base_rsi & (df['gap_pct'].abs() > 0.5)
    rows.append(filter_describe(df, "  + tighter RSI5m>65/<35",
                                bad_gap & rsi5m(65, 35)))
    rows.append(filter_describe(df, "  + SMA50 align (long above / short below)",
                                bad_gap & (
                                    (is_long & (df['above_sma50'] == True)) |
                                    (is_short & (df['above_sma50'] == False))
                                )))
    rows.append(filter_describe(df, "  + SMA200 align",
                                bad_gap & (
                                    (is_long & (df['above_sma200'] == True)) |
                                    (is_short & (df['above_sma200'] == False))
                                )))
    rows.append(filter_describe(df, "  + MACD align",
                                bad_gap & (
                                    (is_long & (df['macd_bullish'] == True)) |
                                    (is_short & (df['macd_bullish'] == False))
                                )))

    # =========== Section F: OR30 variants ===========
    rows.append(filter_describe(df, "OR30m  (no filter)",
                                df['or_min'] == 30))
    rows.append(filter_describe(df, "OR30m  RSI5m>60/<40",
                                (df['or_min'] == 30) & rsi5m(60, 40)))
    rows.append(filter_describe(df, "OR30m  RSI5m>65/<35",
                                (df['or_min'] == 30) & rsi5m(65, 35)))
    rows.append(filter_describe(df, "OR30m  RSI5m>60/<40  |gap|<=0.5%",
                                (df['or_min'] == 30) & rsi5m(60, 40) & (df['gap_pct'].abs() <= 0.5)))

    # =========== Section G: direction split on best variants ===========
    for w in [15, 30]:
        rows.append(filter_describe(df, f"OR{w}m  RSI5m>60  LONG only",
                                    (df['or_min'] == w) & is_long & (df['rsi5m'] > 60)))
        rows.append(filter_describe(df, f"OR{w}m  RSI5m<40  SHORT only",
                                    (df['or_min'] == w) & is_short & (df['rsi5m'] < 40)))

    out = pd.DataFrame(rows)
    # Pretty print
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 50)
    print("=" * 130)
    print("CONSOLIDATED NIFTY ORB OBEDIENCE TABLE")
    print("Win = opposite OR was NOT breached before EOD (= credit-spread theta survives)")
    print("Loss = opposite OR breached at some point (= credit-spread SL would have fired)")
    print("=" * 130)
    print(out.to_string(index=False))

    # Save
    out.to_csv('nifty_orb_consolidated.csv', index=False)
    print(f"\nSaved: nifty_orb_consolidated.csv")


if __name__ == '__main__':
    main()
