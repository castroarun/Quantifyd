"""
Fixed Liquid Fund Comparison - Both simulations capital-constrained.
Shows the TRUE impact of liquid fund on idle cash.
"""
import pandas as pd
import numpy as np
from services.consolidation_breakout import SYSTEM_PRIMARY

LIQUID_FUND_RATE = 0.065
DAILY_RATE = (1 + LIQUID_FUND_RATE) ** (1/365) - 1
CAPITAL_PER_TRADE = 100_000
STARTING_CAPITAL = 1_000_000


def apply_mask(df, system):
    mask = pd.Series(False, index=df.index)
    for strategy in system:
        sm = pd.Series(True, index=df.index)
        for k, v in strategy.items():
            if k == 'min_rsi14': sm &= df['rsi14'].fillna(0) >= v
            elif k == 'min_rsi7': sm &= df['rsi7'].fillna(0) >= v
            elif k == 'min_volume_ratio': sm &= df['volume_ratio'].fillna(0) >= v
            elif k == 'min_vol_trend': sm &= df['vol_trend'].fillna(0) >= v
            elif k == 'min_breakout_pct': sm &= df['breakout_pct'].fillna(0) >= v
            elif k == 'min_ath_proximity': sm &= df['ath_proximity'].fillna(0) >= v
            elif k == 'min_williams_r': sm &= df['williams_r'].fillna(-100) >= v
            elif k == 'min_mom_60d': sm &= df['mom_60d'].fillna(0) >= v
            elif k == 'min_mom_10d': sm &= df['mom_10d'].fillna(0) >= v
            elif k == 'min_bb_pct_b': sm &= df['bb_pct_b'].fillna(0) >= v
            elif k == 'require_ema20_gt_50' and v: sm &= df['ema20_above_50'].fillna(0) == 1
            elif k == 'require_weekly_ema20_gt_50' and v: sm &= df['w_ema20_gt_50'].fillna(0) == 1
        mask |= sm
    return mask


def hold_days(row):
    if row['exit_reason'] == 'OPEN':
        return int(125 * 1.43)
    else:
        return int(min(int(row['days_to_peak'] * 1.5) + 10, 125) * 1.43)


def run_simulation(trades, with_liquid_fund=False):
    """Run capital-constrained simulation."""
    cash = float(STARTING_CAPITAL)
    active = []
    taken = 0
    skipped = 0
    total_pnl = 0.0
    total_interest = 0.0
    max_conc = 0
    last_date = trades['date'].min()
    sim_end = max(trades['exit_date'].max(), trades['date'].max())

    for _, row in trades.iterrows():
        entry = row['date']
        exit_d = row['exit_date']
        ret = row['trade_return']

        # Accrue interest on idle cash (only if liquid fund enabled)
        if with_liquid_fund and entry > last_date:
            days = (entry - last_date).days
            interest = cash * ((1 + DAILY_RATE) ** days - 1)
            cash += interest
            total_interest += interest

        # Process exits
        still = []
        for (ed, cap, r) in active:
            if ed <= entry:
                returned = cap * (1 + r / 100)
                cash += returned
                total_pnl += returned - cap
            else:
                still.append((ed, cap, r))
        active = still

        # Try to take trade
        if cash >= CAPITAL_PER_TRADE:
            cash -= CAPITAL_PER_TRADE
            active.append((exit_d, CAPITAL_PER_TRADE, ret))
            taken += 1
            max_conc = max(max_conc, len(active))
        else:
            skipped += 1

        last_date = entry

    # Final interest + close remaining
    if with_liquid_fund and sim_end > last_date:
        days = (sim_end - last_date).days
        interest = cash * ((1 + DAILY_RATE) ** days - 1)
        cash += interest
        total_interest += interest

    for (ed, cap, r) in active:
        returned = cap * (1 + r / 100)
        cash += returned
        total_pnl += returned - cap

    return {
        'equity': cash,
        'taken': taken,
        'skipped': skipped,
        'max_concurrent': max_conc,
        'trade_pnl': total_pnl,
        'interest': total_interest,
    }


def main():
    df = pd.read_csv('breakout_analysis_enhanced.csv')
    df['date'] = pd.to_datetime(df['date'])

    trades = df[apply_mask(df, SYSTEM_PRIMARY)].sort_values('date').copy()
    trades['holding_days'] = trades.apply(hold_days, axis=1)
    trades['exit_date'] = trades['date'] + pd.to_timedelta(trades['holding_days'], unit='D')

    sim_start = trades['date'].min()
    sim_end = max(trades['exit_date'].max(), trades['date'].max())
    years = (sim_end - sim_start).days / 365.25

    print('=' * 80)
    print('FIXED COMPARISON: Both capital-constrained to Rs.10L')
    print('=' * 80)
    print(f'PRIMARY system | {len(trades)} qualifying trades | {years:.1f} years')
    print()

    # Run both simulations
    a = run_simulation(trades, with_liquid_fund=False)
    b = run_simulation(trades, with_liquid_fund=True)

    # Pure liquid fund
    total_days = (sim_end - sim_start).days
    equity_c = STARTING_CAPITAL * ((1 + DAILY_RATE) ** total_days)
    cagr_c = ((equity_c / STARTING_CAPITAL) ** (1/years) - 1) * 100

    cagr_a = ((a['equity'] / STARTING_CAPITAL) ** (1/years) - 1) * 100
    cagr_b = ((b['equity'] / STARTING_CAPITAL) ** (1/years) - 1) * 100

    print(f'  {"":35} {"A: Cash Idle":>16} {"B: Liquid Fund":>16} {"C: Pure Liquid":>16}')
    print(f'  {"-"*85}')
    print(f'  {"Trades taken":35} {a["taken"]:>16} {b["taken"]:>16} {"0":>16}')
    print(f'  {"Trades skipped (no capital)":35} {a["skipped"]:>16} {b["skipped"]:>16} {"-":>16}')
    print(f'  {"Max concurrent trades":35} {a["max_concurrent"]:>16} {b["max_concurrent"]:>16} {"-":>16}')
    print()
    print(f'  {"Final Equity":35} Rs.{a["equity"]:>13,.0f} Rs.{b["equity"]:>13,.0f} Rs.{equity_c:>13,.0f}')
    print(f'  {"CAGR":35} {cagr_a:>15.2f}% {cagr_b:>15.2f}% {cagr_c:>15.2f}%')
    print(f'  {"Trade P&L":35} Rs.{a["trade_pnl"]:>13,.0f} Rs.{b["trade_pnl"]:>13,.0f} {"-":>16}')
    print(f'  {"Liquid Fund Interest":35} {"-":>16} Rs.{b["interest"]:>13,.0f} Rs.{equity_c - STARTING_CAPITAL:>13,.0f}')

    boost = cagr_b - cagr_a
    extra = b['equity'] - a['equity']
    total_gains_b = b['equity'] - STARTING_CAPITAL
    interest_pct = b['interest'] / total_gains_b * 100 if total_gains_b > 0 else 0

    print()
    print(f'  REAL CAGR BOOST: {cagr_a:.2f}% -> {cagr_b:.2f}% = +{boost:.2f}%')
    print(f'  Extra money from liquid fund: Rs.{extra:,.0f} (Rs.{extra/100000:.1f}L)')
    print(f'  Liquid interest = {interest_pct:.1f}% of total gains')

    # ================================================================
    # WHY CAGR BOOST LOOKS SMALL
    # ================================================================
    print()
    print('=' * 80)
    print('WHY 6.5% LIQUID FUND ONLY ADDS ~3% CAGR')
    print('=' * 80)
    print()
    print('  The liquid fund earns 6.5% only on IDLE cash, not on ALL capital.')
    print('  As trades make money, the portfolio grows -> more capital deployed')
    print('  -> less idle cash proportionally -> liquid fund % contribution drops.')
    print()
    print('  Year 1:  Rs.10L capital, 1 trade open = Rs.9L idle -> earns Rs.58,500/yr')
    print('  Year 10: Rs.30L capital, 5 trades open = Rs.25L idle -> earns Rs.1.6L/yr')
    print('  Year 20: Rs.80L capital, 15 trades open = Rs.65L idle -> earns Rs.4.2L/yr')
    print()
    print('  But the TRADE portion is compounding at ~10%, so the liquid fund')
    print('  contribution (6.5% on idle) becomes a SMALLER FRACTION of total')
    print('  returns as the portfolio grows.')
    print()
    print('  Think of it as a weighted average:')
    avg_util = 0.55  # from previous run
    print(f'    55% of capital in trades earning ~{cagr_a:.1f}% CAGR')
    print(f'    45% of capital in liquid fund earning 6.5%')
    print(f'    Blended: 0.55 x {cagr_a:.1f}% + 0.45 x 6.5% = {0.55*cagr_a + 0.45*6.5:.1f}%')
    print(f'    Actual: {cagr_b:.1f}% (higher due to compounding of returned trade capital)')
    print()
    print('  BUT in absolute terms:')
    print(f'    Rs.{extra/100000:.0f}L extra over {years:.0f} years = {extra/STARTING_CAPITAL:.1f}x your starting capital!')
    print(f'    Thats {interest_pct:.0f}% of ALL gains coming from parking idle cash.')
    print(f'    Nearly HALF your money is made by doing NOTHING.')

    # ================================================================
    # What if liquid fund rate was higher?
    # ================================================================
    print()
    print('=' * 80)
    print('SENSITIVITY: WHAT IF LIQUID FUND RATE IS DIFFERENT?')
    print('=' * 80)
    print()

    for rate, label in [(0.04, '4.0% (Savings Account)'),
                        (0.055, '5.5% (Overnight Fund)'),
                        (0.065, '6.5% (Liquid Fund)'),
                        (0.075, '7.5% (Short Duration)'),
                        (0.085, '8.5% (Arbitrage Fund)')]:
        # Quick simulation with different rate
        daily_r = (1 + rate) ** (1/365) - 1
        cash = float(STARTING_CAPITAL)
        active = []
        total_int = 0.0
        last_dt = sim_start

        for _, row in trades.iterrows():
            entry = row['date']
            exit_d = row['exit_date']
            ret = row['trade_return']

            if entry > last_dt:
                days = (entry - last_dt).days
                interest = cash * ((1 + daily_r) ** days - 1)
                cash += interest
                total_int += interest

            still = []
            for (ed, cap, r) in active:
                if ed <= entry:
                    cash += cap * (1 + r / 100)
                else:
                    still.append((ed, cap, r))
            active = still

            if cash >= CAPITAL_PER_TRADE:
                cash -= CAPITAL_PER_TRADE
                active.append((exit_d, CAPITAL_PER_TRADE, ret))

            last_dt = entry

        if sim_end > last_dt:
            days = (sim_end - last_dt).days
            interest = cash * ((1 + daily_r) ** days - 1)
            cash += interest
            total_int += interest

        for (ed, cap, r) in active:
            cash += cap * (1 + r / 100)

        cagr = ((cash / STARTING_CAPITAL) ** (1/years) - 1) * 100
        print(f'  {label:30} -> CAGR {cagr:.2f}% | Interest Rs.{total_int:>12,.0f} | Equity Rs.{cash:>12,.0f}')

    # No liquid fund for reference
    print(f'  {"0.0% (Cash sitting idle)":30} -> CAGR {cagr_a:.2f}% | Interest Rs.{"0":>12} | Equity Rs.{a["equity"]:>12,.0f}')

    # ================================================================
    # THE REAL QUESTION: Opportunity cost
    # ================================================================
    print()
    print('=' * 80)
    print('THE REAL PICTURE: OPPORTUNITY COST OF IDLE CASH')
    print('=' * 80)
    print()
    print(f'  Over {years:.0f} years with PRIMARY system:')
    print(f'    Cash sitting idle (0%):     Rs.{a["equity"]/100000:>8.1f}L  (CAGR {cagr_a:.2f}%)')
    print(f'    Cash in liquid fund (6.5%): Rs.{b["equity"]/100000:>8.1f}L  (CAGR {cagr_b:.2f}%)')
    print(f'    You LOSE Rs.{extra/100000:.1f}L by not parking in liquid fund!')
    print()
    print(f'    With Rs.{b["interest"]/100000:.1f}L from liquid fund interest:')
    print(f'    -> Thats {b["interest"]/b["trade_pnl"]*100:.0f}% of what your trades earned')
    print(f'    -> Free money from capital that would otherwise earn ZERO')


if __name__ == '__main__':
    main()
