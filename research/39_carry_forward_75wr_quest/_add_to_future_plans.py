"""One-shot helper: insert the carry-forward research/39 entry into
data/future_plans.json. Idempotent — replaces if id already exists.
"""

import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
PATH = os.path.join(ROOT, 'data', 'future_plans.json')


def main() -> int:
    with open(PATH, encoding='utf-8') as f:
        data = json.load(f)

    new_entry = {
        "id": "carry-forward-75wr-quest",
        "title": "Carry-Forward 75% WR Quest - BTST + Multi-Day Swing + Weekly (research/39)",
        "subtitle": "After research/38 found 75% WR + favorable RR is structurally impossible on intraday 5-min equity, the hunt expands to overnight + multi-day holds where post-earnings drift, weekly continuation, and oversold-mean-reversion regimes have demonstrated 70-80% WR in academic literature. 1,623-stock daily universe (2000-2026) + 86-stock F&O cohort. Two sub-agents running in parallel on 5 pattern families.",
        "status": "backtesting",
        "created": "2026-05-06",
        "tags": ["carry-forward", "btst", "swing", "weekly", "daily-bars", "75wr", "favorable-rr", "fno", "research-39"],
        "sections": [
            {
                "title": "Why this thread exists",
                "body": "research/37 + research/38 exhausted the intraday 5-min equity space. Final finding: WR >= 75% with TP > SL is structurally not achievable on simple 5-min signals (random WR with TP/SL=0.5/1.5 is already 75%, real edge was only +3pp). Carry-forward holds (BTST -> multi-week) unlock different regimes where the math allows higher WR with favorable RR. Three locked intraday configs already shipped (see intraday-75wr-three-system-portfolio plan); this is the new hunt for a complement.",
                "items": []
            },
            {
                "title": "Hard gates for any winner",
                "items": [
                    "Walk-forward train WR >= 75% AND test WR >= 70%",
                    "Test PF >= 1.8",
                    "Test MaxDD <= 15%",
                    "Test n_trades >= 30",
                    "TP > SL (favorable RR >= 1.1)",
                    "Cost-stress: positive return at 0.20% round-trip cost (= ~0.10%/side)",
                    "Out-of-sample period 2024-01-01 to 2026-03-19 (~2.25 years held out)"
                ]
            },
            {
                "title": "5 pattern families being hunted (sub-agents in flight)",
                "items": [
                    {
                        "label": "1. BTST (overnight, 1-night hold) - sub-agent 1",
                        "body": "Strong-close + last-15-min momentum + sectoral confirmation entry near close, exit at next-day close. Indian retail-favorite pattern; structural overnight gap-and-go in liquid F&O names.",
                        "pros": ["Single-night exposure (no weekend / multi-day risk)", "Capital recycles fast", "Borrow/SLB risk minimal"],
                        "cons": ["Gap risk on adverse news between close and open", "Cost share bigger as % of small overnight move"]
                    },
                    {
                        "label": "2. Daily breakout swing (2-5 day hold) - sub-agent 1",
                        "body": "Donchian-20 / 52-week-high break + volume + rising 50-day-EMA. Trail with daily-low or fixed exit. Classic Turtle/momentum continuation in daily timeframe on F&O cohort.",
                        "pros": ["Well-documented academic edge in trend-following", "Multi-day target leaves room for big winners", "F&O carry cost ~0.06% round-trip"],
                        "cons": ["Trend-following has lower WR (40-55%) historically - favourable RR needed to compensate", "Whipsaw on choppy markets"]
                    },
                    {
                        "label": "3. Daily RSI mean-reversion (5-10 day hold) - sub-agent 2",
                        "body": "Stock close > 200-day SMA + daily RSI(14) < 25 -> LONG entry next-day open, hold 5-10 days, exit on RSI > 50 or fixed TP/SL. Mirror for SHORT (downtrend + RSI > 75).",
                        "pros": ["Mean-reversion in trend = high-probability setup", "Academic literature shows 65-75% WR on these conditions", "Liquid F&O names have natural reversion"],
                        "cons": ["Catches falling knives in regime breaks (e.g. Mar 2020)", "Long-term trend filter is single point of failure"]
                    },
                    {
                        "label": "4. Earnings post-drift / PEAD (5-20 day hold) - sub-agent 2",
                        "body": "Post-Earnings Announcement Drift - academic finding that stocks with positive earnings surprises drift up over 30-60 days. Uses gap+volume detection as earnings-date proxy if explicit calendar unavailable.",
                        "pros": ["Strong academic edge - Bernard & Thomas 1989 + dozens of replications", "Indian PEAD documented in IIM Bangalore studies", "Catalyst-driven so signal is causally robust"],
                        "cons": ["Need accurate earnings dates (proxy via gap+vol may be noisy)", "Indian small-caps have weaker PEAD than mid-large", "Mean-reversion in non-trending sessions"]
                    },
                    {
                        "label": "5. Pair trading / stat-arb (3-15 day hold) - DEFERRED",
                        "body": "Long sector laggard / short sector leader, mean-revert toward sector mean. Stat-arb with cointegration filtering. Requires sector mapping + cointegration testing - only pursued if patterns 1-4 don't crack 75% WR.",
                        "pros": ["Market-neutral so survives any regime", "Mathematically elegant - exploits relative-value mispricing", "Well-known to clear 75% WR in academic studies"],
                        "cons": ["Complex implementation (cointegration, half-life estimation)", "Borrow availability for short-leg names", "Position-sizing harder than directional"]
                    },
                    {
                        "label": "6. Weekly continuation (5-10 day hold) - sub-agent 1",
                        "body": "Weekly close above prior 8-week high (or below 8-week low) -> enter on next Monday open, hold for X weeks or until weekly close < prior week low.",
                        "pros": ["Higher-timeframe signals have less noise", "Trend-following on weekly bars = robust historical edge", "Lower transaction frequency = lower cost drag"],
                        "cons": ["Lower trade count (~20-40/year on F&O cohort)", "Weekly bars give late entries on fast moves"]
                    }
                ]
            },
            {
                "title": "Universe & data",
                "items": [
                    "Daily data: 1,623 stocks 2000-01-03 to 2026-03-19 in market_data_unified (timeframe='day')",
                    "Starting cohort: 86-stock F&O universe (services/data_manager.py FNO_LOT_SIZES) for borrow availability + tighter spreads",
                    "Optional broader universe: Nifty 500 stocks with >=5 years daily history + Rs.5Cr+ avg daily turnover",
                    "60-min data: 95 symbols 2018-2025 - for finer entry timing if Stage B requires it"
                ]
            },
            {
                "title": "Cost model - different from intraday MIS",
                "items": [
                    "F&O futures carry-forward (preferred): brokerage Rs.20/0.03% per side + STT 0.0125% sell + small exchange/GST = ~0.06% round-trip",
                    "CNC equity delivery: Rs.0 brokerage but STT 0.1% buy + 0.1% sell + small = ~0.21% round-trip (3.5x higher than F&O)",
                    "Default: 0.06% round-trip (F&O futures)",
                    "Stress test: 0.20% round-trip - winners must remain positive at this level"
                ]
            },
            {
                "title": "Walk-forward split",
                "items": [
                    "Train: 2018-01-01 to 2023-12-31 (6 years)",
                    "Test:  2024-01-01 to 2026-03-19 (~2.25 years)",
                    "Earnings pattern uses longer train (2010-2019) due to lower event count"
                ]
            },
            {
                "title": "Live status (2026-05-06)",
                "items": [
                    "Sub-agent 1 in flight: BTST + daily breakout swing + weekly continuation. 86-stock F&O cohort. Hold 1-10 days.",
                    "Sub-agent 2 in flight: RSI mean-reversion + earnings post-drift. F&O + Nifty 500 broader. Hold 5-20 days.",
                    "Pattern 5 (pair trading) deferred until 1+2 produce baseline - only pursued if intraday-style patterns don't crack 75%.",
                    "Time budget: multi-hour OK. Will surface results as each sub-agent completes."
                ]
            },
            {
                "title": "Files / references",
                "items": [
                    "research/39_carry_forward_75wr_quest/CARRY_FORWARD_75WR_DAILY_SWEEP_STATUS.md - live STATUS doc",
                    "research/39_carry_forward_75wr_quest/scripts/ - pattern modules + walk-forward",
                    "research/39_carry_forward_75wr_quest/results/ - per-stock screens + sweep CSVs",
                    "Final CARRY_FORWARD_75WR_DAILY_SWEEP_RESULTS.md will be written once sub-agents complete"
                ]
            }
        ]
    }

    # Idempotent: replace if id exists, else insert at top
    found = False
    for i, p in enumerate(data['plans']):
        if p['id'] == new_entry['id']:
            data['plans'][i] = new_entry
            found = True
            print(f'Replaced existing plan at index {i}')
            break
    if not found:
        data['plans'].insert(0, new_entry)
        print('Inserted as first plan')

    with open(PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f'Total plans: {len(data["plans"])}')
    for p in data['plans']:
        print(f'  - {p["id"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
