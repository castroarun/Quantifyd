"""Simulate NAS OTM strangle trades for 2026-03-30 using real options snapshots."""
import sqlite3

db = sqlite3.connect('backtest_data/options_data.db')

# Config
TARGET_PREM = 20.0
MIN_PREM = 5.0
MAX_PREM = 24.0
ADJ_MIN_PREM = 4.0
ADJ_MAX_PREM = 24.0
MIN_OTM = 100
LOTS = 10
LOT_SIZE = 75
IMBALANCE_TRIGGER = 2.0
EXPIRY = '2026-03-30'

# Snapshot times (round)
snapshot_times = [
    '2026-03-30T10:00:00', '2026-03-30T10:30:00', '2026-03-30T11:00:00',
    '2026-03-30T11:30:00', '2026-03-30T12:00:00', '2026-03-30T12:30:00',
    '2026-03-30T13:00:00', '2026-03-30T14:00:00', '2026-03-30T15:00:00',
    '2026-03-30T15:20:00',
]

# Get spot at each time
spots = {}
for t in snapshot_times:
    r = db.execute(
        "SELECT spot_price FROM underlying_spot WHERE snapshot_time=? AND symbol='NIFTY'", (t,)
    ).fetchone()
    if r:
        spots[t] = r[0]


def find_strike(snapshot_time, inst_type, spot, target_prem, min_p=None, max_p=None):
    """Find OTM strike closest to target premium."""
    if min_p is None:
        min_p = MIN_PREM
    if max_p is None:
        max_p = MAX_PREM

    if inst_type == 'CE':
        min_strike = spot + MIN_OTM
        max_strike = spot + 3000
    else:
        min_strike = spot - 3000
        max_strike = spot - MIN_OTM

    rows = db.execute("""
        SELECT strike, ltp FROM option_chain
        WHERE snapshot_time=? AND symbol='NIFTY' AND expiry_date=?
          AND instrument_type=? AND strike BETWEEN ? AND ?
        ORDER BY strike
    """, (snapshot_time, EXPIRY, inst_type, min_strike, max_strike)).fetchall()

    # Deduplicate
    seen = {}
    for strike, ltp in rows:
        if strike not in seen:
            seen[strike] = ltp

    best = None
    best_diff = float('inf')
    for strike, ltp in seen.items():
        if min_p <= ltp <= max_p:
            diff = abs(ltp - target_prem)
            if diff < best_diff:
                best_diff = diff
                best = (strike, ltp)
    return best


def get_premium(snapshot_time, strike, inst_type):
    r = db.execute("""
        SELECT ltp FROM option_chain
        WHERE snapshot_time=? AND symbol='NIFTY' AND expiry_date=?
          AND instrument_type=? AND strike=?
        LIMIT 1
    """, (snapshot_time, EXPIRY, inst_type, strike)).fetchone()
    return r[0] if r else None


# ==================== SIMULATE ====================
print("=" * 90)
print("NAS OTM SIMULATION - 2026-03-30 (Expiry Day)")
print("=" * 90)
print(f"Config: Target Rs {TARGET_PREM}, Bounds [{MIN_PREM}-{MAX_PREM}], "
      f"Min OTM {MIN_OTM}pts, {LOTS} lots ({LOTS * LOT_SIZE} qty/leg)")
print()

positions = []
trades = []
adj_count = 0
adj_direction = 'OUT'
total_pnl = 0

# Entry at 10:00
entry_time = '2026-03-30T10:00:00'
spot = spots[entry_time]
print(f"[10:00] Spot: {spot:.1f} - Squeeze detected, looking for entry...")

ce_result = find_strike(entry_time, 'CE', spot, TARGET_PREM)
pe_result = find_strike(entry_time, 'PE', spot, TARGET_PREM)

if ce_result and pe_result:
    ce_strike, ce_prem = ce_result
    pe_strike, pe_prem = pe_result
    print(f"  ENTRY: SELL {ce_strike:.0f}CE @ Rs {ce_prem:.2f} + SELL {pe_strike:.0f}PE @ Rs {pe_prem:.2f}")
    print(f"  Total premium collected: Rs {ce_prem + pe_prem:.2f}")
    positions = [
        {'leg': 'CE', 'strike': ce_strike, 'entry_prem': ce_prem, 'entry_time': entry_time, 'current_prem': ce_prem},
        {'leg': 'PE', 'strike': pe_strike, 'entry_prem': pe_prem, 'entry_time': entry_time, 'current_prem': pe_prem},
    ]
else:
    print("  No valid strikes found!")

print()

# Walk through snapshots
for t in snapshot_times[1:]:
    if not positions:
        break
    spot = spots.get(t)
    if not spot:
        continue

    time_str = t.split('T')[1][:5]

    # Update premiums
    for pos in positions:
        ltp = get_premium(t, pos['strike'], pos['leg'])
        if ltp is not None:
            pos['current_prem'] = ltp

    ce_pos = next((p for p in positions if p['leg'] == 'CE'), None)
    pe_pos = next((p for p in positions if p['leg'] == 'PE'), None)

    if ce_pos and pe_pos:
        ce_ltp = ce_pos['current_prem']
        pe_ltp = pe_pos['current_prem']

        if ce_ltp >= pe_ltp:
            expensive, cheap = ce_pos, pe_pos
        else:
            expensive, cheap = pe_pos, ce_pos
        exp_ltp = expensive['current_prem']
        chp_ltp = cheap['current_prem']
        ratio = exp_ltp / chp_ltp if chp_ltp > 0 else 999

        print(f"[{time_str}] Spot: {spot:.1f} | "
              f"{ce_pos['strike']:.0f}CE={ce_ltp:.2f} | "
              f"{pe_pos['strike']:.0f}PE={pe_ltp:.2f} | "
              f"Ratio: {ratio:.1f}x")

        # Check cross-leg imbalance
        if ratio >= IMBALANCE_TRIGGER:
            adj_count += 1

            if adj_direction == 'OUT':
                adj_leg = expensive
                target_prem = chp_ltp
            else:
                adj_leg = cheap
                target_prem = exp_ltp

            # Check bounds
            if target_prem < ADJ_MIN_PREM or target_prem > ADJ_MAX_PREM:
                # Try flip
                if adj_direction == 'OUT':
                    adj_leg = cheap
                    target_prem = exp_ltp
                else:
                    adj_leg = expensive
                    target_prem = chp_ltp

                if target_prem < ADJ_MIN_PREM or target_prem > ADJ_MAX_PREM:
                    # Close both
                    pnl_ce = (ce_pos['entry_prem'] - ce_ltp) * LOTS * LOT_SIZE
                    pnl_pe = (pe_pos['entry_prem'] - pe_ltp) * LOTS * LOT_SIZE
                    total_pnl += pnl_ce + pnl_pe
                    print(f"  >>> BOUNDARY EXIT: Both directions outside [{ADJ_MIN_PREM}-{ADJ_MAX_PREM}]")
                    print(f"      CE P&L: Rs {pnl_ce:,.0f} | PE P&L: Rs {pnl_pe:,.0f}")
                    trades.append({'leg': 'CE', 'strike': ce_pos['strike'], 'entry': ce_pos['entry_prem'],
                                   'exit': ce_ltp, 'pnl': pnl_ce, 'time': t, 'reason': 'boundary'})
                    trades.append({'leg': 'PE', 'strike': pe_pos['strike'], 'entry': pe_pos['entry_prem'],
                                   'exit': pe_ltp, 'pnl': pnl_pe, 'time': t, 'reason': 'boundary'})
                    positions = []
                    continue

            # Close old leg
            old_prem = adj_leg['current_prem']
            old_pnl = (adj_leg['entry_prem'] - old_prem) * LOTS * LOT_SIZE
            total_pnl += old_pnl
            action = 'ROLL_OUT' if adj_direction == 'OUT' else 'ROLL_IN'

            print(f"  >>> ADJ #{adj_count} {action}: Close {adj_leg['strike']:.0f}{adj_leg['leg']} "
                  f"@ {old_prem:.2f} (entry: {adj_leg['entry_prem']:.2f}, P&L: Rs {old_pnl:,.0f})")
            trades.append({'leg': adj_leg['leg'], 'strike': adj_leg['strike'], 'entry': adj_leg['entry_prem'],
                           'exit': old_prem, 'pnl': old_pnl, 'time': t, 'reason': action})

            # Find new strike
            new_result = find_strike(t, adj_leg['leg'], spot, target_prem, ADJ_MIN_PREM, ADJ_MAX_PREM)
            if new_result:
                new_strike, new_prem = new_result
                print(f"      -> New: SELL {new_strike:.0f}{adj_leg['leg']} @ Rs {new_prem:.2f} "
                      f"(target was {target_prem:.2f})")
                adj_leg['strike'] = new_strike
                adj_leg['entry_prem'] = new_prem
                adj_leg['entry_time'] = t
                adj_leg['current_prem'] = new_prem
                adj_direction = 'IN' if adj_direction == 'OUT' else 'OUT'
            else:
                print(f"      -> No valid strike for {adj_leg['leg']} @ target {target_prem:.2f}, leg removed")
                positions = [p for p in positions if p is not adj_leg]

    # EOD squareoff
    if time_str >= '15:15' and positions:
        print(f"  >>> EOD SQUAREOFF")
        for pos in positions:
            ltp = pos['current_prem']
            pnl = (pos['entry_prem'] - ltp) * LOTS * LOT_SIZE
            total_pnl += pnl
            print(f"      Close {pos['strike']:.0f}{pos['leg']} @ {ltp:.2f} "
                  f"(entry: {pos['entry_prem']:.2f}, P&L: Rs {pnl:,.0f})")
            trades.append({'leg': pos['leg'], 'strike': pos['strike'], 'entry': pos['entry_prem'],
                           'exit': ltp, 'pnl': pnl, 'time': t, 'reason': 'eod'})
        positions = []
        break

# Summary
print()
print("=" * 90)
print("TRADE SUMMARY")
print("=" * 90)
print(f"{'#':<4} {'Time':<8} {'Leg':<5} {'Strike':<8} {'Entry':>8} {'Exit':>8} {'P&L':>12} {'Reason':<12}")
print("-" * 70)
for i, tr in enumerate(trades, 1):
    tstr = tr['time'].split('T')[1][:5]
    print(f"{i:<4} {tstr:<8} {tr['leg']:<5} {tr['strike']:<8.0f} {tr['entry']:>8.2f} {tr['exit']:>8.2f} "
          f"{tr['pnl']:>12,.0f} {tr['reason']:<12}")
print("-" * 70)
print(f"{'TOTAL':<43} {total_pnl:>12,.0f}")
print(f"\nAdjustments: {adj_count}")
print(f"Qty per leg: {LOTS * LOT_SIZE}")

db.close()
