"""
NWV — Nifty Weekly View — View Engine
=======================================
Phase 0: computes the Monday-morning weekly view. No Kite orders.

Pipeline (see docs/NWV-PLAN.md for the authoritative design):

    Weekly state (prev week HLC, pivots, CPR, CPR bucket, monthly CPR)
        |
    Mon 09:45 first 30-min candle + gap %
        |
    Base view = matrix_lookup(cpr_bucket, candle_pos, candle_body)
        |
    Gap dampener
        |
    Monthly CPR override (SOFT, demote one step if against macro)
        |
    Conviction score build-up (ADX, VIX percentile, first-candle quality)
        |
    Stacked-level detection (weekly + daily pivot clusters within 0.25%)
        |
    Instrument + expected range + time stop
        |
    Persist to nwv_views, return dict
"""

import json
import logging
import math
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List, Tuple

from services.nwv_db import get_nwv_db

logger = logging.getLogger(__name__)

# ─── Thresholds (locked with user) ──────────────────────────
CPR_WIDE_PCT = 0.80           # >= 0.80% of spot ⇒ wide
CPR_NARROW_PCT = 0.40         # <= 0.40% of spot ⇒ narrow
GAP_CONSIDERABLE_PCT = 0.70
GAP_SMALL_PCT = 0.30
STACKED_LEVEL_DISTANCE_PCT = 0.25   # weekly-vs-daily pivots within 0.25% cluster
VIX_LOOKBACK_DAYS = 60
ADX_CHOP_MAX = 20
ADX_BUILDING_MIN = 20
ADX_TREND_MIN = 25

# ─── View constants ─────────────────────────────────────────
VIEW_BEARISH = 'bearish'
VIEW_NTB     = 'neutral_to_bearish'
VIEW_NEUTRAL = 'neutral'
VIEW_NTBULL  = 'neutral_to_bullish'
VIEW_BULLISH = 'bullish'
VIEW_IGNORE  = 'ignore'

_DEMOTE_BEARISH = {
    VIEW_BEARISH: VIEW_NTB,
    VIEW_NTB: VIEW_NEUTRAL,
    VIEW_NEUTRAL: VIEW_NEUTRAL,
    VIEW_NTBULL: VIEW_NEUTRAL,
    VIEW_BULLISH: VIEW_NTBULL,
    VIEW_IGNORE: VIEW_IGNORE,
}
_DEMOTE_BULLISH = {
    VIEW_BULLISH: VIEW_NTBULL,
    VIEW_NTBULL: VIEW_NEUTRAL,
    VIEW_NEUTRAL: VIEW_NEUTRAL,
    VIEW_NTB: VIEW_NEUTRAL,
    VIEW_BEARISH: VIEW_NTB,
    VIEW_IGNORE: VIEW_IGNORE,
}


# ─── Pure helpers ───────────────────────────────────────────

def compute_cpr(prev_high: float, prev_low: float, prev_close: float) -> Dict[str, float]:
    pp = (prev_high + prev_low + prev_close) / 3.0
    bc = (prev_high + prev_low) / 2.0
    tc = 2.0 * pp - bc
    # Normalize so tc > bc always (symmetric formula can invert in rare setups)
    if tc < bc:
        tc, bc = bc, tc
    return {'pp': pp, 'tc': tc, 'bc': bc}


def compute_pivots(prev_high: float, prev_low: float, prev_close: float) -> Dict[str, float]:
    pp = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2.0 * pp - prev_low
    s1 = 2.0 * pp - prev_high
    r2 = pp + (prev_high - prev_low)
    s2 = pp - (prev_high - prev_low)
    return {'pp': pp, 'r1': r1, 's1': s1, 'r2': r2, 's2': s2}


def classify_cpr_width(cpr_width_pct: float) -> str:
    if cpr_width_pct >= CPR_WIDE_PCT:
        return 'wide'
    if cpr_width_pct <= CPR_NARROW_PCT:
        return 'narrow'
    return 'normal'


def classify_gap(gap_pct: float) -> Tuple[str, str]:
    """Returns (tier, direction). Tier in {none, small, considerable}."""
    abs_g = abs(gap_pct)
    if abs_g < GAP_SMALL_PCT:
        tier = 'none'
    elif abs_g < GAP_CONSIDERABLE_PCT:
        tier = 'small'
    else:
        tier = 'considerable'
    direction = 'up' if gap_pct > 0 else ('down' if gap_pct < 0 else 'flat')
    return tier, direction


def classify_first_candle(o: float, h: float, l: float, c: float,
                           cpr_tc: float, cpr_bc: float) -> Dict[str, Any]:
    """Classify the first 30-min candle on two axes.

    1) Position vs CPR — above_cpr / below_cpr / inside_cpr (vs TC and BC).
    2) Body direction — bullish (c > o), bearish (c < o), doji (c == o,
       small body).

    Also reports:
      - range_pct: (h-l) / ((h+l)/2) * 100
      - wick_pos_pct: where the body center sits within the range,
        0=bottom, 100=top. Useful for detecting wick-rejection candles
        (body near high or low).
    """
    # Position
    if c > cpr_tc:
        pos = 'above_cpr'
    elif c < cpr_bc:
        pos = 'below_cpr'
    else:
        pos = 'inside_cpr'

    # Body direction with doji tolerance (<0.03% = doji)
    mid = (h + l) / 2.0 if (h + l) else 0.0
    body_pct = abs(c - o) / mid * 100.0 if mid else 0.0
    if body_pct < 0.03:
        body = 'doji'
    elif c > o:
        body = 'bullish'
    else:
        body = 'bearish'

    range_pct = (h - l) / mid * 100.0 if mid else 0.0
    body_center = (o + c) / 2.0
    wick_pos_pct = ((body_center - l) / (h - l) * 100.0) if (h > l) else 50.0

    return {
        'pos': pos,
        'body': body,
        'range_pct': round(range_pct, 4),
        'wick_pos_pct': round(wick_pos_pct, 2),
        'body_pct': round(body_pct, 4),
    }


def base_view_matrix(cpr_bucket: str, candle_pos: str, candle_body: str) -> str:
    """The 5×2 table from the system design (extended with inside_cpr + doji)."""
    if cpr_bucket == 'narrow':
        return VIEW_IGNORE

    if candle_pos == 'inside_cpr' or candle_body == 'doji':
        return VIEW_NEUTRAL

    if cpr_bucket == 'wide':
        if candle_pos == 'below_cpr' and candle_body == 'bearish':
            return VIEW_NTB
        if candle_pos == 'below_cpr' and candle_body == 'bullish':
            return VIEW_NEUTRAL
        if candle_pos == 'above_cpr' and candle_body == 'bullish':
            return VIEW_NTBULL
        if candle_pos == 'above_cpr' and candle_body == 'bearish':
            return VIEW_NEUTRAL

    elif cpr_bucket == 'normal':
        if candle_pos == 'below_cpr' and candle_body == 'bearish':
            return VIEW_BEARISH
        if candle_pos == 'below_cpr' and candle_body == 'bullish':
            return VIEW_NEUTRAL
        if candle_pos == 'above_cpr' and candle_body == 'bullish':
            return VIEW_BULLISH
        if candle_pos == 'above_cpr' and candle_body == 'bearish':
            return VIEW_NEUTRAL

    return VIEW_NEUTRAL


def apply_gap_dampener(base_view: str, gap_tier: str, gap_direction: str) -> Tuple[str, bool]:
    """If gap is considerable AND in the same direction as the view, demote by one step."""
    if gap_tier != 'considerable':
        return base_view, False
    # Bearish-side dampening (gap DOWN on a bearish/NtB setup)
    if gap_direction == 'down' and base_view in (VIEW_BEARISH, VIEW_NTB):
        return _DEMOTE_BEARISH[base_view], True
    # Bullish-side dampening (gap UP on a bullish/NtB setup)
    if gap_direction == 'up' and base_view in (VIEW_BULLISH, VIEW_NTBULL):
        return _DEMOTE_BULLISH[base_view], True
    return base_view, False


def apply_monthly_override(view: str, spot: float, monthly_tc: Optional[float],
                            monthly_bc: Optional[float]) -> Tuple[str, Optional[str]]:
    """SOFT monthly CPR override. Returns (new_view, override_side_or_None).

    - Spot firmly above monthly TC → macro bullish → demote bearish-side views one step.
    - Spot firmly below monthly BC → macro bearish → demote bullish-side views one step.
    """
    if monthly_tc is None or monthly_bc is None or spot is None:
        return view, None
    if spot > monthly_tc and view in (VIEW_BEARISH, VIEW_NTB):
        return _DEMOTE_BEARISH[view], 'macro_bullish'
    if spot < monthly_bc and view in (VIEW_BULLISH, VIEW_NTBULL):
        return _DEMOTE_BULLISH[view], 'macro_bearish'
    return view, None


def score_conviction(first_candle: Dict[str, Any], adx: Optional[float],
                      vix_pct_rank: Optional[float]) -> int:
    """Conviction 0..5. Baseline 3. Bump on high-energy candles + trend agreement."""
    conv = 3

    # First-candle range % — Tier 2.5
    rng = first_candle.get('range_pct') or 0
    if rng < 0.15:
        conv -= 1
    elif rng > 0.5:
        conv += 1

    # Wick position — confirming moves land at the extreme of the range
    wp = first_candle.get('wick_pos_pct')
    if wp is not None:
        body = first_candle.get('body')
        if body == 'bearish' and wp < 30:
            conv += 1          # body near low = strong bearish close
        elif body == 'bullish' and wp > 70:
            conv += 1          # body near high = strong bullish close
        elif wp is not None and 40 <= wp <= 60:
            conv -= 1          # indecisive, middle of range

    # ADX — Tier 1.3
    if adx is not None:
        if adx < ADX_CHOP_MAX:
            pass  # no bump — confirms neutral regimes, no conviction bonus
        elif ADX_TREND_MIN <= adx:
            conv += 1          # a trend is a trend: more conviction for directional views

    # VIX — Tier 1.4. No conviction bump; this is an instrument-choice modifier,
    # applied downstream. Keep conviction pure to view strength.

    return max(0, min(5, conv))


def adx_bucket(adx: Optional[float]) -> str:
    if adx is None:
        return 'unknown'
    if adx < ADX_CHOP_MAX:
        return 'chop'
    if adx < ADX_TREND_MIN:
        return 'building'
    return 'trend'


def detect_stacked_levels(weekly: Dict[str, float], daily: Dict[str, float],
                           spot: float) -> Dict[str, List[Dict[str, Any]]]:
    """Find weekly + daily pivot clusters within STACKED_LEVEL_DISTANCE_PCT of spot.

    Returns:
      {'supports': [{'price': ..., 'components': ['wS1', 'dS2']}, ...],
       'resistances': [...]}
    """
    if spot is None or spot <= 0:
        return {'supports': [], 'resistances': []}
    tol = spot * STACKED_LEVEL_DISTANCE_PCT / 100.0

    # Support candidates are pivots that sit BELOW current spot.
    weekly_supports = {f"w{k.upper()}": v for k, v in weekly.items()
                        if v is not None and v < spot and k in ('s1', 's2', 'pp')}
    daily_supports = {f"d{k.upper()}": v for k, v in daily.items()
                       if v is not None and v < spot and k in ('s1', 's2', 'pp')}
    weekly_resistances = {f"w{k.upper()}": v for k, v in weekly.items()
                           if v is not None and v > spot and k in ('r1', 'r2', 'pp')}
    daily_resistances = {f"d{k.upper()}": v for k, v in daily.items()
                          if v is not None and v > spot and k in ('r1', 'r2', 'pp')}

    clusters = {'supports': [], 'resistances': []}

    for bucket_name, weekly_set, daily_set in (
        ('supports', weekly_supports, daily_supports),
        ('resistances', weekly_resistances, daily_resistances),
    ):
        for wname, wval in weekly_set.items():
            for dname, dval in daily_set.items():
                if abs(wval - dval) <= tol:
                    price = (wval + dval) / 2.0
                    clusters[bucket_name].append({
                        'price': round(price, 2),
                        'components': [wname, dname],
                        'weekly_price': round(wval, 2),
                        'daily_price': round(dval, 2),
                    })
    # Sort supports descending (nearest-to-spot first), resistances ascending
    clusters['supports'].sort(key=lambda x: -x['price'])
    clusters['resistances'].sort(key=lambda x: x['price'])
    return clusters


def select_instrument(view: str, vix_pct_rank: Optional[float]) -> str:
    """Map view → instrument template. VIX regime tilts the choice."""
    if view == VIEW_IGNORE:
        return 'none'
    if view == VIEW_NEUTRAL:
        # VIX-rich environments strongly favor strangles (sell premium).
        # VIX-cheap environments tilt toward defined-risk iron condors —
        # out of scope for Phase 0, default back to strangle.
        return 'strangle'
    if view in (VIEW_NTB, VIEW_NTBULL):
        # Near-neutral directional — debit spread with potential OTM short overlay (Phase 1).
        return 'put_debit_spread' if view == VIEW_NTB else 'call_debit_spread'
    if view == VIEW_BEARISH:
        return 'put_debit_spread'
    if view == VIEW_BULLISH:
        return 'call_debit_spread'
    return 'none'


def compute_expected_range(view: str, pivots: Dict[str, float]) -> Tuple[Optional[float], Optional[float]]:
    """Return (low, high) of the expected price range for the week."""
    s2, s1, pp, r1, r2 = pivots.get('s2'), pivots.get('s1'), pivots.get('pp'), pivots.get('r1'), pivots.get('r2')
    if view == VIEW_NEUTRAL:
        # Default to symmetric S1..R1 on pure neutral
        return s1, r1
    if view == VIEW_NTB:
        return s2, r1
    if view == VIEW_NTBULL:
        return s1, r2
    if view == VIEW_BEARISH:
        return None, s1   # expected below S1; no upper
    if view == VIEW_BULLISH:
        return r1, None   # expected above R1
    return None, None


# ─── Core engine ────────────────────────────────────────────

class NwvEngine:
    """Phase 0 engine — compute the view, persist it, return it."""

    def __init__(self):
        self.db = get_nwv_db()

    def compute_view(
        self,
        *,
        week_start: date,
        weekly_state: Dict[str, Any],
        mon_open: float,
        prev_fri_close: float,
        first_candle: Dict[str, float],   # {'open','high','low','close','volume'}
        daily_pivots: Optional[Dict[str, float]] = None,
        vix_value: Optional[float] = None,
        vix_pct_rank: Optional[float] = None,
        adx_daily: Optional[float] = None,
        spot: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute a full view dict. Caller provides inputs; engine does the math.

        Returns a rich dict (persisted AND returned) with every intermediate
        so the dashboard can show the derivation step-by-step.
        """
        spot = spot or first_candle['close']
        weekly_pivots = {
            's2': weekly_state['pivot_s2'],
            's1': weekly_state['pivot_s1'],
            'pp': weekly_state['pivot_pp'],
            'r1': weekly_state['pivot_r1'],
            'r2': weekly_state['pivot_r2'],
        }
        cpr_tc = weekly_state['cpr_tc']
        cpr_bc = weekly_state['cpr_bc']
        cpr_bucket = weekly_state['cpr_bucket']

        # 1) First-candle classification
        fc = classify_first_candle(
            first_candle['open'], first_candle['high'],
            first_candle['low'], first_candle['close'],
            cpr_tc, cpr_bc,
        )

        # 2) Gap tier
        gap_pct = ((mon_open - prev_fri_close) / prev_fri_close * 100.0) if prev_fri_close else 0.0
        gap_tier, gap_direction = classify_gap(gap_pct)

        # 3) Base view from matrix
        base_view = base_view_matrix(cpr_bucket, fc['pos'], fc['body'])

        # 4) Gap dampener
        view_after_gap, gap_demoted = apply_gap_dampener(base_view, gap_tier, gap_direction)

        # 5) Monthly override (SOFT)
        view_after_monthly, monthly_override = apply_monthly_override(
            view_after_gap, spot,
            weekly_state.get('monthly_tc'),
            weekly_state.get('monthly_bc'),
        )

        final_view = view_after_monthly

        # 6) Conviction score
        conviction = score_conviction(fc, adx_daily, vix_pct_rank)

        # 7) Stacked levels (needs daily pivots — if absent, empty)
        clusters = detect_stacked_levels(
            weekly=weekly_pivots,
            daily=(daily_pivots or {}),
            spot=spot,
        )

        # 8) Instrument choice + expected range
        instrument = select_instrument(final_view, vix_pct_rank)
        exp_low, exp_high = compute_expected_range(final_view, weekly_pivots)

        # Persist
        view_row = {
            'week_start': week_start.isoformat() if isinstance(week_start, date) else str(week_start),
            'generated_at': datetime.now().isoformat(),
            'mon_open': round(mon_open, 2),
            'first_candle_open': round(first_candle['open'], 2),
            'first_candle_high': round(first_candle['high'], 2),
            'first_candle_low': round(first_candle['low'], 2),
            'first_candle_close': round(first_candle['close'], 2),
            'first_candle_volume': first_candle.get('volume'),
            'first_candle_body': fc['body'],
            'first_candle_pos': fc['pos'],
            'first_candle_range_pct': fc['range_pct'],
            'first_candle_wick_pos_pct': fc['wick_pos_pct'],
            'gap_pct': round(gap_pct, 4),
            'gap_tier': gap_tier,
            'gap_direction': gap_direction,
            'vix_value': round(vix_value, 2) if vix_value is not None else None,
            'vix_pct_rank': round(vix_pct_rank, 2) if vix_pct_rank is not None else None,
            'adx_daily': round(adx_daily, 2) if adx_daily is not None else None,
            'adx_bucket': adx_bucket(adx_daily),
            'monthly_override_side': monthly_override,
            'monthly_override_applied': 1 if monthly_override else 0,
            'stacked_supports': json.dumps(clusters['supports']),
            'stacked_resistances': json.dumps(clusters['resistances']),
            'base_view': base_view,
            'final_view': final_view,
            'conviction': conviction,
            'instrument_choice': instrument,
            'expected_range_low': round(exp_low, 2) if exp_low is not None else None,
            'expected_range_high': round(exp_high, 2) if exp_high is not None else None,
            'time_stop': 'Fri 15:15',
            'notes': f'gap_demoted={gap_demoted}, monthly_override={monthly_override}',
        }

        try:
            self.db.insert_view(view_row)
        except Exception as e:
            logger.error(f"[NWV] insert_view failed: {e}", exc_info=True)

        # Return the enriched dict (with cluster objects, not JSON strings)
        view_row['stacked_supports'] = clusters['supports']
        view_row['stacked_resistances'] = clusters['resistances']
        view_row['weekly_pivots'] = weekly_pivots
        view_row['cpr_bucket'] = cpr_bucket
        view_row['cpr_tc'] = cpr_tc
        view_row['cpr_bc'] = cpr_bc
        view_row['cpr_width_pct'] = weekly_state.get('cpr_width_pct')
        return view_row

    # ─── Weekly state computation ─────────────────────────────

    def build_weekly_state(
        self,
        *,
        week_start: date,
        prev_high: float,
        prev_low: float,
        prev_close: float,
        prev_fri_close: Optional[float] = None,
        spot_ref: Optional[float] = None,
        monthly: Optional[Dict[str, float]] = None,
        notes: str = '',
    ) -> Dict[str, Any]:
        """Compute + persist weekly state. Returns the row inserted.

        spot_ref is used to compute CPR width as a percent. Defaults to
        prev_close if not given (acceptable — CPR width changes slightly
        with Monday open but is stable enough).
        """
        cpr = compute_cpr(prev_high, prev_low, prev_close)
        pivots = compute_pivots(prev_high, prev_low, prev_close)

        ref = spot_ref or prev_close
        cpr_width_pct = abs(cpr['tc'] - cpr['bc']) / ref * 100.0 if ref else 0.0
        bucket = classify_cpr_width(cpr_width_pct)

        row = {
            'week_start': week_start.isoformat() if isinstance(week_start, date) else str(week_start),
            'prev_week_high': prev_high,
            'prev_week_low': prev_low,
            'prev_week_close': prev_close,
            'prev_fri_close': prev_fri_close,
            'pivot_pp': round(pivots['pp'], 2),
            'pivot_s1': round(pivots['s1'], 2),
            'pivot_s2': round(pivots['s2'], 2),
            'pivot_r1': round(pivots['r1'], 2),
            'pivot_r2': round(pivots['r2'], 2),
            'cpr_tc': round(cpr['tc'], 2),
            'cpr_bc': round(cpr['bc'], 2),
            'cpr_width_pct': round(cpr_width_pct, 4),
            'cpr_bucket': bucket,
            'monthly_tc': round(monthly['tc'], 2) if monthly and monthly.get('tc') else None,
            'monthly_bc': round(monthly['bc'], 2) if monthly and monthly.get('bc') else None,
            'monthly_pivot': round(monthly['pp'], 2) if monthly and monthly.get('pp') else None,
            'notes': notes or None,
        }
        self.db.upsert_weekly_state(row)
        return row


# ─── module-level singleton ─────────────────────────────────

_engine_singleton: Optional[NwvEngine] = None


def get_nwv_engine() -> NwvEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = NwvEngine()
    return _engine_singleton
