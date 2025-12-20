/**
 * Warm-up set calculator
 * Calculates progressive warm-up sets based on target weight
 * Max 4 sets: Bar(10) → 50%(5) → 70%(3) → 85%(2)
 */

export interface WarmupSet {
  weight: number
  reps: number
  purpose: string
}

const BAR_WEIGHT = 20 // Standard Olympic bar

/**
 * Round weight to nearest 2.5kg (plate-friendly)
 */
function roundToPlate(weight: number): number {
  return Math.round(weight / 2.5) * 2.5
}

/**
 * Calculate warm-up sets for a target working weight
 * Only returns sets if target > 40kg (otherwise warm-up isn't meaningful)
 * Max 4 sets with decreasing reps: 10, 5, 3, 2
 */
export function calculateWarmupSets(targetWeight: number): WarmupSet[] | null {
  // Don't suggest warm-up for light weights
  if (targetWeight <= 40) return null

  const sets: WarmupSet[] = []

  // Set 1: Empty bar (only for heavier lifts)
  if (targetWeight >= 60) {
    sets.push({
      weight: BAR_WEIGHT,
      reps: 10,
      purpose: 'Mobility'
    })
  }

  // Set 2: ~50% (only if meaningful gap from bar)
  const fiftyPercent = roundToPlate(targetWeight * 0.5)
  if (fiftyPercent > BAR_WEIGHT + 10 && fiftyPercent < targetWeight * 0.65) {
    sets.push({
      weight: fiftyPercent,
      reps: 5,
      purpose: 'Groove'
    })
  }

  // Set 3: ~70%
  const seventyPercent = roundToPlate(targetWeight * 0.7)
  if (seventyPercent > BAR_WEIGHT && seventyPercent < targetWeight * 0.82) {
    sets.push({
      weight: seventyPercent,
      reps: 3,
      purpose: 'Activate'
    })
  }

  // Set 4: ~85% (final warm-up, only for heavy lifts)
  const eightyFivePercent = roundToPlate(targetWeight * 0.85)
  if (eightyFivePercent > BAR_WEIGHT && eightyFivePercent < targetWeight - 5) {
    sets.push({
      weight: eightyFivePercent,
      reps: 2,
      purpose: 'Prime'
    })
  }

  // Need at least 2 sets to be meaningful
  return sets.length >= 2 ? sets : null
}

/**
 * Calculate estimated warm-up time in minutes
 */
export function estimateWarmupTime(sets: WarmupSet[]): number {
  return sets.length
}

/**
 * Format warm-up sets for display
 */
export function formatWarmupPlan(sets: WarmupSet[]): string {
  return sets
    .map((set, i) => `${set.weight}kg×${set.reps}`)
    .join(' → ')
}
