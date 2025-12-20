import { WeightUnit, KG_TO_LBS, LBS_TO_KG } from '@/types'

const STORAGE_KEY = 'strength-tracker-unit-preference'

/**
 * Convert kg to lbs
 */
export function kgToLbs(kg: number): number {
  return Math.round(kg * KG_TO_LBS * 10) / 10
}

/**
 * Convert lbs to kg
 */
export function lbsToKg(lbs: number): number {
  return Math.round(lbs * LBS_TO_KG * 10) / 10
}

/**
 * Format weight with unit
 */
export function formatWeight(weightKg: number, unit: WeightUnit): string {
  if (unit === 'lbs') {
    return `${kgToLbs(weightKg)} lbs`
  }
  return `${weightKg} kg`
}

/**
 * Format weight number only (no unit suffix)
 */
export function formatWeightValue(weightKg: number, unit: WeightUnit): number {
  if (unit === 'lbs') {
    return kgToLbs(weightKg)
  }
  return weightKg
}

/**
 * Convert weight from display unit to storage unit (kg)
 */
export function convertToKg(value: number, unit: WeightUnit): number {
  if (unit === 'lbs') {
    return lbsToKg(value)
  }
  return value
}

/**
 * Get unit preference from localStorage
 */
export function getUnitPreference(): WeightUnit {
  if (typeof window === 'undefined') return 'kg'
  const stored = localStorage.getItem(STORAGE_KEY)
  if (stored === 'lbs') return 'lbs'
  return 'kg'
}

/**
 * Save unit preference to localStorage
 */
export function setUnitPreference(unit: WeightUnit): void {
  if (typeof window === 'undefined') return
  localStorage.setItem(STORAGE_KEY, unit)
}

/**
 * Muted, professional colors for profile avatars
 * Not too bright, suitable for both light and dark modes
 */
const PROFILE_COLORS = [
  '#5B7B9D', // Steel blue
  '#7B6B8D', // Muted purple
  '#6B8E7B', // Sage green
  '#9D7B6B', // Warm taupe
  '#6B8D9D', // Ocean teal
  '#8D7B6B', // Mocha
  '#7B8D6B', // Olive
  '#6B7B8D', // Slate
]

/**
 * Generate a consistent color for a profile based on its ID
 */
export function getProfileColor(profileId: string): string {
  const hash = profileId.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0)
  return PROFILE_COLORS[hash % PROFILE_COLORS.length]
}
