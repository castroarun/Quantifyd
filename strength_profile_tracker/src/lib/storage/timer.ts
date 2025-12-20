import { Exercise, TimerSettings, ExerciseTimerHistory, DEFAULT_TIMER_SETTINGS } from '@/types'

const TIMER_SETTINGS_KEY = 'strength_profile_timer_settings'
const TIMER_HISTORY_KEY = 'strength_profile_timer_history'

/**
 * Get timer settings from localStorage
 */
export function getTimerSettings(): TimerSettings {
  if (typeof window === 'undefined') return DEFAULT_TIMER_SETTINGS

  const stored = localStorage.getItem(TIMER_SETTINGS_KEY)
  if (!stored) return DEFAULT_TIMER_SETTINGS

  try {
    return { ...DEFAULT_TIMER_SETTINGS, ...JSON.parse(stored) }
  } catch {
    return DEFAULT_TIMER_SETTINGS
  }
}

/**
 * Save timer settings to localStorage
 */
export function saveTimerSettings(settings: Partial<TimerSettings>): TimerSettings {
  if (typeof window === 'undefined') return DEFAULT_TIMER_SETTINGS

  const current = getTimerSettings()
  const updated = { ...current, ...settings }
  localStorage.setItem(TIMER_SETTINGS_KEY, JSON.stringify(updated))
  return updated
}

/**
 * Get all exercise timer history
 */
function getTimerHistory(): ExerciseTimerHistory[] {
  if (typeof window === 'undefined') return []

  const stored = localStorage.getItem(TIMER_HISTORY_KEY)
  if (!stored) return []

  try {
    return JSON.parse(stored)
  } catch {
    return []
  }
}

/**
 * Save timer history
 */
function saveTimerHistory(history: ExerciseTimerHistory[]): void {
  if (typeof window === 'undefined') return
  localStorage.setItem(TIMER_HISTORY_KEY, JSON.stringify(history))
}

/**
 * Get last used timer duration for an exercise
 * Returns default duration if no history
 */
export function getExerciseTimerDuration(exerciseId: Exercise): number {
  const history = getTimerHistory()
  const entry = history.find(h => h.exerciseId === exerciseId)

  if (entry) {
    return entry.lastDuration
  }

  return getTimerSettings().defaultDuration
}

/**
 * Save the last used timer duration for an exercise
 */
export function saveExerciseTimerDuration(exerciseId: Exercise, duration: number): void {
  const history = getTimerHistory()
  const existingIndex = history.findIndex(h => h.exerciseId === exerciseId)

  const entry: ExerciseTimerHistory = {
    exerciseId,
    lastDuration: duration
  }

  if (existingIndex >= 0) {
    history[existingIndex] = entry
  } else {
    history.push(entry)
  }

  saveTimerHistory(history)
}

/**
 * Reset all timer history (used in settings)
 */
export function resetTimerHistory(): void {
  if (typeof window === 'undefined') return
  localStorage.removeItem(TIMER_HISTORY_KEY)
}

/**
 * Reset timer settings to defaults
 */
export function resetTimerSettings(): TimerSettings {
  if (typeof window === 'undefined') return DEFAULT_TIMER_SETTINGS
  localStorage.removeItem(TIMER_SETTINGS_KEY)
  return DEFAULT_TIMER_SETTINGS
}

/**
 * Format seconds to MM:SS display
 */
export function formatTime(seconds: number): string {
  const mins = Math.floor(Math.abs(seconds) / 60)
  const secs = Math.abs(seconds) % 60
  const sign = seconds < 0 ? '-' : ''
  return `${sign}${mins}:${secs.toString().padStart(2, '0')}`
}
