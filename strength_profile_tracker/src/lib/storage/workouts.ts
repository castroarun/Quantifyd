import { Exercise, WorkoutSession, WorkoutSet } from '@/types'

const WORKOUTS_KEY = 'strength_profile_workouts'

/**
 * Get all workout sessions from localStorage
 */
export function getAllWorkouts(): WorkoutSession[] {
  if (typeof window === 'undefined') return []

  const stored = localStorage.getItem(WORKOUTS_KEY)
  if (!stored) return []

  try {
    return JSON.parse(stored)
  } catch {
    return []
  }
}

/**
 * Save all workout sessions to localStorage
 */
function saveWorkouts(workouts: WorkoutSession[]): void {
  if (typeof window === 'undefined') return
  localStorage.setItem(WORKOUTS_KEY, JSON.stringify(workouts))
}

/**
 * Get workout sessions for a specific exercise and profile
 * Returns the last N sessions, sorted by date (most recent first)
 */
export function getExerciseSessions(
  profileId: string,
  exerciseId: Exercise,
  limit: number = 3
): WorkoutSession[] {
  const allWorkouts = getAllWorkouts()

  return allWorkouts
    .filter(w => w.profileId === profileId && w.exerciseId === exerciseId)
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
    .slice(0, limit)
}

/**
 * Get today's date in ISO format (YYYY-MM-DD)
 */
export function getTodayDate(): string {
  return new Date().toISOString().split('T')[0]
}

/**
 * Get or create today's workout session for an exercise
 */
export function getTodaySession(
  profileId: string,
  exerciseId: Exercise
): WorkoutSession | null {
  const today = getTodayDate()
  const allWorkouts = getAllWorkouts()

  return allWorkouts.find(
    w => w.profileId === profileId && w.exerciseId === exerciseId && w.date === today
  ) || null
}

/**
 * Save or update today's workout session
 */
export function saveWorkoutSession(
  profileId: string,
  exerciseId: Exercise,
  sets: WorkoutSet[]
): WorkoutSession {
  const allWorkouts = getAllWorkouts()
  const today = getTodayDate()

  // Find existing session for today
  const existingIndex = allWorkouts.findIndex(
    w => w.profileId === profileId && w.exerciseId === exerciseId && w.date === today
  )

  const session: WorkoutSession = {
    id: existingIndex >= 0 ? allWorkouts[existingIndex].id : crypto.randomUUID(),
    date: today,
    exerciseId,
    profileId,
    sets
  }

  if (existingIndex >= 0) {
    allWorkouts[existingIndex] = session
  } else {
    allWorkouts.push(session)
  }

  saveWorkouts(allWorkouts)
  return session
}

/**
 * Format date for display (e.g., "Dec 5" or "Today")
 */
export function formatSessionDate(dateStr: string): string {
  const today = getTodayDate()
  if (dateStr === today) return 'TODAY'

  const date = new Date(dateStr)
  const month = date.toLocaleString('en', { month: 'short' })
  const day = date.getDate()
  return `${month} ${day}`
}

/**
 * Create empty sets array (3 sets)
 */
export function createEmptySets(): WorkoutSet[] {
  return [
    { weight: null, reps: null },
    { weight: null, reps: null },
    { weight: null, reps: null }
  ]
}
