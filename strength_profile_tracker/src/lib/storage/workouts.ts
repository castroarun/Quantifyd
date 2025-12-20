import { Exercise, WorkoutSession, WorkoutSet, Level, Sex } from '@/types'
import { getExerciseById, calculateStrength } from '@/lib/calculations/strength'
import { addToSyncQueue } from './sync'

const WORKOUTS_KEY = 'strength_profile_workouts'

/**
 * PR Celebration Messages - 10 excited variations
 */
export const PR_CELEBRATIONS = [
  "🎉 NEW PR! {exercise}! You're crushing it!",
  "🔥 BOOM! New {exercise} record! Way to go!",
  "💪 YES! That's a new {exercise} PR! Beast mode!",
  "🏆 NEW PERSONAL BEST! {exercise}! You're unstoppable!",
  "⚡ INCREDIBLE! New {exercise} PR! Keep that energy!",
  "🚀 LIFT OFF! You just hit a new {exercise} record!",
  "👑 KING/QUEEN of {exercise}! New PR achieved!",
  "🌟 STELLAR! New {exercise} PR! The gains are real!",
  "💥 SMASHED IT! New {exercise} record! Legendary!",
  "🎊 CELEBRATION TIME! New {exercise} PR! You're on fire!"
]

/**
 * Get a random celebration message for a PR
 */
export function getCelebrationMessage(exerciseName: string): string {
  const index = Math.floor(Math.random() * PR_CELEBRATIONS.length)
  return PR_CELEBRATIONS[index].replace('{exercise}', exerciseName)
}

/**
 * Get level thresholds for an exercise based on user weight
 */
export function getLevelThresholds(
  exerciseId: Exercise,
  userWeight: number,
  sex?: Sex
): Record<Level, number> | null {
  const exercise = getExerciseById(exerciseId)
  if (!exercise) return null

  return {
    beginner: calculateStrength(userWeight, exercise, 'beginner', sex),
    novice: calculateStrength(userWeight, exercise, 'novice', sex),
    intermediate: calculateStrength(userWeight, exercise, 'intermediate', sex),
    advanced: calculateStrength(userWeight, exercise, 'advanced', sex)
  }
}

/**
 * Calculate what level a weight corresponds to
 */
export function getLevelForWeight(
  exerciseId: Exercise,
  userWeight: number,
  liftedWeight: number,
  sex?: Sex
): Level {
  const thresholds = getLevelThresholds(exerciseId, userWeight, sex)
  if (!thresholds) return 'beginner'

  if (liftedWeight >= thresholds.advanced) return 'advanced'
  if (liftedWeight >= thresholds.intermediate) return 'intermediate'
  if (liftedWeight >= thresholds.novice) return 'novice'
  return 'beginner'
}

/**
 * Check if workout PR triggers a level upgrade
 * Returns new level if upgrade, null if no change
 */
export function checkLevelUpgrade(
  exerciseId: Exercise,
  userWeight: number,
  newPR: number,
  currentLevel: Level | null,
  sex?: Sex
): Level | null {
  const achievedLevel = getLevelForWeight(exerciseId, userWeight, newPR, sex)

  const levelOrder: Level[] = ['beginner', 'novice', 'intermediate', 'advanced']
  const currentIndex = currentLevel ? levelOrder.indexOf(currentLevel) : -1
  const achievedIndex = levelOrder.indexOf(achievedLevel)

  return achievedIndex > currentIndex ? achievedLevel : null
}

/**
 * Check if user should be downgraded based on last N workouts
 * Returns new (lower) level if downgrade needed, null if no change
 *
 * Logic: If user hasn't hit their current level threshold in the last 4 workouts,
 * downgrade to the highest level they've actually achieved in those workouts.
 */
export function checkLevelDowngrade(
  profileId: string,
  exerciseId: Exercise,
  userWeight: number,
  currentLevel: Level | null,
  sex?: Sex,
  lookbackWorkouts: number = 4
): Level | null {
  if (!currentLevel || currentLevel === 'beginner') return null

  const thresholds = getLevelThresholds(exerciseId, userWeight, sex)
  if (!thresholds) return null

  // Get last N workout sessions for this exercise
  const sessions = getExerciseSessions(profileId, exerciseId, lookbackWorkouts)

  // Need at least 4 workouts to consider downgrade
  if (sessions.length < lookbackWorkouts) return null

  // Get max weight from each of the last N sessions
  const sessionMaxes = sessions.map(session => {
    return Math.max(...session.sets.map(set => set.weight || 0))
  })

  // Check if any session hit the current level threshold
  const currentThreshold = thresholds[currentLevel]
  const hitCurrentLevel = sessionMaxes.some(max => max >= currentThreshold)

  if (hitCurrentLevel) {
    // User maintained their level
    return null
  }

  // User didn't hit their level in last N workouts - find what level they actually achieved
  const overallMax = Math.max(...sessionMaxes)
  const achievedLevel = getLevelForWeight(exerciseId, userWeight, overallMax, sex)

  const levelOrder: Level[] = ['beginner', 'novice', 'intermediate', 'advanced']
  const currentIndex = levelOrder.indexOf(currentLevel)
  const achievedIndex = levelOrder.indexOf(achievedLevel)

  // Only return if it's actually a downgrade
  return achievedIndex < currentIndex ? achievedLevel : null
}

/**
 * Downgrade messages
 */
export const DOWNGRADE_MESSAGES = [
  "Level adjusted to {level}. Keep pushing to get back up!",
  "Dropped to {level}. You've got this - time to rebuild!",
  "Now at {level}. Every champion has setbacks. Come back stronger!",
  "Reset to {level}. Focus on form and the gains will follow!"
]

export function getDowngradeMessage(level: string): string {
  const index = Math.floor(Math.random() * DOWNGRADE_MESSAGES.length)
  return DOWNGRADE_MESSAGES[index].replace('{level}', level.charAt(0).toUpperCase() + level.slice(1))
}

/**
 * Get PR (max weight) from workout history for an exercise
 */
export function getExercisePR(profileId: string, exerciseId: Exercise): number {
  const sessions = getExerciseSessions(profileId, exerciseId, 100)
  let maxWeight = 0

  sessions.forEach(session => {
    session.sets.forEach(set => {
      if (set.weight && set.weight > maxWeight) {
        maxWeight = set.weight
      }
    })
  })

  return maxWeight
}

/**
 * Check if this is a new PR compared to history
 */
export function isNewPR(profileId: string, exerciseId: Exercise, newWeight: number): boolean {
  const allWorkouts = getAllWorkouts()
  const today = getTodayDate()

  // Get max weight from sessions excluding today
  let previousMax = 0
  allWorkouts
    .filter(w => w.profileId === profileId && w.exerciseId === exerciseId && w.date !== today)
    .forEach(session => {
      session.sets.forEach(set => {
        if (set.weight && set.weight > previousMax) {
          previousMax = set.weight
        }
      })
    })

  return newWeight > previousMax && previousMax > 0
}

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

  const isUpdate = existingIndex >= 0

  if (isUpdate) {
    allWorkouts[existingIndex] = session
  } else {
    allWorkouts.push(session)
  }

  saveWorkouts(allWorkouts)

  // Queue for cloud sync
  addToSyncQueue('workout', isUpdate ? 'update' : 'create', session)

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
