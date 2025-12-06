import { Profile, WorkoutSession } from '@/types'

const PROFILES_KEY = 'strength_profiles_v2'
const WORKOUTS_KEY = 'strength_profile_workouts'
const SCORE_HISTORY_KEY = 'strength_score_history'

export interface ScoreHistoryEntry {
  date: string
  score: number
}

/**
 * Sample profiles with exercise ratings
 */
const sampleProfiles: Profile[] = [
  {
    id: 'sample_alex_001',
    name: 'Alex',
    age: 28,
    height: 178,
    weight: 82,
    exerciseRatings: {
      benchPress: 'intermediate',
      inclineBench: 'novice',
      deadlift: 'intermediate',
      barbellRow: 'novice',
      pullUps: 'intermediate',
      squat: 'intermediate',
      legPress: 'advanced',
      shoulderPressBarbell: 'novice',
      sideLateralDumbbell: 'beginner',
      bicepCurlBarbell: 'intermediate',
      tricepPushdown: 'novice',
    },
    createdAt: '2024-11-01T10:00:00Z',
    updatedAt: new Date().toISOString()
  },
  {
    id: 'sample_jordan_002',
    name: 'Jordan',
    age: 32,
    height: 165,
    weight: 68,
    exerciseRatings: {
      benchPress: 'novice',
      deadlift: 'advanced',
      squat: 'advanced',
      legPress: 'advanced',
      romanianDeadlift: 'intermediate',
      pullUps: 'novice',
      barbellRow: 'intermediate',
      legCurl: 'advanced',
    },
    createdAt: '2024-10-15T08:00:00Z',
    updatedAt: new Date().toISOString()
  }
]

/**
 * Generate workout sessions for the past few weeks
 */
function generateSampleWorkouts(): WorkoutSession[] {
  const workouts: WorkoutSession[] = []
  const today = new Date()

  // Alex's workout history - bench press progression
  const alexBenchDates = [-21, -17, -14, -10, -7, -3, 0]
  const alexBenchWeights = [70, 72.5, 72.5, 75, 75, 77.5, 80]

  alexBenchDates.forEach((daysAgo, i) => {
    const date = new Date(today)
    date.setDate(date.getDate() + daysAgo)
    workouts.push({
      id: `workout_alex_bench_${i}`,
      date: date.toISOString().split('T')[0],
      exerciseId: 'benchPress',
      profileId: 'sample_alex_001',
      sets: [
        { weight: alexBenchWeights[i], reps: 8 },
        { weight: alexBenchWeights[i], reps: 7 },
        { weight: alexBenchWeights[i] - 5, reps: 10 }
      ]
    })
  })

  // Alex's squat progression
  const alexSquatDates = [-20, -15, -11, -6, -2]
  const alexSquatWeights = [90, 95, 97.5, 100, 105]

  alexSquatDates.forEach((daysAgo, i) => {
    const date = new Date(today)
    date.setDate(date.getDate() + daysAgo)
    workouts.push({
      id: `workout_alex_squat_${i}`,
      date: date.toISOString().split('T')[0],
      exerciseId: 'squat',
      profileId: 'sample_alex_001',
      sets: [
        { weight: alexSquatWeights[i], reps: 6 },
        { weight: alexSquatWeights[i], reps: 5 },
        { weight: alexSquatWeights[i] - 10, reps: 8 }
      ]
    })
  })

  // Alex's deadlift
  const alexDeadliftDates = [-19, -12, -5]
  const alexDeadliftWeights = [120, 125, 130]

  alexDeadliftDates.forEach((daysAgo, i) => {
    const date = new Date(today)
    date.setDate(date.getDate() + daysAgo)
    workouts.push({
      id: `workout_alex_deadlift_${i}`,
      date: date.toISOString().split('T')[0],
      exerciseId: 'deadlift',
      profileId: 'sample_alex_001',
      sets: [
        { weight: alexDeadliftWeights[i], reps: 5 },
        { weight: alexDeadliftWeights[i], reps: 4 },
        { weight: alexDeadliftWeights[i] - 10, reps: 6 }
      ]
    })
  })

  // Jordan's workout history - squat progression (strong legs)
  const jordanSquatDates = [-18, -14, -10, -7, -3, -1]
  const jordanSquatWeights = [100, 105, 107.5, 110, 112.5, 115]

  jordanSquatDates.forEach((daysAgo, i) => {
    const date = new Date(today)
    date.setDate(date.getDate() + daysAgo)
    workouts.push({
      id: `workout_jordan_squat_${i}`,
      date: date.toISOString().split('T')[0],
      exerciseId: 'squat',
      profileId: 'sample_jordan_002',
      sets: [
        { weight: jordanSquatWeights[i], reps: 6 },
        { weight: jordanSquatWeights[i], reps: 5 },
        { weight: jordanSquatWeights[i], reps: 5 }
      ]
    })
  })

  // Jordan's deadlift progression
  const jordanDeadliftDates = [-16, -12, -8, -4, 0]
  const jordanDeadliftWeights = [130, 135, 140, 142.5, 145]

  jordanDeadliftDates.forEach((daysAgo, i) => {
    const date = new Date(today)
    date.setDate(date.getDate() + daysAgo)
    workouts.push({
      id: `workout_jordan_deadlift_${i}`,
      date: date.toISOString().split('T')[0],
      exerciseId: 'deadlift',
      profileId: 'sample_jordan_002',
      sets: [
        { weight: jordanDeadliftWeights[i], reps: 5 },
        { weight: jordanDeadliftWeights[i], reps: 4 },
        { weight: jordanDeadliftWeights[i] - 10, reps: 6 }
      ]
    })
  })

  // Jordan's leg press
  const jordanLegPressDates = [-15, -11, -6, -2]
  const jordanLegPressWeights = [180, 190, 200, 210]

  jordanLegPressDates.forEach((daysAgo, i) => {
    const date = new Date(today)
    date.setDate(date.getDate() + daysAgo)
    workouts.push({
      id: `workout_jordan_legpress_${i}`,
      date: date.toISOString().split('T')[0],
      exerciseId: 'legPress',
      profileId: 'sample_jordan_002',
      sets: [
        { weight: jordanLegPressWeights[i], reps: 10 },
        { weight: jordanLegPressWeights[i], reps: 8 },
        { weight: jordanLegPressWeights[i], reps: 8 }
      ]
    })
  })

  return workouts
}

/**
 * Generate sample score history for profiles
 */
function generateSampleScoreHistory(): Record<string, ScoreHistoryEntry[]> {
  const today = new Date()
  const history: Record<string, ScoreHistoryEntry[]> = {}

  // Alex's score history - steady improvement
  const alexScores = [42, 44, 45, 48, 50, 52, 55, 57]
  history['sample_alex_001'] = alexScores.map((score, i) => {
    const date = new Date(today)
    date.setDate(date.getDate() - (alexScores.length - 1 - i) * 3)
    return {
      date: date.toISOString().split('T')[0],
      score
    }
  })

  // Jordan's score history - strong start, consistent gains
  const jordanScores = [58, 60, 61, 63, 65, 67]
  history['sample_jordan_002'] = jordanScores.map((score, i) => {
    const date = new Date(today)
    date.setDate(date.getDate() - (jordanScores.length - 1 - i) * 4)
    return {
      date: date.toISOString().split('T')[0],
      score
    }
  })

  return history
}

/**
 * Get score history for a profile
 */
export function getScoreHistory(profileId: string): ScoreHistoryEntry[] {
  if (typeof window === 'undefined') return []

  try {
    const data = localStorage.getItem(SCORE_HISTORY_KEY)
    if (!data) return []
    const history = JSON.parse(data)
    return history[profileId] || []
  } catch {
    return []
  }
}

/**
 * Add a score entry to history
 */
export function addScoreToHistory(profileId: string, score: number): void {
  if (typeof window === 'undefined') return

  const today = new Date().toISOString().split('T')[0]
  const allHistory: Record<string, ScoreHistoryEntry[]> = JSON.parse(
    localStorage.getItem(SCORE_HISTORY_KEY) || '{}'
  )

  if (!allHistory[profileId]) {
    allHistory[profileId] = []
  }

  // Only add if it's a new day or score changed
  const lastEntry = allHistory[profileId][allHistory[profileId].length - 1]
  if (!lastEntry || lastEntry.date !== today || lastEntry.score !== score) {
    // Update today's entry or add new
    if (lastEntry && lastEntry.date === today) {
      lastEntry.score = score
    } else {
      allHistory[profileId].push({ date: today, score })
    }
    localStorage.setItem(SCORE_HISTORY_KEY, JSON.stringify(allHistory))
  }
}

/**
 * Check if sample data already exists
 */
export function hasSampleData(): boolean {
  if (typeof window === 'undefined') return false

  const profiles = localStorage.getItem(PROFILES_KEY)
  if (!profiles) return false

  try {
    const parsed = JSON.parse(profiles)
    return parsed.some((p: Profile) => p.id.startsWith('sample_'))
  } catch {
    return false
  }
}

/**
 * Load sample profiles and workout data into localStorage
 */
export function loadSampleData(): void {
  if (typeof window === 'undefined') return

  // Get existing data
  const existingProfiles: Profile[] = JSON.parse(localStorage.getItem(PROFILES_KEY) || '[]')
  const existingWorkouts: WorkoutSession[] = JSON.parse(localStorage.getItem(WORKOUTS_KEY) || '[]')
  const existingScoreHistory: Record<string, ScoreHistoryEntry[]> = JSON.parse(
    localStorage.getItem(SCORE_HISTORY_KEY) || '{}'
  )

  // Remove any existing sample data first
  const filteredProfiles = existingProfiles.filter(p => !p.id.startsWith('sample_'))
  const filteredWorkouts = existingWorkouts.filter(w => !w.profileId.startsWith('sample_'))
  const filteredScoreHistory: Record<string, ScoreHistoryEntry[]> = {}
  Object.keys(existingScoreHistory).forEach(key => {
    if (!key.startsWith('sample_')) {
      filteredScoreHistory[key] = existingScoreHistory[key]
    }
  })

  // Add sample data
  const newProfiles = [...filteredProfiles, ...sampleProfiles]
  const newWorkouts = [...filteredWorkouts, ...generateSampleWorkouts()]
  const newScoreHistory = { ...filteredScoreHistory, ...generateSampleScoreHistory() }

  // Save to localStorage
  localStorage.setItem(PROFILES_KEY, JSON.stringify(newProfiles))
  localStorage.setItem(WORKOUTS_KEY, JSON.stringify(newWorkouts))
  localStorage.setItem(SCORE_HISTORY_KEY, JSON.stringify(newScoreHistory))
}

/**
 * Remove sample data from localStorage
 */
export function removeSampleData(): void {
  if (typeof window === 'undefined') return

  const existingProfiles: Profile[] = JSON.parse(localStorage.getItem(PROFILES_KEY) || '[]')
  const existingWorkouts: WorkoutSession[] = JSON.parse(localStorage.getItem(WORKOUTS_KEY) || '[]')
  const existingScoreHistory: Record<string, ScoreHistoryEntry[]> = JSON.parse(
    localStorage.getItem(SCORE_HISTORY_KEY) || '{}'
  )

  const filteredProfiles = existingProfiles.filter(p => !p.id.startsWith('sample_'))
  const filteredWorkouts = existingWorkouts.filter(w => !w.profileId.startsWith('sample_'))
  const filteredScoreHistory: Record<string, ScoreHistoryEntry[]> = {}
  Object.keys(existingScoreHistory).forEach(key => {
    if (!key.startsWith('sample_')) {
      filteredScoreHistory[key] = existingScoreHistory[key]
    }
  })

  localStorage.setItem(PROFILES_KEY, JSON.stringify(filteredProfiles))
  localStorage.setItem(WORKOUTS_KEY, JSON.stringify(filteredWorkouts))
  localStorage.setItem(SCORE_HISTORY_KEY, JSON.stringify(filteredScoreHistory))
}
