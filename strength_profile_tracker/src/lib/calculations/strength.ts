import {
  Exercise,
  Level,
  BodyPart,
  ExerciseInfo,
  ExerciseRatings,
  CalculatedStrength,
  ALL_LEVELS,
  LEVEL_VALUES
} from '@/types'

// All exercises with their multipliers from DEVELOPMENT-PLAN v2.0
export const EXERCISES: ExerciseInfo[] = [
  // CHEST
  { id: 'benchPress', name: 'Bench Press', bodyPart: 'chest', multipliers: { beginner: 0.59, novice: 0.82, intermediate: 1.06, advanced: 1.29 } },
  { id: 'inclineBench', name: 'Incline Bench', bodyPart: 'chest', multipliers: { beginner: 0.50, novice: 0.70, intermediate: 0.90, advanced: 1.10 } },
  { id: 'dumbbellPress', name: 'Dumbbell Press', bodyPart: 'chest', multipliers: { beginner: 0.25, novice: 0.35, intermediate: 0.45, advanced: 0.55 }, isDumbbell: true },
  { id: 'cableFly', name: 'Cable Fly', bodyPart: 'chest', multipliers: { beginner: 0.15, novice: 0.22, intermediate: 0.30, advanced: 0.40 } },

  // BACK
  { id: 'deadlift', name: 'Deadlift', bodyPart: 'back', multipliers: { beginner: 0.94, novice: 1.29, intermediate: 1.71, advanced: 2.18 } },
  { id: 'barbellRow', name: 'Barbell Row', bodyPart: 'back', multipliers: { beginner: 0.50, novice: 0.70, intermediate: 0.90, advanced: 1.15 } },
  { id: 'latPulldown', name: 'Lat Pulldown', bodyPart: 'back', multipliers: { beginner: 0.50, novice: 0.70, intermediate: 0.90, advanced: 1.10 } },
  { id: 'pullUps', name: 'Pull-ups', bodyPart: 'back', multipliers: { beginner: 0.50, novice: 0.80, intermediate: 1.00, advanced: 1.30 } },
  { id: 'cableRow', name: 'Cable Row', bodyPart: 'back', multipliers: { beginner: 0.50, novice: 0.70, intermediate: 0.90, advanced: 1.10 } },

  // SHOULDERS
  { id: 'shoulderPressBarbell', name: 'Shoulder Press (Barbell)', bodyPart: 'shoulders', multipliers: { beginner: 0.41, novice: 0.59, intermediate: 0.76, advanced: 1.00 } },
  { id: 'shoulderPressMachine', name: 'Shoulder Press (Machine)', bodyPart: 'shoulders', multipliers: { beginner: 0.35, novice: 0.50, intermediate: 0.65, advanced: 0.85 } },
  { id: 'shoulderPressDumbbell', name: 'Shoulder Press (Dumbbell)', bodyPart: 'shoulders', multipliers: { beginner: 0.18, novice: 0.26, intermediate: 0.35, advanced: 0.45 }, isDumbbell: true },
  { id: 'sideLateralDumbbell', name: 'Side Lateral (Dumbbell)', bodyPart: 'shoulders', multipliers: { beginner: 0.08, novice: 0.12, intermediate: 0.16, advanced: 0.22 }, isDumbbell: true },
  { id: 'sideLateralCable', name: 'Side Lateral (Cable)', bodyPart: 'shoulders', multipliers: { beginner: 0.06, novice: 0.10, intermediate: 0.14, advanced: 0.20 } },
  { id: 'frontRaise', name: 'Front Raise', bodyPart: 'shoulders', multipliers: { beginner: 0.10, novice: 0.15, intermediate: 0.20, advanced: 0.28 } },

  // LEGS
  { id: 'squat', name: 'Squat', bodyPart: 'legs', multipliers: { beginner: 0.76, novice: 1.12, intermediate: 1.47, advanced: 1.88 } },
  { id: 'legPress', name: 'Leg Press', bodyPart: 'legs', multipliers: { beginner: 1.50, novice: 2.20, intermediate: 3.00, advanced: 3.80 } },
  { id: 'romanianDeadlift', name: 'Romanian Deadlift', bodyPart: 'legs', multipliers: { beginner: 0.60, novice: 0.85, intermediate: 1.10, advanced: 1.40 } },
  { id: 'legCurl', name: 'Leg Curl', bodyPart: 'legs', multipliers: { beginner: 0.30, novice: 0.45, intermediate: 0.60, advanced: 0.80 } },
  { id: 'legExtension', name: 'Leg Extension', bodyPart: 'legs', multipliers: { beginner: 0.40, novice: 0.55, intermediate: 0.75, advanced: 0.95 } },
  { id: 'calfRaise', name: 'Calf Raise', bodyPart: 'legs', multipliers: { beginner: 0.80, novice: 1.20, intermediate: 1.60, advanced: 2.00 } },

  // ARMS
  { id: 'bicepCurlBarbell', name: 'Bicep Curl (Barbell)', bodyPart: 'arms', multipliers: { beginner: 0.25, novice: 0.38, intermediate: 0.50, advanced: 0.65 } },
  { id: 'bicepCurlDumbbell', name: 'Bicep Curl (Dumbbell)', bodyPart: 'arms', multipliers: { beginner: 0.12, novice: 0.18, intermediate: 0.25, advanced: 0.32 }, isDumbbell: true },
  { id: 'tricepPushdown', name: 'Tricep Pushdown', bodyPart: 'arms', multipliers: { beginner: 0.25, novice: 0.38, intermediate: 0.50, advanced: 0.65 } },
  { id: 'skullCrushers', name: 'Skull Crushers', bodyPart: 'arms', multipliers: { beginner: 0.20, novice: 0.30, intermediate: 0.40, advanced: 0.50 } },
]

/**
 * Get exercise info by ID
 */
export function getExerciseById(exerciseId: Exercise): ExerciseInfo | undefined {
  return EXERCISES.find(e => e.id === exerciseId)
}

/**
 * Get exercises by body part
 */
export function getExercisesByBodyPart(bodyPart: BodyPart): ExerciseInfo[] {
  return EXERCISES.filter(e => e.bodyPart === bodyPart)
}

/**
 * Calculate strength standard for a specific exercise and level
 */
export function calculateStrength(
  bodyWeight: number,
  exercise: ExerciseInfo,
  level: Level
): number {
  const multiplier = exercise.multipliers[level]
  return Math.round(bodyWeight * multiplier)
}

/**
 * Calculate strength for a single exercise with all levels
 */
export function calculateExerciseStrength(
  bodyWeight: number,
  exercise: ExerciseInfo,
  selectedLevel?: Level
): CalculatedStrength {
  return {
    exercise: exercise.id,
    exerciseName: exercise.name,
    bodyPart: exercise.bodyPart,
    isDumbbell: exercise.isDumbbell,
    levels: ALL_LEVELS.map(level => ({
      level,
      weight: calculateStrength(bodyWeight, exercise, level),
      isSelected: selectedLevel === level
    }))
  }
}

/**
 * Calculate all strengths, optionally filtered by body part
 */
export function calculateAllStrengths(
  bodyWeight: number,
  ratings: ExerciseRatings,
  bodyPartFilter?: BodyPart | 'all'
): CalculatedStrength[] {
  let exercises = EXERCISES

  if (bodyPartFilter && bodyPartFilter !== 'all') {
    exercises = exercises.filter(e => e.bodyPart === bodyPartFilter)
  }

  return exercises.map(exercise =>
    calculateExerciseStrength(bodyWeight, exercise, ratings[exercise.id])
  )
}

/**
 * Get rated exercises (user has selected a level)
 */
export function getRatedExercises(
  bodyWeight: number,
  ratings: ExerciseRatings,
  bodyPartFilter?: BodyPart | 'all'
): CalculatedStrength[] {
  return calculateAllStrengths(bodyWeight, ratings, bodyPartFilter)
    .filter(s => s.levels.some(l => l.isSelected))
}

/**
 * Get unrated exercises (user hasn't selected a level yet)
 */
export function getUnratedExercises(
  bodyWeight: number,
  ratings: ExerciseRatings,
  bodyPartFilter?: BodyPart | 'all'
): CalculatedStrength[] {
  return calculateAllStrengths(bodyWeight, ratings, bodyPartFilter)
    .filter(s => !s.levels.some(l => l.isSelected))
}

/**
 * Calculate overall profile level based on rated exercises
 * Returns the average level rounded to nearest level
 */
export function calculateOverallLevel(ratings: ExerciseRatings): Level | null {
  const ratedLevels = Object.values(ratings).filter((l): l is Level => l !== undefined)

  if (ratedLevels.length === 0) return null

  const sum = ratedLevels.reduce((acc, level) => acc + LEVEL_VALUES[level], 0)
  const average = sum / ratedLevels.length

  // Convert average back to level
  if (average < 1.5) return 'beginner'
  if (average < 2.5) return 'novice'
  if (average < 3.5) return 'intermediate'
  return 'advanced'
}

/**
 * Get count of rated exercises
 */
export function getRatedCount(ratings: ExerciseRatings): number {
  return Object.values(ratings).filter(l => l !== undefined).length
}

/**
 * Get total exercises count
 */
export function getTotalExercisesCount(): number {
  return EXERCISES.length
}
