import {
  Exercise,
  Level,
  ExerciseStandards,
  CalculatedStrength,
  EXERCISE_NAMES,
  LEVEL_NAMES
} from '@/types'

// Body weight multipliers from PRD Section 3.2.3
export const STRENGTH_STANDARDS: ExerciseStandards = {
  benchPress: {
    beginner: 0.59,
    novice: 0.82,
    intermediate: 1.06,
    advanced: 1.29
  },
  squat: {
    beginner: 0.76,
    novice: 1.12,
    intermediate: 1.47,
    advanced: 1.88
  },
  deadlift: {
    beginner: 0.94,
    novice: 1.29,
    intermediate: 1.71,
    advanced: 2.18
  },
  shoulderPress: {
    beginner: 0.41,
    novice: 0.59,
    intermediate: 0.76,
    advanced: 1.00
  }
}

const LEVELS: Level[] = ['beginner', 'novice', 'intermediate', 'advanced']
const EXERCISES: Exercise[] = ['benchPress', 'squat', 'deadlift', 'shoulderPress']

/**
 * Calculate strength standard for a specific exercise and level
 * @param bodyWeight - User's body weight in kg
 * @param exercise - The exercise type
 * @param level - The difficulty level
 * @returns Calculated weight in kg (rounded to nearest integer)
 */
export function calculateStrength(
  bodyWeight: number,
  exercise: Exercise,
  level: Level
): number {
  const multiplier = STRENGTH_STANDARDS[exercise][level]
  return Math.round(bodyWeight * multiplier)
}

/**
 * Calculate all strength standards for a given body weight
 * @param bodyWeight - User's body weight in kg
 * @param selectedLevels - Currently selected levels for each exercise
 * @returns Array of calculated strengths for all exercises
 */
export function calculateAllStrengths(
  bodyWeight: number,
  selectedLevels: Record<Exercise, Level>
): CalculatedStrength[] {
  return EXERCISES.map(exercise => ({
    exercise,
    exerciseName: EXERCISE_NAMES[exercise],
    levels: LEVELS.map(level => ({
      level,
      weight: calculateStrength(bodyWeight, exercise, level),
      isSelected: selectedLevels[exercise] === level
    }))
  }))
}

/**
 * Get the display name for a level
 */
export function getLevelDisplayName(level: Level): string {
  return LEVEL_NAMES[level]
}

/**
 * Get all exercises
 */
export function getAllExercises(): Exercise[] {
  return [...EXERCISES]
}

/**
 * Get all levels
 */
export function getAllLevels(): Level[] {
  return [...LEVELS]
}
