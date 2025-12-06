// Profile types based on PRD Section 5.2

export type Level = 'beginner' | 'novice' | 'intermediate' | 'advanced'

export type Exercise = 'benchPress' | 'squat' | 'deadlift' | 'shoulderPress'

export interface ExerciseLevels {
  benchPress: Level
  squat: Level
  deadlift: Level
  shoulderPress: Level
}

export interface Profile {
  id: string
  name: string                  // max 50 characters
  age: number                   // 13-100
  height: number                // 100-250 cm
  weight: number                // 30-300 kg
  currentLevels: ExerciseLevels
  createdAt: string             // ISO date string
  updatedAt: string             // ISO date string
}

// Strength standard multipliers from PRD Section 3.2.3
export interface StrengthMultipliers {
  beginner: number
  novice: number
  intermediate: number
  advanced: number
}

export interface ExerciseStandards {
  benchPress: StrengthMultipliers
  squat: StrengthMultipliers
  deadlift: StrengthMultipliers
  shoulderPress: StrengthMultipliers
}

// Calculated strength values for display
export interface CalculatedStrength {
  exercise: Exercise
  exerciseName: string
  levels: {
    level: Level
    weight: number
    isSelected: boolean
  }[]
}

// Validation constraints from PRD Section 3.1.1
export const VALIDATION = {
  name: { min: 1, max: 50 },
  age: { min: 13, max: 100 },
  height: { min: 100, max: 250 },
  weight: { min: 30, max: 300 },
  maxProfiles: 5
} as const

// Level colors from PRD Section 4.1
export const LEVEL_COLORS: Record<Level, string> = {
  beginner: '#2ECC71',      // Green
  novice: '#3498DB',        // Blue
  intermediate: '#F39C12',  // Orange
  advanced: '#E74C3C'       // Coral red
} as const

// Exercise display names
export const EXERCISE_NAMES: Record<Exercise, string> = {
  benchPress: 'Bench Press',
  squat: 'Squat',
  deadlift: 'Deadlift',
  shoulderPress: 'Shoulder Press'
} as const

// Level display names
export const LEVEL_NAMES: Record<Level, string> = {
  beginner: 'BEG',
  novice: 'NOV',
  intermediate: 'INT',
  advanced: 'ADV'
} as const
