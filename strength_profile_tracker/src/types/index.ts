// Profile types based on DEVELOPMENT-PLAN v2.0

export type Level = 'beginner' | 'novice' | 'intermediate' | 'advanced'

export type BodyPart = 'chest' | 'back' | 'shoulders' | 'legs' | 'arms' | 'core'

// All exercises from DEVELOPMENT-PLAN v2.0
export type Exercise =
  // Chest
  | 'benchPress'
  | 'inclineBench'
  | 'dumbbellPress'
  | 'cableFly'
  // Back
  | 'deadlift'
  | 'barbellRow'
  | 'latPulldown'
  | 'pullUps'
  | 'cableRow'
  // Shoulders
  | 'shoulderPressBarbell'
  | 'shoulderPressMachine'
  | 'shoulderPressDumbbell'
  | 'sideLateralDumbbell'
  | 'sideLateralCable'
  | 'frontRaise'
  // Legs
  | 'squat'
  | 'legPress'
  | 'romanianDeadlift'
  | 'legCurl'
  | 'legExtension'
  | 'calfRaise'
  // Arms
  | 'bicepCurlBarbell'
  | 'bicepCurlDumbbell'
  | 'tricepPushdown'
  | 'skullCrushers'

// Exercise metadata
export interface ExerciseInfo {
  id: Exercise
  name: string
  bodyPart: BodyPart
  multipliers: StrengthMultipliers
  isDumbbell?: boolean  // Per-hand weight
}

export interface StrengthMultipliers {
  beginner: number
  novice: number
  intermediate: number
  advanced: number
}

// User's exercise ratings stored in profile
export type ExerciseRatings = Partial<Record<Exercise, Level>>

export interface Profile {
  id: string
  name: string                  // max 50 characters
  age: number                   // 13-100
  height: number                // 100-250 cm
  weight: number                // 30-300 kg
  exerciseRatings: ExerciseRatings  // Only rated exercises stored
  createdAt: string
  updatedAt: string
}

// Calculated strength for display
export interface CalculatedStrength {
  exercise: Exercise
  exerciseName: string
  bodyPart: BodyPart
  isDumbbell?: boolean
  levels: {
    level: Level
    weight: number
    isSelected: boolean
  }[]
}

// Validation constraints
export const VALIDATION = {
  name: { min: 1, max: 50 },
  age: { min: 13, max: 100 },
  height: { min: 100, max: 250 },
  weight: { min: 30, max: 300 },
  maxProfiles: 5
} as const

// Level colors
export const LEVEL_COLORS: Record<Level, string> = {
  beginner: '#2ECC71',
  novice: '#3498DB',
  intermediate: '#F39C12',
  advanced: '#E74C3C'
} as const

// Level display names
export const LEVEL_NAMES: Record<Level, string> = {
  beginner: 'BEG',
  novice: 'NOV',
  intermediate: 'INT',
  advanced: 'ADV'
} as const

// Level full names for display
export const LEVEL_FULL_NAMES: Record<Level, string> = {
  beginner: 'Beginner',
  novice: 'Novice',
  intermediate: 'Intermediate',
  advanced: 'Advanced'
} as const

// Body part display names
export const BODY_PART_NAMES: Record<BodyPart, string> = {
  chest: 'Chest',
  back: 'Back',
  shoulders: 'Shoulders',
  legs: 'Legs',
  arms: 'Arms',
  core: 'Core'
} as const

// All body parts for filtering
export const ALL_BODY_PARTS: BodyPart[] = ['chest', 'back', 'shoulders', 'legs', 'arms', 'core']

// All levels
export const ALL_LEVELS: Level[] = ['beginner', 'novice', 'intermediate', 'advanced']

// Level numeric values for calculating overall level
export const LEVEL_VALUES: Record<Level, number> = {
  beginner: 1,
  novice: 2,
  intermediate: 3,
  advanced: 4
} as const

// Motivational quotes types
export type QuoteCategory = 'motivation' | 'science' | 'benefit'

export interface Quote {
  id: number
  text: string
  author: string
  category: QuoteCategory
}
