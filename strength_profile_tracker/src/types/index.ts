// Profile types based on DEVELOPMENT-PLAN v2.0

export type Level = 'beginner' | 'novice' | 'intermediate' | 'advanced'

export type BodyPart = 'chest' | 'back' | 'shoulders' | 'legs' | 'arms' | 'core'

// Phase 4: New profile fields
export type Sex = 'male' | 'female'
export type ActivityLevel = 'sedentary' | 'light' | 'moderate' | 'active' | 'very_active'
export type Goal = 'lose' | 'maintain' | 'gain'

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
  multipliers: StrengthMultipliers           // Male multipliers (default)
  femaleMultipliers?: StrengthMultipliers    // Female multipliers (Phase 4)
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
  sex?: Sex                     // For gender-specific standards & BMR (Phase 4)
  avatarUrl?: string            // Profile picture URL (optional)
  dailySteps?: number           // Average daily steps, 0-50000 (Phase 4)
  activityLevel?: ActivityLevel // General activity level (Phase 4)
  goal?: Goal                   // Fitness/nutrition goal (Phase 4)
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
  dailySteps: { min: 0, max: 50000 },
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

// Badge types
export type BadgeId =
  | 'firstStep'
  | 'gettingStarted'
  | 'dedicatedTrainer'
  | 'fullAssessment'
  | 'noviceMilestone'
  | 'breakingIntermediate'
  | 'advancedLifter'
  | 'eliteStatus'

export interface Badge {
  id: BadgeId
  name: string
  description: string
  icon: string
  earned: boolean
}

// Workout Logger types
export interface WorkoutSet {
  weight: number | null  // kg, null if not logged
  reps: number | null    // null if not logged
}

export interface WorkoutSession {
  id: string
  date: string           // ISO date string (YYYY-MM-DD)
  exerciseId: Exercise
  profileId: string
  sets: WorkoutSet[]     // Always 3 sets
}

// Suggested rep scheme for each set
export const SUGGESTED_REPS = [12, 10, 8] as const

// Phase 4: Activity level display names and multipliers
export const ACTIVITY_LEVEL_INFO: Record<ActivityLevel, { name: string; description: string; multiplier: number }> = {
  sedentary: { name: 'Sedentary', description: 'Little to no exercise, desk job', multiplier: 1.2 },
  light: { name: 'Light', description: 'Light exercise 1-3 days/week', multiplier: 1.375 },
  moderate: { name: 'Moderate', description: 'Moderate exercise 3-5 days/week', multiplier: 1.55 },
  active: { name: 'Active', description: 'Hard exercise 6-7 days/week', multiplier: 1.725 },
  very_active: { name: 'Very Active', description: 'Very hard exercise, physical job', multiplier: 1.9 }
} as const

// Phase 4: Goal display names
export const GOAL_INFO: Record<Goal, { name: string; calorieAdjustment: number; description: string }> = {
  lose: { name: 'Lose Weight', calorieAdjustment: -500, description: 'Moderate calorie deficit' },
  maintain: { name: 'Maintain', calorieAdjustment: 0, description: 'Maintain current weight' },
  gain: { name: 'Gain Muscle', calorieAdjustment: 300, description: 'Lean bulk surplus' }
} as const

// Phase 4: Sex display
export const SEX_INFO: Record<Sex, { name: string; icon: string }> = {
  male: { name: 'Male', icon: '♂' },
  female: { name: 'Female', icon: '♀' }
} as const

// All activity levels
export const ALL_ACTIVITY_LEVELS: ActivityLevel[] = ['sedentary', 'light', 'moderate', 'active', 'very_active']

// All goals
export const ALL_GOALS: Goal[] = ['lose', 'maintain', 'gain']

// Unit types for weight display
export type WeightUnit = 'kg' | 'lbs'

export const WEIGHT_UNIT_INFO: Record<WeightUnit, { name: string; shortName: string }> = {
  kg: { name: 'Kilograms', shortName: 'kg' },
  lbs: { name: 'Pounds', shortName: 'lbs' }
} as const

// Conversion constants
export const KG_TO_LBS = 2.20462
export const LBS_TO_KG = 0.453592
