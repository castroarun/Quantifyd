import { Profile, Sex, ActivityLevel, Goal, ACTIVITY_LEVEL_INFO, GOAL_INFO } from '@/types'

/**
 * Nutrition calculation result
 */
export interface NutritionInfo {
  bmr: number              // Basal Metabolic Rate
  tdee: number             // Total Daily Energy Expenditure (maintenance)
  targetCalories: number   // Goal-adjusted calories
  bmi: number              // Body Mass Index
  bmiCategory: BMICategory
  stepsCalories: number    // Additional calories from steps
  weeklyChange: number     // Projected weekly weight change in kg
}

export type BMICategory = 'underweight' | 'normal' | 'overweight' | 'obese'

export interface BMICategoryInfo {
  name: string
  color: string
  recommendation: string
}

export const BMI_CATEGORIES: Record<BMICategory, BMICategoryInfo> = {
  underweight: {
    name: 'Underweight',
    color: '#3B82F6', // blue
    recommendation: 'Consider a calorie surplus for healthy weight gain'
  },
  normal: {
    name: 'Normal',
    color: '#22C55E', // green
    recommendation: 'Maintain your current healthy weight'
  },
  overweight: {
    name: 'Overweight',
    color: '#F59E0B', // amber
    recommendation: 'A moderate calorie deficit may be beneficial'
  },
  obese: {
    name: 'Obese',
    color: '#EF4444', // red
    recommendation: 'Consider consulting a healthcare provider'
  }
}

/**
 * Calculate BMR using Mifflin-St Jeor Equation
 * Most accurate formula for estimating basal metabolic rate
 *
 * Male:   BMR = (10 × weight) + (6.25 × height) - (5 × age) + 5
 * Female: BMR = (10 × weight) + (6.25 × height) - (5 × age) - 161
 */
export function calculateBMR(
  weight: number,    // kg
  height: number,    // cm
  age: number,
  sex: Sex
): number {
  const base = (10 * weight) + (6.25 * height) - (5 * age)
  return sex === 'male' ? base + 5 : base - 161
}

/**
 * Calculate calories burned from daily steps
 * Approximately 40 calories per 1000 steps
 */
export function calculateStepsCalories(steps: number): number {
  return Math.round(steps * 0.04)
}

/**
 * Calculate BMI (Body Mass Index)
 * BMI = weight / (height in meters)²
 */
export function calculateBMI(weight: number, height: number): number {
  const heightInMeters = height / 100
  return Math.round((weight / (heightInMeters * heightInMeters)) * 10) / 10
}

/**
 * Get BMI category based on BMI value
 */
export function getBMICategory(bmi: number): BMICategory {
  if (bmi < 18.5) return 'underweight'
  if (bmi < 25) return 'normal'
  if (bmi < 30) return 'overweight'
  return 'obese'
}

/**
 * Calculate TDEE (Total Daily Energy Expenditure)
 * TDEE = (BMR × Activity Multiplier) + Steps Calories
 */
export function calculateTDEE(
  bmr: number,
  activityLevel: ActivityLevel,
  dailySteps: number = 0
): number {
  const activityMultiplier = ACTIVITY_LEVEL_INFO[activityLevel].multiplier
  const stepsCalories = calculateStepsCalories(dailySteps)
  return Math.round((bmr * activityMultiplier) + stepsCalories)
}

/**
 * Calculate target calories based on goal
 */
export function calculateTargetCalories(tdee: number, goal: Goal): number {
  const adjustment = GOAL_INFO[goal].calorieAdjustment
  return Math.round(tdee + adjustment)
}

/**
 * Calculate projected weekly weight change
 * 7700 calories = approximately 1 kg of body weight
 */
export function calculateWeeklyChange(tdee: number, targetCalories: number): number {
  const dailyDifference = targetCalories - tdee
  const weeklyDifference = dailyDifference * 7
  return Math.round((weeklyDifference / 7700) * 10) / 10
}

/**
 * Get complete nutrition info for a profile
 * Returns null if required fields are missing
 */
export function getNutritionInfo(profile: Profile): NutritionInfo | null {
  // Require sex and activity level for nutrition calculations
  if (!profile.sex || !profile.activityLevel) {
    return null
  }

  const bmr = calculateBMR(profile.weight, profile.height, profile.age, profile.sex)
  const stepsCalories = calculateStepsCalories(profile.dailySteps || 0)
  const tdee = calculateTDEE(bmr, profile.activityLevel, profile.dailySteps || 0)
  const goal = profile.goal || 'maintain'
  const targetCalories = calculateTargetCalories(tdee, goal)
  const bmi = calculateBMI(profile.weight, profile.height)
  const bmiCategory = getBMICategory(bmi)
  const weeklyChange = calculateWeeklyChange(tdee, targetCalories)

  return {
    bmr: Math.round(bmr),
    tdee,
    targetCalories,
    bmi,
    bmiCategory,
    stepsCalories,
    weeklyChange
  }
}

/**
 * Get minimum recommended daily calories
 * Generally 1500 for men, 1200 for women
 */
export function getMinimumCalories(sex: Sex): number {
  return sex === 'male' ? 1500 : 1200
}

/**
 * Check if target calories are below minimum recommended
 */
export function isBelowMinimum(targetCalories: number, sex: Sex): boolean {
  return targetCalories < getMinimumCalories(sex)
}

/**
 * Format calorie number with commas
 */
export function formatCalories(calories: number): string {
  return calories.toLocaleString()
}
