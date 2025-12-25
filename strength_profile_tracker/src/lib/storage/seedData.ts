import { Profile, WorkoutSession } from '@/types'
import { formatLocalDate } from './workouts'

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
    id: 'sample_arun_001',
    name: 'Arun',
    age: 42,
    height: 168,
    weight: 70,
    sex: 'male',
    avatarUrl: '/avatars/arun.jpg',
    dailySteps: 8000,
    activityLevel: 'moderate',
    goal: 'maintain',
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
    createdAt: '2024-10-01T10:00:00Z',
    updatedAt: new Date().toISOString()
  },
  {
    id: 'sample_huimin_002',
    name: 'Hui Min',
    age: 32,
    height: 165,
    weight: 68,
    sex: 'female',
    avatarUrl: 'https://randomuser.me/api/portraits/women/44.jpg',
    dailySteps: 6000,
    activityLevel: 'active',
    goal: 'lose',
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
    createdAt: '2024-09-15T08:00:00Z',
    updatedAt: new Date().toISOString()
  },
  {
    id: 'sample_viraj_003',
    name: 'Viraj',
    age: 25,
    height: 175,
    weight: 78,
    sex: 'male',
    avatarUrl: 'https://randomuser.me/api/portraits/men/45.jpg',
    dailySteps: 10000,
    activityLevel: 'active',
    goal: 'gain',
    exerciseRatings: {
      benchPress: 'novice',
      inclineBench: 'beginner',
      dumbbellPress: 'novice',
      deadlift: 'novice',
      barbellRow: 'beginner',
      latPulldown: 'novice',
      squat: 'novice',
      legPress: 'novice',
      romanianDeadlift: 'beginner',
      shoulderPressDumbbell: 'beginner',
      bicepCurlDumbbell: 'novice',
      tricepPushdown: 'beginner',
    },
    createdAt: '2024-10-15T09:00:00Z',
    updatedAt: new Date().toISOString()
  },
  {
    id: 'sample_tanvi_004',
    name: 'Tanvi',
    age: 29,
    height: 162,
    weight: 58,
    sex: 'female',
    avatarUrl: 'https://randomuser.me/api/portraits/women/32.jpg',
    dailySteps: 7500,
    activityLevel: 'moderate',
    goal: 'maintain',
    exerciseRatings: {
      benchPress: 'intermediate',
      dumbbellPress: 'intermediate',
      deadlift: 'intermediate',
      barbellRow: 'novice',
      latPulldown: 'intermediate',
      pullUps: 'novice',
      squat: 'intermediate',
      legPress: 'intermediate',
      legCurl: 'intermediate',
      shoulderPressMachine: 'novice',
      sideLateralCable: 'intermediate',
      bicepCurlBarbell: 'novice',
      tricepPushdown: 'intermediate',
    },
    createdAt: '2024-09-20T11:00:00Z',
    updatedAt: new Date().toISOString()
  }
]

/**
 * Helper to generate dates for workout sessions
 */
function getWorkoutDate(daysAgo: number): string {
  const date = new Date()
  date.setDate(date.getDate() + daysAgo)
  return formatLocalDate(date)
}

/**
 * Generate progressive weights over weeks with some variation
 */
function generateProgressiveWeights(startWeight: number, endWeight: number, weeks: number): number[] {
  const weights: number[] = []
  const increment = (endWeight - startWeight) / (weeks - 1)
  for (let i = 0; i < weeks; i++) {
    // Add slight variation (±2.5kg)
    const variation = Math.random() > 0.7 ? 2.5 : 0
    weights.push(Math.round((startWeight + increment * i + variation) * 2) / 2)
  }
  return weights
}

/**
 * Generate workout sessions for the past year for all profiles
 */
function generateSampleWorkouts(): WorkoutSession[] {
  const workouts: WorkoutSession[] = []

  // ========== ARUN (sample_arun_001) - Intermediate lifter, 52 weeks (1 year) ==========
  const YEAR_WEEKS = 52

  // Arun's bench press - twice per week for 52 weeks (104 sessions)
  for (let week = 0; week < YEAR_WEEKS; week++) {
    const daysAgo = -((YEAR_WEEKS - week) * 7)
    // Progress from 50kg to 85kg over the year
    const weight = Math.round((50 + (35 * week / YEAR_WEEKS)) * 2) / 2

    // Session 1 (Monday-ish)
    workouts.push({
      id: `workout_arun_bench_${week * 2}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'benchPress',
      profileId: 'sample_arun_001',
      sets: [
        { weight, reps: 8 },
        { weight, reps: 7 },
        { weight: weight - 5, reps: 10 }
      ]
    })

    // Session 2 (Thursday-ish)
    workouts.push({
      id: `workout_arun_bench_${week * 2 + 1}`,
      date: getWorkoutDate(daysAgo + 3),
      exerciseId: 'benchPress',
      profileId: 'sample_arun_001',
      sets: [
        { weight, reps: 8 },
        { weight, reps: 6 },
        { weight: weight - 5, reps: 10 }
      ]
    })
  }

  // Arun's squat - twice per week for 52 weeks
  for (let week = 0; week < YEAR_WEEKS; week++) {
    const daysAgo = -((YEAR_WEEKS - week) * 7) - 1
    // Progress from 60kg to 112.5kg over the year
    const weight = Math.round((60 + (52.5 * week / YEAR_WEEKS)) * 2) / 2

    workouts.push({
      id: `workout_arun_squat_${week * 2}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'squat',
      profileId: 'sample_arun_001',
      sets: [
        { weight, reps: 6 },
        { weight, reps: 5 },
        { weight: weight - 10, reps: 8 }
      ]
    })

    workouts.push({
      id: `workout_arun_squat_${week * 2 + 1}`,
      date: getWorkoutDate(daysAgo + 4),
      exerciseId: 'squat',
      profileId: 'sample_arun_001',
      sets: [
        { weight, reps: 6 },
        { weight, reps: 5 },
        { weight: weight - 10, reps: 8 }
      ]
    })
  }

  // Arun's deadlift - once per week for 52 weeks
  for (let week = 0; week < YEAR_WEEKS; week++) {
    const daysAgo = -((YEAR_WEEKS - week) * 7) - 2
    // Progress from 80kg to 135kg over the year
    const weight = Math.round((80 + (55 * week / YEAR_WEEKS)) * 2) / 2

    workouts.push({
      id: `workout_arun_deadlift_${week}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'deadlift',
      profileId: 'sample_arun_001',
      sets: [
        { weight, reps: 5 },
        { weight, reps: 4 },
        { weight: weight - 10, reps: 6 }
      ]
    })
  }

  // Arun's overhead press - once per week for 52 weeks
  for (let week = 0; week < YEAR_WEEKS; week++) {
    const daysAgo = -((YEAR_WEEKS - week) * 7) - 3
    // Progress from 30kg to 52.5kg over the year
    const weight = Math.round((30 + (22.5 * week / YEAR_WEEKS)) * 2) / 2

    workouts.push({
      id: `workout_arun_ohp_${week}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'shoulderPressBarbell',
      profileId: 'sample_arun_001',
      sets: [
        { weight, reps: 8 },
        { weight, reps: 7 },
        { weight, reps: 6 }
      ]
    })
  }

  // Arun's barbell row - once per week for 52 weeks
  for (let week = 0; week < YEAR_WEEKS; week++) {
    const daysAgo = -((YEAR_WEEKS - week) * 7) - 4
    // Progress from 40kg to 72.5kg over the year
    const weight = Math.round((40 + (32.5 * week / YEAR_WEEKS)) * 2) / 2

    workouts.push({
      id: `workout_arun_row_${week}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'barbellRow',
      profileId: 'sample_arun_001',
      sets: [
        { weight, reps: 10 },
        { weight, reps: 8 },
        { weight, reps: 8 }
      ]
    })
  }

  // ========== HUI MIN (sample_huimin_002) - Advanced lower body, 8 weeks ==========

  // Hui Min's squat - strong progression
  const huiminSquatDates = [-56, -52, -49, -45, -42, -38, -35, -31, -28, -24, -21, -17, -14, -10, -7, -3]
  const huiminSquatWeights = [85, 87.5, 90, 92.5, 95, 97.5, 100, 102.5, 105, 107.5, 110, 112.5, 115, 117.5, 120, 122.5]
  huiminSquatDates.forEach((daysAgo, i) => {
    workouts.push({
      id: `workout_huimin_squat_${i}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'squat',
      profileId: 'sample_huimin_002',
      sets: [
        { weight: huiminSquatWeights[i], reps: 6 },
        { weight: huiminSquatWeights[i], reps: 5 },
        { weight: huiminSquatWeights[i], reps: 5 }
      ]
    })
  })

  // Hui Min's deadlift - very strong
  const huiminDeadliftDates = [-55, -48, -41, -34, -27, -20, -13, -6]
  const huiminDeadliftWeights = [115, 120, 125, 130, 135, 140, 145, 150]
  huiminDeadliftDates.forEach((daysAgo, i) => {
    workouts.push({
      id: `workout_huimin_deadlift_${i}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'deadlift',
      profileId: 'sample_huimin_002',
      sets: [
        { weight: huiminDeadliftWeights[i], reps: 5 },
        { weight: huiminDeadliftWeights[i], reps: 4 },
        { weight: huiminDeadliftWeights[i] - 10, reps: 6 }
      ]
    })
  })

  // Hui Min's leg press
  const huiminLegPressDates = [-54, -47, -40, -33, -26, -19, -12, -5]
  const huiminLegPressWeights = [160, 170, 180, 190, 200, 210, 220, 230]
  huiminLegPressDates.forEach((daysAgo, i) => {
    workouts.push({
      id: `workout_huimin_legpress_${i}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'legPress',
      profileId: 'sample_huimin_002',
      sets: [
        { weight: huiminLegPressWeights[i], reps: 10 },
        { weight: huiminLegPressWeights[i], reps: 8 },
        { weight: huiminLegPressWeights[i], reps: 8 }
      ]
    })
  })

  // Hui Min's Romanian deadlift
  const huiminRDLDates = [-53, -46, -39, -32, -25, -18, -11, -4]
  const huiminRDLWeights = [70, 72.5, 75, 77.5, 80, 82.5, 85, 87.5]
  huiminRDLDates.forEach((daysAgo, i) => {
    workouts.push({
      id: `workout_huimin_rdl_${i}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'romanianDeadlift',
      profileId: 'sample_huimin_002',
      sets: [
        { weight: huiminRDLWeights[i], reps: 10 },
        { weight: huiminRDLWeights[i], reps: 10 },
        { weight: huiminRDLWeights[i], reps: 8 }
      ]
    })
  })

  // Hui Min's barbell row
  const huiminRowDates = [-51, -44, -37, -30, -23, -16, -9, -2]
  const huiminRowWeights = [45, 47.5, 50, 52.5, 55, 57.5, 60, 62.5]
  huiminRowDates.forEach((daysAgo, i) => {
    workouts.push({
      id: `workout_huimin_row_${i}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'barbellRow',
      profileId: 'sample_huimin_002',
      sets: [
        { weight: huiminRowWeights[i], reps: 10 },
        { weight: huiminRowWeights[i], reps: 8 },
        { weight: huiminRowWeights[i], reps: 8 }
      ]
    })
  })

  // ========== VIRAJ (sample_viraj_003) - Beginner-Novice, building foundation, 6 weeks ==========

  // Viraj's bench press - learning form, steady progress
  const virajBenchDates = [-42, -38, -35, -31, -28, -24, -21, -17, -14, -10, -7, -3]
  const virajBenchWeights = [40, 42.5, 45, 47.5, 50, 50, 52.5, 55, 55, 57.5, 60, 60]
  virajBenchDates.forEach((daysAgo, i) => {
    workouts.push({
      id: `workout_viraj_bench_${i}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'benchPress',
      profileId: 'sample_viraj_003',
      sets: [
        { weight: virajBenchWeights[i], reps: 10 },
        { weight: virajBenchWeights[i], reps: 8 },
        { weight: virajBenchWeights[i], reps: 8 }
      ]
    })
  })

  // Viraj's squat - beginner gains
  const virajSquatDates = [-41, -37, -34, -30, -27, -23, -20, -16, -13, -9, -6, -2]
  const virajSquatWeights = [50, 55, 57.5, 60, 62.5, 65, 67.5, 70, 72.5, 75, 77.5, 80]
  virajSquatDates.forEach((daysAgo, i) => {
    workouts.push({
      id: `workout_viraj_squat_${i}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'squat',
      profileId: 'sample_viraj_003',
      sets: [
        { weight: virajSquatWeights[i], reps: 8 },
        { weight: virajSquatWeights[i], reps: 6 },
        { weight: virajSquatWeights[i] - 5, reps: 8 }
      ]
    })
  })

  // Viraj's deadlift
  const virajDeadliftDates = [-40, -33, -26, -19, -12, -5]
  const virajDeadliftWeights = [60, 70, 80, 85, 90, 95]
  virajDeadliftDates.forEach((daysAgo, i) => {
    workouts.push({
      id: `workout_viraj_deadlift_${i}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'deadlift',
      profileId: 'sample_viraj_003',
      sets: [
        { weight: virajDeadliftWeights[i], reps: 5 },
        { weight: virajDeadliftWeights[i], reps: 5 },
        { weight: virajDeadliftWeights[i], reps: 5 }
      ]
    })
  })

  // Viraj's lat pulldown
  const virajLatDates = [-39, -32, -25, -18, -11, -4]
  const virajLatWeights = [35, 40, 42.5, 45, 47.5, 50]
  virajLatDates.forEach((daysAgo, i) => {
    workouts.push({
      id: `workout_viraj_lat_${i}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'latPulldown',
      profileId: 'sample_viraj_003',
      sets: [
        { weight: virajLatWeights[i], reps: 12 },
        { weight: virajLatWeights[i], reps: 10 },
        { weight: virajLatWeights[i], reps: 10 }
      ]
    })
  })

  // Viraj's dumbbell press
  const virajDBPressDates = [-38, -31, -24, -17, -10, -3]
  const virajDBPressWeights = [12, 14, 14, 16, 16, 18]
  virajDBPressDates.forEach((daysAgo, i) => {
    workouts.push({
      id: `workout_viraj_dbpress_${i}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'dumbbellPress',
      profileId: 'sample_viraj_003',
      sets: [
        { weight: virajDBPressWeights[i], reps: 12 },
        { weight: virajDBPressWeights[i], reps: 10 },
        { weight: virajDBPressWeights[i], reps: 10 }
      ]
    })
  })

  // Viraj's leg press
  const virajLegPressDates = [-36, -29, -22, -15, -8, -1]
  const virajLegPressWeights = [80, 90, 100, 110, 120, 130]
  virajLegPressDates.forEach((daysAgo, i) => {
    workouts.push({
      id: `workout_viraj_legpress_${i}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'legPress',
      profileId: 'sample_viraj_003',
      sets: [
        { weight: virajLegPressWeights[i], reps: 12 },
        { weight: virajLegPressWeights[i], reps: 10 },
        { weight: virajLegPressWeights[i], reps: 10 }
      ]
    })
  })

  // Viraj's bicep curls
  const virajCurlDates = [-35, -28, -21, -14, -7, 0]
  const virajCurlWeights = [8, 10, 10, 12, 12, 14]
  virajCurlDates.forEach((daysAgo, i) => {
    workouts.push({
      id: `workout_viraj_curl_${i}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'bicepCurlDumbbell',
      profileId: 'sample_viraj_003',
      sets: [
        { weight: virajCurlWeights[i], reps: 12 },
        { weight: virajCurlWeights[i], reps: 10 },
        { weight: virajCurlWeights[i], reps: 10 }
      ]
    })
  })

  // ========== TANVI (sample_tanvi_004) - Intermediate all-around, 7 weeks ==========

  // Tanvi's bench press
  const tanviBenchDates = [-49, -45, -42, -38, -35, -31, -28, -24, -21, -17, -14, -10, -7, -3]
  const tanviBenchWeights = [32.5, 35, 35, 37.5, 37.5, 40, 40, 42.5, 42.5, 45, 45, 47.5, 47.5, 50]
  tanviBenchDates.forEach((daysAgo, i) => {
    workouts.push({
      id: `workout_tanvi_bench_${i}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'benchPress',
      profileId: 'sample_tanvi_004',
      sets: [
        { weight: tanviBenchWeights[i], reps: 10 },
        { weight: tanviBenchWeights[i], reps: 8 },
        { weight: tanviBenchWeights[i], reps: 8 }
      ]
    })
  })

  // Tanvi's squat
  const tanviSquatDates = [-48, -44, -41, -37, -34, -30, -27, -23, -20, -16, -13, -9, -6, -2]
  const tanviSquatWeights = [50, 52.5, 55, 57.5, 60, 62.5, 65, 67.5, 70, 72.5, 75, 77.5, 80, 82.5]
  tanviSquatDates.forEach((daysAgo, i) => {
    workouts.push({
      id: `workout_tanvi_squat_${i}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'squat',
      profileId: 'sample_tanvi_004',
      sets: [
        { weight: tanviSquatWeights[i], reps: 8 },
        { weight: tanviSquatWeights[i], reps: 6 },
        { weight: tanviSquatWeights[i], reps: 6 }
      ]
    })
  })

  // Tanvi's deadlift
  const tanviDeadliftDates = [-47, -40, -33, -26, -19, -12, -5]
  const tanviDeadliftWeights = [60, 65, 70, 75, 80, 85, 90]
  tanviDeadliftDates.forEach((daysAgo, i) => {
    workouts.push({
      id: `workout_tanvi_deadlift_${i}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'deadlift',
      profileId: 'sample_tanvi_004',
      sets: [
        { weight: tanviDeadliftWeights[i], reps: 5 },
        { weight: tanviDeadliftWeights[i], reps: 5 },
        { weight: tanviDeadliftWeights[i] - 5, reps: 6 }
      ]
    })
  })

  // Tanvi's lat pulldown
  const tanviLatDates = [-46, -39, -32, -25, -18, -11, -4]
  const tanviLatWeights = [35, 37.5, 40, 42.5, 45, 47.5, 50]
  tanviLatDates.forEach((daysAgo, i) => {
    workouts.push({
      id: `workout_tanvi_lat_${i}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'latPulldown',
      profileId: 'sample_tanvi_004',
      sets: [
        { weight: tanviLatWeights[i], reps: 12 },
        { weight: tanviLatWeights[i], reps: 10 },
        { weight: tanviLatWeights[i], reps: 10 }
      ]
    })
  })

  // Tanvi's dumbbell press
  const tanviDBPressDates = [-45, -38, -31, -24, -17, -10, -3]
  const tanviDBPressWeights = [10, 12, 12, 14, 14, 16, 16]
  tanviDBPressDates.forEach((daysAgo, i) => {
    workouts.push({
      id: `workout_tanvi_dbpress_${i}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'dumbbellPress',
      profileId: 'sample_tanvi_004',
      sets: [
        { weight: tanviDBPressWeights[i], reps: 12 },
        { weight: tanviDBPressWeights[i], reps: 10 },
        { weight: tanviDBPressWeights[i], reps: 10 }
      ]
    })
  })

  // Tanvi's leg press
  const tanviLegPressDates = [-43, -36, -29, -22, -15, -8, -1]
  const tanviLegPressWeights = [100, 110, 120, 130, 140, 150, 160]
  tanviLegPressDates.forEach((daysAgo, i) => {
    workouts.push({
      id: `workout_tanvi_legpress_${i}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'legPress',
      profileId: 'sample_tanvi_004',
      sets: [
        { weight: tanviLegPressWeights[i], reps: 12 },
        { weight: tanviLegPressWeights[i], reps: 10 },
        { weight: tanviLegPressWeights[i], reps: 10 }
      ]
    })
  })

  // Tanvi's leg curl
  const tanviLegCurlDates = [-42, -35, -28, -21, -14, -7, 0]
  const tanviLegCurlWeights = [25, 27.5, 30, 32.5, 35, 37.5, 40]
  tanviLegCurlDates.forEach((daysAgo, i) => {
    workouts.push({
      id: `workout_tanvi_legcurl_${i}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'legCurl',
      profileId: 'sample_tanvi_004',
      sets: [
        { weight: tanviLegCurlWeights[i], reps: 12 },
        { weight: tanviLegCurlWeights[i], reps: 10 },
        { weight: tanviLegCurlWeights[i], reps: 10 }
      ]
    })
  })

  // Tanvi's tricep pushdown
  const tanviTricepDates = [-41, -34, -27, -20, -13, -6]
  const tanviTricepWeights = [20, 22.5, 25, 27.5, 30, 32.5]
  tanviTricepDates.forEach((daysAgo, i) => {
    workouts.push({
      id: `workout_tanvi_tricep_${i}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'tricepPushdown',
      profileId: 'sample_tanvi_004',
      sets: [
        { weight: tanviTricepWeights[i], reps: 12 },
        { weight: tanviTricepWeights[i], reps: 10 },
        { weight: tanviTricepWeights[i], reps: 10 }
      ]
    })
  })

  // Tanvi's side lateral cable
  const tanviLateralDates = [-40, -33, -26, -19, -12, -5]
  const tanviLateralWeights = [5, 5, 7.5, 7.5, 10, 10]
  tanviLateralDates.forEach((daysAgo, i) => {
    workouts.push({
      id: `workout_tanvi_lateral_${i}`,
      date: getWorkoutDate(daysAgo),
      exerciseId: 'sideLateralCable',
      profileId: 'sample_tanvi_004',
      sets: [
        { weight: tanviLateralWeights[i], reps: 15 },
        { weight: tanviLateralWeights[i], reps: 12 },
        { weight: tanviLateralWeights[i], reps: 12 }
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

  // Arun's score history - 52 weeks (1 year) of steady improvement
  // Progress from 25 to 64 (beginner to intermediate-advanced)
  const arunScores: number[] = []
  for (let week = 0; week < 52; week++) {
    // Score increases from 25 to ~64 over the year with some plateau periods
    const baseScore = 25 + (39 * week / 51)
    // Add small variations (±2)
    const variation = Math.sin(week * 0.5) * 2
    arunScores.push(Math.round(baseScore + variation))
  }
  history['sample_arun_001'] = arunScores.map((score, i) => {
    const date = new Date(today)
    date.setDate(date.getDate() - (arunScores.length - 1 - i) * 7)
    return {
      date: formatLocalDate(date),
      score
    }
  })

  // Hui Min's score history - strong start, consistent gains over 8 weeks
  const huiminScores = [55, 57, 59, 61, 63, 65, 67, 69]
  history['sample_huimin_002'] = huiminScores.map((score, i) => {
    const date = new Date(today)
    date.setDate(date.getDate() - (huiminScores.length - 1 - i) * 7)
    return {
      date: formatLocalDate(date),
      score
    }
  })

  // Viraj's score history - beginner making quick gains over 6 weeks
  const virajScores = [22, 26, 30, 34, 38, 42]
  history['sample_viraj_003'] = virajScores.map((score, i) => {
    const date = new Date(today)
    date.setDate(date.getDate() - (virajScores.length - 1 - i) * 7)
    return {
      date: formatLocalDate(date),
      score
    }
  })

  // Tanvi's score history - intermediate, steady progress over 7 weeks
  const tanviScores = [45, 47, 49, 51, 53, 55, 57]
  history['sample_tanvi_004'] = tanviScores.map((score, i) => {
    const date = new Date(today)
    date.setDate(date.getDate() - (tanviScores.length - 1 - i) * 7)
    return {
      date: formatLocalDate(date),
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

  const today = formatLocalDate(new Date())
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
