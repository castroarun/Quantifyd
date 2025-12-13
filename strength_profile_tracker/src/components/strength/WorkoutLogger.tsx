'use client'

import { useState, useEffect, useMemo } from 'react'
import { Exercise, WorkoutSession, WorkoutSet, SUGGESTED_REPS, Level } from '@/types'
import {
  getExerciseSessions,
  getTodaySession,
  saveWorkoutSession,
  formatSessionDate,
  createEmptySets,
  getTodayDate,
  getCelebrationMessage,
  isNewPR,
  checkLevelUpgrade,
  getExercisePR
} from '@/lib/storage/workouts'
import { getExerciseById } from '@/lib/calculations/strength'
import { getProfileById, updateExerciseRating } from '@/lib/storage/profiles'
import { useUnit } from '@/contexts'
import { formatWeightValue } from '@/lib/utils/units'

interface WorkoutLoggerProps {
  profileId: string
  exerciseId: Exercise
  onLevelUp?: (newLevel: Level) => void
}

interface CelebrationState {
  show: boolean
  message: string
}

interface Suggestion {
  weight: number
  reps: number
}

/**
 * Calculate smart progression suggestions based on previous session
 *
 * Logic:
 * - If S1 reps >= 10: PROGRESS (add 5kg, adjust reps)
 * - If S1 reps < 10: MAINTAIN (same weights, try to beat reps)
 */
function calculateSuggestions(prevSets: WorkoutSet[]): Suggestion[] | null {
  // Need at least some data from previous session
  const s1 = prevSets[0]
  const s2 = prevSets[1]
  const s3 = prevSets[2]

  if (!s1?.weight || !s1?.reps) return null

  const s1Reps = s1.reps
  const s1Weight = s1.weight
  const s2Weight = s2?.weight || s1Weight + 5
  const s2Reps = s2?.reps || 10
  const s3Weight = s3?.weight || s2Weight + 5
  const s3Reps = s3?.reps || 8

  // Decide: PROGRESS or MAINTAIN
  if (s1Reps >= 10) {
    // PROGRESS: Add 5kg to all sets
    return [
      { weight: s1Weight + 5, reps: 12 },
      { weight: s2Weight + 5, reps: 10 },
      { weight: s3Weight + 5, reps: Math.max(5, s3Reps - 1) }
    ]
  } else {
    // MAINTAIN: Same weights, try to increase reps
    return [
      { weight: s1Weight, reps: Math.min(12, s1Reps + 1) },
      { weight: s2Weight, reps: Math.min(10, s2Reps + 1) },
      { weight: s3Weight, reps: Math.min(8, s3Reps + 1) }
    ]
  }
}

export default function WorkoutLogger({ profileId, exerciseId, onLevelUp }: WorkoutLoggerProps) {
  const { unit } = useUnit()
  const [pastSessions, setPastSessions] = useState<WorkoutSession[]>([])
  const [todaySets, setTodaySets] = useState<WorkoutSet[]>(createEmptySets())
  const [isLoaded, setIsLoaded] = useState(false)
  const [celebration, setCelebration] = useState<CelebrationState>({ show: false, message: '' })
  const [levelUpMessage, setLevelUpMessage] = useState<string | null>(null)
  const [trackedPRs, setTrackedPRs] = useState<Set<number>>(new Set())

  useEffect(() => {
    // Load only last 2 past sessions (excluding today) for 3-column display
    const sessions = getExerciseSessions(profileId, exerciseId, 3)
      .filter(s => s.date !== getTodayDate())
      .slice(0, 2)
    setPastSessions(sessions)

    // Load today's session if exists
    const today = getTodaySession(profileId, exerciseId)
    if (today) {
      setTodaySets(today.sets)
    }

    setIsLoaded(true)
  }, [profileId, exerciseId])

  // Calculate smart suggestions based on most recent session
  const suggestions = useMemo(() => {
    if (pastSessions.length === 0) return null
    // Most recent session (after reverse, it's the last one, but we have them sorted newest first)
    const mostRecent = pastSessions[0]
    if (!mostRecent) return null
    return calculateSuggestions(mostRecent.sets)
  }, [pastSessions])

  // Check if suggested S3 weight would be a new PR
  const prEncouragement = useMemo(() => {
    if (!suggestions) return null
    const suggestedMax = Math.max(...suggestions.map(s => s.weight))
    const currentPR = getExercisePR(profileId, exerciseId)
    if (suggestedMax > currentPR && currentPR > 0) {
      return `💪 Go for ${suggestedMax}${unit}! That's a new PR!`
    }
    if (suggestedMax > currentPR && currentPR === 0 && pastSessions.length > 0) {
      return `🔥 Time to progress! Aim for ${suggestedMax}${unit}!`
    }
    return null
  }, [suggestions, profileId, exerciseId, unit, pastSessions.length])

  const handleSetChange = (setIndex: number, field: 'weight' | 'reps', value: string) => {
    const numValue = value === '' ? null : parseInt(value, 10)
    if (value !== '' && (isNaN(numValue!) || numValue! < 0)) return

    const newSets = [...todaySets]
    newSets[setIndex] = {
      ...newSets[setIndex],
      [field]: numValue
    }
    setTodaySets(newSets)

    // Auto-save
    saveWorkoutSession(profileId, exerciseId, newSets)

    // Check for new PR and level upgrade when weight is entered
    if (field === 'weight' && numValue && numValue > 0) {
      const exercise = getExerciseById(exerciseId)
      const profile = getProfileById(profileId)

      if (exercise && profile && !trackedPRs.has(numValue)) {
        // Check if this is a new PR (compare against history OR today's other sets)
        const isPRvsHistory = isNewPR(profileId, exerciseId, numValue)

        // Also check if this is higher than other sets entered today
        const todayMax = Math.max(...newSets.filter((_, i) => i !== setIndex).map(s => s.weight || 0))
        const isPRvsToday = todayMax > 0 && numValue > todayMax

        const isPR = isPRvsHistory || isPRvsToday

        if (isPR) {
          // Mark this weight as celebrated
          setTrackedPRs(prev => new Set([...prev, numValue]))

          // Show PR celebration
          const message = getCelebrationMessage(exercise.name)
          setCelebration({ show: true, message })

          // Auto-hide after 4 seconds
          setTimeout(() => {
            setCelebration({ show: false, message: '' })
          }, 4000)
        }

        // Check for level upgrade (independent of PR - works on first workout too)
        const currentLevel = profile.exerciseRatings[exerciseId] || null
        const newLevel = checkLevelUpgrade(
          exerciseId,
          profile.weight,
          numValue,
          currentLevel,
          profile.sex
        )

        if (newLevel) {
          // Mark as tracked to avoid duplicate notifications
          setTrackedPRs(prev => new Set([...prev, numValue]))

          // Auto-update the level
          updateExerciseRating(profileId, exerciseId, newLevel)
          setLevelUpMessage(`Level up! You're now ${newLevel.charAt(0).toUpperCase() + newLevel.slice(1)}!`)

          // Notify parent component
          if (onLevelUp) {
            onLevelUp(newLevel)
          }

          // Hide level up message after 5 seconds
          setTimeout(() => {
            setLevelUpMessage(null)
          }, 5000)
        }
      }
    }
  }

  if (!isLoaded) {
    return (
      <div className="py-2 text-center text-sm text-gray-400">
        Loading...
      </div>
    )
  }

  // Build columns: past sessions → TODAY → TARGET
  const columns = [
    ...pastSessions.slice().reverse().map(s => ({
      date: formatSessionDate(s.date),
      sets: s.sets,
      isToday: false,
      isSuggestion: false
    })),
    {
      date: 'TODAY',
      sets: todaySets,
      isToday: true,
      isSuggestion: false
    },
    // Add target column AFTER today
    ...(suggestions ? [{
      date: 'TARGET',
      sets: suggestions.map(s => ({ weight: s.weight, reps: s.reps })),
      isToday: false,
      isSuggestion: true
    }] : [])
  ]

  // Copy suggestion to today's set
  const copyToToday = (setIndex: number) => {
    if (!suggestions?.[setIndex]) return
    const suggestion = suggestions[setIndex]
    const newSets = [...todaySets]
    newSets[setIndex] = { weight: suggestion.weight, reps: suggestion.reps }
    setTodaySets(newSets)
    saveWorkoutSession(profileId, exerciseId, newSets)
  }

  return (
    <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-700 relative">
      {/* PR Celebration Toast */}
      {celebration.show && (
        <div className="absolute -top-2 left-0 right-0 z-10 animate-pulse">
          <div className="bg-gradient-to-r from-yellow-400 via-orange-500 to-red-500 text-white text-center py-2 px-3 rounded-lg shadow-lg text-sm font-bold">
            {celebration.message}
          </div>
        </div>
      )}

      {/* Level Up Message */}
      {levelUpMessage && (
        <div className="absolute -top-2 left-0 right-0 z-10">
          <div className="bg-gradient-to-r from-purple-500 via-pink-500 to-red-500 text-white text-center py-2 px-3 rounded-lg shadow-lg text-sm font-bold animate-bounce">
            🚀 {levelUpMessage}
          </div>
        </div>
      )}

      <div className="flex items-center justify-between mb-2">
        <p className="text-xs text-gray-500 dark:text-gray-400 font-medium">
          Workout Log
        </p>
      </div>

      {/* PR Encouragement Message */}
      {prEncouragement && !celebration.show && !levelUpMessage && (
        <div className="mb-2 bg-gradient-to-r from-green-500/10 to-blue-500/10 border border-green-500/30 text-green-600 dark:text-green-400 text-center py-1.5 px-2 rounded-lg text-xs font-medium">
          {prEncouragement}
        </div>
      )}

      {/* Workout columns - horizontally scrollable */}
      <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-visible">
        {/* Set labels column */}
        <div className="flex flex-col gap-1 flex-shrink-0">
          <div className="h-8"></div>
          {[0, 1, 2].map(setIndex => (
            <div
              key={setIndex}
              className="w-8 h-10 text-xs text-gray-400 dark:text-gray-500 flex items-center justify-center font-medium"
            >
              S{setIndex + 1}
            </div>
          ))}
        </div>

        {/* Past session columns */}
        {pastSessions.slice().reverse().map((session, idx) => (
          <div key={idx} className="flex flex-col gap-1 flex-shrink-0 min-w-[80px]">
            <div className="text-center text-xs font-medium py-1.5 px-2 rounded h-8 flex items-center justify-center text-gray-400 dark:text-gray-500">
              {formatSessionDate(session.date)}
            </div>
            {[0, 1, 2].map(setIndex => {
              const set = session.sets[setIndex]
              const hasData = set?.weight !== null && set?.reps !== null
              return (
                <div
                  key={setIndex}
                  className="text-center text-sm py-2 rounded bg-gray-100 dark:bg-gray-700/50 h-10 flex items-center justify-center"
                >
                  {hasData ? (
                    <span className="text-gray-600 dark:text-gray-300">
                      {formatWeightValue(set.weight!, unit)}×{set.reps}
                    </span>
                  ) : (
                    <span className="text-gray-300 dark:text-gray-600">-</span>
                  )}
                </div>
              )
            })}
          </div>
        ))}

        {/* TODAY column */}
        <div className="flex flex-col gap-1 flex-shrink-0 min-w-[100px]">
          <div className="text-center text-xs font-medium py-1.5 px-2 rounded h-8 flex items-center justify-center bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300">
            TODAY
          </div>
          {[0, 1, 2].map(setIndex => {
            const suggestion = suggestions?.[setIndex]
            return (
              <div
                key={setIndex}
                className="flex items-center gap-1 bg-blue-50 dark:bg-blue-900/20 rounded px-2 py-1 h-10"
              >
                <input
                  type="number"
                  inputMode="numeric"
                  placeholder={suggestion ? String(suggestion.weight) : '-'}
                  value={todaySets[setIndex].weight ?? ''}
                  onChange={(e) => handleSetChange(setIndex, 'weight', e.target.value)}
                  className="w-12 text-center text-sm bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 border border-gray-200 dark:border-gray-600 rounded py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
                <span className="text-xs text-gray-400 dark:text-gray-500">×</span>
                <input
                  type="number"
                  inputMode="numeric"
                  placeholder={suggestion ? String(suggestion.reps) : SUGGESTED_REPS[setIndex].toString()}
                  value={todaySets[setIndex].reps ?? ''}
                  onChange={(e) => handleSetChange(setIndex, 'reps', e.target.value)}
                  className="w-10 text-center text-sm bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 border border-gray-200 dark:border-gray-600 rounded py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>
            )
          })}
        </div>

        {/* Copy buttons column (if suggestions exist) */}
        {suggestions && (
          <div className="flex flex-col gap-1 flex-shrink-0">
            <div className="h-8"></div>
            {[0, 1, 2].map(setIndex => (
              <button
                key={setIndex}
                onClick={() => copyToToday(setIndex)}
                className="w-8 h-10 flex items-center justify-center text-green-500 hover:text-green-600 hover:bg-green-100 dark:hover:bg-green-900/30 rounded transition-colors"
                title="Copy target to today"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
                </svg>
              </button>
            ))}
          </div>
        )}

        {/* TARGET column (if suggestions exist) */}
        {suggestions && (
          <div className="flex flex-col gap-1 flex-shrink-0 min-w-[80px]">
            <div className="text-center text-xs font-medium py-1.5 px-2 rounded h-8 flex items-center justify-center bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300">
              🎯 TARGET
            </div>
            {[0, 1, 2].map(setIndex => {
              const suggestion = suggestions[setIndex]
              return (
                <div
                  key={setIndex}
                  className="text-center text-sm py-2 rounded bg-green-50 dark:bg-green-900/20 h-10 flex items-center justify-center border border-green-200 dark:border-green-800 cursor-pointer hover:bg-green-100 dark:hover:bg-green-900/40 transition-colors"
                  onClick={() => copyToToday(setIndex)}
                  title="Click to copy to today"
                >
                  <span className="text-green-600 dark:text-green-400 font-medium">
                    {suggestion.weight}×{suggestion.reps}
                  </span>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Rep scheme suggestion */}
      <p className="text-[10px] text-gray-400 dark:text-gray-500 mt-2 text-center">
        Suggested reps: 12 / 10 / 8
      </p>
    </div>
  )
}
