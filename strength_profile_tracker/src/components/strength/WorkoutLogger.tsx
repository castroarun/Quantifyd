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
  checkLevelDowngrade,
  getExercisePR,
  getDowngradeMessage
} from '@/lib/storage/workouts'
import { getExerciseById } from '@/lib/calculations/strength'
import { getProfileById, updateExerciseRating } from '@/lib/storage/profiles'
import { useUnit } from '@/contexts'
import { formatWeightValue } from '@/lib/utils/units'
import { RestTimer, TimerSettings, FullScreenTimer } from '@/components/timer'
import { getTimerSettings, saveExerciseTimerDuration } from '@/lib/storage/timer'
import { getFormTip } from '@/data/formTips'
import { calculateWarmupSets } from '@/lib/utils/warmup'

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
  const [levelDownMessage, setLevelDownMessage] = useState<string | null>(null)
  const [trackedPRs, setTrackedPRs] = useState<Set<number>>(new Set())

  // Timer state
  const [showTimer, setShowTimer] = useState(true)
  const [showTimerSettings, setShowTimerSettings] = useState(false)
  const [showFullScreenTimer, setShowFullScreenTimer] = useState(false)
  const [timerTrigger, setTimerTrigger] = useState(0) // Increments to trigger auto-start
  const [timerSetInfo, setTimerSetInfo] = useState<{ setNumber: number; weight: number; reps: number | null } | null>(null)

  // Coaching state - subtle, user-initiated
  const [showFormTip, setShowFormTip] = useState(false)
  const [showWarmup, setShowWarmup] = useState(false)

  // Get exercise name for display
  const exercise = getExerciseById(exerciseId)

  // Get user's current level for this exercise
  const profile = getProfileById(profileId)
  const currentLevel = profile?.exerciseRatings[exerciseId] || 'beginner'

  // Get form tip for current level
  const formTip = useMemo(() => {
    return getFormTip(exerciseId, currentLevel)
  }, [exerciseId, currentLevel])

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

    // Check for level downgrade on mount (based on last 4 workouts)
    const profile = getProfileById(profileId)
    if (profile) {
      const currentLevel = profile.exerciseRatings[exerciseId] || null
      const downgradeLevel = checkLevelDowngrade(
        profileId,
        exerciseId,
        profile.weight,
        currentLevel,
        profile.sex
      )

      if (downgradeLevel) {
        // Apply the downgrade
        updateExerciseRating(profileId, exerciseId, downgradeLevel)

        // Show notification
        const message = getDowngradeMessage(downgradeLevel)
        setLevelDownMessage(message)

        // Notify parent
        if (onLevelUp) {
          onLevelUp(downgradeLevel)
        }

        // Hide after 6 seconds
        setTimeout(() => {
          setLevelDownMessage(null)
        }, 6000)
      }
    }

    setIsLoaded(true)
  }, [profileId, exerciseId, onLevelUp])

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

  // Calculate warm-up sets based on first target weight
  const warmupSets = useMemo(() => {
    if (!suggestions || suggestions.length === 0) return null
    // Use Set 1 target weight as the working weight
    return calculateWarmupSets(suggestions[0].weight)
  }, [suggestions])

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

    // NOTE: Timer is now triggered by the Done button, not on input change

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

  // Copy suggestion to today's set
  const copyToToday = (setIndex: number) => {
    if (!suggestions?.[setIndex]) return
    const suggestion = suggestions[setIndex]
    const newSets = [...todaySets]
    newSets[setIndex] = { weight: suggestion.weight, reps: suggestion.reps }
    setTodaySets(newSets)
    saveWorkoutSession(profileId, exerciseId, newSets)
  }

  // Handle "Done" button - marks set complete and starts timer
  const handleSetDone = (setIndex: number) => {
    const currentSet = todaySets[setIndex]
    if (!currentSet.weight || currentSet.weight <= 0) return

    // Set timer info for this set
    setTimerSetInfo({
      setNumber: setIndex + 1,
      weight: currentSet.weight,
      reps: currentSet.reps
    })

    // Increment trigger to reset/restart the timer (this pauses any running timer)
    setTimerTrigger(prev => prev + 1)

    // Open full-screen timer if enabled
    if (showTimer) {
      setShowFullScreenTimer(true)
    }
  }

  // Handle timer close - resets timer state
  const handleTimerClose = () => {
    setShowFullScreenTimer(false)
    // Reset timer trigger to stop any running timer
    setTimerTrigger(prev => prev + 1)
    setTimerSetInfo(null)
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

      {/* Level Down Message */}
      {levelDownMessage && (
        <div className="absolute -top-2 left-0 right-0 z-10">
          <div className="bg-gradient-to-r from-gray-500 via-gray-600 to-gray-700 text-white text-center py-2 px-3 rounded-lg shadow-lg text-sm font-medium">
            {levelDownMessage}
          </div>
        </div>
      )}

      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <p className="text-xs text-gray-500 dark:text-gray-400 font-medium">
            Workout Log
          </p>
          {/* Form tip trigger - subtle lightbulb icon */}
          {formTip && (
            <button
              onClick={() => setShowFormTip(!showFormTip)}
              className={`p-1 rounded-full transition-colors ${
                showFormTip
                  ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600'
                  : 'text-gray-400 hover:text-yellow-500 hover:bg-yellow-50 dark:hover:bg-yellow-900/20'
              }`}
              title="Form tip"
            >
              <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M11 3a1 1 0 10-2 0v1a1 1 0 102 0V3zM15.657 5.757a1 1 0 00-1.414-1.414l-.707.707a1 1 0 001.414 1.414l.707-.707zM18 10a1 1 0 01-1 1h-1a1 1 0 110-2h1a1 1 0 011 1zM5.05 6.464A1 1 0 106.464 5.05l-.707-.707a1 1 0 00-1.414 1.414l.707.707zM5 10a1 1 0 01-1 1H3a1 1 0 110-2h1a1 1 0 011 1zM8 16v-1h4v1a2 2 0 11-4 0zM12 14c.015-.34.208-.646.477-.859a4 4 0 10-4.954 0c.27.213.462.519.476.859h4.002z" />
              </svg>
            </button>
          )}
        </div>
        {/* Timer toggle button */}
        <button
          onClick={() => setShowTimer(!showTimer)}
          className={`flex items-center gap-1 text-xs px-2 py-1 rounded-full transition-colors ${
            showTimer
              ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
              : 'bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400'
          }`}
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Timer
        </button>
      </div>

      {/* Form Tip - subtle tooltip that appears on tap */}
      {showFormTip && formTip && (
        <div className="mb-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800/50 rounded-lg px-3 py-2">
          <div className="flex items-start gap-2">
            <span className="text-yellow-500 text-sm flex-shrink-0">💡</span>
            <p className="text-xs text-yellow-800 dark:text-yellow-200 leading-relaxed">
              {formTip}
            </p>
            <button
              onClick={() => setShowFormTip(false)}
              className="text-yellow-400 hover:text-yellow-600 dark:hover:text-yellow-300 flex-shrink-0"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* Rest Timer Preview - Double-click to open full-screen */}
      {showTimer && (
        <div className="mb-3">
          <div className="flex justify-end mb-1">
            <button
              onClick={() => setShowTimerSettings(true)}
              className="text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 flex items-center gap-1"
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              Settings
            </button>
          </div>
          <RestTimer
            key={timerTrigger}
            exerciseId={exerciseId}
            exerciseName={exercise?.name}
            setNumber={timerSetInfo?.setNumber}
            weight={timerSetInfo?.weight}
            reps={timerSetInfo?.reps ?? undefined}
            onExpand={() => setShowFullScreenTimer(true)}
            autoStart={false}
          />
        </div>
      )}

      {/* Full Screen Timer Modal */}
      {showFullScreenTimer && (
        <FullScreenTimer
          exerciseId={exerciseId}
          exerciseName={exercise?.name}
          setNumber={timerSetInfo?.setNumber}
          weight={timerSetInfo?.weight}
          reps={timerSetInfo?.reps ?? undefined}
          onClose={handleTimerClose}
          autoStart={true}
        />
      )}

      {/* Timer Settings Modal */}
      {showTimerSettings && (
        <TimerSettings onClose={() => setShowTimerSettings(false)} />
      )}

      {/* PR Encouragement Message */}
      {prEncouragement && !celebration.show && !levelUpMessage && (
        <div className="mb-2 bg-gradient-to-r from-green-500/10 to-blue-500/10 border border-green-500/30 text-green-600 dark:text-green-400 text-center py-1.5 px-2 rounded-lg text-xs font-medium">
          {prEncouragement}
        </div>
      )}

      {/* Warm-up suggestion - collapsible, only shows for heavier weights */}
      {warmupSets && warmupSets.length > 0 && (
        <div className="mb-2">
          <button
            onClick={() => setShowWarmup(!showWarmup)}
            className="flex items-center gap-1 text-[11px] text-orange-500 dark:text-orange-400 hover:text-orange-600 dark:hover:text-orange-300"
          >
            <span>🔥</span>
            <span>Warm-up ({warmupSets.length} sets)</span>
            <svg
              className={`w-3 h-3 transition-transform ${showWarmup ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          {showWarmup && (
            <div className="mt-1.5 bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800/50 rounded-lg px-3 py-2">
              <div className="flex flex-wrap gap-2">
                {warmupSets.map((set, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-1 bg-white dark:bg-gray-800 rounded px-2 py-1 text-xs"
                  >
                    <span className="text-orange-500 font-medium">{set.weight}{unit}</span>
                    <span className="text-gray-400">×</span>
                    <span className="text-gray-600 dark:text-gray-300">{set.reps}</span>
                  </div>
                ))}
              </div>
              <p className="text-[10px] text-orange-600/70 dark:text-orange-400/70 mt-1.5">
                {warmupSets.map(s => s.purpose).join(' → ')}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Main Workout Entry - TARGET and TODAY visible by default */}
      <div className="space-y-3">
        {/* Set rows - each row is a full-width card */}
        {[0, 1, 2].map(setIndex => {
          const suggestion = suggestions?.[setIndex]
          const todaySet = todaySets[setIndex]
          const hasWeight = todaySet.weight !== null && todaySet.weight > 0
          const hasReps = todaySet.reps !== null && todaySet.reps > 0
          const isComplete = hasWeight && hasReps

          return (
            <div
              key={setIndex}
              className={`rounded-xl p-4 border-2 transition-all ${
                isComplete
                  ? 'bg-green-50 dark:bg-green-900/20 border-green-300 dark:border-green-700'
                  : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700'
              }`}
            >
              {/* Set header with TARGET */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span className={`text-lg font-bold ${
                    isComplete ? 'text-green-600 dark:text-green-400' : 'text-gray-700 dark:text-gray-300'
                  }`}>
                    Set {setIndex + 1}
                  </span>
                  {isComplete && (
                    <svg className="w-5 h-5 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                  )}
                </div>
                {/* TARGET badge */}
                {suggestion && (
                  <button
                    onClick={() => copyToToday(setIndex)}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-green-100 dark:bg-green-900/40 hover:bg-green-200 dark:hover:bg-green-800/60 transition-colors"
                    title="Tap to auto-fill"
                  >
                    <span className="text-xs text-green-600 dark:text-green-400 font-medium">🎯 TARGET</span>
                    <span className="text-sm font-bold text-green-700 dark:text-green-300">
                      {suggestion.weight}{unit} × {suggestion.reps}
                    </span>
                  </button>
                )}
              </div>

              {/* Input row - large touch targets */}
              <div className="flex items-center gap-3">
                {/* Weight input */}
                <div className="flex-1">
                  <label className="text-xs text-gray-500 dark:text-gray-400 mb-1 block">Weight ({unit})</label>
                  <input
                    type="number"
                    inputMode="numeric"
                    placeholder={suggestion ? String(suggestion.weight) : '-'}
                    value={todaySet.weight ?? ''}
                    onChange={(e) => handleSetChange(setIndex, 'weight', e.target.value)}
                    className="w-full text-center text-2xl font-bold bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-300 dark:placeholder-gray-500 border-2 border-gray-200 dark:border-gray-600 rounded-xl py-4 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-blue-400"
                  />
                </div>

                {/* Separator */}
                <span className="text-2xl text-gray-300 dark:text-gray-600 font-bold mt-5">×</span>

                {/* Reps input */}
                <div className="flex-1">
                  <label className="text-xs text-gray-500 dark:text-gray-400 mb-1 block">Reps</label>
                  <input
                    type="number"
                    inputMode="numeric"
                    placeholder={suggestion ? String(suggestion.reps) : SUGGESTED_REPS[setIndex].toString()}
                    value={todaySet.reps ?? ''}
                    onChange={(e) => handleSetChange(setIndex, 'reps', e.target.value)}
                    className="w-full text-center text-2xl font-bold bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-300 dark:placeholder-gray-500 border-2 border-gray-200 dark:border-gray-600 rounded-xl py-4 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-blue-400"
                  />
                </div>

                {/* Done button - triggers timer */}
                <button
                  onClick={() => handleSetDone(setIndex)}
                  disabled={!hasWeight}
                  className={`mt-5 p-4 rounded-xl transition-all ${
                    isComplete
                      ? 'bg-green-500 text-white'
                      : hasWeight
                        ? 'bg-blue-500 hover:bg-blue-600 text-white'
                        : 'bg-gray-200 dark:bg-gray-700 text-gray-400 cursor-not-allowed'
                  }`}
                  title={isComplete ? 'Set complete!' : 'Mark set done & start timer'}
                >
                  {isComplete ? (
                    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  ) : (
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  )}
                </button>
              </div>
            </div>
          )
        })}
      </div>

      {/* Past Sessions - collapsible */}
      {pastSessions.length > 0 && (
        <details className="mt-4">
          <summary className="text-xs text-gray-500 dark:text-gray-400 cursor-pointer hover:text-gray-700 dark:hover:text-gray-300 py-2">
            📊 Past Sessions ({pastSessions.length})
          </summary>
          <div className="flex gap-2 overflow-x-auto pb-2 mt-2 scrollbar-visible" style={{ WebkitOverflowScrolling: 'touch', touchAction: 'pan-y pan-x' }}>
            {/* Set labels column */}
            <div className="flex flex-col gap-1 flex-shrink-0">
              <div className="h-6"></div>
              {[0, 1, 2].map(setIndex => (
                <div
                  key={setIndex}
                  className="w-6 h-8 text-xs text-gray-400 dark:text-gray-500 flex items-center justify-center font-medium"
                >
                  S{setIndex + 1}
                </div>
              ))}
            </div>

            {/* Past session columns */}
            {pastSessions.slice().reverse().map((session, idx) => (
              <div key={idx} className="flex flex-col gap-1 flex-shrink-0 min-w-[70px]">
                <div className="text-center text-[10px] font-medium py-1 px-1 rounded h-6 flex items-center justify-center text-gray-400 dark:text-gray-500">
                  {formatSessionDate(session.date)}
                </div>
                {[0, 1, 2].map(setIndex => {
                  const set = session.sets[setIndex]
                  const hasData = set?.weight !== null && set?.reps !== null
                  return (
                    <div
                      key={setIndex}
                      className="text-center text-xs py-1.5 rounded bg-gray-100 dark:bg-gray-700/50 h-8 flex items-center justify-center"
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
          </div>
        </details>
      )}

      {/* Rep scheme suggestion */}
      <p className="text-[10px] text-gray-400 dark:text-gray-500 mt-2 text-center">
        Suggested reps: 12 / 10 / 8
      </p>
    </div>
  )
}
