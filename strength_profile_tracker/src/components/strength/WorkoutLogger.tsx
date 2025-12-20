'use client'

import { useState, useEffect, useMemo, useRef } from 'react'
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
import { formatWeightValue, convertToKg } from '@/lib/utils/units'
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
 * Calculate smart progression suggestions based on previous sessions
 *
 * Logic:
 * - If most recent session is incomplete (1-2 sets), look at earlier sessions too
 * - For each set, use the HIGHER suggestion between recent incomplete + earlier complete data
 * - If S1 reps >= 10: PROGRESS (add 5kg, adjust reps)
 * - If S1 reps < 10: MAINTAIN (same weights, try to beat reps)
 */
function calculateSuggestions(sessions: WorkoutSession[]): Suggestion[] | null {
  if (sessions.length === 0) return null

  const mostRecent = sessions[0]
  if (!mostRecent?.sets[0]?.weight || !mostRecent?.sets[0]?.reps) return null

  // Find the best available data for each set
  // For sets missing in recent session, look at earlier sessions
  const getBestSetData = (setIndex: number): WorkoutSet | null => {
    for (const session of sessions) {
      const set = session.sets[setIndex]
      if (set?.weight && set?.reps) {
        return set
      }
    }
    return null
  }

  // Calculate suggestion from a single set's data
  const calculateFromSet = (set: WorkoutSet, setIndex: number): Suggestion => {
    const targetReps = [12, 10, 8]
    const setReps = set.reps ?? 0
    const setWeight = set.weight ?? 0
    if (setReps >= 10) {
      // PROGRESS
      return { weight: setWeight + 5, reps: targetReps[setIndex] }
    } else {
      // MAINTAIN
      return { weight: setWeight, reps: Math.min(targetReps[setIndex], setReps + 1) }
    }
  }

  // Get the most recent S1 for base progression decision
  const s1 = mostRecent.sets[0]
  const shouldProgress = s1.reps! >= 10

  // Calculate suggestions for each set
  const suggestions: Suggestion[] = []

  for (let i = 0; i < 3; i++) {
    const recentSet = mostRecent.sets[i]
    const hasRecentData = recentSet?.weight && recentSet?.reps

    if (hasRecentData) {
      // Use the most recent data directly
      suggestions.push(calculateFromSet(recentSet, i))
    } else {
      // Recent session is incomplete for this set - look for earlier data
      const earlierSet = getBestSetData(i)

      if (earlierSet) {
        // Calculate suggestion from earlier session
        const earlierSuggestion = calculateFromSet(earlierSet, i)

        // Also calculate a suggestion based on the recent S1 progression
        const baseWeight = s1.weight!
        const setOffset = i * 5 // S2 is +5kg, S3 is +10kg from S1 base
        const progressionSuggestion: Suggestion = shouldProgress
          ? { weight: baseWeight + 5 + setOffset, reps: [12, 10, 8][i] }
          : { weight: baseWeight + setOffset, reps: [12, 10, 8][i] }

        // Use the HIGHER weight suggestion
        if (earlierSuggestion.weight >= progressionSuggestion.weight) {
          suggestions.push(earlierSuggestion)
        } else {
          suggestions.push(progressionSuggestion)
        }
      } else {
        // No earlier data either - derive from S1
        const baseWeight = s1.weight!
        const setOffset = i * 5
        suggestions.push(
          shouldProgress
            ? { weight: baseWeight + 5 + setOffset, reps: [12, 10, 8][i] }
            : { weight: baseWeight + setOffset, reps: [12, 10, 8][i] }
        )
      }
    }
  }

  return suggestions
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
  const [minimizedTimerState, setMinimizedTimerState] = useState<{ timeLeft: number; duration: number; isRunning: boolean } | null>(null)
  const [completedSets, setCompletedSets] = useState<Set<number>>(new Set()) // Track sets marked done by user

  // Ref for scroll container to auto-scroll to show Target column
  const scrollContainerRef = useRef<HTMLDivElement>(null)

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

  // Auto-scroll to show Target column on load
  useEffect(() => {
    if (isLoaded && scrollContainerRef.current) {
      // Small delay to ensure DOM is rendered
      setTimeout(() => {
        if (scrollContainerRef.current) {
          scrollContainerRef.current.scrollLeft = scrollContainerRef.current.scrollWidth
        }
      }, 50)
    }
  }, [isLoaded])

  // Calculate smart suggestions based on past sessions
  // Handles incomplete sessions by looking at earlier complete data
  const suggestions = useMemo(() => {
    if (pastSessions.length === 0) return null
    return calculateSuggestions(pastSessions)
  }, [pastSessions])

  // Check if suggested S3 weight would be a new PR
  const prEncouragement = useMemo(() => {
    if (!suggestions) return null
    const suggestedMax = Math.max(...suggestions.map(s => s.weight))
    const currentPR = getExercisePR(profileId, exerciseId)
    const displayMax = formatWeightValue(suggestedMax, unit)
    if (suggestedMax > currentPR && currentPR > 0) {
      return `💪 Go for ${displayMax}${unit}! That's a new PR!`
    }
    if (suggestedMax > currentPR && currentPR === 0 && pastSessions.length > 0) {
      return `🔥 Time to progress! Aim for ${displayMax}${unit}!`
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
    const numValue = value === '' ? null : parseFloat(value)
    if (value !== '' && (isNaN(numValue!) || numValue! < 0)) return

    // Convert weight from display unit to storage unit (kg)
    const storageValue = field === 'weight' && numValue !== null
      ? Math.round(convertToKg(numValue, unit))
      : numValue

    const newSets = [...todaySets]
    newSets[setIndex] = {
      ...newSets[setIndex],
      [field]: storageValue
    }
    setTodaySets(newSets)

    // Auto-save
    saveWorkoutSession(profileId, exerciseId, newSets)

    // Remove from completed sets if user edits a completed record
    if (completedSets.has(setIndex)) {
      setCompletedSets(prev => {
        const next = new Set(prev)
        next.delete(setIndex)
        return next
      })
    }

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

  // Handle "Done" button - marks set complete and optionally starts timer based on settings
  const handleSetDone = (setIndex: number) => {
    const currentSet = todaySets[setIndex]
    if (!currentSet.weight || currentSet.weight <= 0) return

    // Mark this set as completed by user
    setCompletedSets(prev => new Set(prev).add(setIndex))

    // Check global timer settings - only trigger timer if auto-start is enabled
    const timerSettings = getTimerSettings()
    if (!timerSettings.autoStart) {
      // Auto-start disabled - just mark set complete, no timer
      return
    }

    // Set timer info for this set (convert weight to display unit)
    setTimerSetInfo({
      setNumber: setIndex + 1,
      weight: formatWeightValue(currentSet.weight, unit),
      reps: currentSet.reps
    })

    // Clear any minimized state - we're starting fresh
    setMinimizedTimerState(null)

    // Increment trigger to reset/restart the timer
    setTimerTrigger(prev => prev + 1)

    // Open full-screen timer
    setShowFullScreenTimer(true)
  }

  // Handle timer minimize (double-tap on full-screen)
  const handleTimerMinimize = (state: { timeLeft: number; duration: number; isRunning: boolean }) => {
    setMinimizedTimerState(state)
    setShowFullScreenTimer(false)
  }

  // Handle timer close - resets timer state
  const handleTimerClose = () => {
    setShowFullScreenTimer(false)
    setMinimizedTimerState(null)
    // Reset timer trigger to stop any running timer
    setTimerTrigger(prev => prev + 1)
    setTimerSetInfo(null)
  }

  // Handle expanding RestTimer back to full-screen - receive current state for sync
  const handleTimerExpand = (state: { timeLeft: number; duration: number; isRunning: boolean }) => {
    setMinimizedTimerState(state)
    setShowFullScreenTimer(true)
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

      {/* Full Screen Timer Modal */}
      {showFullScreenTimer && (
        <FullScreenTimer
          exerciseId={exerciseId}
          exerciseName={exercise?.name}
          setNumber={timerSetInfo?.setNumber}
          weight={timerSetInfo?.weight}
          reps={timerSetInfo?.reps ?? undefined}
          onClose={handleTimerClose}
          onMinimize={handleTimerMinimize}
          autoStart={!minimizedTimerState && getTimerSettings().autoStart}
          initialTimeLeft={minimizedTimerState?.timeLeft}
          initialDuration={minimizedTimerState?.duration}
          initialIsRunning={minimizedTimerState?.isRunning}
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
                    <span className="text-orange-500 font-medium">{formatWeightValue(set.weight, unit)}{unit}</span>
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

      {/* Main Workout Entry - Horizontal scroll: History (left) → Today → Target (right) */}
      <div
        ref={scrollContainerRef}
        className="flex gap-1.5 overflow-x-auto pb-2 scrollbar-visible -mx-4 px-4"
        style={{ WebkitOverflowScrolling: 'touch', touchAction: 'pan-y pan-x' }}
      >
        {/* Past Sessions columns (LEFT side - history first) */}
        {pastSessions.slice().reverse().map((session, idx) => (
          <div key={idx} className="flex flex-col gap-1.5 flex-shrink-0 min-w-[70px]">
            <div className="text-center py-1.5">
              <span className="text-[11px] text-gray-500 dark:text-gray-400 font-medium">
                {formatSessionDate(session.date)}
              </span>
            </div>
            {[0, 1, 2].map(setIndex => {
              const set = session.sets[setIndex]
              const hasData = set?.weight !== null && set?.reps !== null
              return (
                <div
                  key={setIndex}
                  className="h-11 flex items-center justify-center rounded-lg bg-gray-100 dark:bg-gray-700/50 px-1.5"
                >
                  {hasData ? (
                    <span className="text-sm font-bold text-gray-600 dark:text-gray-300">
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

        {/* TODAY column - inputs in the middle */}
        <div className="flex flex-col gap-1.5 flex-shrink-0 min-w-[145px] flex-1">
          <div className="text-center py-1.5">
            <span className="text-[11px] text-blue-600 dark:text-blue-400 font-bold uppercase tracking-wider">Today</span>
          </div>
          {[0, 1, 2].map(setIndex => {
            const todaySet = todaySets[setIndex]
            const suggestion = suggestions?.[setIndex]
            const isCompleted = completedSets.has(setIndex)
            const hasWeight = todaySet.weight !== null && todaySet.weight > 0

            return (
              <div
                key={setIndex}
                className={`h-11 flex items-center gap-0.5 px-1.5 rounded-lg border transition-colors ${
                  isCompleted
                    ? 'bg-green-50 dark:bg-green-900/20 border-green-300 dark:border-green-700'
                    : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700'
                }`}
              >
                {/* Weight */}
                <input
                  type="number"
                  inputMode="numeric"
                  placeholder={suggestion ? String(formatWeightValue(suggestion.weight, unit)) : '-'}
                  value={todaySet.weight !== null ? formatWeightValue(todaySet.weight, unit) : ''}
                  onChange={(e) => handleSetChange(setIndex, 'weight', e.target.value)}
                  onFocus={(e) => e.target.select()}
                  className="w-11 text-center text-base font-bold bg-transparent text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 border-0 focus:outline-none focus:ring-0"
                />
                <span className="text-gray-400 dark:text-gray-500 text-sm">×</span>
                {/* Reps */}
                <input
                  type="number"
                  inputMode="numeric"
                  placeholder={suggestion ? String(suggestion.reps) : SUGGESTED_REPS[setIndex].toString()}
                  value={todaySet.reps ?? ''}
                  onChange={(e) => handleSetChange(setIndex, 'reps', e.target.value)}
                  onFocus={(e) => e.target.select()}
                  className="w-8 text-center text-base font-bold bg-transparent text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 border-0 focus:outline-none focus:ring-0"
                />
                {/* Done checkmark button */}
                <button
                  onClick={() => handleSetDone(setIndex)}
                  disabled={!hasWeight}
                  className={`ml-0.5 w-7 h-7 flex-shrink-0 rounded-full flex items-center justify-center transition-all ${
                    isCompleted
                      ? 'bg-green-500 text-white'
                      : hasWeight
                        ? 'bg-gray-100 dark:bg-gray-700 text-gray-400 hover:bg-green-100 hover:text-green-600 dark:hover:bg-green-900/30 dark:hover:text-green-400'
                        : 'bg-gray-50 dark:bg-gray-800 text-gray-300 dark:text-gray-600 cursor-not-allowed'
                  }`}
                  title={isCompleted ? 'Set completed' : 'Mark set as done'}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                  </svg>
                </button>
              </div>
            )
          })}
        </div>

        {/* Chevron buttons column - copy from target */}
        {suggestions && (
          <div className="flex flex-col gap-1.5 flex-shrink-0">
            {/* Header placeholder - matches other column headers exactly */}
            <div className="text-center py-1.5">
              <span className="text-[11px] font-bold uppercase tracking-wider invisible">—</span>
            </div>
            {[0, 1, 2].map(setIndex => (
              <button
                key={setIndex}
                onClick={() => copyToToday(setIndex)}
                className="h-11 px-1 flex items-center justify-center text-blue-500 dark:text-blue-400 hover:text-blue-600 dark:hover:text-blue-300 transition-colors"
                title="Copy to today"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
                </svg>
              </button>
            ))}
          </div>
        )}

        {/* TARGET column (RIGHT side) */}
        {suggestions && (
          <div className="flex flex-col gap-1.5 flex-shrink-0 min-w-[75px]">
            <div className="text-center py-1.5">
              <span className="text-[11px] text-green-600 dark:text-green-400 font-bold uppercase tracking-wider">🎯 Target</span>
            </div>
            {[0, 1, 2].map(setIndex => {
              const suggestion = suggestions[setIndex]
              return (
                <button
                  key={setIndex}
                  onClick={() => copyToToday(setIndex)}
                  className="h-11 px-2 rounded-lg bg-green-100 dark:bg-green-900/30 border border-green-200 dark:border-green-800 flex items-center justify-center hover:bg-green-200 dark:hover:bg-green-800/50 transition-colors"
                >
                  <span className="text-sm font-bold text-green-700 dark:text-green-400">
                    {suggestion?.weight ? formatWeightValue(suggestion.weight, unit) : '-'}×{suggestion?.reps ?? '-'}
                  </span>
                </button>
              )
            })}
          </div>
        )}
      </div>

      {/* Rep scheme suggestion */}
      <p className="text-[10px] text-gray-400 dark:text-gray-500 mt-2 text-center">
        Suggested reps: 12 / 10 / 8
      </p>

      {/* Rest Timer Preview - Double-click to open full-screen */}
      {showTimer && (
        <div className="mt-4">
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
            key={minimizedTimerState ? `minimized-${timerTrigger}` : `fresh-${timerTrigger}`}
            exerciseId={exerciseId}
            exerciseName={exercise?.name}
            setNumber={timerSetInfo?.setNumber}
            weight={timerSetInfo?.weight}
            reps={timerSetInfo?.reps ?? undefined}
            onExpand={handleTimerExpand}
            autoStart={false}
            initialTimeLeft={minimizedTimerState?.timeLeft}
            initialDuration={minimizedTimerState?.duration}
            initialIsRunning={minimizedTimerState?.isRunning}
          />
        </div>
      )}
    </div>
  )
}
