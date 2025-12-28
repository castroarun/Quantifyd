'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { Level, LEVEL_COLORS, LEVEL_NAMES, BODY_PART_NAMES, CalculatedStrength } from '@/types'
import { Card } from '@/components/ui'
import { useUnit } from '@/contexts'
import { formatWeightValue } from '@/lib/utils/units'
import { getTimerSettings } from '@/lib/storage/timer'
import { useWakeLock } from '@/hooks/useWakeLock'
import { getInProgressExercise, setInProgressExercise, clearInProgressExercise, onInProgressChange } from '@/lib/storage/inProgress'
import { needsGesturesOnboarding } from '@/lib/storage/onboarding'
import { GesturesOnboarding, ContextualTip } from '@/components/onboarding'
import WorkoutLogger from './WorkoutLogger'

// Long press duration in milliseconds
const LONG_PRESS_DURATION = 600

// Double-tap detection threshold
const DOUBLE_TAP_THRESHOLD = 300

interface StrengthCardProps {
  strength: CalculatedStrength
  onLevelSelect: (level: Level) => void
  showBodyPart?: boolean
  profileId?: string  // Required for workout logging
}

export default function StrengthCard({ strength, onLevelSelect, showBodyPart = false, profileId }: StrengthCardProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [isInProgress, setIsInProgress] = useState(false)
  const [isFullScreen, setIsFullScreen] = useState(false)
  const [showOnboarding, setShowOnboarding] = useState(false)
  const { unit } = useUnit()
  const isRated = strength.levels.some(l => l.isSelected)
  const { requestWakeLock, releaseWakeLock, isActive: isWakeLockActive } = useWakeLock()

  // Long press handling refs
  const longPressTimerRef = useRef<NodeJS.Timeout | null>(null)
  const isLongPressRef = useRef(false)

  // Double-tap detection for full-screen minimize
  const fullScreenLastTapRef = useRef<number>(0)

  // Check if this exercise is in-progress on mount and when storage changes
  useEffect(() => {
    const checkInProgress = () => {
      const inProgressExercise = getInProgressExercise()
      setIsInProgress(inProgressExercise === strength.exercise)
    }

    checkInProgress()

    // Subscribe to in-progress changes (same-tab and cross-tab)
    const unsubscribe = onInProgressChange(checkInProgress)
    return unsubscribe
  }, [strength.exercise])

  // Set as in-progress when expanded
  useEffect(() => {
    if (isExpanded) {
      setInProgressExercise(strength.exercise)
      setIsInProgress(true)
    }
  }, [isExpanded, strength.exercise])

  // Manage wake lock when workout logger is expanded
  useEffect(() => {
    const settings = getTimerSettings()

    if (isExpanded && settings.keepAwakeDuringWorkout) {
      // Request wake lock when expanded and setting is enabled
      requestWakeLock()
    } else if (!isExpanded && isWakeLockActive) {
      // Release wake lock when collapsed
      releaseWakeLock()
    }

    // Cleanup on unmount
    return () => {
      if (isWakeLockActive) {
        releaseWakeLock()
      }
    }
  }, [isExpanded, requestWakeLock, releaseWakeLock, isWakeLockActive])

  // Long press handlers for clearing in-progress
  const handlePressStart = useCallback(() => {
    if (!isInProgress || isExpanded) return // Only allow long press when collapsed and in-progress

    isLongPressRef.current = false
    longPressTimerRef.current = setTimeout(() => {
      isLongPressRef.current = true
      clearInProgressExercise()
      setIsInProgress(false)
    }, LONG_PRESS_DURATION)
  }, [isInProgress, isExpanded])

  const handlePressEnd = useCallback(() => {
    if (longPressTimerRef.current) {
      clearTimeout(longPressTimerRef.current)
      longPressTimerRef.current = null
    }
  }, [])

  const handleClick = useCallback(() => {
    // Don't toggle if this was a long press
    if (isLongPressRef.current) {
      isLongPressRef.current = false
      return
    }

    // Show onboarding when first expanding
    if (!isExpanded && needsGesturesOnboarding()) {
      setShowOnboarding(true)
    }

    setIsExpanded(!isExpanded)
  }, [isExpanded])

  // Handler for when WorkoutLogger requests full screen (via double-tap)
  const handleRequestFullScreen = useCallback(() => {
    setIsFullScreen(true)
  }, [])

  // Handler for double-tap on full-screen to minimize
  const handleFullScreenDoubleTap = useCallback((e: React.MouseEvent) => {
    // Only trigger on the container itself, not on interactive children
    if (e.target !== e.currentTarget) return

    const now = Date.now()
    const timeSinceLastTap = now - fullScreenLastTapRef.current
    fullScreenLastTapRef.current = now

    if (timeSinceLastTap < DOUBLE_TAP_THRESHOLD) {
      setIsFullScreen(false)
    }
  }, [])

  // Show indicator when this exercise is in-progress (even when collapsed)
  const showInProgressIndicator = isInProgress || isExpanded

  return (
    <Card
      padding="sm"
      className={`mb-3 transition-all ${
        showInProgressIndicator
          ? 'ring-2 ring-blue-400 dark:ring-blue-500 ring-offset-2 dark:ring-offset-gray-900'
          : ''
      }`}
    >
      {/* Header - clickable to expand, long-press to clear in-progress */}
      <button
        onClick={handleClick}
        onMouseDown={handlePressStart}
        onMouseUp={handlePressEnd}
        onMouseLeave={handlePressEnd}
        onTouchStart={handlePressStart}
        onTouchEnd={handlePressEnd}
        onTouchCancel={handlePressEnd}
        className="w-full flex items-center justify-between mb-3"
      >
        <div className="flex items-center gap-2">
          {/* In-progress indicator dot */}
          {showInProgressIndicator && (
            <span className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
          )}
          <h3 className="font-semibold text-[#2C3E50] dark:text-gray-100 text-sm">
            {strength.exerciseName}
          </h3>
          {strength.isDumbbell && (
            <span className="text-xs text-gray-400 dark:text-gray-500">(per hand)</span>
          )}
          {/* In-progress badge */}
          {showInProgressIndicator && (
            <span className="text-[10px] px-1.5 py-0.5 bg-blue-100 dark:bg-blue-900/40 text-blue-600 dark:text-blue-400 rounded-full font-medium">
              Active
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {showBodyPart && (
            <span className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 rounded">
              {BODY_PART_NAMES[strength.bodyPart]}
            </span>
          )}
          {/* Expand/collapse indicator */}
          <svg
            className={`w-4 h-4 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </div>
      </button>

      {/* Level standards - display only (determined by workout performance) */}
      <div className="grid grid-cols-4 gap-2">
        {strength.levels.map(({ level, weight, isSelected }) => (
          <div
            key={level}
            className={`
              relative p-2 rounded-lg border-2 transition-all
              flex flex-col items-center justify-center
              min-h-[70px]
              ${isSelected
                ? 'border-[#2C3E50] dark:border-gray-400'
                : 'border-transparent bg-gray-50 dark:bg-gray-700'
              }
            `}
            style={{
              backgroundColor: isSelected ? `${LEVEL_COLORS[level]}15` : undefined
            }}
          >
            {/* Checkmark for current level */}
            {isSelected && (
              <div
                className="absolute top-1 right-1 w-4 h-4 rounded-full flex items-center justify-center"
                style={{ backgroundColor: LEVEL_COLORS[level] }}
              >
                <svg
                  className="w-3 h-3 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={3}
                    d="M5 13l4 4L19 7"
                  />
                </svg>
              </div>
            )}

            {/* Weight */}
            <span
              className="text-lg font-bold"
              style={{ color: LEVEL_COLORS[level] }}
            >
              {formatWeightValue(weight, unit)}
            </span>
            <span className="text-xs text-gray-500 dark:text-gray-400">{unit}</span>

            {/* Level label */}
            <span
              className="text-xs font-medium mt-1"
              style={{ color: LEVEL_COLORS[level] }}
            >
              {LEVEL_NAMES[level]}
            </span>
          </div>
        ))}
      </div>

      {/* Info text about auto-leveling */}
      {!isRated && (
        <p className="text-[10px] text-gray-400 dark:text-gray-500 mt-2 text-center">
          Log a workout to set your level
        </p>
      )}

      {/* Long press hint - shown when in-progress but collapsed */}
      {isInProgress && !isExpanded && (
        <p className="text-[10px] text-gray-400 dark:text-gray-500 mt-2 text-center italic">
          Long press to remove active status
        </p>
      )}

      {/* Workout Logger - shown when expanded */}
      {isExpanded && profileId && !isFullScreen && (
        <WorkoutLogger
          profileId={profileId}
          exerciseId={strength.exercise}
          onLevelUp={onLevelSelect}
          onRequestFullScreen={handleRequestFullScreen}
        />
      )}

      {/* Full Screen Workout Logger Modal */}
      {isFullScreen && profileId && (
        <div
          className="fixed inset-0 z-50 bg-white dark:bg-gray-900 overflow-auto"
          onClick={handleFullScreenDoubleTap}
        >
          {/* Header */}
          <div className="sticky top-0 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
              <h2 className="font-bold text-lg text-[#2C3E50] dark:text-gray-100">
                {strength.exerciseName}
              </h2>
              {strength.isDumbbell && (
                <span className="text-xs text-gray-400 dark:text-gray-500">(per hand)</span>
              )}
            </div>
            <button
              onClick={() => setIsFullScreen(false)}
              className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            >
              <svg className="w-6 h-6 text-gray-500 dark:text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Level standards in full screen */}
          <div className="p-4">
            <div className="grid grid-cols-4 gap-3 mb-4">
              {strength.levels.map(({ level, weight, isSelected }) => (
                <div
                  key={level}
                  className={`
                    relative p-3 rounded-lg border-2 transition-all
                    flex flex-col items-center justify-center
                    ${isSelected
                      ? 'border-[#2C3E50] dark:border-gray-400'
                      : 'border-transparent bg-gray-50 dark:bg-gray-700'
                    }
                  `}
                  style={{
                    backgroundColor: isSelected ? `${LEVEL_COLORS[level]}15` : undefined
                  }}
                >
                  {isSelected && (
                    <div
                      className="absolute top-1 right-1 w-4 h-4 rounded-full flex items-center justify-center"
                      style={{ backgroundColor: LEVEL_COLORS[level] }}
                    >
                      <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                  )}
                  <span className="text-xl font-bold" style={{ color: LEVEL_COLORS[level] }}>
                    {formatWeightValue(weight, unit)}
                  </span>
                  <span className="text-xs text-gray-500 dark:text-gray-400">{unit}</span>
                  <span className="text-xs font-medium mt-1" style={{ color: LEVEL_COLORS[level] }}>
                    {LEVEL_NAMES[level]}
                  </span>
                </div>
              ))}
            </div>

            {/* Workout Logger in full screen */}
            <WorkoutLogger
              profileId={profileId}
              exerciseId={strength.exercise}
              onLevelUp={onLevelSelect}
            />

            {/* Double-tap hint */}
            <p className="text-[10px] text-gray-400 dark:text-gray-500 mt-4 text-center italic">
              Double-tap empty space to minimize
            </p>
          </div>
        </div>
      )}

      {/* Gestures Onboarding Overlay */}
      {showOnboarding && (
        <GesturesOnboarding onComplete={() => setShowOnboarding(false)} />
      )}

      {/* Contextual Tip - Workout timer settings (shows when expanded) */}
      {isExpanded && (
        <ContextualTip
          tipId="workout_timer_settings"
          title="Rest Timer Settings"
          message="Timer starts automatically after logging a set. Adjust rest time per exercise—your changes are saved. Set defaults in Settings."
          icon="💪"
          position="bottom"
        />
      )}
    </Card>
  )
}
