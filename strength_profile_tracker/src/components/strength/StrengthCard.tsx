'use client'

import { useState, useEffect } from 'react'
import { Level, LEVEL_COLORS, LEVEL_NAMES, BODY_PART_NAMES, CalculatedStrength } from '@/types'
import { Card } from '@/components/ui'
import { useUnit } from '@/contexts'
import { formatWeightValue } from '@/lib/utils/units'
import { getTimerSettings } from '@/lib/storage/timer'
import { useWakeLock } from '@/hooks/useWakeLock'
import WorkoutLogger from './WorkoutLogger'

interface StrengthCardProps {
  strength: CalculatedStrength
  onLevelSelect: (level: Level) => void
  showBodyPart?: boolean
  profileId?: string  // Required for workout logging
}

export default function StrengthCard({ strength, onLevelSelect, showBodyPart = false, profileId }: StrengthCardProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const { unit } = useUnit()
  const isRated = strength.levels.some(l => l.isSelected)
  const { requestWakeLock, releaseWakeLock, isActive: isWakeLockActive } = useWakeLock()

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

  return (
    <Card padding="sm" className="mb-3">
      {/* Header - clickable to expand */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between mb-3"
      >
        <div className="flex items-center gap-2">
          <h3 className="font-semibold text-[#2C3E50] dark:text-gray-100 text-sm">
            {strength.exerciseName}
          </h3>
          {strength.isDumbbell && (
            <span className="text-xs text-gray-400 dark:text-gray-500">(per hand)</span>
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

      {/* Workout Logger - shown when expanded */}
      {isExpanded && profileId && (
        <WorkoutLogger
          profileId={profileId}
          exerciseId={strength.exercise}
          onLevelUp={onLevelSelect}
        />
      )}
    </Card>
  )
}
