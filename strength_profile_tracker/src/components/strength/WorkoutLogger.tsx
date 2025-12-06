'use client'

import { useState, useEffect } from 'react'
import { Exercise, WorkoutSession, WorkoutSet, SUGGESTED_REPS } from '@/types'
import {
  getExerciseSessions,
  getTodaySession,
  saveWorkoutSession,
  formatSessionDate,
  createEmptySets,
  getTodayDate
} from '@/lib/storage/workouts'

interface WorkoutLoggerProps {
  profileId: string
  exerciseId: Exercise
}

export default function WorkoutLogger({ profileId, exerciseId }: WorkoutLoggerProps) {
  const [pastSessions, setPastSessions] = useState<WorkoutSession[]>([])
  const [todaySets, setTodaySets] = useState<WorkoutSet[]>(createEmptySets())
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    // Load past sessions (excluding today)
    const sessions = getExerciseSessions(profileId, exerciseId, 4)
      .filter(s => s.date !== getTodayDate())
      .slice(0, 3)
    setPastSessions(sessions)

    // Load today's session if exists
    const today = getTodaySession(profileId, exerciseId)
    if (today) {
      setTodaySets(today.sets)
    }

    setIsLoaded(true)
  }, [profileId, exerciseId])

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
  }

  if (!isLoaded) {
    return (
      <div className="py-2 text-center text-sm text-gray-400">
        Loading...
      </div>
    )
  }

  // Build columns: past sessions (up to 3) + TODAY
  const columns = [
    ...pastSessions.map(s => ({
      date: formatSessionDate(s.date),
      sets: s.sets,
      isToday: false
    })),
    // Fill empty past columns if less than 3
    ...Array(Math.max(0, 3 - pastSessions.length)).fill(null).map(() => ({
      date: '-',
      sets: createEmptySets(),
      isToday: false,
      isEmpty: true
    })),
    {
      date: 'TODAY',
      sets: todaySets,
      isToday: true
    }
  ].slice(-4) // Ensure exactly 4 columns

  return (
    <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-700">
      <p className="text-xs text-gray-500 dark:text-gray-400 mb-2 font-medium">
        Workout Log
      </p>

      {/* Header row with dates */}
      <div className="grid grid-cols-[auto_1fr_1fr_1fr_1fr] gap-1 mb-1">
        <div className="w-8"></div>
        {columns.map((col, i) => (
          <div
            key={i}
            className={`text-center text-[10px] font-medium py-1 rounded ${
              col.isToday
                ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                : 'text-gray-400 dark:text-gray-500'
            }`}
          >
            {col.date}
          </div>
        ))}
      </div>

      {/* Set rows */}
      {[0, 1, 2].map(setIndex => (
        <div key={setIndex} className="grid grid-cols-[auto_1fr_1fr_1fr_1fr] gap-1 mb-1">
          {/* Set label */}
          <div className="w-8 text-[10px] text-gray-400 dark:text-gray-500 flex items-center justify-center">
            S{setIndex + 1}
          </div>

          {/* Past session columns */}
          {columns.slice(0, -1).map((col, colIndex) => {
            const set = col.sets[setIndex]
            const hasData = set.weight !== null && set.reps !== null

            return (
              <div
                key={colIndex}
                className="text-center text-[11px] py-1.5 rounded bg-gray-50 dark:bg-gray-700/50"
              >
                {hasData ? (
                  <span className="text-gray-600 dark:text-gray-300">
                    {set.weight}<span className="text-gray-400">kg</span> x {set.reps}
                  </span>
                ) : (
                  <span className="text-gray-300 dark:text-gray-600">-</span>
                )}
              </div>
            )
          })}

          {/* Today's input column */}
          <div className="flex items-center gap-0.5 bg-blue-50 dark:bg-blue-900/20 rounded px-1 py-0.5">
            <input
              type="number"
              inputMode="numeric"
              placeholder="kg"
              value={todaySets[setIndex].weight ?? ''}
              onChange={(e) => handleSetChange(setIndex, 'weight', e.target.value)}
              className="w-8 text-center text-[11px] bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded px-0.5 py-1 focus:outline-none focus:ring-1 focus:ring-blue-400"
            />
            <span className="text-[10px] text-gray-400">x</span>
            <input
              type="number"
              inputMode="numeric"
              placeholder={SUGGESTED_REPS[setIndex].toString()}
              value={todaySets[setIndex].reps ?? ''}
              onChange={(e) => handleSetChange(setIndex, 'reps', e.target.value)}
              className="w-6 text-center text-[11px] bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded px-0.5 py-1 focus:outline-none focus:ring-1 focus:ring-blue-400"
            />
          </div>
        </div>
      ))}

      {/* Rep scheme suggestion */}
      <p className="text-[10px] text-gray-400 dark:text-gray-500 mt-2 text-center">
        Suggested reps: 12 / 10 / 8
      </p>
    </div>
  )
}
