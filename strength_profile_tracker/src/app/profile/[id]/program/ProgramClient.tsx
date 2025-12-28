'use client'

import { useEffect, useState, use } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { Profile } from '@/types'
import { getProfileById } from '@/lib/storage/profiles'
import {
  getAllRoutines,
  getSelectedRoutineId,
  setSelectedRoutine,
  clearSelectedRoutine,
  WorkoutRoutine,
  getSelectedExercises,
  toggleExerciseSelection,
  initializeDefaultExercises
} from '@/lib/storage/routines'
import { Exercise } from '@/types'
import { EXERCISES } from '@/lib/calculations/strength'
import { ThemeToggle, UnitToggle, Logo } from '@/components/ui'
import { ContextualTip } from '@/components/onboarding'

interface ProgramClientProps {
  params: Promise<{ id: string }>
}

export default function ProgramClient({ params }: ProgramClientProps) {
  const { id } = use(params)
  const router = useRouter()
  const [profile, setProfile] = useState<Profile | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [routines, setRoutines] = useState<WorkoutRoutine[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [expandedRoutine, setExpandedRoutine] = useState<string | null>(null)
  const [selectedExercises, setSelectedExercises] = useState<Exercise[]>([])

  useEffect(() => {
    const loadedProfile = getProfileById(id)
    setProfile(loadedProfile)
    setRoutines(getAllRoutines())
    setSelectedId(getSelectedRoutineId())
    setSelectedExercises(getSelectedExercises())
    setIsLoading(false)
  }, [id])

  const handleSelectRoutine = (routineId: string) => {
    if (selectedId === routineId) {
      // Deselect
      clearSelectedRoutine()
      setSelectedId(null)
      setSelectedExercises([])
    } else {
      setSelectedRoutine(routineId)
      setSelectedId(routineId)
      // Initialize default exercises when selecting a routine
      const routine = routines.find(r => r.id === routineId)
      if (routine) {
        const defaultExercises = initializeDefaultExercises(routine)
        setSelectedExercises(defaultExercises)
      }
    }
  }

  const handleToggleExercise = (exerciseId: Exercise) => {
    const updated = toggleExerciseSelection(exerciseId)
    setSelectedExercises(updated)
  }

  const toggleExpand = (routineId: string) => {
    setExpandedRoutine(expandedRoutine === routineId ? null : routineId)
  }

  const getExerciseName = (exerciseId: string): string => {
    const exercise = EXERCISES.find(e => e.id === exerciseId)
    return exercise?.name || exerciseId
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-gray-500">Loading...</div>
      </div>
    )
  }

  if (!profile) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center p-4">
        <h2 className="text-lg font-semibold text-[#2C3E50] mb-2">
          Profile not found
        </h2>
        <Link href="/" className="text-blue-500 hover:underline">
          Back to Profiles
        </Link>
      </div>
    )
  }

  return (
    <div className="min-h-screen pb-20">
      {/* Header */}
      <header className="bg-[#2C3E50] text-white px-4 py-3">
        <div className="flex items-center justify-between">
          {/* Left: Back + Title */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => router.back()}
              className="text-white hover:text-gray-300 -ml-1"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
            <Logo size="sm" showText={false} />
          </div>

          {/* Center: Title */}
          <h1 className="text-lg font-semibold">Training Program</h1>

          {/* Right: Toggles */}
          <div className="flex items-center gap-2">
            <UnitToggle />
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="p-4 max-w-lg mx-auto">
        {/* Intro */}
        <div className="mb-6">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Select a training program to organize your workouts. The app will suggest exercises based on your program&apos;s schedule.
          </p>
        </div>

        {/* Current Selection */}
        {selectedId && (
          <div className="mb-6 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
            <div className="flex items-center gap-2 text-green-700 dark:text-green-400">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              <span className="font-medium">
                Active: {routines.find(r => r.id === selectedId)?.name}
              </span>
            </div>
            <p className="text-xs text-green-600 dark:text-green-500 mt-2 pl-7">
              {selectedExercises.length} exercise{selectedExercises.length !== 1 ? 's' : ''} selected • Tap exercises below to toggle
            </p>
          </div>
        )}

        {/* Routines List */}
        <div className="space-y-3">
          {routines.map(routine => {
            const isSelected = selectedId === routine.id
            const isExpanded = expandedRoutine === routine.id

            return (
              <div
                key={routine.id}
                className={`bg-white dark:bg-gray-800 rounded-lg border overflow-hidden transition-all ${
                  isSelected
                    ? 'border-green-500 dark:border-green-600 ring-1 ring-green-500/20'
                    : 'border-gray-200 dark:border-gray-700'
                }`}
              >
                {/* Routine Header */}
                <div className="p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <h3 className="font-semibold text-[#2C3E50] dark:text-gray-100 truncate">
                          {routine.name}
                        </h3>
                        {routine.isCustom && (
                          <span className="text-xs px-2 py-0.5 bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 rounded-full">
                            Custom
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                        {routine.description}
                      </p>
                      <div className="flex items-center gap-3 mt-2 text-xs text-gray-400 dark:text-gray-500">
                        <span className="flex items-center gap-1">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                          </svg>
                          {routine.daysPerWeek} days/week
                        </span>
                        <span className="flex items-center gap-1">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                          </svg>
                          {routine.days.length} workouts
                        </span>
                      </div>
                    </div>

                    {/* Select Button */}
                    <button
                      onClick={() => handleSelectRoutine(routine.id)}
                      className={`shrink-0 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                        isSelected
                          ? 'bg-green-500 text-white'
                          : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                      }`}
                    >
                      {isSelected ? 'Active' : 'Select'}
                    </button>
                  </div>

                  {/* Expand/Collapse */}
                  <button
                    onClick={() => toggleExpand(routine.id)}
                    className="mt-3 flex items-center gap-1 text-xs text-blue-500 hover:text-blue-600 dark:text-blue-400"
                  >
                    <svg
                      className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                    {isExpanded ? 'Hide details' : 'View workout days'}
                  </button>
                </div>

                {/* Expanded: Workout Days */}
                {isExpanded && (
                  <div className="border-t border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 p-4">
                    <div className="space-y-3">
                      {routine.days.map((day, index) => (
                        <div key={index} className="bg-white dark:bg-gray-800 rounded-lg p-3 border border-gray-100 dark:border-gray-700">
                          <h4 className="font-medium text-sm text-[#2C3E50] dark:text-gray-200 mb-2">
                            {day.name}
                          </h4>
                          <div className="flex flex-wrap gap-1.5">
                            {day.exercises.map((exerciseId, exIndex) => {
                              const isExerciseSelected = selectedExercises.includes(exerciseId)
                              return (
                                <button
                                  key={exIndex}
                                  onClick={() => isSelected && handleToggleExercise(exerciseId)}
                                  disabled={!isSelected}
                                  className={`text-xs px-2 py-1 rounded transition-all ${
                                    isSelected
                                      ? isExerciseSelected
                                        ? 'bg-green-500 text-white shadow-sm'
                                        : 'bg-gray-200 dark:bg-gray-600 text-gray-500 dark:text-gray-400 hover:bg-gray-300 dark:hover:bg-gray-500'
                                      : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                                  }`}
                                >
                                  {getExerciseName(exerciseId)}
                                </button>
                              )
                            })}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {/* Create Custom (Coming Soon) */}
        <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-dashed border-gray-300 dark:border-gray-600">
          <div className="text-center">
            <svg className="w-8 h-8 mx-auto text-gray-400 dark:text-gray-500 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
            </svg>
            <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
              Create Custom Program
            </p>
            <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
              Coming soon
            </p>
          </div>
        </div>
      </main>

      {/* Contextual Tip */}
      <ContextualTip
        tipId="program_intro"
        title="Choose Your Training Split"
        message="Select a program and tap the exercises to customize which ones you want to track. Green = selected, grey = skipped."
        icon="🏋️"
        position="bottom"
      />
    </div>
  )
}
