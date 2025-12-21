'use client'

import { useEffect, useState, use } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { Profile, Exercise, Level, BodyPart, SEX_INFO, LEVEL_COLORS } from '@/types'
import { getProfileById, updateExerciseRating, deleteProfile } from '@/lib/storage/profiles'
import { getExercisesWithHistory } from '@/lib/storage/workouts'
import {
  getRatedExercises,
  getUnratedExercises,
  calculateOverallLevel,
  calculateStrengthScore,
  generateCoachTips,
  getRatedCount,
  getTotalExercisesCount
} from '@/lib/calculations/strength'
import { ExerciseRatings } from '@/types'
import { formatWeightValue, getProfileColor } from '@/lib/utils/units'
import { useUnit } from '@/contexts'
import { Button, ThemeToggle, UnitToggle, Logo } from '@/components/ui'
import { StrengthCard, LevelLegend, BodyPartFilter } from '@/components/strength'
import { MotivationalQuote } from '@/components/quotes'

interface ProfileDetailPageProps {
  params: Promise<{ id: string }>
}

export default function ProfileDetailPage({ params }: ProfileDetailPageProps) {
  const { id } = use(params)
  const router = useRouter()
  const { unit } = useUnit()
  const [profile, setProfile] = useState<Profile | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [bodyPartFilter, setBodyPartFilter] = useState<BodyPart | 'all'>('all')

  useEffect(() => {
    const loadedProfile = getProfileById(id)
    setProfile(loadedProfile)
    setIsLoading(false)

    // Save this profile as the last visited for auto-redirect on next app load
    if (loadedProfile) {
      localStorage.setItem('lastVisitedProfile', id)
    }
  }, [id])

  const handleLevelSelect = (exercise: Exercise, level: Level) => {
    if (!profile) return

    const updatedProfile = updateExerciseRating(profile.id, exercise, level)
    setProfile(updatedProfile)
  }

  const handleDelete = () => {
    if (!profile) return

    deleteProfile(profile.id)
    router.push('/')
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
        <p className="text-gray-500 mb-4">
          This profile may have been deleted.
        </p>
        <Link href="/">
          <Button variant="primary">Back to Profiles</Button>
        </Link>
      </div>
    )
  }

  // Filter ratings to only include exercises with actual workout history
  // This prevents showing levels for exercises that were pre-set but never logged
  const exercisesWithHistory = getExercisesWithHistory(profile.id)
  const validatedRatings: ExerciseRatings = {}
  Object.entries(profile.exerciseRatings).forEach(([exercise, level]) => {
    if (exercisesWithHistory.has(exercise as Exercise)) {
      validatedRatings[exercise as Exercise] = level
    }
  })

  const ratedExercises = getRatedExercises(profile.weight, validatedRatings, bodyPartFilter, profile.sex)
  const unratedExercises = getUnratedExercises(profile.weight, validatedRatings, bodyPartFilter, profile.sex)
  const overallLevel = calculateOverallLevel(validatedRatings)
  const strengthScore = calculateStrengthScore(validatedRatings)
  const coachTips = generateCoachTips(validatedRatings)
  const ratedCount = getRatedCount(validatedRatings)
  const totalCount = getTotalExercisesCount()

  return (
    <div className="min-h-screen pb-20">
      {/* Header */}
      <header className="bg-[#2C3E50] text-white px-4 py-3">
        <div className="flex items-center justify-between">
          {/* Left: Back + Logo */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                localStorage.removeItem('lastVisitedProfile')
                router.push('/')
              }}
              className="text-white hover:text-gray-300 -ml-1"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
            <Logo size="sm" showText={false} />
          </div>

          {/* Center: Quote */}
          <div className="flex-1 mx-3 min-w-0 overflow-hidden">
            <MotivationalQuote variant="compact" />
          </div>

          {/* Right: Toggles */}
          <div className="flex items-center gap-2">
            <UnitToggle />
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="p-4 max-w-lg mx-auto">
        {/* Profile Info */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-[#E0E0E0] dark:border-gray-700 p-4 mb-4">
          <div className="flex items-center gap-4">
            {/* Avatar */}
            <div className="relative">
              <div className="w-16 h-16 rounded-full overflow-hidden">
                {profile.avatarUrl ? (
                  <img
                    src={profile.avatarUrl}
                    alt={profile.name}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div
                    className="w-full h-full flex items-center justify-center text-white font-semibold text-xl"
                    style={{ backgroundColor: getProfileColor(profile.id) }}
                  >
                    {profile.name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)}
                  </div>
                )}
              </div>
              {profile.sex && (
                <span className={`absolute -bottom-1 -right-1 w-6 h-6 rounded-full flex items-center justify-center text-sm ${
                  profile.sex === 'male' ? 'bg-blue-500' : 'bg-pink-500'
                } text-white border-2 border-white dark:border-gray-800`}>
                  {SEX_INFO[profile.sex].icon}
                </span>
              )}
            </div>

            {/* Info */}
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <h2 className="font-semibold text-[#2C3E50] dark:text-gray-100 text-lg">
                    {profile.name}
                  </h2>
                  {/* Overall Level Badge */}
                  {overallLevel && (() => {
                    const levelKey = overallLevel.toLowerCase() as Level
                    const bgColor = LEVEL_COLORS[levelKey] || '#9CA3AF'
                    return (
                      <span
                        className="text-xs px-2 py-0.5 rounded-full text-white font-medium"
                        style={{ backgroundColor: bgColor }}
                      >
                        {overallLevel}
                      </span>
                    )
                  })()}
                </div>
                <Link
                  href={`/profile/${profile.id}/edit`}
                  className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 p-1"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                  </svg>
                </Link>
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-2 grid grid-cols-4 gap-2">
                <div className="text-center">
                  <span className="block font-medium text-[#2C3E50] dark:text-gray-200">{profile.age}</span>
                  <span className="text-xs">years</span>
                </div>
                <div className="text-center">
                  <span className="block font-medium text-[#2C3E50] dark:text-gray-200">{profile.height}</span>
                  <span className="text-xs">cm</span>
                </div>
                <div className="text-center">
                  <span className="block font-medium text-[#2C3E50] dark:text-gray-200">{formatWeightValue(profile.weight, unit)}</span>
                  <span className="text-xs">{unit}</span>
                </div>
                <div className="text-center">
                  {(() => {
                    const bmi = profile.weight / Math.pow(profile.height / 100, 2)
                    return (
                      <>
                        <span className="block font-medium text-[#2C3E50] dark:text-gray-200">{bmi.toFixed(1)}</span>
                        <span className="text-xs">BMI</span>
                      </>
                    )
                  })()}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Strength Score and Progress Link - side by side */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          {/* Strength Score with rated count */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-[#E0E0E0] dark:border-gray-700 p-4">
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-1 text-center">Strength Score</p>
            <div className="flex items-center justify-center gap-3">
              <div className="relative w-16 h-16">
                <svg className="w-16 h-16 -rotate-90" viewBox="0 0 36 36">
                  <path
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none"
                    stroke="#E5E7EB"
                    strokeWidth="3"
                    className="dark:stroke-gray-700"
                  />
                  <path
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none"
                    stroke="#F97316"
                    strokeWidth="3"
                    strokeDasharray={`${strengthScore}, 100`}
                    strokeLinecap="round"
                  />
                </svg>
                <span className="absolute inset-0 flex items-center justify-center text-xl font-bold text-[#2C3E50] dark:text-gray-100">
                  {strengthScore}
                </span>
              </div>
              <div className="text-center">
                <span className="block text-lg font-semibold text-[#2C3E50] dark:text-gray-200">{ratedCount}/{totalCount}</span>
                <span className="text-xs text-gray-500 dark:text-gray-400">rated</span>
              </div>
            </div>
          </div>

          {/* Progress Link */}
          <Link
            href={`/profile/${profile.id}/progress`}
            className="bg-white dark:bg-gray-800 rounded-lg border border-[#E0E0E0] dark:border-gray-700 p-3 hover:bg-gray-50 dark:hover:bg-gray-700 transition-all flex flex-col items-center justify-center"
          >
            {/* Simple Progress Chart Icon */}
            <svg className="w-12 h-12 mb-1" viewBox="0 0 48 48" fill="none">
              {/* Bar chart with trending up */}
              <rect x="6" y="28" width="8" height="14" rx="2" fill="#F97316" opacity="0.5"/>
              <rect x="17" y="20" width="8" height="22" rx="2" fill="#F97316" opacity="0.7"/>
              <rect x="28" y="12" width="8" height="30" rx="2" fill="#F97316"/>
              {/* Trend line */}
              <path d="M10 26 L21 18 L32 10" stroke="#22c55e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
              {/* Arrow head */}
              <path d="M28 8 L33 10 L30 14" stroke="#22c55e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
            </svg>
            <span className="text-sm font-medium text-[#2C3E50] dark:text-gray-200">View Progress</span>
            <span className="text-xs text-gray-500 dark:text-gray-400">Charts & History</span>
          </Link>
        </div>

        {/* Coach Tip Hint */}
        {coachTips.length > 0 && (
          <div className="mb-4 px-3 py-2 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700">
            <p className="text-xs text-gray-600 dark:text-gray-400 flex items-center gap-2">
              <span>{coachTips[0].icon}</span>
              <span>{coachTips[0].message}</span>
            </p>
          </div>
        )}

        {/* Body Part Filter */}
        <div className="mb-4">
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">Filter by body part:</p>
          <BodyPartFilter
            selected={bodyPartFilter}
            onChange={setBodyPartFilter}
          />
        </div>

        {/* Rated Exercises */}
        {ratedExercises.length > 0 && (
          <div className="mb-6">
            <h3 className="font-semibold text-[#2C3E50] dark:text-gray-100 text-sm mb-3 flex items-center gap-2">
              <span className="w-2 h-2 bg-[#27AE60] rounded-full"></span>
              Your Rated Exercises ({ratedExercises.length})
            </h3>
            {ratedExercises.map(strength => (
              <StrengthCard
                key={strength.exercise}
                strength={strength}
                onLevelSelect={(level) => handleLevelSelect(strength.exercise, level)}
                showBodyPart={bodyPartFilter === 'all'}
                profileId={profile.id}
              />
            ))}
          </div>
        )}

        {/* Unrated Exercises */}
        {unratedExercises.length > 0 && (
          <div className="mb-6">
            <h3 className="font-semibold text-[#2C3E50] dark:text-gray-100 text-sm mb-3 flex items-center gap-2">
              <span className="w-2 h-2 bg-gray-300 dark:bg-gray-600 rounded-full"></span>
              Not Yet Rated ({unratedExercises.length})
            </h3>
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
              Tap a level to rate your current performance.
            </p>
            {unratedExercises.map(strength => (
              <StrengthCard
                key={strength.exercise}
                strength={strength}
                onLevelSelect={(level) => handleLevelSelect(strength.exercise, level)}
                showBodyPart={bodyPartFilter === 'all'}
                profileId={profile.id}
              />
            ))}
          </div>
        )}

        {/* Level Legend */}
        <div className="mt-4">
          <LevelLegend />
        </div>

        {/* Delete Button */}
        <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
          {!showDeleteConfirm ? (
            <Button
              variant="ghost"
              fullWidth
              onClick={() => setShowDeleteConfirm(true)}
              className="text-[#E74C3C]"
            >
              Delete Profile
            </Button>
          ) : (
            <div className="space-y-2">
              <p className="text-sm text-center text-gray-600 dark:text-gray-300">
                Are you sure you want to delete this profile?
              </p>
              <div className="flex gap-2">
                <Button
                  variant="secondary"
                  fullWidth
                  onClick={() => setShowDeleteConfirm(false)}
                >
                  Cancel
                </Button>
                <Button
                  variant="danger"
                  fullWidth
                  onClick={handleDelete}
                >
                  Delete
                </Button>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
