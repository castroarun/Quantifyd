'use client'

import { useEffect, useState, use } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { Profile, Exercise, Level, BodyPart, SEX_INFO, ACTIVITY_LEVEL_INFO, GOAL_INFO } from '@/types'
import { getProfileById, updateExerciseRating, deleteProfile } from '@/lib/storage/profiles'
import {
  getRatedExercises,
  getUnratedExercises,
  calculateOverallLevel,
  calculateStrengthScore,
  calculateBadges,
  generateCoachTips,
  getRatedCount,
  getTotalExercisesCount
} from '@/lib/calculations/strength'
import { getNutritionInfo, BMI_CATEGORIES, formatCalories, isBelowMinimum } from '@/lib/calculations/nutrition'
import { Button, ThemeToggle } from '@/components/ui'
import { StrengthCard, LevelLegend, BodyPartFilter, OverallLevel, StrengthScore, Badges, CoachTips } from '@/components/strength'

interface ProfileDetailPageProps {
  params: Promise<{ id: string }>
}

export default function ProfileDetailPage({ params }: ProfileDetailPageProps) {
  const { id } = use(params)
  const router = useRouter()
  const [profile, setProfile] = useState<Profile | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [bodyPartFilter, setBodyPartFilter] = useState<BodyPart | 'all'>('all')

  useEffect(() => {
    const loadedProfile = getProfileById(id)
    setProfile(loadedProfile)
    setIsLoading(false)
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

  const ratedExercises = getRatedExercises(profile.weight, profile.exerciseRatings, bodyPartFilter, profile.sex)
  const unratedExercises = getUnratedExercises(profile.weight, profile.exerciseRatings, bodyPartFilter, profile.sex)
  const overallLevel = calculateOverallLevel(profile.exerciseRatings)
  const strengthScore = calculateStrengthScore(profile.exerciseRatings)
  const badges = calculateBadges(profile.exerciseRatings)
  const coachTips = generateCoachTips(profile.exerciseRatings)
  const ratedCount = getRatedCount(profile.exerciseRatings)
  const totalCount = getTotalExercisesCount()

  return (
    <div className="min-h-screen pb-20">
      {/* Header */}
      <header className="bg-[#2C3E50] text-white px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href="/" className="text-white hover:text-gray-300">
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 19l-7-7 7-7"
                />
              </svg>
            </Link>
            <h1 className="text-lg font-semibold">{profile.name}</h1>
          </div>
          <ThemeToggle />
        </div>
      </header>

      {/* Content */}
      <main className="p-4 max-w-lg mx-auto">
        {/* Profile Info */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-[#E0E0E0] dark:border-gray-700 p-4 mb-4">
          <div className="flex items-center gap-4">
            {/* Avatar */}
            <div className="w-16 h-16 rounded-full bg-[#3498DB] flex items-center justify-center text-white font-semibold text-xl relative">
              {profile.name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)}
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
                <h2 className="font-semibold text-[#2C3E50] dark:text-gray-100 text-lg">
                  {profile.name}
                </h2>
                <Link
                  href={`/profile/${profile.id}/edit`}
                  className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 p-1"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                  </svg>
                </Link>
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-1 grid grid-cols-3 gap-2">
                <div>
                  <span className="block font-medium text-[#2C3E50] dark:text-gray-200">{profile.age}</span>
                  <span className="text-xs">years</span>
                </div>
                <div>
                  <span className="block font-medium text-[#2C3E50] dark:text-gray-200">{profile.height}</span>
                  <span className="text-xs">cm</span>
                </div>
                <div>
                  <span className="block font-medium text-[#2C3E50] dark:text-gray-200">{profile.weight}</span>
                  <span className="text-xs">kg</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Strength Score and Overall Level */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          <StrengthScore score={strengthScore} />
          <OverallLevel
            level={overallLevel}
            ratedCount={ratedCount}
            totalCount={totalCount}
          />
        </div>

        {/* Achievements/Badges */}
        <div className="mb-4">
          <Badges badges={badges} />
        </div>

        {/* Nutrition Summary */}
        {(() => {
          const nutritionInfo = getNutritionInfo(profile)
          if (!nutritionInfo) {
            // Show prompt to complete profile
            return (
              <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700 p-4 mb-4">
                <div className="flex items-start gap-3">
                  <div className="text-2xl">🔥</div>
                  <div>
                    <h3 className="font-semibold text-gray-700 dark:text-gray-200 text-sm">
                      Unlock Nutrition Insights
                    </h3>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      Add your sex and activity level to see personalized calorie recommendations.
                    </p>
                    <Link
                      href={`/profile/${profile.id}/edit`}
                      className="text-xs text-blue-500 hover:text-blue-600 mt-2 inline-block"
                    >
                      Complete Profile →
                    </Link>
                  </div>
                </div>
              </div>
            )
          }

          const bmiInfo = BMI_CATEGORIES[nutritionInfo.bmiCategory]
          const goalName = profile.goal ? GOAL_INFO[profile.goal].name : 'Maintain'
          const belowMin = profile.sex ? isBelowMinimum(nutritionInfo.targetCalories, profile.sex) : false

          return (
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 mb-4">
              <h3 className="font-semibold text-gray-700 dark:text-gray-200 text-sm mb-3 flex items-center gap-2">
                <span className="text-lg">🔥</span>
                Daily Nutrition
              </h3>

              <div className="space-y-3">
                {/* Maintenance & Target */}
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-500 dark:text-gray-400">Maintenance</span>
                  <span className="font-semibold text-gray-800 dark:text-gray-100">
                    {formatCalories(nutritionInfo.tdee)} kcal
                  </span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-500 dark:text-gray-400">Goal: {goalName}</span>
                  <span className={`font-semibold ${
                    nutritionInfo.targetCalories < nutritionInfo.tdee
                      ? 'text-orange-500'
                      : nutritionInfo.targetCalories > nutritionInfo.tdee
                      ? 'text-green-500'
                      : 'text-gray-800 dark:text-gray-100'
                  }`}>
                    {formatCalories(nutritionInfo.targetCalories)} kcal
                  </span>
                </div>

                {/* Calorie bar */}
                <div className="h-2 bg-gray-100 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all ${
                      nutritionInfo.targetCalories < nutritionInfo.tdee
                        ? 'bg-orange-500'
                        : nutritionInfo.targetCalories > nutritionInfo.tdee
                        ? 'bg-green-500'
                        : 'bg-blue-500'
                    }`}
                    style={{ width: `${Math.min(100, (nutritionInfo.targetCalories / nutritionInfo.tdee) * 100)}%` }}
                  />
                </div>

                {/* BMI */}
                <div className="pt-2 border-t border-gray-100 dark:border-gray-700">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-500 dark:text-gray-400">BMI</span>
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-gray-800 dark:text-gray-100">
                        {nutritionInfo.bmi}
                      </span>
                      <span
                        className="text-xs px-2 py-0.5 rounded-full"
                        style={{ backgroundColor: `${bmiInfo.color}20`, color: bmiInfo.color }}
                      >
                        {bmiInfo.name}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Weekly projection */}
                {nutritionInfo.weeklyChange !== 0 && (
                  <div className="text-xs text-center text-gray-500 dark:text-gray-400 pt-1">
                    {nutritionInfo.weeklyChange > 0 ? '↑' : '↓'} ~{Math.abs(nutritionInfo.weeklyChange)} kg/week
                  </div>
                )}

                {/* Warning if below minimum */}
                {belowMin && (
                  <div className="text-xs text-orange-600 dark:text-orange-400 bg-orange-50 dark:bg-orange-900/20 p-2 rounded">
                    ⚠️ Target is below recommended minimum. Consider a smaller deficit.
                  </div>
                )}
              </div>
            </div>
          )
        })()}

        {/* Progress Link */}
        <Link
          href={`/profile/${profile.id}/progress`}
          className="block mb-4 bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-200 rounded-lg p-3 text-center hover:bg-gray-200 dark:hover:bg-gray-700 transition-all"
        >
          <div className="flex items-center justify-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <span className="font-medium">View Progress Charts</span>
          </div>
        </Link>

        {/* AI Coach Tips */}
        <div className="mb-4">
          <CoachTips tips={coachTips} />
        </div>

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
