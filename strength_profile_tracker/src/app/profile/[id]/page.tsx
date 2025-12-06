'use client'

import { useEffect, useState, use } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { Profile, Exercise, Level } from '@/types'
import { getProfileById, updateProfileLevel, deleteProfile } from '@/lib/storage/profiles'
import { calculateAllStrengths } from '@/lib/calculations/strength'
import { Button } from '@/components/ui'
import { StrengthCard, LevelLegend } from '@/components/strength'

interface ProfileDetailPageProps {
  params: Promise<{ id: string }>
}

export default function ProfileDetailPage({ params }: ProfileDetailPageProps) {
  const { id } = use(params)
  const router = useRouter()
  const [profile, setProfile] = useState<Profile | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)

  useEffect(() => {
    const loadedProfile = getProfileById(id)
    setProfile(loadedProfile)
    setIsLoading(false)
  }, [id])

  const handleLevelSelect = (exercise: Exercise, level: Level) => {
    if (!profile) return

    const updatedProfile = updateProfileLevel(profile.id, exercise, level)
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

  const strengths = calculateAllStrengths(profile.weight, profile.currentLevels)

  return (
    <div className="min-h-screen pb-20">
      {/* Header */}
      <header className="bg-[#2C3E50] text-white px-4 py-4">
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
      </header>

      {/* Content */}
      <main className="p-4 max-w-lg mx-auto">
        {/* Profile Info */}
        <div className="bg-white rounded-lg border border-[#E0E0E0] p-4 mb-4">
          <div className="flex items-center gap-4">
            {/* Avatar */}
            <div className="w-16 h-16 rounded-full bg-[#3498DB] flex items-center justify-center text-white font-semibold text-xl">
              {profile.name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)}
            </div>

            {/* Info */}
            <div className="flex-1">
              <h2 className="font-semibold text-[#2C3E50] text-lg">
                {profile.name}
              </h2>
              <div className="text-sm text-gray-500 mt-1 grid grid-cols-3 gap-2">
                <div>
                  <span className="block font-medium text-[#2C3E50]">{profile.age}</span>
                  <span className="text-xs">years</span>
                </div>
                <div>
                  <span className="block font-medium text-[#2C3E50]">{profile.height}</span>
                  <span className="text-xs">cm</span>
                </div>
                <div>
                  <span className="block font-medium text-[#2C3E50]">{profile.weight}</span>
                  <span className="text-xs">kg</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Strength Standards */}
        <h3 className="font-semibold text-[#2C3E50] text-sm mb-3">
          Your Strength Standards
        </h3>
        <p className="text-xs text-gray-500 mb-3">
          Tap a level to mark it as your current performance.
        </p>

        {strengths.map(strength => (
          <StrengthCard
            key={strength.exercise}
            strength={strength}
            onLevelSelect={(level) => handleLevelSelect(strength.exercise, level)}
          />
        ))}

        {/* Level Legend */}
        <div className="mt-4">
          <LevelLegend />
        </div>

        {/* Delete Button */}
        <div className="mt-6 pt-4 border-t border-gray-200">
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
              <p className="text-sm text-center text-gray-600">
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
