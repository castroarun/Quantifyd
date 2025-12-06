'use client'

import { useEffect, useState } from 'react'
import { Profile, VALIDATION } from '@/types'
import { getProfiles } from '@/lib/storage/profiles'
import { ProfileCard, EmptyProfileSlot } from '@/components/profile'

export default function HomePage() {
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    setProfiles(getProfiles())
    setIsLoading(false)
  }, [])

  const emptySlots = VALIDATION.maxProfiles - profiles.length

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-gray-500">Loading...</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="bg-[#2C3E50] text-white px-4 py-4">
        <h1 className="text-lg font-semibold">Strength Profiles</h1>
        <p className="text-sm text-gray-300 mt-1">
          {profiles.length} of {VALIDATION.maxProfiles} profiles
        </p>
      </header>

      {/* Content */}
      <main className="p-4 max-w-lg mx-auto">
        {/* Existing Profiles */}
        <div className="space-y-3">
          {profiles.map(profile => (
            <ProfileCard key={profile.id} profile={profile} />
          ))}

          {/* Empty Slots */}
          {Array.from({ length: emptySlots }).map((_, index) => (
            <EmptyProfileSlot
              key={`empty-${index}`}
              disabled={index > 0}
            />
          ))}
        </div>

        {/* Empty State */}
        {profiles.length === 0 && (
          <div className="text-center py-8">
            <div className="text-gray-400 mb-2">
              <svg
                className="w-16 h-16 mx-auto"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
                />
              </svg>
            </div>
            <h2 className="text-lg font-semibold text-[#2C3E50] mb-1">
              No profiles yet
            </h2>
            <p className="text-sm text-gray-500">
              Create your first profile to get started tracking your strength standards.
            </p>
          </div>
        )}
      </main>
    </div>
  )
}
