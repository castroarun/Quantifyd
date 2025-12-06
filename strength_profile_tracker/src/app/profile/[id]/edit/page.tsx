'use client'

import { use, useEffect, useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { Profile } from '@/types'
import { getProfileById } from '@/lib/storage/profiles'
import { ProfileForm } from '@/components/profile'
import { ThemeToggle } from '@/components/ui'

interface EditProfilePageProps {
  params: Promise<{ id: string }>
}

export default function EditProfilePage({ params }: EditProfilePageProps) {
  const { id } = use(params)
  const router = useRouter()
  const [profile, setProfile] = useState<Profile | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const loadedProfile = getProfileById(id)
    if (loadedProfile) {
      setProfile(loadedProfile)
    }
    setIsLoading(false)
  }, [id])

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-gray-500">Loading...</div>
      </div>
    )
  }

  if (!profile) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-500 mb-4">Profile not found</p>
          <Link href="/" className="text-blue-500 hover:underline">
            Go back home
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="bg-[#2C3E50] text-white px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href={`/profile/${id}`} className="text-white hover:text-gray-300">
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
            <h1 className="text-lg font-semibold">Edit Profile</h1>
          </div>
          <ThemeToggle />
        </div>
      </header>

      {/* Content */}
      <main className="p-4 max-w-lg mx-auto">
        <ProfileForm
          profile={profile}
          onCancel={() => router.push(`/profile/${id}`)}
        />
      </main>
    </div>
  )
}
