'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { Profile, VALIDATION } from '@/types'
import { getProfiles, getProfileById, syncProfilesFromCloud } from '@/lib/storage/profiles'
import { loadSampleData, hasSampleData, removeSampleData } from '@/lib/storage/seedData'
import { setupSyncListeners, hasPendingSync, processSyncQueue } from '@/lib/storage/sync'
import { ProfileCard, EmptyProfileSlot } from '@/components/profile'
import { ThemeToggle, UnitToggle, Logo } from '@/components/ui'
import { LoginScreen } from '@/components/auth'
import { useAuth } from '@/contexts'

export default function HomePage() {
  const router = useRouter()
  const { user, isLoading: authLoading, isConfigured } = useAuth()
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [hasSamples, setHasSamples] = useState(false)
  const [showLogin, setShowLogin] = useState(false)
  const [isSyncing, setIsSyncing] = useState(false)

  // Setup sync listeners
  useEffect(() => {
    const cleanup = setupSyncListeners()
    return cleanup
  }, [])

  // Sync on auth or when coming online
  useEffect(() => {
    if (user && isConfigured) {
      handleSync()
    }
  }, [user, isConfigured])

  // Load profiles
  useEffect(() => {
    if (authLoading) return

    // Check if first-time user without account
    const hasSkippedLogin = localStorage.getItem('skippedLogin')
    if (!user && isConfigured && !hasSkippedLogin) {
      setShowLogin(true)
      setIsLoading(false)
      return
    }

    // Check for last visited profile and redirect if it still exists
    const lastVisitedProfileId = localStorage.getItem('lastVisitedProfile')
    if (lastVisitedProfileId) {
      const lastProfile = getProfileById(lastVisitedProfileId)
      if (lastProfile) {
        router.push(`/profile/${lastVisitedProfileId}`)
        return
      } else {
        // Profile was deleted, clear the saved preference
        localStorage.removeItem('lastVisitedProfile')
      }
    }

    setProfiles(getProfiles())
    setHasSamples(hasSampleData())
    setIsLoading(false)
  }, [router, authLoading, user, isConfigured])

  const handleSync = async () => {
    setIsSyncing(true)
    try {
      // Push pending changes first
      if (hasPendingSync()) {
        await processSyncQueue()
      }
      // Then pull from cloud
      await syncProfilesFromCloud()
      setProfiles(getProfiles())
    } catch (error) {
      console.error('Sync failed:', error)
    } finally {
      setIsSyncing(false)
    }
  }

  const handleSkipLogin = () => {
    localStorage.setItem('skippedLogin', 'true')
    setShowLogin(false)
    setProfiles(getProfiles())
    setHasSamples(hasSampleData())
  }

  const handleLoadSampleData = () => {
    loadSampleData()
    setProfiles(getProfiles())
    setHasSamples(true)
  }

  const handleRemoveSampleData = () => {
    removeSampleData()
    setProfiles(getProfiles())
    setHasSamples(false)
  }

  const emptySlots = VALIDATION.maxProfiles - profiles.length

  // Show login screen for first-time users
  if (showLogin) {
    return <LoginScreen onSkip={handleSkipLogin} />
  }

  if (isLoading || authLoading) {
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
        <div className="flex items-center justify-between">
          <div>
            <Logo size="lg" showRings={true} />
            <p className="text-sm text-gray-300 mt-1">
              {profiles.length} of {VALIDATION.maxProfiles} profiles
              {isSyncing && <span className="ml-2">• Syncing...</span>}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Link
              href="/settings"
              className="p-2 text-white/80 hover:text-white transition-colors"
              title="Settings"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </Link>
            <UnitToggle />
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Sync Status Banner */}
      {!user && isConfigured && (
        <div className="bg-amber-50 dark:bg-amber-900/20 border-b border-amber-200 dark:border-amber-800 px-4 py-2">
          <p className="text-sm text-amber-700 dark:text-amber-300 text-center">
            <Link href="/settings" className="underline hover:no-underline">
              Sign in
            </Link>
            {' '}to sync your data across devices
          </p>
        </div>
      )}

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
            <div className="text-gray-400 dark:text-gray-500 mb-2">
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
            <h2 className="text-lg font-semibold text-[#2C3E50] dark:text-gray-100 mb-1">
              No profiles yet
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Create your first profile to get started tracking your strength standards.
            </p>
          </div>
        )}

        {/* Sample Data Button */}
        <div className="mt-8 text-center">
          {!hasSamples ? (
            <button
              onClick={handleLoadSampleData}
              className="text-sm text-blue-500 hover:text-blue-600 dark:text-blue-400 dark:hover:text-blue-300 underline"
            >
              Load sample profiles with workout data
            </button>
          ) : (
            <button
              onClick={handleRemoveSampleData}
              className="text-sm text-gray-400 hover:text-red-500 dark:text-gray-500 dark:hover:text-red-400 underline"
            >
              Remove sample profiles
            </button>
          )}
        </div>
      </main>
    </div>
  )
}
