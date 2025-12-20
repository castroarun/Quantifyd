'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { useAuth } from '@/contexts'
import { Button, ThemeToggle, UnitToggle } from '@/components/ui'
import { getPendingSyncCount, processSyncQueue } from '@/lib/storage/sync'
import { getLastSyncTime } from '@/lib/storage/profiles'
import {
  requestNotificationPermission,
  getNotificationPermission,
  notificationsSupported
} from '@/lib/pwa/notifications'

export default function SettingsPage() {
  const router = useRouter()
  const { user, isConfigured, signInWithGoogle, signOut, deleteAccount } = useAuth()
  const [isSigningIn, setIsSigningIn] = useState(false)
  const [isSigningOut, setIsSigningOut] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [isSyncing, setIsSyncing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notificationStatus, setNotificationStatus] = useState<NotificationPermission | 'unsupported'>('default')
  const [isRequestingNotifications, setIsRequestingNotifications] = useState(false)

  useEffect(() => {
    setNotificationStatus(getNotificationPermission())
  }, [])

  const handleEnableNotifications = async () => {
    setIsRequestingNotifications(true)
    const granted = await requestNotificationPermission()
    setNotificationStatus(granted ? 'granted' : 'denied')
    setIsRequestingNotifications(false)
  }

  const pendingCount = getPendingSyncCount()
  const lastSync = getLastSyncTime()

  const handleSignIn = async () => {
    setIsSigningIn(true)
    setError(null)

    const { error } = await signInWithGoogle()
    if (error) {
      setError(error.message)
    }
    setIsSigningIn(false)
  }

  const handleSignOut = async () => {
    setIsSigningOut(true)
    await signOut()
    localStorage.removeItem('skippedLogin')
    setIsSigningOut(false)
    router.push('/')
  }

  const handleDeleteAccount = async () => {
    setIsDeleting(true)
    const { error } = await deleteAccount()
    if (error) {
      setError(error.message)
      setIsDeleting(false)
    } else {
      localStorage.clear()
      router.push('/')
    }
  }

  const handleManualSync = async () => {
    setIsSyncing(true)
    try {
      await processSyncQueue()
    } catch (err) {
      console.error('Sync failed:', err)
    }
    setIsSyncing(false)
  }

  const formatLastSync = (timestamp: string | null) => {
    if (!timestamp) return 'Never'
    const date = new Date(timestamp)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins} min ago`
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)} hours ago`
    return date.toLocaleDateString()
  }

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="bg-[#2C3E50] text-white px-4 py-4">
        <div className="flex items-center gap-3">
          <Link href="/" className="text-white hover:text-gray-300">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </Link>
          <h1 className="text-lg font-semibold">Settings</h1>
        </div>
      </header>

      <main className="p-4 max-w-lg mx-auto space-y-6">
        {/* Account Section */}
        <section className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
            <h2 className="font-semibold text-[#2C3E50] dark:text-gray-100">Account</h2>
          </div>
          <div className="p-4">
            {user ? (
              <div className="space-y-4">
                {/* User Info */}
                <div className="flex items-center gap-3">
                  {user.user_metadata?.avatar_url ? (
                    <img
                      src={user.user_metadata.avatar_url}
                      alt="Profile"
                      className="w-12 h-12 rounded-full"
                    />
                  ) : (
                    <div className="w-12 h-12 rounded-full bg-blue-500 flex items-center justify-center text-white font-semibold">
                      {(user.email?.[0] || 'U').toUpperCase()}
                    </div>
                  )}
                  <div>
                    <p className="font-medium text-[#2C3E50] dark:text-gray-100">
                      {user.user_metadata?.full_name || 'User'}
                    </p>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      {user.email}
                    </p>
                  </div>
                </div>

                {/* Sync Status */}
                <div className="flex items-center justify-between py-2 border-t border-gray-100 dark:border-gray-700">
                  <div>
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-200">
                      Sync Status
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      Last sync: {formatLastSync(lastSync)}
                      {pendingCount > 0 && (
                        <span className="text-amber-500 ml-2">
                          • {pendingCount} pending
                        </span>
                      )}
                    </p>
                  </div>
                  <button
                    onClick={handleManualSync}
                    disabled={isSyncing || pendingCount === 0}
                    className="text-sm text-blue-500 hover:text-blue-600 disabled:text-gray-400 disabled:cursor-not-allowed"
                  >
                    {isSyncing ? 'Syncing...' : 'Sync Now'}
                  </button>
                </div>

                {/* Sign Out */}
                <Button
                  variant="secondary"
                  fullWidth
                  onClick={handleSignOut}
                  disabled={isSigningOut}
                >
                  {isSigningOut ? 'Signing out...' : 'Sign Out'}
                </Button>
              </div>
            ) : isConfigured ? (
              <div className="space-y-4">
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Sign in to sync your data across all your devices.
                </p>
                <button
                  onClick={handleSignIn}
                  disabled={isSigningIn}
                  className="w-full flex items-center justify-center gap-3 bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 font-medium py-3 px-6 rounded-lg border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-600 transition-colors disabled:opacity-50"
                >
                  {isSigningIn ? (
                    <div className="w-5 h-5 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <>
                      <svg className="w-5 h-5" viewBox="0 0 24 24">
                        <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                        <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                        <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                        <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
                      </svg>
                      Sign in with Google
                    </>
                  )}
                </button>
                {pendingCount > 0 && (
                  <p className="text-xs text-amber-600 dark:text-amber-400 text-center">
                    {pendingCount} changes waiting to sync
                  </p>
                )}
              </div>
            ) : (
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Cloud sync is not configured. Your data is stored locally on this device.
              </p>
            )}

            {error && (
              <p className="mt-3 text-sm text-red-500 text-center">{error}</p>
            )}
          </div>
        </section>

        {/* Preferences Section */}
        <section className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
            <h2 className="font-semibold text-[#2C3E50] dark:text-gray-100">Preferences</h2>
          </div>
          <div className="p-4 space-y-4">
            {/* Theme */}
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium text-gray-700 dark:text-gray-200">Theme</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">Light or dark mode</p>
              </div>
              <ThemeToggle />
            </div>

            {/* Units */}
            <div className="flex items-center justify-between border-t border-gray-100 dark:border-gray-700 pt-4">
              <div>
                <p className="font-medium text-gray-700 dark:text-gray-200">Weight Unit</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">Kilograms or pounds</p>
              </div>
              <UnitToggle />
            </div>

            {/* Notifications */}
            {notificationsSupported() && (
              <div className="flex items-center justify-between border-t border-gray-100 dark:border-gray-700 pt-4">
                <div>
                  <p className="font-medium text-gray-700 dark:text-gray-200">Notifications</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {notificationStatus === 'granted'
                      ? 'Enabled - workout reminders active'
                      : notificationStatus === 'denied'
                      ? 'Blocked - enable in browser settings'
                      : 'Get workout reminders'}
                  </p>
                </div>
                {notificationStatus === 'granted' ? (
                  <span className="text-green-500 text-sm font-medium flex items-center gap-1">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    On
                  </span>
                ) : notificationStatus === 'denied' ? (
                  <span className="text-red-500 text-sm">Blocked</span>
                ) : (
                  <button
                    onClick={handleEnableNotifications}
                    disabled={isRequestingNotifications}
                    className="px-3 py-1.5 text-sm bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 transition-colors"
                  >
                    {isRequestingNotifications ? 'Enabling...' : 'Enable'}
                  </button>
                )}
              </div>
            )}
          </div>
        </section>

        {/* About Section */}
        <section className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
            <h2 className="font-semibold text-[#2C3E50] dark:text-gray-100">About</h2>
          </div>
          <div className="p-4 space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-500 dark:text-gray-400">App</span>
              <span className="text-gray-700 dark:text-gray-200">REPPIT</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-500 dark:text-gray-400">Version</span>
              <span className="text-gray-700 dark:text-gray-200">1.0.0</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-500 dark:text-gray-400">Build</span>
              <span className="text-gray-700 dark:text-gray-200">2025.01</span>
            </div>
          </div>
        </section>

        {/* Dev: Timer UI Options */}
        <section className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
            <h2 className="font-semibold text-[#2C3E50] dark:text-gray-100">Developer</h2>
          </div>
          <div className="p-4">
            <a
              href="/mockups/timer-options.html"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-between text-sm hover:bg-gray-50 dark:hover:bg-gray-700 -mx-4 px-4 py-3 transition-colors"
            >
              <div>
                <p className="font-medium text-gray-700 dark:text-gray-200">Timer UI Options</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">Preview different timer designs</p>
              </div>
              <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </a>
          </div>
        </section>

        {/* Danger Zone */}
        {user && (
          <section className="bg-white dark:bg-gray-800 rounded-lg border border-red-200 dark:border-red-900 overflow-hidden">
            <div className="px-4 py-3 border-b border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-900/20">
              <h2 className="font-semibold text-red-600 dark:text-red-400">Danger Zone</h2>
            </div>
            <div className="p-4">
              {!showDeleteConfirm ? (
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-3">
                    Delete your account and all associated data. This action cannot be undone.
                  </p>
                  <Button
                    variant="danger"
                    fullWidth
                    onClick={() => setShowDeleteConfirm(true)}
                  >
                    Delete Account
                  </Button>
                </div>
              ) : (
                <div className="space-y-3">
                  <p className="text-sm text-red-600 dark:text-red-400 font-medium">
                    Are you sure? This will permanently delete:
                  </p>
                  <ul className="text-sm text-gray-600 dark:text-gray-300 list-disc list-inside space-y-1">
                    <li>Your account</li>
                    <li>All profiles and strength data</li>
                    <li>All workout history</li>
                  </ul>
                  <div className="flex gap-2 mt-4">
                    <Button
                      variant="secondary"
                      fullWidth
                      onClick={() => setShowDeleteConfirm(false)}
                      disabled={isDeleting}
                    >
                      Cancel
                    </Button>
                    <Button
                      variant="danger"
                      fullWidth
                      onClick={handleDeleteAccount}
                      disabled={isDeleting}
                    >
                      {isDeleting ? 'Deleting...' : 'Yes, Delete'}
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

        {/* Privacy Link */}
        <div className="text-center pt-4">
          <a
            href="/privacy"
            className="text-sm text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          >
            Privacy Policy
          </a>
        </div>
      </main>
    </div>
  )
}
