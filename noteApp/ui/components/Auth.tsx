'use client'

/**
 * Authentication component for Google OAuth signin/signout
 *
 * Input: None
 * Output: Authentication UI (signin button or user info)
 *
 * Called by: app/page.tsx when user is not authenticated
 * Calls: createClient from lib/supabase
 */

import { useState } from 'react'
import { createClient } from '@/lib/supabase-client'

export function AuthSignIn() {
  const [isLoading, setIsLoading] = useState(false)
  const supabase = createClient()

  /**
   * Handle Google OAuth signin
   *
   * Input: None
   * Output: Redirects to Google OAuth or shows error
   *
   * Called by: Sign in button click
   * Calls: supabase.auth.signInWithOAuth
   */
  const handleGoogleSignIn = async () => {
    setIsLoading(true)
    try {
      const { error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo: `${window.location.origin}/auth/callback`,
        },
      })
      if (error) throw error
    } catch (error) {
      console.error('Error signing in:', error)
      alert('Failed to sign in. Please try again.')
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="text-center space-y-8 p-8">
        <div className="space-y-4">
          <h1 className="text-5xl font-light text-text-primary">
            NoteApp
          </h1>
          <p className="text-lg text-text-secondary">
            Clean, simple, fast note-taking
          </p>
          <p className="text-sm text-text-muted">
            Auto-titles • Templates • Simple formatting
          </p>
        </div>

        <button
          onClick={handleGoogleSignIn}
          disabled={isLoading}
          className="inline-flex items-center gap-3 px-6 py-3 bg-accent-blue text-white rounded-lg hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <>
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Signing in...
            </>
          ) : (
            <>
              <svg className="w-5 h-5" viewBox="0 0 24 24">
                <path
                  fill="currentColor"
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                />
                <path
                  fill="currentColor"
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                />
                <path
                  fill="currentColor"
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                />
                <path
                  fill="currentColor"
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                />
              </svg>
              Sign in with Google
            </>
          )}
        </button>
      </div>
    </div>
  )
}

/**
 * User menu component for authenticated users
 *
 * Input: user object
 * Output: User info and signout button
 *
 * Called by: Main app layout when user is authenticated
 * Calls: supabase.auth.signOut
 */
export function UserMenu({ user }: { user: any }) {
  const [isLoading, setIsLoading] = useState(false)
  const supabase = createClient()

  /**
   * Handle user signout
   *
   * Input: None
   * Output: Signs out user and refreshes page
   *
   * Called by: Sign out button click
   * Calls: supabase.auth.signOut
   */
  const handleSignOut = async () => {
    setIsLoading(true)
    try {
      const { error } = await supabase.auth.signOut()
      if (error) throw error
      window.location.href = '/'
    } catch (error) {
      console.error('Error signing out:', error)
      alert('Failed to sign out. Please try again.')
      setIsLoading(false)
    }
  }

  return (
    <div className="flex items-center gap-3">
      {user?.user_metadata?.avatar_url && (
        <img
          src={user.user_metadata.avatar_url}
          alt={user.user_metadata.name || user.email}
          className="w-8 h-8 rounded-full"
        />
      )}
      <span className="text-sm text-text-secondary">
        {user?.user_metadata?.name || user?.email}
      </span>
      <button
        onClick={handleSignOut}
        disabled={isLoading}
        className="px-3 py-1 text-sm text-text-secondary hover:text-text-primary border border-border rounded hover:border-text-muted transition-colors disabled:opacity-50"
      >
        {isLoading ? 'Signing out...' : 'Sign out'}
      </button>
    </div>
  )
}
