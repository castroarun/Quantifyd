'use client'

import { useState, useEffect, useCallback, useRef } from 'react'

interface WakeLockState {
  isSupported: boolean
  isActive: boolean
  error: string | null
}

/**
 * Hook to manage Screen Wake Lock API
 * Keeps the screen on while the lock is active (phone can still dim)
 */
export function useWakeLock() {
  const [state, setState] = useState<WakeLockState>({
    isSupported: false,
    isActive: false,
    error: null
  })

  const wakeLockRef = useRef<WakeLockSentinel | null>(null)
  // Track if we should have a wake lock (for re-acquisition on visibility change)
  const shouldHaveWakeLockRef = useRef(false)

  // Check support on mount
  useEffect(() => {
    const isSupported = 'wakeLock' in navigator
    setState(prev => ({ ...prev, isSupported }))
  }, [])

  // Request wake lock
  const requestWakeLock = useCallback(async () => {
    // Track that we want a wake lock (for re-acquisition on visibility change)
    shouldHaveWakeLockRef.current = true

    if (!('wakeLock' in navigator)) {
      setState(prev => ({ ...prev, error: 'Wake Lock not supported' }))
      return false
    }

    // Don't request if page is not visible
    if (document.visibilityState !== 'visible') {
      return false
    }

    try {
      wakeLockRef.current = await navigator.wakeLock.request('screen')

      // Handle release event (e.g., when tab becomes inactive)
      wakeLockRef.current.addEventListener('release', () => {
        wakeLockRef.current = null
        setState(prev => ({ ...prev, isActive: false }))
      })

      setState(prev => ({ ...prev, isActive: true, error: null }))
      return true
    } catch (err) {
      const error = err instanceof Error ? err.message : 'Failed to request wake lock'
      setState(prev => ({ ...prev, error, isActive: false }))
      return false
    }
  }, [])

  // Release wake lock
  const releaseWakeLock = useCallback(async () => {
    // Clear the intent to have a wake lock
    shouldHaveWakeLockRef.current = false

    if (wakeLockRef.current) {
      try {
        await wakeLockRef.current.release()
        wakeLockRef.current = null
        setState(prev => ({ ...prev, isActive: false }))
        return true
      } catch (err) {
        console.error('Failed to release wake lock:', err)
        return false
      }
    }
    return false
  }, [])

  // Re-acquire wake lock when page becomes visible again
  useEffect(() => {
    const handleVisibilityChange = async () => {
      // Use shouldHaveWakeLockRef instead of state.isActive because
      // when the browser releases the wake lock on background, isActive becomes false
      // but we still want to re-acquire when coming back to foreground
      if (document.visibilityState === 'visible' && shouldHaveWakeLockRef.current && !wakeLockRef.current) {
        await requestWakeLock()
      }
    }

    document.addEventListener('visibilitychange', handleVisibilityChange)
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [requestWakeLock])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wakeLockRef.current) {
        wakeLockRef.current.release().catch(() => {})
      }
    }
  }, [])

  return {
    ...state,
    requestWakeLock,
    releaseWakeLock
  }
}
