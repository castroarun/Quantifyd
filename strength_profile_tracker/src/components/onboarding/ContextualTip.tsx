'use client'

import { useState, useEffect } from 'react'
import { areTipsEnabled } from '@/lib/storage/onboarding'

export type TipId =
  | 'program_intro'
  | 'profile_gestures'
  | 'progress_calendar'
  | 'workout_log'
  | 'profile_settings'
  | 'workout_timer_settings'
  | 'timer_settings'

interface ContextualTipProps {
  tipId: TipId
  title: string
  message: string
  icon?: string
  position?: 'top' | 'bottom'
}

const TIP_STORAGE_KEY = 'strength_profile_tips_seen'

function getSeenTips(): TipId[] {
  if (typeof window === 'undefined') return []
  const stored = localStorage.getItem(TIP_STORAGE_KEY)
  if (!stored) return []
  try {
    return JSON.parse(stored)
  } catch {
    return []
  }
}

function markTipSeen(tipId: TipId): void {
  if (typeof window === 'undefined') return
  const seen = getSeenTips()
  if (!seen.includes(tipId)) {
    seen.push(tipId)
    localStorage.setItem(TIP_STORAGE_KEY, JSON.stringify(seen))
  }
}

export function hasSeenTip(tipId: TipId): boolean {
  return getSeenTips().includes(tipId)
}

export function resetAllTips(): void {
  if (typeof window === 'undefined') return
  localStorage.removeItem(TIP_STORAGE_KEY)
}

export default function ContextualTip({
  tipId,
  title,
  message,
  icon = '💡',
  position = 'top'
}: ContextualTipProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [isAnimating, setIsAnimating] = useState(false)

  useEffect(() => {
    // Check if tips are enabled and if this tip hasn't been seen
    if (areTipsEnabled() && !hasSeenTip(tipId)) {
      // Small delay before showing tip
      const timer = setTimeout(() => {
        setIsVisible(true)
        setIsAnimating(true)
      }, 500)
      return () => clearTimeout(timer)
    }
  }, [tipId])

  const handleDismiss = () => {
    setIsAnimating(false)
    setTimeout(() => {
      setIsVisible(false)
      markTipSeen(tipId)
    }, 200)
  }

  if (!isVisible) return null

  return (
    <div
      className={`fixed ${position === 'top' ? 'top-20' : 'bottom-24'} left-4 right-4 z-50 max-w-md mx-auto transition-all duration-200 ${
        isAnimating ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'
      }`}
    >
      <div className="bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-700 rounded-xl shadow-lg p-4">
        <div className="flex items-start gap-3">
          <span className="text-2xl flex-shrink-0">{icon}</span>
          <div className="flex-1 min-w-0">
            <h4 className="font-semibold text-blue-800 dark:text-blue-200 text-sm">
              {title}
            </h4>
            <p className="text-xs text-blue-600 dark:text-blue-300 mt-1 leading-relaxed">
              {message}
            </p>
          </div>
          <button
            onClick={handleDismiss}
            className="flex-shrink-0 text-blue-400 hover:text-blue-600 dark:text-blue-500 dark:hover:text-blue-300 transition-colors p-1"
            aria-label="Dismiss tip"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <button
          onClick={handleDismiss}
          className="mt-3 w-full text-xs text-blue-500 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-200 font-medium"
        >
          Got it!
        </button>
      </div>
    </div>
  )
}
