'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { Exercise, TIMER_INCREMENT, TimerMode, getDynamicTimerPresets } from '@/types'
import {
  getTimerSettings,
  getExerciseTimerDuration,
  saveExerciseTimerDuration,
  formatTime,
  onTimerSettingsChange
} from '@/lib/storage/timer'
import { useWakeLock } from '@/hooks/useWakeLock'
import { useUnit } from '@/contexts'
import { ContextualTip } from '@/components/onboarding'
import { showWorkoutNotification } from '@/lib/pwa/notifications'

interface FullScreenTimerProps {
  exerciseId?: Exercise
  exerciseName?: string
  setNumber?: number
  weight?: number
  reps?: number
  onTimerEnd?: () => void
  onClose: () => void
  onMinimize?: (state: { timeLeft: number; duration: number; isRunning: boolean }) => void
  autoStart?: boolean
  initialTimeLeft?: number
  initialDuration?: number
  initialIsRunning?: boolean
}

export default function FullScreenTimer({
  exerciseId,
  exerciseName,
  setNumber,
  weight,
  reps,
  onTimerEnd,
  onClose,
  onMinimize,
  autoStart = true,
  initialTimeLeft,
  initialDuration,
  initialIsRunning
}: FullScreenTimerProps) {
  const { unit } = useUnit()
  const [duration, setDuration] = useState(initialDuration ?? 90)
  const [timeLeft, setTimeLeft] = useState(initialTimeLeft ?? 90)
  const [isRunning, setIsRunning] = useState(initialIsRunning ?? false)
  const [mode, setMode] = useState<TimerMode>('countdown')
  const [hasEnded, setHasEnded] = useState(false)
  const [showWarningFlash, setShowWarningFlash] = useState(false)
  const [warningTime, setWarningTime] = useState(30)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const { requestWakeLock, releaseWakeLock, isActive: wakeLockActive } = useWakeLock()

  // Load settings on mount (only if not resuming from minimize)
  useEffect(() => {
    const settings = getTimerSettings()
    setWarningTime(settings.warningTime ?? 30)

    // If we have initial values from minimize, don't override them
    if (initialTimeLeft !== undefined && initialDuration !== undefined) {
      if (initialIsRunning) {
        setIsRunning(true)
      }
      return
    }

    let loadedDuration = settings.defaultDuration

    if (exerciseId) {
      loadedDuration = getExerciseTimerDuration(exerciseId)
    }

    setDuration(loadedDuration)
    setTimeLeft(loadedDuration)

    if (autoStart) {
      setIsRunning(true)
    }
  }, [exerciseId, autoStart, initialTimeLeft, initialDuration, initialIsRunning])

  // Listen for timer settings changes (e.g., when user changes default duration in settings)
  useEffect(() => {
    const unsubscribe = onTimerSettingsChange((newSettings) => {
      // Only update if timer is not currently running
      if (!isRunning) {
        setDuration(newSettings.defaultDuration)
        setTimeLeft(newSettings.defaultDuration)
      }
      setWarningTime(newSettings.warningTime ?? 30)
    })
    return unsubscribe
  }, [isRunning])

  // Handle minimize (double-click on background)
  const handleMinimize = () => {
    if (onMinimize) {
      onMinimize({ timeLeft, duration, isRunning })
    }
  }

  // Manage wake lock based on timer state
  useEffect(() => {
    if (isRunning && !wakeLockActive) {
      requestWakeLock()
    } else if (!isRunning && wakeLockActive) {
      releaseWakeLock()
    }
  }, [isRunning, wakeLockActive, requestWakeLock, releaseWakeLock])

  // Show notification when app goes to background while timer is running
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'hidden' && isRunning) {
        // App went to background - show notification
        const timerDisplay = formatTime(timeLeft)
        showWorkoutNotification(
          `REPPIT - Rest Timer ${timerDisplay}`,
          exerciseName ? `${exerciseName} - Tap to return` : 'Tap to return to your workout'
        )
      }
    }

    document.addEventListener('visibilitychange', handleVisibilityChange)
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [isRunning, timeLeft, exerciseName])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
      releaseWakeLock()
    }
  }, [releaseWakeLock])

  // Play alert sound - longer duration for timer completion
  const playAlert = useCallback(() => {
    const settings = getTimerSettings()

    if (settings.vibrationEnabled && 'vibrate' in navigator) {
      navigator.vibrate([200, 100, 200, 100, 200, 100, 300, 100, 400])
    }

    if (settings.soundEnabled) {
      try {
        const audioContext = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)()
        const oscillator = audioContext.createOscillator()
        const gainNode = audioContext.createGain()

        oscillator.connect(gainNode)
        gainNode.connect(audioContext.destination)

        oscillator.frequency.value = 800
        oscillator.type = 'sine'
        gainNode.gain.value = 0.4

        oscillator.start()

        // Extended melody pattern for timer end - total ~2 seconds
        setTimeout(() => { oscillator.frequency.value = 1000 }, 200)
        setTimeout(() => { oscillator.frequency.value = 1200 }, 400)
        setTimeout(() => { oscillator.frequency.value = 800 }, 600)
        setTimeout(() => { oscillator.frequency.value = 1000 }, 800)
        setTimeout(() => { oscillator.frequency.value = 1200 }, 1000)
        setTimeout(() => { oscillator.frequency.value = 1400 }, 1200)
        setTimeout(() => { oscillator.frequency.value = 1600 }, 1400)
        setTimeout(() => {
          // Fade out
          gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3)
        }, 1700)
        setTimeout(() => {
          oscillator.stop()
          audioContext.close()
        }, 2000)
      } catch (e) {
        console.log('Audio not supported')
      }
    }
  }, [])

  // Play warning blip at 30 seconds remaining
  const playWarningBlip = useCallback(() => {
    const settings = getTimerSettings()

    if (settings.vibrationEnabled && 'vibrate' in navigator) {
      navigator.vibrate([100, 50, 100])
    }

    if (settings.soundEnabled) {
      try {
        const audioContext = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)()
        const oscillator = audioContext.createOscillator()
        const gainNode = audioContext.createGain()

        oscillator.connect(gainNode)
        gainNode.connect(audioContext.destination)

        oscillator.frequency.value = 600
        oscillator.type = 'sine'
        gainNode.gain.value = 0.2

        oscillator.start()

        // Two quick blips
        setTimeout(() => { oscillator.frequency.value = 800 }, 100)
        setTimeout(() => { gainNode.gain.value = 0 }, 200)
        setTimeout(() => { gainNode.gain.value = 0.2 }, 300)
        setTimeout(() => { oscillator.frequency.value = 600 }, 300)
        setTimeout(() => { oscillator.frequency.value = 800 }, 400)
        setTimeout(() => {
          oscillator.stop()
          audioContext.close()
        }, 500)
      } catch (e) {
        console.log('Audio not supported')
      }
    }
  }, [])

  // Track if warning has been played
  const hasPlayedWarningRef = useRef(false)

  // Timer logic
  useEffect(() => {
    if (isRunning) {
      intervalRef.current = setInterval(() => {
        setTimeLeft(prev => {
          if (mode === 'countdown') {
            const newTime = prev - 1

            // Play warning at configured time (only once, and only if duration > warningTime)
            if (warningTime > 0 && newTime === warningTime && !hasPlayedWarningRef.current && duration > warningTime) {
              hasPlayedWarningRef.current = true
              playWarningBlip()
              // Show visual flash
              setShowWarningFlash(true)
              setTimeout(() => setShowWarningFlash(false), 1000)
            }

            if (newTime <= 0 && !hasEnded) {
              setHasEnded(true)
              playAlert()
              onTimerEnd?.()
            }
            return newTime
          } else {
            return prev + 1
          }
        })
      }, 1000)
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [isRunning, mode, hasEnded, playAlert, playWarningBlip, duration, warningTime, onTimerEnd])

  const start = () => {
    setIsRunning(true)
    setHasEnded(false)
  }

  const pause = () => {
    setIsRunning(false)
  }

  const reset = () => {
    setIsRunning(false)
    setTimeLeft(duration)
    setHasEnded(false)
    hasPlayedWarningRef.current = false
  }

  const setPreset = (seconds: number) => {
    setDuration(seconds)
    setTimeLeft(seconds)
    setIsRunning(false)
    setHasEnded(false)
    hasPlayedWarningRef.current = false

    if (exerciseId) {
      saveExerciseTimerDuration(exerciseId, seconds)
    }
  }

  const adjustTime = (seconds: number) => {
    // Temporary adjustment - does NOT save to exercise memory
    const newDuration = Math.max(15, duration + seconds)
    setDuration(newDuration)
    setTimeLeft(prev => Math.max(0, prev + seconds))
  }

  const toggleMode = () => {
    if (mode === 'countdown') {
      // Switching to countup: preserve elapsed time (duration - timeLeft)
      const elapsed = Math.max(0, duration - timeLeft)
      setMode('countup')
      setTimeLeft(elapsed)
      // Keep duration for potential switch back
    } else {
      // Switching back to countdown: calculate remaining time from elapsed
      const settings = getTimerSettings()
      const targetDuration = exerciseId ? getExerciseTimerDuration(exerciseId) : settings.defaultDuration
      const remaining = Math.max(0, targetDuration - timeLeft)
      setMode('countdown')
      setDuration(targetDuration)
      setTimeLeft(remaining)
    }
    // Keep timer running - don't pause on mode switch
    setHasEnded(false)
  }

  const skipRest = () => {
    releaseWakeLock()
    onClose()
  }

  // Calculate progress
  const progress = mode === 'countdown' && duration > 0
    ? Math.max(0, timeLeft / duration)
    : 0

  // Format weight with unit
  const formatWeight = (w: number) => {
    return unit === 'kg' ? `${w}kg` : `${Math.round(w * 2.205)}lb`
  }

  // Colors based on state
  const getTimerColor = () => {
    if (hasEnded || timeLeft < 0) return 'text-red-400'
    if (timeLeft <= 10 && mode === 'countdown') return 'text-yellow-400'
    return 'text-white'
  }

  const getProgressGradient = () => {
    if (hasEnded || timeLeft < 0) return 'from-red-500 to-orange-400'
    if (timeLeft <= 10 && mode === 'countdown') return 'from-yellow-500 to-amber-400'
    return 'from-blue-500 to-cyan-400'
  }

  const getButtonGradient = () => {
    if (hasEnded || timeLeft < 0) return 'from-red-500 to-orange-500'
    return 'from-blue-500 to-cyan-500'
  }

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black overflow-y-auto py-4 landscape:py-2"
      onDoubleClick={handleMinimize}
    >
      {/* 30-second warning flash overlay */}
      {showWarningFlash && (
        <div className="absolute inset-0 z-10 pointer-events-none animate-pulse">
          <div className="absolute inset-0 bg-gradient-to-b from-yellow-500/40 via-orange-500/30 to-transparent" />
          <div className="absolute inset-x-0 top-0 h-32 bg-gradient-to-b from-yellow-400/60 to-transparent" />
        </div>
      )}

      {/* Close button */}
      <button
        onClick={(e) => { e.stopPropagation(); skipRest() }}
        onDoubleClick={(e) => e.stopPropagation()}
        className="absolute top-4 landscape:top-2 right-4 landscape:right-2 z-10 w-12 h-12 landscape:w-10 landscape:h-10 flex items-center justify-center rounded-full bg-gray-800/80 text-gray-300 hover:text-white hover:bg-gray-700 text-2xl transition-colors"
        aria-label="Close timer"
      >
        <svg className="w-6 h-6 landscape:w-5 landscape:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>

      {/* Exercise name at top */}
      {exerciseName && (
        <p className="absolute top-8 landscape:top-2 left-0 right-0 text-center text-blue-400 text-lg landscape:text-sm uppercase tracking-[0.3em] font-medium">
          {exerciseName}
        </p>
      )}

      {/* Mode toggle */}
      <button
        onClick={(e) => { e.stopPropagation(); toggleMode() }}
        onDoubleClick={(e) => e.stopPropagation()}
        className="absolute top-8 landscape:top-2 left-6 landscape:left-4 text-gray-500 hover:text-white flex items-center gap-2 text-xs transition-colors uppercase tracking-wider p-2"
      >
        <svg className="w-4 h-4 landscape:w-3 landscape:h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
        </svg>
        <span className="landscape:hidden">{mode === 'countdown' ? 'Timer' : 'Stopwatch'}</span>
      </button>

      {/* Main timer - double-click on timer text should not minimize */}
      <div className="flex flex-col items-center landscape:mt-8" onDoubleClick={(e) => e.stopPropagation()}>
        {/* Time display */}
        <div className={`font-mono-timer font-bold ${getTimerColor()} breathe`} style={{ fontSize: 'clamp(3.5rem, 18vw, 10rem)', lineHeight: 1 }}>
          {formatTime(timeLeft)}
        </div>

        {/* Rest complete message */}
        {hasEnded && mode === 'countdown' && (
          <span className="text-red-400 text-xl font-semibold animate-pulse mt-4">
            REST COMPLETE!
          </span>
        )}
        {mode === 'countup' && (
          <span className="text-gray-500 text-sm mt-2 uppercase tracking-wider">elapsed</span>
        )}
      </div>

      {/* Progress bar with glow effect when running */}
      {mode === 'countdown' && (
        <div className="w-80 max-w-[80vw] mt-6 landscape:mt-2">
          <div className="h-2 landscape:h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <div
              className={`h-full bg-gradient-to-r ${getProgressGradient()} rounded-full transition-all duration-1000 ${isRunning ? 'glow-pulse' : ''}`}
              style={{ width: `${progress * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Stats row */}
      {(setNumber || weight || reps) && (
        <div className="flex gap-12 landscape:gap-8 mt-10 landscape:mt-4" onDoubleClick={(e) => e.stopPropagation()}>
          {setNumber && (
            <div className="text-center">
              <p className="text-3xl landscape:text-2xl font-bold text-white">S{setNumber}</p>
              <p className="text-xs text-gray-500 uppercase tracking-wider mt-1 landscape:hidden">Set</p>
            </div>
          )}
          {weight !== undefined && weight > 0 && (
            <div className="text-center">
              <p className="text-3xl landscape:text-2xl font-bold text-green-400">{formatWeight(weight)}</p>
              <p className="text-xs text-gray-500 uppercase tracking-wider mt-1 landscape:hidden">Weight</p>
            </div>
          )}
          {reps !== undefined && reps > 0 && (
            <div className="text-center">
              <p className="text-3xl landscape:text-2xl font-bold text-white">{reps}</p>
              <p className="text-xs text-gray-500 uppercase tracking-wider mt-1 landscape:hidden">Reps</p>
            </div>
          )}
        </div>
      )}

      {/* Preset buttons - show 5 options around current duration */}
      {mode === 'countdown' && (
        <div className="flex gap-2 mt-10 landscape:mt-3 px-4" onClick={(e) => e.stopPropagation()} onDoubleClick={(e) => e.stopPropagation()}>
          {getDynamicTimerPresets(duration).map(preset => (
            <button
              key={preset}
              onClick={() => setPreset(preset)}
              className={`px-3 py-2 landscape:px-2 landscape:py-1 rounded-lg text-xs font-medium transition-all ${
                duration === preset
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {formatTime(preset)}
            </button>
          ))}
        </div>
      )}

      {/* Control buttons */}
      <div className="flex items-center gap-8 landscape:gap-6 mt-8 landscape:mt-3" onClick={(e) => e.stopPropagation()} onDoubleClick={(e) => e.stopPropagation()}>
        {/* -15s */}
        {mode === 'countdown' && (
          <button
            onClick={(e) => { e.stopPropagation(); adjustTime(-TIMER_INCREMENT) }}
            onDoubleClick={(e) => e.stopPropagation()}
            className="w-16 h-16 landscape:w-12 landscape:h-12 rounded-full border-2 border-gray-700 text-gray-500 text-xl landscape:text-lg font-bold hover:border-blue-500 hover:text-blue-400 transition-all"
          >
            -15
          </button>
        )}

        {/* Start/Pause */}
        <button
          onClick={(e) => { e.stopPropagation(); isRunning ? pause() : start() }}
          onDoubleClick={(e) => e.stopPropagation()}
          className={`w-24 h-24 landscape:w-16 landscape:h-16 rounded-full flex items-center justify-center shadow-lg transition-all transform hover:scale-105 bg-gradient-to-r ${
            isRunning ? 'from-orange-500 to-amber-500 shadow-orange-500/30' : `${getButtonGradient()} shadow-blue-500/30`
          }`}
        >
          {isRunning ? (
            <svg className="w-12 h-12 landscape:w-8 landscape:h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
              <rect x="6" y="4" width="4" height="16" />
              <rect x="14" y="4" width="4" height="16" />
            </svg>
          ) : (
            <svg className="w-12 h-12 landscape:w-8 landscape:h-8 text-white ml-1" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
          )}
        </button>

        {/* Reset / +15s */}
        {mode === 'countdown' ? (
          <button
            onClick={(e) => { e.stopPropagation(); adjustTime(TIMER_INCREMENT) }}
            onDoubleClick={(e) => e.stopPropagation()}
            className="w-16 h-16 landscape:w-12 landscape:h-12 rounded-full border-2 border-gray-700 text-gray-500 text-xl landscape:text-lg font-bold hover:border-blue-500 hover:text-blue-400 transition-all"
          >
            +15
          </button>
        ) : (
          <button
            onClick={(e) => { e.stopPropagation(); reset() }}
            onDoubleClick={(e) => e.stopPropagation()}
            className="w-16 h-16 landscape:w-12 landscape:h-12 rounded-full border-2 border-gray-700 text-gray-500 hover:border-blue-500 hover:text-blue-400 flex items-center justify-center transition-all"
          >
            <svg className="w-6 h-6 landscape:w-5 landscape:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        )}
      </div>

      {/* End Timer / Skip Rest button */}
      <button
        onClick={(e) => { e.stopPropagation(); skipRest() }}
        onDoubleClick={(e) => e.stopPropagation()}
        className="mt-6 landscape:mt-2 px-5 py-2 landscape:px-4 landscape:py-1.5 bg-gray-800 hover:bg-gray-700 border border-gray-600 hover:border-gray-500 text-white rounded-lg transition-all text-sm landscape:text-xs font-medium uppercase tracking-wider flex items-center gap-2"
      >
        <svg className="w-4 h-4 landscape:w-3 landscape:h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
        </svg>
        End Timer
      </button>

      {/* Bottom hints - combined to avoid overlap */}
      <p className="absolute bottom-6 landscape:bottom-2 right-0 left-0 text-center text-gray-600 text-xs flex items-center justify-center gap-2">
        {wakeLockActive && (
          <>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-green-500" />
              Screen kept on
            </span>
            <span className="text-gray-700">|</span>
          </>
        )}
        <span className="text-gray-700">Double-tap to minimize</span>
      </p>

      {/* Contextual Tip - Timer settings */}
      <ContextualTip
        tipId="timer_settings"
        title="Timer Adjusts Per Exercise"
        message="Use +/- buttons or tap presets to change rest time. Each exercise remembers its own timer. Default can be set in Settings."
        icon="⏱️"
        position="top"
      />
    </div>
  )
}
