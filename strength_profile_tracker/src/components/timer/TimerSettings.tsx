'use client'

import { useState, useEffect } from 'react'
import { TIMER_PRESETS } from '@/types'
import {
  getTimerSettings,
  saveTimerSettings,
  resetTimerSettings,
  formatTime
} from '@/lib/storage/timer'

interface TimerSettingsProps {
  onClose: () => void
}

export default function TimerSettings({ onClose }: TimerSettingsProps) {
  const [defaultDuration, setDefaultDuration] = useState(90)
  const [soundEnabled, setSoundEnabled] = useState(true)
  const [vibrationEnabled, setVibrationEnabled] = useState(true)
  const [autoStart, setAutoStart] = useState(true)
  const [showResetConfirm, setShowResetConfirm] = useState(false)

  // Load settings on mount
  useEffect(() => {
    const settings = getTimerSettings()
    setDefaultDuration(settings.defaultDuration)
    setSoundEnabled(settings.soundEnabled)
    setVibrationEnabled(settings.vibrationEnabled)
    setAutoStart(settings.autoStart)
  }, [])

  // Save settings whenever they change
  const updateSettings = (updates: Parameters<typeof saveTimerSettings>[0]) => {
    saveTimerSettings(updates)
  }

  const handleDefaultDurationChange = (duration: number) => {
    setDefaultDuration(duration)
    updateSettings({ defaultDuration: duration })
  }

  const handleSoundToggle = () => {
    const newValue = !soundEnabled
    setSoundEnabled(newValue)
    updateSettings({ soundEnabled: newValue })
  }

  const handleVibrationToggle = () => {
    const newValue = !vibrationEnabled
    setVibrationEnabled(newValue)
    updateSettings({ vibrationEnabled: newValue })
  }

  const handleAutoStartToggle = () => {
    const newValue = !autoStart
    setAutoStart(newValue)
    updateSettings({ autoStart: newValue })
  }

  const handleResetAll = () => {
    resetTimerSettings()
    const settings = getTimerSettings()
    setDefaultDuration(settings.defaultDuration)
    setSoundEnabled(settings.soundEnabled)
    setVibrationEnabled(settings.vibrationEnabled)
    setAutoStart(settings.autoStart)
    setShowResetConfirm(false)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-2xl w-full max-w-sm shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Timer Settings
          </h2>
          <button
            onClick={onClose}
            className="p-2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Settings content */}
        <div className="p-4 space-y-5">
          {/* Default Duration */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Default Rest Duration
            </label>
            <div className="flex flex-wrap gap-2">
              {TIMER_PRESETS.map(preset => (
                <button
                  key={preset}
                  onClick={() => handleDefaultDurationChange(preset)}
                  className={`px-3 py-2 text-sm rounded-lg transition-colors ${
                    defaultDuration === preset
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                  }`}
                >
                  {formatTime(preset)}
                </button>
              ))}
            </div>
          </div>

          {/* Sound Toggle */}
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Sound Alert
              </span>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Play sound when timer ends
              </p>
            </div>
            <button
              onClick={handleSoundToggle}
              className={`relative w-12 h-6 rounded-full transition-colors ${
                soundEnabled ? 'bg-blue-500' : 'bg-gray-300 dark:bg-gray-600'
              }`}
            >
              <span
                className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
                  soundEnabled ? 'left-7' : 'left-1'
                }`}
              />
            </button>
          </div>

          {/* Vibration Toggle */}
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Vibration Alert
              </span>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Vibrate when timer ends
              </p>
            </div>
            <button
              onClick={handleVibrationToggle}
              className={`relative w-12 h-6 rounded-full transition-colors ${
                vibrationEnabled ? 'bg-blue-500' : 'bg-gray-300 dark:bg-gray-600'
              }`}
            >
              <span
                className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
                  vibrationEnabled ? 'left-7' : 'left-1'
                }`}
              />
            </button>
          </div>

          {/* Auto-start Toggle */}
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Auto-start Timer
              </span>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Start timer after logging a set
              </p>
            </div>
            <button
              onClick={handleAutoStartToggle}
              className={`relative w-12 h-6 rounded-full transition-colors ${
                autoStart ? 'bg-blue-500' : 'bg-gray-300 dark:bg-gray-600'
              }`}
            >
              <span
                className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
                  autoStart ? 'left-7' : 'left-1'
                }`}
              />
            </button>
          </div>

          {/* Reset All */}
          <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
            {showResetConfirm ? (
              <div className="flex items-center gap-2">
                <span className="text-sm text-red-600 dark:text-red-400">Reset all?</span>
                <button
                  onClick={handleResetAll}
                  className="px-3 py-1 text-sm bg-red-500 text-white rounded-md hover:bg-red-600"
                >
                  Yes
                </button>
                <button
                  onClick={() => setShowResetConfirm(false)}
                  className="px-3 py-1 text-sm bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-300 dark:hover:bg-gray-600"
                >
                  No
                </button>
              </div>
            ) : (
              <button
                onClick={() => setShowResetConfirm(true)}
                className="text-sm text-red-600 dark:text-red-400 hover:underline"
              >
                Reset timer settings
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
