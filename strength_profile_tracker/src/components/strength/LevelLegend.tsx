'use client'

import { Level, LEVEL_COLORS, LEVEL_NAMES } from '@/types'

const LEVEL_DESCRIPTIONS: Record<Level, string> = {
  beginner: 'Just starting strength training',
  novice: '6-12 months of training',
  intermediate: '1-2 years of consistent training',
  advanced: '2+ years of serious training'
}

export default function LevelLegend() {
  const levels: Level[] = ['beginner', 'novice', 'intermediate', 'advanced']

  return (
    <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
      <h4 className="text-sm font-semibold text-[#2C3E50] dark:text-gray-100 mb-3">Level Guide</h4>
      <div className="space-y-2">
        {levels.map(level => (
          <div key={level} className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded-full flex-shrink-0"
              style={{ backgroundColor: LEVEL_COLORS[level] }}
            />
            <span
              className="text-xs font-medium w-8"
              style={{ color: LEVEL_COLORS[level] }}
            >
              {LEVEL_NAMES[level]}
            </span>
            <span className="text-xs text-gray-600 dark:text-gray-400">
              {LEVEL_DESCRIPTIONS[level]}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
