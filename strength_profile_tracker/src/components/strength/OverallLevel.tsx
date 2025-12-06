'use client'

import { Level, LEVEL_COLORS, LEVEL_FULL_NAMES } from '@/types'

interface OverallLevelProps {
  level: Level | null
  ratedCount: number
  totalCount: number
}

export default function OverallLevel({ level, ratedCount, totalCount }: OverallLevelProps) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-[#E0E0E0] dark:border-gray-700 p-4 flex flex-col">
      <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3 text-center">
        Overall Level
      </h3>

      <div className="flex-1 flex flex-col items-center justify-center">
        {level ? (
          <p
            className="text-xl font-bold"
            style={{ color: LEVEL_COLORS[level] }}
          >
            {LEVEL_FULL_NAMES[level]}
          </p>
        ) : (
          <p className="text-base text-gray-400 dark:text-gray-500">Not rated</p>
        )}

        <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
          <span className="font-semibold text-[#2C3E50] dark:text-gray-100">{ratedCount}</span>
          <span className="text-xs"> / {totalCount} rated</span>
        </p>
      </div>

      {/* Progress bar */}
      <div className="mt-3">
        <div className="h-1.5 bg-gray-100 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-[#3498DB] transition-all duration-300"
            style={{ width: `${(ratedCount / totalCount) * 100}%` }}
          />
        </div>
      </div>
    </div>
  )
}
