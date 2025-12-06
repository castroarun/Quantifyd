'use client'

import { Level, LEVEL_COLORS, LEVEL_FULL_NAMES } from '@/types'

interface OverallLevelProps {
  level: Level | null
  ratedCount: number
  totalCount: number
}

export default function OverallLevel({ level, ratedCount, totalCount }: OverallLevelProps) {
  return (
    <div className="bg-white rounded-lg border border-[#E0E0E0] p-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-medium text-gray-500">Overall Level</h3>
          {level ? (
            <p
              className="text-2xl font-bold mt-1"
              style={{ color: LEVEL_COLORS[level] }}
            >
              {LEVEL_FULL_NAMES[level]}
            </p>
          ) : (
            <p className="text-lg text-gray-400 mt-1">Not rated yet</p>
          )}
        </div>

        <div className="text-right">
          <p className="text-sm text-gray-500">Exercises Rated</p>
          <p className="text-xl font-semibold text-[#2C3E50] mt-1">
            {ratedCount} / {totalCount}
          </p>
        </div>
      </div>

      {/* Progress bar */}
      <div className="mt-3">
        <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-[#3498DB] transition-all duration-300"
            style={{ width: `${(ratedCount / totalCount) * 100}%` }}
          />
        </div>
      </div>
    </div>
  )
}
