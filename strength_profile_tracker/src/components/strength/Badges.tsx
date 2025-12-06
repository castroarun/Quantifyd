'use client'

import { Badge } from '@/types'

interface BadgesProps {
  badges: Badge[]
}

export default function Badges({ badges }: BadgesProps) {
  const earnedCount = badges.filter(b => b.earned).length
  const totalCount = badges.length

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-[#E0E0E0] dark:border-gray-700 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">
          Achievements
        </h3>
        <span className="text-xs text-gray-400 dark:text-gray-500">
          {earnedCount} / {totalCount}
        </span>
      </div>

      <div className="grid grid-cols-4 gap-2">
        {badges.map(badge => (
          <div
            key={badge.id}
            className={`relative flex flex-col items-center p-2 rounded-lg transition-all ${
              badge.earned
                ? 'bg-amber-50 dark:bg-amber-900/20'
                : 'bg-gray-50 dark:bg-gray-700/50 opacity-40'
            }`}
            title={`${badge.name}: ${badge.description}`}
          >
            <span className="text-2xl" role="img" aria-label={badge.name}>
              {badge.icon}
            </span>
            <span className={`text-[10px] mt-1 text-center leading-tight ${
              badge.earned
                ? 'text-gray-700 dark:text-gray-200'
                : 'text-gray-400 dark:text-gray-500'
            }`}>
              {badge.name}
            </span>
            {badge.earned && (
              <div className="absolute -top-1 -right-1 w-3.5 h-3.5 bg-green-500 rounded-full flex items-center justify-center">
                <svg
                  className="w-2 h-2 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={3}
                    d="M5 13l4 4L19 7"
                  />
                </svg>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
