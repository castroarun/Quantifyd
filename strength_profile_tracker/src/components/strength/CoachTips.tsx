'use client'

import { CoachTip } from '@/lib/calculations/strength'

interface CoachTipsProps {
  tips: CoachTip[]
}

export default function CoachTips({ tips }: CoachTipsProps) {
  if (tips.length === 0) return null

  return (
    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-gray-800 dark:to-gray-700 rounded-lg border border-blue-100 dark:border-gray-600 p-4">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-lg">🤖</span>
        <h3 className="text-sm font-medium text-blue-800 dark:text-blue-300">
          Coach Tips
        </h3>
      </div>

      <div className="space-y-2">
        {tips.map((tip, index) => (
          <div
            key={index}
            className="flex items-start gap-2 bg-white/60 dark:bg-gray-900/40 rounded-md p-2"
          >
            <span className="text-base flex-shrink-0" role="img" aria-hidden="true">
              {tip.icon}
            </span>
            <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
              {tip.message}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}
